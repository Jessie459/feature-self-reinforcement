import argparse
import datetime
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import datasets.voc as voc
from losses import FSRLoss
from models.losses import (
    DenseEnergyLoss,
    get_energy_loss,
    get_masked_ptc_loss,
    get_seg_loss,
)
from models.network import build_model
from models.PAR import PAR
from transforms import MultiviewTransform
from utils import evaluate, imutils, optim
from utils.camutils import (
    cam_to_label,
    label_to_aff_mask,
    multi_scale_cam,
    refine_cams_with_bkg_v2,
)
from utils.distutils import reduce_tensor, setup_distributed
from utils.masking import MaskGenerator
from utils.pyutils import (
    AverageMeter,
    cal_eta,
    cosine_scheduler,
    fix_random_seed,
    format_tabs,
    setup_logger,
    str2bool,
)
from utils.tbutils import make_grid_image, make_grid_label

torch.hub.set_dir("./pretrained")

parser = argparse.ArgumentParser()

# model parameters
parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
parser.add_argument("--drop_path_rate", type=float, default=0.1)
parser.add_argument("--cls_depth", type=int, default=2)
parser.add_argument("--out_dim", type=int, default=4096)

parser.add_argument("--data_folder", default=os.path.expanduser("~/data/VOCdevkit/VOC2012"), type=str)
parser.add_argument("--list_folder", default="datasets/voc", type=str)
parser.add_argument("--work_dir", default="work_dir_voc/fsr", type=str)

parser.add_argument("--num_classes", default=21, type=int)
parser.add_argument("--input_size", default=448, type=int)
parser.add_argument("--ignore_index", default=255, type=int)

parser.add_argument("--spg", default=2, type=int, help="samples per GPU")

parser.add_argument("--optimizer", type=str, default="CosWarmupAdamW")
parser.add_argument("--lr", type=float, default=6e-5)
parser.add_argument("--lr_scale", type=float, default=10.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--betas", nargs="+", default=(0.9, 0.999))

parser.add_argument("--max_iters", default=20000, type=int)
parser.add_argument("--log_iters", default=200, type=int)
parser.add_argument("--eval_iters", default=2000, type=int)
parser.add_argument("--warmup_iters", default=1500, type=int)

parser.add_argument("--high_thre", default=0.7, type=float)
parser.add_argument("--low_thre", default=0.25, type=float)
parser.add_argument("--bkg_thre", default=0.5, type=float)
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5))

parser.add_argument("--w_aff", default=0.2, type=float)
parser.add_argument("--w_seg", default=0.1, type=float)
parser.add_argument("--w_reg", default=0.05, type=float)

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--find_unused_parameters", type=str2bool, default=False)

parser.add_argument("--tensorboard", type=str2bool, default=False)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")

# multi-crop parameters
parser.add_argument("--global_crops_number", type=int, default=2)
parser.add_argument("--local_crops_number", type=int, default=0)

# knowledge distillation
parser.add_argument("--momentum", type=float, default=0.9995,
                    help="If EMA the entire encoder, setting momentum to a small value works")

# masked image modeling parameters
parser.add_argument("--use_mim", type=str2bool, default=True)
parser.add_argument("--block_size", type=int, default=64)
parser.add_argument("--mask_ratio", type=float, default=0.4)
parser.add_argument("--interpolate_mode", type=str, default="nearest")

parser.add_argument("--w_class", type=float, default=0.1)
parser.add_argument("--w_patch", type=float, default=0.1)
parser.add_argument("--fsr_iters", type=int, default=20000)

parser.add_argument("--use_aa", type=str2bool, default=False)
parser.add_argument("--use_gauss", type=str2bool, default=False)
parser.add_argument("--use_solar", type=str2bool, default=False)


@torch.no_grad()
def validate(model, data_loader, args):
    model.eval()

    pred_list, gt_list = [], []
    cam_list, aux_cam_list = [], []

    avg_meter = AverageMeter()

    for data in tqdm(data_loader, total=len(data_loader), ncols=100, disable=(args.local_rank > 0)):
        name, inputs, labels, cls_label = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_label = cls_label.cuda()

        inputs = F.interpolate(inputs, size=[args.input_size] * 2, mode="bilinear", align_corners=False)
        outputs = model(inputs)
        cls_logits = outputs["top_cls_out"]
        seg_logits = outputs["seg_out"]

        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], (cls_logits > 0).int().cpu().numpy()[0])
        avg_meter.add({"cls_score": cls_score})

        cam, aux_cam = multi_scale_cam(model, inputs, scales=args.cam_scales)

        resized_cam = F.interpolate(cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
        _, cam_label = cam_to_label(
            resized_cam,
            cls_label=cls_label,
            ignore_mid=False,
            bkg_thre=args.bkg_thre,
            ignore_index=args.ignore_index,
        )

        resized_aux_cam = F.interpolate(aux_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
        _, aux_cam_label = cam_to_label(
            resized_aux_cam,
            cls_label=cls_label,
            ignore_mid=False,
            bkg_thre=args.bkg_thre,
            ignore_index=args.ignore_index,
        )

        seg_logits = F.interpolate(seg_logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
        pred_list += list(torch.argmax(seg_logits, dim=1).cpu().numpy().astype(int))
        gt_list += list(labels.cpu().numpy().astype(int))
        cam_list += list(cam_label.cpu().numpy().astype(int))
        aux_cam_list += list(aux_cam_label.cpu().numpy().astype(int))

    cls_score = avg_meter.pop("cls_score")
    seg_score = evaluate.scores(gt_list, pred_list)
    cam_score = evaluate.scores(gt_list, cam_list)
    aux_cam_score = evaluate.scores(gt_list, aux_cam_list)

    model.train()

    tab_results = format_tabs(
        [cam_score, aux_cam_score, seg_score],
        name_list=["cam", "aux cam", "seg"],
        cat_list=voc.class_list,
    )
    return (cls_score, seg_score["miou"], cam_score["miou"], aux_cam_score["miou"]), tab_results


def train(args):
    setup_distributed(args)

    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(os.path.join(args.work_dir, "train.log"))
        logging.info("Pytorch version: %s" % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info("Arguments:\n" + "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    fix_random_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    tb_logger = None
    if args.local_rank == 0:
        if args.tensorboard is True:
            from torch.utils.tensorboard import SummaryWriter

            if args.log_dir is None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                log_dir = os.path.join(args.work_dir, "runs", timestamp + args.comment)
            else:
                log_dir = args.log_dir
            tb_logger = SummaryWriter(log_dir=log_dir)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ building student and teacher networks ... ============
    student = build_model(args, pretrained=True)
    teacher = build_model(args, pretrained=False)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    param_groups = student.get_param_groups()
    optimizer = getattr(optim, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": 0,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * args.lr_scale,
                "weight_decay": 0,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * args.lr_scale,
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        warmup_ratio=args.warmup_ratio,
    )

    student.cuda()
    teacher.cuda()
    if args.dist:
        student = DistributedDataParallel(student, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters)
        student_wo_ddp = student.module
    else:
        student_wo_ddp = student

    par_model = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    # ============ preparing data ... ============
    train_transform = MultiviewTransform(
        size1=args.input_size,
        num1=args.global_crops_number,
        num2=args.local_crops_number,
        use_aa=args.use_aa,
        use_gauss=args.use_gauss,
        use_solar=args.use_solar,
    )

    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split="train_aug",
        stage="train",
        ignore_index=args.ignore_index,
        transform=train_transform,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split="val",
        stage="val",
        aug=False,
        ignore_index=args.ignore_index,
    )

    if args.dist:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # ============ preparing loss ... ============
    fsr_loss_func = FSRLoss(
        out_dim=args.out_dim,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        student_temp=0.1,
        warmup_epochs=int(args.max_iters / 8),
        epochs=args.max_iters,
        num_gcrops=args.global_crops_number,
        num_lcrops=args.local_crops_number,
    ).cuda()
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    # ============ init schedulers ... ============
    momentum_schedule = cosine_scheduler(args.momentum, 1.0, args.fsr_iters)
    _schedule = np.repeat(1.0, args.max_iters - args.fsr_iters)
    momentum_schedule = np.concatenate([momentum_schedule, _schedule], axis=0)

    w_class_schedule = np.repeat(args.w_class, args.fsr_iters)
    w_patch_schedule = np.repeat(args.w_patch, args.fsr_iters)
    _schedule = np.repeat(0.0, args.max_iters - args.fsr_iters)
    w_class_schedule = np.concatenate([w_class_schedule, _schedule], axis=0)
    w_patch_schedule = np.concatenate([w_patch_schedule, _schedule], axis=0)

    # ============ mask generator ============
    if args.use_mim:
        mask_generator = MaskGenerator(
            input_size=args.input_size,
            patch_size=16,
            block_size=args.block_size,
            mask_ratio=args.mask_ratio,
            mode=args.interpolate_mode,
        )
    else:
        mask_generator = None

    avg_meter = AverageMeter()

    if sampler is not None:
        sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)

    for n_iter in range(args.max_iters):
        global_step = n_iter + 1
        try:
            _, images, cls_labels = next(train_loader_iter)
        except:
            if sampler is not None:
                sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            _, images, cls_labels = next(train_loader_iter)

        images = [x.cuda(non_blocking=True) for x in images]
        cls_labels = cls_labels.cuda(non_blocking=True)

        # generate random mask
        if args.use_mim:
            mask_strategy = "random" if (n_iter + 1) <= args.max_iters // 5 else "uncertain"
            _cam, _aux_cam = multi_scale_cam(student, images[1], scales=args.cam_scales)
            mask2 = mask_generator(_aux_cam, cls_labels, 0.2, 0.7, mask_strategy)
            mask1 = torch.zeros_like(mask2)
            mask = torch.cat([mask1, mask2], dim=0).cuda(non_blocking=True)
        else:
            mask1, mask2, mask = None, None, None

        # forward propagation
        outputs = student(images, mask=mask)
        student_outputs = outputs
        with torch.no_grad():
            teacher_outputs = teacher(images[:2])  # only the 2 global views pass through the teacher

        # classification loss
        top_cls_loss = F.multilabel_soft_margin_loss(outputs["top_cls_out"], cls_labels)
        aux_cls_loss = F.multilabel_soft_margin_loss(outputs["aux_cls_out"], cls_labels)
        cls_loss = top_cls_loss + aux_cls_loss

        with torch.no_grad():
            cam1, aux_cam1 = multi_scale_cam(student, images[0], scales=args.cam_scales)
            cam2, aux_cam2 = multi_scale_cam(student, images[1], scales=args.cam_scales)

            images_denorm1 = imutils.denormalize_img2(images[0].clone())
            images_denorm2 = imutils.denormalize_img2(images[1].clone())

            valid_cam1, _ = cam_to_label(
                cam1,
                cls_labels,
                img_box=None,
                ignore_mid=True,
                high_thre=args.high_thre,
                low_thre=args.low_thre,
                ignore_index=args.ignore_index,
            )
            valid_cam2, _ = cam_to_label(
                cam2,
                cls_labels,
                img_box=None,
                ignore_mid=True,
                high_thre=args.high_thre,
                low_thre=args.low_thre,
                ignore_index=args.ignore_index,
            )

            pseudo_label1 = refine_cams_with_bkg_v2(
                par_model,
                images_denorm1,
                cams=valid_cam1,
                cls_labels=cls_labels,
                img_box=None,
                high_thre=args.high_thre,
                low_thre=args.low_thre,
                ignore_index=args.ignore_index,
            )
            pseudo_label2 = refine_cams_with_bkg_v2(
                par_model,
                images_denorm2,
                cams=valid_cam2,
                cls_labels=cls_labels,
                img_box=None,
                high_thre=args.high_thre,
                low_thre=args.low_thre,
                ignore_index=args.ignore_index,
            )

        # segmentation loss
        seg_pred = F.interpolate(outputs["seg_out"], size=pseudo_label1.shape[-2:], mode="bilinear", align_corners=False)
        seg_loss = get_seg_loss(seg_pred, pseudo_label1.long(), ignore_index=args.ignore_index)

        # segmentation regularization loss
        reg_loss = get_energy_loss(img=images[0], logit=seg_pred, label=pseudo_label1, img_box=None, loss_layer=loss_layer)

        # affinity loss
        fmap = outputs["top_fmap"]
        resized_aux_cam = F.interpolate(aux_cam1, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(
            resized_aux_cam.detach(),
            cls_labels,
            img_box=None,
            ignore_mid=True,
            high_thre=args.high_thre,
            low_thre=args.low_thre,
            ignore_index=args.ignore_index,
        )
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        aff_loss = get_masked_ptc_loss(fmap, aff_mask)

        # feature self-reinforcement loss
        fsr_cls_loss, fsr_pat_loss = fsr_loss_func(
            teacher_output=teacher_outputs["project_out"],
            student_output=student_outputs["project_out"],
            student_output2=student_outputs["project_out2"],
            student_mask=mask,
            epoch=n_iter,
        )

        w_class = w_class_schedule[n_iter]
        w_patch = w_patch_schedule[n_iter]
        if (n_iter + 1) <= 2000:
            loss = cls_loss + w_class * fsr_cls_loss + w_patch * fsr_pat_loss + args.w_aff * aff_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = cls_loss + w_class * fsr_cls_loss + w_patch * fsr_pat_loss + args.w_aff * aff_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        # student update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # teacher EMA update
        with torch.no_grad():
            momentum = momentum_schedule[n_iter]
            for param_s, param_t in zip(student_wo_ddp.parameters(), teacher.parameters()):
                param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

        with torch.no_grad():
            avg_meter.add({
                "cls_loss": reduce_tensor(cls_loss).item(),
                "aff_loss": reduce_tensor(aff_loss).item(),
                "seg_loss": reduce_tensor(seg_loss).item(),
                "fsr_cls_loss": reduce_tensor(fsr_cls_loss).item(),
                "fsr_pat_loss": reduce_tensor(fsr_pat_loss).item(),
            })

        if (n_iter + 1) % args.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)

            cls_loss_value = avg_meter.pop("cls_loss")
            aff_loss_value = avg_meter.pop("aff_loss")
            seg_loss_value = avg_meter.pop("seg_loss")
            class_loss_value = avg_meter.pop("fsr_cls_loss")
            patch_loss_value = avg_meter.pop("fsr_pat_loss")

            if args.local_rank == 0:
                msg = f"Iter: {n_iter + 1} Elasped: {delta} ETA: {eta}"
                msg += f" cls_loss: {cls_loss_value:.4f}"
                msg += f" aff_loss: {aff_loss_value:.4f}"
                msg += f" seg_loss: {seg_loss_value:.4f}"
                msg += f" fsr_cls_loss: {class_loss_value:.4f}"
                msg += f" fsr_pat_loss: {patch_loss_value:.4f}"
                logging.info(msg)

            if tb_logger is not None and (n_iter + 1) % 1000 == 0:
                grid_img1, grid_cam1 = make_grid_image(images[0].detach(), cam1.detach(), cls_labels.detach())
                grid_img2, grid_cam2 = make_grid_image(images[1].detach(), cam2.detach(), cls_labels.detach(), mask=mask2)
                _, grid_aux_cam1 = make_grid_image(images[0].detach(), aux_cam1.detach(), cls_labels.detach())
                _, grid_aux_cam2 = make_grid_image(images[1].detach(), aux_cam2.detach(), cls_labels.detach(), mask=mask2)
                grid_seg_gt1 = make_grid_label(pseudo_label1.detach())
                grid_seg_gt2 = make_grid_label(pseudo_label2.detach())
                grid_seg_pred = make_grid_label(torch.argmax(seg_pred.detach(), dim=1))

                tb_logger.add_image("img1", grid_img1, global_step=global_step)
                tb_logger.add_image("img2", grid_img2, global_step=global_step)
                tb_logger.add_image("cam1", grid_cam1, global_step=global_step)
                tb_logger.add_image("cam2", grid_cam2, global_step=global_step)
                tb_logger.add_image("aux_cam1", grid_aux_cam1, global_step=global_step)
                tb_logger.add_image("aux_cam2", grid_aux_cam2, global_step=global_step)
                tb_logger.add_image("seg_gt1", grid_seg_gt1, global_step=global_step)
                tb_logger.add_image("seg_gt2", grid_seg_gt2, global_step=global_step)
                tb_logger.add_image("seg_pred", grid_seg_pred, global_step=global_step)

                tb_logger.add_scalar("train/cls_loss", cls_loss_value, global_step=global_step)
                tb_logger.add_scalar("train/aff_loss", aff_loss_value, global_step=global_step)
                tb_logger.add_scalar("train/seg_loss", seg_loss_value, global_step=global_step)
                tb_logger.add_scalar("train/fsr_cls_loss", class_loss_value, global_step=global_step)
                tb_logger.add_scalar("train/fsr_pat_loss", patch_loss_value, global_step=global_step)

                tb_logger.add_scalar("schedule/momentum", momentum, global_step=global_step)
                tb_logger.add_scalar("schedule/w_class", w_class, global_step=global_step)
                tb_logger.add_scalar("schedule/w_patch", w_patch, global_step=global_step)

        if (n_iter + 1) % args.eval_iters == 0:
            if args.local_rank == 0:
                torch.save(student_wo_ddp.state_dict(), os.path.join(args.ckpt_dir, "student.pth"))
                torch.save(teacher.state_dict(), os.path.join(args.ckpt_dir, "teacher.pth"))

            scores, tab_results = validate(student, val_loader, args)
            if args.local_rank == 0:
                logging.info("[STUDENT] cls score: %.4f" % (scores[0]))
                logging.info("[STUDENT]\n" + tab_results)
            if tb_logger is not None:
                tb_logger.add_scalar("val/student_cls_score", scores[0], global_step=global_step)
                tb_logger.add_scalar("val/student_seg_score", scores[1], global_step=global_step)
                tb_logger.add_scalar("val/student_cam_score", scores[2], global_step=global_step)
                tb_logger.add_scalar("val/student_aux_cam_score", scores[3], global_step=global_step)

            scores, tab_results = validate(teacher, val_loader, args)
            if args.local_rank == 0:
                logging.info("[TEACHER] cls score: %.4f" % (scores[0]))
                logging.info("[TEACHER]\n" + tab_results)
            if tb_logger is not None:
                tb_logger.add_scalar("val/teacher_cls_score", scores[0], global_step=global_step)
                tb_logger.add_scalar("val/teacher_seg_score", scores[1], global_step=global_step)
                tb_logger.add_scalar("val/teacher_cam_score", scores[2], global_step=global_step)
                tb_logger.add_scalar("val/teacher_aux_cam_score", scores[3], global_step=global_step)


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
