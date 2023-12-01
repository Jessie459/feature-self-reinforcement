import argparse
import os
import sys

sys.path.append(".")

import os

import imageio
import joblib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets.coco as coco
from models.network import Network
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.distutils import setup_distributed
from utils.pyutils import format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="work_dir_coco/vit_base_dino_tsd/checkpoints/model.pth", type=str)

parser.add_argument("--data_folder", default=os.path.expanduser("~/data/COCO"), type=str)
parser.add_argument("--image_folder", default=os.path.expanduser("~/data/COCO/JPEGImages"), type=str)
parser.add_argument("--label_folder", default=os.path.expanduser("~/data/COCO/SegmentationClass"), type=str)
parser.add_argument("--list_folder", default="datasets/coco", type=str)

parser.add_argument("--backbone", default="vit_base_patch16_224", type=str, help="vit_base_patch16_224")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")

parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--scales", default=(1.0, 1.25, 1.5), help="multi_scales for seg")


parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", type=int, default=0)



def get_colormap(N=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


COLORMAP = get_colormap()


def _validate_one(pid, model, dataset, args):
    data_loader = DataLoader(dataset[pid], batch_size=1, num_workers=args.num_workers)
    print(f"local rank: {args.local_rank} number of samples: {len(data_loader)}")

    model.cuda()
    model.eval()

    if "val" in args.infer_set:
        image_dir = os.path.join(args.image_folder, "val2014")
        label_dir = os.path.join(args.label_folder, "val2014")
    elif "train" in args.infer_set:
        image_dir = os.path.join(args.image_folder, "train2014")
        label_dir = os.path.join(args.label_folder, "train2014")

    post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=1,  # 3
        pos_w=1,  # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,  # 5, 5
        bi_w=4,  # 4, 5
    )

    with torch.no_grad():
        pred_list = []
        true_list = []
        pred_crf_list = []

        for step, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >=", disable=(args.local_rank > 0)):
            name, inputs, labels, _ = data
            name = name[0]
            # if os.path.exists(os.path.join(args.logits_dir, name + ".npy")):
            #     continue

            inputs = inputs.cuda()
            # labels = labels.cuda()
            _, _, H, W = inputs.shape
            inputs = F.interpolate(inputs, size=[448, 448], mode="bilinear", align_corners=False)
            _, _, h, w = inputs.shape

            seg_list = []
            for scale in args.scales:
                _h, _w = int(h * scale), int(w * scale)
                _inputs = F.interpolate(inputs, size=[_h, _w], mode="bilinear", align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat, aux_layer=-3)[1]
                segs = F.interpolate(segs, size=[H, W], mode="bilinear", align_corners=False)
                seg = torch.max(segs[:1, ...], segs[1:, ...].flip(-1))
                seg_list.append(seg)
            seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)

            image_path = os.path.join(image_dir, name + ".jpg")
            label_path = os.path.join(label_dir, name + ".png")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            label = np.asarray(Image.open(label_path))

            prob = F.softmax(seg, dim=1)[0].cpu().numpy()
            prob = post_processor(image, prob)
            pred = np.argmax(prob, axis=0)

            imageio.imsave(os.path.join(args.segs_dir, name + ".png"), COLORMAP[pred])

            pred_crf_list.append(pred)
            pred_list.append(torch.argmax(seg, dim=1)[0].cpu().numpy())
            true_list.append(label)

            # seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            # gts += list(labels.cpu().numpy().astype(np.int16))
            # np.save(os.path.join(args.logits_dir, name[0] + ".npy"), {"msc_seg": seg.cpu().numpy()})

    seg_score = evaluate.scores(true_list, pred_list, num_classes=81)
    crf_score = evaluate.scores(true_list, pred_crf_list, num_classes=81)
    seg_score_tab = format_tabs([seg_score], name_list=["seg"], cat_list=coco.class_list)
    crf_score_tab = format_tabs([crf_score], name_list=["seg"], cat_list=coco.class_list)
    print(seg_score_tab)
    print(crf_score_tab)
    with open(os.path.join(args.base_dir, args.infer_set + "_seg_score.txt"), "a") as f:
        f.write(seg_score_tab + "\n")
    with open(os.path.join(args.base_dir, args.infer_set + "_crf_score.txt"), "a") as f:
        f.write(crf_score_tab + "\n")


def _validate(pid, model, dataset, args):
    data_loader = DataLoader(dataset[pid], batch_size=1, num_workers=args.num_workers)

    model.cuda()
    model.eval()

    with torch.no_grad():
        gts, seg_pred = [], []

        for step, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >=", disable=(args.local_rank > 0)):
            name, inputs, labels, _ = data
            if os.path.exists(os.path.join(args.logits_dir, name[0] + ".npy")):
                continue

            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = F.interpolate(inputs, size=[448, 448], mode="bilinear", align_corners=False)
            _, _, h, w = inputs.shape

            seg_list = []
            for sc in args.scales:
                _h, _w = int(h * sc), int(w * sc)

                _inputs = F.interpolate(inputs, size=[_h, _w], mode="bilinear", align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat, aux_layer=-3)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode="bilinear", align_corners=False)
                seg = torch.max(segs[:1, ...], segs[1:, ...].flip(-1))
                seg_list.append(seg)

            seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(os.path.join(args.logits_dir, name[0] + ".npy"), {"msc_seg": seg.cpu().numpy()})

    seg_score = evaluate.scores(gts, seg_pred, num_classes=81)
    seg_score_tab = format_tabs([seg_score], name_list=["seg"], cat_list=coco.class_list)
    print(seg_score_tab)
    with open(os.path.join(args.base_dir, args.infer_set + "_seg_score.txt"), "a") as f:
        f.write(seg_score_tab + "\n")


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + ".txt"
    with open(txt_name) as f:
        name_list = [x for x in f.read().split("\n") if x]

    if "val" in args.infer_set:
        images_path = os.path.join(args.image_folder, "val2014")
        labels_path = os.path.join(args.label_folder, "val2014")
    elif "train" in args.infer_set:
        images_path = os.path.join(args.image_folder, "train2014")
        labels_path = os.path.join(args.label_folder, "train2014")

    post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=1,  # 3
        pos_w=1,  # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,  # 5, 5
        bi_w=4,  # 4, 5
    )

    def _job(i):
        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit["msc_seg"]

        image_name = os.path.join(images_path, name + ".jpg")
        image = coco.robust_read_image(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)  # [None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))]
    )

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds, num_classes=81)
    crf_score_tab = format_tabs([crf_score], name_list=["crf"], cat_list=coco.class_list)
    print(crf_score_tab)
    with open(os.path.join(args.base_dir, args.infer_set + "_crf_score.txt"), "a") as f:
        f.write(crf_score_tab)


def validate(args):
    setup_distributed(args)

    val_dataset = coco.COCOSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage="val",
        aug=False,
        ignore_index=args.ignore_index,
    )

    model = Network(
        args.backbone,
        num_classes=args.num_classes,
        global_crops_number=2,
        out_dim=4096,
        cls_depth=2,
        aggregate="cls",
        drop_path_rate=0.0,
        use_mim=False,
        shared_head=False,
        pretrained=False,
    )

    state_dict = torch.load(args.model_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    if args.local_rank == 0:
        print(f"missing_keys: {msg.missing_keys}")
        print(f"unexpected_keys: {msg.unexpected_keys}")

    model.cuda()
    model.eval()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.dist:
        n_gpus = dist.get_world_size()
        dataset_list = [Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range(n_gpus)]
    else:
        dataset_list = [val_dataset]

    _validate_one(pid=args.local_rank, model=model, dataset=dataset_list, args=args)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.base_dir = base_dir
    args.logits_dir = os.path.join(base_dir, "logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    if args.local_rank == 0:
        print(vars(args))
    validate(args)
    # if args.local_rank == 0:
    #     crf_proc()
