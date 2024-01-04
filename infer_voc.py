import argparse
import os

import cv2
import imageio
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import voc
from models.network import build_model
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs, str2bool

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", type=str, required=True)

parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
parser.add_argument("--num_classes", type=int, default=21)
parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--drop_path_rate", type=float, default=0.0)
parser.add_argument("--cls_depth", type=int, default=2)
parser.add_argument("--out_dim", type=int, default=4096)

parser.add_argument("--data_folder", type=str, default="~/data/VOCdevkit/VOC2012")
parser.add_argument("--list_folder", type=str, default="datasets/voc")
parser.add_argument("--infer_set", type=str, default="val")
parser.add_argument("--ignore_index", type=int, default=255)
parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 1.25, 1.5))

parser.add_argument("--save_cam", type=str2bool, default=True)


def cam_on_img(cam, img=None):
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.addWeighted(cam, 0.5, img, 0.5, 0)
    else:
        out = cam
    return out


@torch.no_grad()
def main_evaluate(model, data_loader):
    model.cuda()
    model.eval()

    img_dir = os.path.join(args.data_folder, "JPEGImages")
    img_dir = os.path.expanduser(img_dir)

    gts, seg_pred = [], []

    for data in tqdm(data_loader, total=len(data_loader), ncols=100):
        name, inputs, labels, cls_labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        _, _, h, w = inputs.shape
        seg_list = []
        cam_list = []
        for sc in args.scales:
            _h, _w = int(h * sc), int(w * sc)
            _inputs = F.interpolate(inputs, size=[_h, _w], mode="bilinear", align_corners=False)
            _inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

            outputs = model(_inputs_cat)
            segs = outputs["seg_out"]
            cams = outputs["top_cam"]

            segs = F.interpolate(segs, size=labels.shape[1:], mode="bilinear", align_corners=False)
            segs = segs[:1, ...] + segs[1:, ...].flip(-1)

            cams = F.interpolate(cams, size=labels.shape[1:], mode="bilinear", align_corners=False)
            cams = cams[:1, ...] + cams[1:, ...].flip(-1)

            seg_list.append(segs)
            cam_list.append(cams)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = F.relu(cam)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)
        cam = cam.cpu().numpy()

        if args.save_cam:
            img = np.array(Image.open(os.path.join(img_dir, name[0] + ".jpg")).convert("RGB"))
            for i in range(args.num_classes - 1):
                if cls_labels[0, i].item() == 1:
                    cam_img = cam_on_img((cam[0, i] * 255).astype(np.uint8), img)
                    Image.fromarray(cam_img).save(os.path.join(args.cams_dir, f"{name[0]}_cls{i}.png"))

        seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
        seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(int))
        gts += list(labels.cpu().numpy().astype(int))

        np.save(os.path.join(args.logits_dir, f"{name[0]}.npy"), {"msc_seg": seg.cpu().numpy()})

    seg_score = evaluate.scores(gts, seg_pred)
    print(format_tabs([seg_score], ["seg_pred"], cat_list=voc.class_list))
    return seg_score


def main_crf():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + ".txt"
    with open(txt_name) as f:
        name_list = [x for x in f.read().split("\n") if x]

    images_path = os.path.join(args.data_folder, "JPEGImages")
    labels_path = os.path.join(args.data_folder, "SegmentationClassAug")

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
        image = imageio.imread(image_name).astype(np.float32)
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

        imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
        imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))]
    )

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    print(format_tabs([crf_score], ["seg_crf"], cat_list=voc.class_list))
    return crf_score


def main():
    dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage="val",
        aug=False,
        ignore_index=args.ignore_index,
    )
    data_loader = DataLoader(dataset, batch_size=1)

    model = build_model(args, pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    seg_score = main_evaluate(model, data_loader)
    torch.cuda.empty_cache()

    crf_score = main_crf()


if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.checkpoint.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/seg_preds_rgb", args.infer_set)
    args.cams_dir = os.path.join(base_dir, "segs/cams", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.cams_dir, exist_ok=True)

    print(args)
    main()
