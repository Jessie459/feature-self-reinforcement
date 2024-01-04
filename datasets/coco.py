import os

import imageio
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from . import transforms

class_list = [
    "_background_",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_cls_label_list(name_list_dir):
    return np.load(os.path.join(name_list_dir, "cls_labels_onehot.npy"), allow_pickle=True).item()


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class COCODataset(Dataset):
    def __init__(
        self,
        root_dir,
        name_list_dir,
        split="train",
        stage="train",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.stage = stage
        self.split = split
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.seg_dir = os.path.join(root_dir, "SegmentationClass")

        self.name_list = np.loadtxt(os.path.join(name_list_dir, split + ".txt"), dtype=str)
        self.cls_label_dict = np.load(os.path.join(name_list_dir, "cls_labels_onehot.npy"), allow_pickle=True).item()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]

        image_path = os.path.join(self.img_dir, self.stage + "2014", name + ".jpg")
        image = np.asarray(pil_loader(image_path))

        label_path = os.path.join(self.seg_dir, self.stage + "2014", name + ".png")
        label = np.asarray(imageio.imread(label_path))

        return name, image, label


class COCOClsDataset(COCODataset):
    def __init__(
        self,
        root_dir,
        name_list_dir,
        split="train",
        stage="train",
        crop_size=448,
        ignore_index=255,
        transform=None,
    ):
        super().__init__(root_dir, name_list_dir, split, stage)
        self.crop_size = crop_size
        self.ignore_index = ignore_index

        self.transform = transform
        self.random_hflip = T.RandomHorizontalFlip(p=0.5)
        self.random_color_jitter = T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8)

    def __len__(self):
        return len(self.name_list)

    def transform_image(self, img):
        img = transforms.random_scaling(img, scale_range=[0.5, 2.0])
        img = self.random_hflip(img)
        img = self.random_color_jitter(img)
        img = np.asarray(img)
        img, img_box = transforms.random_crop(img, crop_size=self.crop_size, mean_rgb=[0, 0, 0], ignore_index=self.ignore_index)
        img = transforms.normalize_img(img)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img, img_box

    def __getitem__(self, index):
        name = self.name_list[index]

        pil_image = pil_loader(os.path.join(self.img_dir, self.stage + "2014", name + ".jpg"))
        cls_label = self.cls_label_dict[name]

        images = self.transform(pil_image)  # 2 global crops and several local crops

        return name, images, cls_label


class COCOSegDataset(COCODataset):
    def __init__(
        self,
        root_dir,
        name_list_dir,
        split="train",
        stage="train",
        resize_range=[512, 640],
        rescale_range=[0.5, 2.0],
        crop_size=512,
        img_fliplr=True,
        ignore_index=255,
        aug=False,
    ):
        super().__init__(root_dir, name_list_dir, split, stage)
        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53],
                    ignore_index=self.ignore_index,
                )
        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))
        return image, label

    def __getitem__(self, index):
        name, image, label = super().__getitem__(index)
        image, label = self.__transforms(image=image, label=label)
        cls_label = self.cls_label_dict[name]
        return name, image, label, cls_label
