import random

import torchvision.transforms as T
from PIL import ImageFilter, ImageOps
from timm.data import create_transform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        else:
            return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class MultiviewTransform:
    def __init__(
        self,
        size1=448,
        size2=96,
        scale1=(0.32, 1.0),
        scale2=(0.05, 0.32),
        num1=2,
        num2=0,
        interpolation=T.InterpolationMode.BICUBIC,
        use_aa=False,
        use_gauss=False,
        use_solar=False,
    ):
        self.num1 = num1
        self.num2 = num2

        if use_aa is True:
            self.transform1_view1 = create_transform(
                input_size=size1,
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation="bicubic",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
            )
            self.transform1_view2 = create_transform(
                input_size=size1,
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation="bicubic",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
            )
        else:
            transform_list = [
                T.RandomResizedCrop(size1, scale=scale1, interpolation=interpolation),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.5),
            ]
            if use_gauss is True:
                transform_list.append(GaussianBlur(0.1))
            if use_solar is True:
                transform_list.append(Solarization(0.2))
            transform_list.extend([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
            self.transform1_view1 = T.Compose(transform_list)
            self.transform1_view2 = T.Compose(transform_list)

        self.transform2 = T.Compose([
            T.RandomResizedCrop(size2, scale=scale2, interpolation=interpolation),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.transform1_view1(image))
        crops.append(self.transform1_view2(image))
        for _ in range(self.num2):
            crops.append(self.transform2(image))
        return crops
