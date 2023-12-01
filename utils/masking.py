import math
import random

import numpy as np
import torch
import torch.nn.functional as F


class MaskGenerator:
    def __init__(self, input_size=224, patch_size=16, block_size=32, mask_ratio=0.75):
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.block_size = block_size

        assert self.input_size % self.block_size == 0
        self.mask_size = self.input_size // self.block_size

        assert self.block_size % self.patch_size == 0
        self.scale = self.block_size // self.patch_size

        length = self.mask_size**2
        self.remove_length = int(np.ceil(length * self.mask_ratio))
        self.retain_length = length - self.remove_length

    def __call__(self, cam, cls_label=None, low_thresh=None, high_thresh=None, mask_strategy="random"):
        B = cam.shape[0]

        noise = torch.rand(B, self.mask_size**2)
        if mask_strategy != "random":
            cam = cam * cls_label.unsqueeze(-1).unsqueeze(-1)
            cam = torch.max(cam, dim=1)[0]  # (B, H, W)
            roi = torch.logical_and(cam > low_thresh, cam < high_thresh)  # (B, H, W)
            roi = F.interpolate(roi.float().unsqueeze(1), size=[self.mask_size, self.mask_size], mode="nearest")
            roi = roi.reshape(B, -1).bool()
            noise[roi] += 1

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, self.mask_size**2], dtype=torch.bool)
        mask[:, :self.retain_length] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = mask.reshape(B, self.mask_size, self.mask_size)
        mask = mask.repeat_interleave(self.scale, dim=1).repeat_interleave(self.scale, dim=2)
        return mask


class MaskGeneratorBEIT:
    def __init__(
        self,
        input_size,
        patch_size,
        pred_shape="random",
        pred_ratio=0.75,
        pred_ratio_var=0,
        pred_aspect_ratio=(0.3, 1.0 / 0.3),
    ):
        assert pred_shape in ["block", "random"]
        self.input_size = input_size
        self.patch_size = patch_size
        self.resolution = input_size // patch_size
        self.pred_shape = pred_shape
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))

    def get_pred_ratio(self):
        assert self.pred_ratio >= self.pred_ratio_var
        if self.pred_ratio_var > 0:
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + self.pred_ratio_var)
            return pred_ratio
        else:
            return self.pred_ratio

    def __call__(self):
        H, W = self.resolution, self.resolution
        high = self.get_pred_ratio() * H * W

        if self.pred_shape == "block":
            # following BEiT (https://arxiv.org/abs/2106.08254), see at
            # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(H, W) // 3) ** 2
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top : top + h, left : left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta
        else:
            mask = np.hstack([np.zeros(H * W - int(high)), np.ones(int(high))]).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(H, W)

        return mask
