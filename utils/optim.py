import numpy as np
import torch


class CosWarmupAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, weight_decay, betas, warmup_iters, max_iters, warmup_ratio):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        self.global_step = 0
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.__init_lr = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iters:
            lr_mult = 1 - (1 - self.global_step / self.warmup_iters) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iters:
            lr_mult = np.cos((self.global_step - self.warmup_iters) / (self.max_iters - self.warmup_iters) * np.pi) * 0.5 + 0.5
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1


class PolyWarmupAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, weight_decay, betas, warmup_iters, max_iters, warmup_ratio, power=0.9):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        self.global_step = 0
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.max_iters = max_iters
        self.power = power
        self.__init_lr = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iters:
            lr_mult = 1 - (1 - self.global_step / self.warmup_iters) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iters:
            lr_mult = (1 - self.global_step / self.max_iters) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1


def get_param_groups(model, weight_decay, skip_list=None):
    skip_list = set(skip_list) if skip_list is not None else set()
    not_wd_params = []
    has_wd_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad():
            continue
        if name.endswith(".bias") or len(param.shape) == 1 or name in skip_list:
            not_wd_params.append(param)
        else:
            has_wd_params.append(param)
    param_groups = [
        {"params": not_wd_params, "weight_decay": 0.0},
        {"params": has_wd_params, "weight_decay": weight_decay},
    ]
    return param_groups
