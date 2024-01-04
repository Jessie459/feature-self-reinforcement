import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def is_distributed():
    return dist.is_available() and dist.is_initialized()


class FSRLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        student_temp=0.1,
        momentum=0.9,
        num_gcrops=2,
        num_lcrops=0,
        warmup_epochs=None,
        epochs=None,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(epochs - warmup_epochs) * teacher_temp,
        ))
        self.momentum = momentum
        self.num_gcrops = num_gcrops
        self.num_lcrops = num_lcrops

        self.register_buffer("class_center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, 1, out_dim))

    def forward(self, teacher_output, student_output, student_output2=None, student_mask=None, epoch=None):
        class_loss = self.forward_class(teacher_output, student_output, student_output2, epoch)
        patch_loss = self.forward_patch(teacher_output, student_output, student_mask, epoch)
        return class_loss, patch_loss

    def forward_patch(self, teacher_output, student_output, student_mask=None, epoch=None):
        if student_mask is None:
            return torch.tensor(0.0).to(student_output.device)

        student_patch = student_output[:, 1:].chunk(2)[1]
        student_value = student_patch / self.student_temp
        with torch.no_grad():
            teacher_patch = teacher_output[:, 1:].chunk(2)[1]
            teacher_value = (teacher_patch - self.patch_center) / self.teacher_temp[epoch]
            teacher_value = F.softmax(teacher_value, dim=-1)

        mask = student_mask.chunk(2)[1].float().flatten(1)
        loss = torch.sum(-teacher_value * F.log_softmax(student_value, dim=-1), dim=-1)
        loss = torch.sum(loss * mask, dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        loss = loss.mean()

        self.update_patch_center(teacher_patch)
        return loss

    def forward_class(self, teacher_output, student_output, student_output2=None, epoch=None):
        student_class = student_output[:, 0]
        if student_output2 is not None:
            student_class = torch.cat([student_class, student_output2], dim=0)
        student_value = student_class / self.student_temp
        student_value = student_value.chunk(self.num_gcrops + self.num_lcrops)

        with torch.no_grad():
            teacher_class = teacher_output[:, 0].chunk(2)[0]
            teacher_value = (teacher_class - self.class_center) / self.teacher_temp[epoch]
            teacher_value = F.softmax(teacher_value, dim=-1)

        total_loss = 0
        loss_terms = 0
        for i in range(1, len(student_value)):
            loss = torch.sum(-teacher_value * F.log_softmax(student_value[i], dim=-1), dim=-1)
            total_loss += loss.mean()
            loss_terms += 1
        total_loss /= loss_terms

        self.update_class_center(teacher_class)
        return total_loss

    @torch.no_grad()
    def update_patch_center(self, teacher_output):
        center = teacher_output.mean(dim=1, keepdim=True).sum(dim=0, keepdim=True)
        if is_distributed():
            dist.all_reduce(center)
            batch_size = teacher_output.shape[0] * dist.get_world_size()
        else:
            batch_size = teacher_output.shape[0]
        center /= batch_size
        self.patch_center = self.patch_center * self.momentum + center * (1 - self.momentum)

    @torch.no_grad()
    def update_class_center(self, teacher_output):
        center = teacher_output.sum(dim=0, keepdim=True)
        if is_distributed():
            dist.all_reduce(center)
            batch_size = teacher_output.shape[0] * dist.get_world_size()
        else:
            batch_size = teacher_output.shape[0]
        center /= batch_size
        self.class_center = self.class_center * self.momentum + center * (1 - self.momentum)
