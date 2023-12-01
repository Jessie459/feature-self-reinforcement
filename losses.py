import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def is_distributed():
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def concat_all_gather(tensor):
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


class FSRLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        teacher_temp=0.04,
        student_temp=0.1,
        momentum=0.9,
        num_gcrops=2,
        num_lcrops=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.momentum = momentum

        self.num_gcrops = num_gcrops
        self.num_lcrops = num_lcrops

        self.register_buffer("class_center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, 1, out_dim))

    def forward(self, teacher_output, student_output, student_mask=None):
        class_loss = self.forward_class(teacher_output, student_output)
        patch_loss = self.forward_patch(teacher_output, student_output, student_mask)
        return class_loss, patch_loss

    def forward_patch(self, teacher_output, student_output, student_mask=None):
        if student_mask is None:
            return torch.tensor(0.0).to(student_output.device)

        student_patch = student_output.chunk(2)[1][:, 1:]
        student_value = student_patch / self.student_temp
        with torch.no_grad():
            teacher_patch = teacher_output.chunk(2)[1][:, 1:]
            teacher_value = (teacher_patch - self.patch_center) / self.teacher_temp

        mask = student_mask.chunk(2)[1].float().flatten(1)
        loss = torch.sum(-F.softmax(teacher_value, dim=-1) * F.log_softmax(student_value, dim=-1), dim=-1)
        loss = torch.sum(loss * mask, dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        loss = loss.mean()

        self.update_patch_center(teacher_patch)
        return loss

    def forward_class(self, teacher_output, student_output):
        student_class_temp = student_output[:, 0] / self.student_temp
        student_class_list = student_class_temp.chunk(2)
        with torch.no_grad():
            teacher_class_temp = (teacher_output[:, 0] - self.class_center) / self.teacher_temp
            teacher_class_list = teacher_class_temp.chunk(2)

        t = teacher_class_list[0]
        s = student_class_list[1]
        loss = torch.sum(-F.softmax(t, dim=-1) * F.log_softmax(s, dim=-1), dim=-1)
        loss = loss.mean()

        self.update_class_center(teacher_output.chunk(self.num_gcrops)[0][:, 0])
        return loss

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
