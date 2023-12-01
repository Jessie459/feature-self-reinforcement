import os

import torch
import torch.distributed as dist
import torch.nn as nn


def is_distributed():
    if dist.is_available() and dist.is_initialized():
        return True
    else:
        return False


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    if is_distributed():
        return dist.get_rank()
    else:
        return 0


def is_main_process():
    return get_rank() == 0


def reduce_tensor(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    dist.all_reduce(tensor)
    tensor /= world_size

    return tensor


def gather_tensor(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def setup_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.dist = True
        print(f"Will run the code on {args.world_size} GPUs.")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )

        torch.cuda.set_device(args.local_rank)
        dist.barrier()
    else:
        args.world_size = 1
        args.local_rank = 0
        args.rank = 0
        args.dist = False
        print(f"Will run the code on one GPU.")


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
