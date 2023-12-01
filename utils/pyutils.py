import argparse
import datetime
import logging
import random

import numpy as np
import torch
from texttable import Texttable


def cosine_scheduler(value, final_value, total_iters, warmup_iters=0, start_value=0.0):
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_value, value, warmup_iters)
    else:
        warmup_schedule = np.array([])

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def str2bool(s: str) -> bool:
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_tabs(scores, name_list, cat_list=None):
    _keys = list(scores[0]["iou"].keys())
    _values = []

    for i in range(len(name_list)):
        _values.append(list(scores[i]["iou"].values()))

    _values = np.array(_values) * 100

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values[:, i]))

    t.add_row(["mIoU"] + list(_values.mean(1)))

    return t.draw()


def setup_logger(filename="log.txt"):
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(formatter)
    logger.addHandler(cHandler)

    fHandler = logging.FileHandler(filename, mode="w")
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    # time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
