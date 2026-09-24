"""
This is all boilerplate code, safe to skip / ignore!
"""

import logging
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict

import neptune
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def setup_logging(cfg):
    if not cfg.logfile:
        return
    tee = subprocess.Popen(
        ["tee", "-a", f"workdir/{cfg.name}/{cfg.logfile}"], stdin=subprocess.PIPE
    )
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def get_logger(name):
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([lg.get(key, []) for lg in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except Exception as _:
            pass
    return out


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = defaultdict(int)
        self.prefix = None
        self.masks = {}
        if cfg.neptune:
            self.neptune_init()

    def step(self, trainer, prefix):
        self.iter_step[prefix] += 1
        self._log[prefix + "/dur"].append(time.time() - self.last_log_time)
        self.last_log_time = time.time()

        interval = {"train": self.cfg.train_log_freq, "val": self.cfg.val_log_freq}[
            prefix
        ]
        if interval is not None and self.iter_step[prefix] % interval == 0:
            self.print_log(trainer, prefix)

    def register_masks(self, batch):
        for key in batch:
            if key[0] == "/":
                self.masks[key] = batch[key]

    def clear_masks(self):
        self.masks = {}

    # logs with every masked combination
    def masked_log(self, key, data, mask=None, post=None, sum=False, dims=1):
        self.log(key, data, mask, post, sum)
        for sub_key, sub_mask in self.masks.items():
            for i in range(dims):
                sub_mask = sub_mask.unsqueeze(-1)
            if mask is not None:
                new_mask = mask * sub_mask
            else:
                new_mask = sub_mask
            self.log(sub_key[1:] + "/" + key, data, new_mask, post, sum)

    def log(self, key, data, mask=None, post=None, sum=False):
        if isinstance(data, torch.Tensor):
            if mask is not None:  # we want this to be NaN if the mask is all zeros!
                if sum:
                    data = (data * mask).sum()
                else:
                    data = (data * mask).sum() / mask.expand(data.shape).sum()
            else:
                if sum:
                    data = data.sum()
                else:
                    data = data.mean()
            data = data.item()
            if post is not None:
                data = post(data)
        self._log[self.prefix + "/" + key].append(data)

    def epoch_end(self, trainer, prefix):
        interval = {"train": self.cfg.train_log_freq, "val": self.cfg.val_log_freq}[
            prefix
        ]
        if interval is None:
            self.print_log(trainer, prefix)

    def print_log(self, trainer, prefix="train", save=False, extra_logs=None):
        assert not save, "print_log(save=True) does not work yet"
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}/" in key}
        log = gather_log(log, trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update(
            {
                prefix + "/epoch": trainer.current_epoch,
                prefix + "/global_step": trainer.global_step,
                prefix + "/iter_step": self.iter_step[prefix],
                prefix + "/count": len(log[next(iter(log))]),
            }
        )
        if extra_logs:
            mean_log.update(extra_logs)

        if trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.cfg.neptune:
                self.neptune_log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv",
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}/" in key:
                del self._log[key]

    @rank_zero_only
    def neptune_init(self):
        self.run = neptune.init_run(
            project=self.cfg.project,
            with_id=self.cfg.run_id,
            name=os.environ["MODEL_DIR"].split("/")[1],
            source_files=[os.environ["CONFIG"]],
        )

    @rank_zero_only
    def neptune_log(self, log):
        for key in log:
            self.run[key].append(log[key])
