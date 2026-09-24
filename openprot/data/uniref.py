import torch
import numpy as np
import pandas as pd
# import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset
import json

import threading

lock = threading.Lock()

class UnirefDataset(OpenProtDataset):
    def setup(self):
        # self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        
        with open(self.cfg.path) as db:
            start = self.index[idx]
            end = self.index[idx + 1]
            db.seek(start)
            item = db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        
        seq_mask = np.ones(len(seqres), dtype=np.float32)
    
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        name = header if len(header.split()) == 0 else header.split()[0]
        residx = np.arange(len(seqres), dtype=np.float32)
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
            residx=residx,
        )
