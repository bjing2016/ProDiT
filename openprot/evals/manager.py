from ..utils.misc_utils import autoimport
import torch
import numpy as np
from ..data.data import OpenProtData
from ..tracks.manager import OpenProtTrackManager


class OpenProtEvalManager:
    def __init__(self, cfg, tracks: OpenProtTrackManager):
        self.cfg = cfg
        
        self.tracks = tracks

        self.evals = {}

        self.loaders = []

        collate = lambda batch: OpenProtData.batch([
            self.tracks.tokenize(data) for data in batch
        ])
        
        for name in cfg.evals:  # autoload the eval tasks
            type_ = cfg.evals[name].type
            eval_ = autoimport(f"openprot.evals.{type_}")(
                cfg.evals[name], cfg.features, tracks
            )
            eval_.cfg.name = name  # know its own name
            self.evals[name] = eval_
            
            self.loaders.append(torch.utils.data.DataLoader(
                eval_,
                num_workers=cfg.data.num_workers,
                batch_size=cfg.evals[name].batch,
                collate_fn=collate,
            ))
