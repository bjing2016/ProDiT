import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from typing import override
from .track import OpenProtTrack

import pandas as pd
import math


class CATHTrack(OpenProtTrack):

    def setup(self):
        self.lookup = {}

        def add(key):
            self.lookup[key] = self.lookup.get(key, len(self.lookup)+1)
        with open(self.cfg.idx) as f:
            for line in f:
                if line[0] == '#': continue
                label = line.split()[0].split('.')
                add('.'.join(label[:1]))
                add('.'.join(label[:2]))
                add('.'.join(label[:3]))
            
    def tokenize(self, data):    
        # ugly and not vectorized 
        for i in range(len(data['cath'])):
            key = ".".join(map(str, data['cath'][i].astype(int)))
            data['cath'][i,0] = self.lookup.get('.'.join(key.split('.')[:1]), 0)
            data['cath'][i,1] = self.lookup.get('.'.join(key.split('.')[:2]), 0)
            data['cath'][i,2] = self.lookup.get('.'.join(key.split('.')[:3]), 0)
    

    def add_modules(self, model):
        # zero is not used
        model.cath_embed = nn.Embedding(len(self.lookup)+1, model.cfg.dim)
        model.cath_embed_cond = nn.Embedding(len(self.lookup)+1, model.cfg.dim)
        
        torch.nn.init.zeros_(model.cath_embed.weight)
        torch.nn.init.zeros_(model.cath_embed_cond.weight)
        
    def corrupt(self, batch, noisy_batch, target, logger=None):
        pass # deprecated        
        # drop_prob = np.cumsum(self.cfg.drop_prob)
        # B = len(batch['name'])
        # rand = 1 - batch['_cath_noise']
        # num_to_keep = (rand[:,None] > torch.from_numpy(drop_prob).to(rand.device)).sum(-1)
        # noisy_batch['cath'] = torch.where(
        #     (torch.arange(3, device=rand.device) < num_to_keep[:,None])[:,None],
        #     batch['cath'],
        #     0.0
        # )
        
    def embed(self, model, batch, inp):
        pass # deprecated
        # inp["x"] += torch.where(
        #     (batch['cath'] > 0)[...,None],
        #     model.cath_embed(batch['cath'].int()), 
        #     0.0
        # ).sum(-2)
        # inp["x_cond"] += torch.where(
        #     (batch['cath'] > 0)[...,None],
        #     model.cath_embed_cond(batch['cath'].int()), 
        #     0.0
        # ).sum(-2)
        
    def predict(self, model, inp, out, readout):
        pass
        
    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        return 0.0
