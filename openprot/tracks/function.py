import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from typing import override
from .track import OpenProtTrack
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype
from functools import partial

import pandas as pd
import math, json

class GOEmbeddingModule(nn.Module): # nn.Module for GO embeddings
    def __init__(self, num_go_terms, embedding_dim, padding_idx=0):
        super().__init__()
        self.go_embedding = nn.Embedding(num_go_terms, embedding_dim, padding_idx=padding_idx)
        torch.nn.init.zeros_(self.go_embedding.weight)
    def forward(self, go_terms):
        go_term_embeddings_lookup = self.go_embedding(go_terms)
        residue_go_embeddings = torch.sum(go_term_embeddings_lookup, dim=2) 
        return residue_go_embeddings
    
class FunctionTrack(OpenProtTrack):

    def setup(self):
        self.go_vocab = json.loads(open(self.cfg.go_vocab).read())
        self.num_go_terms = len(self.go_vocab) + 1 

    def tokenize(self, data):
        pass 

    def add_modules(self, model):
        # function conditioning 
        model.func_embed = GOEmbeddingModule(self.num_go_terms, model.cfg.dim, padding_idx=0)

    def corrupt(self, batch, noisy_batch, target, logger=None):

        noisy_batch['go_terms'] = torch.where(batch['go_noise'] == 0, batch['go_terms'], 0.0)
                                            
        return 
        
        # similar to ESM3, drop annotation across the protein with some probability
        drop_prob = self.cfg.drop_prob
        B = len(batch['name'])
        unpadded_noise = [arr[arr != -1] for arr in batch['_go_noise']]
        rand = 1 - unpadded_noise

        # get terms to drop for each array in batch
        drop_terms_list = [(unique_terms := np.unique(arr[arr != 0]))[np.array(random_vals) < drop_prob] 
                           for arr, random_vals in zip(batch['go_terms'], rand)]
        masks = [np.isin(arr, terms_to_drop) for arr, terms_to_drop in zip(batch['go_terms'], drop_terms_list)]
        noisy_batch['go_terms'] = np.where(np.stack(masks), 0, batch['go_terms'])

    def embed(self, model, batch, inp):
        # add function to sequence embeddings
        residue_go_embeddings = model.func_embed.go_embedding(batch['go_terms'].int()) 
        # model.func_embed(batch['go_terms'])
        inp["x"] += residue_go_embeddings # add go embeddings
        
    def predict(self, model, inp, out, readout):
        pass
        
    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        return 0.0
