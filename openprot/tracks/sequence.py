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
import math
from transformers import EsmTokenizer

MASK_IDX = 20
NUM_TOKENS = MASK_IDX+1

import esm
from esm.data import Alphabet

RNA_LETTERS = {"A": 0, "G": 1, "C": 2, "U": 3}
DNA_LETTERS = {"A": 0, "G": 1, "C": 2, "T": 3}

load_fn = esm.pretrained.load_model_and_alphabet
esm_registry = {
    "esm2_8M": partial(load_fn, "esm2_t6_8M_UR50D_500K"),
    "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": partial(load_fn, "esm2_t12_35M_UR50D_500K"),
    "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": partial(load_fn, "esm2_t30_150M_UR50D_500K"),
    "esm2_150M_270K": partial(load_fn, "esm2_t30_150M_UR50D_270K"),
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_650M_270K": partial(load_fn, "esm2_t33_650M_270K_UR50D"),
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_3B_270K": partial(load_fn, "esm2_t36_3B_UR50D_500K"),
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.dense = nn.Linear(in_dim, in_dim)
        self.layer_norm2 = nn.LayerNorm(in_dim)

        self.decoder = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm2(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
    
class SequenceTrack(OpenProtTrack):

    def setup(self):
        self.alphabet = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

        if self.cfg.all_atom:
            self.ntoks = 21 + 5 + 5 + 128
        else:
            self.ntoks = NUM_TOKENS # 21
            
    def tokenize(self, data):
        
        prot_aatype = np.array(seqres_to_aatype(data["seqres"]))
        rna_aatype = np.array([RNA_LETTERS.get(c, 4) for c in data["seqres"]])
        dna_aatype = np.array([DNA_LETTERS.get(c, 4) for c in data["seqres"]])
        lig_aatype = data["atom_num"].astype(int)
        data['aatype'] = (data['mol_type'] == 0) * prot_aatype
        data['aatype'] += (data['mol_type'] == 1) * (dna_aatype + 21)
        data['aatype'] += (data['mol_type'] == 2) * (rna_aatype + 26)
        data['aatype'] += (data['mol_type'] == 3) * (lig_aatype + 31)
        data['aatype'] = data['aatype'].astype(int)    
        
    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in rc.restypes_with_x]
        return torch.tensor(esm_reorder)

    def _compute_language_model_representations(self, esmaa):
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=self.cfg.use_esm_attn_map,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.cfg.use_esm_attn_map
            else None
        )
        return esm_s, esm_z

    def add_modules(self, model):
        
        model.seq_embed = nn.Embedding(self.ntoks, model.cfg.dim)
        if self.cfg.init:
            torch.nn.init.normal_(model.seq_embed.weight, std=self.cfg.init)
        if self.cfg.esm_lm_head:
            model.seq_out = EsmLMHead(model.cfg.dim, self.ntoks)
            if self.cfg.tied_weights:
                model.seq_out.decoder.weight = model.seq_embed.weight
        else:
            model.seq_out = nn.Linear(model.cfg.dim, self.ntoks)

        if self.cfg.all_atom:
            model.mol_type_cond = nn.Embedding(4, model.cfg.dim)
        
        if self.cfg.esm is not None:
            model.esm, self.esm_dict = esm_registry.get(self.cfg.esm)()
            model.esm.requires_grad_(False)
            model.esm.half()
            model.esm.eval()
            esm_dim = model.esm.layers[0].final_layer_norm.bias.shape[0]
            
            model.esm_s_combine = nn.Parameter(torch.zeros(model.esm.num_layers + 1))
            model.esm_s_mlp = nn.Sequential(
                nn.LayerNorm(esm_dim),
                nn.Linear(esm_dim, model.cfg.dim),
                nn.ReLU(),
                nn.Linear(model.cfg.dim, model.cfg.dim),
            )

            model.register_buffer("af2_to_esm", self._af2_to_esm(self.esm_dict))

        # if self.cfg.func_cond:
        #     num_go_terms = 8225 + 1 # TODO: read in num GO terms as argument
        #     model.func_embed = GOEmbeddingModule(num_go_terms, model.cfg.dim, padding_idx=0)
        #    # model.func_embed = nn.Embedding(num_go_terms, model.cfg.dim, padding_idx=0) 

    def corrupt(self, batch, noisy_batch, target, logger=None):
        eps = 1e-6
        
        tokens = batch["aatype"]
        rand_mask = torch.rand_like(batch['seq_noise']) < batch['seq_noise']
        rand_mask &= batch['mol_type'] == 0 # corrupt protein sequences only

        mask = rand_mask | ~batch["seq_mask"].bool() # these will be input as MASK
        sup = rand_mask & batch["seq_mask"].bool() # these will be actually supervised

        noisy_batch["aatype"] = torch.where(mask, MASK_IDX, tokens)
        target["seq_supervise"] = sup.float()


        if self.cfg.reweight == 'linear':
            target["seq_supervise"] *= (1-batch['seq_noise'] + self.cfg.reweight_eps)
        elif self.cfg.reweight == 'inverse':
            target["seq_supervise"] *= 1/(batch['seq_noise'] + self.cfg.reweight_eps)
        target["aatype"] = tokens

        # noisy_batch['func_cond'] = batch['func_cond'] 
        
        mlm_mask = (
            (torch.rand_like(batch['seq_noise']) < self.cfg.mlm_prob)
            & (batch['motif_mask'] == 0) & (batch['motif_float_mask'] == 0)
            & ~mask
        )

        noisy_batch['aatype'] = torch.where(
            mlm_mask & (torch.rand_like(batch['seq_noise']) < self.cfg.mlm_ratio),
            torch.randint(0, 20, mlm_mask.shape, device=mlm_mask.device),
            noisy_batch['aatype']
        )
        target["seq_supervise"].masked_fill_(mlm_mask, self.cfg.mlm_weight)

        gen_mask = batch['seq_mask'].bool() # & (batch['motif_mask'] == 0)
        if logger:
            logger.masked_log("seq/toks", batch["seq_mask"], sum=True)
            logger.masked_log("seq/length", gen_mask.sum(-1).float(), dims=0)
            
            
            
    def embed(self, model, batch, inp):

        def _af2_idx_to_esm_idx(aa, mask):
            aa = (aa + 1).masked_fill(mask != 1, 0)
            return model.af2_to_esm[aa]

        if self.cfg.esm is not None:

            esmaa = _af2_idx_to_esm_idx(batch["aatype"], batch["pad_mask"])
            res = model.esm(esmaa, repr_layers=range(model.esm.num_layers + 1))
            esm_s = torch.stack(
                [v for _, v in sorted(res["representations"].items())], dim=2
            )
            # rep = res['representations'][model.esm.num_layers]
            esm_s = esm_s.detach().float()

            # === preprocessing ===
            esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            inp["x"] += model.esm_s_mlp(esm_s)

        inp["x"] += model.seq_embed(batch["aatype"])

        # if self.cfg.func_cond:
        #     residue_go_embeddings = model.func_embed(batch['func_cond'])
        #     inp["x"] += residue_go_embeddings # add go embeddings

        #     # look up embeddings for all GO terms in the input
        #     # shape: (batch_size, seq_len, max_depth, embedding_dim)
        #     go_term_embeddings_lookup = model.func_embed(batch['func_cond'])

        #     # sum embeddings along the 'max_depth' dimension (dimension 2)
        #     # shape: (batch_size, seq_len, embedding_dim)
        #     residue_go_embeddings = torch.sum(go_term_embeddings_lookup, dim=2)

        if self.cfg.all_atom:
            inp["x_cond"] += model.mol_type_cond(batch['mol_type'].int())
        
        
    def predict(self, model, inp, out, readout):
        readout["aatype"] = model.seq_out(out["x"])
        if not model.training:
            readout["aatype"][...,MASK_IDX:] = -np.inf # only predict proteins
        
    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        
        if logger:
            logger.masked_log("seq/loss", loss, mask=mask)
            logger.masked_log("seq/perplexity", loss, mask=mask, post=np.exp)

        return loss * mask
