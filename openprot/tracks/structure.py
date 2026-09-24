import torch
import torch.nn as nn
from .track import OpenProtTrack
from ..utils.geometry import (
    atom37_to_frames,
    atom37_to_torsions,
    compute_fape,
    gram_schmidt,
    compute_pade,
    compute_lddt,
    compute_rmsd,
    rmsdalign,
    compute_pseudo_tm,
)
from ..utils.rotation_conversions import axis_angle_to_matrix, random_rotations
from ..utils.rigid_utils import Rigid, Rotation
from ..utils import residue_constants as rc
from ..generate.diffusion import EDMDiffusion, GaussianFM
from functools import partial

import numpy as np
import math

def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale) + shift
    else:
        return x

def gate(x, gate_):
    if gate_ is not None:
        return x * gate_
    else:
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, dim, out):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
        
def sinusoidal_embedding(pos, n_freqs, max_period, min_period):
    periods = torch.exp(
        torch.linspace(
            math.log(min_period), math.log(max_period), n_freqs, device=pos.device
        )
    )
    freqs = 2 * np.pi / periods
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )


class StructureTrack(OpenProtTrack):

    def setup(self):
        if self.cfg.edm:
            self.diffusion = EDMDiffusion(self.cfg.edm)
        elif self.cfg.gfm:
            self.diffusion = GaussianFM(self.cfg.gfm)

    def tokenize(self, data):
        pass 

    def add_modules(self, model):
        model.trans_in = nn.Linear(3, model.cfg.dim)
        if self.cfg.readout_adaLN:
            model.trans_out = FinalLayer(model.cfg.dim, 3)
        else:
            model.trans_out = nn.Sequential(
                nn.LayerNorm(model.cfg.dim), nn.Linear(model.cfg.dim, 3)
            )
        
        if self.cfg.embed_motif:
            model.motif_in = nn.Linear(3, model.cfg.dim)
            model.motif_cond = nn.Parameter(torch.zeros(model.cfg.dim))

        if self.cfg.embed_float_motif:
            model.float_motif_in = nn.Linear(3, model.cfg.dim)
            model.float_motif_cond = nn.Parameter(torch.zeros(model.cfg.dim))
        
        if self.cfg.embed_ligand:
            model.ligand_cond = nn.Parameter(torch.zeros(model.cfg.dim))
        
        if self.cfg.embed_nma:
            model.nma_in = nn.Linear(3, model.cfg.dim)
                                    
            
        
    def corrupt(self, batch, noisy_batch, target, logger=None):

        noisy, target_tensor = self.diffusion.add_noise(
            batch["struct"],
            batch["struct_noise"],
        ) 
        if self.cfg.get('embed_mask_as_noise', True):
            embed_as_mask = (
                (batch["struct_noise"] >= self.diffusion.cfg.noise_max-0.01) | 
                (~batch["struct_mask"].bool())
            )
    
            noisy = torch.where(
                embed_as_mask[...,None], 
                self.diffusion.cfg.sigma_max * torch.randn_like(batch["struct"]),
                noisy
            )
            noise_level = torch.where(
                batch['struct_mask'].bool(),
                batch["struct_noise"],
                self.diffusion.cfg.noise_max,
            ) # not present will have adaLN conditioning like max noise
            # I think this should be a no-op though
            
            # overwrite
            noisy_batch['struct_noise'] = target['struct_noise'] = noise_level
        
        
        noisy_batch["struct"] = noisy

        
        # struct exists but is not ligand
        target["struct_supervise"] = (
            batch['struct_mask'].bool() #  struct exists
            & (batch['struct_noise'] > self.cfg.edm.sigma_min+1e-6) # nonzero noise
        )
        
        target["struct"] = target_tensor 

        if logger:
            logger.masked_log("struct/toks", batch["struct_mask"], sum=True)
            logger.masked_log("struct/motif_toks", batch["motif_mask"], sum=True)
        
    def embed(self, model, batch, inp):

        coords = batch["struct"]
        noise = batch["struct_noise"]

        # linear embed coords if specified
        precond = self.diffusion.precondition(coords, noise)
        if self.cfg.get('embed_mask_as_noise', True):
            inp["x"] += torch.where(
                ~batch['motif_float_mask'][...,None].bool(),
                model.trans_in(precond),
                0.0,
            )
        else:
            inp["x"] += torch.where(
                batch['struct_mask'][...,None].bool(),
                model.trans_in(precond),
                0.0,
            )
            
        if self.cfg.embed_motif: # now motifs
            inp["x"] += torch.where(
                batch['motif_mask'][...,None].bool(),
                model.motif_in(batch['motif']),
                0.0
            )
            inp["x_cond"] += torch.where(
                batch['motif_mask'][...,None].bool(),
                model.motif_cond,
                0.0
            )
        if self.cfg.embed_float_motif: # now motifs
            inp["x"] += torch.where(
                batch['motif_float_mask'][...,None].bool(),
                model.float_motif_in(batch['motif']),
                0.0
            )
            inp["x_cond"] += torch.where(
                batch['motif_float_mask'][...,None].bool(),
                model.float_motif_cond,
                0.0
            )
        if self.cfg.embed_ligand:
            inp["x_cond"] += torch.where(
                batch['ligand_mask'][...,None].bool(),
                model.ligand_cond,
                0.0
            )
        if self.cfg.embed_nma:
            inp["x"] += torch.where(
                batch['motif_nma_mask'][...,None].bool(),
                model.nma_in(batch['motif_nma']),
                0.0
            )

        # embed sigma
        def sigma_transform(noise_level):
            t_emb = noise_level ** (1/self.diffusion.cfg.sched_p)
            t_emb = t_emb / self.diffusion.cfg.noise_max ** (1/self.diffusion.cfg.sched_p)
            return t_emb
            
        t_emb = sigma_transform(noise) 
        t_emb = sinusoidal_embedding(t_emb, model.cfg.dim // 2, 1, 0.01).float()
        
        if self.cfg.get('embed_mask_as_noise', True):
            inp["x_cond"] += torch.where(
                ~batch['motif_float_mask'][...,None].bool(), t_emb, 0.0
            )
            
        else:
            inp["x_cond"] += t_emb
                
        inp["postcond_fn"] = lambda x: self.diffusion.postcondition(
            coords, x, noise,
        )
        
    def predict(self, model, inp, out, readout):
        if self.cfg.readout_adaLN:
            readout["trans"] = inp["postcond_fn"](
                model.trans_out(out["x"], inp["x_cond"])
            )
        else:
            readout["trans"] = inp["postcond_fn"](model.trans_out(out["x"]))

    def compute_loss(self, readout, target, logger=None, **kwargs):


        mse = self.compute_mse_loss(
            readout, target, aligned=False, logger=logger
        )
        aligned_mse = self.compute_mse_loss(
            readout, target, aligned=True, logger=logger
        )
        soft_lddt = self.compute_lddt_loss(
            readout, target, logger=logger
        )

        loss = 0
        loss = self.cfg.loss['aligned_mse'] * aligned_mse 
        loss += self.cfg.loss['lddt'] * soft_lddt
        loss += self.cfg.loss['mse'] * mse

        self.compute_metrics(readout, target, logger=logger)
        mask = target["struct_supervise"]
        if logger:
            logger.masked_log("struct/toks_sup", mask, sum=True)
            logger.masked_log("struct/loss", loss, mask=mask)

        loss = loss * mask

        return loss

    @torch.no_grad()
    def compute_metrics(self, readout, target, logger=None):
        
        lddt = compute_lddt(
            readout["trans"], target["struct"], target["struct_supervise"]
        )
        rmsd = compute_rmsd(
            readout["trans"], target["struct"], target["struct_supervise"]
        )
        pseudo_tm = compute_pseudo_tm(
            readout["trans"], target["struct"], target["struct_supervise"]
        )
        
        mask = torch.any(target['struct_supervise'].bool(), -1).float()
        if logger:
            logger.masked_log("struct/lddt", lddt, mask=mask, dims=0)
            logger.masked_log("struct/rmsd", rmsd, mask=mask, dims=0)
            logger.masked_log("struct/pseudo_tm", pseudo_tm, mask=mask, dims=0)
            
    def compute_lddt_loss(self, readout, target, logger=None, eps=1e-5):

        pred = readout["trans"]
        gt = target["struct"]
        mask = (target["struct_supervise"] > eps).float() # lddt loss mask is binary
        soft_lddt = compute_lddt(pred, gt, mask, soft=True, reduce=(-1,))

        soft_lddt_loss = 1 - soft_lddt  # [9, B, L]

        if logger:
            logger.masked_log("struct/lddt_loss", soft_lddt_loss, mask=mask)
            

        return soft_lddt_loss
        
    def compute_mse_loss(self, readout, target, logger=None, aligned=False, eps=1e-5):

        pred = readout["trans"]
        gt = target["struct"]
        
        mask = target["struct_supervise"] # weighted mse!
        t = target["struct_noise"]
        
        mse = self.diffusion.compute_loss(
            pred=pred,
            target=gt,
            t=t,
            mask=mask, # used for alignment
            aligned=aligned,
        ) 
       
        if aligned: key = 'aligned_mse'
        else: key = 'mse'
        if logger:
            logger.masked_log(f"struct/{key}_loss", mse, mask=mask)
            logger.masked_log(f"struct/lig/{key}_loss", mse, mask=mask * (target['mol_type'] == 3).float())
        
        return mse


        