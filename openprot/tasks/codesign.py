from .task import OpenProtTask
import numpy as np
import math
from ..utils import residue_constants as rc
from ..generate.diffusion import t_to_sigma
from scipy.spatial.transform import Rotation as R
import torch, scipy
import os
from ..data.data import OpenProtData

class CodesignTask(OpenProtTask):
            
    def sample_motifs(self, data, crop=None):
        N = len(data['seqres'])
        Ns = np.random.randint(1, self.cfg.motif.nmax)
        Nr = np.random.randint(
            int(math.floor(N * self.cfg.motif.ymin)),
            int(math.ceil(N * self.cfg.motif.ymax)) + 1,
        )
        if Ns > Nr: return # segments cannot exceed residues
        
        B = [
            0,
            *sorted(np.random.choice(np.arange(1, Nr), size=Ns-1, replace=False)),
            Nr
        ]
        L = np.diff(B)
        class Motif(list):
            def __init__(self, l):
                super().__init__(l)
                self.rand = np.random.rand()
            
        M = [[False] for _ in range(N-Nr)] + [Motif([True]*l) for l in L]
        np.random.shuffle(M)
        
        is_motif = np.array([i for a in M for i in a])
        rand = np.array([getattr(a, 'rand', 0) for a in M for i in a])
        
        data['motif'][is_motif] = data['struct'][is_motif]
        data['motif_mask'][is_motif] = data['struct_mask'][is_motif]
        
        if np.random.rand() < self.cfg.motif.multi_prob:
            data['motif_idx'][is_motif] = (rand < 0.5).astype(np.float32)[is_motif]

        if np.random.rand() < self.cfg.motif.nma_prob:
            self.compute_motif_nma(data)

        if np.random.rand() < self.cfg.motif.float_prob:
            self.make_float_motif(data, crop=crop)

    def make_float_motif(self, data, crop=None):
        L = len(data['struct'])
        M = data['motif_mask'].sum()
        if L + M > crop:
            old_motif_mask =  data['motif_mask']
            cutoff = (np.cumsum(1 + data['motif_mask']) <= crop).sum() - 1
            data.crop(idx=np.arange(int(cutoff)))
            
            L = len(data['struct'])
            M = data['motif_mask'].sum()
            
        data.pad(int(L+M))
        idx = np.argwhere(data['motif_mask']).flatten()
        data['seqres'] = data['seqres'][:L] + ''.join([data['seqres'][i] for i in idx])
        data['seq_mask'][L:] = data['seq_mask'][data['motif_mask'].astype(bool)]
        data['motif'][L:] = data['motif'][data['motif_mask'].astype(bool)]
        data['motif_float_mask'][L:] = 1
        data['go_terms'][:] = data['go_terms'][0]
        data['go_noise'][:] = data['go_noise'][0]

        data['motif'][:L] = 0.
        data['motif_mask'][:] = 0.
        data['motif_idx'][:] = 0. # incompatible with multi-motifs atm

        del data['pad_mask']
        
    def compute_motif_nma(self, data):

        idx = np.unique(data['motif_idx'])
        for i in idx: 
            mask = (data['motif_mask'] == 1) & (data['motif_idx'] == i)
            if mask.sum() == 0: continue
            nma = data['nma'][mask]
            nma = nma - nma.mean(0)
            data['motif_nma'][mask] = nma
            data['motif_nma_mask'][mask] = data['nma_mask'][mask]
            
    def center_random_rot(self, data, eps=1e-6):
        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)

        # if self.cfg.struct.get('ligand_center', False):
        #     K = data['ligand']['struct_mask'].sum()
        #     if K > 0:
        #         pos = data['ligand']["struct"]
        #         mask = data['ligand']["struct_mask"][..., None]
        #         com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        # else:            
        #     com += np.random.randn(3) * self.cfg.edm.sigma_perturb

        data["struct"] -= com
        data['ligand']["struct"] -= com

        randrot = R.random().as_matrix()
        data["struct"] @= randrot.T
        data['ligand']["struct"] @= randrot.T

        idx = np.unique(data['motif_idx'])
        for i in idx: 
            # note that multi-residue ligands will be wrong
            # this also handles motifs, so far correct but brittle
            motif = data["motif"][data['motif_idx'] == i]
            motif_mask = data['motif_mask'][data['motif_idx'] == i][...,None]
            motif -= (motif * motif_mask).sum(-2) / (motif_mask.sum(-2) + eps)
            randrot = R.random().as_matrix()
            motif @= randrot.T
            data["motif"][data['motif_idx'] == i] = motif
            data["motif_nma"][data['motif_idx'] == i] @= randrot.T
            
        
    def add_sequence_noise(self, data, noise_level=None):
        
        def sample_noise_level():
            rand = np.random.rand()
            probs = [
                self.cfg.seq.get('zero_prob', 0),
                self.cfg.seq.get('max_prob', 0),
                self.cfg.seq.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.seq.beta)
            return noise_level
        
        L = len(data["seqres"])

        if noise_level is None:
            noise_level = sample_noise_level()
        
        data["seq_noise"] = np.where( # not ligand or motif
            (data['motif_mask'] == 0), # & (data['ligand_mask'] == 0),
            noise_level,
            0,
        ).astype(np.float32)

    def add_structure_noise(self, data, eps=1e-6, noise_level=None):
        

        def sample_noise_level():
            rand = np.random.rand()
            
            probs = [
                self.cfg.struct.get('zero_prob', 0),
                self.cfg.struct.get('max_prob', 0),
                self.cfg.struct.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.struct.beta)
            
            return noise_level
        
        L = len(data["seqres"])

        if noise_level is None:
            noise_level = sample_noise_level()

        if self.cfg.get('max_noise_as_mask', False) and noise_level == 1.0:
            data['struct_mask'][:] = 0

        if self.cfg.edm:
            data["struct_noise"][:] = t_to_sigma(self.cfg.edm, noise_level)
        else:
            data["struct_noise"][:] = noise_level
        return noise_level
   

class Codesign(CodesignTask):
    def register_loss_masks(self):
        return ["/codesign"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        rand = np.random.rand()
        
        if rand < self.cfg.motif_prob:
            self.sample_motifs(data, crop=crop)
                
        self.add_sequence_noise(data)
        self.add_structure_noise(data)

        # this must happen after motifs have been assigned
        self.center_random_rot(data)

        # only generate random numbers for unique GO indices in data
        # go_terms_array = data['go_terms'] 
        # num_unique_terms = len(np.unique(go_terms_array[go_terms_array > 0]))
        # data["_go_noise"] = np.array(np.random.rand(num_unique_terms), dtype=np.float32)

        data["go_noise"][:] = np.random.rand() < 0.15 # drop 15% the time
        
        # # only generate random numbers for unique site indices in data
        # site_array = data['sites'] 
        # num_unique_sites = len(np.unique(site_array[site_array > 0]))
        # data["_site_noise"] = np.array(np.random.rand(num_unique_sites), dtype=np.float32)
        data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
    