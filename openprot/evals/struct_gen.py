from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj, seqres_to_aatype
from ..utils.geometry import compute_lddt, rmsdalign
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
import numpy as np
from ..utils.prot_utils import compute_tmscore
from ..tasks import StructurePrediction
import torch
import os, tqdm, math, subprocess
import pandas as pd
from .codesign import CodesignEval
from collections import defaultdict


class StructureGenerationEval(CodesignEval):
    def setup(self):
        pass

    def run(self, model):
        NotImplemented

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        max_noise = self.cfg.struct.edm.sigma_max
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A" * L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            struct_noise=np.ones(L, dtype=np.float32) * max_noise,
            struct=np.zeros((L, 3), dtype=np.float32),
            struct_mask=np.ones(L, dtype=np.float32),
            residx=np.arange(L, dtype=np.float32),
        )
        return data


    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):

        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

        df = defaultdict(dict) 
                   
        if self.cfg.run_secondary:
            self.run_secondary(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_pmpnn_designability:
            torch.cuda.empty_cache()
            self.run_pmpnn_designability(idx, rank, world_size, savedir, logger, df)
        
        # this has to be last
        self.save_df(idx, rank, world_size, savedir, logger, df)
        if world_size > 1:
            torch.distributed.barrier()

        if self.cfg.run_diversity and rank == 0:
            self.run_diversity(idx, rank, world_size, savedir, logger, df)

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):

        def edm_sched_fn(t):
            p = self.cfg.struct.edm.sched_p
            sigma_max = self.cfg.struct.edm.sigma_max
            sigma_min = self.cfg.struct.edm.sigma_min 
            return (
                sigma_min ** (1 / p)
                + (1-t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
            ) ** p

        schedules = {
            'structure': edm_sched_fn,
            #'struct_weight': lambda t: (1-t) * self.cfg.struct.sde_weight.start + t * self.cfg.struct.sde_weight.end,
            #'struct_temp': lambda t: (1-t) * self.cfg.struct.temp_factor.start + t * self.cfg.struct.temp_factor.end
        }

        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct, mask=None),
        ])

    
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)


        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        B = len(sample['struct'])
        
        for i in range(B):
            if 'aatype' in batch:
                aatype = batch["aatype"][i].cpu().numpy()
            else:
                aatype = np.array(seqres_to_aatype('A'*self.cfg.sample_length))
            prot = make_ca_prot(
                sample['struct'][i].cpu().numpy(),
                aatype,
                batch["struct_mask"][i].cpu().numpy(),
            )
    
            ref_str = protein.to_pdb(prot)
            name = batch["name"][i]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(ref_str)
    
            with open(f"{savedir}/{name}_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))
