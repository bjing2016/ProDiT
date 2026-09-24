from .eval import OpenProtEval
from ..utils import protein
from ..utils.prot_utils import aatype_to_seqres
from ..utils.geometry import compute_lddt, compute_rmsd
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
import torch
import os
import math
import torch.nn.functional as F
import pandas as pd
import subprocess
from torch.distributions.categorical import Categorical
from biopandas.pdb import PandasPdb

class InverseFoldingEval(OpenProtEval):
    def setup(self):
        if self.cfg.split:
            self.df = pd.read_csv(self.cfg.split, index_col="name")
            if self.cfg.limit:
                self.df = self.df[:self.cfg.limit]
            

    def run(self, model):
        NotImplemented

    def __len__(self):
        if self.cfg.split:
            return len(self.df)
        else:
            return self.cfg.limit

    def get_name(self, idx):
        if self.cfg.split:
            name = self.df.index[idx]
        else:
            name = f"sample{idx}"
        return name
    def __getitem__(self, idx):
        name = self.get_name(idx)
        # prot = dict(
        #     np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        # )
        with open(f"{self.cfg.path}/{name}.pdb") as f:
            prot = protein.from_pdb_string(f.read())
            
        # seqres = self.df.seqres[name]
        seqres = aatype_to_seqres(prot.aatype)
        L = len(seqres)
        data = self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            seq_noise=np.ones(L, dtype=np.float32),
            struct=prot.atom_positions[:,1].astype(np.float32),
            struct_mask=prot.atom_mask[:,1].astype(np.float32),
            residx=np.arange(L, dtype=np.float32),
        )
        
    
        return data

    def compute_sequence_entropy(self, seq):
        p = np.zeros(21)
        for s in seq:
            p[rc.restype_order_with_x[s]] += 1
        p /= p.sum()
        return np.e ** (-np.nansum(p * np.log(p)))
        
    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        
        torch.cuda.empty_cache()

        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)
        if self.cfg.run_pmpnn:
            os.makedirs(f"{savedir}/rank{rank}/pmpnn/pdbs", exist_ok=True)
            
        for i in idx:
            name = self.get_name(i)
            cmd = ['cp', f"{savedir}/{name}.fasta", f"{savedir}/rank{rank}"]
            subprocess.run(cmd)
            if self.cfg.run_pmpnn:
                name = self.get_name(idx)
                cmd = [
                    'cp',
                    f"{savedir}/sample{i}.pdb",
                    f"{savedir}/rank{rank}/pmpnn/pdbs"
                ]
                subprocess.run(cmd)
            
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "eval",
            "python",
            "-m",
            "scripts.esmfold",
            "--outdir",
            savedir,
            "--dir",
            f"{savedir}/rank{rank}",
            # "--print",
            "--device",
            str(torch.cuda.current_device())
        ]
        
        out = subprocess.run(cmd)  
        
        for i in idx:
            name = self.get_name(i)
            with open(f"{self.cfg.path}/{name}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            with open(f"{savedir}/{name}.pdb") as f:
                pred = protein.from_pdb_string(f.read())
            lddt = compute_lddt(
                torch.from_numpy(pred.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_mask[:,1])
            )
            rmsd = compute_rmsd(
                torch.from_numpy(pred.atom_positions[:,1]),  
                torch.from_numpy(prot.atom_positions[:,1])
            )
            if logger is not None:
                logger.log(f"{self.cfg.name}/sclddt", lddt)
                logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())

        if self.cfg.run_pmpnn:
            cmd = [
                "bash",
                "scripts/run_genie_pipeline.sh",
                f"{savedir}/rank{rank}/pmpnn",
            ]
            cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cvd:
                dev = cvd.split(',')[torch.cuda.current_device()]
            else:
                dev = torch.cuda.current_device()
            subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  
    
            pmpnn_df = pd.read_csv(
                f"{savedir}/rank{rank}/pmpnn/info.csv", index_col="domain"
            )
            pmpnn_df["designable"] = pmpnn_df["scRMSD"] < 2
            if logger is not None:
                for col in pmpnn_df.columns:
                    for val in pmpnn_df[col].tolist():
                        logger.log(f"{self.cfg.name}/pmpnn_{col}", val)
            

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):
        # have to maintain train-test alignment
        noisy_batch['struct_noise'].fill_(self.cfg.sigma) 
             
        sampler = OpenProtSampler(schedules={
            'sequence': lambda t: 1-t,
        }, steppers=[
            SequenceUnmaskingStepper(self.cfg)
        ])

        L = self.cfg.steps or noisy_batch['struct_noise'].shape[-1]
        sample, extra = sampler.sample(model, noisy_batch, L)

        B = len(batch['name'])
        for i in range(B):
            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            
            recov = (sample['aatype'][i] == batch['aatype'][i]).float()
            recov = (recov * batch['struct_mask'][i]).sum() / batch['struct_mask'][i].sum()
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/recov", recov.item())
            
            name = batch['name'][i]
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            print(seq, recov)
