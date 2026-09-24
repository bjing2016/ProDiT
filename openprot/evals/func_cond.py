from collections import defaultdict
from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
from ..tracks.sequence import MASK_IDX
from ..generate.sampler import OpenProtSampler
from ..generate.sequence import SequenceUnmaskingStepper
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..utils.prot_utils import make_ca_prot, write_ca_traj, compute_tmscore, aatype_to_seqres

import numpy as np
import torch
import os
import math
import tqdm
import shutil
import torch.nn.functional as F
import subprocess
import json
import pandas as pd

from .seq_gen import SequenceGenerationEval
from .codesign import CodesignEval

class FunctionConditioningEval(CodesignEval):
    def setup(self):
        super().setup()
        
        # Load GO vocabulary and ancestors
        with open(self.cfg.go_vocab, 'r') as f:
            self.go_vocab = json.load(f)

    def __len__(self):
        return len(self.cfg.terms) * self.cfg.num_samples

    def __getitem__(self, idx):
        
        term = self.cfg.terms[idx % len(self.cfg.terms)]
        
        L = self.cfg.sample_length
        
        func_cond = np.ones(L) * self.go_vocab[term][0]

        # Create data with function conditioning
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A"*L,
            seq_mask=np.ones(L),
            seq_noise=np.ones(L),
            struct_noise=np.ones(L) * self.cfg.struct.edm.noise_max,
            struct=np.zeros((L, 3)),
            struct_mask=np.ones(L),
            residx=np.arange(L),
            go_terms=func_cond,
        )
        data['term'] = term
        
        return data


    
    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        if world_size > 1:
            torch.distributed.barrier()

        idx = list(range(rank, len(self), world_size))
        
        
        df = defaultdict(dict)
        
        self.run_deepfri_recall(idx, rank, world_size, savedir, logger, df)

        torch.cuda.empty_cache()
        self.run_designability(idx, rank, world_size, savedir, logger, df)
        self.run_deepfri_recall(idx, rank, world_size, savedir, logger, df, refolded=True)

        # Save DataFrame with DeepFRI metrics
        df = pd.DataFrame(df).T
        df.to_csv(f"{savedir}/rank{rank}.csv")

        if world_size > 1:
            torch.distributed.barrier()

        if rank == 0:
            dfs = []
            for r in range(world_size):
                dfs.append(pd.read_csv(f"{savedir}/rank{r}.csv", index_col=0))
                # subprocess.run(["rm", f"{savedir}/rank{r}.csv"])
            df = pd.concat(dfs).sort_values('term')
            df.to_csv(f"{savedir}/info.csv")

    def run_deepfri_recall(self, idx, rank, world_size, savedir, logger, df, refolded=False):
        """
        Run DeepFRI evaluation on generated proteins and calculate metrics.
        """
        
        # Copy files to the directory
        # these must not have trailing slash!
        if refolded:
            src = f"{savedir}/rank{rank}"
            dst = f"{savedir}/rank{rank}/deepfri_refolded"   
        else:
            src = savedir
            dst = f"{savedir}/rank{rank}/deepfri"    
        os.makedirs(dst, exist_ok=True)
        for i in idx:
            name = self[i]['name']
            cmd = ['cp', f"{src}/{name}.pdb", dst]
            subprocess.run(cmd)

        # # Run DeepFRI with sequence model
        # seq_output_dir = f"{deepfri_dir}/seq_results"
        # os.makedirs(seq_output_dir, exist_ok=True)
        cwd = os.getcwd()
        cmd = [
            "bash",
            f"{cwd}/scripts/switch_conda_env.sh",
            "deepfri",
            "python",
            "predict.py",
            "--pdb_dir",
            f"{cwd}/{dst}",
            "-ont", 
            "mf",
            "--output_fn_prefix",
            f"{cwd}/{dst}",
        ]
        
        subprocess.run(cmd, cwd='../DeepFRI')

        js = json.load(open(f"{dst}_MF_pred_scores.json"))
        
        for i in idx:
            name = self[i]['name']
            term = self[i]['term']
            preds = js['Y_hat'][js['pdb_chains'].index(name)]
            score = preds[js['goterms'].index(term)]
            if logger is not None:
                logger.log(f"{self.cfg.name}/deepfri_score{'_r' if refolded else ''}", score)
            df[name]['term'] = term
            df[name]['score'+('_r' if refolded else '')] = score

            
    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):


        if self.cfg.struct.edm:
            StructureStepper = EDMDiffusionStepper
        elif self.cfg.struct.gfm:
            StructureStepper = GaussianFMStepper
        sampler = OpenProtSampler(schedules={
            'structure': self.struct_sched_fn,
            'sequence': self.seq_sched_fn,
        }, steppers=[
            StructureStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq)
        ])
        
        def model_func(noisy_batch):
            
            dup_batch = {**noisy_batch}
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    dup_batch[key] = torch.stack([
                        dup_batch[key].clone(), 
                        dup_batch[key].clone()
                    ])
            dup_batch['go_terms'][0] = 0
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    shape = dup_batch[key].shape
                    new_shape = (shape[0] * shape[1], *shape[2:])
                    dup_batch[key] = dup_batch[key].reshape(new_shape)
            
            out, readout = model.forward(dup_batch)        
            
            for key in readout:
                if type(readout[key]) is torch.Tensor:
                    shape = readout[key].shape
                    new_shape = (2, shape[0] // 2, *shape[1:])
                    readout[key] = readout[key].reshape(new_shape)
                
            new_readout = {}
            for key in readout:
                new_readout[key] = readout[key][0].clone()
                new_readout[key] += self.cfg.guidance * (readout[key][1] - readout[key][0])
            new_readout['aatype'][...,-1] = -np.inf # otherwise nan
            
            return out, new_readout
        
        
        sample, extra = sampler.sample(
            model_func,
            noisy_batch,
            self.cfg.steps,
        )
        
        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        B = len(sample['struct'])

        batch['struct'] = sample['struct']
        batch['aatype'] = sample['aatype']
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):
            
            prot = make_ca_prot(
                sample['struct'][i].cpu().numpy(),
                sample["aatype"][i].cpu().numpy(),
                batch["struct_mask"][i].cpu().numpy(),
            )
            
            ref_str = protein.to_pdb(prot)
            name = batch["name"][i]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(ref_str)
    
            # with open(f"{savedir}/{name}_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            # with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))

            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            # with open(f"{savedir}/{name}_traj.fasta", "w") as f:
            #     for seqs in extra['seq_traj']:
            #         seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
            #         seq = seq.replace('X', '-')
            #         f.write(seq+'\n')
