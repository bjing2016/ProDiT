from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils.prot_utils import compute_tmscore
from ..utils import residue_constants as rc
from ..tracks.sequence import MASK_IDX
from ..generate.sampler import OpenProtSampler
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
import torch
import os
import math
import tqdm
import shutil
import torch.nn.functional as F
import subprocess

from biopandas.pdb import PandasPdb

class SequenceGenerationEval(OpenProtEval):
    def setup(self):
        pass

    def run(self, model):
        NotImplemented

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A"*L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
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

        idx = list(range(rank, self.cfg.num_samples, world_size))
        if self.cfg.get('run_esmfold', True):
            os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)
            for i in idx:
                shutil.copy(f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}")
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
                "--print",
            ]
            cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cvd:
                dev = cvd.split(',')[torch.cuda.current_device()]
            else:
                dev = torch.cuda.current_device()
            out = subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  
            for i in idx:
                try:
                    plddt = PandasPdb().read_pdb(f"{savedir}/sample{i}.pdb").df['ATOM']['b_factor'].mean()
                    if logger is not None:
                        logger.log(f"{self.cfg.name}/plddt", plddt)
                except:
                    pass

        ## the proteins must have been folded first
        if world_size > 1:
            torch.distributed.barrier()

        if rank == 0 and self.cfg.get('run_vendi', True):
            cmd = [
                'python',
                '-m',
                'scripts.compute_vendi',
                '--dir',
                savedir,
                '--count',
                str(len(self)),
                '--num_workers=100',
            ]
            print(' '.join(cmd))
            out = subprocess.run(cmd + ['--out', 'tmscore.npy'])
            try:
                tm_arr = np.load(f"{savedir}/tmscore.npy")
                tm_arr = tm_arr/2 + tm_arr.T/2
                eigvals = np.linalg.eigvals(tm_arr / len(tm_arr))
                vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
            except Exception as e:
                vendi = np.nan
            if logger is not None:
                logger.log(f"{self.cfg.name}/vendi", vendi)

            # print(' '.join(cmd))
            # out = subprocess.run(cmd + ['--seq', '--out', 'tmscore_seq.npy'])
            # try:
            #     tm_arr = np.load(f"{savedir}/tmscore_seq.npy")
            #     tm_arr = tm_arr/2 + tm_arr.T/2
            #     eigvals = np.linalg.eigvals(tm_arr / len(tm_arr))
            #     vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
            # except Exception as e:
            #     vendi = np.nan
            # if logger is not None:
            #     logger.log(f"{self.cfg.name}/vendi_seq", vendi)

            print(' '.join(cmd))
            out = subprocess.run(cmd + ['--exc=TMalign', '--out', 'tmalign.npy'])
            try:
                tm_arr = np.load(f"{savedir}/tmalign.npy")
                tm_arr = tm_arr/2 + tm_arr.T/2
                eigvals = np.linalg.eigvals(tm_arr / len(tm_arr))
                vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
            except Exception as e:
                vendi = np.nan
            if logger is not None:
                logger.log(f"{self.cfg.name}/vendi_tmalign", vendi)


    
    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):


        sampler = OpenProtSampler(schedules={
            'sequence': lambda t: 1-t,
        }, steppers=[
            SequenceUnmaskingStepper(self.cfg)
        ])
        
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)
        B = len(sample['aatype'])
        for i in range(B):
            name = batch["name"][i]

            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            if logger is not None:
                logger.log(f"{self.cfg.name}/seqent", self.compute_sequence_entropy(seq))

            with open(f"{savedir}/{name}_traj.fasta", "w") as f:
                for seqs in extra['seq_traj']:
                    seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
                    seq = seq.replace('X', '-')
                    f.write(seq+'\n')
                
