from .codesign import CodesignEval
import pandas as pd
import pickle, os
import torch
from ..utils.misc_utils import temp_seed
import numpy as np
#from rdkit import Chem
from ..utils import residue_constants as rc
from ..data.data import OpenProtData
from ..generate.sampler import OpenProtSampler
from ..utils.prot_utils import make_ca_prot, aatype_to_seqres
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..generate.sequence import SequenceUnmaskingStepper
from ..utils.prot_utils import write_mmcif, aatype_to_seqres, compute_tmscore
from ..utils import protein 
from ..utils.geometry import rmsdalign, compute_rmsd, compute_lddt
from ..utils.motif_utils import load_motif_spec, sample_motif_mask, save_motif_pdb
from ..utils.structure import Structure, Polypeptide, Ligand
from collections import defaultdict
from biopandas.pdb import PandasPdb
import subprocess

def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class MotifEval(CodesignEval):
    def __len__(self):
        return len(self.cfg.path) * self.cfg.num_samples
        
    def __getitem__(self, idx):

        path = self.cfg.path[idx % len(self.cfg.path)]
        spec = load_motif_spec(path)
        masks = sample_motif_mask(spec)
        motif_mask = masks['sequence']
        motif_idx = masks['group']
        with open(path) as f:
            prot = protein.from_pdb_string(f.read())

        L = len(motif_mask)
        motif_ca = np.zeros((L, 3))
        motif_ca[motif_mask] = prot.atom_positions[:,1]
        motif_aatype = np.zeros(L, dtype=int)
        motif_aatype[motif_mask] = prot.aatype

        name = path.split('/')[-1].split('.')[0]
        data = self.make_data(
            name=f"{name}_sample{idx // len(self.cfg.path)}",
            seqres=aatype_to_seqres(motif_aatype),
            seq_mask=np.ones(L),
            seq_noise=~motif_mask,
            struct_noise=np.ones(L) * self.cfg.struct.edm.sigma_max,
            struct=np.zeros((L, 3)),
            struct_mask=np.ones(L),
            motif=masked_center(motif_ca, motif_mask),
            motif_mask=motif_mask,
            motif_idx=motif_idx,
            residx=np.arange(L),
        )
        data['path'] = path
        return data

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        if world_size > 1:
            torch.distributed.barrier()

        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

        df = defaultdict(dict) 
        
        if self.cfg.run_designability:
            torch.cuda.empty_cache()
            self.run_designability(idx, rank, world_size, savedir, logger, df)
                   
        # if self.cfg.run_secondary:
        #     self.run_secondary(idx, rank, world_size, savedir, logger, df)

        # if self.cfg.run_diversity:
        #     self.run_diversity(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_pmpnn_designability:
            torch.cuda.empty_cache()
            self.run_pmpnn_designability(idx, rank, world_size, savedir, logger, df)
        
        # this has to be last
        self.save_df(idx, rank, world_size, savedir, logger, df)
        # if self.cfg.run_plot and rank == 0:
        #     self.make_plot(idx, rank, world_size, savedir, logger, df)


    def run_designability(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            name = self[i]['name']
            cmd = ['cp', f"{savedir}/{name}.fasta", f"{savedir}/rank{rank}"]
            subprocess.run(cmd)
                
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "eval",
            "python",
            "-m",
            "scripts.esmfold",
            "--outdir",
            f"{savedir}/rank{rank}",
            "--dir",
            f"{savedir}/rank{rank}",
            # "--print",
            "--device",
            str(torch.cuda.current_device())
        ]
        out = subprocess.run(cmd) 
        
        for i in idx:
            name = self[i]['name']
            with open(f"{savedir}/{name}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            with open(f"{savedir}/rank{rank}/{name}.pdb") as f:
                pred = protein.from_pdb_string(f.read())

            with open(f"{savedir}/{name}_motif.pdb") as f:
                motif = protein.from_pdb_string(f.read())

            
            motif_rmsd = []
            for j in np.unique(motif.segment_index):
                motif_residx = motif.residue_index[motif.segment_index == j]
                motif_rmsd.append(compute_rmsd(
                    torch.from_numpy(pred.atom_positions[motif_residx-1,1]),
                    torch.from_numpy(prot.atom_positions[motif_residx-1,1]),
                ))
            motif_rmsd = torch.stack(motif_rmsd).max()
                            
            lddt = compute_lddt(
                torch.from_numpy(pred.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_mask[:,1])
            )
            rmsd = compute_rmsd(
                torch.from_numpy(pred.atom_positions[:,1]),  
                torch.from_numpy(prot.atom_positions[:,1])
            )
            tmscore = compute_tmscore(  # second is reference
                coords1=pred.atom_positions[:,1],
                coords2=prot.atom_positions[:,1],
            )['tm']

            plddt = PandasPdb().read_pdb(f"{savedir}/rank{rank}/{name}.pdb").df['ATOM']['b_factor'].mean()
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/sclddt", lddt)
                logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                logger.log(f"{self.cfg.name}/scTM", tmscore)
                logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
                logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
                logger.log(f"{self.cfg.name}/plddt", plddt)
                logger.log(f"{self.cfg.name}/mRMSD", motif_rmsd)
                logger.log(
                    f"{self.cfg.name}/success", 
                    (rmsd < 2).float() * (motif_rmsd < 1).float()
                )

            df[f"sample{i}"]["plddt"] = plddt
            df[f"sample{i}"]["scrmsd"] = float(rmsd)
            df[f"sample{i}"]["sctm"] = tmscore
            df[f"sample{i}"]["sclddt"] = float(lddt)
            df[f"sample{i}"]["mRMSD"] = float(motif_rmsd)
    

    def run_pmpnn_designability(self, idx, rank, world_size, savedir, logger, df):


        os.makedirs(f"{savedir}/rank{rank}/pmpnn/pdbs", exist_ok=True)
        os.makedirs(f"{savedir}/rank{rank}/pmpnn/motif_pdbs", exist_ok=True)
        for i in idx:
            name = self[i]['name']
            
            cmd = [
                'cp',
                f"{savedir}/{name}.pdb",
                f"{savedir}/rank{rank}/pmpnn/pdbs"
            ]
            subprocess.run(cmd)
            cmd = [
                'cp',
                f"{savedir}/{name}_motif.pdb",
                f"{savedir}/rank{rank}/pmpnn/motif_pdbs/{name}.pdb"
            ]
            subprocess.run(cmd)
            
        cmd = [
            "bash",
            "scripts/run_genie_motif_pipeline.sh",
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
        pmpnn_df["success"] = (pmpnn_df["scRMSD"] < 2) & (pmpnn_df.loc[name].motif_ca_rmsd < 1)
        if logger is not None:
            for col in pmpnn_df.columns:
                for val in pmpnn_df[col].tolist():
                    logger.log(f"{self.cfg.name}/pmpnn_{col}", val)
        for i in idx:
            name = self[i]['name']
            df[name]["pmpnn_scrmsd"] = pmpnn_df.loc[name].scRMSD
            df[name]["pmpnn_scTM"] = pmpnn_df.loc[name].scTM
            df[name]["pmpnn_pLDDT"] = pmpnn_df.loc[name].pLDDT
            df[name]["pmpnn_mRMSD"] = pmpnn_df.loc[name].motif_ca_rmsd
            
    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):
        
        schedules = {
            'structure': self.struct_sched_fn,
            'sequence': self.seq_sched_fn,
        }

        mask = batch['seq_noise'].bool()
        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq, mask=mask)
        ])

        def model_func(noisy_batch):
            mul = noisy_batch['motif_idx'].max().int().item()
            dup_batch = {**noisy_batch}
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    dup_batch[key] = torch.stack([
                        dup_batch[key].clone() for _ in range(mul+1)
                    ])
            for i in range(mul+1):
                dup_batch['motif_mask'][i] = noisy_batch['motif_mask'] * (noisy_batch['motif_idx'] == i)
                dup_batch['motif'][i] *= (noisy_batch['motif_idx'] == i)[...,None]
            dup_batch['motif_idx'][:] = 0
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    shape = dup_batch[key].shape
                    new_shape = (shape[0] * shape[1], *shape[2:])
                    dup_batch[key] = dup_batch[key].reshape(new_shape)
            
            out, readout = model.forward(dup_batch)        
            
            for key in readout:
                if type(readout[key]) is torch.Tensor:
                    shape = readout[key].shape
                    new_shape = (mul+1, shape[0] // (mul+1), *shape[1:])
                    readout[key] = readout[key].reshape(new_shape)
                
            
            new_readout = {}
            for key in readout:
                new_readout[key] = readout[key][0].clone()
                for i in range(mul):
                    new_readout[key] += readout[key][i+1] - readout[key][0]
            new_readout['aatype'][...,-1] = -np.inf # otherwise nana
            
            return out, new_readout
        
        sample, extra = sampler.sample(
            model_func,
            noisy_batch,
            self.cfg.steps
        )

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        batch['struct'] = sample['struct']
        batch['aatype'] = sample['aatype']
        
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):

            data.update_seqres()
            name = data["name"]

            
            mask = data['motif_mask'].bool()
            ref_motif = data['motif'][mask]
            samp_motif = data['struct'][mask]
            motif_idx = data['motif_idx'][mask]

            rmsds = []
            for i in torch.unique(motif_idx):
                rmsds.append(compute_rmsd(
                    ref_motif[motif_idx == i],
                    samp_motif[motif_idx == i]
                ))
            rmsd = torch.stack(rmsds).max()
            if logger is not None:
                logger.log(f"{self.cfg.name}/samp_mRMSD", rmsd)

            prot = make_ca_prot(
                data['struct'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["struct_mask"].cpu().numpy(),
            )
            name = data["name"]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(protein.to_pdb(prot))

            seq = aatype_to_seqres(data["aatype"])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")
                
            save_motif_pdb(
                data['path'], 
                data["motif_mask"].cpu().numpy(),
                f"{savedir}/{name}_motif.pdb"
            )
            

            


        
