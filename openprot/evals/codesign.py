from .eval import OpenProtEval
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj, compute_tmscore, aatype_to_seqres
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
from ..tasks import StructurePrediction
import torch
import os, tqdm, math, subprocess
import pandas as pd
from biopandas.pdb import PandasPdb
from ..utils.secondary import assign_secondary_structures
from collections import defaultdict
from openprot.utils.structure import Structure, Ligand, Polypeptide

class CodesignEval(OpenProtEval):
    def setup(self):

        def gfm_sched_fn(t):
            return (10**(-2*t) - 0.01) / 0.99 + 1e-4
            # sched = np.logspace(0, -2, cfg.nsteps + 1)
            # sched = (sched - sched.min()) / (sched.max() - sched.min())

        
        def edm_sched_fn(t):
            p = self.cfg.struct.edm.sched_p
            sigma_max = self.cfg.struct.edm.sigma_max
            sigma_min = self.cfg.struct.edm.sigma_min 
            return (
                sigma_min ** (1 / p)
                + (1-t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
            ) ** p

        if self.cfg.struct.edm:
            sched_fn = edm_sched_fn
        elif self.cfg.struct.gfm:
            sched_fn = gfm_sched_fn
        def t_skew_func(t, skew):
            midpoint_y = 0.5 + skew / 2
            midpoint_x = 0.5 
            if t < midpoint_x:
                return midpoint_y / midpoint_x * t
            else:
                return midpoint_y + (1 - midpoint_y) / (1 - midpoint_x) * (t - midpoint_x)

        self.struct_sched_fn = lambda t: sched_fn(t_skew_func(t, self.cfg.skew))
        self.seq_sched_fn = lambda t: 1-t_skew_func(t, -self.cfg.skew)
        
    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        try:
            max_noise = self.cfg.struct.edm.noise_max
        except:
            max_noise = self.cfg.struct.gfm.noise_max
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A"*L,
            seq_mask=np.ones(L),
            seq_noise=np.ones(L),
            struct_noise=np.ones(L) * max_noise,
            struct=np.zeros((L, 3)),
            struct_mask=np.ones(L),
            residx=np.arange(L),
        )
        
        if self.cfg.get('ligand', False):
            lig = Ligand([self.cfg.ligand])
            lig.atoms['coords'] = lig.atoms['conformer']
            K = len(lig.atoms)
            lig = self.make_data(
                name=self.cfg.ligand,
                seqres='*'*K,
                atom_num=lig.atoms['element'],
                mol_type=np.ones(K) * lig.mol_type,
                seq_mask=np.ones(K),
                struct=lig.atoms['conformer'],
                struct_mask=np.ones(K),
                make_ligand=False,
            )
            lig['struct'] -= lig['struct'].mean(0)
            lig['struct'] += np.random.rand(3) * self.cfg.ligand_sigma
            data['ligand'] = lig
        return data

    def run_designability(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            cmd = ['cp', f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}"]
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
            "--device",
            str(torch.cuda.current_device())
        ]
        print(' '.join(cmd))
        out = subprocess.run(cmd) 
        
        for i in idx:
            
            with open(f"{savedir}/sample{i}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            with open(f"{savedir}/rank{rank}/sample{i}.pdb") as f:
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
            tmscore = compute_tmscore(  # second is reference
                coords1=pred.atom_positions[:,1],
                coords2=prot.atom_positions[:,1],
            )['tm']

            plddt = PandasPdb().read_pdb(f"{savedir}/rank{rank}/sample{i}.pdb").df['ATOM']['b_factor'].mean()
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/sclddt", lddt)
                logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                logger.log(f"{self.cfg.name}/scTM", tmscore)
                logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
                logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
                logger.log(f"{self.cfg.name}/plddt", plddt)

            df[f"sample{i}"]["plddt"] = plddt
            df[f"sample{i}"]["scrmsd"] = float(rmsd)
            df[f"sample{i}"]["sctm"] = tmscore
            df[f"sample{i}"]["sclddt"] = float(lddt)
    
    def run_diversity(self, idx, rank, world_size, savedir, logger, df):
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
        # out = subprocess.run(cmd + ['--out', 'tmscore.npy'])
        # try:
        #     tm_arr = np.load(f"{savedir}/tmscore.npy")
        #     tm_arr = tm_arr/2 + tm_arr.T/2
        #     eigvals = np.linalg.eigvals(tm_arr / len(tm_arr))
        #     vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
        # except Exception as e:
        #     vendi = np.nan
        # if logger is not None:
        #     logger.log(f"{self.cfg.name}/vendi", vendi)

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

        # print(' '.join(cmd))
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

        cmd = [
            'python',
            '-m',
            'scripts.diversity_cluster',
            '--dir',
            savedir,
        ]
        print(' '.join(cmd))
        subprocess.run(cmd)
        if os.path.exists(f"{savedir}/designable/clu_rep_seq.fasta"):
            fs_clus = len(list(open(
                f"{savedir}/designable/clu_rep_seq.fasta"
            ))) // 2
        else:
            fs_clus = 0
        if logger is not None:
            logger.log(f"{self.cfg.name}/foldseek_clusters", fs_clus)
            



    def run_secondary(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            with open(f"{savedir}/sample{i}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            ss = assign_secondary_structures(
                torch.from_numpy(prot.atom_positions[None,:,1]), 
                full=False,
                return_encodings=False
            )[0]
            
            pos = prot.atom_positions[:,1]
            L = len(pos)
            pos = pos - pos.mean(0, keepdims=True)
            ii, jj = np.meshgrid(np.arange(L), np.arange(L))
            dmat = np.square(pos[None] - pos[:,None]).sum(-1) ** 0.5
            lrc = (np.abs(ii-jj) > 12) & (dmat < 8)
            lrc = lrc.sum() / L
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/helix", ss.count('h') / len(ss))
                logger.log(f"{self.cfg.name}/sheet", ss.count('s') / len(ss))
                logger.log(f"{self.cfg.name}/loop", ss.count('-') / len(ss))
                logger.log(f"{self.cfg.name}/lrc", lrc)
            df[f"sample{i}"]["helix"] = ss.count('h') / len(ss)
            df[f"sample{i}"]["sheet"] = ss.count('s') / len(ss)
            df[f"sample{i}"]["loop"] = ss.count('-') / len(ss)
            df[f"sample{i}"]["lrc"] = lrc


    def make_plot(self, idx, rank, world_size, savedir, logger, df):
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "pymol",
            "python",
            "-m",
            "scripts.visualize",
            "--dir",
            savedir,
            "--out",
            f"{savedir}/out.png",
            "--annotate",
        ]
        out = subprocess.run(cmd) 

    def run_chai(self, idx, rank, world_size, savedir, logger, df):
        
        
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cvd:
            dev = cvd.split(',')[torch.cuda.current_device()]
        else:
            dev = torch.cuda.current_device()
        
        for i in idx:
            subprocess.run(['rm', '-r', f"{savedir}/chai/sample{i}"])
            os.makedirs(f"{savedir}/chai/sample{i}", exist_ok=True)
            cmd = [
                "bash",
                "scripts/switch_conda_env.sh",
                "chai",
                "chai-lab",
                "fold",
                f"{savedir}/sample{i}_lig.fasta",
                f"{savedir}/chai/sample{i}"
            ]
            print(' '.join(cmd))  
            subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  
            arr = np.load(f"{savedir}/chai/sample{i}/scores.model_idx_0.npz")
            
            if logger is not None:
                logger.log("chai_ptm", float(arr['ptm']))
                logger.log("chai_iptm", float(arr['iptm']))
                
            if df is not None:
                df[f"sample{i}"]["chai_ptm"] = float(arr['ptm'])
                df[f"sample{i}"]["chai_iptm"] = float(arr['iptm'])
            

        


    def run_pmpnn_designability(self, idx, rank, world_size, savedir, logger, df):

        os.makedirs(f"{savedir}/rank{rank}/pmpnn/pdbs", exist_ok=True)
        for i in idx:
            cmd = [
                'cp',
                f"{savedir}/sample{i}.pdb",
                f"{savedir}/rank{rank}/pmpnn/pdbs"
            ]
            subprocess.run(cmd)
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
        print(' '.join(cmd))  
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
        for i in idx:
            df[f"sample{i}"]["pmpnn_scrmsd"] = pmpnn_df.loc[f"sample{i}"].scRMSD
            df[f"sample{i}"]["pmpnn_scTM"] = pmpnn_df.loc[f"sample{i}"].scTM
            df[f"sample{i}"]["pmpnn_pLDDT"] = pmpnn_df.loc[f"sample{i}"].pLDDT
        
    def save_df(self, idx, rank, world_size, savedir, logger, df):
        df = pd.DataFrame(df).T.astype(float)
        df.to_csv(f"{savedir}/rank{rank}.csv")
    
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            dfs = []
            for r in range(world_size):
                dfs.append(pd.read_csv(f"{savedir}/rank{r}.csv", index_col=0))
                subprocess.run(["rm", f"{savedir}/rank{r}.csv"])
            df = pd.concat(dfs).sort_index()
            df.to_csv(f"{savedir}/info.csv")
        
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
                   
        if self.cfg.run_secondary:
            self.run_secondary(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_pmpnn_designability:
            torch.cuda.empty_cache()
            self.run_pmpnn_designability(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_chai:
            torch.cuda.empty_cache()
            self.run_chai(idx, rank, world_size, savedir, logger, df)
        
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
        if self.cfg.struct.edm:
            StructureStepper = EDMDiffusionStepper
        elif self.cfg.struct.gfm:
            StructureStepper = GaussianFMStepper
        sampler = OpenProtSampler(schedules={
            'structure': self.struct_sched_fn,
            'sequence': self.seq_sched_fn,
            #'struct_weight': lambda t: (1-t) * self.cfg.struct.sde_weight.start + t * self.cfg.struct.sde_weight.end,
            #'struct_temp': lambda t: (1-t) * self.cfg.struct.temp_factor.start + t * self.cfg.struct.temp_factor.end
        }, steppers=[
            StructureStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq)
        ])
        
        sample, extra = sampler.sample(
            model,
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
    
            with open(f"{savedir}/{name}_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))

            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            with open(f"{savedir}/{name}_traj.fasta", "w") as f:
                for seqs in extra['seq_traj']:
                    seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
                    seq = seq.replace('X', '-')
                    f.write(seq+'\n')
                
            if self.cfg.get('ligand', False):
                data.update_seqres()
                struct = Structure.from_chains([
                    Polypeptide(data['seqres'], name='A'),
                    Ligand([self.cfg.ligand], name='B'),
                ])
                prot = struct.get_chain(0)
                prot.atoms['coords'][prot.residues['atom_center']] = data['struct'].cpu()
                prot.atoms['is_present'][prot.residues['atom_center']] = True
    
                lig = struct.get_chain(1)
                lig.atoms['coords'] = data['ligand']['struct'].cpu()
                lig.atoms['is_present'] = True
    
                ref_str = struct.to_mmcif()
                with open(f"{savedir}/{name}.cif", "w") as f:
                    f.write(ref_str)
                
                clash = struct.clash_score(ca_only=True)
                if logger is not None:
                    logger.log(f"{self.cfg.name}/clash", clash)
                
                from rdkit import Chem                
                smi = Chem.MolToSmiles(struct.get_chain(1).get_residue(0).to_rdkit())
                with open(f"{savedir}/{name}_lig.fasta", "w") as f:
                    f.write(f">protein|name={name}\n")
                    f.write(f"{data['seqres']}\n")
                    f.write(f">ligand|name={self.cfg.ligand}\n")
                    f.write(f"{smi}\n")
                