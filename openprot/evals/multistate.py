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
from ..utils.prot_utils import make_ca_prot, write_ca_traj, compute_tmscore, aatype_to_seqres, seqres_to_aatype
from collections import defaultdict
from biopandas.pdb import PandasPdb
from Bio import Align#, pairwise2 
import subprocess
from ..utils.mycif import read_mmcif

def align(seq, motif_seq):
    aligner = Align.PairwiseAligner()
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -100000
    aligner.extend_gap_score = -100000
    aligner.target_end_gap_score = -100000
    aligner.query_end_gap_score = 0.0
    alignment = aligner.align(seq, motif_seq)[0]
    start = np.argwhere(alignment.indices[1] > -1).min()
    end = np.argwhere(alignment.indices[1] > -1).max() + 1
    return alignment, start, end
def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class MultistateEditEval(CodesignEval):

    def setup(self):
        super().setup()
        if self.cfg.prot:
            self.prot = dict(np.load(self.cfg.prot, allow_pickle=True))
            self.mask = np.load(self.cfg.mask)
            self.seqres = self.cfg.seqres 
            self.ref_motif = self.prot['all_atom_positions'][self.mask]
            
        with open('experiments/carbonic_anhydrase/efhand.pdb') as f:
            self.float_motif = protein.from_pdb_string(f.read())
            
        self.df = defaultdict(dict)             
        
    def __len__(self):
        return self.cfg.num_samples
        
    def __getitem__(self, idx):

        if self.cfg.prot:
            prot = self.prot
            motif_mask = self.mask
            seqres = self.seqres
            L = len(seqres)
            motif_ca=prot["all_atom_positions"][:,1]
            motif_ca_mask=prot["all_atom_mask"][:,1]
        elif self.cfg.motif:
            spec = load_motif_spec(self.cfg.motif)
            masks = sample_motif_mask(spec)
            motif_mask = masks['sequence']
            motif_idx = masks['group']
            with open(self.cfg.motif) as f:
                prot = protein.from_pdb_string(f.read())
            L = len(motif_mask)
            motif_ca = np.zeros((L, 3))
            motif_ca[motif_mask] = prot.atom_positions[:,1]
            motif_ca_mask=np.ones_like(motif_mask)
            motif_aatype = np.zeros(L, dtype=int)
            motif_aatype[motif_mask] = prot.aatype
            seqres = aatype_to_seqres(motif_aatype)
                    
        del prot
        with open('experiments/carbonic_anhydrase/efhand.pdb') as f:
            efhand = protein.from_pdb_string(f.read())
        M = len(efhand.atom_positions)
            
        data = self.make_data(
            name=f"sample{idx}",
            seqres=seqres,
            residx=np.arange(L),
            seq_mask=np.ones(L),
            seq_noise=(~motif_mask).astype(float) * self.cfg.seq.noise_level,
            struct=masked_center(motif_ca, motif_ca_mask),
            struct_noise=np.ones(L) * self.cfg.struct.noise_level,
            struct_mask=motif_ca_mask,
            motif_mask=motif_mask,
            motif=np.where(motif_mask[...,None], masked_center(motif_ca, motif_mask), 0.0),
            motif_idx=motif_mask, # motif is motif 1
        )
        seqres = np.array(list(data['seqres']))
        efhand_seqres = np.array(list(aatype_to_seqres(efhand.aatype)))
        # if self.cfg.motif: # this part is NEW!
        #     start = np.random.randint(0, len(seqres)-12)
        #     end = start + 12
        #     collision = np.any(
        #         data['motif_mask'][start:end].astype(bool) \
        #         #& (motif_seqres != seqres[start:end])
        #     )
        #     if collision:
        #         return self[idx]
        # else:
        if self.cfg.motif_pos:
            start = self.cfg.motif_pos
            end = start + 12
        else:
            collision = True
            while collision:
                start = np.random.randint(0, len(seqres)-12)
                end = start + 12
                collision = np.any(
                    data['motif_mask'][start:end].astype(bool) \
                    #& (motif_seqres != seqres[start:end])
                )
                
        
        seqres[start:end] = efhand_seqres
        data['seqres'] = ''.join(list(seqres))
        data['seq_noise'][start:end] = 0.0
        data['motif'][start:end] = masked_center(efhand.atom_positions[:,1])
        data['motif_mask'][start:end] = 1.0
        data['motif_idx'][start:end] = 2.0
        
        # data.pad(L+M)
        # data['seqres'] = seqres + aatype_to_seqres(motif.aatype)
        # data['seq_mask'][:] = 1
        # data['struct_noise'][:] = data['struct_noise'][0]
        # data['motif'][L:] = masked_center(motif.atom_positions[:,1])
        # data['motif_float_mask'][L:] = 1
        # data['L'] = len(seqres)
        # data['pad_mask'][:] = 1
        
        return data

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        
        if world_size > 1:
            torch.distributed.barrier()

        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

        df = self.df
        
        if self.cfg.run_designability:
            torch.cuda.empty_cache()
            self.run_designability(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_chai:
            torch.cuda.empty_cache()
            self.run_chai(idx, rank, world_size, savedir, logger, df)
                   
        self.save_df(idx, rank, world_size, savedir, logger, df)
        
    
    def run_chai(self, idx, rank, world_size, savedir, logger, df):
        
        
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cvd:
            dev = cvd.split(',')[torch.cuda.current_device()]
        else:
            dev = torch.cuda.current_device()
        
        for i in idx:
            subprocess.run(['rm', '-r', f"{savedir}/chai/sample{i}_zn"])
            os.makedirs(f"{savedir}/chai/sample{i}_zn", exist_ok=True)

            subprocess.run(['rm', '-r', f"{savedir}/chai/sample{i}_ca"])
            os.makedirs(f"{savedir}/chai/sample{i}_ca", exist_ok=True)

            seqres = open(f"{savedir}/sample{i}.fasta").read().strip().split('\n')[-1]
            
            with open(f"{savedir}/chai/sample{i}_zn.fasta", "w") as f:
                f.write(f">protein|name=sample{i}\n")
                f.write(f"{seqres}\n")
                if self.cfg.zn:
                    f.write(">ligand|name=ZN\n")
                    f.write("[Zn+2]\n")
            with open(f"{savedir}/chai/sample{i}_ca.fasta", "w") as f:
                f.write(f">protein|name=sample{i}\n")
                f.write(f"{seqres}\n")
                if self.cfg.zn:
                    f.write(">ligand|name=ZN\n")
                    f.write("[Zn+2]\n")
                f.write(">ligand|name=CA\n")
                f.write("[Ca+2]\n")

            cmd = [
                "bash",
                "scripts/switch_conda_env.sh",
                "chai",
                "chai-lab",
                "fold",
                f"{savedir}/chai/sample{i}_zn.fasta",
                f"{savedir}/chai/sample{i}_zn"
            ]
            print(' '.join(cmd), flush=True)  
            subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  
            cmd = [
                "bash",
                "scripts/switch_conda_env.sh",
                "chai",
                "chai-lab",
                "fold",
                f"{savedir}/chai/sample{i}_ca.fasta",
                f"{savedir}/chai/sample{i}_ca"
            ]
            print(' '.join(cmd), flush=True)  
            subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  
            motif_seq = aatype_to_seqres(self.float_motif.aatype)
            aln, start, end = align(seqres, motif_seq)

            def get_ca_coords(path):
                atoms = read_mmcif(path)['_atom_site']
                atoms = atoms.loc[
                    atoms.label_atom_id == 'CA',
                    ['Cartn_x', 'Cartn_y', 'Cartn_z']
                ]
                return atoms.to_numpy().astype(np.float32)

            struct1 = get_ca_coords(
                f"{savedir}/chai/sample{i}_zn/pred.model_idx_0.cif"
            )
            struct2 = get_ca_coords(
                f"{savedir}/chai/sample{i}_ca/pred.model_idx_0.cif"
            )
            if self.cfg.motif:
                with open(f"{savedir}/sample{i}_motif.pdb") as f:
                    motif = protein.from_pdb_string(f.read())
                motif_residx = motif.residue_index                    
                struct1_motif1 = compute_rmsd(
                    torch.from_numpy(struct1[motif_residx-1]).float(), 
                    torch.from_numpy(motif.atom_positions[:,1]).float(),
                )
                struct2_motif1 = compute_rmsd(
                    torch.from_numpy(struct2[motif_residx-1]).float(), 
                    torch.from_numpy(motif.atom_positions[:,1]).float(),
                )

            else:    
                struct1_motif1 = compute_rmsd(
                    torch.from_numpy(struct1[self.mask]), 
                    torch.from_numpy(self.ref_motif[:,1])
                )
                struct2_motif1 = compute_rmsd(
                    torch.from_numpy(struct2[self.mask]), 
                    torch.from_numpy(self.ref_motif[:,1])
                )

            
            struct1_motif2 = compute_rmsd(
                torch.from_numpy(struct1[start:end]), 
                torch.from_numpy(self.float_motif.atom_positions[:,1]).float()
            )
            struct2_motif2 = compute_rmsd(
                torch.from_numpy(struct2[start:end]), 
                torch.from_numpy(self.float_motif.atom_positions[:,1]).float()
            )
            if df is not None:
                df[f"sample{i}"]['struct1_motif1'] = float(struct1_motif1)
                df[f"sample{i}"]['struct1_motif2'] = float(struct1_motif2)
                df[f"sample{i}"]['struct2_motif1'] = float(struct2_motif1)
                df[f"sample{i}"]['struct2_motif2'] = float(struct2_motif2)

            

    def run_designability(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            #name = self[i]['name']
            name = f'sample{i}'
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
            # name = self[i]['name']
            name = f'sample{i}'
            # with open(f"{savedir}/{name}.pdb") as f:
            #     prot = protein.from_pdb_string(f.read()) # len 257
            with open(f"{savedir}/rank{rank}/{name}.pdb") as f:
                pred = protein.from_pdb_string(f.read()) # len 260

            if self.cfg.motif:
                with open(f"{savedir}/{name}_motif.pdb") as f:
                    motif = protein.from_pdb_string(f.read())
                motif_residx = motif.residue_index
                motif_rmsd = compute_rmsd(
                    torch.from_numpy(pred.atom_positions[motif_residx-1,1]),
                    #torch.from_numpy(prot.atom_positions[motif_residx-1,1]),
                    torch.from_numpy(motif.atom_positions[:,1]),
                )
            else:
                motif_rmsd = compute_rmsd(
                    torch.from_numpy(pred.atom_positions[self.mask,1]), 
                    torch.from_numpy(self.ref_motif[:,1])
                )
            motif_seq = aatype_to_seqres(self.float_motif.aatype)
            seq = aatype_to_seqres(pred.aatype)
            aln, start, end = align(seq, motif_seq)
            float_motif_rmsd = compute_rmsd(
                torch.from_numpy(pred.atom_positions[start:end,1]), 
                torch.from_numpy(self.float_motif.atom_positions[:,1])
            )
            
                            
            # lddt = compute_lddt(
            #     torch.from_numpy(pred.atom_positions[:,1]), 
            #     torch.from_numpy(prot.atom_positions[:,1]), 
            #     torch.from_numpy(prot.atom_mask[:,1])
            # )
            # rmsd = compute_rmsd(
            #     torch.from_numpy(pred.atom_positions[:,1]),  
            #     torch.from_numpy(prot.atom_positions[:,1])
            # )
            # tmscore = compute_tmscore(  # second is reference
            #     coords1=pred.atom_positions[:,1],
            #     coords2=prot.atom_positions[:,1],
            # )['tm']

            plddt = PandasPdb().read_pdb(
                f"{savedir}/rank{rank}/{name}.pdb"
            ).df['ATOM']['b_factor'].mean()
            
            if logger is not None:
                # logger.log(f"{self.cfg.name}/sclddt", lddt)
                # logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                # logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                # logger.log(f"{self.cfg.name}/scTM", tmscore)
                # logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
                # logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
                logger.log(f"{self.cfg.name}/plddt", plddt)
                logger.log(f"{self.cfg.name}/mRMSD", motif_rmsd)
                logger.log(f"{self.cfg.name}/float_mRMSD", float_motif_rmsd)
                # logger.log(
                #     f"{self.cfg.name}/success", 
                #     (rmsd < 2).float() * (motif_rmsd < 1).float()
                # )

            df[f"sample{i}"]["plddt"] = plddt
            # df[f"sample{i}"]["scrmsd"] = float(rmsd)
            # df[f"sample{i}"]["sctm"] = tmscore
            # df[f"sample{i}"]["sclddt"] = float(lddt)
            df[f"sample{i}"]["mRMSD"] = float(motif_rmsd)
            df[f"sample{i}"]["float_mRMSD"] = float(float_motif_rmsd)
            df[f"sample{i}"]["aln_score"] = aln.score / len(motif_seq)
                
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
            p = self.cfg.struct.edm_p # self.cfg.struct.edm.sched_p
            sigma_max = self.cfg.struct.noise_level
            sigma_min = self.cfg.struct.edm.sigma_min 
            return (
                sigma_min ** (1 / p)
                + (1-t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
            ) ** p        
        
        """
        def model_func(noisy_batch):
            dup_batch = {**noisy_batch}
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    dup_batch[key] = torch.stack([
                        dup_batch[key].clone(), 
                        dup_batch[key].clone(),
                        dup_batch[key].clone()
                    ])
            ##### index 0 is NO MOTIFS ######
            dup_batch['pad_mask'][0] = torch.where(
                dup_batch['motif_float_mask'][0].bool(),
                0.0,
                dup_batch['pad_mask'][0]
            ) # no float motif
            dup_batch['motif_mask'][0][:] = 0 # no regular motif

            ##### index 1 is NO FLOAT MOTIF
            dup_batch['pad_mask'][1] = torch.where(
                dup_batch['motif_float_mask'][1].bool(),
                0.0,
                dup_batch['pad_mask'][1]
            ) # no float motif

            ##### index 2  is NO REGULAR MOTIF
            dup_batch['motif_mask'][2][:] = 0 # no regular motif
            
            
            for key in dup_batch:
                if type(dup_batch[key]) is torch.Tensor:
                    shape = dup_batch[key].shape
                    new_shape = (shape[0] * shape[1], *shape[2:])
                    dup_batch[key] = dup_batch[key].reshape(new_shape)
            
            out, readout = model.forward(dup_batch)        
            
            for key in readout:
                if type(readout[key]) is torch.Tensor:
                    shape = readout[key].shape
                    new_shape = (3, shape[0] // 3, *shape[1:])
                    readout[key] = readout[key].reshape(new_shape)
                
            
            t = noisy_batch['t']
            w1 = 1 # + 0.5 * t
            w2 = 1 + t**0.5
            w3 = 1
            w4 = 1
            new_readout = {}
            new_readout['trans'] = (
                readout['trans'][0]
                + w1 * (readout['trans'][1] - readout['trans'][0]) 
                + w2 * (readout['trans'][2] - readout['trans'][0])
            )
            new_readout['aatype'] = (
                readout['aatype'][0]
                + w3 * (readout['aatype'][1] - readout['aatype'][0]) 
                + w4 * (readout['aatype'][2] - readout['aatype'][0])
            )
            
            new_readout['aatype'][...,-1] = -np.inf # otherwise nan
            
            readout['aatype'] += noisy_batch['seqres_oh'].float() * self.cfg.seq.bias
            return out, new_readout
        """
        def model_func(noisy_batch):
            #mul = noisy_batch['motif_idx'].max().int().item()
            dup_batch = {**noisy_batch}
            for key in dup_batch:
                if key == 'struct':
                    dup_batch[key] = torch.stack([
                        torch.randn_like(
                            noisy_batch['struct'][0]
                        ) * self.cfg.struct.edm.sigma_max,
                        noisy_batch['struct'][0],
                        noisy_batch['struct'][0],
                        noisy_batch['struct'][0],
                        noisy_batch['struct'][1],
                        noisy_batch['struct'][1],
                        noisy_batch['struct'][1],
                    ])
                    
                elif type(dup_batch[key]) is torch.Tensor:
                    dup_batch[key] = torch.stack([
                        dup_batch[key].clone() for _ in range(7)
                    ])
            dup_batch['struct_noise'][0] = 160.

            #### i=0: no structure, no_motif
            dup_batch['motif_mask'][0] = 0
            dup_batch['motif'][0] = 0

            #### i=1: struct 0, no_motif
            dup_batch['motif_mask'][1] = 0
            dup_batch['motif'][1] = 0

            #### i=2: struct 1, motif 1
            dup_batch['motif_mask'][2] = noisy_batch['motif_mask'] * (noisy_batch['motif_idx'] == 1)
            dup_batch['motif'][2] *= (noisy_batch['motif_idx'] == 1)[...,None]

            #### i=3: struct 1, motif 2
            dup_batch['motif_mask'][3] = noisy_batch['motif_mask'] * (noisy_batch['motif_idx'] == 2)
            dup_batch['motif'][3] *= (noisy_batch['motif_idx'] == 2)[...,None]

            
            #### i=4: struct 2, no motif
            dup_batch['motif_mask'][4] = 0
            dup_batch['motif'][4] = 0

            #### i=5: struct 2, motif 1
            dup_batch['motif_mask'][5] = noisy_batch['motif_mask'] * (noisy_batch['motif_idx'] == 1)
            dup_batch['motif'][5] *= (noisy_batch['motif_idx'] == 1)[...,None]

            #### i=6: struct 2, motif 2
            dup_batch['motif_mask'][6] = noisy_batch['motif_mask'] * (noisy_batch['motif_idx'] == 2)
            dup_batch['motif'][6] *= (noisy_batch['motif_idx'] == 2)[...,None]

            
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
                    new_shape = (7, shape[0] // 7, *shape[1:])
                    readout[key] = readout[key].reshape(new_shape)

            #shape = readout['trans'].shape
            r = 0 # self.cfg.repulsion
            new_readout = {
                'trans': torch.stack([
                    (
                        readout['trans'][1] \
                        + (readout['trans'][2] - readout['trans'][1])
                        - r*(readout['trans'][3] - readout['trans'][1])
                    ),
                    (
                        readout['trans'][4] \
                        - r*(readout['trans'][5] - readout['trans'][4])
                        + (readout['trans'][6] - readout['trans'][4])
                    )
                ]),
                'aatype': (
                    - readout['aatype'][0]\
                    + (
                        readout['aatype'][1] \
                        + (readout['aatype'][2] - readout['aatype'][1])\
                        - r*(readout['aatype'][3] - readout['aatype'][1])\
                    )
                    + (
                        readout['aatype'][4] \
                        - r*(readout['aatype'][5] - readout['aatype'][4])\
                        + (readout['aatype'][6] - readout['aatype'][4])\
                    )
                ),
            }
            
            # breakpoint()
            # new_readout = {}
            # for key in readout:
            #     new_readout[key] = readout[key][0].clone()
            #     for i in range(mul):
            #         new_readout[key] += readout[key][i+1] - readout[key][0]
            new_readout['aatype'][...,-1] = -np.inf # otherwise nana
            
            return out, new_readout

        schedules = {
            'structure': edm_sched_fn,
            'sequence': lambda t: self.cfg.seq.noise_level * (1-t),
        }

        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct, mask=batch['struct_mask'].bool()),
            SequenceUnmaskingStepper(self.cfg.seq, mask=batch['seq_noise'].bool())
        ])
        
        noisy_batch['struct'] = torch.stack([
            noisy_batch['struct'],
            noisy_batch['struct']
        ]) # two struct trajs
        sample, extra = sampler.sample(
            model_func,
            noisy_batch,
            self.cfg.steps
        )

        if self.cfg.redesign:
            # resample the aatypes
            sample['aatype'].masked_fill_(~sample['motif_mask'].bool(), 20)
            sampler = OpenProtSampler(schedules, steppers=[
                SequenceUnmaskingStepper(self.cfg.seq1, mask=batch['seq_noise'].bool())
            ])
    
            sample, extra = sampler.sample(
                model_func,
                sample,
                #len (self.cfg.seqres),
                300,
            )

        
        batch['struct'] = sample['struct']
        batch['aatype'] = sample['aatype']

        batch['struct_0'] = batch['struct'][0]
        batch['struct_1'] = batch['struct'][1]
        del batch['struct']
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):

            data.update_seqres()
            name = data["name"]

            
            mask = data['motif_mask'].bool()
            ref_motif = data['motif'][mask]
            samp_motif_0 = data['struct_0'][mask]
            samp_motif_1 = data['struct_1'][mask]
            motif_idx = data['motif_idx'][mask]

            rmsd_01 = compute_rmsd(
                ref_motif[motif_idx == 1],
                samp_motif_0[motif_idx == 1]
            )
            rmsd_02 = compute_rmsd(
                ref_motif[motif_idx == 2],
                samp_motif_0[motif_idx == 2]
            )
            rmsd_11 = compute_rmsd(
                ref_motif[motif_idx == 1],
                samp_motif_1[motif_idx == 1]
            )
            rmsd_12 = compute_rmsd(
                ref_motif[motif_idx == 2],
                samp_motif_1[motif_idx == 2]
            )
            if logger is not None:
                logger.log(f"{self.cfg.name}/samp_mRMSD_01", rmsd_01)
                logger.log(f"{self.cfg.name}/samp_mRMSD_02", rmsd_02)
                logger.log(f"{self.cfg.name}/samp_mRMSD_11", rmsd_11)
                logger.log(f"{self.cfg.name}/samp_mRMSD_12", rmsd_12)

            self.df[name]['samp_mRMSD_01'] = rmsd_01
            self.df[name]['samp_mRMSD_02'] = rmsd_02
            self.df[name]['samp_mRMSD_11'] = rmsd_11
            self.df[name]['samp_mRMSD_12'] = rmsd_12

            prot = make_ca_prot(
                data['struct_0'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["struct_mask"].cpu().numpy(),
            )
            with open(f"{savedir}/{name}_0.pdb", "w") as f:
                f.write(protein.to_pdb(prot))

            prot = make_ca_prot(
                data['struct_1'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["struct_mask"].cpu().numpy(),
            )
            with open(f"{savedir}/{name}_1.pdb", "w") as f:
                f.write(protein.to_pdb(prot))
    
    
            # with open(f"{savedir}/{name}_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            # with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))

            # L = data['L']
            seq = aatype_to_seqres(data["aatype"])
            # seq, motif_seq = seq[:L], seq[L:]
            
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            if self.cfg.motif:
                save_motif_pdb(
                    self.cfg.motif, 
                    data["motif_idx"].cpu().numpy() == 1, # pick out original motif
                    f"{savedir}/{name}_motif.pdb"
                )

            # with open(f"{savedir}/{name}_traj.fasta", "w") as f:
            #     for seqs in extra['seq_traj']:
            #         seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])[:L]
            #         seq = seq.replace('X', '-')
            #         f.write(seq+'\n')

            # seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            # seq, motif_seq = seq[:L], seq[L:]            

            # alignment, start, end = align(seq, motif_seq)
            # if logger is not None:
            #     logger.log(f"{self.cfg.name}/aln_score", alignment.score / len(motif_seq))
            #     logger.log(f"{self.cfg.name}/aln_score=1", float(alignment.score == len(motif_seq)))
            # with open(f"{savedir}/{name}.aln", "w") as f:
            #     f.write(alignment.format())


            # ref_motif = data['motif'][data['motif_idx'] == 1]
            # samp_motif = data['struct'][data['motif_idx'] == 1]
            # rmsd = compute_rmsd(ref_motif, samp_motif)

            # if logger is not None:
            #     logger.log(f"{self.cfg.name}/samp_mRMSD_1", rmsd)

            # ref_motif = data['motif'][data['motif_idx'] == 2]
            # samp_motif = data['struct'][data['motif_idx'] == 2]
            # rmsd = compute_rmsd(ref_motif, samp_motif)

            # if logger is not None:
            #     logger.log(f"{self.cfg.name}/samp_mRMSD_2", rmsd)
                
            # motif_mask = np.zeros(L)
            
            # motif_mask[start:end] = 1
            # save_motif_pdb(
            #     data['path'], 
            #     motif_mask,
            #     f"{savedir}/{name}_motif.pdb"
            # )
            # if logger is not None:
            #     logger.log(f"{self.cfg.name}/float_samp_mRMSD", rmsd)            
            #     logger.log(
            #         f"{self.cfg.name}/seqID", 
            #         np.mean([a == b for a, b in zip(seq, self.seqres)])
            #     )
        
