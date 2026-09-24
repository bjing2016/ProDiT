import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype

class PDBDataset(OpenProtDataset):

    def setup(self):
        self.df = pd.read_csv(f"{self.cfg.path}/pdb_chains.csv", index_col="name")

        if self.cfg.cutoff is not None:
            self.df = self.df[self.df.release_date < self.cfg.cutoff]

        if self.cfg.clusters is not None:
            self.clusters = []
            with open(self.cfg.clusters) as f:
                for line in f:
                    clus = []
                    names = line.split()
                    for name in names:
                        if name in self.df.index:
                            clus.append(name)
                    if len(clus) > 0:
                        self.clusters.append(clus)

        # PDB blacklist has a peculiar format due to how the foldseek PDB database is constrcted.
        if self.cfg.blacklist is not None:
            blacklist = pd.read_csv(
                self.cfg.blacklist,
                names=["query", "target", "qcov", "tcov", "evalue"],
                sep="\t",
            )
            blacklist = list(set(blacklist["target"]))
            prefix = [name.split("-")[0] for name in blacklist]
            suffix = [name.split("_")[-1] for name in blacklist]
            blacklist = list(zip(prefix, suffix))
            self.blacklist = set(["_".join(tup) for tup in blacklist])

    def __len__(self):
        if self.cfg.clusters:
            return len(self.clusters)
        else:
            return len(self.df)

    def __getitem__(self, idx: int):
        if self.cfg.clusters:
            clus = self.clusters[idx]
            name = np.random.choice(clus)
        else:
            name = self.df.index[idx]

        if self.cfg.blacklist is not None:
            if name in self.blacklist:
                return self[(idx + 1) % len(self)]

        prot = dict(
            np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        )
        seqres = self.df.seqres[name]
        residx = np.arange(len(seqres), dtype=np.float32)
        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        atom37=prot["all_atom_positions"]
        atom37_mask=prot["all_atom_mask"]
        
            
        return self.make_data(
            name=name,
            seqres=seqres,
            residx=residx,
            seq_mask=seq_mask,
            struct=atom37[:,1],
            struct_mask=atom37_mask[:,1]
        )
