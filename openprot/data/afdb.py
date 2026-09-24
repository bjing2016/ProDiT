import torch
import numpy as np
import pandas as pd
try:
    import foldcomp
except:
    print("Could not import foldcomp")
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset
from ..utils.prot_utils import seqres_to_aatype

class AFDBDataset(OpenProtDataset):
    
    def setup(self):
        self.num = 0
        self.denom = 0
        self.db = foldcomp.open(self.cfg.path)
        if self.cfg.blacklist is not None:
            blacklist = pd.read_csv(
                self.cfg.blacklist,
                names=["query", "target", "qcov", "tcov", "evalue"],
                sep="\t",
            )
            self.blacklist = set(blacklist["target"])

        # print(len(self.db))
        self.idx = np.arange(len(self.db))
        self.annotations = pd.read_pickle(self.cfg.annotations)
        if self.cfg.plddt_thresh is not None:
            self.idx = self.idx[self.annotations["plddt"] > self.cfg.plddt_thresh]

        if self.cfg.index is not None:
            names = [f"{s.strip()}.cif.gz" for s in open(self.cfg.index)]
            self.idx = (
                self.annotations.reset_index()
                .set_index("name")
                .loc[names]["index"]
                .to_numpy()
            )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        name, pdb = self.db[self.idx[idx]]
        name = name.split(".")[0]
        if self.cfg.blacklist is not None and name in self.blacklist:
            return self[(idx + 1) % len(self)]

        prot = protein.from_pdb_string(pdb)
        self.num += prot.b_factors[:,1].mean()
        self.denom += 1
        # print(self.num / self.denom)
        seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
        return self.make_data(
            name=name,
            seqres=seqres,
            residx=np.arange(len(seqres), dtype=np.float32),
            seq_mask=np.ones(len(seqres), dtype=np.float32),
            struct=prot.atom_positions[:,1].astype(np.float32),
            struct_mask=prot.atom_mask[:,1].astype(np.float32),
        )
