import torch
import numpy as np
from abc import abstractmethod
RNA_LETTERS = {"A": 0, "G": 1, "C": 2, "U": 3}
DNA_LETTERS = {"A": 0, "G": 1, "C": 2, "T": 3}
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype

class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feats=None, tracks=None):
        super().__init__()
        self.cfg = cfg
        self.feats = feats
        self.tracks = tracks
        self.setup()

    @abstractmethod
    def setup(self):
        NotImplemented

    @abstractmethod
    def __len__(self):
        NotImplemented

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Returns OpenProtData via self.make_data(...) with args
        (1) name and seqres (required)
        (2) any features that should be set to non-default values (np arrays only!)
        """
        NotImplemented

    def make_data(self, make_ligand=True, **kwargs):
        assert "name" in kwargs
        assert "seqres" in kwargs
        for key in kwargs:
            assert key in self.feats or key in ['name', 'seqres'], key

        data = OpenProtData()

        data["name"] = kwargs["name"]
        data["seqres"] = kwargs["seqres"]
        data["dataset"] = self.cfg.name 
        
        L = len(data["seqres"])
        for feat, shape in self.feats.items():
            shape = [(n if n > 0 else L) for n in shape]
            if feat in kwargs and kwargs[feat] is not None:
                assert type(kwargs[feat]) is np.ndarray, (feat, kwargs[feat])
                data[feat] = kwargs[feat].astype(np.float32)
                
            else:
                data[feat] = np.zeros(shape, dtype=np.float32)
        if make_ligand:
            data['ligand'] = self.make_data(
                name=kwargs["name"]+'_lig', 
                seqres="", 
                make_ligand=False
            )
        
        return data

# these keys risk leaking information about the protein
# so we exclude them by default when copying a data object
excluded_keys = [
    "seqres",
    "aatype",
    "atom_num",
    "struct",
    "cath",
    "nma",
]
class OpenProtBatch(dict):
    
    def copy(self, all_keys=False):
        data = OpenProtBatch()
        for key in self:
            if all_keys or (key not in excluded_keys):
                data[key] = self[key]
        if 'ligand' in self:
            data['ligand'] = self['ligand'].copy(all_keys=True)
        return data

    def to(self, device):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        if 'ligand' in self:
            self['ligand'].to(device)
        return self

    def unbatch(self, trim=True):
        datas = [OpenProtData() for _ in self['seqres']]
        for key in self:
            for i in range(len(self['seqres'])):
                if key == 'ligand': continue
                datas[i][key] = self[key][i]

        if trim:
            datas = [data.trim() for data in datas]
        if 'ligand' in self:
            ligands = self['ligand'].unbatch(trim=trim)
            for data, ligand in zip(datas, ligands):
                data['ligand'] = ligand
        return datas

class OpenProtData(dict):

    def copy(self, all_keys=False):
        data = OpenProtData()
        for key in self:
            if all_keys or (key not in excluded_keys):
                data[key] = self[key]
        if 'ligand' in self:
            data['ligand'] = self['ligand'].copy(all_keys=True)
        return data

    def to(self, device):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        if 'ligand' in self:
            self['ligand'].to(device)
        return self
        
    def update_seqres(self):
        
        seqres = []
        for aa, mt in zip(
            self['aatype'].cpu().long(), 
            self['mol_type'].cpu().long(), 
        ):
            if mt == 0:
                seqres += rc.restypes_with_x[aa]
            elif mt == 1:
                seqres += list(DNA_LETTERS.keys())[aa - 21]
            elif mt == 2:
                seqres += list(RNA_LETTERS.keys())[aa - 26]
            else:
                seqres += "*"
        self['seqres'] = ''.join(seqres)
        
    def get_contiguous_crop(self, crop_len):
        L = len(self["seqres"])
        start = np.random.randint(0, L - crop_len + 1)
        end = start + crop_len
        return list(range(start, end))
        
    def crop(self, crop_len=np.inf, idx=None, inplace=True):
        data = self if inplace else OpenProtData()
        # if 'ligand' in self:
        #     data['ligand'] = self['ligand'].crop(crop_len, idx, inplace)
        L = len(self["seqres"])
        if idx is not None or L >= crop_len:  # needs crop
            if idx is None:
                idx = self.get_contiguous_crop(crop_len)
            for key in self.keys():
                # special attribute
                if key == "seqres":
                    data[key] = "".join([self[key][i] for i in idx])
                elif key == 'ligand': 
                    pass
                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    data[key] = self[key]

                # global attribute
                elif key[0] == "/" or key[0] == '_':
                    data[key] = self[key]

                # regular attribute
                else:
                    data[key] = self[key][idx]
        
        return data

    def pad(self, pad_len: int):
        L = len(self["seqres"])
        if pad_len and L < pad_len:  # needs pad
            pad = pad_len - L
            for key in self.keys():
                # special attribute
                if key == "seqres":
                    self[key] = self[key] + " " * pad
                elif key == 'ligand':
                    pass
                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    pass

                # global attribute
                elif key[0] == "/" or key[0] == '_':
                    pass

                # regular attribute
                else:
                    shape = self[key].shape
                    dtype = self[key].dtype
                    padded = np.zeros((pad_len, *shape[1:]), dtype=dtype)
                    padded[:L] = self[key]
                    self[key] = padded

        pad_len = pad_len or L
        pad_mask = np.zeros(pad_len, dtype=np.float32)
        pad_mask[: min(pad_len, L)] = 1.0
        self["pad_mask"] = pad_mask

        # pad is the only one where the ligand is not also padded
        return self

    def batch(datas, pad=True):
        if pad:
            lens = [len(data['seqres']) for data in datas]
            for data in datas:
                data.pad(max(lens))
                
        batch = OpenProtBatch()
        key_union = list(set(sum([list(data.keys()) for data in datas], [])))
        for key in key_union:
            try:
                batch[key] = [data[key] for data in datas]
            except:
                raise Exception(f"Key {key} not present in all batch elements.")
        for key in key_union:
            try:
                if type(batch[key][0]) is np.ndarray and key not in ["_go_noise", "_site_noise"]:
                    batch[key] = torch.from_numpy(np.stack(batch[key]))
                elif key in ["_go_noise", "_site_noise"]: # handle func label noise lengths
                    max_len = max(len(arr) for arr in batch[key]) 
                    padded_batch = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=-1) for arr in batch[key]] # pad w/ -1
                    batch[key] = torch.from_numpy(np.stack(padded_batch))
                elif type(batch[key][0]) is torch.Tensor:
                    batch[key] = torch.stack(batch[key])
            except Exception as e:
                raise Exception(f"Key {key} exception: {e}")
        if 'ligand' in datas[0]:
            batch['ligand'] = OpenProtData.batch([data['ligand'] for data in datas])
        
        return batch

    def trim(self):
        mask = self['pad_mask'].bool()
        for key in self:
            if key == "seqres":
                self[key] = ''.join([self[key][i] for i, m in enumerate(mask) if m])

            elif key == 'ligand':
                pass
            # non-array attribute
            elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                pass

            # global attribute
            elif key[0] == "/" or key[0] == '_':
                pass

            else:
                self[key] = self[key][mask]
        if 'ligand' in self:
            self['ligand'].trim()
        return self

    # def concat(datas):
    #     batch = OpenProtData()
    #     key_union = list(set(sum([list(data.keys()) for data in datas], [])))
    #     for key in key_union:
    #         try:
    #             batch[key] = [data[key] for data in datas]
    #         except:
    #             raise Exception(f"Key {key} not present in all batch elements.")
    #     for key in key_union:
    #         try:
    #             if type(batch[key][0]) is np.ndarray:
    #                 if len(batch[key][0].shape) == 0: batch[key] = batch[key][0]
    #                 else: batch[key] = np.concatenate(batch[key], 0)
    #             elif type(batch[key][0]) is torch.Tensor:
    #                 if len(batch[key][0].shape) == 0: batch[key] = batch[key][0]
    #                 else: batch[key] = torch.cat(batch[key], 0)
    #             elif type(batch[key][0]) is str:
    #                 batch[key] = "".join(batch[key])
    #         except Exception as e:
    #             raise Exception(f"Key {key} exception: {e}")
    #     return batch


