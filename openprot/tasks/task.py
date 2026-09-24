import numpy as np
from ..utils.misc_utils import temp_seed
from abc import abstractmethod
import hashlib

class OpenProtTask:
    def __init__(self, cfg, datasets):
        self.cfg = cfg
        self.datasets = datasets
        print('datasets', datasets)
        self.dataset_probs = np.array(
            [cfg.datasets[ds] for ds in cfg.datasets]
        )
        print('dataset_probs', self.dataset_probs)
        assert self.dataset_probs.sum() == 1
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.shuffle_datasets()

    def register_loss_masks(self):
        return []

    def shuffle_datasets(self):
        self.shuffled_idx = {}
        self.counter = {}

        for name in self.cfg.datasets:
            idx = np.arange(len(self.datasets[name]))
            hash_ = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10000
            hash_ += int(hashlib.sha256(self.cfg.name.encode()).hexdigest(), 16) % 10000
            with temp_seed(hash_+self.cfg.seed):
                np.random.shuffle(idx)
            self.shuffled_idx[name] = idx
            self.counter[name] = 0 # self.cfg.datasets[name].start

        self.curr_ds = self.rng.choice(self.cfg.datasets, p=self.dataset_probs)

    @abstractmethod
    def prep_data(self, data, crop=None):
        """
        (1) Crops data, or otherwise ensures max length = crop
        (2) Sets _noise features for tracks of interest
            - does not need to take mask into account, strictly speaking
            - (i.e., mask doesn't exist yet for structure track)
        (3) any other relevant preprocessing (must be features in config.yaml)
            - NUMPY ARRAYS ONLY
        """
        NotImplemented

    def advance(self):
        self.counter[self.curr_ds] += 1
        self.curr_ds = self.rng.choice(self.cfg.datasets, p=self.dataset_probs)

    def yield_data(self, crop=None):

        name = self.curr_ds
        ds = self.datasets[name]
        order = self.shuffled_idx[name]
        idx = self.counter[name]

        # print(f"i={i} rank={rank} ds={name} idx={idx} actual={order[idx]}")
        data = ds[order[idx % len(order)]]
        data = self.prep_data(data, crop=crop)
        if crop is not None:
            try:
                assert len(data["seqres"]) <= crop
            except:
                raise Exception(f"{self.__class__}.prep_data failed to crop data.")
        return data
