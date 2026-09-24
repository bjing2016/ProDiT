from abc import abstractmethod
import torch
from ..model.model import OpenProtModel
from ..data.data import OpenProtData


class OpenProtTrack:
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup()

    def setup(self):
        pass

    @abstractmethod
    def tokenize(self, data: OpenProtData):
        NotImplemented

    @abstractmethod
    def add_modules(self, model: OpenProtModel):
        NotImplemented

    @abstractmethod
    def corrupt(
        self,
        batch: OpenProtData,
        noisy_batch: OpenProtData,
        target: OpenProtData,
        logger=None,
    ):
        """
        (1) log number of toks (_mask.sum())
        (2) place regression targets in target
        (3) place _supervise mask in target (used to reduce losses)
        (4) place noisy inputs in noisy_batch
        (5) place _noise in noisy_batch (so they can be embedded) (or equivalent)
        (6) if necessary, places any _mask in noisy_batch (so it can be embedded)
        """
        NotImplemented

    @abstractmethod
    def embed(self, model: OpenProtModel, batch: OpenProtData):
        """
        (1) Noise levels must be embedded
        (2) Nonexistent tokens must be marked with NONE
            - so the model knows it won't be supervised
        (3) Noise level 1 tokens should be marked with MASK
        """
        NotImplemented

    @abstractmethod
    def predict(self, model: OpenProtModel, out: torch.Tensor, readout: OpenProtData):
        NotImplemented

    @abstractmethod
    def compute_loss(self, readout: OpenProtData, target: OpenProtData, logger=None):
        """
        (1) Reduce loss using _supervise when logging
        (2) Reduce loss with pad_mask when returning!
            - I.e., (loss * supervise).sum() / pad_mask.sum()
        """
        NotImplemented
