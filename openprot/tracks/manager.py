from ..model.model import OpenProtModel
import torch
from ..utils.misc_utils import autoimport


class OpenProtTrackManager(dict):
    def __init__(self, cfg):
        self.cfg = cfg
        for name in cfg:
            track = autoimport(f"openprot.tracks.{name}")(cfg[name])
            self[name] = track

    def add_modules(self, model: OpenProtModel):
        for track in self.values():
            track.add_modules(model)
        return model

    def tokenize(self, data: dict):
        for track in self.values():
            track.tokenize(data)
        return data

    def embed(self, model: OpenProtModel, batch: dict):
        inp = batch.copy()
        inp["x"] = 0
        inp["x_cond"] = 0
        inp["y"] = 0
        for track in self.values():
            track.embed(model, batch, inp)
        return inp

    def readout(self, model: OpenProtModel, inp: dict, out: torch.Tensor):
        readout = {}
        for track in self.values():
            track.predict(model, inp, out, readout)
        return readout

    def corrupt(self, batch: dict, logger=None):
        noisy_batch = batch.copy()
        target = batch.copy()

        for track in self.values():
            track.corrupt(batch, noisy_batch, target, logger=logger)

        return noisy_batch, target

    def compute_loss(self, readout: dict, target: dict, logger=None, **kwargs):
        loss = 0
        for track in self.values():
            loss_ = track.compute_loss(readout, target, logger=logger, **kwargs)
            loss = loss + track.cfg.loss_weight * loss_
        return loss
