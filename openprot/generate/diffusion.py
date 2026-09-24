import torch
import numpy as np
from abc import abstractmethod
from ..utils.geometry import rmsdalign


class Diffusion:

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def add_noise(self, pos, t, mask=None):
        # return noisy, target
        NotImplemented

    def precondition(self, inp, t):
        return inp

    def postcondition(self, inp, out, t):
        return out
        
    @abstractmethod
    def compute_loss(self, pred, target, t, mask):
        NotImplemented

def t_to_sigma(cfg, t):
    p = cfg.sched_p
    sigma = (
        cfg.sigma_min ** (1 / p)
        + t * (cfg.sigma_max ** (1 / p) - cfg.sigma_min ** (1 / p))
    ) ** p
    return sigma
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class GaussianFM(Diffusion):
    def add_noise(self, pos, t, mask=None):
        
        noise = torch.randn_like(pos) * self.cfg.sigma_max
        
        if self.cfg.center:
            pos = masked_center(pos, mask)

        t = t[..., None]
        noisy = t * noise + (1 - t) * pos
        
        target = pos # - noise
        
        return noisy, target

    def postcondition(self, inp, out, t):
        return inp + t[...,None] * out

    def compute_loss(self, pred, target, t, mask, aligned=False, eps=1e-9):
        return (
            torch.square(pred - target).sum(-1)
            / self.cfg.sigma_max ** 2 / 2 
            / (eps + t**2)
        )

    

    def inference(
        self,
        model,
        cfg=None,
        seed=None,
        mask=None,
        shape=None,
        device=None,
        return_traj=False,
    ):

        if seed is not None:
            x = seed
        else:
            if mask is not None:
                shape = list(mask.shape) + [3]
                device = mask.device
            x = torch.randn(shape, device=device) * self.cfg.prior_sigma

        out = [x]
        if cfg.sched_type == "linear":
            sched = np.linspace(1, 0, cfg.nsteps + 1)
        elif cfg.sched_type == "log":
            sched = np.logspace(0, -2, cfg.nsteps + 1)
            sched = (sched - sched.min()) / (sched.max() - sched.min())

        preds = []
        for t2, t1 in zip(sched[:-1], sched[1:]):
            dt = t2 - t1

            if self.cfg.prediction == "velocity":
                v = model(x, t2)
            elif self.cfg.prediction == "target":
                x0 = model(x, t2)

                if self.cfg.inf_align:
                    x0 = rmsdalign(x, x0, mask)

                v = (x0 - x) / t2

            preds.append(x + v * t2)

            # score = ((1-t)x0 - x_t) / t^2 * sigma^2

            # x0 = xt + t*v
            # score = (xt + t*v - t*xt - t^2*v - xt) / t^2 * sigma^2
            #       = v - xt - t*v / t * sigma^2
            #       = = (1-t)*v - xt / t * sigma^2

            s = ((1 - t2) * v - x) / t2 / self.cfg.prior_sigma**2

            noise = torch.randn_like(x)

            g = cfg.sde_weight * t2 * self.cfg.prior_sigma**2

            g *= sigmoid((t2 - cfg.sde_cutoff_time) / cfg.sde_cutoff_width)

            gamma = cfg.temp_factor
            dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise

            x = x + dx
            out.append(x)

        if return_traj:
            return torch.stack(out), torch.stack(preds)
        else:
            return x

class EDMDiffusion(Diffusion):

    def get_sigma(self, t, eps=1e-4):
        return t

    def add_noise(self, pos, t, mask=None):

        sigma = self.get_sigma(t)[..., None]
        noise = torch.randn_like(pos) * sigma
        if self.cfg.center:
            pos = masked_center(pos, mask)

        noisy = pos + noise
        target = pos

        return noisy, target

    def precondition(self, inp, t):
        sigma = self.get_sigma(t)[..., None]
        return inp / (self.cfg.sigma_data**2 + sigma**2) ** 0.5

    def postcondition(self, inp, out, t):
        
        sigma = self.get_sigma(t)[..., None]
        cskip = self.cfg.sigma_data**2 / (self.cfg.sigma_data**2 + sigma**2)
        cout = sigma * self.cfg.sigma_data / (self.cfg.sigma_data**2 + sigma**2) ** 0.5

        return cskip * inp + cout * out

    def compute_loss(self, pred, target, t, mask, aligned=False, eps=1e-9):

        if aligned:
            target = rmsdalign(pred.detach(), target, mask)
            target = torch.where(mask[..., None].bool(), target, 0.0)
        
        
        sigma = self.get_sigma(t)
        num = sigma**2 + self.cfg.sigma_data**2
        denom = (sigma * self.cfg.sigma_data) ** 2
        weight = num / (denom + eps)
        return weight * torch.square(pred - target).sum(-1)
