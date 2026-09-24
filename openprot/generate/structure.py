import torch
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class EDMDiffusionStepper:
    def __init__(self, cfg=None, mask=None):
        self.cfg = cfg
        self.mask = mask
        
    def set_step(self, batch, sched, extra={}):
        if self.mask is not None:
            mask = self.mask
        else:
            mask = batch['pad_mask'].bool()
        t, _ = sched['structure']
        batch["struct_noise"] = torch.where(
            mask, t, batch["struct_noise"]
        )
        
    def advance(self, batch, sched, out, extra={}):
        
        if self.mask is not None:
            mask = self.mask
        else:
            mask = batch['pad_mask'].bool()

        if 'traj' not in extra: extra['traj'] = []
        if 'preds' not in extra: extra['preds'] = []
            
        x = batch['struct']
        t2, t1 = sched['structure']
    
        dt = t2 - t1
        g = np.sqrt(2 * t2)

        x0 = out['trans']

        if self.cfg.align:
            x0 = masked_center(x0, batch['struct_mask'].bool())
        
        extra['preds'].append(x0)
        
        s = (x0 - x) / t2**2  # score
        noise = torch.randn_like(x)

        if 'struct_temp' in sched:
            gamma, _ = sched['struct_temp']
        else:
            gamma = self.cfg.temp_factor

        if 'struct_weight' in sched:
            w, _ = sched['struct_weight']
        else:
            w = self.cfg.sde_weight

        w = w / (1 + t2/self.cfg.edm.sigma_data) ** self.cfg.get('sde_exp', 0.5)
        
        ode = 0.5 * g**2 * s * dt 
        sde = 0.5 * g**2 * s * dt + g * gamma * np.sqrt(dt) * noise
        dx = ode + w * sde
        
        batch['struct'] = x + torch.where(mask[...,None], dx, 0.0)
        extra['traj'].append(batch['struct'])

class GaussianFMStepper:
    def __init__(self, cfg=None):
        self.cfg = cfg
        
    def set_step(self, batch, sched, extra={}):
        t, _ = sched['structure']
        batch["struct_noise"] = torch.ones_like(batch["struct_noise"]) * t
        
    def advance(self, batch, sched, out, extra={}):

        if 'traj' not in extra: extra['traj'] = []
        if 'preds' not in extra: extra['preds'] = []
        
        x = batch['struct']
        t2, t1 = sched['structure']

        dt = t2 - t1
        
        x0 = out['trans']
        extra['preds'].append(x0)

        v = (x0 - x) / t2
        s = ((1 - t2) * v - x) / t2 / self.cfg.gfm.sigma_max**2

        noise = torch.randn_like(x)
        w = self.cfg.sde_weight

        g = w * t2 * self.cfg.gfm.sigma_max**2  
        g *= sigmoid(t2 / 0.001) # temporary
        gamma = self.cfg.temp_factor
        dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise

        batch['struct'] = x + dx
        extra['traj'].append(batch['struct'])