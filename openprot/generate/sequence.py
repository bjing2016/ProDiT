from ..tracks.sequence import MASK_IDX, NUM_TOKENS
from torch.distributions.categorical import Categorical
import torch
import numpy as np
from ..utils import residue_constants as rc

def topk_masking(scores, k, mask=None, temp=1.0):
    
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    scores = scores + temp * gumbel
    if mask is not None:
        scores = torch.where(mask, scores, -np.inf) 
    ####
    # new = torch.zeros_like(scores).bool()
    # idx = torch.topk(scores, k, dim=-1).indices
    # new.scatter_(-1, idx, True)
    # print(new[0])
    ####
    new = torch.zeros_like(scores).bool()
    idx = scores.argsort(descending=True)
    if type(k) is int:
        k = torch.ones_like(idx[...,0]) * k
    fill = torch.arange(idx.shape[-1], device=idx.device) < k[...,None]
    new.scatter_(-1, idx, fill.bool())
    # print(new[0])
    return new

class SequenceUnmaskingStepper:
    def __init__(self, cfg=None, mask=None):
        self.cfg = cfg
        self.mask = mask
        
    def set_step(self, batch, sched, extra={}):
        pass
        
    def advance(self, batch, sched, out, extra={}):

        if self.mask is not None:
            mask = self.mask
        else:
            mask = batch['pad_mask'].bool()

        if 'seq_traj' not in extra: extra['seq_traj'] = []
        
        t, s = sched['sequence']

        L = mask.sum(-1) # batch['aatype'].shape[1]
        
        num_mask = (t * L).round().int() # int(round(t * L))    
        num_unmask = L - num_mask
        new_num_mask = (s * L).round().int()
        
        logits = out['aatype'] 
        
        
        
        is_unmask = (batch['aatype'] != MASK_IDX) & mask
        is_mask = (batch['aatype'] == MASK_IDX) & mask

        if 'seq_temp' in sched:
            temp, _ = sched['seq_temp']
        else:
            temp = self.cfg.temp_start * t + (1-t) * self.cfg.temp_end

        probs = logits.softmax(-1)
        entropy = -(probs * (probs+1e-12).log()).sum(-1)
        oh = torch.nn.functional.one_hot(
            batch['aatype'], 
            num_classes=probs.shape[-1]
        )
        denom = 0.5 * oh + 0.05
        new_probs = probs / denom
        new_probs /= new_probs.sum(-1, keepdims=True)
        if self.cfg.logits == 'standard':
            curr_tok_logits = Categorical(
                logits=new_probs.log() / temp
            ).log_prob(batch['aatype'])
        elif self.cfg.logits == 'gumbel':
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            curr_tok_logits = Categorical(
                logits = (new_probs.log() + gumbel_noise) / temp
            ).log_prob(batch['aatype'])
        
        # is_mask_prob = ((probs - oh) / (new_probs - oh))[...,0]

        if self.cfg.adjust_unmasked:
            logits = torch.where(is_unmask[...,None], new_probs.log(), logits)
        
        if self.cfg.logits == 'standard':
            cat = Categorical(logits=logits / temp)
            sample_ = cat.sample()
            scores_ = cat.log_prob(sample_)
            
        elif self.cfg.logits == 'gumbel':
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise 
            scores_, sample_ = (logits / temp).log_softmax(dim=-1).max(dim=-1) 

        if self.cfg.strategy == 'dplm':

            if self.cfg.replace_unmasked:
                # used the logits of the current token, not the sampled token
                scores_ = torch.where(is_unmask, curr_tok_logits, scores_)
                sample_ = torch.where(is_unmask, batch['aatype'], sample_)
            
            topk = topk_masking(
                scores_,
                L - new_num_mask, 
                mask=mask,
                temp=self.cfg.topk_temp
            )
            
            if self.cfg.allow_mutation:
                sample = sample_
            else:
                sample = torch.where(is_mask, sample_, batch['aatype'])
            
            sample = torch.where(topk, sample, MASK_IDX)
            batch['aatype'] = torch.where(mask, sample, batch['aatype'])

        elif self.cfg.strategy == 'random':

            topk = topk_masking(
                torch.rand_like(scores_),
                num_mask - new_num_mask,
                mask=is_mask & mask,
            )
            
            batch['aatype'] = torch.where(topk, sample_, batch['aatype'])

        elif self.cfg.strategy == 'entropy':
            
            topk = topk_masking(
                -entropy,
                num_mask - new_num_mask,
                mask=is_mask & mask,
            )
            
            batch['aatype'] = torch.where(topk, sample_, batch['aatype'])
        
        elif self.cfg.strategy == 'purity':
            
            topk = topk_masking(
                scores_,
                num_mask - new_num_mask,
                mask=is_mask & mask,
            )
            
            batch['aatype'] = torch.where(topk, sample_, batch['aatype'])

        batch['aatype'] = batch['aatype'].long()
        extra['seq_traj'].append(batch['aatype'])
        
        return batch
        
