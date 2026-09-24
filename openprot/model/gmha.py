import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from ..utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
from ..utils.rigid_utils import Rotation, Rigid
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except:
    print('Could not import flash attn')
# Inherit from Function
class ScatterAttnBias(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(z, bias):
        return bias[z]

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        z, bias = inputs
        ctx.save_for_backward(z, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        z, bias = ctx.saved_tensors
        grad_bias = torch.zeros_like(bias)
        B, L, L = z.shape
        grad_bias.index_add_(0, z.flatten(), grad_output.reshape(B*L*L, -1))
        return None, grad_bias


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_3d(x, pos, max_period, min_period):
    out = rotary_emb(x[..., None, :], pos, x.shape[-1] // 2, max_period, min_period)
    return out.view(*out.shape[:-2], -1)


def rope_3d_gather(attn, value, pos, max_period, min_period):
    value = rotary_emb(
        value[..., None, :], pos, value.shape[-1] // 2, max_period, min_period
    )
    out = torch.einsum("...ij,...jxc->...ixc", attn, value)
    out = rotary_emb(out, -pos, value.shape[-1] // 2, max_period, min_period)
    return out.mean(-2)


def rotary_emb(x, pos, n_freqs, max_period, min_period):
    periods = torch.exp(
        torch.linspace(
            math.log(min_period), math.log(max_period), n_freqs, device=pos.device
        )
    )
    freqs = 2 * np.pi / periods
    freqs = torch.cat([freqs, freqs], -1)
    cos = torch.cos(pos[..., None] * freqs)
    sin = torch.sin(pos[..., None] * freqs)

    return (x * cos) + (rotate_half(x) * sin)


def sinusoidal_embedding(pos, n_freqs, max_period, min_period):
    periods = torch.exp(
        torch.linspace(
            math.log(min_period), math.log(max_period), n_freqs, device=pos.device
        )
    )
    freqs = 2 * np.pi / periods
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )


class GeometricMultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        rope=False,  # rotate scalar queries and keys
        rope_attn=False,
        rope_values=False,
        attn_mask=False,
        no_qk_points=4,
        no_v_points=8,
        dropout=0.0,
        cross_attn=False,
        qk_norm=False,
    ):
        
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.rope = rope
        self.rope_attn = rope_attn
        self.rope_values = rope_values
        self.cross_attn = cross_attn
    
        if attn_mask:
            self.attn_mask = nn.Parameter(torch.zeros(attn_mask, heads))
        else:
            self.attn_mask = None

        ## basic stuff we always need
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        
        self.q_norm = nn.LayerNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dim) if qk_norm else nn.Identity()
        
        if self.cross_attn:
            self.cross_w_q = nn.Linear(dim, dim, bias=False)
            self.cross_w_k = nn.Linear(dim, dim, bias=False)
            self.cross_w_v = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(dim, dim, bias=False)

       
    def forward(
        self,
        x,
        z,
        mask,
        idx=None,
        chain=None,
        mol_type=None,
        flash_attn=False,
    ):

        B, L, D = x.shape
        dev = x.device
        if idx is None:
            idx = torch.arange(L, device=dev)
        if mask is None:
            mask = torch.ones(B, L, D, dtype=bool, device=dev)

        
        query = self.q_norm(self.w_q(x)).view(B, L, self.heads, -1).transpose(1, 2)  # B H L D
        key = self.k_norm(self.w_k(x)).view(B, L, self.heads, -1).transpose(1, 2)

        ## scalar values
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)
        
        if self.rope_attn:
            query = rotary_emb(query, idx[:,None], query.shape[-1] // 2, 10000, 1)
            key = rotary_emb(key, idx[:,None], key.shape[-1] // 2, 10000, 1)
        
        if self.rope_values:
            value = rotary_emb(value, idx[:,None], value.shape[-1] // 2, 10000, 1)

        if flash_attn:
            mask = mask.to(query).view(B, 1, L, 1)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = torch.nn.functional.scaled_dot_product_attention(
                    query * mask,
                    key * mask,
                    value * mask
                )

        else:
            attn = query @ key.mT / math.sqrt(D / self.heads)  # B H L L
    
            if self.cross_attn:
                cross_query = self.cross_w_q(x).view(B, L, self.heads, -1).transpose(1, 2)  
                cross_key = self.cross_w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
                # B H L L
                cross_attn = cross_query @ cross_key.mT / math.sqrt(D / self.heads)  
            
            
            if self.cross_attn:
                same_poly_chain_mask = (chain[:,None] == chain[:,:,None])
                same_poly_chain_mask &= (mol_type == 0)[:,None]
                attn = torch.where(
                    same_poly_chain_mask.unsqueeze(1),
                    attn,
                    cross_attn
                )
    
            if self.attn_mask is not None:
                attn = attn + ScatterAttnBias.apply(z, self.attn_mask).permute(0, 3, 1, 2)
            
            mask = mask.view(B, 1, 1, -1)
            attn = torch.where(mask, attn, -float("inf"))
            attn = torch.softmax(attn, dim=-1)
            
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attn = self.dropout(attn) 
    
            if self.cross_attn:
                self_attn = torch.where(same_poly_chain_mask.unsqueeze(1), attn, 0.0)
                cross_attn = torch.where(same_poly_chain_mask.unsqueeze(1), 0.0, attn)
                attn = self_attn

            out = attn @ value
            
        if self.rope_values:
            out = rotary_emb(out, -idx[:,None], out.shape[-1] // 2, 10000, 1)
        
        out = out.transpose(1, 2).reshape(B, L, D)

        if self.cross_attn:
            cross_value = self.cross_w_v(x).view(B, L, self.heads, -1).transpose(1, 2)
            cross_out = cross_attn @ cross_value
            cross_out = cross_out.transpose(1, 2).reshape(B, L, D)
            out = out + cross_out
            

        out = self.w_o(out)
        
        return out