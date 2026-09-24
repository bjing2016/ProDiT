import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from ..utils.rigid_utils import Rigid, Rotation
from ..utils.checkpointing import checkpoint_blocks


def modulate(x, shift, scale, sigmoid=False):
    if shift is not None:
        if sigmoid:
            return x * scale.sigmoid() + shift
        else:
            return x * (1 + scale) + shift
    else:
        return x


def gate(x, gate_, sigmoid=False):
    if gate_ is not None:
        if sigmoid:
            return x * gate_.sigmoid()
        else:
            return x * gate_
    else:
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, dim, out):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output
        
class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, layers=2, act=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(dim))
        if act == SwiGLU:
            out_mul = 2
        else:
            out_mul = 1
        self.layers.append(nn.Linear(dim, ff_dim * out_mul))
        for i in range(layers - 2):
            self.layers.append(act())
            self.layers.append(nn.Linear(ff_dim, ff_dim * out_mul))
        self.layers.append(act())
        self.layers.append(nn.Linear(ff_dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_expand=4,
        ff_layers=2,
        pairwise_dim=128,
        pairwise_heads=4,
        rope=False,  # rotate scalar queries and keys
        rope_attn=False,
        rope_values=False,
        adaLN=False,
        readout_adaLN=False,
        attn_mask=False,
        dropout=0.0,
        token_dropout=0.0,
        cross_attn=False,
        qk_norm=False,
        act='relu',
    ):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(
            dim=dim,
            heads=heads,
            rope=rope,
            attn_mask=attn_mask,
            rope_attn=rope_attn,
            rope_values=rope_values,
            dropout=token_dropout,
            cross_attn=cross_attn,
            qk_norm=qk_norm,
        )
        self.ff = FeedForward(
            dim,
            ff_expand * dim,
            layers=ff_layers,
            act={
                'relu': nn.ReLU,
                'swiglu': SwiGLU,
                'gelu': nn.GELU,
                'silu': nn.SiLU,
            }[act]
        )

        self.adaLN = adaLN
        
        self.mha_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)
        self.mha_dropout = nn.Dropout(p=dropout)
        self.ff_dropout = nn.Dropout(p=dropout)

        if adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim)
            )
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

       
    def forward(
        self,
        x,
        z,
        mask,
        x_cond=None,
        idx=None,
        chain=None,
        mol_type=None,
    ):

        ### no pair2sequence

        if self.adaLN:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(x_cond).chunk(6, dim=-1)
            )
        else:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = [None] * 6

        
        x = x + self.mha_dropout(gate(
            self.mha(
                x=modulate(self.mha_norm(x), shift_mha, scale_mha),
                z=self.pair_norm(z) if hasattr(self, "pair_norm") else z,
                mask=mask.bool(),
                idx=idx,
                chain=chain,
                mol_type=mol_type,
            ),
            gate_mha,
        ))

        x = x + self.ff_dropout(gate(self.ff(modulate(
            self.ff_norm(x), shift_mlp, scale_mlp
        )), gate_mlp))

        return x, z


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        default_block_args = dict(
            dim=cfg.dim,
            ff_expand=cfg.ff_expand,
            heads=cfg.heads,
            rope_attn=cfg.rope_attn,
            rope_values=cfg.rope_values,
            adaLN=cfg.adaLN,
            attn_mask=cfg.attn_mask,
            cross_attn=cfg.cross_attn,
            dropout=cfg.dropout,
            token_dropout=cfg.token_dropout,
            qk_norm=cfg.qk_norm,
            act=cfg.act,
        )


        self.blocks = nn.ModuleList()

        for i in range(cfg.blocks):
            block_args = default_block_args 
            self.blocks.append(OpenProtTransformerBlock(**block_args))

        if cfg.self_cond:
            self.self_cond_emb = nn.Sequential(
                nn.LayerNorm(cfg.dim),
                nn.Linear(cfg.dim, cfg.dim),
            )
            torch.nn.init.zeros_(self.self_cond_emb[-1].weight)
            torch.nn.init.zeros_(self.self_cond_emb[-1].bias)

    def get_z(self, inp):
        

        idx = torch.where(
            inp['motif_mask'][:,None].bool() & inp['motif_mask'][:,:,None].bool(),
            inp['motif_idx'][:,None] != inp['motif_idx'][:,:,None],
            0
        )
        return idx

    def forward(self, inp):

        x = inp["x"]
        B, L, _ = x.shape
        
        residx = inp['residx']
        mask = inp["pad_mask"]
        x_cond = inp.get("x_cond", None)
        
        assert "z" not in inp
        z = self.get_z(inp)
        
        if self.cfg.ligand:
            B, K, _ = inp["y"].shape
            x = torch.cat([inp['x'], inp["y"]], 1)
            z = torch.nn.functional.pad(z, (0, K, 0, K))
            mask = torch.cat([mask, inp['ligand']['pad_mask']], 1)
            x_cond = torch.nn.functional.pad(x_cond, (0, 0, 0, K))
            residx = torch.nn.functional.pad(residx, (0, K))

        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                z,
                mask,
                x_cond=x_cond,
                idx=residx,
            )

        if self.cfg.ligand:
            x = x[:,:L]
        return {"x": x, "z": z}
