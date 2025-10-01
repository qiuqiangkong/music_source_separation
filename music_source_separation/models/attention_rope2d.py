from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from music_source_separation.models.rope import apply_rope


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.att_norm = RMSNorm(config.n_embd)
        self.att = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, 
        x: Tensor, 
        rope: Tensor, 
        pos: LongTensor | None, 
        mask: Tensor | None
    ) -> Tensor:
        r"""

        b: batch_size
        l: seq_len
        d: hid_dim
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, h/2)
            pos: (l, rope_dim)
            mask: (1, 1, l, l)

        Outputs:
            x: (b, l, d)
        """
        x = x + self.att(self.att_norm(x), rope, pos, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        r"""RMSNorm.

        Args:
            x: (b, l, d)
           
        Outputs:
            out: (b, l, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        out = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return out


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self, 
        x: Tensor, 
        rope: Tensor, 
        pos: Tensor | None, 
        mask: Tensor
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: heads_num
        h: head_dim
        k: rope_dim

        Args:
            x: (b, l, d)
            rope: (l, h/2, 2)
            pos: (l, k)
            mask: (1, 1, )

        Outputs:
            x: (b, l, d)
        """
        B, L, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, l, d)

        k = k.view(B, L, self.n_head, D // self.n_head)
        q = q.view(B, L, self.n_head, D // self.n_head)
        v = v.view(B, L, self.n_head, D // self.n_head)
        # q, k, v shapes: (b, l, n, h)

        if pos is None:
            q = rope(q)  # (b, l, n, h)
            k = rope(k)  # (b, l, n, h)
        else:
            q = rope.apply_nd(q, pos)  # (b, l, n, h)
            k = rope.apply_nd(k, pos)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=q.transpose(1, 2),  # (b, n, l, h)
            key=k.transpose(1, 2),  # (b, n, l, h)
            value=v.transpose(1, 2),  # (b, n, l, h)
            attn_mask=mask, 
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')

        # output projection
        x = self.c_proj(x)  # shape: (b, l, d)
        
        return x


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # The hyper-parameters follow https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3) 

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Causal self attention.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x