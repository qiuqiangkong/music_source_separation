from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor
import time

from mss.models.rope import RoPE


class Block(nn.Module):
    r"""Self attention block.

    Ref: 
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = SelfAttention(dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(
        self,
        x: Tensor,
        rope: RoPE,
        pos: LongTensor | None,
    ) -> Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)

        Outputs:
            out: (b, l, d)
        """
        T = x.shape[2]
        x = x + self.attn(self.norm_x(x, self.norm1), rope, pos)

        x = rearrange(x, 'b d t f -> b (t f) d')
        x = x + self.ffn(self.norm2(x))
        x = rearrange(x, 'b (t f) d -> b d t f', t=T)
        
        return x
    
    def norm_x(self, x, norm):
        B, D, T, F = x.shape
        x = rearrange(x, 'b d t f -> b (t f) d')
        x = self.norm1(x)
        x = rearrange(x, 'b (t f) d -> b d t f', t=T)
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: nn.Module,
        pos: LongTensor | None
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2, 2)
            mask: (1, 1)

        Outputs:
            x: (b, l, d)
        """
        B, D, T, F_ = x.shape
        x = rearrange(x, 'b d t f -> b (t f) d')
        
        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply RoPE
        if pos is None:
            q = rope(q)  # (b, l, n, h)
            k = rope(k)  # (b, l, n, h)
        else:
            q = rope.apply_nd(q, pos)  # (b, l, n, h)
            k = rope.apply_nd(k, pos)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        y1 = F.scaled_dot_product_attention(
            query=rearrange(q, 'b (t f) n h -> (b t) n f h', t=T), 
            key=rearrange(k, 'b (t f) n h -> (b t) n f h', t=T), 
            value=rearrange(v, 'b (t f) n h -> (b t) n f h', t=T), 
            attn_mask=None, 
            dropout_p=0.0
        )

        y2 = F.scaled_dot_product_attention(
            query=rearrange(q, 'b (t f) n h -> (b f) n t h', f=F_), 
            key=rearrange(k, 'b (t f) n h -> (b f) n t h', f=F_), 
            value=rearrange(v, 'b (t f) n h -> (b f) n t h', f=F_), 
            attn_mask=None, 
            dropout_p=0.0
        )

        T2 = 4
        F2 = 8
        T1 = T // T2
        F1 = F_ // F2
        y3 = F.scaled_dot_product_attention(
            query=rearrange(q, 'b (t1 t2 f1 f2) n h -> b (t1 f1) n (t2 f2) h', t1=T1, t2=T2, f1=F1), 
            key=rearrange(k, 'b (t1 t2 f1 f2) n h -> b (t1 f1) n (t2 f2) h', t1=T1, t2=T2, f1=F1), 
            value=rearrange(v, 'b (t1 t2 f1 f2) n h -> b (t1 f1) n (t2 f2) h', t1=T1, t2=T2, f1=F1), 
            attn_mask=None, 
            dropout_p=0.0
        )

        y1 = rearrange(y1, '(b t) n f h -> b (t f) (n h)', t=T)
        y2 = rearrange(y2, '(b f) n t h -> b (t f) (n h)', f=F_)
        y3 = rearrange(y3, 'b (t1 f1) n (t2 f2) h -> b (t1 t2 f1 f2) (n h)', t1=T1, f1=F1, t2=T2)
        y = y1 + y2 + y3

        x = self.proj(y)  # (b, l, d)
        x = rearrange(x, 'b (t f) d -> b d t f', t=T)
        
        return x