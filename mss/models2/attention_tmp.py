from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor

from mss.models.rope import RoPE
import time



class MLP(nn.Module):
    r"""Ref: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        # n_hidden = int(8 * n_embd / 3)
        n_hidden = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        r"""Causal self attention.

        Args:
            x: (b, l, d)
           
        Outputs:
            x: (b, l, d)
        """

        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.proj(x)
        return x


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
        self.mlp = MLP(dim, num_heads)

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

        torch.cuda.synchronize()
        t1 = time.time()
        # from IPython import embed; embed(using=False); os._exit(0)
        x = x + self.attn(self.norm1(x), rope, pos)

        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()
        x = x + self.mlp(self.norm2(x))

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        return x


'''
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

        torch.cuda.synchronize()
        t1 = time.time()
        # from IPython import embed; embed(using=False); os._exit(0)
        x = x + self.attn(self.norm1(x), rope, pos)

        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()
        x = x + self.ffn(self.norm2(x))

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        return x
'''

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
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=None, 
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x