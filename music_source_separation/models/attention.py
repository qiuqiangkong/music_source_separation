import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2)
            mask: (1, 1, t, t)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(self.att_norm(x), rope, mask)
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
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Causal self attention.

        b: batch size
        t: time steps
        d: latent dim
        h: heads num

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2, 2)
            mask: (1, 1, )

        Outputs:
            x: (b, t, d)
        """
        B, T, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, t, d)

        k = k.view(B, T, self.n_head, D // self.n_head)
        q = q.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head)
        # q, k, v shapes: (b, t, h, head_dim)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # q, k shapes: (b, t, h, head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v shapes: (b, h, t, head_dim)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=mask, 
            dropout_p=0.0
        )
        # shape: (b, h, t, head_dim)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.c_proj(x)  # shape: (b, t, d)
        
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