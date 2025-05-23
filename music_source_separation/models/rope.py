import torch
from torch import Tensor


def build_rope(seq_len: int, head_dim: int, base: int = 10000) -> Tensor:
    r"""Rotary Position Embedding (RoPE).
    Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py

    h: head_dim
    l: seq_len

    Args:
        seq_len: int
        head_dim: int

    Outputs:
        rope: (l, h/2, 2)
    """
    
    # Calculate θ = 1 / 10000**(2i/h)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

    # Make matrix pθ
    mat = torch.outer(torch.arange(seq_len), theta).float()  # (l, h/2)

    # Apply cos and sin
    rope = torch.stack([torch.cos(mat), torch.sin(mat)], dim=-1)  # (l, h/2, 2)

    return rope


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    r"""Apply RoPE on a tensor.

    b: batch_size
    t: time_steps
    n: head_num
    h: head_dim
    l: rope_len

    Args:
        x: (b, t, n, h)
        rope: (l, h/2, 2)

    Outputs:
        out: (b, t, n, h)
    """
    
    B, T, N, H = x.shape

    x = x.reshape(B, T, N, H//2, 2)  # (b, t, n, h/2, 2)
    rope = rope[:T][None, :, None, :, :]  # (1, t, 1, h/2, 2)
    
    x = torch.stack([
        x[..., 0] * rope[..., 0] - x[..., 1] * rope[..., 1],
        x[..., 1] * rope[..., 0] + x[..., 0] * rope[..., 1]
        ],
        dim=-1,
    )  # (b, t, n, h/2, 2)
    
    x = x.flatten(3)  # (b, t, n, h)
    
    return x


if __name__ == '__main__':

    B = 4  # batch_size
    T = 100  # time_steps
    N = 16  # heads_num
    H = 32  # head_dim
    L = 1000  # rope_len

    rope = build_rope(seq_len=L, head_dim=H)  # (l, h/2, 2)
    x = torch.Tensor(B, T, N, H)  # (b, t, n, h)

    # Apply RoPE
    out = apply_rope(x, rope)  # (b, t, n, h)

    # Plot RoPE for visualization
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(rope[:, :, 0].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(rope[:, :, 1].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("rope.pdf")
    print("Write out to rope.pdf")