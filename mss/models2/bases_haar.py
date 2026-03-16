from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from scipy.linalg import hadamard
import numpy as np


def transform(
    x: Tensor, 
    w: Tensor,
    n_window: int, 
    hop_length: int, 
    window: Tensor
) -> Tensor:
    r"""Compute fractional Short-time Fourier Transform (STFT).

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: fractions
    f: freq_bins

    Args:
        x: (b, L)

    Returns:
        out: (b, n, f)
    """

    # Enframe    
    N = n_window
    x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, L)
    x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x.mul_(window)  # (b, t, n)
    return x @ w


def inverse_transform(
    x: Tensor, 
    w: Tensor,
    n_window: int, 
    hop_length: int, 
    window: Tensor, 
    length=None
) -> Tensor:
    r"""Compute fractional inverse Short-time Fourier Transform (iSTFT).

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: fractions
    f: freq_bins

    Args:
        x: (b, f*r)

    Returns:
        out: (b, L)
    """

    x = x @ w.T
    
    # Reserve space
    B, T = x.shape[0 : 2]
    N = n_window
    out = torch.zeros((B, T, N), device=x.device)
    
    # Overlap add
    out = fold(
        x=x, 
        hop_length=hop_length, 
        window=window
    )  # (b, L)
    
    out = out[:, n_window // 2 :]  # (b, L)
    
    if length is not None:
        out = out[:, 0 : length]  # (b, L)

    return out


def fold(x: Tensor, hop_length: int, window: Tensor | None):
    r"""Overlap-add.

    b: batch_size
    t: n_frames
    n: frame_samples
    l: segment_samples

    Args:
        x: (b, n, t)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length
    
    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)
        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


def haar_matrix(n):
    if n == 1:
        return np.array([[1]])
    H = haar_matrix(n//2)
    I = np.eye(n//2)
    top = np.kron(H, [1,1])
    bottom = np.kron(I, [1,-1])
    w = np.vstack((top, bottom))/np.sqrt(2)
    return w.astype(np.float32)


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_window = 2048
    hop_length = 512
    r = 16
    window = torch.hann_window(n_window)
    x = torch.randn(4, 48000)

    w = torch.from_numpy(haar_matrix(n_window))
    y = transform(x, w, n_window, hop_length, window)
    x_hat = inverse_transform(y, w, n_window, hop_length, window, x.shape[-1])

    print(f"x (B, L): {x.shape}")
    print(f"y (B, T, F): {y.shape}")
    print("Error: {}".format((x - x_hat).abs().mean()))