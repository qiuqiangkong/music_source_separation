from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def fractional_stft(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
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
    N = n_fft
    x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, L)
    x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x.mul_(window)  # (b, t, n)
    
    # Reserve space
    x = x.to(torch.complex64)  # (b, t, n)
    n = torch.arange(0, N, device=x.device)  # (n,)
    a = torch.exp(-1.j * 2 * math.pi / N * n / r)  # (n,)
    B, T, N = x.shape
    out = torch.zeros((B, T, N // 2 + 1, r), dtype=torch.complex64, device=x.device)  # (b, t, f, r)

    # Compute STFT for each fraction
    for i in range(r):    
        y = torch.fft.fft(x, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, f)
        out[:, :, :, i] = y[:, :, 0 : N // 2 + 1]  # (b, t, f, r)
        x.mul_(a)  # (b, t, n)

    out = rearrange(out, 'b t f r -> b t (f r)')  # (b, t, f*r)
    out = out[..., 0 : N * r // 2 + 1]  # (b, t, f*r)

    return out


def fractional_istft(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
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

    N = n_fft
    n = torch.arange(0, N, device=x.device)  # (n,)
    
    # Reserve space
    B, T = x.shape[0 : 2]
    out = torch.zeros((B, T, N), device=x.device)
    
    # Compute iSTFT for each fraction
    for i in range(r):
        y = x[:, :, i :: r]
        if i == 0:
            y_flip = torch.flip(y[..., 1 : -1], dims=[-1]).conj()
        else:
            y_flip = torch.flip(x[:, :, r - i :: r], dims=[-1]).conj()

        y = torch.cat([y, y_flip], dim=-1)  # (b, f)
        a = torch.exp(1.j * 2 * math.pi / N * n / r * i)  # (n, r)
        y = torch.fft.ifft(y, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, n, r)
        out.add_((y * a).real)
    
    # Overlap add
    out = fold(
        x=out, 
        hop_length=hop_length, 
        window=window
    )  # (b, L)
    
    out = out[:, n_fft // 2 :]  # (b, L)
    
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


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_fft = 2048
    hop_length = 512
    r = 16
    window = torch.hann_window(n_fft)
    x = torch.randn(4, 48000)

    y = fractional_stft(x, n_fft, hop_length, r, window)
    x_hat = fractional_istft(y, n_fft, hop_length, r, window, x.shape[-1])

    print(f"x (B, L): {x.shape}")
    print(f"y (B, T, F): {y.shape}")
    print("Error: {}".format((x - x_hat).abs().mean()))