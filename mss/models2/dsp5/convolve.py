from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .utils import fix_length


def fftconvolve(x: Tensor, h: Tensor, use_complex_fft=True, mode="same") -> Tensor:
    """Convolution with FFT.

    b: batch_size
    i: in_channels
    o: out_channels
    l: len_x
    n: len_h

    Args:
        x: (b, i, l), real/complex
        h: (o, i, n), real/complex

    Returns:
        out: (b, o, l), real/complex
    """

    L1 = x.shape[-1]
    L2 = h.shape[-1]
    L = L1 + L2 - 1  # full convolution length

    if use_complex_fft:
        fft = torch.fft.fft
        ifft = torch.fft.ifft
    else:
        fft = torch.fft.rfft
        ifft = torch.fft.irfft

    # pad both sequences to length L
    X = fft(torch.nn.functional.pad(x, (0, L - L1)))
    H = fft(torch.nn.functional.pad(h, (0, L - L2)))
    
    # multiply in frequency domain and IFFT
    Y = torch.einsum('bil,oil->bol', X, H)  # (b, o, l)
    y = ifft(Y, n=L)

    if mode == "full":
        return y

    elif mode == "same":
        start = (L2 - 1) // 2
        end = start + L1
        return y[:, :, start:end]  # (b, o, l)

    else:
        raise ValueError(mode)


def polyphase_fftconvolve(
    x: Tensor,
    h: Tensor,
    stride: int,
) -> Tensor:
    r"""Strided convolution using polyphase decomposition and FFT.

    Equivalent to:
        np.convolve(x, h, mode="same")[::stride]

    Args:
        x: (b, l), real/complex
        h: (k, n), real/complex
        stride: int

    Returns:
        out: (b, k, l'), real/complex
    """

    S = stride
    L = x.shape[-1]
    N = h.shape[-1]

    assert L >= N

    # Starting index of NumPy "same" in the full convolution.
    start = (N - 1) // 2
    phase = start % S
    offset = start // S

    # Align the desired output phase to phase 0.
    if phase:
        x = F.pad(x, (S - phase, 0))
        offset += 1

    # Pad lengths to multiples of stride.
    x = F.pad(x, (0, (-x.shape[-1]) % S))
    h = F.pad(h, (0, (-h.shape[-1]) % S))

    # Polyphase decomposition.
    x = rearrange(x, "b (l s) -> b s l", s=S)
    h = rearrange(h, "k (n s) -> k s n", s=S)

    K, _, Np = h.shape

    # Align filter phases.
    h_poly = torch.zeros(
        K, S, Np + 1,
        dtype=h.dtype,
        device=h.device,
    )

    h_poly[:, 0, :Np] = h[:, 0]

    if S > 1:
        h_poly[:, 1:, 1:] = torch.flip(h[:, 1:], dims=[1])

    # Convolve and sum over polyphase branches.
    out = fftconvolve(
        x,
        h_poly,
        use_complex_fft=True,
        mode="full",
    )

    # Equivalent to np.convolve(..., mode="same")[::S].
    out_len = (L - 1) // S + 1
    out = out[..., offset : offset + out_len]

    return out


if __name__ == '__main__':

    import numpy  as np

    rs = np.random.RandomState(1234)
    L1 = 100
    L2 = 21
    S = 5

    x = rs.uniform(low=-1, high=1, size=L1)
    h = rs.uniform(low=-1, high=1, size=L2)

    x2 = torch.from_numpy(x)
    h2 = torch.from_numpy(h)

    # PyTorch
    u = np.convolve(x, h, mode="same")
    v = fftconvolve(x2[None, None, :], h2[None, None, :])[0, 0].numpy()
    w = polyphase_fftconvolve(x2[None, :], h2[None, :], stride=S)[0, 0].numpy()

    print(np.abs(u - v).max())
    print(np.abs(u[::S] - w).max())