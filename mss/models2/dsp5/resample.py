from __future__ import annotations

from einops import rearrange
from torch import Tensor

from .convolve import fftconvolve
from .utils import fix_length


def polyphase_fftinterpolate(x: Tensor, h: Tensor, stride: int) -> Tensor:
    r"""Interpolate with polylphase convolution.

    b: batch_size
    l': n_downloaded_samples
    l: n_samples
    n: filter_len
    s: stride

    Args:
        x: (b, l')
        h: (n,)

    Returns:
        out: (b, l)
    """

    L = x.shape[-1] * stride
    N = h.shape[-1]
    x = rearrange(x, 'b l -> b 1 l')  # (b, 1, l')
    h = rearrange(h, '(n s) -> s 1 n', s=stride)  # (s, 1, n)

    x = fftconvolve(x, h, use_complex_fft=True, mode="full")  # (b, k, l')
    x = rearrange(x, 'b s t -> b (t s)')  # (b, l)
    out = fix_length(x, (N - 1) // 2, L)
    
    return out