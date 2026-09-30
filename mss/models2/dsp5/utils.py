import math
import numpy as np

from scipy.signal.windows import get_window
import torch
from torch import Tensor


def real_to_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return {
        torch.float16: torch.complex32,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[dtype]


def fix_length(x: Tensor, start: int, length: int) -> Tensor:
    end = start + length
    return x[..., start : end]


def to_analytic_filter(h: np.ndarray, window: str) -> np.ndarray:
    """Convert a real FIR filter to an analytic complex FIR filter.

    Args:
        h: (n_taps,)

    Returns:
        out: (n_taps,)
    """

    n = len(h)
    H = np.fft.fft(h)
    H[1:(n + 1) // 2] *= 2
    H[n // 2 + 1:] = 0

    h = np.fft.ifft(H)
    h *= get_window(window, n, fftbins=False)

    return h


def shift_frequency(x: Tensor, f: Tensor, sr: int, n=1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)
    """

    omega = f / (sr / 2) * math.pi  # (k,)
    t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
    a = torch.exp(1.j * omega[:, None] * t[None, :])
    out = x * a
    return out


def fast_sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    r"""Compute SDR."""

    noise = est - ref
    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr