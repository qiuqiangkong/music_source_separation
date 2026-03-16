import torch
from torch import Tensor


def fftconvolve(x: Tensor, h: Tensor) -> Tensor:
    """Use FFT to implement convolution.

    Args:
        x: (b, i, l), real signal
        h: (o, i, l), real signal

    Returns:
        out: (b, o, l)
    """

    L1 = x.shape[-1]
    L2 = h.shape[-1]
    L = L1 + L2 - 1  # full convolution length

    # pad both sequences to length L
    X = torch.fft.rfft(torch.nn.functional.pad(x, (0, L - L1)))
    H = torch.fft.rfft(torch.nn.functional.pad(h, (0, L - L2)))
    
    # multiply in frequency domain and IFFT
    Y = torch.einsum('bil,oil->bol', X, H)  # (b, o, l)
    y = torch.fft.irfft(Y)

    start = (L2 - 1) // 2
    end = start + L1
    return y[:, :, start:end]  # (b, o, l)