import numpy as np
import torch
import torch.nn as nn
from scipy.signal import firwin
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange

from .convolve import fftconvolve, fftconvolve_complex


class UpSample(nn.Module):
    def __init__(self, factor: int, filter_len=10001):
        super().__init__()

        self.factor = factor
        self.filter_len = filter_len

        w = Tensor(firwin(
            numtaps=to_odd(filter_len), 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        )) * self.factor

        w = pad_filter_to_even(w)

        w = rearrange(w, '(n r) -> r 1 n', r=factor)
        self.register_buffer("w", Tensor(w))

    def __call__(self, x: Tensor) -> Tensor:
        r"""Upsample.
        
        Args:
            x: (b, l_in)

        Returns:
            out: (b, l_out)
        """

        # from IPython import embed; embed(using=False); os._exit(0)
        # import matplotlib.pyplot as plt
        # plt.plot(self.w.cpu().numpy())
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        L = x.shape[-1] * self.factor
        x = fftconvolve_complex(x[:, None, :], self.w, mode="full")  # (b, r, l)
        x = rearrange(x, 'b r l -> b (l r)')
        out = to_same(x, self.filter_len, L)
        # from IPython import embed; embed(using=False); os._exit(0)



        # B, L = x.shape
        # out = torch.zeros((B, L * self.factor), device=x.device, dtype=x.dtype)  # (b, l_out)
        # out[:, ::self.factor] = x  # (b, l_out)
        # out = fftconvolve(out[:, None, :], self.w[None, None, :])[:, 0, :]  # (b, out)

        # import numpy as np
        # b1 = np.convolve(x[0].data.cpu().numpy(), self.w[0, 0].data.cpu().numpy(), mode='full')

        return out


class UpSample3(nn.Module):
    def __init__(self, factor: int, filter_len: int, n_bands: int, rotate_w):
        super().__init__()

        self.factor = factor
        self.filter_len = filter_len

        w = Tensor(firwin(
            numtaps=to_odd(filter_len), 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        )) * self.factor
        w = pad_filter_to_even(w)
        
        w = rearrange(w, '(n r) -> r 1 n', r=factor)
        w = w.repeat(1, n_bands, 1)  # (r, k, n)
        w = w * rearrange(rotate_w, 'k r -> r k 1')
        self.register_buffer("w", Tensor(w))

    def __call__(self, x: Tensor) -> Tensor:
        r"""Upsample.
        
        Args:
            x: (b, l_in)

        Returns:
            out: (b, l_out)
        """

        # from IPython import embed; embed(using=False); os._exit(0)
        # import matplotlib.pyplot as plt
        # plt.plot(self.w.cpu().numpy())
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        L = x.shape[-1] * self.factor
        x = fftconvolve_complex(x, self.w, mode="full")  # (b, r, l)
        # x = fftconvolve(x.real, self.w, mode="full") + 1.j * fftconvolve(x.imag, self.w, mode="full")  # (b*c, k, l)
        x = rearrange(x, 'b r l -> b (l r)')
        out = to_same(x, self.filter_len, L)
        # from IPython import embed; embed(using=False); os._exit(0)



        # B, L = x.shape
        # out = torch.zeros((B, L * self.factor), device=x.device, dtype=x.dtype)  # (b, l_out)
        # out[:, ::self.factor] = x  # (b, l_out)
        # out = fftconvolve(out[:, None, :], self.w[None, None, :])[:, 0, :]  # (b, out)

        # import numpy as np
        # b1 = np.convolve(x[0].data.cpu().numpy(), self.w[0, 0].data.cpu().numpy(), mode='full')

        return out


class UpSample4(nn.Module):
    def __init__(self, factor: int, filter_len: int):
        super().__init__()

        self.factor = factor
        self.filter_len = filter_len

        w = Tensor(firwin(
            numtaps=to_odd(filter_len), 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        )) * self.factor
        w = pad_filter_to_even(w)
        
        w = rearrange(w, '(n r) -> r 1 n', r=factor)
        self.register_buffer("w", Tensor(w))

    def __call__(self, x: Tensor) -> Tensor:
        r"""Upsample.
        
        Args:
            x: (b, l_in)

        Returns:
            out: (b, l_out)
        """

        L = x.shape[-1] * self.factor
        x = fftconvolve_complex(x, self.w, mode="full")  # (b, r, l)
        x = rearrange(x, 'b r l -> b (l r)')
        out = to_same(x, self.filter_len, L)

        return out


def to_odd(N):
    if N % 2 == 0:
        return N - 1
    else:
        return N


def pad_filter_to_even(h):
    if h.shape[-1] % 2 == 1:
        return F.pad(h, (0, 1))  # (n,)
    else:
        return h


def to_same(x, N, length):
    start = (N - 1) // 2
    end = start + length
    return x[..., start : end]
