import torch
import torch.nn as nn
from scipy.signal import firwin
from torch import Tensor
import torch.nn.functional as F

from .convolve import fftconvolve, fftconvolve_complex


'''
class UpSample(nn.Module):
    def __init__(self, factor: int, filter_len=10001):
        super().__init__()

        self.factor = factor

        w = firwin(
            numtaps=filter_len, 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        ) * self.factor
        
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

        B, L = x.shape
        out = torch.zeros((B, L * self.factor), device=x.device, dtype=x.dtype)  # (b, l_out)
        out[:, ::self.factor] = x  # (b, l_out)
        out = fftconvolve(out[:, None, :], self.w[None, None, :])[:, 0, :]  # (b, out)
        return out
'''

class UpSample(nn.Module):
    def __init__(self, factor: int, filter_len=10001):
        super().__init__()

        self.factor = factor

        w = firwin(
            numtaps=filter_len, 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        ) * self.factor
        
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
        B, L = x.shape
        out = torch.zeros((B, L * self.factor), device=x.device, dtype=x.dtype)  # (b, l_out)
        out[:, ::self.factor] = x  # (b, l_out)
        out = fftconvolve_complex(out[:, None, :], self.w[None, None, :])[:, 0, :]  # (b, out)
        return out

'''
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

        B = x.shape[0]
        L = x.shape[-1] * self.factor
        out = torch.zeros((B, L), device=x.device, dtype=x.dtype)  # (b, l_out)
        out[:, ::self.factor] = x  # (b, l_out)
        out = fftconvolve_complex(out[:, None, :], self.w[None, None, :], mode="full")[:, 0, :]  # (b, out)
        out = to_same(out, self.filter_len, L)
        # from IPython import embed; embed(using=False); os._exit(0)

        # import numpy as np
        # b1 = np.convolve(x0[0].data.cpu().numpy(), self.w.data.cpu().numpy(), mode='full')

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(b1[0 : 1000], c="r")
        # plt.plot(out[0, 0:1000].data.cpu().numpy(), c="b")
        # plt.savefig("_zz.pdf")
        

        return out
'''

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