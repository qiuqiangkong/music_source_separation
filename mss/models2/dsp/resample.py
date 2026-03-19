import torch
import torch.nn as nn
from scipy.signal import firwin
from torch import Tensor

from .convolve import fftconvolve


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