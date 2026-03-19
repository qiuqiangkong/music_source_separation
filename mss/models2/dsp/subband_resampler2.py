import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor, LongTensor

from mss.models2.dsp.analytic import analytic_to_real, real_to_analytic
from mss.models2.dsp.banks import mel_linear_banks
from mss.models2.dsp.subband import SubbandFilter
from mss.models2.dsp.convolve import fftconvolve
from mss.models2.dsp.resample import UpSample
from mss.utils import fast_sdr

    
class SubbandResampler2(nn.Module):
    def __init__(
        self, 
        sr: int,
        banks: list[tuple[float, float]],
        indices: list[list[int]],
        factors: list[int],
        filter_len=10001
    ):
        super().__init__()

        self.sr = sr
        # self.indices = indices
        self.factors = factors
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        # bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        # assert max(bandwidths) <= sr / self.factor

        for i in range(len(indices)):
            self.register_buffer(f"indices_{i}", LongTensor(indices[i]))  # (k,)

        self.upsamples = nn.ModuleList([UpSample(factor, filter_len) for factor in self.factors]) 

    def analysis(self, x: Tensor) -> Tensor:
        r"""Downsample subband signals.

        b: batch_size
        c: audio_channels
        k: n_bands

        Args:
            x: (b, c, k, l_in)

        Returns:
            out: (b, c, k, l_out)
        """
        
        # Get analytic signal
        x = real_to_analytic(x)  # (b, c, k, l_in)

        # Move ω0 to center
        x = self.shift_frequency(x, -self.f_center)  # (b, c, k, l_in)

        # Downsample
        out = []
        for i in range(len(self.factors)):
            indices = getattr(self, f"indices_{i}")
            out.append(x[:, :, indices, 0 :: self.factors[i]])  # (b, c, k_i, l_out)

        return out

    def synthesis(self, xs: Tensor) -> Tensor:
        r"""Upsample subband signals.

        b: batch_size
        c: audio_channels
        k: n_bands

        Args:
            x: (b, c, k, l_in)

        Returns:
            out: (b, c, k, l_out)
        """

        out = []

        for i in range(len(self.factors)):
           
            x = xs[i]
            B, C, K, L = x.shape

            # Upsample
            x = rearrange(x, 'b c k l -> (b c k) l')
            x = self.upsamples[i](x.real) + 1.j * self.upsamples[i](x.imag)
            x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)

            # Move back to ω0
            indices = getattr(self, f"indices_{i}")
            x = self.shift_frequency(x, self.f_center[indices])

            # Analytic to original signal
            x = analytic_to_real(x)
            out.append(x)

        out = torch.cat(out, dim=2)
        return out

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x


if __name__ == '__main__':
    
    sr = 48000
    n_bands = 64
    W = 1200
    filter_len = 10001
    factor = sr // W
    device = "cuda"

    factors = [factor * 8, factor * 4, factor * 2, factor]
    
    # Melbanks
    banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=W)
    sb_filter = SubbandFilter(sr, banks, filter_len).to(device)
    sb_resampler = SubbandResampler2(sr, banks, factors, filter_len).to(device)

    for _ in range(20):

        # Audio
        audio = np.random.uniform(low=-1, high=1, size=(4, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (c, l)
        
        # Analysis
        t0 = time.time()
        x = sb_filter.analysis(audio)  # (b, c, k, l)
        latent = sb_resampler.analysis(x)
        
        # Synthesis
        y = sb_resampler.synthesis(latent)
        y = sb_filter.synthesis(y)
        
        # Print
        t1 = time.time() - t0
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"time: {t1:.4f} s, latent: {latent.shape}, SDR: {sdr:.2f} dB")