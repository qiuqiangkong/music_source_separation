import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor

from mss.models2.dsp.analytic import analytic_to_real, real_to_analytic
from mss.models2.dsp.banks import mel_linear_banks
from mss.models2.dsp.convolve import fftconvolve
from mss.models2.dsp.resample import UpSample
from mss.utils import fast_sdr


class SubbandFilter(nn.Module):
    r"""Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        filter_len: int = 10001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        self.sr = sr
        self.banks = banks
        self.filter_len = filter_len
        self.window_type = "hamming"

        n_banks = len(banks)
        w = torch.empty((n_banks, self.filter_len))  # (k, m)

        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])

        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

    def analysis(self, x: Tensor) -> Tensor:
        r"""Split signal into subbands.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, l)

        Returns:
            out: (b, c, k, l)
        """
        
        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)
        x = fftconvolve(x, self.w)  # (b*c, k, l)
        return rearrange(x, '(b c) k l -> b c k l', b=B)  # (b, c, k, l)

    def synthesis(self, x: Tensor) -> Tensor:
        r"""Sum subband signals into original signal.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, k, l)

        Returns:
            out: (b, c, l)
        """
        return x.sum(2)

    
class SubbandResampler(nn.Module):
    def __init__(
        self, 
        sr: int,
        banks: list[tuple[float, float]],
        factor: int,
        filter_len=10001
    ):
        super().__init__()

        self.sr = sr
        self.factor = factor
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.upsample = UpSample(self.factor, filter_len)

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
        x = x[:, :, :, 0 :: self.factor]  # (b, c, k, l_out)

        return x

    def synthesis(self, x: Tensor) -> Tensor:
        r"""Upsample subband signals.

        b: batch_size
        c: audio_channels
        k: n_bands

        Args:
            x: (b, c, k, l_in)

        Returns:
            out: (b, c, k, l_out)
        """
        
        B, C, K, L = x.shape

        # Upsample
        x = rearrange(x, 'b c k l -> (b c k) l')
        x = self.upsample(x.real) + 1.j * self.upsample(x.imag)
        x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)
        
        # Move back to ω0
        x = self.shift_frequency(x, self.f_center)

        # Analytic to original signal
        x = analytic_to_real(x)

        return x

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
    max_bandwidth = 1200
    filter_len = 10001
    factor = sr // max_bandwidth
    device = "cuda"
    
    # Melbanks
    banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, filter_len).to(device)
    sb_resampler = SubbandResampler(sr, banks, factor, filter_len).to(device)

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