from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.signal import firwin, firwin2
from torch import Tensor

from .banks import filterbank_frequencies
from .convolve import fftconvolve
from .utils import (complex_to_real_dtype, fast_sdr, real_to_complex_dtype,
                    shift_frequency, to_analytic_filter)


class Subband(nn.Module):
    """Split a signal into complex subbands."""

    def __init__(
        self,
        sr: int,
        freqs: list[float],
        downsample_factor: int,
        chunk_size: int = 4,
    ):
        super().__init__()

        self.sr = sr
        self.freqs = freqs
        self.factor = downsample_factor
        self.chunk_size = chunk_size
        
        # Precision
        self.real_dtype = torch.float64
        self.cmplx_dtype = torch.complex128

        # Analysis bandpass filter
        self.bp_n_taps = sr  # bandpass n_taps
        self.bp_window = ("kaiser", 8.0)  # bandpass window

        # Synthesis interpolation filter
        self.interp_n_taps = sr
        self.interp_window = ("kaiser", 8.0)

        bandwidths = np.r_[freqs[1] - freqs[0], freqs[2:] - freqs[:-2], freqs[-1] - freqs[-2]]
        assert bandwidths.max() < sr / self.factor

        # Analysis filter
        filters = [self.build_subband_filter(k) for k in range(len(freqs))]
        self.register_buffer("filters", torch.stack(filters).to(self.cmplx_dtype))  # (k, 1, m)

        # Center frequency
        self.register_buffer("f_center", torch.from_numpy(freqs).to(self.real_dtype))  # (k,)

        # Synthesis interpolation filter
        interp = self.build_interp_filter(self.interp_n_taps, self.factor, self.interp_window)
        self.register_buffer("interp", interp.to(self.real_dtype))

    def analysis(self, x: Tensor, pad: int = 0) -> Tensor:
        """Convert a real signal into downsampled complex subbands.

        b: batch_size
        c: n_channels
        k: n_subbands
        l: n_samples
        l': n_downsampled_samples

        Args:
            x: (b, c, l), real signal
            pad: int

        Returns:
            out: (b, c, k, l'), complex subbands
        """

        with torch.autocast(device_type=x.device.type, enabled=False):

            out_dtype = real_to_complex_dtype(x.dtype)
            x = x.to(self.real_dtype)

            if pad:
                ex_pad = -(x.shape[-1] + pad + pad) % self.factor
                x = F.pad(x, (pad, pad + ex_pad))

            # 1. Subband
            B = x.shape[0]
            x = rearrange(x, "b c l -> (b c) 1 l")
            x = fftconvolve(x, self.filters.unsqueeze(1), True)  # (b*c, k, l)
            x = rearrange(x, "(b c) k l -> b c k l", b=B)

            # 2. Demodulate
            x = shift_frequency(x, -self.f_center, self.sr)  # (b, c, k, l)

            # 3. Downsample
            x = x[..., 0 :: self.factor]  # (b, c, k, l')

            x = x.to(out_dtype)

        return x

    def synthesis(self, x: Tensor, unpad: int = 0, length=None) -> Tensor:
        """Reconstruct a real signal from complex subbands.

        b: batch_size
        c: n_channels
        k: n_subbands
        l: n_samples
        l': n_downsampled_samples

        Args:
            x: (b, c, k, l'), complex subbands
            unpad: int
            length: int

        Returns:
            out: (b, c, l), real signal
        """

        with torch.autocast(device_type=x.device.type, enabled=False):

            out_dtype = complex_to_real_dtype(x.dtype)
            x = x.to(self.cmplx_dtype)

            # 1. Upsample
            B, C, K, L = x.shape
            x = rearrange(x, 'b c k l -> (b c k) 1 l')  # (b*c*k, 1, l')
            out = torch.zeros((B * C * K, 1, L * self.factor), device=x.device, dtype=torch.complex64)
            out[..., ::self.factor] = x  # (b*c*k, 1, l)

            x = fftconvolve(out, self.interp[None, None, :], True)  # (b*c*k, 1, l)
            x = rearrange(x, '(b c k) 1 l -> b c k l', b=B, c=C)  # (b, c, k, l)

            # 2. Modulate
            x = shift_frequency(x, self.f_center, self.sr)  # (b, c, k, l)

            # 3. Sum bands
            x = x.sum(dim=2).real  # (b, c, l)

            if unpad:
                assert length is not None
                x = x[..., unpad : unpad + length]

            x = x.to(out_dtype)

        return x

    def build_subband_filter(self, k: int) -> Tensor:
        """Design the analytic analysis filter for the k-th subband.

        Args:
            k: subband_index

        Returns:
            h: (n_taps,), complex filter
        """

        # Lowpass
        if k == 0:
            freq = [0, self.freqs[1], self.sr / 2]
            gain = [1, 0, 0]

        # Bandpass
        elif k == 1:
            freq = [self.freqs[k - 1], self.freqs[k], self.freqs[k + 1], self.sr / 2]
            gain = [0, 1, 0, 0]

        elif 2 <= k < len(self.freqs) - 2:
            freq = [0, self.freqs[k - 1], self.freqs[k], self.freqs[k + 1], self.sr / 2]
            gain = [0, 0, 1, 0, 0]

        elif  k == len(self.freqs) - 2:
            freq = [0, self.freqs[k - 1], self.freqs[k], self.freqs[k + 1]]
            gain = [0, 0, 1, 0]

        # Highpass
        elif k == len(self.freqs) - 1:
            freq = [0, self.freqs[-2], self.sr / 2]
            gain = [0, 0, 1]

        return self.build_firwin(freq, gain)

    def build_firwin(
        self,
        freq: list[float],
        gain: list[float],
    ) -> Tensor:
        """Design a complex filter."""

        n = self.bp_n_taps
        n_design = n if n % 2 else n - 1

        h = firwin2(
            numtaps=n_design,
            freq=freq,
            gain=gain,
            fs=self.sr,
            window=None,
        )
        h = to_analytic_filter(h, self.bp_window)
        h = self.pad(h)
        return torch.from_numpy(h)

    def build_interp_filter(
        self,
        n_taps: int,
        factor: int,
        window,
    ) -> Tensor:
        """Design the interpolation low-pass filter for upsampling."""

        n_design = n_taps if n_taps % 2 else n_taps - 1

        h = factor * firwin(
            numtaps=n_design,
            cutoff=1 / factor,
            window=window,
        )
        h = self.pad(h)
        return torch.from_numpy(h)

    def pad(self, h: np.ndarray) -> np.ndarray:
        """Pad an odd-length FIR filter to an even length."""

        if len(h) % 2 == 1:
            h = np.pad(h, (0, 1))
        return h


if __name__ == "__main__":

    sr = 48000
    n_bands = 128
    subband_max_bandwidth = 800
    subband_passband = 780

    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    downsample_factor = sr // subband_max_bandwidth
    device = "cuda"

    # Mel-to-linear frequency points
    freqs = filterbank_frequencies(
        sr=sr, 
        n_freqs=n_bands, 
        max_spacing=subband_passband / 2, 
        freq_scale="mel_linear"
    )

    # Subband filter
    sb_filter = Subband(
        sr=sr, 
        freqs=freqs, 
        downsample_factor=downsample_factor, 
        chunk_size=chunk_size
    ).to(device)

    rs = np.random.RandomState(1234)

    for _ in range(20):
        # Audio
        audio = rs.uniform(low=-1, high=1, size=(3, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (b, c, l)
                
        # Analysis
        t0 = time.time()
        if False:  # No padding, 36 dB.
            x = sb_filter.analysis(audio)  # (b, c, k, l')
            y = sb_filter.synthesis(x)  # (b, c, l)
        else:   # Padding. 95 dB.
            pad = 1000
            x = sb_filter.analysis(audio, pad=pad)  # (b, c, k, l')
            y = sb_filter.synthesis(x, unpad=pad, length=audio.shape[-1])  # (b, c, l)
        
        t1 = time.time() - t0
        
        # Print
        print(x.shape, y.shape)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"time: {t1:.4f} s, latent: {x.shape}, SDR: {sdr:.2f} dB")