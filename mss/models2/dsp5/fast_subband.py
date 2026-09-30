from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.signal import firwin, firwin2
from torch import Tensor

from .banks import filterbank_frequencies
from .convolve import polyphase_fftconvolve
from .resample import polyphase_fftinterpolate
from .utils import (complex_to_real_dtype, fast_sdr, real_to_complex_dtype,
                    shift_frequency, to_analytic_filter)


class FastSubband(nn.Module):
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

            # Subband
            B = x.shape[0]
            x = bandpass_demodulate_downsample(
                x=rearrange(x, 'b c l -> (b c) l'),
                h=self.filters, 
                sr=self.sr, 
                freq=-self.f_center, 
                downsample_factor=self.factor, 
                chunk_size=self.chunk_size
            )  # (b*c, k, l')
            x = rearrange(x, '(b c) k l -> b c k l', b=B)

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

            B = x.shape[0]
            x = upsample_modulate_sum(
                x=rearrange(x, 'b c k l -> (b c) k l'), 
                interp=self.interp, 
                sr=self.sr, 
                freq=self.f_center, 
                upsample_factor=self.factor, 
                chunk_size=self.chunk_size
            )
            x = rearrange(x, '(b c) l -> b c l', b=B)
            
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


def bandpass_demodulate_downsample(
    x: Tensor, 
    h: Tensor, 
    sr: int, 
    freq: Tensor, 
    downsample_factor: int,
    chunk_size: int
) -> Tensor:
    r"""

    b: batch_size
    l: n_samples
    l': n_donwsampled_samples
    k: n_bands
    m: filter_len

    Args:
        x: (b, l), real
        h: (k, m), complex
        sr: int
        freq: (k,)
        downsample_factor: int

    Returns:
        out: (b, k, l'), complex
    """

    B = x.shape[0]
    K = h.shape[0]
    L = math.ceil(x.shape[1] / downsample_factor)

    out = torch.empty(B, K, L, dtype=real_to_complex_dtype(x.dtype), device=x.device)  # (b, k, l')
    k = 0

    while k < K:

        # Band pass and downsample
        e = polyphase_fftconvolve(x, h[k : k + chunk_size, :], downsample_factor)  # (b, s, l_up)

        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr, downsample_factor)  # (b, s, l_up)
        
        out[:, k : k + chunk_size, :] = e  # (b, s, l_down)

        k += chunk_size

    return out


def upsample_modulate_sum(
    x: Tensor, 
    interp: Tensor, 
    sr: int, 
    freq: Tensor, 
    upsample_factor: int,
    chunk_size: int
):
    r"""

    b: batch_size
    k: n_bands
    l': n_downsampled_samples
    l: n_samples
    n: filter_len

    Args:
        x: (b, k, l'), complex
        interp: (n,), real

    Returns:
        out: (b, l), real
    """

    B, K = x.shape[0 : 2]
    L = x.shape[2] * upsample_factor
    
    out = torch.zeros(B, L, dtype=complex_to_real_dtype(x.dtype), device=x.device)  # (b, l)
    k = 0

    while k < K:
        
        # 1. Upsample
        e = x[:, k : k + chunk_size, :]
        e = rearrange(e, 'b k l -> (b k) l')  # (b*k, 1, l)
        e = polyphase_fftinterpolate(e, interp, upsample_factor)  # (b*k, l)
        e = rearrange(e, '(b k) l -> b k l', b=B)
        
        # 2. Modulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, k, l)

        # 3. Sum bands
        out.add_(e.sum(dim=1).real)  # (b, l)
        
        k += chunk_size

    return out


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
    sb_filter = FastSubband(
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