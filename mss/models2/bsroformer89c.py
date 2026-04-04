from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.dsp3.banks import mel_linear_banks, erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.utils import fast_sdr


class BSRoformer89c(nn.Module):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()
        
        n_bands = 128
        self.n_fft = 32
        self.hop_length = 4
        self.patch_size_t = 4
        max_bandwidth = 400
        factor = sample_rate // max_bandwidth
        chunk_size = 16

        # Subband filter
        
        banks = erb_linear_banks(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)
        
        # Patch
        in_channels = audio_channels * self.n_fft * 2
        self.patch = Patch(in_channels, dim, (self.patch_size_t, 1))
        self.unpatch = UnPatch(dim, in_channels, (self.patch_size_t, 1))

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        k: n_bands
        l'
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, l)

        Returns:
            out: (b, c, l)
        """

        # Subband Analysis
        x = self.sb_filter.analysis(audio)  # (b, c, k, l')
        complex_sp = self.stft(x)  # (b, c, k, t, f)

        if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
            self.check_sdr(audio, complex_sp)
            os._exit(0)

        # Patchify
        B, C, K, T = complex_sp.shape[0 : 4]
        x = rearrange(torch.view_as_real(complex_sp), 'b c k t f x -> b (c f x) t k')
        x = self.pad_tensor(x, self.patch_size_t)  # x: (b, d, t, k)
        x = self.patch(x)

        # --- 2. Transformer along time and frequency axes ---
        for t_block, k_block in zip(self.t_blocks, self.k_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = k_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # Unpatchify
        x = self.unpatch(x)
        x = x[:, :, 0 : T, :]
        x = rearrange(x, 'b (c f x) t k -> b c k t f x', c=C, x=2)
        mask = torch.view_as_complex(x.contiguous())
        sep_stft = complex_sp * mask

        # Subband synthesis
        x = self.istft(sep_stft)  # (b, c, k, t, f)
        out = self.sb_filter.synthesis(x)
     
        return out

    def stft(self, x):
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c k l -> (b c k) l')
        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c k) f t -> b c k t f', b=B, c=C)
        return x

    def istft(self, x):
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c k t f -> (b c k) f t')
        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)
        return x

    def pad_tensor(self, x: Tensor, patch_size_t) -> Tensor:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t, f)
        
        Returns:
            out: E.g., (b, c, t f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % patch_size_t  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))
        return x

    def check_sdr(self, audio, complex_sp) -> None:
        y = self.istft(complex_sp)  # (b, c, k, t, f)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")


class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        return self.conv(x)


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        return self.conv(x)