from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.dsp.banks import erb_linear_banks
from mss.models2.dsp.subband import SubbandFilter, SubbandResampler
from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.utils import fast_sdr


class BSRoformer84a(nn.Module):
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
        
        self.n_fft = 64
        self.hop_length = 32
        self.patch_size_t = 1
        max_bandwidth = 800
        factor = sample_rate // max_bandwidth
        
        # Subband filter
        banks = erb_linear_banks(sr=sample_rate, n_bands=64, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks)
        self.sb_resampler = SubbandResampler(sample_rate, banks, factor)

        in_channels = audio_channels * 2
        self.patch = nn.Conv1d(in_channels=in_channels, out_channels=dim, kernel_size=65, stride=32, padding=32)
        self.unpatch = nn.ConvTranspose1d(in_channels=dim, out_channels=in_channels, kernel_size=65, stride=32, padding=32)

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
        x = self.sb_resampler.analysis(x)  # (b, c, k, l')

        B, C, K, L = x.shape[0 : 4]
        x = F.pad(x, (0, 10))
        x = rearrange(torch.view_as_real(x), 'b c k l x -> (b k) (c x) l')
        x = self.patch(x)
        x = rearrange(x, '(b k) d t -> b d t k', k=K)

        if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
            self.check_sdr(audio, complex_sp)
            os._exit(0)

        # --- 2. Transformer along time and frequency axes ---
        for t_block, k_block in zip(self.t_blocks, self.k_blocks):

            x = rearrange(x, 'b d t k -> (b k) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b k) t d -> (b t) k d', b=B)
            x = k_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) k d -> b d t k', b=B)  # shape: (b, d, t, f)

        # Unpatchify
        x = rearrange(x, 'b d t k -> (b k) d t')
        x = self.unpatch(x)[:, :, 0 : L]
        x = rearrange(x, '(b k) (c x) l -> b c k l x', b=B, c=C)
        x = torch.view_as_complex(x.contiguous())
        
        x = self.sb_resampler.synthesis(x)
        out = self.sb_filter.synthesis(x)
     
        return out