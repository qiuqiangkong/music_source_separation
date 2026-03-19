from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.dsp.banks import mel_linear_banks
from mss.models2.dsp.subband import SubbandFilter, SubbandResampler
from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.utils import fast_sdr


class BSRoformer82a(nn.Module):
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
        
        self.ac = audio_channels
        max_bandwidth = 800
        factor = sample_rate // max_bandwidth
        
        # Subband filter
        banks = mel_linear_banks(sr=sample_rate, n_bands=64, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks)
        self.sb_resampler = SubbandResampler(sample_rate, banks, factor)

        self.pre = nn.Linear(self.ac * len(banks) * 2, dim)
        self.post = nn.Linear(dim, self.ac * len(banks) * 2)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)
        self.blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

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

        if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
            self.check_sdr(audio, x)
            os._exit(0)

        x = rearrange(torch.view_as_real(x), 'b c k l x -> b l (c k x)')
        x = self.pre(x)

        for block in self.blocks:
            x = block(x, rope=self.rope, pos=None)

        x = self.post(x)
        x = rearrange(x, 'b l (c k x) -> b c k l x', c=self.ac, x=2)
        x = torch.view_as_complex(x.contiguous())

        # Subbnad
        x = self.sb_resampler.synthesis(x)
        out = self.sb_filter.synthesis(x)

        return out

    def check_sdr(self, audio, x_hat) -> None:
        y = self.sb_resampler.synthesis(x_hat)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")