from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE


class BSRoformer23a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.patch_size = patch_size

        self.n_ffts = [512, 2048, 8192]

        # Band split
        self.fouriers = nn.ModuleList([Fourier(n_fft=n_fft, hop_length=480) for n_fft in self.n_ffts])
            
        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft * 4, 
            n_bands=n_bands,
            in_channels=audio_channels * 2 * len(self.n_ffts),  # real + imag
            out_channels=band_dim
        )

        self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sps = []
        xs = []
        for fourier, n_fft in zip(self.fouriers, self.n_ffts):
            complex_sp = fourier.stft(audio)  # shape: (b, c, t, f)
            complex_sps.append(complex_sp)
            if n_fft == 512:
                x = complex_sp.repeat_interleave(repeats=16, dim=-1)[..., 0 : 4097]
            elif n_fft == 2048:
                x = complex_sp.repeat_interleave(repeats=4, dim=-1)[..., 0 : 4097]
            elif n_fft == 8192:
                x = complex_sp[..., 0 : 4097]
            else:
                raise ValueError

            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = torch.view_as_real(x)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)
        T0 = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        # 1.4 Patchify
        x = self.patch(x)  # shape: (b, d, t, f)
        B = x.shape[0]
        T1 = x.shape[2]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, d, t, f)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        # Unpad
        x = x[:, :, 0 : T0, :]

        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2).contiguous()
        x = torch.view_as_complex(x)  # shape: (b, c, t, f)

        xs = x.chunk(dim=1, chunks=len(self.fouriers))

        outs = []
        for i in range(len(self.fouriers)):

            if self.n_ffts[i] == 512:
                mask = xs[i][..., 0 :: 16]
            elif self.n_ffts[i] == 2048:
                mask = xs[i][..., 0 :: 4]
            elif self.n_ffts[i] == 8192:
                mask = xs[i][..., 0 :: 1]
            else:
                raise ValueError
                
            sep_stft = mask * complex_sps[i]  # shape: (b, c, t, f)
            output = self.fouriers[i].istft(sep_stft)  # shape: (b, c, l)
            outs.append(output)

        output = torch.stack(outs, dim=0).sum(dim=0)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x