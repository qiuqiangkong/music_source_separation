from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models2.bandsplit25a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE

from mss.models2.gabor_transform import GaborTransform


class BSRoformer25b(nn.Module): 
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

        # super().__init__(
        #     n_fft=n_fft, 
        #     hop_length=hop_length, 
        #     return_complex=True, 
        #     normalized=True
        # )
        super().__init__()

        # self.n_ffts = [512, 2048, 8192]
        # self.patch_size = [16, 4, 1]
        self.n_ffts = [2048]
        self.patch_size_t = 4
        self.patch_size = [e * self.patch_size_t for e in [1]]
        self.hop_lengths = [n // 4 for n in self.n_ffts]
        
        r = 16
        self.n_windows = len(self.n_ffts)

        self.fourier = GaborTransform(
            n_ffts=self.n_ffts,
            hop_lengths=self.hop_lengths,
            r=r,
        )

        self.bandsplits = nn.ModuleList([])
        for i in range(self.n_windows):
            bs = BandSplit(
                sr=sample_rate, 
                n_fft=self.n_ffts[i] * r, 
                n_bands=n_bands,
                in_channels=audio_channels * self.patch_size[i] * 2,  # real + imag
                out_channels=dim
            )
            self.bandsplits.append(bs)

        # self.patch_size = patch_size

        # # Band split
        # self.bandsplit = BandSplit(
        #     sr=sample_rate, 
        #     n_fft=n_fft, 
        #     n_bands=n_bands,
        #     in_channels=audio_channels * 2,  # real + imag
        #     out_channels=band_dim
        # )

        # self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        # self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

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
        
        x = self.pad_tensor(audio)
        
        features0 = self.fourier.encode(x)
        features = [feat[:, :, 0 : -1, :] for feat in features0]

        emb = 0

        for i in range(self.n_windows):
            x = torch.view_as_real(features[i])  # shape: (b, c, t, f, 2)
            x = rearrange(x, 'b c (t1 t2) f k -> b (c t2 k) t1 f', t2=self.patch_size[i])  # shape: (b, d, t, f)
            x = self.bandsplits[i].transform(x)  # (b, d, t, f)
            emb += x

        x = emb
        B = audio.shape[0]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        masks = []
        for i in range(self.n_windows):
            y = self.bandsplits[i].inverse_transform(x)
            y = rearrange(y, 'b (c t2 k) t1 f -> b c (t1 t2) f k', t2=self.patch_size[i], k=2)
            y = torch.view_as_complex(y)
            masks.append(y)

        # from IPython import embed; embed(using=False); os._exit(0)
        masks = [F.pad(e, (0, 0, 0, 1)) for e in masks]

        out = [feat * mask for feat, mask in zip(features0, masks)]
        out = self.fourier.decode(out, audio.shape[-1])

        return out

    def pad_tensor(self, x: Tensor) -> Tensor:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        pad1 = self.n_ffts[-1] // 2
        pad2 = -(x.shape[-1] + pad1) % (self.hop_lengths[-1] * self.patch_size_t)
        out = F.pad(x, pad=(0, pad1 + pad2))

        return out