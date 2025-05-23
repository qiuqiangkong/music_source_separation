from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import librosa
from dataclasses import dataclass

from music_source_separation.models.fourier import Fourier
from music_source_separation.models.attention import Block
from music_source_separation.models.rope import build_rope
from music_source_separation.models.bandsplit import BandSplit


@dataclass
class BSRoformerConfig:

    name: str

    # STFT params
    audio_channels: int = 2
    sr: float = 44100
    n_fft: int = 2048
    hop_length: int = 441

    # Mel params
    mel_bins: int = 256
    mel_channels: int = 64

    # Transformer params
    patch_size: tuple[int, int] = (4, 4)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384


class BSRoformer(Fourier):
    def __init__(self, config: BSRoformerConfig) -> None:

        super(BSRoformer, self).__init__(
            n_fft=config.n_fft, 
            hop_length=config.hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.cmplx_num = 2
        self.in_channels = config.audio_channels * self.cmplx_num
        self.fps = config.sr // config.hop_length

        self.patch_size = config.patch_size
        self.ds_factor = self.patch_size[0]
        self.head_dim = config.n_embd // config.n_head

        # Band split
        self.bandsplit = BandSplit(
            sr=config.sr, 
            n_fft=config.n_fft, 
            bands_num=config.mel_bins,
            in_channels=self.in_channels, 
            out_channels=config.mel_channels
        )

        # Patch STFT
        self.patch = Patch(
            patch_size=self.patch_size,
            in_channels=config.mel_channels * np.prod(self.patch_size),
            out_channels=config.n_embd
        )

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.f_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # Build RoPE cache
        t_rope = build_rope(seq_len=config.n_fft, head_dim=self.head_dim)
        f_rope = build_rope(seq_len=self.fps * 20, head_dim=self.head_dim)
        self.register_buffer(name="t_rope", tensor=t_rope)  # shape: (t, head_dim/2, 2)
        self.register_buffer(name="f_rope", tensor=f_rope)  # shape: (t, head_dim/2, 2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Separation model.

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
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)

        # 1.2 Pad stft
        x, pad_t = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        # 1.4 Patchify
        x = self.patch.patchify(x)  # shape: (b, d, t, f)

        B = x.shape[0]  # batch size

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, self.t_rope, mask=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, self.f_rope, mask=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # --- 1. Decode ---
        # 3.1 Unpatchify
        x = self.patch.unpatchify(x)  # shape: (b, d, t, f)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.4 Unpad mask to the original shape
        mask = self.unpad_tensor(mask, pad_t)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        T = x.shape[2]
        pad_t = -T % self.ds_factor
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x, pad_t

    def unpad_tensor(self, x: torch.Tensor, pad_t: int) -> torch.Tensor:
        """Unpad a spectrum to the original shape.

        Args:
            x: E.g., (b, c, t=204, f)
        
        Outpus:
            x: E.g., (b, c, t=201, f)
        """

        # Unpad last frames, e.g., 204 -> 201
        if pad_t > 0:
            x = x[:, :, 0 : -pad_t, :]

        return x


class Patch(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, out_channels: int):
        super().__init__()

        self.patch_size = patch_size

        self.fc_in = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.fc_out = nn.Linear(in_features=out_channels, out_features=in_channels)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Patchify STFT. 

        b: batch_size
        d_in: in_channels
        d_out: out_channels
        t: time_frames
        f: freq_bins

        Args:
            x: (b, d_in, t, f)

        Outputs:
            x: (b, d_out, t/t2, f/f2)
        """

        t2, f2 = self.patch_size

        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=t2, f2=f2)
        x = self.fc_in(x)  # (b, t, f, d_out)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Unpatchify STFT.

        b: batch_size
        d_in: in_channels
        d_out: out_channels
        t: time_frames
        f: freq_bins

        Args:
            x: (b, d_out, t/t2, f/t2)

        Outputs:
            x: (b, d_in, t, f)
        """
        
        t2, f2 = self.patch_size
        
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_out(x)  # (b, t, f, d_in)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=t2, f2=f2)

        return x