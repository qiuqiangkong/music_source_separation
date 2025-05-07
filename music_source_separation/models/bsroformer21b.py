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

        self.cmplx_num = 1
        self.in_channels = config.audio_channels * self.cmplx_num
        self.fps = config.sr // config.hop_length

        self.patch_size = config.patch_size
        self.ds_factor = 8
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
            in_channels=config.mel_channels,
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

    def forward(self, audio: torch.Tensor, target) -> torch.Tensor:
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
        complex_sp_tar = self.stft(target)

        x = torch.abs(complex_sp)

        # x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        # x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f) 

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

        # 3.4 Unpad mask to the original shape
        x = self.unpad_tensor(x, pad_t)  # shape: (b, c, t, f)

        x = x * torch.exp(1.j * torch.angle(complex_sp_tar))

        # from IPython import embed; embed(using=False); os._exit(0)

        # 3.6 ISTFT
        output = self.istft(x)  # shape: (b, c, l)

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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super(EncoderBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            latent: (b, c_out, t, f)
            output: (b, c_out, t/2, f/2)
        """
        s = rearrange(x, 'b (d1 d2) (t1 t2) (f1 f2) -> b (d1 t2 f2) t1 f1 d2', d2=2, t2=2, f2=2)
        s = s.mean(-1)

        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))
        h = F.avg_pool2d(h, kernel_size=2)  # shape: (b, c_out, t/2, f/2)

        out = s + h

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super(DecoderBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            latent: (b, c_out, t, f)
            output: (b, c_out, t/2, f/2)
        """

        s = rearrange(x, 'b (d t2 f2) t1 f1 -> b d (t1 t2) (f1 f2)', t2=2, f2=2)
        s = s.repeat(1, 2, 1, 1)
        
        x = F.interpolate(x, scale_factor=2)
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))
        out = s + h

        return out


class Patch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.encoder1 = EncoderBlock(in_channels, in_channels * 2)
        self.encoder2 = EncoderBlock(in_channels * 2, in_channels * 4)
        self.encoder3 = EncoderBlock(in_channels * 4, in_channels * 8)

        self.decoder1 = DecoderBlock(in_channels * 8, in_channels * 4)
        self.decoder2 = DecoderBlock(in_channels * 4, in_channels * 2)
        self.decoder3 = DecoderBlock(in_channels * 2, in_channels)

        self.fc_in = nn.Linear(in_features=in_channels * 8, out_features=out_channels)
        self.fc_out = nn.Linear(in_features=out_channels, out_features=in_channels * 8)

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

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_in(x)
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

        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_out(x)
        x = rearrange(x, 'b t f d -> b d t f')

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        return x