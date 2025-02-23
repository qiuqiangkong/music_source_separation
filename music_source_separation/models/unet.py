from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
from dataclasses import dataclass

from music_source_separation.models.fourier import Fourier


@dataclass
class UNetConfig:
    n_fft: int = 2048
    hop_length: int = 441


class UNet(Fourier):
    def __init__(self, config: UNetConfig) -> None:
        
        super(UNet, self).__init__(
            n_fft=config.n_fft, 
            hop_length=config.hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.ds_factor = 16  # Downsample factor
        
        self.audio_channels = 2
        self.cmplx_num = 2
        in_channels = self.audio_channels * self.cmplx_num

        self.encoder_block1 = EncoderBlock(in_channels, 16)
        self.encoder_block2 = EncoderBlock(16, 64)
        self.encoder_block3 = EncoderBlock(64, 256)
        self.encoder_block4 = EncoderBlock(256, 1024)
        self.middle = EncoderBlock(1024, 1024)
        self.decoder_block1 = DecoderBlock(1024, 256)
        self.decoder_block2 = DecoderBlock(256, 64)
        self.decoder_block3 = DecoderBlock(64, 16)
        self.decoder_block4 = DecoderBlock(16, 16)

        self.post_fc = nn.Conv2d(
            in_channels=16, 
            out_channels=in_channels, 
            kernel_size=1, 
            padding=0,
        )

    def forward(self, audio):
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

        # Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)

        # pad stft
        x, pad_t = self.pad_tensor(x)  # x: (b, d, t, f)

        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        _, h = self.middle(x4)
        x5 = self.decoder_block1(h, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)
        x = self.post_fc(x8)

        x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
        x = x.to(torch.float)  # compatible with bf16
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        
        # Unpad mask to the original shape
        mask = self.unpad_tensor(mask, pad_t)  # shape: (b, c, t, f)

        # Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f=1025)
        
        Outpus:
            output: E.g., (b, c, t=208, f=1024)
        """

        # Pad last frames, e.g., 201 -> 208
        T = x.shape[2]
        pad_t = -T % self.ds_factor
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        # Remove last frequency bin, e.g., 1025 -> 1024
        x = x[:, :, :, 0 : -1]

        return x, pad_t

    def unpad_tensor(self, x: torch.Tensor, pad_t: int) -> torch.Tensor:
        """Unpad a spectrum to the original shape.

        Args:
            x: E.g., (b, c, t=208, f=1024)
        
        Outpus:
            x: E.g., (b, c, t=201, f=1025)
        """

        # Pad last frequency bin, e.g., 1024 -> 1025
        x = F.pad(x, pad=(0, 1))

        # Unpad last frames, e.g., 208 -> 201
        x = x[:, :, 0 : -pad_t, :]

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

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

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            output: (b, c_out, t, f)
        """
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))

        if self.is_shortcut:
            return self.shortcut(x) + h
        else:
            return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(EncoderBlock, self).__init__()

        self.pool_size = 2

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            latent: (b, c_out, t, f)
            output: (b, c_out, t/2, f/2)
        """

        latent = self.conv_block(x)  # shape: (b, c_out, t, f)
        output = F.avg_pool2d(latent, kernel_size=self.pool_size)  # shape: (b, c_out, t/2, f/2)
        return output, latent 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(DecoderBlock, self).__init__()

        stride = 2

        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=(0, 0),
            bias=False,
        )

        self.conv_block = ConvBlock(in_channels * 2, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, c_in, t/2, f/2)

        Returns:
            output: (b, c_out, t, f)
        """

        x = self.upsample(x)  # shape: (b, c_in, t, f)
        x = torch.cat((x, latent), dim=1)  # shape: (b, 2*c_in, t, f)
        x = self.conv_block(x)  # shape: (b, c_out, t, f)
        
        return x