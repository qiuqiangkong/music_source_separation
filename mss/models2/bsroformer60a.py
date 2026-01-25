from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# from mss.models.attention import Block
from mss.models2.attention60a import Block
from mss.models2.bandsplit60a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import time



class BSRoformer60a(Fourier):
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

        self.ac = audio_channels
        self.patch_size = patch_size

        self.encoder1 = EncoderFreq(self.ac * 2, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bandsplit = BandSplit(sr=sample_rate, n_fft=n_fft, n_bands=256, in_channels=32, out_channels=64)
        self.encoder2 = Encoder(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), n_layers=1)
        self.encoder3 = Encoder(64, 128, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), n_layers=2)
        self.encoder4 = Encoder(128, 256, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), n_layers=4)

        self.decoder4 = Decoder(256, 128, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), n_layers=4)
        self.decoder3 = Decoder(128, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), n_layers=2)
        self.decoder2 = Decoder(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), n_layers=1)
        self.decoder1 = DecoderFreq(32, self.ac * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # kernel_size = (4, 4)
        # kernel_size = patch_size
        # self.patch = Patch(band_dim * audio_channels, dim, kernel_size)
        # self.unpatch = UnPatch(dim, band_dim * audio_channels, kernel_size)

        # # RoPE
        head_dim = dim // n_heads
        # head_dim = 16
        self.rope = RoPE(head_dim=head_dim, max_len=rope_len)

        # # Transformer blocks
        # self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        # self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    
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
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)
        x = complex_sp
        T0 = x.shape[2]

        x = self.pad_tensor(x)

        x = torch.view_as_real(x)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')

        enc1 = self.encoder1(x, self.rope)  # (B, D, 201, 1024)
        enc2 = self.encoder2(self.bandsplit.transform(enc1), self.rope)
        enc3 = self.encoder3(enc2, self.rope)
        enc4 = self.encoder4(enc3, self.rope)

        dec3 = self.decoder4(enc4, self.rope)
        dec2 = self.decoder3(enc3 + dec3, self.rope)
        dec1 = self.decoder2(enc2 + dec2, self.rope)
        x = self.decoder1(enc1 + self.bandsplit.inverse_transform(dec1), self.rope)

        # Unpad
        x = F.pad(x[:, :, 0 : T0, :], pad=(0, 1))
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2)
        
        # 3.3 Get complex mask
        mask = torch.view_as_complex(x.contiguous())  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

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
        x = x[:, :, :, 0 : -1]

        return x
    '''

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
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)
        x = complex_sp
        T0 = x.shape[2]

        x = self.pad_tensor(x)

        x = torch.view_as_real(x)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')

        torch.cuda.synchronize()
        t1 = time.time()
        enc1 = self.encoder1(x, self.rope)  # (B, D, 201, 1024)
        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        tmp = self.bandsplit.transform(enc1)
        torch.cuda.synchronize()
        t1 = time.time()
        enc2 = self.encoder2(tmp, self.rope)
        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()
        enc3 = self.encoder3(enc2, self.rope)
        torch.cuda.synchronize()
        print("a3", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()
        enc4 = self.encoder4(enc3, self.rope)
        torch.cuda.synchronize()
        print("a4", time.time() - t1)
        torch.cuda.synchronize()

        dec3 = self.decoder4(enc4, self.rope)
        dec2 = self.decoder3(enc3 + dec3, self.rope)
        dec1 = self.decoder2(enc2 + dec2, self.rope)
        x = self.decoder1(enc1 + self.bandsplit.inverse_transform(dec1), self.rope)

        # Unpad
        x = F.pad(x[:, :, 0 : T0, :], pad=(0, 1))
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2)
        
        # 3.3 Get complex mask
        mask = torch.view_as_complex(x.contiguous())  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output
    '''
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
        x = x[:, :, :, 0 : -1]

        return x

class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

    def __call__(self, x):
        x = self.conv(x)
        return x


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x, audio_channels):
        x = self.conv(x)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=audio_channels)
        
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_layers):
        super().__init__()

        n_heads = out_channels // 32
        # n_heads = out_channels // 16 

        self.patch = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

        self.t_blocks = nn.ModuleList(Block(out_channels, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(out_channels, n_heads) for _ in range(n_layers))

    def __call__(self, x, rope):

        B = x.shape[0]
        # torch.cuda.synchronize()
        # t1 = time.time()
        x = self.patch(x)
        # torch.cuda.synchronize()
        # print(" c0", time.time() - t1)
        # torch.cuda.synchronize()

        # torch.cuda.synchronize()
        # t1 = time.time()
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):
            # from IPython import embed; embed(using=False); os._exit(0)
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # torch.cuda.synchronize()
        # print(" c1", time.time() - t1)
        # torch.cuda.synchronize()

        return x


class EncoderFreq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        n_heads = out_channels // 32
        # n_heads = out_channels // 16

        self.patch = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

        self.f_blocks = nn.ModuleList(Block(out_channels, n_heads) for _ in range(1))
        
    def __call__(self, x, rope):

        B = x.shape[0]
        # torch.cuda.synchronize()
        # t1 = time.time()
        x = self.patch(x)
        # torch.cuda.synchronize()
        # print(" c0", time.time() - t1)
        # torch.cuda.synchronize()

        # torch.cuda.synchronize()
        # t1 = time.time()
        for f_block in self.f_blocks:
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = f_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)
        # torch.cuda.synchronize()
        # print(" c1", time.time() - t1)
        # torch.cuda.synchronize()

        return x
    '''
    def __call__(self, x, rope):

        B = x.shape[0]
        x = self.patch(x)
        # from IPython import embed; embed(using=False); os._exit(0)

        for f_block in self.f_blocks:
            x = rearrange(x, 'b d t (f1 f2) -> (b t f1) f2 d', f1=16)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b t f1) f2 d -> b d t (f1 f2)', b=B, f1=16)
            # x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x
    '''


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_layers):
        super().__init__()

        n_heads = in_channels // 32
        # n_heads = in_channels // 16

        self.t_blocks = nn.ModuleList(Block(in_channels, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(in_channels, n_heads) for _ in range(n_layers))

        self.unpatch = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=kernel_size,
            padding=padding
        )

    def __call__(self, x, rope):

        B = x.shape[0]
        
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):
            # from IPython import embed; embed(using=False); os._exit(0)
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        x = self.unpatch(x)

        return x



class DecoderFreq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        n_heads = in_channels // 32
        # n_heads = in_channels // 16

        self.f_blocks = nn.ModuleList(Block(in_channels, n_heads) for _ in range(1))

        self.unpatch = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=kernel_size,
            padding=padding
        )

    def __call__(self, x, rope):

        B = x.shape[0]
        
        for f_block in self.f_blocks:
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = f_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        x = self.unpatch(x)

        return x