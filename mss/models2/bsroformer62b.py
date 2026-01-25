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



class BSRoformer62b(Fourier):
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
        head_dim = 32

        self.patch1 = nn.Conv2d(in_channels=self.ac * 2, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.encoder1 = EncoderFreq(32, 32//head_dim, n_layers=1)

        self.bandsplit = BandSplit(sr=sample_rate, n_fft=n_fft, n_bands=256, in_channels=32, out_channels=64)
        self.encoder2 = Encoder(64, 64//head_dim, n_layers=1)

        self.patch3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.encoder3 = Encoder(128, 128//head_dim, n_layers=2)

        self.patch4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.encoder4 = Encoder(256, 256//head_dim, n_layers=4)

        self.decoder4 = Decoder(256, 256//head_dim, n_layers=4)
        self.unpatch4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))

        self.decoder3 = Decoder(128, 128//head_dim, n_layers=2)
        self.unpatch3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.decoder2 = Decoder(64, 64//head_dim, n_layers=1)
        # self.unpatch2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2))

        self.decoder1 = DecoderFreq(32, 32//head_dim, n_layers=1)
        self.unpatch1 = nn.ConvTranspose2d(in_channels=32, out_channels=self.ac * 2, kernel_size=(1, 1), stride=(1, 1))

        # # RoPE
        # head_dim = dim // n_heads
        # head_dim = 16
        self.rope = RoPE(head_dim=head_dim, max_len=rope_len)

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
        
        #
        x = self.patch1(x)
        # x = self.encoder1(x, self.rope)
        enc1 = x

        x = self.bandsplit.transform(x)
        # x = self.encoder2(x, self.rope)
        # enc2 = x

        x = self.patch3(x)
        # x = self.encoder3(x, self.rope)
        # enc3 = x

        x = self.patch4(x)
        x = self.encoder4(x, self.rope)
        # enc4 = x
        
        # 
        x = self.decoder4(x, self.rope)
        x = self.unpatch4(x)
        # dec3 = x

        # x = enc3 + dec3
        # x = self.decoder3(x, self.rope)
        x = self.unpatch3(x)
        # dec2 = x

        # x = enc2 + dec2
        # x = self.decoder2(x, self.rope)
        x = self.bandsplit.inverse_transform(x)
        # dec1 = x

        # x = enc1 + dec1
        # x = self.decoder1(x, self.rope)
        x = self.unpatch1(x)

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

# class Patch(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels=in_channels, 
#             out_channels=out_channels, 
#             kernel_size=kernel_size, 
#             stride=stride, 
#             padding=padding
#         )

#     def __call__(self, x):
#         x = self.conv(x)
#         return x


# class UnPatch(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

#     def __call__(self, x, audio_channels):
#         x = self.conv(x)
#         x = rearrange(x, 'b (c i) t f -> b c t f i', c=audio_channels)
        
#         return x


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()

        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def __call__(self, x, rope):

        B = x.shape[0]

        for t_block, f_block in zip(self.t_blocks, self.f_blocks):
            
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        return x


class EncoderFreq(nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()

        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def __call__(self, x, rope):

        B = x.shape[0]

        for f_block in self.f_blocks:
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = f_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()

        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def __call__(self, x, rope):

        B = x.shape[0]
        
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):
            # from IPython import embed; embed(using=False); os._exit(0)
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        return x



class DecoderFreq(nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()

        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def __call__(self, x, rope):

        B = x.shape[0]
        
        for f_block in self.f_blocks:
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = f_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x