import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import time
import librosa
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from models.fourier import Fourier


'''
class BSRoformer17a(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        # depth: int = 12,
        # dim: int = 384,
        # n_heads: int = 8
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        # self.depth = 
        # self.dim = dim

        self.cmplx_num = 2
        n_heads = 8
        
        # self.head_dim = self.dim // self.n_heads

        self.patch_size = (1, 1)
        sr = 44100
        mel_bins = 256
        out_channels = 64
        self.ds_t = 16
        self.ds_f = 16

        self.stft_to_image = StftToImage(
            in_channels=self.input_channels * self.cmplx_num, 
            sr=sr, 
            n_fft=n_fft, 
            mel_bins=mel_bins,
            out_channels=out_channels
        )

        self.fc_in = nn.Linear(
            in_features=out_channels * np.prod(self.patch_size), 
            out_features=64
        )

        if True:

            dim1 = 64
            head_dim = dim1 // n_heads
            rotary_emb1_t = RotaryEmbedding(dim=head_dim)
            rotary_emb1_f = RotaryEmbedding(dim=head_dim)
            
            # from IPython import embed; embed(using=False); os._exit(0)
            self.enc1_transformers = nn.ModuleList([])
            for _ in range(2):
                self.enc1_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim1, n_heads=n_heads, rotary_emb=rotary_emb1_t),
                    TransformerBlock(dim=dim1, n_heads=n_heads, rotary_emb=rotary_emb1_f)
                ]))

            self.dec1_transformers = nn.ModuleList([])
            for _ in range(2):
                self.dec1_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim1, n_heads=n_heads, rotary_emb=rotary_emb1_t),
                    TransformerBlock(dim=dim1, n_heads=n_heads, rotary_emb=rotary_emb1_f)
                ]))


        if True:

            dim2 = 256
            head_dim = dim2 // n_heads
            rotary_emb2_t = RotaryEmbedding(dim=head_dim)
            rotary_emb2_f = RotaryEmbedding(dim=head_dim)

            self.enc2_fc = nn.Linear(dim1 * 16, dim2)
            
            self.enc2_transformers = nn.ModuleList([])
            for _ in range(2):
                self.enc2_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim2, n_heads=n_heads, rotary_emb=rotary_emb2_t),
                    TransformerBlock(dim=dim2, n_heads=n_heads, rotary_emb=rotary_emb2_f)
                ]))

            self.dec2_transformers = nn.ModuleList([])
            for _ in range(2):
                self.dec2_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim2, n_heads=n_heads, rotary_emb=rotary_emb2_t),
                    TransformerBlock(dim=dim2, n_heads=n_heads, rotary_emb=rotary_emb2_f)
                ]))

        if True:

            dim3 = 1024
            head_dim = dim3 // n_heads
            rotary_emb3_t = RotaryEmbedding(dim=head_dim)
            rotary_emb3_f = RotaryEmbedding(dim=head_dim)

            self.enc3_fc = nn.Linear(dim2 * 16, dim3)
            
            self.enc3_transformers = nn.ModuleList([])
            for _ in range(2):
                self.enc3_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim3, n_heads=n_heads, rotary_emb=rotary_emb3_t),
                    TransformerBlock(dim=dim3, n_heads=n_heads, rotary_emb=rotary_emb3_f)
                ]))

            self.dec2_transformers = nn.ModuleList([])
            for _ in range(2):
                self.dec2_transformers.append(nn.ModuleList([
                    TransformerBlock(dim=dim3, n_heads=n_heads, rotary_emb=rotary_emb3_t),
                    TransformerBlock(dim=dim3, n_heads=n_heads, rotary_emb=rotary_emb3_f)
                ]))

        if True:

            dim_latent = 1024
            head_dim = dim_latent // n_heads
            rotary_emb_latent = RotaryEmbedding(dim=head_dim)

            self.latent_fc = nn.Linear(dim3 * 16, dim_latent)
            
            self.latent_transformers = nn.ModuleList([])
            for _ in range(4):
                self.latent_transformers.append(
                    TransformerBlock(dim=dim_latent, n_heads=n_heads, rotary_emb=rotary_emb_latent))

        self.fc_out = nn.Linear(
            in_features=64, 
            out_features=out_channels * np.prod(self.patch_size),
        )
        
    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)

        Constants:
            b: batch_size
            c: channels_num=2
            z: complex_num=2
        """
        
        t1 = time.time()

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (b, c, t, f)

        batch_size = complex_sp.shape[0]
        time_steps = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)
        # shape: (b, c, t, f, z)

        x = rearrange(x, 'b c t f z -> b (c z) t f')

        x = self.stft_to_image.transform(x)
        # shape: (b, d, t, f)

        x = self.patchify(x)
        # shape: (b, d, t, f)

        print("a1", time.time() - t1)
        t1 = time.time()

        # Enc1
        for t_transformer, f_transformer in self.enc1_transformers:

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_transformer(x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            x = f_transformer(x)

            x = rearrange(x, '(b t) f d -> b d t f', b=batch_size)

        latent1 = x
        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b (d t2 f2) t1 f1', t2=4, f2=4)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.enc2_fc(x)
        x = rearrange(x, 'b t f d -> b d t f')

        print("a2", time.time() - t1)
        t1 = time.time()

        # Enc2
        for t_transformer, f_transformer in self.enc2_transformers:

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_transformer(x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            x = f_transformer(x)

            x = rearrange(x, '(b t) f d -> b d t f', b=batch_size)

        latent2 = x
        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b (d t2 f2) t1 f1', t2=4, f2=4)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.enc3_fc(x)
        x = rearrange(x, 'b t f d -> b d t f')

        print("a3", time.time() - t1)
        t1 = time.time()

        # Enc3
        for t_transformer, f_transformer in self.enc3_transformers:

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_transformer(x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            x = f_transformer(x)

            x = rearrange(x, '(b t) f d -> b d t f', b=batch_size)

        from IPython import embed; embed(using=False); os._exit(0)

        _F = x.shape[-1]
        x = rearrange(x, 'b d t f -> b (t f) d')

        print("a4", time.time() - t1)
        t1 = time.time()

        # Enc latent
        for transformer in self.latent_transformers:

            x = transformer(x)

        x = rearrange(x, 'b (t f) d -> b d t f', f=_F)


        print("a5", time.time() - t1)
        t1 = time.time()        

        

        """
        from IPython import embed; embed(using=False); os._exit(0)

        x = self.unpatchify(x, time_steps)

        x = self.stft_to_image.inverse_transform(x)

        x = rearrange(x, 'b (c z) t f -> b c t f z', c=self.input_channels)
        # shape: (b, c, t, f, z)

        mask = torch.view_as_complex(x.contiguous())
        # shape: (b, c, t, f)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)
        """

        # return output

        return mixture + torch.mean(x)

    def patchify(self, x):

        B, C, T, Freq = x.shape
        pad_len = int(np.ceil(T / self.ds_t)) * self.ds_t - T
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        t2, f2 = self.patch_size
        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=t2, f2=f2)
        x = self.fc_in(x)  # (b, t, f, d)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def unpatchify(self, x, time_steps):
        t2, f2 = self.patch_size
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_out(x)  # (b, t, f, d)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=t2, f2=f2)

        x = x[:, :, 0 : time_steps, :]

        return x
'''

class BSRoformer17a(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        self.cmplx_num = 2
        n_heads = 12
        
        self.patch_size = (1, 1)
        sr = 44100
        mel_bins = 256
        out_channels = 64
        self.ds_t = 16
        self.ds_f = 16

        self.stft_to_image = StftToImage(
            in_channels=self.input_channels * self.cmplx_num, 
            sr=sr, 
            n_fft=n_fft, 
            mel_bins=mel_bins,
            out_channels=out_channels
        )

        gamma = 4
        self.encoder1 = BSTransformerLayers(dim=96, n_heads=n_heads, depth=2)
        self.encoder2 = BSTransformerLayers(dim=192, n_heads=n_heads, depth=2)
        self.encoder3 = BSTransformerLayers(dim=384, n_heads=n_heads, depth=2)
        self.encoder4 = BSTransformerLayers(dim=768, n_heads=n_heads, depth=2)
        # self.latent_layers = TransformerLayers(dim=1536, n_heads=n_heads, depth=4)

        
        self.fc_in = nn.Linear(out_channels * np.prod(self.patch_size), 96)
        self.encoder2_proj = nn.Linear(96 * gamma, 192)
        self.encoder3_proj = nn.Linear(192 * gamma, 384)
        self.encoder4_proj = nn.Linear(384 * gamma, 768)
        self.latent_proj = nn.Linear(768 * gamma, 1536)

        #
        self.decoder1 = BSTransformerLayers(dim=768, n_heads=n_heads, depth=2)
        self.decoder2 = BSTransformerLayers(dim=384, n_heads=n_heads, depth=2)
        self.decoder3 = BSTransformerLayers(dim=192, n_heads=n_heads, depth=2)
        self.decoder4 = BSTransformerLayers(dim=96, n_heads=n_heads, depth=2)

        # self.decoder1_proj = nn.Linear(1536 // gamma, 768)
        self.decoder1_proj = nn.Linear(768, 768)
        self.decoder2_proj = nn.Linear(768 // gamma, 384)
        self.decoder3_proj = nn.Linear(384 // gamma, 192)
        self.decoder4_proj = nn.Linear(192 // gamma, 96)


        self.fc_out = nn.Linear(
            in_features=96, 
            out_features=out_channels * np.prod(self.patch_size),
        )
        
    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)

        Constants:
            b: batch_size
            c: channels_num=2
            z: complex_num=2
        """
        
        t1 = time.time()

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (b, c, t, f)

        batch_size = complex_sp.shape[0]
        time_steps = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)
        # shape: (b, c, t, f, z)

        x = rearrange(x, 'b c t f z -> b (c z) t f')

        x = self.stft_to_image.transform(x)
        # shape: (b, d, t, f)

        x = self.patchify(x)
        # shape: (b, d, t, f)

        #
        x1 = self.encoder1(x)
        x1_pool = rearrange(x1, 'b d (t1 t2) (f1 f2) -> b (t2 f2 d) t1 f1', t2=2, f2=2)

        #
        x = rearrange(x1_pool, 'b d t f -> b t f d')
        x = self.encoder2_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x2 = self.encoder2(x)
        x2_pool = rearrange(x2, 'b d (t1 t2) (f1 f2) -> b (t2 f2 d) t1 f1', t2=2, f2=2)

        #
        x = rearrange(x2_pool, 'b d t f -> b t f d')
        x = self.encoder3_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x3 = self.encoder3(x)
        x3_pool = rearrange(x3, 'b d (t1 t2) (f1 f2) -> b (t2 f2 d) t1 f1', t2=2, f2=2)

        #
        x = rearrange(x3_pool, 'b d t f -> b t f d')
        x = self.encoder4_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x4 = self.encoder4(x)
        x4_pool = rearrange(x4, 'b d (t1 t2) (f1 f2) -> b (t2 f2 d) t1 f1', t2=2, f2=2)

        # Latent
        # x = rearrange(x4_pool, 'b d t f -> b t f d')
        # x = self.latent_proj(x)
        # x = rearrange(x, 'b t f d -> b d t f')
        # x = self.latent_layers(x)
        x = x4_pool
        
        # Decoders
        x = rearrange(x, 'b (t2 f2 d) t1 f1 -> b d (t1 t2) (f1 f2)', t2=2, f2=2)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.decoder1_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x = self.decoder1(x + x4)

        #
        x = rearrange(x, 'b (t2 f2 d) t1 f1 -> b d (t1 t2) (f1 f2)', t2=2, f2=2)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.decoder2_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x = self.decoder2(x + x3)

        #
        x = rearrange(x, 'b (t2 f2 d) t1 f1 -> b d (t1 t2) (f1 f2)', t2=2, f2=2)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.decoder3_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x = self.decoder3(x + x2)

        #
        x = rearrange(x, 'b (t2 f2 d) t1 f1 -> b d (t1 t2) (f1 f2)', t2=2, f2=2)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.decoder4_proj(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x = self.decoder4(x + x1)

        x = self.unpatchify(x, time_steps)

        x = self.stft_to_image.inverse_transform(x)

        x = rearrange(x, 'b (c z) t f -> b c t f z', c=self.input_channels)
        # shape: (b, c, t, f, z)

        mask = torch.view_as_complex(x.contiguous())
        # shape: (b, c, t, f)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)
        

        # return output

        return output

    def patchify(self, x):

        B, C, T, Freq = x.shape
        pad_len = int(np.ceil(T / self.ds_t)) * self.ds_t - T
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        t2, f2 = self.patch_size
        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=t2, f2=f2)
        x = self.fc_in(x)  # (b, t, f, d)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def unpatchify(self, x, time_steps):
        t2, f2 = self.patch_size
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_out(x)  # (b, t, f, d)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=t2, f2=f2)

        x = x[:, :, 0 : time_steps, :]

        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class StftToImage(nn.Module):

    def __init__(self, in_channels: int, sr: float, n_fft: int, mel_bins: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.n_fft = n_fft
        self.mel_bins = mel_bins

        melbanks = librosa.filters.mel(
            sr=sr, 
            n_fft=n_fft, 
            n_mels=self.mel_bins - 2, 
            norm=None
        )

        melbank_first = np.zeros(melbanks.shape[-1])
        melbank_first[0] = 1.

        melbank_last = np.zeros(melbanks.shape[-1])
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]

        melbanks = np.concatenate(
            [melbank_first[None, :], melbanks, melbank_last[None, :]], axis=0
        )

        sum_banks = np.sum(melbanks, axis=0)
        assert np.allclose(a=sum_banks, b=1.)

        self.band_nets = nn.ModuleList([])
        self.inv_band_nets = nn.ModuleList([])
        self.indexes = []
        # 
        for f in range(self.mel_bins):
            
            idxes = (melbanks[f] != 0).nonzero()[0]
            self.indexes.append(idxes)
            
            in_dim = len(idxes) * in_channels
            self.band_nets.append(nn.Linear(in_dim, out_channels))
            self.inv_band_nets.append(nn.Linear(out_channels, in_dim))

        # 
        self.register_buffer(name='melbanks', tensor=torch.Tensor(melbanks))

    def transform(self, x):

        vs = []

        for f in range(self.mel_bins):
            
            idxes = self.indexes[f]

            bank = self.melbanks[f, idxes]  # (banks,)
            stft_bank = x[..., idxes]  # (b, c, t, banks)

            v = stft_bank * bank  # (b, c, t, banks)
            v = rearrange(v, 'b c t q -> b t (c q)')

            v = self.band_nets[f](v)  # (b, t, d)
            vs.append(v)

        x = torch.stack(vs, dim=2)  # (b, t, f, d)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def inverse_transform(self, x):

        B, _, T, _ = x.shape
        y = torch.zeros(B, self.in_channels, T, self.n_fft // 2 + 1).to(x.device)

        for f in range(self.mel_bins):

            idxes = self.indexes[f]
            v = x[..., f]  # (b, d, t)
            v = rearrange(v, 'b d t -> b t d')
            v = self.inv_band_nets[f](v)  # (b, t, d)
            v = rearrange(v, 'b t (c q) -> b c t q', q=len(idxes))
            y[..., idxes] += v

        return y


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim: int, n_heads: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_emb = rotary_emb

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Must have flash attention."
        
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        r"""
        Args:
            x: (b, t, h*d)

        Constants:
            b: batch_size
            t: time steps
            r: 3
            h: heads_num
            d: heads_dim
        """
        B, T, C = x.size()

        q, k, v = rearrange(self.c_attn(x), 'b t (r h d) -> r b h t d', r=3, h=self.n_heads)
        # q, k, v: (b, h, t, d)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        
        y = rearrange(y, 'b h t d -> b t (h d)')

        y = self.c_proj(y)
        # shape: (b, t, h*d)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_emb: RotaryEmbedding):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb)
        self.mlp = MLP(dim=dim)
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x



class BSTransformerLayers(nn.Module):
    def __init__(self, dim, n_heads, depth):
        super().__init__()

        head_dim = dim // n_heads
        rotary_emb_t = RotaryEmbedding(dim=head_dim)
        rotary_emb_f = RotaryEmbedding(dim=head_dim)
        
        self.transformers = nn.ModuleList([])

        for _ in range(depth):
            self.transformers.append(nn.ModuleList([
                TransformerBlock(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb_t),
                TransformerBlock(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb_f)
            ]))

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """
        batch_size = x.shape[0]

        for t_transformer, f_transformer in self.transformers:

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_transformer(x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            x = f_transformer(x)

            x = rearrange(x, '(b t) f d -> b d t f', b=batch_size)
        
        return x


class TransformerLayers(nn.Module):
    def __init__(self, dim, n_heads, depth):
        super().__init__()

        head_dim = dim // n_heads
        rotary_emb = RotaryEmbedding(dim=head_dim)
        
        self.transformers = nn.ModuleList([])

        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(dim=dim, n_heads=n_heads, rotary_emb=rotary_emb)
            )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """
        
        _B, _D, _T, _F = x.shape

        x = rearrange(x, 'b d t f -> b (t f) d')

        for transformer in self.transformers:
            x = transformer(x)

        x = rearrange(x, 'b (t f) d -> b d t f', t=_T)
        
        return x