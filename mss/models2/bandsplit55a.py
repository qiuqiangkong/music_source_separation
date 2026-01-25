from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


class BandSplit(nn.Module):
    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        n_bands: int, 
        in_channels: int, 
        out_channels: int
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band width
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name='ola_window', tensor=Tensor(ola_window))  # (f,)
        
        nonzero_indexes = []  # (k, w)
        nonzero_melbanks = []  # (k, w)
        
        for f in range(self.n_bands):    
            idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[0]  # shape: (w,)
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(self.melbanks[f, idxes])

        max_len = max([len(idxes) for idxes in nonzero_indexes])

        self.pre_w = nn.Parameter(torch.zeros((in_channels, max_len, out_channels)))  # (i, f, o)
        self.post_w = nn.Parameter(torch.zeros((out_channels, in_channels, max_len)))  # (o, i, f)
        self.pre_b = nn.Parameter(torch.zeros(out_channels))  # (k, o)
        self.post_b = nn.Parameter(torch.zeros((in_channels, max_len)))  # (i, f)

        gain = np.sqrt(np.linalg.pinv(self.in_channels * melbanks**2) @ np.ones(self.n_bands))  # (f)
        self.register_buffer(name='gain', tensor=Tensor(gain))  # (k, f)
        
        nn.init.uniform_(self.pre_w, -1, 1)
        bound = 1 / math.sqrt(out_channels)
        nn.init.uniform_(self.post_w, -bound, bound)

        Q = 4
        self.Q = Q
        cumsum = np.cumsum([len(idxes) for idxes in nonzero_indexes])
        total = cumsum[-1]
        subbands = []

        for q in range(Q):
            subband = []
            for i in range(n_bands):
                if q / Q * total <= cumsum[i] < (q + 1) / Q * total:
                    subband.append(i)
            subbands.append(subband)
        subbands[-1].append(i)
        
        #
        for q in range(Q):
            sb_idxes = []
            sb_melbanks = []
            for f in subbands[q]:
                sb_idxes.append(nonzero_indexes[f])
                sb_melbanks.append(nonzero_melbanks[f])

            sb_idxes = pad_sequence(sequences=sb_idxes, batch_first=True, padding_value=-1)  # (k, w)
            sb_masks = (sb_idxes != -1).float()
            sb_idxes[sb_idxes == -1] = n_fft // 2
            sb_melbanks = pad_sequence(sequences=sb_melbanks, batch_first=True, padding_value=0)  # (k, w)

            self.register_buffer(name=f'sb_idxes_{q}', tensor=sb_idxes)
            self.register_buffer(name=f'sb_melbanks_{q}', tensor=sb_melbanks)
            self.register_buffer(name=f'sb_masks_{q}', tensor=sb_masks)
            self.register_buffer(name=f'sb_subbands_{q}', tensor=LongTensor(subbands[q]))

    def init_melbanks(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize mel bins from librosa.

        f: stft bins
        k: mel bins

        Args:
            None

        Returns:
            melbanks: (k, f)
            ola_window: (f,)
        """

        melbanks = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.n_bands - 2, 
            norm=None
        )  # shape: (k, f)

        F = self.n_fft // 2 + 1
        
        # The zeroth bank, e.g., [1., 0.66, 0.32, 0, ..., 0.]
        melbank_0 = np.zeros(F)
        idx = np.argmax(melbanks[0])
        melbank_0[0 : idx] = 1. - melbanks[0, 0 : idx]  # (f,)

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]  # (f,)

        # Concatenate
        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # (n_mels, f)

        # Calculate overlap-add window
        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert ola_window.max() >= 0.5

        return melbanks, ola_window

    def transform(self, x: Tensor) -> Tensor:
        r"""Convert STFT to mel scale STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, f)

        Returns:
            x: (b, d, t, k)
        """

        B, I, T, F_ = x.shape
        O_ = self.out_channels
        out = torch.zeros((B, O_, T, self.n_bands), device=x.device)  # (b, o, t, f)
        
        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)
            W = idxes.shape[-1]
            
            _x = x[..., idxes] * melbanks * masks  # (b, i, t, s, w)
            _w = self.pre_w[:, None, 0 : W, :] * masks[None, :, :, None] * self.gain[None, subbands, None, None]  # (i, s, w, o)
            _b = self.pre_b
            _y = torch.einsum('bitsw,iswo->btso', _x, _w) + _b  # (b, c, t, s, o)
            _y = rearrange(_y, 'b t s o -> b o t s')
            out[:, :, :, subbands] = _y
            
        return out

    def inverse_transform(self, x: Tensor) -> Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, f, o)

        Outputs:
            y: (b, c, t, f)
        """
        
        # B, C, T, F, O_ = x.shape
        B, O_, T, F = x.shape
        I = self.in_channels
        out = torch.zeros((B, I, T, self.n_fft // 2 + 1), device=x.device)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)
            W = idxes.shape[-1]

            _x = x[:, :, :, subbands]  # (b, o, t, s)
            _w = self.post_w[:, :, None, 0 : W] * masks  # (o, i, s, w)
            _b = self.post_b[:, None, 0 : W] * masks  # (i, s, w)
            _y = torch.einsum('bots,oisw->bitsw', _x, _w) + _b[:, None, :, :]  # (b, o, t, s, w)
            _y = _y * melbanks
            out.scatter_add_(dim=3, index=idxes.flatten()[None, None, None, :].repeat(B, I, T, 1), src=_y.flatten(3, 4))
            
        # Divide overlap add window
        out /= self.ola_window

        return out


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(8, 4, 201, 1025)  # (b, c, t, f)

    bandsplit = BandSplit(
        sr=44100, 
        n_fft=2048, 
        n_bands=64,
        in_channels=4, 
        out_channels=64
    )

    y = bandsplit.transform(x)  # (b, d, t, k)
    x_hat = bandsplit.inverse_transform(y)  # (b, c, t, f)
    print(x - x_hat)
    print(y.abs().mean())
    print(x_hat.abs().mean())
    from IPython import embed; embed(using=False); os._exit(0)