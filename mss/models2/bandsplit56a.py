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
        # melbanks = np.pad(melbanks, pad_width=(0, 1))
        self.register_buffer(name='melbanks', tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name='ola_window', tensor=Tensor(ola_window))  # (f,)

        self.pre_w = nn.Parameter(torch.zeros((in_channels, n_fft // 2 + 1, out_channels)))  # (i, f, o)
        self.post_w = nn.Parameter(torch.zeros((out_channels, in_channels, n_fft // 2 + 1)))  # (o, i, f)
        self.pre_b = nn.Parameter(torch.zeros((self.n_bands, out_channels)))  # (k, o)
        self.post_b = nn.Parameter(torch.zeros((in_channels, n_fft // 2 + 1)))  # (i, f)

        # bound = 1 / math.sqrt(in_channels * (n_fft // 2 + 1))
        gain = np.sqrt(np.linalg.pinv(self.in_channels * melbanks**2) @ np.ones(self.n_bands))  # (f)
        gain = torch.from_numpy(gain[None, :, None])
        
        nn.init.uniform_(self.pre_w, -1, 1)
        with torch.no_grad():
            self.pre_w *= gain

        bound = 1 / math.sqrt(out_channels)
        nn.init.uniform_(self.post_w, -bound, bound)

        # debug = False
        # if debug:
        #     nn.init.uniform_(self.pre_b, -1, 1)
        #     nn.init.uniform_(self.post_b, -1, 1)

        nonzero_indexes = []  # (k, w)
        nonzero_melbanks = []  # (k, w)

        for f in range(self.n_bands):    
            idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[0]  # shape: (w,)
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(self.melbanks[f, idxes])

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

        ##
        melbanks = np.zeros((self.n_bands, self.n_fft // 2 + 1))

        for i in range(128):
            melbanks[i, i] = 1

        for i in range(32):
            melbanks[i + 128, 128 + i*4 : 128 + (i+1)*4] = 1

        for i in range(32):
            melbanks[i + 160, 256 + i*8 : 256 + (i+1)*8] = 1

        for i in range(64):
            melbanks[i + 192, 512 + i*8 : 512 + (i+1)*8] = 1

        melbanks[-1, -1] = 1
        # import matplotlib.pyplot as plt
        # plt.matshow(melbanks, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        # from IPython import embed; embed(using=False); os._exit(0)

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

        B, C, T, F_, I = x.shape
        O_ = self.out_channels
        out = torch.zeros((B, C, T, self.n_bands, O_), device=x.device)  # (b, c, t, f, o)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)
            
            _x = x[..., idxes, :] * melbanks[..., :, :, None] * masks[..., :, :, None]  # (b, c, t, s, w, i)
            _w = self.pre_w[:, idxes, :] * masks[None, :, :, None]  # (i, s, w, o)
            _b = self.pre_b[subbands, :]  # (s, o)
            _y = torch.einsum('bctswi,iswo->bctso', _x, _w) + _b  # (b, c, t, s, o)
            # from IPython import embed; embed(using=False); os._exit(0)
            out[:, :, :, subbands, :] = _y
            
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
        
        B, C, T, F, O_ = x.shape
        I = self.in_channels
        out = torch.zeros((B, C, T, self.n_fft // 2 + 1, I), device=x.device)

        for q in range(self.Q):
            subbands = getattr(self, f"sb_subbands_{q}")  # (s,)
            idxes = getattr(self, f"sb_idxes_{q}")  # (s, w)
            melbanks = getattr(self, f"sb_melbanks_{q}")  # (s, w)
            masks = getattr(self, f"sb_masks_{q}")  # (s, w)

            _x = x[:, :, :, subbands, :]  # (b, c, t, s, o)
            _w = self.post_w[:, :, idxes] * masks  # (o, i, s, w)
            _b = self.post_b[:, idxes] * masks  # (i, s, w)
            _b = rearrange(_b, 'i s w -> s w i')
            _y = torch.einsum('bctso,oisw->bctswi', _x, _w) + _b  # (b, c, t, s, w, i)
            _y = _y * melbanks[..., :, :, None] * masks[..., :, :, None]
            out.scatter_add_(dim=3, index=idxes.flatten()[None, None, None, :, None].repeat(B, C, T, 1, I), src=_y.flatten(3, 4))

        # Divide overlap add window
        out /= self.ola_window[..., :, None]

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