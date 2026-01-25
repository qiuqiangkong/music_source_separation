import random
from torch import Tensor
import numpy as np
import librosa


class RandomResample:
    r"""Applies random resample to the audio. 

    NumPy is be faster than PyTorch when the source and target sample rates 
    have a small GCD.
    """

    def __init__(
        self, 
        sr: int, 
        min_ratio: float = 0.95,
        max_ratio: float = 1.05
    ):
        self.sr = sr
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, x: Tensor) -> Tensor:
        r"""Random gain.

        c: audio_channels
        l: audio_samples

        Args:
            x: (c, l)

        Output:
            out: (c, l)
        """
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        target_sr = round(self.sr * ratio)
        out = np.stack([librosa.resample(e, orig_sr=self.sr, target_sr=target_sr) for e in x], axis=0)
        # print("resample:", ratio)
        return out