import random
from torch import Tensor
import numpy as np
import librosa


class RandomPitch:
    r"""Applies pitch shifting to the audio."""

    def __init__(
        self, 
        sr: float,
        min_step: float = -1.0,
        max_step: float = 1.0
    ):
        self.sr = sr
        self.min_step = min_step
        self.max_step = max_step

    def __call__(self, x: Tensor) -> Tensor:
        r"""Random pitch.

        c: audio_channels
        l: audio_samples

        Args:
            x: (c, l)

        Output:
            out: (c, l)
        """
        step = random.uniform(self.min_step, self.max_step)
        out = np.stack([librosa.effects.pitch_shift(e, sr=self.sr, n_steps=step) for e in x], axis=0)
        
        return out