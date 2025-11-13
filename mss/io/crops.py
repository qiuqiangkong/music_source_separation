from __future__ import annotations

import random


class RandomCrop:
    r"""Prepare start time and duration of to crop from random time.
    """

    def __init__(
        self, 
        clip_duration: float, 
        end_pad: float = 0  # Pad silent at the end (s)
    ):
        self.clip_duration = clip_duration
        self.end_pad = end_pad

    def __call__(self, audio_duration: float) -> tuple[float, float]:

        padded_duration = audio_duration + self.end_pad

        if self.clip_duration <= padded_duration:
            start_time = random.uniform(0., padded_duration - self.clip_duration)

        else:
            start_time = 0

        return start_time, self.clip_duration