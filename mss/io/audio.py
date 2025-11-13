from __future__ import annotations

import librosa
import numpy as np
import torch
import torchaudio


def load(
    path: str,  # Audio path
    sr: int,  # Sample rate
    offset: float = 0.,  # Load start time (s)
    duration: float | None = None,  # Load duration (s)
    mono: bool = False
) -> np.ndarray:
    r"""Load audio.

    c: audio_channels
    L: audio_samples

    Returns:
       audio: (channels, audio_samples) 

    Examples:
        >>> audio = load_audio(path="xx/yy.wav", sr=16000)
    """
    
    # Load audio. librosa.load is faster than torchaudio.load
    audio, orig_sr = librosa.load(
        path, 
        sr=sr, 
        mono=mono, 
        offset=offset, 
        duration=duration
    )  # (c, L) or (L,)
    
    if audio.ndim == 1:
        audio = audio[None, :]  # (c, L)

    # Resample. torchaudio's resample faster than librosa's resample
    audio = torchaudio.functional.resample(
        waveform=torch.Tensor(audio), 
        orig_freq=orig_sr, 
        new_freq=sr
    ).numpy()  # (c, L)

    if duration:
        # Fix length to address 1) after resampling. 2) audio is shorter than duration
        audio = librosa.util.fix_length(
            data=audio, 
            size=round(duration * sr), 
            axis=-1
        )  # (c, L)

    if mono:
        audio = np.mean(audio, axis=0, keepdims=True)  # (1, L)
    
    return audio