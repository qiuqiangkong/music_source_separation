from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile

from mss.utils import parse_yaml, separate_overlap_add
from train import get_model


def inference(args) -> None:
    r"""Separate an audio.

    c: channels_num
    L: audio_samples
    """

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    output_path = args.output_path
    batch_size = args.batch_size
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    segment_samples = round(segment_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Load audio
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)  # (c, L) or (L,)
    
    # Copy channels to stereo
    if audio.ndim == 1:
        audio = np.array([audio, audio])  # (c, L)
    
    # Foward
    output = separate_overlap_add(
        model=model, 
        audio=audio, 
        segment_samples=segment_samples, 
        hop_length=segment_samples // 4,
        batch_size=batch_size
    )  # (c, L)

    # Write out audio
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    soundfile.write(file=output_path, data=output.T, samplerate=sr)
    print("Write out to {}".format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    inference(args)