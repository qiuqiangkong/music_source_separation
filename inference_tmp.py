from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile

from music_source_separation.utils import parse_yaml
# from train import get_model, separate
from tmp import get_model, separate


def inference(args) -> None:
    r"""Separate an audio."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    output_path = args.output_path
    device = "cuda"
    batch_size = 4

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    clip_samples = round(clip_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Load audio
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)  # shape: (c, l)
    
    # Copy channels to stereo
    if audio.ndim == 1:
        audio = np.array([audio, audio])  # shape: (c, l)
    
    # Foward
    output = separate(
        model=model, 
        audio=audio, 
        clip_samples=clip_samples, 
        batch_size=batch_size
    )  # shape: (c, l)

    # Write out to MIDI
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    soundfile.write(file=output_path, data=output.T, samplerate=sr)
    print("Write out to {}".format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)