from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile

from music_source_separation.utils import calculate_sdr, parse_yaml
from train import get_model, separate


def evaluate(args) -> None:
    r"""Evaluate on the test set of MUSDB18HQ."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    results_dir = args.results_dir
    device = "cuda"
    split = "test"
    batch_size = 4

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    target_stem = configs["target_stem"]
    clip_samples = round(clip_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Data paths
    root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    # Data buffer
    stems = ["vocals", "bass", "drums", "other"]
    sdrs = []
    fast_sdrs = []

    for idx, audio_name in enumerate(audio_names):
    
        # Get data
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, "{}.wav".format(stem))
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # shape: (c, l)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # shape: (c, l)

        # Foward
        output = separate(
            model=model, 
            audio=data["mixture"], 
            clip_samples=clip_samples, 
            batch_size=batch_size
        )  # shape: (c, l)

        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode="default")
        print("{}/{}, {}: {}".format(idx, len(audio_names), audio_name, sdr))
        sdrs.append(sdr)
        
        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode="fast")
        print("Fast SDR: {}".format(sdr))
        fast_sdrs.append(sdr)

        if results_dir:
            
            # Write out separated audio
            output_path = Path(results_dir, audio_name, f"{stem}.wav")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            soundfile.write(file=output_path, data=output.T, samplerate=sr)
            print("Write out to {}".format(output_path))

    print("====== Overall metrics ====== ")
    print("SDR: {}".format(np.nanmedian(sdrs)))
    print("Fast SDR: {}".format(np.nanmedian(fast_sdrs)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)

    args = parser.parse_args()

    evaluate(args)