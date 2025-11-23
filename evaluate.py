from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile

from mss.utils import parse_yaml
from train import get_model, validate


def evaluate(args) -> None:
    r"""Evaluate on the test set of MUSDB18HQ."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_yaml)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Compute SDRs
    sdr = validate(
        configs=configs,
        model=model,
        split="test",
        audios_num=None,
        hop_ratio=4
    )

    print("====== Overall metrics ====== ")
    print(f"Median SDR: {sdr:.2f} dB")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)

    args = parser.parse_args()

    evaluate(args)