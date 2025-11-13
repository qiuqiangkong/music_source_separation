from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from mss.utils import (parse_yaml, requires_grad, update_ema, LinearWarmUp, 
    separate_overlap_add, calculate_sdr)


def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Loss function
    loss_fn = get_loss_fn(configs)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Logger
    if wandb_log:
        wandb.init(project="music_source_separation", name=f"{config_name}")

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Training ------
        # 1.1 Data
        target = data["target"].to(device)
        mixture = data["mixture"].to(device)

        # 1.1 Forward
        model.train()
        output = model(mixture)

        # 1.2 Loss
        loss = loss_fn(output=output, target=target)
        
        # 1.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        scheduler.step()
        update_ema(ema, model, decay=0.999)

        if step % 100 == 0:
            print(loss)

        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_sdr = validate(
                configs=configs,
                model=ema,
                split="train",
                valid_audios=10,
                eval_mode="default",
            )

            test_sdr = validate(
                configs=configs,
                model=ema,
                split="test",
                valid_audios=10,
                eval_mode="default",
            )

            if wandb_log:
                wandb.log(
                    data={
                        "train_sdr": train_sdr, 
                        "test_sdr": test_sdr,
                    },
                    step=step
                )

            print("====== Overall metrics ====== ")
            print(f"Train SDR fast: {train_sdr:.2f}")
            print(f"Test SDR fast: {train_sdr:.2f}")
        
        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(ema.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
        

def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from audidata.io.crops import RandomCrop

    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
    ds = f"{split}_datasets"

    for name in configs[ds].keys():
    
        if name == "MUSDB18HQ":
            from mss.datasets.musdb18hq import MUSDB18HQ
            return MUSDB18HQ(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=segment_duration, end_pad=0.),
                target_stems=[target_stem],
                time_align="group",
                mixture_transform=None,
                group_transform=None,
                stem_transform=None
            )
        else:
            raise ValueError(name)
            

def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RandomSongSampler":
        from mss.samplers.random_song_sampler import RandomSongSampler
        return RandomSongSampler(dataset)

    else:
        raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "BSRoformer":

        from mss.models.bsroformer import BSRoformer
        model = BSRoformer(**configs["model"])

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_loss_fn(configs: dict) -> callable:
    r"""Get loss function."""

    loss_type = configs["train"]["loss"]

    if loss_type == "l1":
        from mss.losses.l1 import l1
        return l1

    else:
        raise ValueError(loss_type)


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def validate(
    configs: dict,
    model: nn.Module,
    split: str,
    valid_audios: None | int = None,
    eval_mode: str = "default"
) -> float:
    r"""Validate the model on part of data.

    c: channels_num
    L: audio_samples
    """

    root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
    batch_size = configs["train"]["batch_size_per_device"]
    segment_samples = round(segment_duration * sr)

    # Paths
    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    if valid_audios:
        # Evaluate only part of data
        skip_n = max(1, len(audio_names) // valid_audios)
    else:
        skip_n = 1
    
    stems = ["vocals", "bass", "drums", "other"]
    sdrs = []

    for idx in range(0, len(audio_names), skip_n):

        # Get data
        audio_name = audio_names[idx]    
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, f"{stem}.wav")
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # (c, L)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # (c, L)

        # Foward
        output = separate_overlap_add(
            model=model, 
            audio=data["mixture"], 
            segment_samples=segment_samples,
            hop_length=segment_samples // 4,
            batch_size=batch_size
        )  # (c, L)
        
        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode=eval_mode)
        print("{}/{}, {}: {:.3f}".format(idx, len(audio_names), audio_name, sdr))

        sdrs.append(sdr)

    return np.nanmedian(sdrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)