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
from music_source_separation.utils import (LinearWarmUp, calculate_sdr,
    parse_yaml, requires_grad, update_ema)


def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
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
        wandb.init(project="music_source_separation", name="{}".format(config_name))

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
        update_ema(ema, model, decay=0.999)

        # 1.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)

        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_sdr = validate(
                configs=configs,
                model=model,
                split="train",
                valid_audios=10
            )

            test_sdr = validate(
                configs=configs,
                model=model,
                split="test",
                valid_audios=None
            )

            test_sdr_ema = validate(
                configs=configs,
                model=ema,
                split="test",
                valid_audios=None
            )

            if wandb_log:
                wandb.log(
                    data={
                        "train_sdr": train_sdr, 
                        "test_sdr": test_sdr, 
                        "test_sdr_ema": test_sdr_ema
                    },
                    step=step
                )

            print("====== Overall metrics ====== ")
            print("Train SDR: {}".format(train_sdr))
            print("Test SDR: {}".format(test_sdr))
            print("Test SDR ema: {}".format(test_sdr_ema))
        
        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

            ckpt_path = Path(ckpts_dir, "step={}_ema.pth".format(step))
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
    clip_duration = configs["clip_duration"]
    target_stem = configs["target_stem"]
    datasets_split = "{}_datasets".format(split)

    datasets = []
    
    for name in configs[datasets_split].keys():
    
        if name == "MUSDB18HQ":

            from audidata.datasets import MUSDB18HQ

            dataset = MUSDB18HQ(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=0.),
                target_stems=[target_stem],
                time_align="group",
                mixture_transform=None,
                group_transform=None,
                stem_transform=None
            )
            datasets.append(dataset)

        else:
            raise ValueError(name)

    if len(datasets) == 1:
        return datasets[0]

    else:
        raise ValueError("Do not support multiple datasets in this file.")


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "InfiniteSampler":
        from audidata.samplers import InfiniteSampler
        return InfiniteSampler(dataset)

    elif name == "MUSDB18HQ_RandomSongSampler":
        from audidata.samplers import MUSDB18HQ_RandomSongSampler
        return MUSDB18HQ_RandomSongSampler(dataset)

    else:
        raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "UNet":

        from music_source_separation.models.unet import UNet, UNetConfig

        config = UNetConfig(
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
        )
        model = UNet(config)

    elif name == "BSRoformer":

        import ast
        from music_source_separation.models.bsroformer import BSRoformer, BSRoformerConfig

        config = BSRoformerConfig(
            sr=configs["sample_rate"],
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
            mel_bins=configs["model"]["mel_bins"],
            mel_channels=configs["model"]["mel_channels"],
            patch_size=ast.literal_eval(configs["model"]["patch_size"]),
            n_layer=configs["model"]["n_layer"],
            n_head=configs["model"]["n_head"],
            n_embd=configs["model"]["n_embd"],
        )
        model = BSRoformer(config)        

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
        from music_source_separation.losses import l1_loss
        return l1_loss

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
    valid_audios: None | int = None
) -> float:
    r"""Validate the model on part of data.

    c: channels_num
    l: audio_samples
    """

    root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    target_stem = configs["target_stem"]
    batch_size = configs["train"]["batch_size_per_device"]
    clip_samples = round(clip_duration * sr)

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

        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode="fast")
        print("{}/{}, {}: {:.3f}".format(idx, len(audio_names), audio_name, sdr))

        sdrs.append(sdr)

    return np.nanmedian(sdrs)


def separate(
    model: nn.Module, 
    audio: torch.Tensor, 
    clip_samples: int, 
    batch_size: int
):
    r"""Split audio into clips. Separate each clip. Concatenate the results.

    b: batch_size
    c: channels_num
    t: cilp_samples
    n: clips_num

    Args:
        model: nn.Module
        audio: (c, audio_samples)
        clip_samples: int
        batch_size: int

    Returns:
        output: (c, audio_samples)
    """

    device = next(model.parameters()).device
    
    audio_samples = audio.shape[1]
    full_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
    audio = librosa.util.fix_length(data=audio, size=full_samples, axis=-1)
    # shape: (c, n*t)

    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=clip_samples
    )  # shape: (c, t, n)

    clips = clips.transpose(2, 0, 1)  # shape: (n, c, t)
    clips = torch.Tensor(clips.copy()).to(device)
    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size])
        # shape: (b, c, t)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_clips)
            batch_output = batch_output.cpu().numpy()  # shape: (b, c, t)

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)
    # shape: (n, c, t)

    channels_num = outputs.shape[1]
    outputs = outputs.transpose(1, 0, 2).reshape(channels_num, -1)
    # shape: (c, n*t)

    outputs = outputs[:, 0 : audio_samples]
    # shape: (c, audio_samples)

    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)