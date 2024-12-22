import argparse
import os
import random
from pathlib import Path
from typing import Union
from datetime import timedelta

import librosa
import museval
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

import wandb

wandb.require("core")

from audidata.datasets import MUSDB18HQ
from audidata.io import RandomCrop
from audidata.samplers import InfiniteSampler, MUSDB18HQ_RandomSongSampler
from train import InfiniteSampler, get_model, l1_loss, warmup_lambda
from train2 import validate
from utils import update_ema, requires_grad


def train(args):

    # Arguments
    model_name = args.model_name
    clip_duration = args.clip_duration
    batch_size = args.batch_size
    lr = float(args.lr)

    # Default parameters
    sr = 44100 
    mono = False
    num_workers = 16
    pin_memory = True
    use_scheduler = True
    test_step_frequency = 5000
    save_step_frequency = 10000
    evaluate_num = 10
    training_steps = 1000000
    wandb_log = True
    device = "cuda"
    target_stems = ["vocals"]
    sampler_type = "full_remix"

    filename = Path(__file__).stem

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    root = "/datasets/musdb18hq"

    # Training dataset
    train_dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=clip_duration, end_pad=0.),
        target_stems=target_stems,
        time_align="group",
        mixture_transform=None,
        group_transform=None,
        stem_transform=None
    )
    stems = train_dataset.stems

    # Samplers
    if sampler_type == "infinite":
        train_sampler = InfiniteSampler(train_dataset)
    elif sampler_type == "full_remix":
        train_sampler = MUSDB18HQ_RandomSongSampler(train_dataset)
    else:
        raise ValueError(sampler_type)

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name)

    # EMA
    ema = deepcopy(model)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lambda)

    # Prepare for multiprocessing
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[process_group_kwargs])

    # Logger
    wandb_log = accelerator.is_local_main_process and wandb_log
    if wandb_log:
        config = vars(args) | {
            "filename": filename, 
            "devices_num": torch.cuda.device_count()
        }
        wandb.init(
            project="mini_source_separation2", 
            config=config, 
            name="{}, sampler={}".format(model_name, sampler_type),
            magic=True
        )

    model, ema, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, ema, optimizer, train_dataloader, scheduler)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        mixture = data["mixture"]
        target = data["vocals"]

        # Forward
        model.train()
        output = model(mixture=mixture) 
        
        # Calculate loss
        loss = l1_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        parallel = (accelerator.num_processes > 1)

        update_ema(ema_model=ema, model=accelerator.unwrap_model(model))

        # Learning rate scheduler (optional)
        if use_scheduler:
            scheduler.step()
        
        # Evaluate
        if step % test_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                if accelerator.num_processes == 1:
                    val_model = model
                else:
                    val_model = model.module
                
                sdrs = {}

                for split in ["train", "test"]:
            
                    sdr = validate(
                        root=root, 
                        split=split, 
                        sr=sr,
                        clip_duration=clip_duration,
                        stems=stems, 
                        target_stems=target_stems,
                        batch_size=batch_size,
                        model=model,
                        evaluate_num=evaluate_num,
                    )
                    sdrs[split] = sdr

                sdr = validate(
                    root=root, 
                    split="test", 
                    sr=sr,
                    clip_duration=clip_duration,
                    stems=stems, 
                    target_stems=target_stems,
                    batch_size=batch_size,
                    model=ema,
                    evaluate_num=evaluate_num,
                )
                sdrs["test_ema"] = sdr

                print("--- step: {} ---".format(step))
                print("Evaluate on {} songs.".format(evaluate_num))
                print("Loss: {:.3f}".format(loss))
                print("Train SDR: {:.3f}".format(sdrs["train"]))
                print("Test SDR: {:.3f}".format(sdrs["test"]))
                print("Test SDR ema: {:.3f}".format(sdrs["test_ema"]))

                if wandb_log:
                    wandb.log(
                        data={
                            "train_sdr": sdrs["train"],
                            "test_sdr": sdrs["test"],
                            "test_sdr_ema": sdrs["test_ema"],
                            "loss": loss.item(),
                        },
                        step=step
                    )
        
        # Save model.
        if step % save_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_ema = accelerator.unwrap_model(ema)

                checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(unwrapped_model.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest_ema.pth")
                torch.save(unwrapped_ema.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    parser.add_argument('--clip_duration', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', default=0.001)
    args = parser.parse_args()

    train(args)