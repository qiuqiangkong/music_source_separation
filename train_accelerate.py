from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
# import trackio as wandb
from mss.utils import parse_yaml, requires_grad, update_ema
from train import (get_dataset, get_loss_fn, get_model,
                   get_optimizer_and_scheduler, get_sampler, validate)


def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    precision = configs["train"]["precision"]
    valid_num = configs["validate"]["audios_num"]

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

    # Prepare for acceleration
    print(f"Mixed precision: {precision}")
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        mixed_precision=precision, 
        kwargs_handlers=[process_group_kwargs]
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    ema.to(accelerator.device)

    # Logger
    if wandb_log and accelerator.is_main_process:
        wandb.init(project="mss", name=f"{config_name}")

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Training ------
        # 1.1 Data
        target = data["target"]
        mixture = data["mixture"]

        # 1.1 Forward
        model.train()
        output = model(mixture)

        # 1.2 Loss
        loss = loss_fn(output=output, target=target)
        
        # 1.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        scheduler.step()
        update_ema(ema, model, decay=0.999)

        if step % 100 == 0:
            print(loss)

        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0 and accelerator.is_main_process:

            train_sdr = validate(
                configs=configs,
                model=accelerator.unwrap_model(ema),
                split="train",
                audios_num=valid_num,
            )

            test_sdr = validate(
                configs=configs,
                model=accelerator.unwrap_model(ema),
                split="test",
                audios_num=valid_num,
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
            print(f"Test SDR fast: {test_sdr:.2f}")
        
        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0 and accelerator.is_main_process:
            
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(accelerator.unwrap_model(ema).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)