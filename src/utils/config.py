from os import cpu_count
from pathlib import Path
from typing import Literal

from dataclasses import dataclass

from torch import torch
import tyro


@dataclass
class DatasetConfig:
    name: Literal["TOY2", "SEGTHOR"] = "SEGTHOR"

    num_classes: int = 5

    seed: int = 0

    shape: tuple[int, int] = (256, 256)
    retains: int = 5
    fold: int = 0


@dataclass
class ModelConfig:
    name: Literal["ShallowNet", "ENet"] = "ENet"

    kernels: int = 8
    factor: int = 2


@dataclass
class Config:
    # Destination directory to save the results (predictions and weights).
    dest: Path

    # The dataset to train on
    dataset: DatasetConfig

    model: ModelConfig

    epochs: int = 20

    mode: Literal["partial", "full"] = "full"

    gpu: bool = False

    num_workers: int = 5

    # Keep only a fraction (10 samples) of the datasets, to test the logics around epochs and logging easily.
    debug: bool = False

    wandb_watch: bool = False

    seed: int = 42

    lr: float = 0.0005
    weight_decay: float = 0
    batch_size: int = 8
    betas: tuple[float, float] = (0.9, 0.999)
    dropout: float = 0.01
    temperature: float = 1


def get_config() -> Config:
    config = tyro.cli(Config)

    # Make sure a gpu is available if configured to use
    config.gpu = config.gpu and torch.cuda.is_available()

    return config
