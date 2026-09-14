from pathlib import Path
from typing import Literal

from dataclasses import dataclass

import tyro


@dataclass
class Args:
    # Destination directory to save the results (predictions and weights).
    dest: Path

    # The dataset to train on
    dataset: str = "TOY2"

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
    batch_size: int = 256
    betas: tuple[float, float] = (0.9, 0.999)
    dropout: float = 0.01
    temperature: float = 1


def get_args() -> Args:
    return tyro.cli(Args)
