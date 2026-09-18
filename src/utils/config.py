import argparse
from pathlib import Path
from typing import Literal, Optional
import yaml

import dataclasses
from dataclasses import dataclass, field

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
    dest: Optional[Path] = None

    # The dataset to train on
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    model: ModelConfig = field(default_factory=ModelConfig)

    epochs: int = 20

    mode: Literal["partial", "full"] = "full"

    gpu: bool = True

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


def instantiate_dataclass(cls, data: dict):
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for key, value in data.items():
        # Ignore keys in yaml that don't exist in the dataclass
        if key not in field_types:
            continue

        expected_type = field_types[key]

        # If the expected field is a nested dataclass, instantiate it recursively
        if dataclasses.is_dataclass(expected_type) and isinstance(value, dict):
            kwargs[key] = instantiate_dataclass(expected_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)  # type: ignore


def _load_config_file():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="")
    args, remaining_argv = parser.parse_known_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as fp:
            return yaml.safe_load(fp) or {}, remaining_argv

    return {}, remaining_argv


def get_config() -> Config:
    yaml_dict, remaining_argv = _load_config_file()

    base_inst = Config()
    if yaml_dict:
        base_inst = instantiate_dataclass(Config, yaml_dict)

    config = tyro.cli(Config, default=base_inst, args=remaining_argv)
    if isinstance(config, dict):
        raise Exception("config was loaded as a dict")

    # Make sure a gpu is available if configured to use
    config.gpu = config.gpu and torch.cuda.is_available()

    return config
