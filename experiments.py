#!/usr/bin/env python3

"""
Experiment registry.

One place to define named technique configurations, so that we can:
  * reproduce the *original* pipeline exactly (nothing is thrown away);
  * run the baseline and each improvement with a single ``--experiment NAME`` flag;
  * A/B them fairly and later stack more techniques.

Design principle: **never delete an old code path -- branch on config instead.**
An experiment is a bundle of choices for the two pipeline stages:
  * ``SliceConfig`` -> options that change the built 2D dataset (slice_segthor.py)
  * ``TrainConfig`` -> options that change training (main.py)

Adding a new technique later = add a field to the relevant *Config, wire it in the
one place that reads it, and register a new Experiment. The existing experiments
keep producing the same results, so every comparison stays valid.

Ground truth is a first-class axis: each GT-fix method produces its own corrected
source under ``data/gt/<method>`` (``watershed`` from fix_gt.py, ``watershed_refined``
from fix_segthor_gt.py), and an experiment selects one via ``SliceConfig.source_dir``.
Comparisons that differ only in ``source_dir`` isolate the fix method; those that
differ only in ``window`` isolate the intensity preprocessing. ``original`` is the
exception -- it slices the uncorrected ``data/segthor_part1`` with min-max, so the
literal starting pipeline is preserved end to end.
"""

from dataclasses import dataclass, field

from preprocessing import MEDIASTINAL


@dataclass(frozen=True)
class SliceConfig:
    """Options for slice_segthor.py (the 2D dataset build)."""
    source_dir: str = "data/gt/watershed"               # which GT-fix variant to slice from
    # Default None = legacy per-volume min-max, so a bare `slice_segthor.py` /
    # `make data/SEGTHOR` reproduces the original master behavior. Experiments opt
    # into windowing by setting this explicitly (see watershed_window etc.).
    window: tuple[float, float] | None = None
    shape: tuple[int, int] = (256, 256)
    retains: int = 5                                     # patients held out for validation


@dataclass(frozen=True)
class TrainConfig:
    """Options for main.py (training)."""
    mode: str = "full"      # "full" | "partial"
    loss: str = "ce"        # extension point: "dice", "focal", ... (see main.build_loss)
    augment: bool = False   # spatial/intensity augmentation (wired via SliceDataset)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    base_dataset: str = "SEGTHOR"                            # key into datasets_params (K / net / batch)
    slice: SliceConfig = field(default_factory=SliceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def data_dir(self) -> str:
        """Where this experiment's sliced 2D dataset lives / is built.

        All experiment datasets are grouped under data/experiments/<name> so the
        top-level data/ folder is not cluttered with one SEGTHOR_* folder per run.
        (Nested under data/ -- not a top-level experiments/ folder -- to avoid
        shadowing the experiments.py module on `import experiments`.)
        """
        return f"data/experiments/{self.name}"


EXPERIMENTS: dict[str, Experiment] = {}


def register(exp: Experiment) -> Experiment:
    assert exp.name not in EXPERIMENTS, f"duplicate experiment name: {exp.name}"
    EXPERIMENTS[exp.name] = exp
    return exp


def get(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{name}'. Known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]


# --------------------------------------------------------------------------- #
# Registry                                                                     #
#                                                                             #
# Experiments vary along two axes and are named <gtfix>_<intensity> so the     #
# data/experiments/ listing reads as the ablation itself:                      #
#                                                                             #
#   GT fix       : original (corrupted) | watershed (fix_gt.py) |              #
#                  refined (fix_segthor_gt.py)  -> lives in data/gt/<method>   #
#   intensity    : minmax (per-volume min-max) | window (HU mediastinal)       #
#                                                                             #
#   original            -> watershed_minmax : the GT fix                       #
#   watershed_minmax    -> watershed_window : the HU windowing                 #
#   watershed_window    -> refined_window   : the fix method (basic vs refined)#
# --------------------------------------------------------------------------- #

# The literal original pipeline, kept so the pre-change number stays
# reproducible: uncorrected GT + per-volume min-max.
register(Experiment(
    name="original",
    description="Original pipeline: uncorrected GT (data/segthor_part1) + per-volume min-max.",
    slice=SliceConfig(source_dir="data/segthor_part1", window=None),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))

# Watershed GT fix (fix_gt.py) + min-max. vs original -> isolates the GT fix;
# this is the reference the preprocessing experiments are compared against.
register(Experiment(
    name="watershed_minmax",
    description="Watershed GT fix (fix_gt.py) + per-volume min-max. Reference for preprocessing.",
    slice=SliceConfig(source_dir="data/gt/watershed", window=None),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))

# Watershed GT fix + HU mediastinal window. vs watershed_minmax -> isolates
# the windowing effect.
register(Experiment(
    name="watershed_window",
    description="Watershed GT fix (fix_gt.py) + HU mediastinal windowing.",
    slice=SliceConfig(source_dir="data/gt/watershed", window=MEDIASTINAL),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))

# Refined GT fix (fix_segthor_gt.py) + HU window. vs watershed_window ->
# isolates the fix method (basic vs refined aorta/esophagus split).
register(Experiment(
    name="refined_window",
    description="Refined GT fix (fix_segthor_gt.py) + HU mediastinal windowing.",
    slice=SliceConfig(source_dir="data/gt/watershed_refined", window=MEDIASTINAL),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))
