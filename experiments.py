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

To keep comparisons fair, the *ground truth* is held constant across the
measurement experiments by slicing from the corrected source ``data/segthor_fixed``
(see fix_gt.py). The one exception is ``original``, which reproduces the literal
starting pipeline (uncorrected GT + per-volume min-max) so the pre-change result
is preserved end to end.
"""

from dataclasses import dataclass, field

from preprocessing import MEDIASTINAL


@dataclass(frozen=True)
class SliceConfig:
    """Options for slice_segthor.py (the 2D dataset build)."""
    source_dir: str = "data/segthor_fixed"              # corrected GT, held constant
    window: tuple[float, float] | None = MEDIASTINAL    # None -> legacy per-volume min-max
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
# Ablation ladder: each step changes exactly one thing vs the step above it.  #
# --------------------------------------------------------------------------- #

# 0) The literal original pipeline, preserved so the pre-change number stays
#    reproducible: uncorrected GT + per-volume min-max normalization.
register(Experiment(
    name="original",
    description="Literal original pipeline: uncorrected GT (data/segthor_part1) + per-volume min-max.",
    slice=SliceConfig(source_dir="data/segthor_part1", window=None),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))

# 1) Baseline for measuring *preprocessing*: corrected GT, still min-max. The
#    difference original -> baseline isolates the GT fix; baseline is the
#    reference the intensity/augmentation experiments are compared against.
register(Experiment(
    name="baseline",
    description="Corrected GT + per-volume min-max (isolates the GT fix; reference for preprocessing).",
    slice=SliceConfig(window=None),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))

# 2) HU windowing: corrected GT + fixed mediastinal window; everything else as
#    baseline, so baseline -> hu_window isolates the windowing effect.
register(Experiment(
    name="hu_window",
    description="Corrected GT + HU mediastinal windowing (isolates the windowing effect).",
    slice=SliceConfig(window=MEDIASTINAL),
    train=TrainConfig(mode="full", loss="ce", augment=False),
))
