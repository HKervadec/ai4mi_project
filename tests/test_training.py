"""End-to-end checks on real data, CPU, a few slices. Needs data/SEGTHOR; run inside a Slurm job."""
import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import src.engine
from src import train
from src.checkpoint import seed_everything
from src.config import REPO
from src.run import read_json

CFG = "configs/segthor_enet_ce.yaml"
HAS_DATA = (REPO / "data" / "SEGTHOR" / "train").is_dir()


def run_args(root: str, *extra: str) -> list[str]:
    return ["--config", CFG, "--set", "device=cpu", "wandb.mode=disabled", "train.debug_samples=10",
            f"paths.run_root={root}/runs", f"paths.metrics_dir={root}/metrics", *extra]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(f)]


@unittest.skipUnless(HAS_DATA, "data/SEGTHOR missing")
class TrainingTests(unittest.TestCase):
    def test_parity_with_legacy_main(self):
        """Seeded, the new loop reproduces main.py's losses and Dice exactly."""
        import main as legacy
        with tempfile.TemporaryDirectory() as tmp:
            seed_everything(0)
            legacy.runTraining(Namespace(epochs=2, dataset="SEGTHOR", mode="full", dest=Path(tmp) / "legacy",
                                         gpu=False, debug=True))
            train.main(run_args(tmp, "train.epochs=2", "data.num_workers=5"))
            rows = read_rows(Path(tmp) / "runs" / "segthor_enet_ce" / "seed0" / "epochs.csv")
            for name, column, reduce in (("loss_tra", "train_loss", lambda a: a.mean(1)),
                                         ("loss_val", "val_loss", lambda a: a.mean(1)),
                                         ("dice_val", "val_dice_legacy_fg", lambda a: a[:, :, 1:].mean((1, 2)))):
                expected = reduce(np.load(Path(tmp) / "legacy" / f"{name}.npy"))
                np.testing.assert_allclose([float(r[column]) for r in rows], expected, rtol=1e-5, err_msg=name)

    def test_interrupted_run_resumes_to_identical_result(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            args = ("train.epochs=3", "data.num_workers=0")
            train.main(run_args(tmp_a, *args))

            real_run_epoch, calls = src.engine.run_epoch, []

            def crash_in_second_epoch(split, *a, **kw):
                calls.append(split)
                if calls.count("train") == 2:
                    raise KeyboardInterrupt  # stands in for scancel / node failure
                return real_run_epoch(split, *a, **kw)

            with mock.patch("src.engine.run_epoch", crash_in_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train.main(run_args(tmp_b, *args))
            run_b = Path(tmp_b) / "runs" / "segthor_enet_ce" / "seed0"
            self.assertEqual(read_json(run_b / "manifest.json")["status"], "failed")

            train.main(run_args(tmp_b, *args))  # same command again: resumes automatically
            self.assertEqual(len(read_json(run_b / "manifest.json")["resumes"]), 1)
            run_a = Path(tmp_a) / "runs" / "segthor_enet_ce" / "seed0"
            self.assertEqual(read_rows(run_a / "epochs.csv"), read_rows(run_b / "epochs.csv"))
            self.assertTrue((Path(tmp_b) / "metrics" / "segthor_enet_ce" / "seed0" / "summary.json").exists())

            with mock.patch("src.train.fit") as fit:  # finished: a third call does nothing
                train.main(run_args(tmp_b, *args))
                fit.assert_not_called()

    def test_changed_config_refuses_existing_run(self):
        from src.run import RunConflict
        with tempfile.TemporaryDirectory() as tmp:
            train.main(run_args(tmp, "train.epochs=1", "data.num_workers=0"))
            with self.assertRaises(RunConflict):
                train.main(run_args(tmp, "train.epochs=1", "data.num_workers=0", "optim.kwargs.lr=0.01"))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
