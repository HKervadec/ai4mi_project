import unittest

import numpy as np
import torch

from dataset_analysis.utils import overlap
from src.config import load_config
from src.metrics import dice_from_counts, epoch_metrics, legacy_dice_fg, slice_counts
from src.metrics_3d import volume_metrics
from utils import class2one_hot, dice_coef


def random_batch(seed: int, b: int = 6, k: int = 5, hw: int = 32):
    g = torch.Generator().manual_seed(seed)
    gt = torch.randint(0, k, (b, hw, hw), generator=g)
    gt[0] = 0  # a slice with every foreground class absent
    pred = torch.randint(0, k, (b, hw, hw), generator=g)
    pred[0] = 0  # ... and absent in the prediction too: joint-empty
    return pred, class2one_hot(gt, k)


class SliceMetricTests(unittest.TestCase):
    def test_dice_matches_dataset_analysis_convention(self):
        pred, gt = random_batch(0)
        dice = dice_from_counts(slice_counts(pred, gt))
        gt_cls = gt.argmax(1).numpy()
        for i in range(pred.shape[0]):
            for k in range(1, 5):
                expected = overlap(gt_cls[i] == k, pred[i].numpy() == k)["dice"]
                np.testing.assert_equal(dice[i, k], expected)  # NaN == NaN here
        self.assertTrue(np.isnan(dice[0, 1:]).all())

    def test_legacy_dice_matches_course_code(self):
        pred, gt = random_batch(1)
        expected = dice_coef(class2one_hot(pred, 5), gt)[:, 1:].mean().item()
        self.assertAlmostEqual(legacy_dice_fg(slice_counts(pred, gt)), expected, places=6)

    def test_patient_level_aggregation(self):
        cfg = load_config("configs/segthor_enet_ce.yaml")
        counts = np.zeros((3, 5, 3), dtype=np.int64)
        counts[0, 1] = [10, 20, 20]  # Patient_01 slice 0: esophagus
        counts[1, 1] = [0, 0, 10]    # Patient_01 slice 1: false positive only
        counts[2, 1] = [5, 5, 5]     # Patient_02: perfect
        out = epoch_metrics("val", counts, ["Patient_01_0000", "Patient_01_0001", "Patient_02_0000"], cfg)
        self.assertAlmostEqual(out["val_dice_esophagus"], (2 * 10 / 50 + 1.0) / 2)
        self.assertTrue(np.isnan(out["val_dice_heart"]))  # no GT, no prediction anywhere


class VolumeMetricTests(unittest.TestCase):
    def cube(self, offset=(0, 0, 0)):
        m = np.zeros((40, 40, 40), dtype=bool)
        x, y, z = (10 + o for o in offset)
        m[x:x + 12, y:y + 12, z:z + 12] = True
        return m

    def test_identical_and_empty(self):
        self.assertEqual(volume_metrics(self.cube(), self.cube(), (1, 1, 1)), {"dice": 1.0, "hd95": 0.0, "assd": 0.0})
        empty = np.zeros_like(self.cube())
        both = volume_metrics(empty, empty, (1, 1, 1))
        self.assertTrue(all(np.isnan(v) for v in both.values()))
        missed = volume_metrics(self.cube(), empty, (1, 1, 1))
        self.assertEqual(missed["dice"], 0.0)
        self.assertTrue(np.isnan(missed["hd95"]))

    def test_shift_uses_physical_spacing(self):
        a, b = self.cube(), self.cube((3, 0, 0))
        iso = volume_metrics(a, b, (1, 1, 1))
        aniso = volume_metrics(a, b, (2, 1, 1))
        self.assertAlmostEqual(iso["dice"], 9 / 12)
        self.assertGreater(iso["hd95"], 0)
        self.assertLessEqual(iso["hd95"], 3.0)  # no surface point is further than the shift
        self.assertGreater(aniso["hd95"], iso["hd95"])  # the shift is along the 2 mm axis
        self.assertLessEqual(aniso["hd95"], 6.0)
        swapped = volume_metrics(b, a, (1, 1, 1))
        for m in iso:
            self.assertAlmostEqual(iso[m], swapped[m])  # symmetric


if __name__ == "__main__":
    unittest.main()
