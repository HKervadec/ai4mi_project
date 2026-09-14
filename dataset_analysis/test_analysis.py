"""Small numerical tests for conventions that could bias the scientific results."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from analyze_dataset import original_stats
from analyze_baseline import bin_index, class_summary
from utils import extent, load_png, normalized_z, overlap


class MeasurementTests(unittest.TestCase):
    def test_overlap_empty_false_positive_and_miss(self):
        zero = np.zeros((2, 3), dtype=bool)
        one = zero.copy()
        one[0, 1] = True
        self.assertTrue(np.isnan(overlap(zero, zero)["dice"]))
        self.assertTrue(overlap(zero, zero)["joint_empty"])
        self.assertEqual(overlap(zero, one)["fp_pixels"], 1)
        self.assertEqual(overlap(one, zero)["fn_pixels"], 1)
        self.assertEqual(overlap(one, one)["dice"], 1)
        other = one.copy()
        other[1, 1] = True
        self.assertAlmostEqual(overlap(one, other)["dice"], 2 / 3)

    def test_extent_including_absence_gaps_and_single_slice(self):
        self.assertIsNone(extent([], 0)["organ_relative_z"])
        self.assertIsNone(extent([1, 3], 2)["organ_relative_z"])
        self.assertEqual(extent([4], 4)["organ_relative_z"], .5)
        self.assertEqual(extent([1, 3, 5], 3)["distance_from_first"], 2)
        self.assertEqual(normalized_z(0, 1), 0)
        self.assertEqual(normalized_z(9, 10), 1)

    def test_original_volume_uses_physical_grid(self):
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        data[1:3, 1:3, 0] = 1
        data[1:3, 1:3, 2] = 1
        nii = nib.Nifti1Image(data, np.diag([2, 3, 4, 1]))
        inv, rows = original_stats("Patient_99", "train", nii, data)
        self.assertEqual(rows[0]["voxel_count"], 8)
        self.assertAlmostEqual(rows[0]["volume_mm3"], 192)
        self.assertEqual(rows[0]["occupied_slice_count"], 2)
        self.assertEqual(rows[0]["bbox_extent_z_mm"], 12)
        self.assertEqual(rows[0]["normalized_last_z"], 1)
        self.assertFalse(rows[1]["present"])
        self.assertEqual(inv["foreground_voxels"], 8)

    def test_strict_png_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "mask.png"
            Image.fromarray(np.array([[0, 63, 126, 189]], dtype=np.uint8)).save(p)
            np.testing.assert_array_equal(load_png(p), [[0, 1, 2, 3]])
            Image.fromarray(np.array([[252]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                load_png(p)
            self.assertEqual(load_png(p, prediction=True)[0, 0], 4)
            Image.fromarray(np.array([[64]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                load_png(p, prediction=True)

    def test_summary_excludes_joint_empty_and_fp_only(self):
        zero, one = np.zeros((1, 1), bool), np.ones((1, 1), bool)
        rows = [{"class_id": k, "patient_id": "Patient_99", **overlap(g, p)}
                for k in (1, 2, 3) for g, p in ((zero, zero), (zero, one), (one, zero))]
        for r in class_summary(rows):
            self.assertEqual(r["positive_slice_dice_mean"], 0)
            self.assertEqual(r["joint_empty_slices"], 1)
            self.assertEqual(r["fp_only_slices"], 1)

    def test_bin_endpoints(self):
        edges = np.array([0, .2, .8, 1.])
        self.assertEqual(bin_index(0, edges), 0)
        self.assertEqual(bin_index(.2, edges), 1)
        self.assertEqual(bin_index(.8, edges), 2)
        self.assertEqual(bin_index(1, edges), 2)


if __name__ == "__main__":
    unittest.main()
