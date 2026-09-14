"""src.evaluate.stitch must rebuild exactly what the course stitch.py produced for the baseline."""
import unittest

import nibabel as nib
import numpy as np
from PIL import Image

from src.config import REPO, load_config
from src.evaluate import group_by_patient, stitch

LEGACY_PRED = REPO / "results" / "segthor" / "ce" / "best_epoch" / "val"
LEGACY_VOL = REPO / "volumes" / "segthor" / "ce"


class StitchTests(unittest.TestCase):
    @unittest.skipUnless(LEGACY_PRED.is_dir() and LEGACY_VOL.is_dir(), "baseline outputs missing")
    def test_matches_legacy_stitch(self):
        cfg = load_config("configs/segthor_enet_ce.yaml")
        preds = {p.stem: np.asarray(Image.open(p)) // 63 for p in LEGACY_PRED.glob("*.png")}
        for patient, slices in group_by_patient(preds, cfg["data"]["patient_regex"]).items():
            ref = nib.load(REPO / cfg["data"]["source_pattern"].format(patient=patient))
            legacy = np.asarray(nib.load(LEGACY_VOL / f"{patient}.nii.gz").dataobj)
            np.testing.assert_array_equal(stitch(slices, ref, patient), legacy, err_msg=patient)

    def test_missing_slice_is_an_error(self):
        ref = nib.Nifti1Image(np.zeros((8, 8, 3), dtype=np.uint8), np.eye(4))
        with self.assertRaises(ValueError):
            stitch({0: np.zeros((4, 4), np.uint8), 2: np.zeros((4, 4), np.uint8)}, ref, "Patient_99")


if __name__ == "__main__":
    unittest.main()
