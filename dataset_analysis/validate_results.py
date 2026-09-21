"""Audit generated table identities, physical quantities, metric bounds and input immutability."""
import json
import math
from pathlib import Path

from utils import CLASSES, parser, paths, read_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    args = parser(__doc__).parse_args()
    _, _, _, output = paths(args)
    tables = output / "tables"
    inventory = read_csv(tables / "patient_inventory.csv")
    organs = read_csv(tables / "patient_class_stats.csv")
    slices = read_csv(tables / "slice_class_stats.csv")
    baseline = read_csv(tables / "baseline_slice_metrics.csv")
    patients = read_csv(tables / "baseline_patient_metrics.csv")
    ids = {r["patient_id"] for r in inventory}
    inv = {r["patient_id"]: r for r in inventory}
    require(len(ids) == len(inventory), "Duplicate patients")
    n_classes = len(CLASSES)
    class_ids = {str(c) for c in CLASSES}
    require(len(organs) == n_classes * len(ids), "Unexpected patient-class count")
    require(len(slices) == n_classes * sum(int(r["num_slices"]) for r in inventory), "Unexpected slice-class count")
    require(len(baseline) == n_classes * sum(int(r["num_slices"]) for r in inventory if r["split"] == "val"),
            "Unexpected baseline count")
    for rows, keys in ((organs, ("patient_id", "class_id")),
                       (slices, ("patient_id", "slice_index", "class_id")),
                       (baseline, ("patient_id", "slice_index", "class_id")),
                       (patients, ("patient_id", "class_id"))):
        require(len({tuple(r[k] for k in keys) for r in rows}) == len(rows), "Duplicate table keys")
        require(all(r["class_id"] in class_ids for r in rows), "Unexpected class in analysis")
    for r in organs:
        i = inv[r["patient_id"]]
        expected = int(r["voxel_count"]) * float(i["voxel_volume_mm3"])
        require(math.isclose(float(r["volume_mm3"]), expected, rel_tol=1e-9), "Incorrect physical volume")
        require(0 <= expected <= float(i["image_volume_mm3"]), "Impossible volume")
        require(0 <= float(r["occupied_slice_fraction"]) <= 1, "Invalid occupied-slice fraction")
    for r in slices:
        require(r["split"] == inv[r["patient_id"]]["split"], "Split mismatch")
        require(0 <= float(r["normalized_z"]) <= 1, "Invalid z")
        require(math.isclose(float(r["relative_area"]), int(r["pixel_area"]) / (int(r["height"]) * int(r["width"]))), "Invalid pixel fraction")
        require((r["present"] == "True") == (int(r["pixel_area"]) > 0), "Presence/area mismatch")
    features = {(r["stem"], r["class_id"]): r for r in slices}
    for r in baseline:
        require(int(features[r["stem"], r["class_id"]]["pixel_area"]) == int(r["gt_area"]), "GT area join mismatch")
        g, p, inter = int(r["gt_area"]), int(r["pred_area"]), int(r["intersection"])
        require(0 <= inter <= min(g, p), "Impossible intersection")
        require(int(r["fp_pixels"]) == p - inter and int(r["fn_pixels"]) == g - inter, "Invalid FP/FN")
        if r["joint_empty"] == "True":
            require(g == p == 0 and r["dice"] == "", "Joint empty must be undefined")
        else:
            require(0 <= float(r["dice"]) <= 1 and math.isclose(float(r["dice"]), 2 * inter / (g + p)), "Invalid Dice")
    for r in patients:
        # A class that is joint-empty in 3D too (e.g. aorta, in a release
        # without it) has an undefined, blank dice_3d even when the
        # reconstruction itself is available.
        if r["reconstruction_available"] == "True" and r["dice_3d"] != "":
            require(0 <= float(r["dice_3d"]) <= 1, "Invalid 3D Dice")
    for stage in ("dataset", "baseline"):
        for r in read_csv(tables / f"{stage}_inputs.csv"):
            stat = Path(r["path"]).stat()
            require(stat.st_size == int(r["size_bytes"]) and stat.st_mtime_ns == int(r["mtime_ns"]),
                    f"Input changed since analysis: {r['path']}")
    for split in ("train", "val", "all"):
        rs = [r for r in read_csv(tables / "class_frequency_original.csv") if r["split"] == split]
        if rs:
            require(math.isclose(sum(float(r["fraction_all_voxels"]) for r in rs), 1), "Frequency does not sum to one")
    examples = read_csv(tables / "baseline_examples.csv")
    expected_sheets = len({r["class_id"] for r in examples})
    report = {"status": "passed", "patients": len(ids), "patient_class_rows": len(organs),
              "slice_class_rows": len(slices), "baseline_slice_class_rows": len(baseline),
              "baseline_patient_class_rows": len(patients),
              "figures": len(list((output / "plots").glob("*.png"))),
              "example_sheets": len(list((output / "examples").glob("*.png")))}
    require(report["figures"] == 11 and report["example_sheets"] == expected_sheets, "Missing expected figures")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
