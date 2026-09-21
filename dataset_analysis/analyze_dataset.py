"""Inventory original NIfTI anatomy and separately describe processed PNG targets."""
from __future__ import annotations

import numpy as np
import nibabel as nib
from PIL import Image

from utils import (CLASSES, COLORS, discover, distribution, extent, load_original,
                   load_png, normalized_z, parser, paths, provenance, pyplot, write_csv)


def original_stats(patient, split, nii, data):
    """Compute physical volumes on the original grid; never on resized PNG pixels."""
    shape = data.shape
    spacing = nii.header.get_zooms()[:3]
    voxel_mm3 = abs(float(np.linalg.det(nii.affine[:3, :3])))
    counts = np.bincount(data.ravel(), minlength=len(CLASSES) + 1)
    foreground = int(counts[1:].sum())
    inventory = {"patient_id": patient, "split": split, "source_grid": "original_nifti",
                 "shape_x": shape[0], "shape_y": shape[1], "num_slices": shape[2],
                 "spacing_x_mm": spacing[0], "spacing_y_mm": spacing[1], "spacing_z_mm": spacing[2],
                 "axis_codes": "".join(nib.aff2axcodes(nii.affine)), "voxel_volume_mm3": voxel_mm3,
                 "image_volume_mm3": data.size * voxel_mm3, "background_voxels": int(counts[0]),
                 "foreground_voxels": foreground}
    rows = []
    for k, name in CLASSES.items():
        mask = data == k
        axes = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != axis)))
                for axis in range(3)]
        positive = axes[2].tolist()
        first, last = (positive[0], positive[-1]) if positive else (None, None)
        count = int(counts[k])
        inventory[f"class_{k}_present"] = bool(count)
        inventory[f"class_{k}_voxels"] = count
        rows.append({"patient_id": patient, "split": split, "source_grid": "original_nifti",
                     "class_id": k, "class_name": name, "present": bool(count),
                     "voxel_count": count, "volume_mm3": count * voxel_mm3,
                     "volume_ml": count * voxel_mm3 / 1000,
                     "fraction_of_patient_foreground": count / foreground if foreground else np.nan,
                     "first_slice_index": first, "last_slice_index": last,
                     "occupied_slice_count": len(positive), "num_slices_patient": shape[2],
                     "occupied_slice_fraction": len(positive) / shape[2],
                     "normalized_first_z": normalized_z(first, shape[2]) if positive else np.nan,
                     "normalized_last_z": normalized_z(last, shape[2]) if positive else np.nan,
                     **{f"bbox_extent_{axis}_mm": (a[-1] - a[0] + 1) * s if len(a) else np.nan
                        for axis, a, s in zip("xyz", axes, spacing)}})
    return inventory, rows


def processed_stats(patient, entry, num_slices):
    """Keep all slices, including zero-area targets, and attach processed class extent."""
    if sorted(entry["slices"]) != list(range(num_slices)):
        raise ValueError(f"Incomplete original slice coverage for {patient}")
    counts, shapes = {}, {}
    for z, (image, gt) in sorted(entry["slices"].items()):
        a = load_png(gt)
        with Image.open(image) as im:
            if im.size != a.shape[::-1] or im.mode != "L":
                raise ValueError(f"Image/mask shape or channel mismatch: {image}")
        if a.shape != (256, 256):
            raise ValueError(f"Expected repository's 256x256 grid: {gt}")
        counts[z] = np.bincount(a.ravel(), minlength=len(CLASSES) + 1)
        shapes[z] = a.shape
    positives = {k: [z for z in sorted(counts) if counts[z][k] > 0] for k in CLASSES}
    rows = []
    for z in sorted(counts):
        h, w = shapes[z]
        for k, name in CLASSES.items():
            area = int(counts[z][k])
            rows.append({"patient_id": patient, "split": entry["split"],
                         "slice_index": z, "stem": entry["slices"][z][1].stem,
                         "source_grid": "processed_png", "height": h, "width": w,
                         "num_slices_patient": num_slices, "normalized_z": normalized_z(z, num_slices),
                         "class_id": k, "class_name": name, "present": bool(area),
                         "pixel_area": area, "relative_area": area / (h * w),
                         **extent(positives[k], z)})
    return rows


def summaries(inventory, slices):
    """Summarize pooled voxels and slices explicitly; these are not equal-patient means."""
    frequency, presence = [], []
    for split in ("train", "val", "all"):
        patients = [r for r in inventory if split == "all" or r["split"] == split]
        if not patients:
            continue
        total = sum(r["shape_x"] * r["shape_y"] * r["num_slices"] for r in patients)
        fg = sum(r["foreground_voxels"] for r in patients)
        for k, name in {0: "background", **CLASSES}.items():
            n = sum(r["background_voxels"] if k == 0 else r[f"class_{k}_voxels"] for r in patients)
            frequency.append({"split": split, "source_grid": "original_nifti", "class_id": k,
                              "class_name": name, "num_patients": len(patients), "voxel_count": n,
                              "fraction_all_voxels": n / total,
                              "fraction_foreground_voxels": n / fg if k and fg else np.nan})
        for k, name in CLASSES.items():
            selected = [r for r in slices if r["class_id"] == k and (split == "all" or r["split"] == split)]
            areas = [r["pixel_area"] for r in selected if r["present"]]
            presence.append({"split": split, "source_grid": "processed_png", "class_id": k,
                             "class_name": name, "num_patients": len(patients),
                             "total_slices": len(selected), "positive_slices": len(areas),
                             "percentage_present": 100 * len(areas) / len(selected),
                             **{f"positive_area_{key}": value for key, value in distribution(areas).items()}})
    return frequency, presence


def area_trends(slices):
    """Average slices within each patient/z bin, then summarize equally weighted patients.

    Z is a scan-relative coordinate, not cross-patient anatomical registration.
    Zeros are retained. IQR describes patients, not a confidence interval.
    """
    patient_bins = {}
    for r in slices:
        b = min(int(r["normalized_z"] * 10), 9)
        key = r["split"], r["patient_id"], r["class_id"], b
        patient_bins.setdefault(key, []).append(r["relative_area"])
    per_patient = [{"split": s, "patient_id": p, "class_id": k, "class_name": CLASSES[k],
                    "source_grid": "processed_png", "z_bin": b, "z_midpoint": (b + 0.5) / 10,
                    "num_slices": len(v), "mean_relative_area": float(np.mean(v))}
                   for (s, p, k, b), v in sorted(patient_bins.items())]
    aggregate = []
    for split in ("train", "val", "all"):
        for k in CLASSES:
            for b in range(10):
                rs = [r for r in per_patient if r["class_id"] == k and r["z_bin"] == b
                      and (split == "all" or r["split"] == split)]
                if rs:
                    aggregate.append({"split": split, "class_id": k, "class_name": CLASSES[k],
                                      "source_grid": "processed_png", "z_bin": b,
                                      "z_midpoint": (b + 0.5) / 10, "num_patients": len(rs),
                                      **distribution(r["mean_relative_area"] for r in rs)})
    return per_patient, aggregate


def plots(output, frequency, patients, slices, trends):
    """Generate a compact set of figures with source grid and weighting in titles."""
    plt = pyplot(output)
    def save(fig, name):
        fig.savefig(output / "plots" / name, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, ks, field, title in zip(axes, ([0, *CLASSES], list(CLASSES)),
            ("fraction_all_voxels", "fraction_foreground_voxels"),
            ("All labeled voxels (including background)", "Annotated foreground only")):
        for j, split in enumerate(("train", "val", "all")):
            rows = [r for r in frequency if r["split"] == split and r["class_id"] in ks]
            ax.bar(np.arange(len(ks)) + (j - 1) * .25,
                   [100 * r[field] for r in rows], width=.25, label=split)
        ax.set_xticks(range(len(ks)), [{0: "background", **CLASSES}[k] for k in ks], rotation=15)
        ax.set(ylabel="Pooled voxel fraction (%)", title=title)
        ax.legend()
    fig.suptitle("Original NIfTI: class frequency")
    save(fig, "original_class_frequency.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(axes, ("volume_ml", "occupied_slice_count"),
                               ("Annotated volume (mL)", "Occupied slices")):
        for k in CLASSES:
            rs = [r for r in patients if r["class_id"] == k]
            ax.boxplot([[r[field] for r in rs]], positions=[k], widths=.5, showfliers=False)
            for j, r in enumerate(rs):
                ax.scatter(k + .3 * (j / max(len(rs) - 1, 1) - .5), r[field],
                           marker="^" if r["split"] == "val" else "o", s=23,
                           c=COLORS[k], alpha=.7)
        ax.set_xticks(list(CLASSES), list(CLASSES.values()))
        ax.set(ylabel=title, title=title)
    axes[0].set_yscale("log")
    fig.suptitle("Original NIfTI: patient variation (triangles = validation)")
    save(fig, "original_patient_volume_extent.png")

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 5), sharey=True, layout="constrained")
    ids = sorted({r["patient_id"] for r in patients})
    for ax, k in zip(axes, CLASSES):
        for r in patients:
            if r["class_id"] == k and r["present"]:
                y = ids.index(r["patient_id"])
                ax.plot([r["normalized_first_z"], r["normalized_last_z"]], [y, y],
                        color=COLORS[k], marker="|", linewidth=2)
        ax.set(title=CLASSES[k], xlabel="Scan-relative z", xlim=(0, 1))
    axes[0].set_yticks(range(len(ids)), ids, fontsize=8)
    fig.suptitle("Original NIfTI: first–last labeled slice (gaps, if any, are not shown)")
    save(fig, "original_normalized_extent.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, field, label in zip(axes, ("pixel_area", "relative_area"),
                               ("Positive target area (pixels, log scale)", "Positive target fraction (log scale)")):
        for j, split in enumerate(("train", "val")):
            values = [[r[field] for r in slices if r["class_id"] == k and r["present"]
                       and r["split"] == split] for k in CLASSES]
            boxes = ax.boxplot(values, positions=np.arange(1, len(CLASSES) + 1) + (j - .5) * .3,
                               widths=.25, showfliers=False, patch_artist=True)
            for patch in boxes["boxes"]:
                patch.set_facecolor(("#9ecae1", "#fdae6b")[j])
            ax.plot([], [], color=("#9ecae1", "#fdae6b")[j], linewidth=8, label=split)
        ax.set_xticks(list(CLASSES), list(CLASSES.values()))
        ax.set(yscale="log", ylabel=label)
        ax.legend()
    fig.suptitle("Processed PNG: GT-positive slices; zeros excluded; whiskers = 1.5 IQR")
    save(fig, "processed_positive_area_distribution.png")

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 4), layout="constrained")
    for ax, k in zip(axes, CLASSES):
        for split, style in (("train", "-"), ("val", "--")):
            rs = [r for r in trends if r["class_id"] == k and r["split"] == split]
            x = [r["z_midpoint"] for r in rs]
            ax.plot(x, [r["median"] for r in rs], style, label=split)
            ax.fill_between(x, [r["p25"] for r in rs], [r["p75"] for r in rs], alpha=.15)
        ax.set(title=CLASSES[k], xlabel="Scan-relative z", ylabel="Relative area (zeros included)")
        ax.legend()
    fig.suptitle("Processed PNG: median/IQR of patient-bin mean area; patients equally weighted")
    save(fig, "processed_area_vs_z.png")


def main():
    p = parser(__doc__)
    p.add_argument("--max-patients", type=int, help="Deterministic smoke-test subset; use a separate output directory")
    args = p.parse_args()
    root, original, processed, output = paths(args)
    if args.max_patients is not None and args.max_patients < 1:
        p.error("--max-patients must be positive")
    patients = discover(processed)
    if args.max_patients:
        patients = dict(list(patients.items())[:args.max_patients])
    inventory, organs, slices, inputs = [], [], [], []
    for patient, entry in patients.items():
        print(f"Dataset: {patient} ({entry['split']})", flush=True)
        nii, gt = load_original(original, patient)
        inv, rows = original_stats(patient, entry["split"], nii, gt)
        inventory.append(inv)
        organs.extend(rows)
        slices.extend(processed_stats(patient, entry, gt.shape[2]))
        inputs.extend([original / patient / "GT.nii.gz", original / patient / f"{patient}.nii.gz"])
        inputs.extend(path for pair in entry["slices"].values() for path in pair)
    frequency, presence = summaries(inventory, slices)
    patient_bins, trends = area_trends(slices)
    for name, rows in (("patient_inventory", inventory), ("patient_class_stats", organs),
                       ("class_frequency_original", frequency), ("slice_class_stats", slices),
                       ("slice_presence_summary", presence), ("processed_patient_z_bins", patient_bins),
                       ("processed_area_z_summary", trends)):
        write_csv(output / "tables" / f"{name}.csv", rows)
    plots(output, frequency, organs, slices, trends)
    provenance(output, "dataset", args, inputs, {"patient_ids": list(patients),
               "subset": bool(args.max_patients), "num_slice_class_rows": len(slices),
               "annotated_classes": CLASSES, "normalized_z": "index / (Z-1); Z=1 -> 0",
               "single_slice_organ_relative_z": 0.5})
    print(f"Dataset complete: {len(inventory)} patients, {len(slices)} slice-class rows -> {output}", flush=True)


if __name__ == "__main__":
    main()
