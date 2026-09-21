"""Analyze existing validation hard predictions for annotated classes 1–3 only."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from utils import (CLASSES, COLORS, discover, distribution, identity, load_original,
                   load_png, overlap, parser, paths, provenance, pyplot, read_csv, write_csv)


def class_summary(rows):
    """GT-positive slice summaries, with joint-empty and false-positive-only counts separate."""
    result = []
    for k, name in CLASSES.items():
        rs = [r for r in rows if r["class_id"] == k]
        positive = [r for r in rs if r["gt_present"]]
        patient_means = [np.mean([r["dice"] for r in positive if r["patient_id"] == p])
                         for p in sorted({r["patient_id"] for r in positive})]
        result.append({"class_id": k, "class_name": name, "source_grid": "processed_png",
                       "total_slices": len(rs), "gt_positive_slices": len(positive),
                       "joint_empty_slices": sum(r["joint_empty"] for r in rs),
                       "fp_only_slices": sum(not r["gt_present"] and r["pred_present"] for r in rs),
                       "fp_pixels_on_gt_empty": sum(r["fp_pixels"] for r in rs if not r["gt_present"]),
                       "num_positive_patients": len(patient_means),
                       "equal_patient_mean_positive_slice_dice": float(np.mean(patient_means)) if patient_means else np.nan,
                       **{f"positive_slice_dice_{key}": value for key, value in
                          distribution(r["dice"] for r in positive).items()}})
    return result


def bin_index(value: float, edges: np.ndarray) -> int:
    """Left-closed bins; the final bin includes its upper endpoint."""
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def bin_summaries(rows):
    """Report pooled slices AND equal-patient summaries; no independence assumption."""
    output, patient_output = [], []
    for k, name in CLASSES.items():
        positive = [r for r in rows if r["class_id"] == k and r["gt_present"]]
        if not positive:
            continue
        areas = np.array([r["gt_area"] for r in positive])
        area_edges = np.unique(np.quantile(areas, np.linspace(0, 1, 6)))
        if len(area_edges) == 1:
            area_edges = np.array([areas[0], areas[0] + 1])
        for kind, field, edges in (("area_quantile", "gt_area", area_edges),
                                   ("scan_z", "normalized_z", np.linspace(0, 1, 11)),
                                   ("organ_z", "organ_relative_z", np.array([0, .2, .8, 1.]))):
            for b in range(len(edges) - 1):
                rs = [r for r in positive if bin_index(r[field], edges) == b]
                if not rs:
                    continue
                ids = sorted({r["patient_id"] for r in rs})
                patient_means = []
                label = ("first_20_percent", "middle_60_percent", "last_20_percent")[b] if kind == "organ_z" else str(b)
                for patient in ids:
                    pr = [r for r in rs if r["patient_id"] == patient]
                    mean = float(np.mean([r["dice"] for r in pr]))
                    patient_means.append(mean)
                    patient_output.append({"patient_id": patient, "class_id": k, "class_name": name,
                        "source_grid": "processed_png", "bin_type": kind, "bin_id": b,
                        "bin_label": label, "lower": edges[b], "upper": edges[b + 1],
                        "num_slices": len(pr), "mean_dice": mean,
                        "median_dice": float(np.median([r["dice"] for r in pr]))})
                stats = distribution(r["dice"] for r in rs)
                patient_stats = distribution(patient_means)
                output.append({"class_id": k, "class_name": name, "source_grid": "processed_png",
                    "bin_type": kind, "bin_id": b, "bin_label": label,
                    "lower": edges[b], "upper": edges[b + 1], "num_slices": len(rs), "num_patients": len(ids),
                    "median_gt_area": float(np.median([r["gt_area"] for r in rs])),
                    **{f"slice_dice_{key}": v for key, v in stats.items()},
                    **{f"patient_mean_dice_{key}": v for key, v in patient_stats.items()}})
    return output, patient_output


def patient_metrics(rows, original, volumes, processed_predictions):
    """Compare original-grid reconstructed masks, never mean slice Dice, for 3D Dice.

    Verify reconstructed labels against the current PNG predictions so a stale
    reconstruction cannot silently be attached to a newer best checkpoint.
    """
    results, inputs = [], []
    for patient in sorted({r["patient_id"] for r in rows}):
        prediction_path = volumes / f"{patient}.nii.gz"
        scores = {}
        if prediction_path.exists():
            gt_nii, gt = load_original(original, patient)
            pred_nii = nib.load(prediction_path)
            if (pred_nii.shape != gt.shape or not np.allclose(pred_nii.affine, gt_nii.affine)
                    or pred_nii.header.get_xyzt_units()[0] != "mm"
                    or not np.allclose(pred_nii.header.get_zooms(), gt_nii.header.get_zooms())):
                raise ValueError(f"Reconstructed geometry mismatch: {patient}")
            pred = np.asanyarray(pred_nii.dataobj)
            if not np.issubdtype(pred.dtype, np.integer) or pred.min() < 0 or pred.max() > 4:
                raise ValueError(f"Unexpected reconstructed labels: {patient}")
            for z, image in sorted(processed_predictions[patient].items()):
                # This analysis intentionally supports the repository's exact 2x grid change.
                enlarged = image.repeat(2, axis=0).repeat(2, axis=1)
                if not np.array_equal(enlarged, pred[:, :, z]):
                    raise ValueError(f"Reconstruction differs from current best PNG: {patient}, {z}")
            for k in CLASSES:
                m = overlap(gt == k, pred == k)
                scores[k] = {"dice_3d": m["dice"], "gt_voxels_3d": m["gt_area"],
                             "pred_voxels_3d": m["pred_area"], "intersection_voxels_3d": m["intersection"],
                             "fp_voxels_3d": m["fp_pixels"], "fn_voxels_3d": m["fn_pixels"]}
            inputs.extend([prediction_path, original / patient / "GT.nii.gz",
                           original / patient / f"{patient}.nii.gz"])
            print(f"3D verified: {patient}", flush=True)
        for k, name in CLASSES.items():
            rs = [r for r in rows if r["patient_id"] == patient and r["class_id"] == k]
            positive = [r["dice"] for r in rs if r["gt_present"]]
            stats = distribution(positive)
            results.append({"patient_id": patient, "split": "val", "class_id": k, "class_name": name,
                            "num_gt_positive_slices": len(positive), "mean_positive_slice_dice": stats["mean"],
                            "median_positive_slice_dice": stats["median"],
                            "joint_empty_slices": sum(r["joint_empty"] for r in rs),
                            "fp_only_slices": sum(not r["gt_present"] and r["pred_present"] for r in rs),
                            "reconstruction_available": bool(scores),
                            "slice_metric_grid": "processed_png", "volume_metric_grid": "original_nifti",
                            **scores.get(k, dict.fromkeys(("dice_3d", "gt_voxels_3d", "pred_voxels_3d",
                                "intersection_voxels_3d", "fp_voxels_3d", "fn_voxels_3d"), np.nan))})
    return results, inputs


def plots(output, rows, bins, patients):
    """Show conditional slice quality and patient-level variability without conflating grids."""
    plt = pyplot(output)
    from matplotlib.ticker import NullFormatter
    def save(fig, name):
        fig.savefig(output / "plots" / name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.boxplot([[r["dice"] for r in rows if r["class_id"] == k and r["gt_present"]] for k in CLASSES],
               tick_labels=list(CLASSES.values()), showfliers=True)
    ax.set(ylabel="Slice Dice", ylim=(-.03, 1.03), title="Processed PNG validation: GT-positive slices only")
    save(fig, "baseline_positive_dice_distribution.png")

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 4), sharey=True, layout="constrained")
    ids = sorted({r["patient_id"] for r in rows})
    for ax, k in zip(axes, CLASSES):
        any_positive = False
        for patient in ids:
            rs = [r for r in rows if r["class_id"] == k and r["gt_present"] and r["patient_id"] == patient]
            any_positive = any_positive or bool(rs)
            ax.scatter([r["gt_area"] for r in rs], [r["dice"] for r in rs], s=7, alpha=.35, label=patient)
        # An absent class (e.g. aorta, in a release without it) has no positive
        # area anywhere, and a log-scaled axis with no data raises at draw time.
        ax.set(title=CLASSES[k], xlabel="GT area (processed pixels)", ylim=(-.03, 1.03))
        if any_positive:
            ax.set_xscale("log")
    axes[0].set_ylabel("Slice Dice")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Processed PNG validation: GT-positive slices (adjacent slices are correlated)")
    save(fig, "baseline_dice_vs_area_scatter.png")

    for kind, filename, xlabel in (("area_quantile", "baseline_dice_vs_area_binned.png", "Median GT area (pixels)"),
                                   ("scan_z", "baseline_dice_vs_scan_z.png", "Scan-relative z"),
                                   ("organ_z", "baseline_dice_vs_organ_z.png", "Position in processed class extent")):
        fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 4), sharey=True, layout="constrained")
        for ax, k in zip(axes, CLASSES):
            rs = [r for r in bins if r["class_id"] == k and r["bin_type"] == kind]
            x = [r["median_gt_area"] if kind == "area_quantile" else (r["lower"] + r["upper"]) / 2 for r in rs]
            ax.plot(x, [r["patient_mean_dice_median"] for r in rs], "o-", color=COLORS[k],
                    label="Median patient mean")
            ax.fill_between(x, [r["patient_mean_dice_p25"] for r in rs],
                            [r["patient_mean_dice_p75"] for r in rs], color=COLORS[k], alpha=.18)
            ax.plot(x, [r["slice_dice_median"] for r in rs], "x--", color=".4", label="Pooled slice median")
            for xx, r in zip(x, rs):
                ax.annotate(f"{r['num_patients']}p", (xx, r["patient_mean_dice_median"]),
                            xytext=(0, 7), textcoords="offset points", fontsize=7, ha="center")
            ax.set(title=CLASSES[k], xlabel=xlabel, ylim=(-.03, 1.12))
            if kind == "area_quantile" and x:
                ax.set_xscale("log")
                ax.set_xticks(x, [f"{v:g}" for v in x], rotation=35, ha="right")
                ax.xaxis.set_minor_formatter(NullFormatter())
            if kind == "organ_z":
                ax.set_xticks([.1, .5, .9], ["First 20%", "Middle 60%", "Last 20%"])
        axes[0].set_ylabel("GT-positive slice Dice")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncols=2)
        fig.suptitle("Processed PNG validation: equal-patient summaries; shading = patient IQR, p = patients")
        save(fig, filename)

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 4), sharey=True, layout="constrained")
    for ax, k in zip(axes, CLASSES):
        rs = [r for r in patients if r["class_id"] == k]
        x = np.arange(len(rs))
        ax.plot(x, [r["mean_positive_slice_dice"] for r in rs], "o--", label="Mean positive slice Dice (PNG)")
        ax.plot(x, [r["dice_3d"] for r in rs], "s-", label="3D Dice (original grid)")
        ax.set_xticks(x, [r["patient_id"] for r in rs], rotation=35, ha="right", fontsize=8)
        ax.set(title=CLASSES[k], ylim=(-.03, 1.03))
    axes[0].set_ylabel("Dice (distinct definitions)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncols=2)
    fig.suptitle("Validation patient performance: slice mean and reconstructed 3D overlap")
    save(fig, "baseline_patient_performance.png")


def examples(output, rows, processed, predictions):
    """Six deterministic cases per class; contours and crop retain PNG coordinates."""
    plt = pyplot(output)
    from matplotlib.lines import Line2D
    selected, inputs = [], []
    for k, name in CLASSES.items():
        rs = sorted([r for r in rows if r["class_id"] == k and r["gt_present"]],
                    key=lambda r: (r["patient_id"], r["slice_index"]))
        if not rs:
            continue
        median = np.median([r["gt_area"] for r in rs])
        choices = [("small", min(rs, key=lambda r: r["gt_area"])),
                   ("typical", min(rs, key=lambda r: abs(r["gt_area"] - median))),
                   ("large", max(rs, key=lambda r: r["gt_area"])),
                   ("low Dice", min(rs, key=lambda r: r["dice"])),
                   ("high Dice", max(rs, key=lambda r: r["dice"])),
                   ("extremity", min(rs, key=lambda r: (min(r["distance_from_first"], r["distance_from_last"]), r["dice"])))]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        for ax, (reason, row) in zip(axes.flat, choices):
            stem = row["stem"]
            image_path = processed / "val/img" / f"{stem}.png"
            with Image.open(image_path) as image_file:
                image = np.asarray(image_file)
            gt = load_png(processed / "val/gt" / f"{stem}.png") == k
            pred = load_png(predictions / f"{stem}.png") == k
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            for a, color in ((gt, "#00e5ff"), (pred, "#ff8c00")):
                if a.any() and not a.all():
                    ax.contour(a, levels=[.5], colors=[color], linewidths=1)
            yy, xx = np.nonzero(gt | pred)
            cy, cx = (yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2
            half = max(32, (max(np.ptp(yy), np.ptp(xx)) + 24) / 2)
            ax.set_xlim(max(-.5, cx - half), min(image.shape[1] - .5, cx + half))
            ax.set_ylim(min(image.shape[0] - .5, cy + half), max(-.5, cy - half))
            ax.set_title(f"{reason}: {stem}\narea={row['gt_area']} px, Dice={row['dice']:.3f}", fontsize=9)
            selected.append({"class_id": k, "class_name": name, "selection": reason,
                             "stem": stem, "patient_id": row["patient_id"], "slice_index": row["slice_index"],
                             "gt_area": row["gt_area"], "dice": row["dice"],
                             "source_grid": "processed_png", "figure": f"{name}_examples.png"})
            inputs.append(image_path)
        fig.legend([Line2D([], [], color="#00e5ff"), Line2D([], [], color="#ff8c00")],
                   ["GT contour", "Prediction contour"], loc="outside lower center", ncols=2)
        fig.suptitle(f"{name}: processed validation examples; crops in original PNG pixel coordinates")
        fig.savefig(output / "examples" / f"{name}_examples.png", bbox_inches="tight")
        plt.close(fig)
    write_csv(output / "tables/baseline_examples.csv", selected)
    return inputs


def main():
    p = parser(__doc__)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--reconstructed-volumes", type=Path)
    args = p.parse_args()
    root, original, processed, output = paths(args)
    predictions = (args.predictions or root / "results/segthor/ce/best_epoch/val").resolve()
    volumes = (args.reconstructed_volumes or root / "volumes/segthor/ce").resolve()
    if not predictions.is_dir():
        raise FileNotFoundError(predictions)
    for source in (predictions, volumes):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output cannot overlap baseline input directories")
    run_path = output / "dataset_run.json"
    run = json.loads(run_path.read_text())
    feature_path = output / "tables/slice_class_stats.csv"
    features = read_csv(feature_path)
    patients = discover(processed)
    selected_ids = set(run["patient_ids"])
    expected = {gt.name for patient, entry in patients.items()
                if entry["split"] == "val" and patient in selected_ids for _, gt in entry["slices"].values()}
    actual = {p.name for p in predictions.glob("*.png")}
    if not expected or not expected <= actual or (not run["subset"] and expected != actual):
        raise ValueError("Prediction names do not match validation GT names")
    feature_map = {(r["stem"], int(r["class_id"])): r for r in features if r["split"] == "val"}
    if len(feature_map) != sum(r["split"] == "val" for r in features) or set(feature_map) != {
            (Path(name).stem, k) for name in expected for k in CLASSES}:
        raise ValueError("Duplicate or missing validation feature rows")
    rows, inputs, decoded = [], [run_path, feature_path], {}
    for name in sorted(expected):
        gt_path, pred_path = processed / "val/gt" / name, predictions / name
        gt, pred = load_png(gt_path), load_png(pred_path)
        patient, z = identity(gt_path)
        decoded.setdefault(patient, {})[z] = pred
        for k, class_name in CLASSES.items():
            feature = feature_map[(gt_path.stem, k)]
            m = overlap(gt == k, pred == k)
            if int(feature["pixel_area"]) != m["gt_area"]:
                raise ValueError(f"Stale area features: {name}, {k}")
            rows.append({"patient_id": patient, "split": "val", "stem": gt_path.stem,
                         "slice_index": z, "class_id": k, "class_name": class_name,
                         "source_grid": "processed_png", "normalized_z": float(feature["normalized_z"]),
                         "organ_relative_z": float(feature["organ_relative_z"]) if feature["organ_relative_z"] else np.nan,
                         "distance_from_first": int(feature["distance_from_first"]) if feature["distance_from_first"] else None,
                         "distance_from_last": int(feature["distance_from_last"]) if feature["distance_from_last"] else None,
                         "relative_area": float(feature["relative_area"]), **m})
        inputs.extend([gt_path, pred_path])
    print(f"Baseline: paired {len(expected)} validation slices by exact stem", flush=True)
    summary = class_summary(rows)
    bins, patient_bins = bin_summaries(rows)
    patient_rows, volume_inputs = patient_metrics(rows, original, volumes, decoded)
    inputs.extend(volume_inputs)
    for name, records in (("baseline_slice_metrics", rows), ("baseline_class_summary", summary),
                          ("baseline_binned_summary", bins), ("baseline_patient_bins", patient_bins),
                          ("baseline_patient_metrics", patient_rows)):
        write_csv(output / "tables" / f"{name}.csv", records)
    plots(output, rows, bins, patient_rows)
    inputs.extend(examples(output, rows, processed, predictions))
    checkpoint_note = predictions.parent.parent / "best_epoch.txt"
    if checkpoint_note.exists():
        inputs.append(checkpoint_note)
    provenance(output, "baseline", args, inputs, {"num_slice_class_rows": len(rows),
               "checkpoint_note": checkpoint_note.read_text() if checkpoint_note.exists() else None,
               "annotated_classes": CLASSES, "joint_empty_dice": "undefined (blank CSV cell)",
               "main_summary_population": "GT-positive processed slices only",
               "dice_3d": "full original-grid reconstructed prediction vs original GT",
               "area_bins": "within-class 5 quantile bins, duplicate edges collapsed",
               "position_bins": "left-closed; last includes upper endpoint",
               "num_available_reconstructions": sum(r["reconstruction_available"] for r in patient_rows)
                   // len(CLASSES)})
    print(f"Baseline complete -> {output}", flush=True)


if __name__ == "__main__":
    main()
