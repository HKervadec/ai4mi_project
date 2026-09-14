"""3D evaluation of a run's best checkpoint.

    python -m src.evaluate --run runs/segthor_enet_ce/seed0

Writes into the run directory:
  predictions/<split>/<stem>.png     2D predictions, grey = class * label_scale (viewer-compatible)
  volumes/<split>/<patient>.nii.gz   predictions stitched back onto the original CT grid
  eval/metrics_3d.csv                one row per (patient, class): dice, hd95, assd
  eval/<split>/{dice,hd95,assd}.npz  submission format: patient -> (K,) array, background = NaN
then adds an `eval` block to summary.json and W&B, and refreshes the metrics/ copy.
The val split is scored against `data.source_pattern` volumes; the test split (if
`data.test_source_pattern` is set and <data.root>/test exists) is only predicted and stitched.
"""
import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.config import REPO, read_yaml
from src.data import build_dataset, patient_of
from src.engine import build_model
from src.metrics_3d import METRICS, volume_metrics
from src.run import copy_back, read_json, setup_logging, write_json
from src.wandb_logger import WandbLogger


def predict(net, cfg: dict, split: str, device, out_dir: Path) -> dict[str, np.ndarray]:
    cfg = cfg | {"train": cfg["train"] | {"debug_samples": 0}}  # always the whole split
    loader = DataLoader(build_dataset(cfg, split), batch_size=cfg["data"]["batch_size"],
                        num_workers=cfg["data"]["num_workers"], shuffle=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    net.eval()
    with torch.no_grad():
        for batch in loader:
            classes = net(batch["images"].to(device)).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for stem, pred in zip(batch["stems"], classes):
                preds[stem] = pred
                Image.fromarray(pred * cfg["data"]["label_scale"]).save(out_dir / f"{stem}.png")
    return preds


def stitch(slices: dict[int, np.ndarray], reference: nib.Nifti1Image, patient: str) -> np.ndarray:
    """Inverse of slice_segthor.py: slice z of the volume was resized to 256x256 and saved as one PNG."""
    X, Y, Z = reference.shape
    if sorted(slices) != list(range(Z)):
        raise ValueError(f"{patient}: predicted {len(slices)} slices, volume has {Z}")
    stack = np.stack([slices[z] for z in range(Z)], axis=-1)
    return resize(stack, (X, Y, Z), order=0, preserve_range=True, anti_aliasing=False,
                  mode="constant").astype(np.uint8)


def group_by_patient(preds: dict[str, np.ndarray], regex: str) -> dict[str, dict[int, np.ndarray]]:
    grouped = defaultdict(dict)
    for stem, pred in preds.items():
        grouped[patient_of(stem, regex)][int(stem.rsplit("_", 1)[1])] = pred
    return dict(sorted(grouped.items()))


def mean(values) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def evaluate(run_dir: Path, device: torch.device) -> dict:
    cfg = read_yaml((run_dir / "config.yaml").read_text())
    log = setup_logging(run_dir / "eval.log")
    d, names = cfg["data"], cfg["data"]["class_names"]
    net = build_model(cfg, device)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    net.load_state_dict(best["model"])
    log.info("evaluating %s, best epoch %d, on %s", run_dir, best["epoch"], device)

    spacing_pkl = REPO / d["root"] / "spacing.pkl"
    sliced_spacing = pickle.loads(spacing_pkl.read_bytes()) if spacing_pkl.exists() else {}

    splits = {"val": d["source_pattern"]}
    if d["test_source_pattern"] and (REPO / d["root"] / "test").is_dir():
        splits["test"] = d["test_source_pattern"]
    rows, block = [], {}
    for split, pattern in splits.items():
        preds = predict(net, cfg, split, device, run_dir / "predictions" / split)
        log.info("%s: predicted %d slices -> %s", split, len(preds), run_dir / "predictions" / split)
        if pattern is None:
            continue
        per_metric = {m: {} for m in METRICS}
        for patient, slices in group_by_patient(preds, d["patient_regex"]).items():
            ref = nib.load(REPO / pattern.format(patient=patient))
            spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
            log.info("%s %s: shape %s, spacing %s mm, orientation %s", split, patient, ref.shape,
                     np.round(spacing, 3).tolist(), "".join(nib.aff2axcodes(ref.affine)))
            if patient in sliced_spacing and not np.allclose(sliced_spacing[patient], spacing):
                log.warning("%s: spacing %s differs from spacing.pkl %s", patient, spacing,
                            sliced_spacing[patient])
            vol = stitch(slices, ref, patient)
            out = run_dir / "volumes" / split / f"{patient}.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(vol, ref.affine, ref.header), out)
            if split == "test":
                continue
            gt = np.asarray(ref.dataobj)
            for m in METRICS:
                per_metric[m][patient] = np.full(d["num_classes"], np.nan)
            for k in range(1, d["num_classes"]):
                values = volume_metrics(gt == k, vol == k, spacing)
                for m in METRICS:
                    per_metric[m][patient][k] = values[m]
                rows.append({"split": split, "patient": patient, "class_idx": k, "class_name": names[k],
                             "gt_voxels": int(np.count_nonzero(gt == k)),
                             "pred_voxels": int(np.count_nonzero(vol == k)), **values})
        if split == "test":
            continue
        (run_dir / "eval" / split).mkdir(parents=True, exist_ok=True)
        for m in METRICS:
            np.savez(run_dir / "eval" / split / f"{m}.npz", **per_metric[m])
        for m in METRICS:
            per_class = {k: mean([v[k] for v in per_metric[m].values()]) for k in range(1, d["num_classes"])}
            block |= {f"{split}_{m}_{names[k]}": per_class[k] for k in cfg["eval"]["classes"]}
            block[f"{split}_{m}_fg"] = mean([per_class[k] for k in cfg["eval"]["classes"]])
        block[f"{split}_patients"] = len(per_metric["dice"])

    if rows:
        with (run_dir / "eval" / "metrics_3d.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    block |= {"best_epoch": best["epoch"]}
    summary = read_json(run_dir / "summary.json")
    write_json(run_dir / "summary.json", summary | {"eval": block})
    log.info("3D %s", " | ".join(f"{k} {v:.4f}" for k, v in block.items() if k.endswith("_fg")))

    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("wandb_id"):
        wb = WandbLogger(cfg, run_dir, run_name=f"{cfg['experiment']}/seed{cfg['seed']}",
                         run_id=manifest["wandb_id"], job_type="eval")
        wb.summary({f"eval3d/{k}": v for k, v in block.items()})
        wb.finish()
    dest = copy_back(run_dir, cfg)
    if dest:
        log.info("copied results to %s", dest)
    return block


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory, e.g. runs/<experiment>/seed0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    evaluate(args.run.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
