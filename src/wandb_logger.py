"""Fail-soft Weights & Biases wrapper: any W&B problem (no login, no network, server error)
is logged once and turns W&B off for the rest of the run. Training never depends on it;
everything is also written to the run directory."""
import getpass
import logging

import numpy as np

LOG = logging.getLogger("ai4mi")


class WandbLogger:
    def __init__(self, cfg: dict, run_dir, run_name: str, run_id: str | None = None,
                 job_type: str = "train"):
        self.run = None
        w = cfg["wandb"]
        if w["mode"] == "disabled":
            return
        try:
            import wandb
            self.wandb = wandb
            tags = [cfg["model"]["name"], cfg["loss"]["name"], cfg["data"]["root"].rstrip("/").split("/")[-1],
                    getpass.getuser(), *w["tags"]]
            self.run = wandb.init(project=w["project"], entity=w["entity"] or None, mode=w["mode"],
                                  id=run_id, resume="allow",  # run_id None: W&B picks a new id
                                  name=run_name, group=cfg["experiment"], job_type=job_type,
                                  tags=tags, notes=cfg["notes"] or None, config=cfg, dir=str(run_dir),
                                  settings=wandb.Settings(init_timeout=120))
            LOG.info("W&B run: %s", self.run.url or f"{w['mode']} ({self.run.id})")
        except Exception as exc:  # noqa: BLE001 — W&B must never take down a run
            self._disable("init", exc)

    def _disable(self, where: str, exc: Exception) -> None:
        LOG.warning("W&B disabled after %s failure (%s: %s); results are still saved locally",
                    where, type(exc).__name__, exc)
        self.run = None

    @property
    def id(self) -> str | None:
        return self.run.id if self.run else None

    @property
    def url(self) -> str | None:
        return self.run.url if self.run else None

    def log(self, metrics: dict, step: int) -> None:
        if self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as exc:  # noqa: BLE001
                self._disable("log", exc)

    def log_overlays(self, images: np.ndarray, gts: np.ndarray, preds: np.ndarray, stems: list[str],
                     class_names: list[str], step: int) -> None:
        """images (N,H,W) in [0,1]; gts/preds (N,H,W) class maps. Interactive mask overlays in the UI."""
        if not self.run:
            return
        try:
            labels = dict(enumerate(class_names))
            self.log({"val_overlays": [
                self.wandb.Image(img, caption=stem, masks={
                    "prediction": {"mask_data": pred, "class_labels": labels},
                    "ground_truth": {"mask_data": gt, "class_labels": labels}})
                for img, gt, pred, stem in zip(images, gts, preds, stems)]}, step)
        except Exception as exc:  # noqa: BLE001
            self._disable("image", exc)

    def summary(self, values: dict) -> None:
        if self.run:
            try:
                self.run.summary.update(values)
            except Exception as exc:  # noqa: BLE001
                self._disable("summary", exc)

    def finish(self, exit_code: int = 0) -> None:
        if self.run:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:  # noqa: BLE001
                self._disable("finish", exc)
