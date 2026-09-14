"""Training curves from a run's epochs.csv."""
import csv
from pathlib import Path


def plot_curves(csv_path: Path, out_path: Path, dice_columns: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    for col in ("train_loss", "val_loss"):
        ax_loss.plot(epochs, [float(r[col]) for r in rows], label=col)
    for col in dice_columns:
        ax_dice.plot(epochs, [float(r[col]) for r in rows], label=col.removeprefix("val_dice_"),
                     linewidth=2.5 if col == "val_dice_fg" else 1.2)
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_dice.set(xlabel="epoch", ylabel="Dice", title="Validation Dice (patient-level)", ylim=(0, 1))
    for ax in (ax_loss, ax_dice):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(csv_path.parent.parent.name + " / " + csv_path.parent.name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
