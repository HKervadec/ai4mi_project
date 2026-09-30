# Optimiser experiments

## Implemented

`main.py` now supports:

- `--optimizer adam|adamw|sgd_nesterov` (default: `adam`);
- `--lr` and `--weight-decay`;
- `--lr-scheduler none|polynomial` (default: `none`).

The defaults preserve the original Adam baseline: learning rate `0.0005`, zero weight decay, and no scheduler. The polynomial scheduler reduces the learning rate after each epoch.

## TOY2 check (five epochs)

| Configuration | Best validation foreground Dice |
| --- | ---: |
| Adam baseline | 0.955 |
| AdamW, LR `0.0005`, WD `1e-4`, polynomial | 0.880 |
| AdamW, LR `0.0003`, WD `1e-4`, polynomial | 0.631 |
| Nesterov SGD, LR `0.01`, WD `1e-4`, polynomial | 0.986 |
| Nesterov SGD, LR `0.03`, WD `1e-4`, polynomial | **0.990** |

The toy experiment confirms that all configurations run. It suggests Nesterov SGD is promising, but does not establish performance on medical images.

## SegTHOR plan

Run the following with identical data, model, loss, epoch budget, and evaluation:

1. Adam baseline.
2. AdamW: LR `0.0005`, weight decay `1e-4`, polynomial scheduler.
3. Nesterov SGD: LR `0.01`, weight decay `1e-4`, polynomial scheduler.
4. Nesterov SGD: LR `0.03`, weight decay `1e-4`, polynomial scheduler.

Compare validation Dice during training and final stitched 3D Dice. Repeat the strongest settings with multiple seeds if compute allows.
