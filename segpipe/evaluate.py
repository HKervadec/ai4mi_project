"""3D evaluation of a run's predictions. Not used yet: runs currently report the 2D validation Dice."""

# name -> fn(pred_mask, gt_mask, spacing) -> float
METRICS: dict = {}
