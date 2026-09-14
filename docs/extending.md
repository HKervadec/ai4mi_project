# Extending the pipeline

Every swappable component is a **factory function registered under a name**. Configs select it
by that name and pass their `kwargs` to it. You never edit the training loop to add one.

| kind | lives in | factory receives | config |
|---|---|---|---|
| `model` | `src/models/` | `in_channels, num_classes, **kwargs` | `model.name`, `model.kwargs` |
| `loss` | `src/losses/` | `num_classes, **kwargs` | `loss.name`, `loss.kwargs` |
| `optim` | `src/optim.py` | `params, **kwargs` | `optim.name`, `optim.kwargs` |
| `scheduler` | `src/optim.py` | `optimizer, epochs, **kwargs` | `scheduler.name`, `scheduler.kwargs` |
| `augment` | `src/augment.py` (create it) | `**kwargs` | `data.augment: [{name, kwargs}]` |

`python -c "import src.engine; from src.registry import available; print(available('model'))"`
lists what is registered.

## Add an architecture in 3 steps

**1. Write the network and register a factory**: a new file `src/models/unet.py`:

```python
from torch import nn

from src.registry import register


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32, depth: int = 4):
        super().__init__()
        ...

    def forward(self, x):          # (B, in_channels, H, W) -> logits (B, num_classes, H, W)
        ...


@register("model", "unet")
def build_unet(in_channels: int, num_classes: int, base: int = 32, depth: int = 4) -> nn.Module:
    return UNet(in_channels, num_classes, base=base, depth=depth)
```

The factory is where you adapt an existing class's constructor to the common
`(in_channels, num_classes, **kwargs)` signature; see `src/models/baseline.py`, which does this
for the course's ENet and shallowCNN. Return **logits**: the engine applies the softmax.

**2. Import it** in `src/models/__init__.py` (one line; this is what registers it):

```python
from . import baseline  # noqa: F401  enet, shallow_cnn
from . import unet      # noqa: F401  unet
```

**3. Add a config** in `configs/segthor_unet_ce.yaml`, then smoke-test it:

```yaml
experiment: segthor_unet_ce
notes: plain U-Net, 4 levels
model:
  name: unet
  kwargs: {base: 32, depth: 4}
data:
  batch_size: 16
```

```bash
sbatch --export=ALL,CONFIG=configs/segthor_unet_ce.yaml jobsAndOutputs/pipeline/jobs/smoke.job
```

That's all. Checkpointing, resume, metrics, W&B, 3D evaluation and tables come for free.
Two contracts to respect: output spatial size must equal input size (256x256), and
`checkpoints/best_model.pkl` pickles the whole network, so keep the class importable from its module.

## Add a loss

Losses receive **softmax probabilities** `(B, K, H, W)` and the one-hot target `(B, K, H, W)`,
like the course `CrossEntropy` in `losses.py`, and return a scalar. New file `src/losses/dice.py`:

```python
from torch import Tensor

from src.registry import register


class SoftDice:
    def __init__(self, idk: list[int], eps: float = 1e-6):
        self.idk, self.eps = idk, eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        p, t = probs[:, self.idk], target[:, self.idk].float()
        inter = (p * t).sum((0, 2, 3))
        return 1 - ((2 * inter + self.eps) / (p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + self.eps)).mean()


@register("loss", "soft_dice")
def build_soft_dice(num_classes: int, idk: list[int] | None = None, eps: float = 1e-6):
    return SoftDice(list(range(1, num_classes)) if idk is None else idk, eps)
```

Then `from . import dice` in `src/losses/__init__.py`, and `loss: {name: soft_dice}` in a config.
A combined loss is just another factory that builds two losses and returns their weighted sum.

## Add an online augmentation

Create `src/augment.py`, import it at the top of `src/data.py` (`import src.augment  # noqa: F401`),
and register factories returning a callable `(image (1,H,W), gt (K,H,W)) -> (image, gt)`.
They run on the **train split only**, in DataLoader workers, so use `torch`/`numpy`/`random`
RNGs (each worker is seeded from the run seed):

```python
import random

import torch

from src.registry import register


@register("augment", "random_flip")
def build_random_flip(p: float = 0.5):
    def flip(image, gt):
        if random.random() < p:
            return torch.flip(image, dims=[-1]), torch.flip(gt, dims=[-1])
        return image, gt
    return flip
```

```yaml
data:
  augment:
    - {name: random_flip, kwargs: {p: 0.5}}
```

Spatial transforms must move image and gt together and keep the gt one-hot (use nearest
interpolation for the gt).

## Change the dataset or the preprocessing

Preprocessing (HU windowing, resampling, cropping, a different slice size...) happens **offline**
when slicing the NIfTI volumes, not in the training loop:

1. Extend `slice_segthor.py` (or write a variant) and slice into a **new** directory, keeping
   the layout `<root>/{train,val}/{img,gt}/<Patient_XX>_<zzzz>.png` and the same patient split.
   Never overwrite `data/SEGTHOR`: other runs point at it.
   ```bash
   python slice_segthor.py --source_dir data/segthor_part1 --dest_dir data/SEGTHOR_hu_window ...
   ```
2. Point the config at it: `data: {root: data/SEGTHOR_hu_window}`. The run records `data.root`,
   and the dataset name becomes a W&B tag, so tables show which preprocessing each run used.

**When the final dataset arrives:** slice it the same way into e.g. `data/SEGTHOR_final`, then set
in `configs/base.yaml` (so every experiment follows):

```yaml
data:
  root: data/SEGTHOR_final
  source_pattern: data/<final_dir>/train/{patient}/GT.nii.gz     # val GT volumes for 3D eval
  test_source_pattern: data/<final_dir>/test/{patient}.nii.gz    # test CTs, for stitching predictions
eval:
  classes: [1, 2, 3, 4]      # once the aorta is annotated
```

If slice file names change, update `data.patient_regex` (group 1 must be the patient id; the
part after the last `_` must be the slice index). A different class count means changing
`num_classes`, `class_names` and `label_scale` (grey value per class in the gt PNGs).
