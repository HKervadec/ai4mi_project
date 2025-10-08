import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from losses import (
    CrossEntropy, DiceLoss, DiceLoss2,
    GeneralizedDiceLoss, FocalLoss, ComboLoss1
)

torch.manual_seed(0)


def make_dummy_data(B=2, K=3, W=8, H=8, dtype=torch.float32, requires_grad=False):
    logits = torch.randn(B, K, W, H, dtype=dtype, requires_grad=requires_grad)
    pred_softmax = F.softmax(logits, dim=1)

    target_idx = torch.randint(0, K, (B, W, H))
    target_onehot = F.one_hot(target_idx, num_classes=K).permute(0, 3, 1, 2).to(dtype)
    return logits, pred_softmax, target_onehot


def test_forward_and_backward(loss_fn, name):
    logits, pred_softmax, target_onehot = make_dummy_data(requires_grad=True)

    try:
        loss = loss_fn(pred_softmax, target_onehot)
        loss_value = float(loss.item())

        loss.backward()
        grad_ok = logits.grad is not None and not torch.isnan(logits.grad).any()

        print(f"Good {name:<25} | loss = {loss_value:.6f} | grad ok: {grad_ok}")
    except Exception as e:
        print(f"Bad {name:<25} | ERROR: {e}")


def test_gradcheck(loss_fn, name):
    """Run torch.autograd.gradcheck on a small double-precision input."""
    logits = torch.randn(1, 2, 4, 4, dtype=torch.double, requires_grad=True)
    pred_softmax = F.softmax(logits, dim=1)
    target_idx = torch.randint(0, 2, (1, 4, 4))
    target_onehot = F.one_hot(target_idx, num_classes=2).permute(0, 3, 1, 2).double()

    try:
        result = gradcheck(loss_fn, (pred_softmax, target_onehot), eps=1e-6, atol=1e-4)
        print(f"Good {name:<25} | gradcheck: {result}")
    except Exception as e:
        print(f"Bad {name:<25} | gradcheck ERROR: {e}")


if __name__ == "__main__":
    losses = {
        "CrossEntropy": CrossEntropy(idk=[0, 1, 2]),
        "DiceLoss": DiceLoss(),
        "DiceLoss2": DiceLoss2(),
        "GeneralizedDiceLoss": GeneralizedDiceLoss(),
        "FocalLoss": FocalLoss(alpha=[0.5, 0.3, 0.2]),
        "ComboLoss1": ComboLoss1(alpha=0.5, idk=[0, 1, 2]),
    }

    print("=== Forward & Backward Tests ===")
    for name, loss_fn in losses.items():
        test_forward_and_backward(loss_fn, name)

    print("\n=== Gradient Check (gradcheck) ===")
    gradcheck_losses = {
        "DiceLoss": DiceLoss(smooth=1e-6),
        "GeneralizedDiceLoss": GeneralizedDiceLoss(smooth=1e-6),
        "FocalLoss": FocalLoss(alpha=[0.5, 0.5]),
    }
    for name, loss_fn in gradcheck_losses.items():
        test_gradcheck(loss_fn, name)