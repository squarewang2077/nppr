#!/usr/bin/env python3
# scripts/smoke_test_losses.py - Shape/arity smoke test for the loss modules.
#
# Runs one forward+backward of every training loss in src/ on a tiny random
# input, checking that each returns what its module header promises and that
# gradients actually reach the parameters. This is a wiring check, not a
# correctness check — it will not tell you a loss is wrong, only that it runs.
#
# Usage:
#   python scripts/smoke_test_losses.py

import os
import random
import sys

import torch
import torch.nn as nn

# Runnable without PYTHONPATH (the run_exp/*.sh wrappers set it; this does not
# assume they were used).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adv_loss import pgd_at_loss, trades_loss, mart_loss, fgsm_at_loss
from src.atpr_loss import atpr_loss
from src.mma_loss import mma_loss

EPSILON = 8 / 255
ALPHA = 2 / 255
NUM_STEPS = 5


def build_model(device):
    """Tiny CIFAR-shaped CNN — small enough that every loss runs on CPU."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10),
    ).to(device)


def check(name, model, opt, x, y, fn, expect_extra=False):
    """Run one loss, assert its contract, and confirm gradients flow."""
    model.train()
    opt.zero_grad(set_to_none=True)

    out = fn(model, x, y)
    if expect_extra:
        loss, x_adv, extra = out
    else:
        loss, x_adv = out
        extra = None

    assert loss.dim() == 0, f"{name}: loss should be a scalar, got {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"{name}: loss is not finite ({loss.item()})"
    assert x_adv.shape == x.shape, \
        f"{name}: x_adv shape {tuple(x_adv.shape)} != input {tuple(x.shape)}"
    assert x_adv.min() >= -1e-6 and x_adv.max() <= 1 + 1e-6, \
        f"{name}: x_adv left [0, 1] — got [{x_adv.min():.3f}, {x_adv.max():.3f}]"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name}: no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads), f"{name}: all gradients are zero"
    opt.step()

    extra_msg = ""
    if extra is not None:
        assert extra.shape == (y.size(0),), \
            f"{name}: third return should be (B,), got {tuple(extra.shape)}"
        extra_msg = f"  curr_eps mean {extra.mean().item():.4f}"

    print(f"  {name:<12} loss {loss.item():>8.4f}  x_adv {tuple(x_adv.shape)}{extra_msg}")


def main():
    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = build_model(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    print("\nlosses returning (loss, x_adv):")
    check("pgd_at", model, opt, x, y,
          lambda m, a, b: pgd_at_loss(m, a, b, EPSILON, ALPHA, NUM_STEPS, criterion))
    check("trades", model, opt, x, y,
          lambda m, a, b: trades_loss(m, a, b, EPSILON, ALPHA, NUM_STEPS, 6.0, criterion))
    check("mart", model, opt, x, y,
          lambda m, a, b: mart_loss(m, a, b, EPSILON, ALPHA, NUM_STEPS, 5.0, criterion))
    check("fgsm_at", model, opt, x, y,
          lambda m, a, b: fgsm_at_loss(m, a, b, EPSILON, criterion))
    check("atpr", model, opt, x, y,
          lambda m, a, b: atpr_loss(m, a, b, EPSILON, ALPHA, NUM_STEPS, criterion,
                                    num_candidates=3, max_refine_steps=10))

    print("\nlosses returning (loss, x_adv, curr_eps):")
    check("mma", model, opt, x, y,
          lambda m, a, b: mma_loss(m, a, b, EPSILON, criterion,
                                   num_steps=NUM_STEPS, num_search_steps=5),
          expect_extra=True)

    print("\nOK")


if __name__ == "__main__":
    main()
