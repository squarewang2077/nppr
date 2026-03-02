# smoke_test.py - Quick end-to-end test for adv_pgd, trades, and pr training paths
# Runs only a few batches per mode (no full epoch) to verify everything wires up correctly.

import sys
import torch
import torch.nn as nn
import torch.optim as optim

from model_zoo import build_model
from utils.data_preprocessing import get_dataset, get_img_size
from utils.adv_attacker import pgd_at_loss, trades_loss
from utils.pr_generator import pr_generator
from config_fitting import build_sigma_list

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "cifar10"
DATA_ROOT = "./dataset"
ARCH = "resnet18"
BATCH_SIZE = 32
NUM_BATCHES = 3   # only test this many batches per mode
EPSILON = 8 / 255
ALPHA = 2 / 255
NUM_STEPS = 3     # reduced for speed

def get_loader():
    img_size = get_img_size(DATASET)
    train_set, num_classes = get_dataset(DATASET, DATA_ROOT, True, img_size)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    return loader, num_classes

def make_model(num_classes):
    model = build_model(ARCH, num_classes, DATASET).to(DEVICE)
    return model

# ------------------------------------------------------------------
def test_adv_pgd(norm="linf"):
    print("\n" + "="*60)
    print(f"TEST: adv_pgd (PGD-AT) — {norm}")
    print("="*60)
    loader, num_classes = get_loader()
    model = make_model(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for i, (x, y) in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss, x_adv = pgd_at_loss(model, x, y, EPSILON, ALPHA, NUM_STEPS, criterion, norm=norm)
        loss.backward()
        optimizer.step()
        delta = x_adv - x
        delta_linf = delta.abs().max().item()
        delta_l2   = delta.view(x.size(0), -1).norm(p=2, dim=1).max().item()
        print(f"  batch {i+1}/{NUM_BATCHES}  loss={loss.item():.4f}  "
              f"delta_linf={delta_linf:.4f}  delta_l2={delta_l2:.4f}  (budget={EPSILON:.4f})")

    print(f"PASS: adv_pgd ({norm})")

# ------------------------------------------------------------------
def test_trades(norm="linf"):
    print("\n" + "="*60)
    print(f"TEST: trades (TRADES) — {norm}")
    print("="*60)
    loader, num_classes = get_loader()
    model = make_model(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    BETA = 6.0

    model.train()
    for i, (x, y) in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss, x_adv = trades_loss(model, x, y, EPSILON, ALPHA, NUM_STEPS, BETA, criterion, norm=norm)
        loss.backward()
        optimizer.step()
        delta = x_adv - x
        delta_linf = delta.abs().max().item()
        delta_l2   = delta.view(x.size(0), -1).norm(p=2, dim=1).max().item()
        print(f"  batch {i+1}/{NUM_BATCHES}  loss={loss.item():.4f}  "
              f"delta_linf={delta_linf:.4f}  delta_l2={delta_l2:.4f}  (budget={EPSILON:.4f})")

    print(f"PASS: trades ({norm})")

# ------------------------------------------------------------------
def test_pr():
    print("\n" + "="*60)
    print("TEST: pr (PR / Bayesian)")
    print("="*60)
    loader, num_classes = get_loader()
    model = make_model(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    K = 2
    sigma_list = build_sigma_list(epsilon=EPSILON, K=K, mode_type="linear")
    pr_config = dict(
        norm="linf", epsilon=EPSILON,
        beta_mix=0.5, kappa=1.0,
        K=K, sigma_list=sigma_list, fisher_damping=1e-4, tau=1.0,
        num_samples=4, noise_scale=1.0,
    )
    generator_kwargs = {k: v for k, v in pr_config.items()}

    model.train()
    for i, (x, y) in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        x_adv = pr_generator(model, x, y, **generator_kwargs)
        B, N = x_adv.shape[0], x_adv.shape[1]
        x_adv_flat = x_adv.view(B * N, *x_adv.shape[2:])
        y_rep = y.repeat_interleave(N)

        logits = model(x_adv_flat)
        loss = criterion(logits, y_rep)
        loss.backward()
        optimizer.step()

        delta_max = (x_adv - x.unsqueeze(1)).abs().max().item()
        print(f"  batch {i+1}/{NUM_BATCHES}  loss={loss.item():.4f}  "
              f"x_adv shape={tuple(x_adv.shape)}  "
              f"delta_max={delta_max:.4f}  (budget={EPSILON:.4f})")

    print("PASS: pr")

# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    passed = []
    failed = []

    for name, fn in [
        ("adv_pgd (linf)",  lambda: test_adv_pgd("linf")),
        ("adv_pgd (l2)",    lambda: test_adv_pgd("l2")),
        ("trades  (linf)",  lambda: test_trades("linf")),
        ("trades  (l2)",    lambda: test_trades("l2")),
        ("pr",              test_pr),
    ]:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            print(f"\nFAIL: {name}  ->  {type(e).__name__}: {e}")
            failed.append(name)

    print("\n" + "="*60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if passed:
        print(f"  PASSED: {', '.join(passed)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
