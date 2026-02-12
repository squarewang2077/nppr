# fit_classifiers.py - Train image classifiers from scratch
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15
#   numpy
#   tqdm
#
# This script relies on:
#   - `utils/data_preprocessing.py` for dataset loading and preprocessing utilities
#   - `utils/ad_attacker.py` for adversarial training losses (PGD-AT, TRADES)
#   - `model_zoo/` for building various model architectures

import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model_zoo import build_model
from utils.data_preprocessing import get_dataset, get_img_size
from utils.ad_attacker import pgd_at_loss, trades_loss


def set_seed(seed: int = 42):
    """Make training as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    """Return accuracy on `loader`."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# ------------------------------ Training Loops ------------------------------

def train_one_epoch(model, loader, optimizer, device, criterion, epoch=None, total_epochs=None):
    """Standard training loop."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Train Epoch [{epoch}/{total_epochs}]" if epoch else "Training", leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        avg_loss = running_loss / total_samples

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / len(loader.dataset)


def train_one_epoch_adv(model, loader, optimizer, device, criterion, adv_config, epoch=None, total_epochs=None):
    """
    Adversarial training loop (outer loop).
    Inner loop (attack generation) is handled by ad_attacker functions.
    """
    running_loss = 0.0
    total_samples = 0

    adv_type = adv_config["type"]
    epsilon = adv_config["epsilon"]
    alpha = adv_config["alpha"]
    num_steps = adv_config["num_steps"]
    beta = adv_config.get("beta", 6.0)  # TRADES only

    pbar = tqdm(loader, desc=f"Adv Train [{epoch}/{total_epochs}]" if epoch else "Adv Training", leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Compute adversarial loss (inner loop + outer loss)
        if adv_type == "pgd":
            loss, _ = pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion)
        elif adv_type == "trades":
            loss, _ = trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        avg_loss = running_loss / total_samples

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / len(loader.dataset)


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()

    # Dataset & model
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_b_16"
    ], default="resnet18")
    ap.add_argument("--img_size", type=int, default=None,
                    help="Input image size. If None, uses dataset-native size (32 for CIFAR, 64 for TinyImageNet)")

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    # Adversarial training
    ap.add_argument("--adv_training", choices=["none", "pgd", "trades"], default="none",
                    help="Adversarial training method: none (standard), pgd (PGD-AT), trades (TRADES)")
    ap.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation budget (L-inf)")
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD step size")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="Number of PGD steps")
    ap.add_argument("--beta", type=float, default=6.0,
                    help="TRADES regularization weight")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/standard_training",
                    help="Directory to save best checkpoint")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = get_img_size(args.dataset, args.img_size)
    is_adv_training = args.adv_training != "none"

    # Print config
    print(f"[config] dataset={args.dataset}, arch={args.arch}")
    print(f"[config] img_size={img_size}")
    if is_adv_training:
        print(f"[config] adv_training={args.adv_training}, epsilon={args.epsilon:.4f}, "
              f"alpha={args.alpha:.4f}, num_steps={args.num_steps}")
        if args.adv_training == "trades":
            print(f"[config] beta={args.beta}")

    # Build datasets/loaders
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True, img_size)
    test_set, _ = get_dataset(args.dataset, args.data_root, False, img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Build model
    model = build_model(args.arch, num_classes)
    model.to(device)

    # Optional DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Adversarial config
    adv_config = {
        "type": args.adv_training,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "num_steps": args.num_steps,
        "beta": args.beta,
    }

    # Output path
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{args.arch.lower()}_{args.dataset.lower()}.pth")
    print(f"[save] best checkpoint will be written to: {out_path}")

    # Train
    best_acc = 0.0
    start = time.time()

    for ep in range(1, args.epochs + 1):
        if is_adv_training:
            train_loss = train_one_epoch_adv(model, train_loader, optimizer, device, criterion, adv_config,
                                             epoch=ep, total_epochs=args.epochs)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                         epoch=ep, total_epochs=args.epochs)

        acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"[{ep:03d}/{args.epochs}] loss={train_loss:.4f}  val_acc={acc*100:.2f}%  lr={scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint
        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "epoch": ep,
                "arch": args.arch,
                "dataset": args.dataset,
                "img_size": img_size,
                "adv_training": args.adv_training,
                "model_state": model.state_dict()
            }
            if is_adv_training:
                ckpt["adv_config"] = adv_config
            torch.save(ckpt, out_path)
            print(f"  -> saved best to {out_path}")

    elapsed = time.time() - start
    print(f"Done. Best val acc: {best_acc*100:.2f}%  (time: {elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
