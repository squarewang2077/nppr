# eva_classifier.py - Evaluate trained image classifiers
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15
#   numpy, tqdm, pandas
#
# Loads a trained checkpoint and runs three evaluations:
#   1. Standard (clean) evaluation
#   2. PGD adversarial evaluation  (utils/adv_attacker.py)
#   3. PR  adversarial evaluation  (utils/pr_generator.py)
#
# Usage example:
#   python eva_classifier.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/pr_training/resnet18_cifar10.pth \
#       --norm linf --epsilon 0.03137 --pgd_steps 20 --pr_reduce worst

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pandas as pd

from model_zoo import build_model
from utils.data_preprocessing import get_dataset, get_img_size
from utils.adv_attacker import pgd_attack
from utils.pr_generator import pr_generator
from config_fitting import build_sigma_list
from pathlib import Path

def set_seed(seed: int = 42):
    """Make evaluation as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# ------------------------------------------------------------------
#                           Evaluation Functions
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """
    Standard (clean) evaluation.

    Returns:
        acc: clean accuracy (float)
        avg_loss: mean CE loss, or None if criterion is not provided
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Clean eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)

        if criterion is not None:
            running_loss += criterion(logits, y).item() * y.size(0)

        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    acc = total_correct / total_samples
    avg_loss = running_loss / total_samples if criterion else None
    return acc, avg_loss


def evaluate_with_pgd_attack(model, loader, device, criterion,
                              epsilon, alpha, num_steps, norm="linf"):
    """
    Evaluate robust accuracy / loss under PGD attack.

    Args:
        model      : classifier (expected in [0,1] input space)
        loader     : dataloader yielding (x, y)
        device     : torch device
        criterion  : loss function (e.g. CrossEntropyLoss)
        epsilon    : perturbation budget radius
        alpha      : PGD step size
        num_steps  : number of PGD iterations
        norm       : "linf" or "l2"

    Returns:
        rob_acc  : robust accuracy (float)
        rob_loss : robust loss    (float)
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    running_loss  = 0.0

    pbar = tqdm(loader, desc=f"PGD eval ({norm}, ε={epsilon:.4f}, steps={num_steps})", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # pgd_attack requires grad internally; x_adv is returned detached
        x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps, norm=norm)

        with torch.no_grad():
            logits = model(x_adv)
            loss   = criterion(logits, y)

        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        running_loss  += loss.item() * y.size(0)

    rob_acc  = total_correct / max(1, total_samples)
    rob_loss = running_loss  / max(1, total_samples)
    return rob_acc, rob_loss


def evaluate_with_pr_attack(model, loader, device, criterion,
                             pr_generator, generator_kwargs=None,
                             reduce="worst"):
    """
    Evaluate robust accuracy / loss using a probabilistic (PR / Bayesian) attack.

    Args:
        model             : classifier
        loader            : dataloader yielding (x, y)
        device            : torch device
        criterion         : loss function (default: CrossEntropyLoss)
        pr_generator      : function(model, x, y, ...) -> (x_adv, stats)
                            x_adv shape: (B,N,C,H,W) or (B,C,H,W)
        generator_kwargs  : dict forwarded to pr_generator (excluding "type")
        reduce            : how to aggregate N samples per input:
                              "mean"  — average probability over N samples
                              "worst" — pick the sample that maximises loss
                              "none"  — treat every sample independently

    Returns:
        rob_acc  : robust accuracy (float)
        rob_loss : robust loss    (float)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    if generator_kwargs is None:
        generator_kwargs = {}

    # Strip "type" key if present (not a valid kwarg for pr_generator)
    generator_kwargs = {k: v for k, v in generator_kwargs.items() if k != "type"}

    model.eval()

    total_correct = 0
    total_samples = 0
    running_loss  = 0.0

    pbar = tqdm(loader, desc=f"PR eval (reduce={reduce})", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # pr_generator needs gradients internally and leaves model in train()
        x_adv, _ = pr_generator(model, x, y, **generator_kwargs)
        model.eval()  # restore eval mode after pr_generator

        # --- 4-D case: single sample per input (B,C,H,W) ---
        if x_adv.dim() == 4:
            with torch.no_grad():
                logits = model(x_adv)
                loss   = criterion(logits, y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
            running_loss  += loss.item() * y.size(0)
            continue

        if x_adv.dim() != 5:
            raise ValueError(f"x_adv must be 4-D or 5-D, got {tuple(x_adv.shape)}")

        B, N = x_adv.shape[:2]
        x_adv_flat = x_adv.view(B * N, *x_adv.shape[2:])   # (B*N, C, H, W)
        y_rep      = y.repeat_interleave(N)                  # (B*N,)

        with torch.no_grad():
            logits    = model(x_adv_flat)                    # (B*N, C)
            loss_each = F.cross_entropy(logits, y_rep, reduction="none")  # (B*N,)
        loss_each = loss_each.view(B, N)                     # (B, N)

        # --- robust loss aggregation ---
        if reduce == "mean":
            loss_per = loss_each.mean(dim=1)                 # (B,)
        elif reduce == "worst":
            loss_per = loss_each.max(dim=1).values           # (B,)
        elif reduce == "none":
            loss_per = loss_each.view(-1)                    # (B*N,)
        else:
            raise ValueError("reduce must be 'mean', 'worst', or 'none'")

        running_loss += loss_per.sum().item()

        # --- robust prediction aggregation ---
        logits = logits.view(B, N, -1)                       # (B, N, C)

        if reduce == "mean":
            preds = torch.softmax(logits, dim=-1).mean(dim=1).argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += B
        elif reduce == "worst":
            idx   = loss_each.argmax(dim=1)
            preds = logits[torch.arange(B, device=device), idx].argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += B
        elif reduce == "none":
            preds = logits.view(B * N, -1).argmax(dim=1)
            total_correct += (preds == y_rep).sum().item()
            total_samples += B * N

    rob_acc  = total_correct / max(1, total_samples)
    rob_loss = running_loss  / max(1, total_samples)
    return rob_acc, rob_loss


# ------------------------------------------------------------------
#                           Main Function
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained image classifier")

    # ---- Model / checkpoint ----
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_b_16",
    ], default="resnet18",
        help="Architecture (overridden by value stored in checkpoint if present)")
    ap.add_argument("--ckp_path", type=str, default='ckp/standard_training/resnet/resnet18_cifar10.pth',
                    help="Direct path to .pth checkpoint file. "
                         "If omitted, resolved as <ckp_dir>/<arch>_<dataset>.pth")
 
    # ---- Dataset ----
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"],
                    default="cifar10",
                    help="Dataset (overridden by value stored in checkpoint if present)")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--img_size", type=int, default=None,
                    help="Override image size (default: inferred from dataset / checkpoint)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_train", action="store_true", default=False,
                    help="Also evaluate on the training split")

    # ---- PGD attack settings ----
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Perturbation norm for both PGD and PR attacks")
    ap.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation budget radius")
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD step size")
    ap.add_argument("--pgd_steps", type=int, default=20,
                    help="Number of PGD iterations")

    # ---- PR attack settings ----
    ap.add_argument("--beta_mix", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=2,
                    help="Number of GMM components for PR attack")
    ap.add_argument("--sigma_dist_type", type=str, default="linear",
                    choices=["linear", "geometric", "full"])
    ap.add_argument("--fisher_damping", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--num_samples", type=int, default=32,
                    help="Number of perturbation samples per input (N) for PR attack")
    ap.add_argument("--pr_reduce", choices=["mean", "worst", "none"], default="worst",
                    help="Aggregation method over N PR samples")

    # ---- Misc ----
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to save the results table as a CSV file")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Resolve and load checkpoint
    # ------------------------------------------------------------------
    if args.ckp_path is not None:
        ckp_path = args.ckp_path
    else:
        raise ValueError("ckp_path must be provided explicitly (no default resolution)")

    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

    print(f"[ckp] loading from: {ckp_path}")
    ckpt = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # Prefer metadata stored inside the checkpoint; fall back to CLI args
    arch          = ckpt.get("arch",          args.arch)
    dataset       = ckpt.get("dataset",       args.dataset)
    training_type = ckpt.get("training_type", "unknown")
    img_size      = get_img_size(dataset, args.img_size or ckpt.get("img_size"))

    # recheck the training_type if it is not stored in the checkpoint, by looking at the ckp_path structure
    if training_type == "unknown":
        if Path(args.ckp_path).parts[1] == "standard_training":
            training_type = "standard_training"
        elif Path(args.ckp_path).parts[1] == "adv_training":
            training_type = "adv_training"
        elif Path(args.ckp_path).parts[1] == "pr_training":
            training_type = "pr_training"   
        else:
            training_type = "unknown"

    print(f"[ckp] arch={arch}, dataset={dataset}, img_size={img_size}, "
          f"training_type={training_type}, epoch={ckpt.get('epoch', '?')}")

    # ------------------------------------------------------------------
    # Build datasets and loaders
    # ------------------------------------------------------------------
    test_set, num_classes = get_dataset(dataset, args.data_root, False, img_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    train_loader = None
    if args.eval_train:
        train_set, _ = get_dataset(dataset, args.data_root, True, img_size, augment=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Build and load model
    # ------------------------------------------------------------------
    model = build_model(arch, num_classes, dataset)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    print(f"[model] {arch} loaded — {num_classes} classes on {device}")

    # ------------------------------------------------------------------
    # Shared criterion and PR config
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    sigma_list = build_sigma_list(epsilon=args.epsilon, K=args.K,
                                  mode_type=args.sigma_dist_type)
    pr_config = {
        "norm":           args.norm,
        "epsilon":        args.epsilon,
        "beta_mix":       args.beta_mix,
        "kappa":          args.kappa,
        "K":              args.K,
        "sigma_list":     sigma_list,
        "fisher_damping": args.fisher_damping,
        "tau":            args.tau,
        "noise_scale":    args.noise_scale,
        "num_samples":    args.num_samples,
    }

    # ------------------------------------------------------------------
    # Run evaluations across requested splits
    # ------------------------------------------------------------------
    results = {
        "arch": arch, "dataset": dataset,
        "training_type": training_type, "ckp_path": ckp_path,
        "norm": args.norm, "epsilon": args.epsilon,
        "pgd_steps": args.pgd_steps, "pr_reduce": args.pr_reduce,
        "num_samples": args.num_samples,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        # 1. Standard clean evaluation
        print("[1/3] Standard (clean) evaluation ...")
        clean_acc, clean_loss = evaluate(model, loader, device, criterion)
        print(f"      clean_acc={clean_acc*100:.2f}%  clean_loss={clean_loss:.4f}")
        results[f"{split}_clean_acc"]  = clean_acc
        results[f"{split}_clean_loss"] = clean_loss

        # 2. PGD adversarial evaluation
        print(f"[2/3] PGD evaluation  "
              f"(norm={args.norm}, ε={args.epsilon:.4f}, α={args.alpha:.4f}, "
              f"steps={args.pgd_steps}) ...")
        pgd_acc, pgd_loss = evaluate_with_pgd_attack(
            model, loader, device, criterion,
            epsilon=args.epsilon, alpha=args.alpha,
            num_steps=args.pgd_steps, norm=args.norm,
        )
        print(f"      pgd_acc={pgd_acc*100:.2f}%  pgd_loss={pgd_loss:.4f}")
        results[f"{split}_pgd_acc"]  = pgd_acc
        results[f"{split}_pgd_loss"] = pgd_loss

        # 3. PR adversarial evaluation
        print(f"[3/3] PR evaluation  "
              f"(reduce={args.pr_reduce}, N={args.num_samples}, K={args.K}) ...")
        pr_acc, pr_loss = evaluate_with_pr_attack(
            model, loader, device, criterion,
            pr_generator, generator_kwargs=pr_config,
            reduce=args.pr_reduce,
        )
        print(f"      pr_acc={pr_acc*100:.2f}%  pr_loss={pr_loss:.4f}")
        results[f"{split}_pr_acc"]  = pr_acc
        results[f"{split}_pr_loss"] = pr_loss

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print(f"  PGD           : norm={args.norm}, ε={args.epsilon:.4f}, "
          f"α={args.alpha:.4f}, steps={args.pgd_steps}")
    print(f"  PR            : reduce={args.pr_reduce}, N={args.num_samples}, K={args.K}")
    print()
    print(f"  {'Split':<6}  {'Clean Acc':>10}  {'Clean Loss':>11}  "
          f"{'PGD Acc':>9}  {'PGD Loss':>9}  {'PR Acc':>8}  {'PR Loss':>8}")
    print("  " + "-" * 73)
    for split, _ in splits:
        print(f"  {split:<6}  "
              f"{results[f'{split}_clean_acc']*100:>9.2f}%  "
              f"{results[f'{split}_clean_loss']:>11.4f}  "
              f"{results[f'{split}_pgd_acc']*100:>8.2f}%  "
              f"{results[f'{split}_pgd_loss']:>9.4f}  "
              f"{results[f'{split}_pr_acc']*100:>7.2f}%  "
              f"{results[f'{split}_pr_loss']:>8.4f}")

    # ------------------------------------------------------------------
    # Optional CSV save
    # ------------------------------------------------------------------
    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        pd.DataFrame([results]).to_csv(args.save_csv, index=False)
        print(f"\n[save] results written to: {args.save_csv}")


if __name__ == "__main__":
    main()
