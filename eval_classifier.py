# eval_classifier.py - Evaluate trained image classifiers
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15
#   numpy, pandas
#
# Loads a trained checkpoint and runs three evaluations:
#   1. Standard (clean) evaluation
#   2. PGD adversarial evaluation  (utils/adv_attacker.py)
#   3. PR  adversarial evaluation  (utils/pr_generator.py)
#
# Usage example:
#   python eval_classifier.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/pr_training/resnet18_cifar10.pth \
#       --norm linf --epsilon 0.03137 --pgd_steps 20

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

from model_zoo import build_model
from utils.data_preprocessing import get_dataset, get_img_size
from utils.adv_attacker import pgd_attack
from utils.pr_generator import pr_generator
from utils.evaluator import Evaluator
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
    ap.add_argument("--ckp_path", type=str, default='ckp/tmp/cifar10/resnet18_cifar10.pth',
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
    ap.add_argument("--beta_mix", type=float, default=1)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=3,
                    help="Number of GMM components for PR attack")
    ap.add_argument("--sigma_dist_type", type=str, default="geometric",
                    choices=["linear", "geometric", "full"])
    ap.add_argument("--fisher_damping", type=float, default=1e-7)
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--num_samples", type=int, default=32,
                    help="Number of perturbation samples per input (N) for PR attack")

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
        parts = Path(args.ckp_path).parts
        if "standard_training" in parts:
            training_type = "standard_training"
        elif "adv_training" in parts:
            training_type = "adv_training"
        elif "pr_training" in parts:
            training_type = "pr_training"

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
        "pgd_steps": args.pgd_steps, "num_samples": args.num_samples,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        # 1. Standard clean evaluation
        print("[1/3] Standard (clean) evaluation ...")
        clean = evaluator.evaluate_standard(eval_name="clean")
        print(f"      clean_acc={clean['acc']*100:.2f}%  clean_loss={clean['loss']:.4f}")
        results[f"{split}_clean_acc"]  = clean["acc"]
        results[f"{split}_clean_loss"] = clean["loss"]

        # 2. PGD adversarial evaluation
        print(f"[2/3] PGD evaluation  "
              f"(norm={args.norm}, ε={args.epsilon:.4f}, α={args.alpha:.4f}, "
              f"steps={args.pgd_steps}) ...")
        pgd = evaluator.evaluate_adversarial(
            attacker=pgd_attack,
            eval_name=f"PGD-{args.pgd_steps}",
            epsilon=args.epsilon, alpha=args.alpha,
            num_steps=args.pgd_steps, norm=args.norm,
        )
        print(f"      pgd_acc={pgd['acc']*100:.2f}%  pgd_loss={pgd['loss']:.4f}")
        results[f"{split}_pgd_acc"]  = pgd["acc"]
        results[f"{split}_pgd_loss"] = pgd["loss"]

        # 3. PR adversarial evaluation
        print(f"[3/3] PR evaluation  (N={args.num_samples}, K={args.K}) ...")
        pr = evaluator.evaluate_pr(
            pr_generator=pr_generator,
            eval_name=f"PR-{args.num_samples}",
            **pr_config,
        )
        print(f"      pr={pr['pr']*100:.2f}%  "
              f"D={pr['stats']['D_proxy']:.3e}  Hpi={pr['stats']['pi_entropy']:.3f}")
        results[f"{split}_pr"] = pr["pr"]
        for k, v in pr.get("stats", {}).items():
            results[f"{split}_pr_{k}"] = v

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
    print(f"  PR            : N={args.num_samples}, K={args.K}")
    print()
    print(f"  {'Split':<6}  {'Clean Acc':>10}  {'Clean Loss':>11}  "
          f"{'PGD Acc':>9}  {'PGD Loss':>9}  {'PR':>8}")
    print("  " + "-" * 62)
    for split, _ in splits:
        print(f"  {split:<6}  "
              f"{results[f'{split}_clean_acc']*100:>9.2f}%  "
              f"{results[f'{split}_clean_loss']:>11.4f}  "
              f"{results[f'{split}_pgd_acc']*100:>8.2f}%  "
              f"{results[f'{split}_pgd_loss']:>9.4f}  "
              f"{results[f'{split}_pr']*100:>7.2f}%")

    # ------------------------------------------------------------------
    # Optional CSV save
    # ------------------------------------------------------------------
    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        pd.DataFrame([results]).to_csv(args.save_csv, index=False)
        print(f"\n[save] results written to: {args.save_csv}")


if __name__ == "__main__":
    main()
