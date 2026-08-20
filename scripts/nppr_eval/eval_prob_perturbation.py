# scripts/eval_prob_perturbation.py — Evaluate GMM-based probabilistic robustness
#
# Purpose:
#   Test how GMM PR performance changes when the evaluation radius (epsilon)
#   differs from the radius used during GMM training.
#
# Reports:
#   PR-GMM  — Trained GMM-based probabilistic robustness
#
# Usage example:
#   python scripts/eval_prob_perturbation.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/pr_training/resnet18_cifar10.pth \
#       --gmm_path ./ckp/gmm/gmm_resnet50_cifar10.pt \
#       --gmm_epsilon 0.06274 \
#       --num_samples 32

import os
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from utils.evaluator import Evaluator
from pathlib import Path


def set_seed(seed: int = 42):
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
    ap = argparse.ArgumentParser(
        description="Evaluate GMM-based probabilistic robustness — "
                    "supports testing at a different radius than training."
    )

    # ---- Model / checkpoint ----
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet34", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_tiny", "vit_small", "convit_tiny", "convit_small",
        ], default="resnet50")
    ap.add_argument("--ckp_path", type=str,
                    default="./ckp/standard/resnet/resnet50_cifar10.pth")

    # ---- Dataset ----
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet", "svhn"],
                    default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_train", action="store_true", default=False,
                    help="Also evaluate on the training split")

    # ---- GMM PR evaluation ----
    ap.add_argument("--gmm_path", type=str,
                    default='./ckp/gmm_fitting/resnet/resnet18_on_cifar10/gmm_K3_cond(x)_decoder(nontrainable)_linf(16)_reg(none).pt',
                    help="Path to a trained GMM4PR checkpoint (.pt).")
    ap.add_argument("--gmm_epsilon", type=float, default=None,
                    help="Override the perturbation radius for GMM evaluation "
                         "(test at a different radius than training).")
    ap.add_argument("--gmm_norm", type=str, default=None,
                    choices=["linf", "l2"],
                    help="Override the perturbation norm for GMM evaluation.")
    ap.add_argument("--num_samples", type=int, default=32,
                    help="Number of perturbation samples per input (N)")

    # ---- Misc ----
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_csv", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    if not os.path.isfile(args.ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckp_path}")

    print(f"[ckp] loading from: {args.ckp_path}")
    ckpt = torch.load(args.ckp_path, map_location="cpu", weights_only=False)

    arch          = ckpt.get("arch",          args.arch)
    dataset       = ckpt.get("dataset",       args.dataset)
    training_type = ckpt.get("training_type", "unknown")
    img_size      = get_img_size(dataset, args.img_size or ckpt.get("img_size"))

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
    # Load trained GMM (required)
    # ------------------------------------------------------------------
    if args.gmm_path is None:
        raise ValueError("--gmm_path is required for GMM PR evaluation.")

    from utils.utils import load_gmm_model
    print(f"[gmm] loading from: {args.gmm_path}")
    gmm = load_gmm_model(
        args.gmm_path,
        dataset=dataset,
        device=str(device),
    )
    _gmm_feat_arch = gmm._feat_arch
    _gmm_eval_eps  = args.gmm_epsilon if args.gmm_epsilon is not None else gmm.budget["eps"]
    _gmm_eval_norm = args.gmm_norm    if args.gmm_norm    is not None else gmm.budget["norm"]
    print(f"[gmm] loaded — K={gmm.K}, latent_dim={gmm.latent_dim}, "
          f"cond_mode={gmm.cond_mode}")
    print(f"[gmm] feat_arch={_gmm_feat_arch}  (classifier arch={arch})")
    print(f"[gmm] training budget: eps={gmm.budget['eps']:.4f}, norm={gmm.budget['norm']}")
    print(f"[gmm] eval budget:     eps={_gmm_eval_eps:.4f}, norm={_gmm_eval_norm}")

    criterion = nn.CrossEntropyLoss()

    results = {
        "arch": arch, "dataset": dataset,
        "training_type": training_type, "ckp_path": args.ckp_path,
        "gmm_train_eps": gmm.budget["eps"], "gmm_train_norm": gmm.budget["norm"],
        "gmm_eval_eps": _gmm_eval_eps, "gmm_eval_norm": _gmm_eval_norm,
        "num_samples": args.num_samples,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # ------------------------------------------------------------------
    # Run GMM PR evaluation
    # ------------------------------------------------------------------
    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        print(f"[PR-GMM] N={args.num_samples}, feat={_gmm_feat_arch}, "
              f"ε={_gmm_eval_eps:.4f}, norm={_gmm_eval_norm} ...")
        _t0 = time.perf_counter()
        pr_gmm_res = evaluator.evaluate_pr_gmm(
            gmm=gmm,
            eval_name="PR-GMM",
            num_samples=args.num_samples,
            epsilon=args.gmm_epsilon,
            norm=args.gmm_norm,
        )
        _t_pr_gmm = time.perf_counter() - _t0
        print(f"    pr_gmm={pr_gmm_res['pr']*100:.2f}%  "
              f"D={pr_gmm_res['stats']['D_proxy']:.3e}"
              f"  [{_t_pr_gmm:.1f}s]")
        results[f"{split}_pr_gmm"] = pr_gmm_res["pr"]
        results[f"{split}_pr_gmm_time"] = _t_pr_gmm
        for k, v in pr_gmm_res.get("stats", {}).items():
            if isinstance(v, (int, float)):
                results[f"{split}_pr_gmm_{k}"] = v

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {args.ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print(f"  PR GMM        : N={args.num_samples}, K={gmm.K}, "
          f"cond={gmm.cond_mode}, feat={_gmm_feat_arch}")
    print(f"                  train budget: eps={gmm.budget['eps']:.4f}, "
          f"norm={gmm.budget['norm']}")
    print(f"                  eval  budget: eps={_gmm_eval_eps:.4f}, "
          f"norm={_gmm_eval_norm}")
    print()

    # Timing
    print(f"  {'Eval':<18}  " + "  ".join(f"{s:>8}" for s, _ in splits))
    print("  " + "-" * (18 + 2 + 10 * len(splits)))
    vals = "  ".join(
        f"{results[f'{s}_pr_gmm_time']:>7.1f}s"
        if f"{s}_pr_gmm_time" in results else f"{'N/A':>8}"
        for s, _ in splits
    )
    print(f"  {'pr-gmm':<18}  {vals}")
    print()

    # Accuracy table
    header  = f"  {'Split':<6}  {'PR-GMM':>8}"
    divider = 6 + 2 + 8
    print(header)
    print("  " + "-" * divider)
    for split, _ in splits:
        row = f"  {split:<6}  {results[f'{split}_pr_gmm']*100:>7.2f}%"
        print(row)

    # ------------------------------------------------------------------
    # Optional CSV save
    # ------------------------------------------------------------------
    if args.save_csv:
        csv_dir = os.path.dirname(os.path.abspath(args.save_csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pd.DataFrame([results]).to_csv(args.save_csv, index=False)
        print(f"\n[save] results written to: {args.save_csv}")


if __name__ == "__main__":
    main()
