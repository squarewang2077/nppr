# scripts/eval_corruptions.py — Evaluate classifier robustness to image corruptions
#
# Applies every corruption in utils/corrupter.py across severities 1-5 and
# reports the top-1 accuracy under each (corruption, severity) pair.
#
# Usage example:
#   python scripts/eval_corruptions.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/standard_training/resnet18_cifar10.pth \
#       --corruptions salt_pepper motion_blur \
#       --severities 1 3 5

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
from utils.corrupter import CORRUPTION_FNS, apply_corruption_batch
from pathlib import Path


def corruption_attacker(model, x, y, corruption_name, severity):
    """Wrap apply_corruption_batch as a pointwise attacker for Evaluator."""
    return apply_corruption_batch(x, corruption_name, severity)


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
        description="Evaluate a trained classifier under image corruptions"
    )

    # ---- Model / checkpoint ----
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_b_16",
        ], default="resnet18")
    ap.add_argument("--ckp_path", type=str,
                    default="ckp/tmp/cifar10/resnet18_cifar10.pth")

    # ---- Dataset ----
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"],
                    default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_train", action="store_true", default=False)

    # ---- Corruption settings ----
    ap.add_argument("--corruptions", nargs="+",
                    default=list(CORRUPTION_FNS.keys()),
                    choices=list(CORRUPTION_FNS.keys()),
                    help="Corruptions to evaluate (default: all)")
    ap.add_argument("--severities", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5],
                    help="Severity levels 1-5 to evaluate (default: all)")

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

    # Validate severity values
    for s in args.severities:
        if not (1 <= s <= 5):
            raise ValueError(f"Severity must be in [1, 5], got {s}")

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    results_meta = {
        "arch": arch, "dataset": dataset,
        "training_type": training_type, "ckp_path": args.ckp_path,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # acc_table[split][corruption][severity] = acc
    acc_table = {split: {} for split, _ in splits}

    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        # 1. Clean baseline
        print("[clean] Standard (clean) evaluation ...")
        _t0 = time.perf_counter()
        clean = evaluator.evaluate_standard(eval_name="clean")
        _t_clean = time.perf_counter() - _t0
        print(f"  clean_acc={clean['acc']*100:.2f}%  [{_t_clean:.1f}s]")
        results_meta[f"{split}_clean_acc"] = clean["acc"]

        # 2. Corruptions
        for corruption in args.corruptions:
            acc_table[split][corruption] = {}
            for severity in args.severities:
                eval_name = f"{corruption}-s{severity}"
                _t0 = time.perf_counter()
                res = evaluator.evaluate_adversarial(
                    attacker=corruption_attacker,
                    eval_name=eval_name,
                    corruption_name=corruption,
                    severity=severity,
                )
                _t = time.perf_counter() - _t0
                acc = res["acc"]
                acc_table[split][corruption][severity] = acc
                print(f"  [{eval_name}] acc={acc*100:.2f}%  [{_t:.1f}s]")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {args.ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print(f"  Corruptions   : {', '.join(args.corruptions)}")
    print(f"  Severities    : {args.severities}")
    print()

    for split, loader in splits:
        print(f"  Split: {split}")
        print(f"  {'Clean Acc':>10}: {results_meta[f'{split}_clean_acc']*100:.2f}%")
        print()

        # Header: corruption name + one column per severity
        sev_header = "".join(f"  {f'S{s}':>6}" for s in args.severities)
        print(f"  {'Corruption':<16}{sev_header}")
        print("  " + "-" * (16 + 8 * len(args.severities)))

        for corruption in args.corruptions:
            row = f"  {corruption:<16}"
            for severity in args.severities:
                acc = acc_table[split][corruption].get(severity, float("nan"))
                row += f"  {acc*100:>5.2f}%"
            print(row)
        print()

    # ------------------------------------------------------------------
    # Optional CSV save
    # ------------------------------------------------------------------
    if args.save_csv:
        rows = []
        for split, _ in splits:
            row = dict(results_meta)
            row["split"] = split
            row[f"clean_acc"] = results_meta[f"{split}_clean_acc"]
            for corruption in args.corruptions:
                for severity in args.severities:
                    acc = acc_table[split][corruption].get(severity, float("nan"))
                    row[f"{corruption}_s{severity}_acc"] = acc
            rows.append(row)

        csv_dir = os.path.dirname(os.path.abspath(args.save_csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_csv, index=False)
        print(f"[save] results written to: {args.save_csv}")


if __name__ == "__main__":
    main()
