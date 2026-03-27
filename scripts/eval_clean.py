# scripts/eval_clean.py — Evaluate trained classifier: clean (standard) accuracy
#
# Usage example:
#   python scripts/eval_clean.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/standard_training/resnet18_cifar10.pth

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
        description="Evaluate a trained image classifier — clean accuracy"
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
    ap.add_argument("--eval_train", action="store_true", default=False,
                    help="Also evaluate on the training split")

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

    criterion = nn.CrossEntropyLoss()

    results = {
        "arch": arch, "dataset": dataset,
        "training_type": training_type, "ckp_path": args.ckp_path,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # ------------------------------------------------------------------
    # Run clean evaluation
    # ------------------------------------------------------------------
    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        print("[1] Standard (clean) evaluation ...")
        _t0 = time.perf_counter()
        clean = evaluator.evaluate_standard(eval_name="clean")
        _t_clean = time.perf_counter() - _t0
        print(f"    clean_acc={clean['acc']*100:.2f}%  clean_loss={clean['loss']:.4f}"
              f"  [{_t_clean:.1f}s]")
        results[f"{split}_clean_acc"]  = clean["acc"]
        results[f"{split}_clean_loss"] = clean["loss"]
        results[f"{split}_clean_time"] = _t_clean

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {args.ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print()

    print(f"  {'Split':<6}  {'Clean Acc':>10}  {'Clean Loss':>11}")
    print("  " + "-" * (6 + 2 + 10 + 2 + 11))
    for split, _ in splits:
        print(f"  {split:<6}  "
              f"{results[f'{split}_clean_acc']*100:>9.2f}%  "
              f"{results[f'{split}_clean_loss']:>11.4f}")

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
