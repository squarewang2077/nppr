# scripts/eval_adv_examples.py — Evaluate trained classifier: PGD / CW / AutoAttack
#
# Reports:
#   PGD Acc — Projected Gradient Descent adversarial accuracy
#   CW Acc  — Carlini–Wagner (L-inf variant) adversarial accuracy
#   AA Acc  — AutoAttack adversarial accuracy
#
# Usage example:
#   python scripts/eval_adv_examples.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/adv_training/resnet18_cifar10.pth \
#       --norm linf --epsilon 0.03137 --pgd_steps 20

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
from src.adv_attacker import pgd_attack, cw_attack
from utils.evaluator import Evaluator
from pathlib import Path


def autoattack_wrapper(model, x, y, norm, epsilon, version="standard", **_):
    """
    Thin wrapper around the AutoAttack library so it fits the
    attacker(model, x, y, **kwargs) -> x_adv interface expected by
    Evaluator.evaluate_adversarial().
    """
    import autoattack
    norm_str = "Linf" if norm.lower() == "linf" else "L2"
    adversary = autoattack.AutoAttack(
        model, norm=norm_str, eps=epsilon, version=version, verbose=False
    )
    return adversary.run_standard_evaluation(x, y, bs=x.size(0))


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
        description="Evaluate a trained image classifier — adversarial examples "
                    "(PGD, CW, AutoAttack)"
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

    # ---- PGD attack settings ----
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf")
    ap.add_argument("--epsilon", type=float, default=8/255)
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD / CW step size")
    ap.add_argument("--pgd_steps", type=int, default=20)

    # ---- CW attack settings ----
    ap.add_argument("--cw_steps", type=int, default=30)
    ap.add_argument("--no_cw", action="store_true", default=False,
                    help="Skip the CW adversarial evaluation")

    # ---- AutoAttack settings ----
    ap.add_argument("--aa_version", type=str, default="standard",
                    choices=["standard", "plus", "rand", "custom"])
    ap.add_argument("--no_aa", action="store_true", default=False,
                    help="Skip the AutoAttack evaluation")

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
        "norm": args.norm, "epsilon": args.epsilon,
        "pgd_steps": args.pgd_steps,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # ------------------------------------------------------------------
    # Run adversarial evaluations
    # ------------------------------------------------------------------
    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        # 1. PGD
        print(f"[1] PGD evaluation  "
              f"(norm={args.norm}, ε={args.epsilon:.4f}, α={args.alpha:.4f}, "
              f"steps={args.pgd_steps}) ...")
        _t0 = time.perf_counter()
        pgd = evaluator.evaluate_adversarial(
            attacker=pgd_attack,
            eval_name=f"PGD-{args.pgd_steps}",
            epsilon=args.epsilon, alpha=args.alpha,
            num_steps=args.pgd_steps, norm=args.norm,
        )
        _t_pgd = time.perf_counter() - _t0
        print(f"    pgd_acc={pgd['acc']*100:.2f}%  pgd_loss={pgd['loss']:.4f}"
              f"  [{_t_pgd:.1f}s]")
        results[f"{split}_pgd_acc"]  = pgd["acc"]
        results[f"{split}_pgd_loss"] = pgd["loss"]
        results[f"{split}_pgd_time"] = _t_pgd

        # 2. CW
        if not args.no_cw:
            print(f"[2] CW evaluation  "
                  f"(ε={args.epsilon:.4f}, step={args.alpha:.4f}, "
                  f"steps={args.cw_steps}) ...")
            _t0 = time.perf_counter()
            cw = evaluator.evaluate_adversarial(
                attacker=cw_attack,
                eval_name=f"CW-{args.cw_steps}",
                epsilon=args.epsilon,
                step_size=args.alpha,
                num_steps=args.cw_steps,
                num_classes=num_classes,
                device=device,
            )
            _t_cw = time.perf_counter() - _t0
            print(f"    cw_acc={cw['acc']*100:.2f}%  cw_loss={cw['loss']:.4f}"
                  f"  [{_t_cw:.1f}s]")
            results[f"{split}_cw_acc"]  = cw["acc"]
            results[f"{split}_cw_loss"] = cw["loss"]
            results[f"{split}_cw_time"] = _t_cw

        # 3. AutoAttack
        if not args.no_aa:
            print(f"[3] AutoAttack evaluation  "
                  f"(version={args.aa_version}, norm={args.norm}, "
                  f"ε={args.epsilon:.4f}) ...")
            _t0 = time.perf_counter()
            aa = evaluator.evaluate_adversarial(
                attacker=autoattack_wrapper,
                eval_name=f"AA-{args.aa_version}",
                norm=args.norm,
                epsilon=args.epsilon,
                version=args.aa_version,
            )
            _t_aa = time.perf_counter() - _t0
            print(f"    aa_acc={aa['acc']*100:.2f}%  aa_loss={aa['loss']:.4f}"
                  f"  [{_t_aa:.1f}s]")
            results[f"{split}_aa_acc"]  = aa["acc"]
            results[f"{split}_aa_loss"] = aa["loss"]
            results[f"{split}_aa_time"] = _t_aa

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {args.ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print(f"  PGD           : norm={args.norm}, ε={args.epsilon:.4f}, "
          f"α={args.alpha:.4f}, steps={args.pgd_steps}")
    if not args.no_cw:
        print(f"  CW            : norm=linf, ε={args.epsilon:.4f}, "
              f"step={args.alpha:.4f}, steps={args.cw_steps}")
    if not args.no_aa:
        print(f"  AutoAttack    : version={args.aa_version}, norm={args.norm}, "
              f"ε={args.epsilon:.4f}")
    print()

    # Timing
    _timing_rows = [
        ("pgd",        "pgd_time",  True),
        ("cw",         "cw_time",   not args.no_cw),
        ("autoattack", "aa_time",   not args.no_aa),
    ]
    print(f"  {'Eval':<14}  " + "  ".join(f"{s:>8}" for s, _ in splits))
    print("  " + "-" * (14 + 2 + 10 * len(splits)))
    for label, key, active in _timing_rows:
        if not active:
            continue
        vals = "  ".join(
            f"{results[f'{s}_{key}']:>7.1f}s"
            if f"{s}_{key}" in results else f"{'N/A':>8}"
            for s, _ in splits
        )
        print(f"  {label:<14}  {vals}")
    print()

    # Accuracy table
    header  = f"  {'Split':<6}  {'PGD Acc':>9}"
    divider = 6 + 2 + 9
    if not args.no_cw:
        header  += f"  {'CW Acc':>8}"
        divider += 10
    if not args.no_aa:
        header  += f"  {'AA Acc':>8}"
        divider += 10
    print(header)
    print("  " + "-" * divider)

    for split, _ in splits:
        row = f"  {split:<6}  {results[f'{split}_pgd_acc']*100:>8.2f}%"
        if not args.no_cw:
            row += f"  {results[f'{split}_cw_acc']*100:>7.2f}%"
        if not args.no_aa:
            row += f"  {results[f'{split}_aa_acc']*100:>7.2f}%"
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
