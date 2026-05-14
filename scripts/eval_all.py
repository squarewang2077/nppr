# scripts/eval_all.py - One-shot evaluation of a trained classifier.
#
# Loads a checkpoint saved by train_classifiers_pr.py or
# train_classifiers_adv.py, builds the matching test set, and reports:
#
#   * Clean test accuracy
#   * PGD-10 robust accuracy
#   * Probabilistic Robustness under random Gaussian / Uniform / Laplace
#     noise (mean fraction of N draws that stay correctly classified)
#   * Accuracy under 4 corruption methods (salt_pepper, motion_blur,
#     brightness, jpeg) at one or more severity levels
#
# A formatted summary is printed to the terminal. Optionally writes a flat
# one-row CSV with every metric for downstream aggregation.
#
# Dataset and architecture are auto-detected from the checkpoint
# (saved as `arch` / `dataset` keys by both training scripts). They can be
# overridden with --dataset / --arch.
#
# Examples:
#   python scripts/eval_all.py \
#       --ckpt ./ckp/nppr_training/pr_training/cifar10/resnet18/loc_entropy/loc_ent_eps10_L100_G0.05_N1.pth
#
#   python scripts/eval_all.py \
#       --ckpt ./ckp/nppr_training/adv_training/cifar10/resnet18/pgd10_Aug/resnet18_cifar10_adv_pgd_Aug.pth \
#       --corruption_severities 1 3 5 \
#       --save_csv ./results/eval_pgd_aug.csv

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is importable when invoked as `python scripts/eval_all.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from utils.corrupter import CORRUPTION_FNS
from scripts.train_classifiers_pr import evaluate_per_epoch, set_seed


# ------------------------------------------------------------------
#                              CLI
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a trained classifier on clean / PGD / random-PR / corruption metrics."
    )
    ap.add_argument("--ckpt", required=True, type=str,
                    help="Path to .pth checkpoint saved by the training scripts.")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override dataset (default: read from ckpt).")
    ap.add_argument("--arch", type=str, default=None,
                    help="Override arch (default: read from ckpt).")
    ap.add_argument("--img_size", type=int, default=None,
                    help="Override input size (default: ckpt or dataset default).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    # Adversarial / random-PR knobs
    ap.add_argument("--epsilon", type=float, default=8 / 255,
                    help="Perturbation budget. Default 8/255 (linf, CIFAR convention).")
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf")
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--random_n", type=int, default=8,
                    help="Number of random draws per sample for random-PR.")
    ap.add_argument("--random_dist", nargs="+",
                    choices=["gaussian", "uniform", "laplace"],
                    default=["gaussian", "uniform", "laplace"])

    # Corruption knobs
    ap.add_argument("--corruption_names", nargs="+",
                    default=list(CORRUPTION_FNS.keys()),
                    choices=list(CORRUPTION_FNS.keys()))
    ap.add_argument("--corruption_severities", nargs="+", type=int, default=[3],
                    help="Severity levels (1-5). Default: 3 (moderate).")

    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional CSV path for a flat one-row summary.")
    return ap.parse_args()


# ------------------------------------------------------------------
#                          Helpers
# ------------------------------------------------------------------

def _pct(v):
    return f"{v * 100:6.2f}%" if v is not None else "    --"


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path} "
                         f"(expected dict with key 'model_state').")
    return ckpt


def print_summary(args, ckpt, metrics):
    bar = "=" * 72
    print()
    print(bar)
    print(f"  Evaluation summary  -  {os.path.basename(args.ckpt)}")
    print(bar)
    print(f"  arch          : {ckpt.get('arch', '?')}")
    print(f"  dataset       : {ckpt.get('dataset', '?')}")
    print(f"  training_type : {ckpt.get('training_type', '?')}")
    print(f"  epoch         : {ckpt.get('epoch', '?')}")
    print(f"  num_samples   : {metrics['num_samples']}")
    print(f"  epsilon       : {args.epsilon:.5f}   norm: {args.norm}")
    print("-" * 72)
    label_w = 32  # column width for metric labels in this block
    print(f"  {'Clean test accuracy':<{label_w}}: {_pct(metrics['clean_acc'])}")
    print(f"  {f'PGD-{args.pgd_steps} robust accuracy':<{label_w}}: {_pct(metrics['pgd_acc'])}")

    rb = metrics.get("random_pr_breakdown") or {}
    print(f"  Random-PR (N={args.random_n})")
    for d in args.random_dist:
        print(f"     - {d:<{label_w - 5}}: {_pct(rb.get(d))}")

    cb = metrics.get("corr_breakdown") or {}
    print(f"  Corruption accuracy")
    single_sev = len(args.corruption_severities) == 1
    for c in args.corruption_names:
        if single_sev:
            s = args.corruption_severities[0]
            v = cb.get((c, s))
            label = f"{c} (sev={s})"
            print(f"     - {label:<{label_w - 5}}: {_pct(v)}")
        else:
            sev_str = "  ".join(
                f"sev{s}={_pct(cb.get((c, s)))}" for s in args.corruption_severities
            )
            print(f"     - {c:<{label_w - 5}}: {sev_str}")
    if metrics["corr_acc"] is not None:
        print(f"  {'Corruption mean accuracy':<{label_w}}: {_pct(metrics['corr_acc'])}")
    print(bar)
    print()


def write_csv(args, ckpt, metrics, dataset, arch):
    import pandas as pd

    row = {
        "ckpt":          args.ckpt,
        "arch":          arch,
        "dataset":       dataset,
        "training_type": ckpt.get("training_type"),
        "epoch":         ckpt.get("epoch"),
        "epsilon":       args.epsilon,
        "norm":          args.norm,
        "num_samples":   metrics["num_samples"],
        "clean_acc":     metrics["clean_acc"],
        "pgd_acc":       metrics["pgd_acc"],
    }
    rb = metrics.get("random_pr_breakdown") or {}
    for d in args.random_dist:
        row[f"random_{d}"] = rb.get(d)

    cb = metrics.get("corr_breakdown") or {}
    for c in args.corruption_names:
        for s in args.corruption_severities:
            row[f"corr_{c}_s{s}"] = cb.get((c, s))
    row["corr_mean"] = metrics["corr_acc"]

    out_dir = os.path.dirname(os.path.abspath(args.save_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.save_csv, index=False)
    print(f"[eval] saved csv -> {args.save_csv}")


# ------------------------------------------------------------------
#                              Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt, device)
    dataset = args.dataset or ckpt.get("dataset")
    arch = args.arch or ckpt.get("arch")
    if dataset is None or arch is None:
        raise ValueError("Could not infer dataset/arch from checkpoint; "
                         "pass --dataset and --arch explicitly.")

    img_size = get_img_size(dataset, args.img_size or ckpt.get("img_size"))

    test_set, num_classes = get_dataset(
        dataset, args.data_root, train=False, img_size=img_size, augment=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Build model, then load weights into the unwrapped module. Training
    # scripts strip the DataParallel wrapper before saving (`model.module
    # .state_dict()`), so the keys here match without a `module.` prefix.
    model = build_model(arch, num_classes, dataset, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    pgd_cfg = {
        "epsilon":   args.epsilon,
        "alpha":     args.epsilon / 4.0,
        "num_steps": args.pgd_steps,
        "norm":      args.norm,
    }
    random_cfgs = [
        {
            "epsilon":      args.epsilon,
            "norm":         args.norm,
            "num_samples":  args.random_n,
            "noise_dist":   d,
            "return_stats": False,
        }
        for d in args.random_dist
    ]

    print(f"[eval] arch={arch} dataset={dataset} "
          f"img_size={img_size} ckpt={args.ckpt}")
    t0 = time.time()
    metrics = evaluate_per_epoch(
        model=model,
        loader=test_loader,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        pgd_cfg=pgd_cfg,
        locent_cfg=None,
        random_cfgs=random_cfgs,
        corruptions=args.corruption_names,
        severities=args.corruption_severities,
        eval_name="eval-test",
    )
    print(f"[eval] elapsed: {time.time() - t0:.1f}s")

    print_summary(args, ckpt, metrics)

    if args.save_csv:
        write_csv(args, ckpt, metrics, dataset, arch)


if __name__ == "__main__":
    main()
