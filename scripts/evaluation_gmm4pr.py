# scripts/evaluation_gmm4pr.py — Unified evaluation of a trained classifier
# under multiple robustness metrics.
#
# Supports:
#   * Random PR — Gaussian / Uniform / Laplace noise (any subset)
#   * Adversarial accuracy — PGD, CW, AutoAttack (any subset)
#   * GMM4PR — probabilistic robustness from a trained GMM4PR checkpoint
#
# The classifier checkpoint and the GMM4PR checkpoint are *independent*: the
# GMM uses its own frozen feature extractor (potentially a different arch on
# a different training run), and the only thing it shares with the classifier
# under evaluation is the dataset. Pass any combination of arch / training
# regime for the classifier with any compatible GMM.
#
# Designed to be driven from a shell script — every evaluation method is
# opt-in via a CLI flag so the same script supports quick spot checks and
# full-protocol runs.
#
# Architectures targeted: resnet18 / resnet50 / wide_resnet50_2 (others work too).
# Datasets targeted:      cifar10 / cifar100 / tinyimagenet.
#
# Examples:
#   # Random-PR + PGD + GMM on a PR-trained resnet18 / cifar10:
#   python scripts/evaluation_gmm4pr.py \
#       --ckp_path ./ckp/.../resnet18_cifar10.pth \
#       --eval_random --eval_pgd --eval_gmm \
#       --gmm_path ./ckp/gmm_fitting/.../gmm.pt
#
#   # Full protocol (everything):
#   python scripts/evaluation_gmm4pr.py \
#       --ckp_path ./ckp/.../model.pth \
#       --eval_all --gmm_path ./ckp/.../gmm.pt

import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from utils.evaluator import Evaluator
from utils.utils import load_gmm_model, check_mode_collapse
from src.adv_attacker import pgd_attack, cw_attack


# ------------------------------------------------------------------
#                           Helpers
# ------------------------------------------------------------------

def autoattack_wrapper(model, x, y, norm, epsilon, version="standard", **_):
    """Adapt the AutoAttack library to the (model, x, y, **kw) -> x_adv interface."""
    import autoattack  # local import; only required when --eval_aa is set
    norm_str = "Linf" if norm.lower() == "linf" else "L2"
    adversary = autoattack.AutoAttack(
        model, norm=norm_str, eps=float(epsilon), version=version, verbose=False,
    )
    return adversary.run_standard_evaluation(x, y, bs=x.size(0))


def set_seed(seed: int = 42):
    """Seed every RNG used by the eval pipeline + force deterministic cuDNN.

    Eval-time determinism is more important than throughput, so we *disable*
    cuDNN benchmark and pin deterministic kernels — different from training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def infer_training_type(ckpt, ckp_path):
    tt = ckpt.get("training_type", "unknown")
    if tt != "unknown":
        return tt
    parts = Path(ckp_path).parts
    for tag in ("standard_training", "adv_training", "pr_training",
                "standard", "adv_pgd", "pr", "loc_entropy", "trades"):
        if tag in parts:
            return tag
    return tt


# ------------------------------------------------------------------
#                              CLI
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified evaluation: random-PR, PGD, CW, AutoAttack, GMM4PR. "
                    "Classifier ckpt and GMM4PR ckpt are independent.",
    )

    # ---- Classifier checkpoint / arch ----
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_b_16",
    ], default="resnet18",
                    help="Fallback when the checkpoint lacks an `arch` field.")
    ap.add_argument("--ckp_path", required=True, type=str,
                    help="Path to the classifier .pth checkpoint to evaluate.")

    # ---- Dataset ----
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"],
                    default=None,
                    help="Dataset to evaluate on (default: read from ckpt).")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_train", action="store_true", default=False,
                    help="Also evaluate on the training split (no augmentation).")

    # ---- Which evaluation methods to run ----
    ap.add_argument("--eval_random", action="store_true",
                    help="Random-PR (Gaussian/Uniform/Laplace).")
    ap.add_argument("--eval_pgd",    action="store_true",
                    help="PGD adversarial accuracy.")
    ap.add_argument("--eval_cw",     action="store_true",
                    help="Carlini-Wagner adversarial accuracy.")
    ap.add_argument("--eval_aa",     action="store_true",
                    help="AutoAttack adversarial accuracy.")
    ap.add_argument("--eval_gmm",    action="store_true",
                    help="GMM4PR probabilistic robustness (requires --gmm_path).")
    ap.add_argument("--eval_all",    action="store_true",
                    help="Enable every eval method (random + PGD + CW + AA + GMM).")

    # ---- Randomness ----
    ap.add_argument("--seed", type=int, default=42,
                    help="Master seed for python / numpy / torch / cuDNN.")
    ap.add_argument("--deterministic", action="store_true", default=True,
                    help="(default on) Force deterministic cuDNN kernels.")
    ap.add_argument("--device", type=str, default="cuda")

    # ---- Shared attack / PR budget ----
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Perturbation norm used by random-PR / PGD / CW / AA.")
    ap.add_argument("--epsilon", type=float, default=8 / 255,
                    help="Perturbation radius. Default 8/255 (CIFAR convention).")
    ap.add_argument("--alpha", type=float, default=2 / 255,
                    help="PGD / CW step size.")

    # ---- Random-PR knobs ----
    ap.add_argument("--random_dist", nargs="+",
                    choices=["gaussian", "uniform", "laplace"],
                    default=["gaussian", "uniform", "laplace"],
                    help="Sampling distributions to evaluate under random-PR.")
    ap.add_argument("--num_samples_random", type=int, default=32,
                    help="Number of random draws per input (N) for random-PR.")

    # ---- PGD knobs ----
    ap.add_argument("--pgd_steps", type=int, default=20)

    # ---- CW knobs ----
    ap.add_argument("--cw_steps", type=int, default=30)

    # ---- AutoAttack knobs ----
    ap.add_argument("--aa_version", choices=["standard", "plus", "rand", "custom"],
                    default="standard")

    # ---- GMM4PR knobs (independent of the classifier ckpt) ----
    ap.add_argument("--gmm_path", type=str, default=None,
                    help="Path to a trained GMM4PR .pt checkpoint. "
                         "Required iff --eval_gmm is set.")
    ap.add_argument("--gmm_epsilon", type=float, default=None,
                    help="Override the GMM eval radius (default: GMM's train eps).")
    ap.add_argument("--gmm_norm", choices=["linf", "l2"], default=None,
                    help="Override the GMM eval norm (default: GMM's train norm).")
    ap.add_argument("--gmm_num_samples", type=int, default=32,
                    help="Number of GMM perturbation draws per input (N).")
    ap.add_argument("--gmm_diagnostics", action="store_true", default=False,
                    help="Also run check_mode_collapse on the GMM and log its "
                         "per-component stats (max_pi/min_pi/std_pi/entropy/"
                         "entropy_ratio) into the result row. Off by default.")
    ap.add_argument("--gmm_diag_num_batches", type=int, default=10,
                    help="Number of batches sampled by check_mode_collapse "
                         "when --gmm_diagnostics is set.")

    # ---- Where to save the results CSV ----
    ap.add_argument("--save_dir", type=str, default="./results/nppr_eval",
                    help="Directory for per-ckpt result CSVs.")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Explicit CSV path (overrides --save_dir auto-naming).")
    ap.add_argument("--csv_suffix", type=str, default="",
                    help="Optional suffix appended to the auto-named CSV "
                         "(e.g. '_seed42'); ignored when --save_csv is set.")

    return ap.parse_args()


# ------------------------------------------------------------------
#                              Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    if args.eval_all:
        args.eval_random = args.eval_pgd = args.eval_cw = args.eval_aa = True
        args.eval_gmm = True

    enabled = [args.eval_random, args.eval_pgd, args.eval_cw, args.eval_aa, args.eval_gmm]
    if not any(enabled):
        raise SystemExit(
            "ERROR: no evaluation methods enabled. Pass at least one of "
            "--eval_random / --eval_pgd / --eval_cw / --eval_aa / --eval_gmm, "
            "or --eval_all."
        )
    if args.eval_gmm and not args.gmm_path:
        raise SystemExit("ERROR: --eval_gmm requires --gmm_path.")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load classifier checkpoint
    # ------------------------------------------------------------------
    if not os.path.isfile(args.ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckp_path}")

    print(f"[ckp] loading from: {args.ckp_path}")
    ckpt = torch.load(args.ckp_path, map_location="cpu", weights_only=False)

    arch          = ckpt.get("arch", args.arch)
    dataset       = args.dataset or ckpt.get("dataset")
    if dataset is None:
        raise SystemExit("ERROR: dataset could not be inferred from ckpt; "
                         "pass --dataset explicitly.")
    training_type = infer_training_type(ckpt, args.ckp_path)
    img_size      = get_img_size(dataset, args.img_size or ckpt.get("img_size"))
    print(f"      arch={arch}, dataset={dataset}, img_size={img_size}, "
          f"training_type={training_type}, epoch={ckpt.get('epoch', '?')}")

    # ------------------------------------------------------------------
    # Build datasets and loaders
    # ------------------------------------------------------------------
    test_set, num_classes = get_dataset(dataset, args.data_root, False, img_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    train_loader = None
    if args.eval_train:
        train_set, _ = get_dataset(dataset, args.data_root, True, img_size, augment=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Build and load classifier
    # ------------------------------------------------------------------
    model = build_model(arch, num_classes, dataset)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    print(f"[model] {arch} loaded — {num_classes} classes on {device}")

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Load GMM (optional) — independent of the classifier above
    # ------------------------------------------------------------------
    gmm = None
    gmm_eval_eps = None
    gmm_eval_norm = None
    if args.eval_gmm:
        print(f"[gmm] loading from: {args.gmm_path}")
        gmm = load_gmm_model(args.gmm_path, dataset=dataset, device=str(device))
        gmm_eval_eps  = args.gmm_epsilon if args.gmm_epsilon is not None else gmm.budget["eps"]
        gmm_eval_norm = args.gmm_norm    if args.gmm_norm    is not None else gmm.budget["norm"]
        print(f"      feat_arch={gmm._feat_arch}  (classifier arch={arch})")
        print(f"      K={gmm.K}, latent_dim={gmm.latent_dim}, cond_mode={gmm.cond_mode}")
        print(f"      train budget: eps={gmm.budget['eps']:.4f}, norm={gmm.budget['norm']}")
        print(f"      eval  budget: eps={gmm_eval_eps:.4f}, norm={gmm_eval_norm}")

    # ------------------------------------------------------------------
    # Results dictionary (one row per ckpt)
    # ------------------------------------------------------------------
    results = {
        "arch":          arch,
        "dataset":       dataset,
        "training_type": training_type,
        "ckp_path":      args.ckp_path,
        "epoch":         ckpt.get("epoch"),
        "seed":          args.seed,
        "norm":          args.norm,
        "epsilon":       args.epsilon,
        "alpha":         args.alpha,
    }
    if args.eval_random:
        results["num_samples_random"] = args.num_samples_random
        results["random_dists"]       = ",".join(args.random_dist)
    if args.eval_pgd:
        results["pgd_steps"] = args.pgd_steps
    if args.eval_cw:
        results["cw_steps"]  = args.cw_steps
    if args.eval_aa:
        results["aa_version"] = args.aa_version
    if args.eval_gmm:
        results["gmm_path"]        = args.gmm_path
        results["gmm_feat_arch"]   = gmm._feat_arch
        results["gmm_train_eps"]   = gmm.budget["eps"]
        results["gmm_train_norm"]  = gmm.budget["norm"]
        results["gmm_eval_eps"]    = gmm_eval_eps
        results["gmm_eval_norm"]   = gmm_eval_norm
        results["gmm_num_samples"] = args.gmm_num_samples

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # ------------------------------------------------------------------
    # Evaluation loops
    # ------------------------------------------------------------------
    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        ev = Evaluator(model, loader, criterion, device)

        # --- Clean accuracy (always, one extra forward pass) ---
        print(f"[clean] ...")
        t0 = time.perf_counter()
        r = ev.evaluate_standard(eval_name="clean")
        dt = time.perf_counter() - t0
        print(f"    clean_acc={r['acc']*100:.2f}%  clean_loss={r['loss']:.4f}"
              f"  [{dt:.1f}s]")
        results[f"{split}_clean_acc"]  = r["acc"]
        results[f"{split}_clean_loss"] = r["loss"]
        results[f"{split}_clean_time"] = dt

        # --- Random PR ---
        if args.eval_random:
            for dist in args.random_dist:
                print(f"[random-PR] dist={dist}, N={args.num_samples_random}, "
                      f"ε={args.epsilon:.4f}, norm={args.norm} ...")
                t0 = time.perf_counter()
                r = ev.evaluate_pr_random(
                    eval_name=f"PR-rand-{dist}",
                    epsilon=args.epsilon, norm=args.norm,
                    num_samples=args.num_samples_random, noise_dist=dist,
                    return_stats=False,
                )
                dt = time.perf_counter() - t0
                print(f"    pr_{dist}={r['pr']*100:.2f}%  [{dt:.1f}s]")
                results[f"{split}_pr_{dist}"]      = r["pr"]
                results[f"{split}_pr_{dist}_time"] = dt

        # --- PGD ---
        if args.eval_pgd:
            print(f"[PGD] norm={args.norm}, ε={args.epsilon:.4f}, α={args.alpha:.4f}, "
                  f"steps={args.pgd_steps} ...")
            t0 = time.perf_counter()
            r = ev.evaluate_adversarial(
                attacker=pgd_attack, eval_name=f"PGD-{args.pgd_steps}",
                epsilon=args.epsilon, alpha=args.alpha,
                num_steps=args.pgd_steps, norm=args.norm,
            )
            dt = time.perf_counter() - t0
            print(f"    pgd_acc={r['acc']*100:.2f}%  [{dt:.1f}s]")
            results[f"{split}_pgd_acc"]  = r["acc"]
            results[f"{split}_pgd_loss"] = r["loss"]
            results[f"{split}_pgd_time"] = dt

        # --- CW ---
        if args.eval_cw:
            print(f"[CW] ε={args.epsilon:.4f}, step={args.alpha:.4f}, "
                  f"steps={args.cw_steps} ...")
            t0 = time.perf_counter()
            r = ev.evaluate_adversarial(
                attacker=cw_attack, eval_name=f"CW-{args.cw_steps}",
                epsilon=args.epsilon, step_size=args.alpha,
                num_steps=args.cw_steps, num_classes=num_classes,
                device=device,
            )
            dt = time.perf_counter() - t0
            print(f"    cw_acc={r['acc']*100:.2f}%  [{dt:.1f}s]")
            results[f"{split}_cw_acc"]  = r["acc"]
            results[f"{split}_cw_loss"] = r["loss"]
            results[f"{split}_cw_time"] = dt

        # --- AutoAttack ---
        if args.eval_aa:
            print(f"[AA] version={args.aa_version}, norm={args.norm}, "
                  f"ε={args.epsilon:.4f} ...")
            t0 = time.perf_counter()
            r = ev.evaluate_adversarial(
                attacker=autoattack_wrapper, eval_name=f"AA-{args.aa_version}",
                norm=args.norm, epsilon=args.epsilon, version=args.aa_version,
            )
            dt = time.perf_counter() - t0
            print(f"    aa_acc={r['acc']*100:.2f}%  [{dt:.1f}s]")
            results[f"{split}_aa_acc"]  = r["acc"]
            results[f"{split}_aa_loss"] = r["loss"]
            results[f"{split}_aa_time"] = dt

        # --- GMM4PR ---
        if args.eval_gmm:
            print(f"[GMM] N={args.gmm_num_samples}, "
                  f"ε={gmm_eval_eps:.4f}, norm={gmm_eval_norm} ...")
            t0 = time.perf_counter()
            r = ev.evaluate_pr_gmm(
                gmm=gmm, eval_name="PR-GMM",
                num_samples=args.gmm_num_samples,
                epsilon=args.gmm_epsilon, norm=args.gmm_norm,
            )
            dt = time.perf_counter() - t0
            print(f"    pr_gmm={r['pr']*100:.2f}%  [{dt:.1f}s]")
            results[f"{split}_pr_gmm"]      = r["pr"]
            results[f"{split}_pr_gmm_time"] = dt
            for k, v in r.get("stats", {}).items():
                if isinstance(v, (int, float)):
                    results[f"{split}_pr_gmm_{k}"] = v

            # Optional GMM diagnostics: π-distribution stats from
            # check_mode_collapse (same helper used in scripts/train_gmm.py).
            # Off by default; only scalar stats are logged into the CSV.
            if args.gmm_diagnostics:
                print(f"[GMM-diag] check_mode_collapse "
                      f"(num_batches={args.gmm_diag_num_batches}) ...")
                t0 = time.perf_counter()
                diag = check_mode_collapse(
                    gmm, loader, device,
                    num_batches=args.gmm_diag_num_batches,
                )
                gmm.eval()  # check_mode_collapse leaves the GMM in train mode
                dt = time.perf_counter() - t0
                for k, v in diag.items():
                    if isinstance(v, (int, float)):
                        results[f"{split}_gmm_diag_{k}"] = v
                results[f"{split}_gmm_diag_time"] = dt

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {args.ckp_path}")
    print(f"  Arch / Dataset: {arch} / {dataset}  [{training_type}]")
    print(f"  Budget        : norm={args.norm}, ε={args.epsilon:.4f}, α={args.alpha:.4f}")
    print(f"  Seed          : {args.seed}")
    if args.eval_gmm:
        print(f"  GMM           : feat_arch={gmm._feat_arch}, K={gmm.K}, "
              f"cond={gmm.cond_mode}")
        print(f"                  train budget eps={gmm.budget['eps']:.4f}/{gmm.budget['norm']} "
              f"-> eval eps={gmm_eval_eps:.4f}/{gmm_eval_norm}")
    print()

    label_w = 22
    header = " " * (label_w + 2) + "  ".join(f"{s:>8}" for s, _ in splits)
    print("  " + header)
    print("  " + "-" * (label_w + 2 + 10 * len(splits)))

    def _row(label, key_tmpl):
        cells = []
        for s, _ in splits:
            v = results.get(key_tmpl.format(split=s))
            cells.append(f"{v*100:>7.2f}%" if v is not None else f"{'--':>8}")
        print(f"  {label:<{label_w}}  " + "  ".join(cells))

    _row("Clean", "{split}_clean_acc")
    if args.eval_random:
        for d in args.random_dist:
            _row(f"PR-rand-{d}", f"{{split}}_pr_{d}")
    if args.eval_pgd:
        _row(f"PGD-{args.pgd_steps}",    "{split}_pgd_acc")
    if args.eval_cw:
        _row(f"CW-{args.cw_steps}",      "{split}_cw_acc")
    if args.eval_aa:
        _row(f"AA-{args.aa_version}",    "{split}_aa_acc")
    if args.eval_gmm:
        _row("PR-GMM",                   "{split}_pr_gmm")

    # ------------------------------------------------------------------
    # CSV save — always on (target dir defaults to ./results/nppr_eval)
    # ------------------------------------------------------------------
    save_csv = args.save_csv
    if save_csv is None:
        os.makedirs(args.save_dir, exist_ok=True)
        base = Path(args.ckp_path).stem
        save_csv = os.path.join(
            args.save_dir, f"{base}{args.csv_suffix}_eval.csv"
        )

    csv_dir = os.path.dirname(os.path.abspath(save_csv))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Append-on-existence so the same CSV accumulates rows across runs (e.g.
    # one row per seed). We read-concat-write rather than `mode='a'` so that
    # columns merge safely if a later run enables a different eval method —
    # missing values become NaN instead of silently misaligning.
    new_row = pd.DataFrame([results])
    if os.path.isfile(save_csv):
        existing = pd.read_csv(save_csv)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
        combined.to_csv(save_csv, index=False)
        print(f"\n[save] appended row to: {save_csv}  (total rows: {len(combined)})")
    else:
        new_row.to_csv(save_csv, index=False)
        print(f"\n[save] results written to: {save_csv}")


if __name__ == "__main__":
    main()
