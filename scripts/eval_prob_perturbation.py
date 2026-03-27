# scripts/eval_prob_perturbation.py — Evaluate trained classifier: probabilistic robustness
#
# Reports:
#   PR      — Langevin-sampled probabilistic robustness
#   PR-G    — Gaussian random baseline
#   PR-U    — Uniform random baseline
#   PR-L    — Laplace random baseline
#   PR-GMM  — Trained GMM-based probabilistic robustness (optional)
#
# Usage example:
#   python scripts/eval_prob_perturbation.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/pr_training/resnet18_cifar10.pth \
#       --norm linf --epsilon 0.03137 --num_samples 32 \
#       --gmm_path ./ckp/gmm/gmm_resnet50_cifar10.pt

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
from src.langevin4pr import pr_generator
from utils.evaluator import Evaluator
from configs.train_clf_cfg import build_sigma_list
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
        description="Evaluate a trained image classifier — probabilistic robustness "
                    "(PR, PR-G, PR-U, PR-L, PR-GMM)"
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

    # ---- PR attack settings ----
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf")
    ap.add_argument("--epsilon", type=float, default=8/255)
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
                    help="Number of perturbation samples per input (N)")
    ap.add_argument("--no_pr_random", action="store_true", default=False,
                    help="Skip all random-noise PR baseline evaluations "
                         "(Gaussian, Uniform, Laplace)")

    # ---- GMM PR evaluation ----
    ap.add_argument("--gmm_path", type=str, 
                    default='./ckp/gmm_fitting/resnet/resnet18_on_cifar10/gmm_K3_cond(x)_decoder(nontrainable)_linf(16)_reg(none).pt',
                    help="Path to a trained GMM4PR checkpoint (.pt). "
                         "When provided, a GMM-based PR evaluation is added.")
    ap.add_argument("--gmm_epsilon", type=float, default=None,
                    help="Override the perturbation radius for GMM evaluation.")
    ap.add_argument("--gmm_norm", type=str, default=None,
                    choices=["linf", "l2"],
                    help="Override the perturbation norm for GMM evaluation.")

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
    # Optionally load trained GMM
    # ------------------------------------------------------------------
    gmm = None
    _gmm_eval_eps  = None
    _gmm_eval_norm = None
    if args.gmm_path is not None:
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

    results = {
        "arch": arch, "dataset": dataset,
        "training_type": training_type, "ckp_path": args.ckp_path,
        "norm": args.norm, "epsilon": args.epsilon,
        "num_samples": args.num_samples,
    }

    splits = [("test", test_loader)]
    if train_loader is not None:
        splits.append(("train", train_loader))

    # ------------------------------------------------------------------
    # Run probabilistic robustness evaluations
    # ------------------------------------------------------------------
    for split, loader in splits:
        print(f"\n{'='*60}")
        print(f"Split : {split}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        evaluator = Evaluator(model, loader, criterion, device)

        # 1. PR (Langevin)
        print(f"[1] PR evaluation  (N={args.num_samples}, K={args.K}) ...")
        _t0 = time.perf_counter()
        pr = evaluator.evaluate_pr(
            pr_generator=pr_generator,
            eval_name=f"PR-{args.num_samples}",
            **pr_config,
        )
        _t_pr = time.perf_counter() - _t0
        print(f"    pr={pr['pr']*100:.2f}%  "
              f"D={pr['stats']['D_proxy']:.3e}  Hpi={pr['stats']['pi_entropy']:.3f}"
              f"  [{_t_pr:.1f}s]")
        results[f"{split}_pr"] = pr["pr"]
        results[f"{split}_pr_time"] = _t_pr
        for k, v in pr.get("stats", {}).items():
            results[f"{split}_pr_{k}"] = v

        # 2. PR random baselines — Gaussian, Uniform, Laplace
        if not args.no_pr_random:
            for _dist in ("gaussian", "uniform", "laplace"):
                print(f"[2] PR random evaluation  "
                      f"(N={args.num_samples}, dist={_dist}) ...")
                _t0 = time.perf_counter()
                _pr_rand = evaluator.evaluate_pr_random(
                    eval_name=f"PR-rand-{_dist}",
                    norm=args.norm,
                    epsilon=args.epsilon,
                    num_samples=args.num_samples,
                    noise_dist=_dist,
                )
                _t_rand = time.perf_counter() - _t0
                print(f"    pr_rand_{_dist}={_pr_rand['pr']*100:.2f}%  "
                      f"D={_pr_rand['stats']['D_proxy']:.3e}"
                      f"  [{_t_rand:.1f}s]")
                results[f"{split}_pr_rand_{_dist}"] = _pr_rand["pr"]
                results[f"{split}_pr_rand_{_dist}_time"] = _t_rand
                for k, v in _pr_rand.get("stats", {}).items():
                    if isinstance(v, (int, float)):
                        results[f"{split}_pr_rand_{_dist}_{k}"] = v

        # 3. GMM-based PR
        if gmm is not None:
            print(f"[3] PR GMM evaluation  "
                  f"(N={args.num_samples}, feat={_gmm_feat_arch}, "
                  f"ε={_gmm_eval_eps:.4f}, norm={_gmm_eval_norm}) ...")
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
    print(f"  PR            : N={args.num_samples}, K={args.K}")
    if not args.no_pr_random:
        print(f"  PR random     : N={args.num_samples}, dists=gaussian,uniform,laplace")
    if gmm is not None:
        print(f"  PR GMM        : N={args.num_samples}, K={gmm.K}, "
              f"cond={gmm.cond_mode}, feat={_gmm_feat_arch}")
        print(f"                  train budget: eps={gmm.budget['eps']:.4f}, "
              f"norm={gmm.budget['norm']}")
        print(f"                  eval  budget: eps={_gmm_eval_eps:.4f}, "
              f"norm={_gmm_eval_norm}")
    print()

    # Timing
    _timing_rows = [
        ("pr",                  "pr_time",                 True),
        ("pr-rand-gaussian",    "pr_rand_gaussian_time",   not args.no_pr_random),
        ("pr-rand-uniform",     "pr_rand_uniform_time",    not args.no_pr_random),
        ("pr-rand-laplace",     "pr_rand_laplace_time",    not args.no_pr_random),
        ("pr-gmm",              "pr_gmm_time",             gmm is not None),
    ]
    print(f"  {'Eval':<18}  " + "  ".join(f"{s:>8}" for s, _ in splits))
    print("  " + "-" * (18 + 2 + 10 * len(splits)))
    for label, key, active in _timing_rows:
        if not active:
            continue
        vals = "  ".join(
            f"{results[f'{s}_{key}']:>7.1f}s"
            if f"{s}_{key}" in results else f"{'N/A':>8}"
            for s, _ in splits
        )
        print(f"  {label:<18}  {vals}")
    print()

    # Accuracy table
    header  = f"  {'Split':<6}  {'PR':>8}"
    divider = 6 + 2 + 8
    if not args.no_pr_random:
        header  += f"  {'PR-G':>7}  {'PR-U':>7}  {'PR-L':>7}"
        divider += 27
    if gmm is not None:
        header  += f"  {'PR-GMM':>8}"
        divider += 10
    print(header)
    print("  " + "-" * divider)

    for split, _ in splits:
        row = f"  {split:<6}  {results[f'{split}_pr']*100:>7.2f}%"
        if not args.no_pr_random:
            for _dist in ("gaussian", "uniform", "laplace"):
                row += f"  {results[f'{split}_pr_rand_{_dist}']*100:>6.2f}%"
        if gmm is not None:
            row += f"  {results[f'{split}_pr_gmm']*100:>7.2f}%"
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
