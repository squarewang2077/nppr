# scripts/eval_prob_pert.py — Evaluate trained classifier: probabilistic robustness
#
# Reports:
#   PR      — Langevin-sampled probabilistic robustness
#   PR-G    — Gaussian random baseline
#   PR-U    — Uniform random baseline
#   PR-L    — Laplace random baseline
#   PR-Mix  — Trained Mixture (MixedNoise4PR) probabilistic robustness (optional)
#
# Usage example:
#   python scripts/eval_prob_pert.py \
#       --dataset cifar10 --arch resnet18 \
#       --ckp_path ./ckp/pr_training/resnet18_cifar10.pth \
#       --norm linf --epsilon 0.06274 --num_samples 32 \
#       --mixture_path ./results/gmm_expressivity/resnet18_on_cifar10/mixture_K3_*.pt

import os
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

from arch import build_model, build_feat_extractor
from utils.preprocess_data import get_dataset, get_img_size
from src.langevin4pr import pr_generator
from utils.evaluator import Evaluator
from utils import build_sigma_list
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_mixture_model(mixture_path, dataset, device="cuda"):
    """
    Load a trained MixedNoise4PR checkpoint produced by train_mixture.py.

    The checkpoint stores training args under cfg["config"] (= vars(args)).
    feat_extractor and up_sampler module structures are rebuilt from those
    args so that load_state_dict can fill in the saved weights.
    """
    from src.mixture4pr import MixedNoise4PR, build_decoder_from_flag

    if not os.path.isfile(mixture_path):
        raise FileNotFoundError(f"Mixture checkpoint not found: {mixture_path}")

    ckpt = torch.load(mixture_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    # Training args are stored nested under cfg["config"] (from vars(args))
    train_args = cfg.get("config", {})

    num_cls   = cfg["num_cls"]
    cond_mode = cfg["cond_mode"]   # None / "x" / "y" / "xy"

    # Build feat_extractor module structure when conditioning on x or xy.
    # Weights will be overwritten by load_state_dict — we only need the structure.
    feat_extractor = None
    if cond_mode in ("x", "xy"):
        feat_arch = train_args.get("feat_arch") or train_args.get("arch", "resnet18")
        feat_extractor = build_feat_extractor(feat_arch, num_cls, dataset)

    # Build up_sampler module structure for trainable decoders.
    up_sampler = None
    if train_args.get("use_decoder", False):
        decoder_backend = train_args.get("decoder_backend", "bicubic_trainable")
        latent_dim      = cfg["latent_dim"]
        img_size        = get_img_size(dataset)
        channels        = 1 if dataset.lower() == "mnist" else 3
        out_shape       = (channels, img_size, img_size)
        up_sampler = build_decoder_from_flag(decoder_backend, latent_dim, out_shape, "cpu")

    mixture = MixedNoise4PR.load_from_checkpoint(
        mixture_path,
        feat_extractor=feat_extractor,
        up_sampler=up_sampler,
        map_location=device,
        strict=True,
    )
    mixture = mixture.to(device).eval()
    for p in mixture.parameters():
        p.requires_grad_(False)

    # Stash resolved feat_arch for display
    mixture._feat_arch = train_args.get("feat_arch") or train_args.get("arch", "resnet18")
    return mixture


# ------------------------------------------------------------------
#                           Main Function
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a trained image classifier — probabilistic robustness "
                    "(PR, PR-G, PR-U, PR-L, PR-Mix)"
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

    # ---- Mixture PR evaluation ----
    ap.add_argument("--mixture_path", type=str, default=None,
                    help="Path to a trained MixedNoise4PR checkpoint (.pt) produced "
                         "by train_mixture.py. When provided, a mixture-based PR "
                         "evaluation is added.")
    ap.add_argument("--mixture_epsilon", type=float, default=None,
                    help="Override the perturbation radius for mixture evaluation. "
                         "Defaults to the budget stored in the checkpoint.")
    ap.add_argument("--mixture_norm", type=str, default=None,
                    choices=["linf", "l2"],
                    help="Override the perturbation norm for mixture evaluation. "
                         "Defaults to the budget stored in the checkpoint.")

    # ---- Misc ----
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_csv", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load classifier checkpoint
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
    # Build and load classifier
    # ------------------------------------------------------------------
    model = build_model(arch, num_classes, dataset)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    print(f"[model] {arch} loaded — {num_classes} classes on {device}")

    # ------------------------------------------------------------------
    # Optionally load trained Mixture model
    # ------------------------------------------------------------------
    mixture = None
    _mix_eval_eps  = None
    _mix_eval_norm = None
    if args.mixture_path is not None:
        print(f"[mixture] loading from: {args.mixture_path}")
        mixture = load_mixture_model(
            args.mixture_path,
            dataset=dataset,
            device=str(device),
        )
        _mix_feat_arch = mixture._feat_arch
        _mix_eval_eps  = args.mixture_epsilon if args.mixture_epsilon is not None \
                         else mixture.budget["eps"]
        _mix_eval_norm = args.mixture_norm    if args.mixture_norm    is not None \
                         else mixture.budget["norm"]
        print(f"[mixture] loaded — K={mixture.K}, latent_dim={mixture.latent_dim}, "
              f"cond_mode={mixture._strategy.mode_name}")
        print(f"[mixture] feat_arch={_mix_feat_arch}  (classifier arch={arch})")
        print(f"[mixture] training budget: eps={mixture.budget['eps']:.4f}, "
              f"norm={mixture.budget['norm']}")
        print(f"[mixture] eval budget:     eps={_mix_eval_eps:.4f}, "
              f"norm={_mix_eval_norm}")

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

        # 3. Mixture-based PR
        if mixture is not None:
            print(f"[3] PR Mixture evaluation  "
                  f"(N={args.num_samples}, feat={_mix_feat_arch}, "
                  f"ε={_mix_eval_eps:.4f}, norm={_mix_eval_norm}) ...")
            _t0 = time.perf_counter()
            pr_mix_res = evaluator.evaluate_pr_gmm(
                gmm=mixture,
                eval_name="PR-Mix",
                num_samples=args.num_samples,
                epsilon=args.mixture_epsilon,
                norm=args.mixture_norm,
            )
            _t_pr_mix = time.perf_counter() - _t0
            print(f"    pr_mix={pr_mix_res['pr']*100:.2f}%  "
                  f"D={pr_mix_res['stats']['D_proxy']:.3e}"
                  f"  [{_t_pr_mix:.1f}s]")
            results[f"{split}_pr_mix"] = pr_mix_res["pr"]
            results[f"{split}_pr_mix_time"] = _t_pr_mix
            for k, v in pr_mix_res.get("stats", {}).items():
                if isinstance(v, (int, float)):
                    results[f"{split}_pr_mix_{k}"] = v

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
    if mixture is not None:
        print(f"  PR Mixture    : N={args.num_samples}, K={mixture.K}, "
              f"cond={mixture._strategy.mode_name}, feat={_mix_feat_arch}")
        print(f"                  train budget: eps={mixture.budget['eps']:.4f}, "
              f"norm={mixture.budget['norm']}")
        print(f"                  eval  budget: eps={_mix_eval_eps:.4f}, "
              f"norm={_mix_eval_norm}")
    print()

    # Timing
    _timing_rows = [
        ("pr",                  "pr_time",                 True),
        ("pr-rand-gaussian",    "pr_rand_gaussian_time",   not args.no_pr_random),
        ("pr-rand-uniform",     "pr_rand_uniform_time",    not args.no_pr_random),
        ("pr-rand-laplace",     "pr_rand_laplace_time",    not args.no_pr_random),
        ("pr-mix",              "pr_mix_time",             mixture is not None),
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
    if mixture is not None:
        header  += f"  {'PR-Mix':>8}"
        divider += 10
    print(header)
    print("  " + "-" * divider)

    for split, _ in splits:
        row = f"  {split:<6}  {results[f'{split}_pr']*100:>7.2f}%"
        if not args.no_pr_random:
            for _dist in ("gaussian", "uniform", "laplace"):
                row += f"  {results[f'{split}_pr_rand_{_dist}']*100:>6.2f}%"
        if mixture is not None:
            row += f"  {results[f'{split}_pr_mix']*100:>7.2f}%"
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
