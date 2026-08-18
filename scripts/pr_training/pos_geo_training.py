# pos_geo_training.py - Train on margin level-set perturbations, weighted by
#                       where on the level set each perturbation landed.
#
# Description:
#   Each image gets N random starts inside the epsilon-ball. Every start is
#   driven onto the level set m(x + delta, y) = t, and the model is trained on
#   all N with a weighted PGD-style objective:
#
#       loss = sum_r w_r * CE(f(x + delta_r), y)
#
#   The weighting is the experiment. --weight_mode selects one of five:
#
#     uniform    every valid position counts equally
#     sharp      favour large ||grad_delta CE||_*  — steep spots
#     flat       favour small ||grad_delta CE||_*  — wide-valley spots
#     min_norm   all weight on the position closest to the clean image
#     max_norm   all weight on the position furthest from it
#
#   Sharpness is the dual norm of the *cross-entropy* input gradient (L1 for an
#   L-inf ball, L2 for an L2 ball) — the local steepness of the loss surface,
#   which is what "sharp minimum" and "wide valley" refer to.
#
#   Perturbations that never reached the level set (|m - t| > tol) get zero
#   weight in every mode: the premise is "different positions at the *same*
#   level". Watch valid_rate — if it collapses, raise --num_steps or --epsilon.
#
# Outputs:
#   ckp/pos_geo_training/<name_tag>.pth                      model checkpoint
#   ckp/pos_geo_training/<name_tag>.log                      training log
#   results/pos_geo_training/<name_tag>_training_info.csv    per-epoch summary
#   results/pos_geo_training/<name_tag>_probe_ep{N}.npz      per-delta geometry
#
#   The CSV holds one row per evaluated epoch: clean and PGD accuracy on both
#   the train subset and the test set, plus batch-averaged geometry.
#
#   The .npz files hold the raw per-delta record on a *fixed* probe subset
#   (--probe_n images, the same ones every epoch, so one image's perturbations
#   can be tracked over training). Keys, with R = probe_n and N = num_starts:
#       delta_norm, margin1, margin2, ce, grad_dual, weights,
#       valid_mask, j1, j2                     (R, N)
#       pair_cos                               (R, N, N)
#       eff_rank, sigma_max, sigma_min, anisotropy   (R,)
#       indices                                (R,)  which images these are
#
#   margin1 / margin2 are the margins against the top-1 and top-2 wrong
#   classes; their gap says how firmly one class owns the decision. eff_rank is
#   the participation ratio of the pairwise-cosine spectrum — it reads as "how
#   many independent directions the N perturbations span", 1 when they collapse
#   onto one direction and N when they are mutually orthogonal.
#
# Evaluation is deliberately narrow: clean accuracy and PGD-10 robust accuracy,
# on the train subset and the test set. Nothing else.
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15
#   numpy
#   tqdm
#   pandas
#
# Dependencies:
#   arch/                       model registry and NormalizedModel wrapper
#   utils/preprocess_data.py    dataset loading and preprocessing
#   utils/epoch_eval.py         single-pass clean + PGD evaluation
#   src/pos_geo_loss.py         level-set solver, weighting, geometry
#
# Usage:
#   python scripts/pr_training/pos_geo_training.py [options]
#
# Examples:
#   # Uniform weighting at the decision boundary
#   python scripts/pr_training/pos_geo_training.py \
#       --dataset cifar10 --arch resnet18 --training_type level \
#       --t 0.0 --num_starts 8 --num_steps 50 --weight_mode uniform \
#       --eval_pgd --pgd_steps 10 --epochs 100
#
#   # Wide-valley weighting, tracking 512 probe images
#   python scripts/pr_training/pos_geo_training.py \
#       --dataset cifar100 --arch resnet18 --training_type level \
#       --t 0.0 --weight_mode flat --tau 1.0 --probe_n 512 \
#       --eval_pgd --pgd_steps 10 --epochs 100

import os
import logging
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pandas as pd

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from src.pos_geo_loss import (
    pos_geo_loss,
    level_geometry,
    dispersion,
    WEIGHT_MODES,
)
from utils.epoch_eval import evaluate_per_epoch

def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("pos_geo_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate output if root logger has handlers
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger




def set_seed(seed: int = 42):
    """Seed Python/NumPy/Torch RNGs and enable cuDNN auto-tuning.

    cudnn.benchmark=True lets cuDNN pick the fastest convolution kernel for
    each input shape; this is a sizeable speedup when activations are large
    (the (B*N, C, H, W) perturbation tensor here). We give up bit-exact
    reproducibility across hardware/cuDNN versions in exchange. Run-to-run
    seeding via the lines above is unaffected.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

# ------------------------------------------------------------------
#                       Standard Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, criterion,
                    epoch=None, total_epochs=None):
    """Standard training loop."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Train Epoch [{epoch}/{total_epochs}]" if epoch else "Training", leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()

        avg_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}")

    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)


# ------------------------------------------------------------------
#                    Probabilistic Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch_level(model, loader, optimizer, device, criterion,
                          level_config, epoch=None, total_epochs=None):
    """
    Margin level-set training loop with position weighting.

    Each image gets N random starts inside the epsilon-ball; every start is
    driven onto the level set m = t, and the model is trained on all of them
    with a weighted PGD-style objective. The weighting is the experiment — see
    --weight_mode.

    level_config keys:
        norm          : "linf" | "l2" (default "linf")
        epsilon       : radius of the ball (default 8/255)
        t             : target margin level (default 0.0)
        num_starts    : perturbations per image, i.e. N (default 8)
        num_steps     : solver steps; 0 = plain random sampling (default 50)
        step_size     : solver step size (default 1e-2)
        anchor_lambda : L2 pull toward each random start (default 0.02)
        psi_alpha     : sharpness of the symmetric penalty (default 10.0)
        tol           : |m - t| <= tol counts as reaching the level (default 0.05)
        weight_mode   : one of pos_geo_loss.WEIGHT_MODES (default "uniform")
        tau           : softmax temperature for sharp / flat (default 1.0)

    Returns:
        loss, train_acc, stats — stats is the per-epoch geometry summary that
        the caller writes into the training CSV.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Level Train [{epoch}/{total_epochs}]" if epoch else "Level Training",
                leave=False)

    solver_kwargs = _solver_kwargs(level_config)
    t = level_config.get("t", 0.0)
    epsilon = level_config.get("epsilon", 8 / 255)
    weight_mode = level_config.get("weight_mode", "uniform")
    tau = level_config.get("tau", 1.0)

    keys = ["margin1", "margin2", "margin_gap", "ce", "delta_norm", "grad_dual",
            "valid_rate", "pair_cos", "eff_rank", "sigma_max", "sigma_min",
            "anisotropy", "weight_entropy"]
    stat_sums = {k: 0.0 for k in keys}
    stat_count = 0
    train_acc = 0.0
    avg_stats = {k: 0.0 for k in keys}

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss, x_adv, info = pos_geo_loss(
            model, x, y, criterion, t=t, epsilon=epsilon,
            weight_mode=weight_mode, tau=tau, **solver_kwargs,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        # Reuse the loss's logits so accuracy costs no extra forward pass.
        B, N = x_adv.shape[:2]
        with torch.no_grad():
            preds = info["logits_adv"].argmax(dim=1)
            running_correct += (preds == y.repeat_interleave(N)).sum().item()

            batch = _batch_geometry(x_adv - x.unsqueeze(1), info)
            for k in keys:
                stat_sums[k] += batch[k].mean().item()
        stat_count += 1

        avg_loss = running_loss / total_samples
        avg_stats = {k: stat_sums[k] / stat_count for k in keys}
        train_acc = running_correct / (total_samples * N)

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            m1=f"{avg_stats['margin1']:.3f}",
            valid=f"{avg_stats['valid_rate']:.2f}",
            rank=f"{avg_stats['eff_rank']:.2f}",
            acc=f"{train_acc:.4f}",
        )

    return running_loss / len(loader.dataset), train_acc, avg_stats


def _solver_kwargs(level_config):
    """Pull the solver arguments out of a level_config dict."""
    return dict(
        num_starts=level_config.get("num_starts", 8),
        num_steps=level_config.get("num_steps", 50),
        step_size=level_config.get("step_size", 1e-2),
        anchor_lambda=level_config.get("anchor_lambda", 0.02),
        alpha=level_config.get("psi_alpha", 10.0),
        tol=level_config.get("tol", 0.05),
        norm=level_config.get("norm", "linf"),
    )


def _batch_geometry(delta, info):
    """
    Per-sample geometry for one batch: every value is (B,) or (B, N) so the
    caller can either average it into the epoch CSV or dump it per-delta.
    """
    geom = level_geometry(delta, info["margin1"])
    disp = dispersion(geom["pair_cos"])
    N = delta.shape[1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=delta.device)
    w = info["weights"]

    return {
        "margin1":       info["margin1"],
        "margin2":       info["margin2"],
        "margin_gap":    info["margin2"] - info["margin1"],
        "ce":            info["ce"],
        "delta_norm":    info["delta_norm"],
        "grad_dual":     info["grad_dual"],
        "valid_rate":    info["valid_mask"].float(),
        "pair_cos":      geom["pair_cos"][:, off_diag],
        "eff_rank":      disp["eff_rank"],
        "sigma_max":     disp["sigma_max"],
        "sigma_min":     disp["sigma_min"],
        "anisotropy":    disp["anisotropy"],
        # Shannon entropy of the weight row: 0 when one position takes
        # everything, log(N) when they share equally. Catches sharp / flat
        # collapsing onto a single delta.
        "weight_entropy": -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(dim=1),
    }


@torch.no_grad()
def _probe_indices(dataset, probe_n, seed=0):
    """A fixed, reproducible slice of the training set for per-delta logging."""
    g = torch.Generator().manual_seed(seed)
    n = min(probe_n, len(dataset))
    return torch.randperm(len(dataset), generator=g)[:n]


def probe_geometry(model, probe_loader, device, criterion, level_config):
    """
    Per-delta geometry on the fixed probe subset — the record that lets you
    watch one image's perturbations move over training.

    Returns a dict of numpy arrays ready for np.savez_compressed.
    """
    solver_kwargs = _solver_kwargs(level_config)
    per_delta, per_sample = {}, {}

    for x, y in probe_loader:
        x, y = x.to(device), y.to(device)
        # train_mode=False: this is a measurement, so it must not touch the
        # model's BatchNorm running statistics.
        _, x_adv, info = pos_geo_loss(
            model, x, y, criterion,
            t=level_config.get("t", 0.0),
            epsilon=level_config.get("epsilon", 8 / 255),
            weight_mode=level_config.get("weight_mode", "uniform"),
            tau=level_config.get("tau", 1.0),
            train_mode=False,
            **solver_kwargs,
        )
        with torch.no_grad():
            geom = level_geometry(x_adv - x.unsqueeze(1), info["margin1"])
            disp = dispersion(geom["pair_cos"])
            chunk = {
                "delta_norm": info["delta_norm"], "margin1": info["margin1"],
                "margin2": info["margin2"], "ce": info["ce"],
                "grad_dual": info["grad_dual"], "weights": info["weights"],
                "valid_mask": info["valid_mask"].float(),
                "j1": info["j1"].float(), "j2": info["j2"].float(),
                "pair_cos": geom["pair_cos"],
            }
            for k, v in chunk.items():
                per_delta.setdefault(k, []).append(v.cpu())
            for k, v in disp.items():
                per_sample.setdefault(k, []).append(v.cpu())

    out = {k: torch.cat(v).numpy() for k, v in per_delta.items()}
    out.update({k: torch.cat(v).numpy() for k, v in per_sample.items()})
    return out


# ------------------------------------------------------------------
#                           Main Function
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # ============================================================
    # Dataset & model
    # ============================================================
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet", "svhn"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", choices=[
            "resnet18", "resnet34", "resnet50", "wide_resnet50_2",
            "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
            "vit_tiny", "vit_small", "convit_tiny", "convit_small",
        ], default="resnet18")
    ap.add_argument("--pretrained", action="store_true",
                    help="Load ImageNet pretrained weights. For pretrained models, consider smaller lr.")
    # ============================================================
    # General training settings
    # ============================================================
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--img_size", type=int, default=None,
                    help="Input image size. If None, use the dataset default.")
    # ============================================================
    # Training method
    # ============================================================
    ap.add_argument("--training_type", choices=["standard", "level"], default="level",
                    help="Training method: standard or local-entropy robust training.")
    ap.add_argument("--augment", action="store_true",
                    help="Enable training-set data augmentation (RandomCrop / Flip / "
                         "RandAugment / RandomErasing). When set, output filenames "
                         "are tagged with '_Aug'.")
    # ============================================================
    # Local-entropy particle settings
    # ============================================================
    ap.add_argument("--epsilon", type=float, default=8 / 255,
                    help="Perturbation budget. For linf on CIFAR, 8/255 is standard.")
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for perturbations.")
    ap.add_argument("--num_starts", type=int, default=8,
                    help="Number of random starts (perturbations) per input.")
    # ============================================================
    # Level-set solver
    # ============================================================
    ap.add_argument("--t", type=float, default=0.0,
                    help="Target margin level. Perturbations are driven onto m = t.")
    ap.add_argument("--num_steps", type=int, default=50,
                    help="Solver steps. 0 leaves the random starts unoptimised.")
    ap.add_argument("--step_size", type=float, default=1e-2,
                    help="Solver step size.")
    ap.add_argument("--anchor_lambda", type=float, default=0.02,
                    help="L2 pull toward each random start. Keep small: large "
                         "values stop perturbations reaching the level set.")
    ap.add_argument("--psi_alpha", type=float, default=10.0,
                    help="Sharpness of the symmetric softplus penalty.")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="A perturbation counts as valid when |m - t| <= tol.")
    ap.add_argument("--weight_mode", choices=list(WEIGHT_MODES), default="uniform",
                    help="How to weight the N positions on the level set. "
                         "uniform = all equal; sharp / flat = favour large / "
                         "small CE-gradient dual norm; min_norm / max_norm = "
                         "all weight on the nearest / furthest perturbation.")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Softmax temperature for --weight_mode sharp / flat. "
                         "Large values flatten them back toward uniform.")
    ap.add_argument("--probe_n", type=int, default=256,
                    help="Size of the fixed probe subset whose per-delta "
                         "geometry is dumped to .npz every epoch.")
    ap.add_argument("--results_dir", type=str, default="./results/pos_geo_training",
                    help="Where the per-epoch CSV and probe .npz files go.")
    # ============================================================
    # Per-epoch evaluation knobs
    #
    # Each evaluation is opt-in. Clean accuracy is always reported. For each
    # extra evaluation we expose only the high-level "what level" knobs
    # (steps, num samples, severities, version) and optionally the norm; the
    # finer attack hyperparameters (PGD step size, etc.) are derived from the
    # training epsilon and standard ratios.
    # ============================================================

    # PGD
    ap.add_argument("--eval_pgd", action="store_true",
                    help="Run PGD adversarial eval at every eval cycle.")
    ap.add_argument("--pgd_steps", type=int, default=10,
                    help="Number of PGD steps when --eval_pgd is set.")
    ap.add_argument("--pgd_norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for PGD eval.")

    # ============================================================
    # Misc
    # ============================================================
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/pos_geo_training",
                    help="Directory to save checkpoint.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = get_img_size(args.dataset, args.img_size)

    # Set up output directory and logger early so config lines are captured
    os.makedirs(args.save_dir, exist_ok=True)
    aug_suffix = "_Aug" if args.augment else ""
    name_tag = f"{args.arch.lower()}_{args.dataset.lower()}_{args.training_type}{aug_suffix}"
    log_path = os.path.join(args.save_dir, f"{name_tag}.log")
    logger = setup_logger(log_path)

    # Log config
    logger.info(f"[config] dataset={args.dataset}, arch={args.arch}, pretrained={args.pretrained}")
    aug_state = "ENABLED (RandomCrop+Flip+RandAugment+RandomErasing)" if args.augment else "DISABLED (no-aug train set)"
    logger.info(f"[config] img_size={img_size}, augmentation={aug_state}")
    if args.training_type == "standard":
        logger.info(f"[config] training_type={args.training_type}, no adversarial perturbations")
    elif args.training_type == "level":
        logger.info(f"[config] training_type={args.training_type} (margin level set), "
                    f"epsilon={args.epsilon:.4f}, norm={args.norm}, t={args.t}")
        logger.info(f"         num_starts={args.num_starts}, num_steps={args.num_steps}, "
                    f"step_size={args.step_size}")
        logger.info(f"         anchor_lambda={args.anchor_lambda}, psi_alpha={args.psi_alpha}, "
                    f"tol={args.tol}")
        logger.info(f"         weight_mode={args.weight_mode}, tau={args.tau}, "
                    f"probe_n={args.probe_n}")
    else:
        raise ValueError(f"Unknown training_type: {args.training_type}")

    # accumulate one dict per evaluation epoch; written to CSV incrementally
    training_history = []

    # Build datasets/loaders (augmentation governed by --augment; test set is always no-aug)
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True, img_size, augment=args.augment)
    test_set, _ = get_dataset(args.dataset, args.data_root, False, img_size, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, # this for warm up of particales
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=4, pin_memory=True
    )

    ## Fixed subset of train set (no augmentation) for per-epoch monitoring ##
    subset_size = len(test_set) # match the test set size for a fair comparison of train vs test metrics
    # train_set w/o augmentation to ensure the same samples are selected across epochs and training types
    train_set_NONaug, _ = get_dataset(args.dataset, args.data_root, True, img_size, augment=False)
    rng = np.random.default_rng(seed=args.seed) # for subset selection reproducibility
    # randomly sample
    indices = rng.choice(len(train_set_NONaug), subset_size, replace=False)
    train_subset = Subset(train_set_NONaug, indices)

    subtrain_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,  # no need to shuffle the subset loader since it's only for monitoring
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"[eval] train eval subset: {subset_size}/{len(train_set)} samples ({subset_size/len(train_set)*100:.0f}%, fixed seed)")

    # Build model
    model = build_model(args.arch, num_classes, args.dataset, pretrained=args.pretrained)
    model.to(device)

    # Optional DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Level-set solver config
    level_config = {
        "type": args.training_type,
        "norm": args.norm,
        "epsilon": args.epsilon,
        "t": args.t,
        "num_starts": args.num_starts,
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "anchor_lambda": args.anchor_lambda,
        "psi_alpha": args.psi_alpha,
        "tol": args.tol,
        "weight_mode": args.weight_mode,
        "tau": args.tau,
    }

    # Fixed probe subset: the same images every epoch, so the per-delta dumps
    # can be read as one sample's geometry moving over training.
    probe_idx = _probe_indices(train_set_NONaug, args.probe_n, seed=args.seed)
    probe_loader = torch.utils.data.DataLoader(
        Subset(train_set_NONaug, probe_idx.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Output paths
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{name_tag}.pth")
    info_csv_path = os.path.join(args.results_dir, f"{name_tag}_training_info.csv")
    logger.info(f"[save] checkpoint -> {out_path}")
    logger.info(f"[save] log       -> {log_path}")
    logger.info(f"[save] csv       -> {info_csv_path}")
    logger.info(f"[save] probe npz -> {os.path.join(args.results_dir, name_tag + '_probe_ep{N}.npz')}")

    # Train
    ep = 0  # Initialize epoch counter
    for ep in range(1, args.epochs + 1):
        start = time.time()
        if args.training_type == "standard":
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                                    epoch=ep, total_epochs=args.epochs)
            avg_stats = {}  # Empty stats for standard training
        elif args.training_type == "level":
            train_loss, train_acc, avg_stats = train_one_epoch_level(
                model, train_loader, optimizer, device, criterion, level_config,
                epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        scheduler.step()
        elapsed = time.time() - start
        current_lr = scheduler.get_last_lr()[0]

        # Geometry is recorded every epoch — it is the point of this script and
        # costs one probe pass. The clean/PGD evaluation below is far more
        # expensive, so it stays on the every-5 cadence and its columns are
        # left empty on the epochs it does not run.
        row = {
            'arch':          args.arch,
            'dataset':       args.dataset,
            'training_type': args.training_type,
            'weight_mode':   args.weight_mode,
            'epoch':         ep,
            'lr':            current_lr,
            'time':          elapsed,
            'train_loss':    train_loss,
            'train_acc':     train_acc,
            'trainS_loss':   None, 'trainS_acc': None, 'trainS_pgd': None,
            'val_loss':      None, 'val_acc':    None, 'val_pgd':    None,
        }
        if args.training_type == "level" and avg_stats:
            row.update(avg_stats)

            probe = probe_geometry(model, probe_loader, device, criterion,
                                   level_config)
            probe["indices"] = probe_idx.numpy()
            probe_path = os.path.join(args.results_dir,
                                      f"{name_tag}_probe_ep{ep}.npz")
            np.savez_compressed(probe_path, **probe)

            logger.info(
                f"[{ep:03d}/{args.epochs}] geom: "
                f"m1={avg_stats['margin1']:+.3f} gap={avg_stats['margin_gap']:.3f} "
                f"|d|={avg_stats['delta_norm']:.3f} g={avg_stats['grad_dual']:.3f} "
                f"valid={avg_stats['valid_rate']:.2f} "
                f"eff_rank={avg_stats['eff_rank']:.2f} "
                f"w_ent={avg_stats['weight_entropy']:.3f} -> {os.path.basename(probe_path)}"
            )

        # Evaluation and checkpointing
        if ep % 5 == 0 or ep == args.epochs:

            # ----- Build eval configs from the per-evaluation flags. -----
            # PGD (alpha derived as epsilon/4 — standard PGD ratio).
            pgd_cfg = None
            if args.eval_pgd:
                pgd_cfg = {
                    "epsilon":   args.epsilon,
                    "alpha":     args.epsilon / 4.0,
                    "num_steps": args.pgd_steps,
                    "norm":      args.pgd_norm,
                }

            # Level: inherit level_config, override n/steps/norm for eval.
            ## Evaluation on Test set ##
            test_metrics = evaluate_per_epoch(
                model, test_loader, device, criterion,
                pgd_cfg=pgd_cfg,
                eval_name=f"eval-test [{ep}/{args.epochs}]",
            )

            ## Evaluation on Train subset (same size as test set) ##
            train_metrics = evaluate_per_epoch(
                model, subtrain_loader, device, criterion,
                pgd_cfg=pgd_cfg,
                eval_name=f"eval-trainS [{ep}/{args.epochs}]",
            )

            # Build log message — only show enabled metrics.
            def _pct(v): return f"{v*100:.2f}%" if v is not None else None

            def _line(prefix, m):
                parts = []
                if m["clean_loss"] is not None:
                    parts.append(f"loss={m['clean_loss']:.4f}")
                parts.append(f"clean={_pct(m['clean_acc'])}")
                if m["pgd_acc"] is not None:
                    parts.append(f"pgd{args.pgd_steps}={_pct(m['pgd_acc'])}")
                return f"  {prefix}: " + " ".join(parts)

            log_msg = (
                f"[{ep:03d}/{args.epochs}] "
                f"lr={current_lr:.5f} time={elapsed:.1f}s "
                f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%\n"
                + _line("trainS", train_metrics) + "\n"
                + _line("val   ", test_metrics)
            )

            logger.info(log_msg)

            row.update({
                'trainS_loss': train_metrics['clean_loss'],
                'trainS_acc':  train_metrics['clean_acc'],
                'trainS_pgd':  train_metrics['pgd_acc'],
                'val_loss':    test_metrics['clean_loss'],
                'val_acc':     test_metrics['clean_acc'],
                'val_pgd':     test_metrics['pgd_acc'],
            })
            model.train()

        # One CSV row per epoch. The eval columns stay empty on the epochs that
        # skip evaluation, so every row has the same schema.
        training_history.append(row)
        pd.DataFrame(training_history).to_csv(info_csv_path, index=False)


    # Save last checkpoint
    ckpt = {
        "epoch": ep,
        "arch": args.arch,
        "dataset": args.dataset,
        "img_size": img_size,
        "training_type": args.training_type,
        "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    }
    if args.training_type == "level":
        ckpt["level_config"] = level_config

    torch.save(ckpt, out_path)
    logger.info(f"  -> saved last checkpoint to {out_path}")


if __name__ == "__main__":
    main()
