# train_classifiers_pr.py - Train image classifiers with standard or
#                            probabilistic robustness (PR) training using local entropy.
#
# Description:
#   This script trains an image classifier on CIFAR-10, CIFAR-100, or
#   TinyImageNet using one of two training methods:
#
#     standard     - Vanilla cross-entropy training on clean images.
#     loc_entropy  - Probabilistic Robustness training with Local Entropy.
#                    Perturbations are sampled using local-entropy Langevin
#                    dynamics; the model is trained over N perturbation
#                    particles per image with TRADES-style KL regularization.
#
#   For every training run the script saves:
#     <save_dir>/<arch>_<dataset>_<training_type>.pth          model checkpoint
#     <save_dir>/<arch>_<dataset>_<training_type>.log          training log
#     <save_dir>/<arch>_<dataset>_<training_type>_training_info.csv  per-epoch metrics
#
#   Evaluation is run every 5 epochs on a fixed subset of the training set
#   (same size as the test set) and the full test set, reporting clean
#   accuracy for both splits.
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
#   src/local_entropy4pr.py     Local-entropy Langevin dynamics for PR training
#
# Usage:
#   python scripts/train_classifiers_pr.py [options]
#
# Key arguments:
#   --dataset       {cifar10, cifar100, tinyimagenet}  (default: cifar10)
#   --arch          {resnet18, resnet50, wide_resnet50_2, vgg16,
#                    densenet121, mobilenet_v3_large,
#                    efficientnet_b0, vit_b_16}         (default: resnet18)
#   --training_type {standard, loc_entropy}            (default: loc_entropy)
#   --epochs        number of training epochs           (default: 100)
#   --batch_size    mini-batch size                     (default: 128)
#   --lr            initial learning rate               (default: 0.01)
#   --save_dir      output directory                    (default: ./ckp/pr_training)
#   --device        compute device, e.g. cuda or cpu    (default: cuda)
#
#   PR specific (local entropy):
#   --epsilon           perturbation budget                 (default: 8/255)
#   --norm              norm constraint (linf/l2)           (default: linf)
#   --num_particles     number of particles per sample      (default: 8)
#   --langevin_steps    number of Langevin steps            (default: 5)
#   --step_size         Langevin step size                  (default: 1e-2)
#   --noise_scale       Langevin noise scale                (default: 1.0)
#   --gamma             localization strength               (default: 1.0)
#   --beta_outer        TRADES KL regularization weight     (default: 6.0)
#   --psi_type          energy function (softplus/hinge)    (default: softplus)
#   --psi_alpha         energy smoothing parameter          (default: 10.0)
#   --threshold_mode    threshold update (fixed/adaptive)   (default: fixed)
#   --t0                initial/fixed threshold             (default: 0.0)
#   --t_floor           minimum threshold for adaptive      (default: 0.0)
#   --scope_mode        scope update (fixed/dynamic)        (default: fixed)
#   --init_method       particle initialization method      (default: uniform)
#
# Examples:
#   # Standard training on CIFAR-10 with ResNet-18
#   python scripts/train_classifiers_pr.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type standard --epochs 100
#
#   # Local-Entropy training on CIFAR-10 with ResNet-18
#   python scripts/train_classifiers_pr.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type loc_entropy \
#       --num_particles 8 --langevin_steps 5 --gamma 1.0 \
#       --epochs 100 --save_dir ./ckp/pr_training
#
#   # Local-Entropy training on CIFAR-100 with WideResNet-50-2
#   python scripts/train_classifiers_pr.py \
#       --dataset cifar100 --arch wide_resnet50_2 \
#       --training_type loc_entropy \
#       --num_particles 8 --langevin_steps 5 --gamma 1.0 --beta_outer 6.0 \
#       --threshold_mode adaptive --scope_mode dynamic \
#       --epochs 100 --save_dir ./ckp/pr_training

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
from src.local_entropy4pr import (
    ParticleState,
    EnergyConfig,
    LangevinConfig,
    compute_margins,
    fixed_threshold_update,
    adaptive_threshold_update,
    fixed_scope,
    dynamic_scope,
    langevin_update_local_entropy,
    local_entropy_trades_loss,
)
from src.adv_attacker import pgd_attack
from utils.corrupter import apply_corruption_batch, CORRUPTION_FNS
from utils.utils import pr_random_generator

def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("train_classifiers_pr")
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


def local_entropy_generator(model, x, y, **kwargs):
    """
    Wrapper function for local-entropy adversarial generation.
    Compatible with evaluation interface.

    Returns:
        x_adv: adversarial examples, shape (B, N, C, H, W)
    """
    # Extract parameters
    norm = kwargs.get("norm", "linf")
    epsilon = kwargs.get("epsilon", 8/255)
    num_particles = kwargs.get("num_particles", 8)
    langevin_steps = kwargs.get("langevin_steps", 5)
    step_size = kwargs.get("step_size", 1e-2)
    noise_scale = kwargs.get("noise_scale", 1.0)
    gamma = kwargs.get("gamma", 1.0)
    psi_type = kwargs.get("psi_type", "softplus")
    psi_alpha = kwargs.get("psi_alpha", 10.0)
    threshold_mode = kwargs.get("threshold_mode", "fixed")
    t0 = kwargs.get("t0", 0.0)
    t_floor = kwargs.get("t_floor", 0.0)
    scope_mode = kwargs.get("scope_mode", "fixed")
    init_method = kwargs.get("init_method", "uniform")

    # Create particle state
    particle_state = ParticleState(epsilon=epsilon, norm=norm, num_particles=num_particles)

    # Create configs
    energy_cfg = EnergyConfig(psi_type=psi_type, psi_alpha=psi_alpha)
    langevin_cfg = LangevinConfig(
        steps=langevin_steps,
        step_size=step_size,
        beta=kwargs.get("langevin_beta", 1.0),
        noise_scale=noise_scale,
    )

    # Initialize particles
    particle_state.init_particles(x, method=init_method, warm_start=False)

    # Update threshold
    if threshold_mode == "fixed":
        # For fixed mode, we only need a dummy tensor with correct shape/device/dtype
        B, N = x.shape[0], num_particles
        margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        t_curr = fixed_threshold_update(margins=margins, state=particle_state, t=t0)
    elif threshold_mode == "adaptive":
        # For adaptive mode, we need actual margins
        margins = compute_margins(model=model, x=x, y=y, state=particle_state)
        t_curr = adaptive_threshold_update(
            margins=margins, state=particle_state, t0=t0, t_floor=t_floor
        )
    else:
        # Default to fixed mode
        B, N = x.shape[0], num_particles
        margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        t_curr = fixed_threshold_update(margins=margins, state=particle_state, t=t0)

    # Update scope
    if scope_mode == "fixed":
        gamma_curr = fixed_scope(t_curr=t_curr, gamma=gamma)
    elif scope_mode == "dynamic":
        gamma_curr = dynamic_scope(t_curr=t_curr, t0=t0, t_floor=t_floor)
    else:
        gamma_curr = fixed_scope(t_curr=t_curr, gamma=gamma)

    # Langevin update
    langevin_update_local_entropy(
        state=particle_state,
        model=model,
        x=x,
        y=y,
        t_curr=t_curr,
        gamma_curr=gamma_curr,
        energy_cfg=energy_cfg,
        cfg=langevin_cfg,
    )

    return particle_state.x_adv


def set_seed(seed: int = 42):
    """Seed Python/NumPy/Torch RNGs and enable cuDNN auto-tuning.

    cudnn.benchmark=True lets cuDNN pick the fastest convolution kernel for
    each input shape; this is a sizeable speedup when activations are large
    (the (B*N, C, H, W) Langevin tensor here). We give up bit-exact
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

def train_one_epoch_pr(model, loader, optimizer, device, criterion,
                        pr_config,
                        particle_state=None,
                        epoch=None, total_epochs=None):
    """
    Local-Entropy Probabilistic Robustness training loop.

    Uses local-entropy Langevin dynamics to generate adversarial particles
    and trains with TRADES-style KL regularization.

    pr_config keys:
        norm            : "linf" | "l2" (default "linf")
        epsilon         : perturbation budget radius (default 8/255)
        num_particles   : number of particles per sample (default 8)
        langevin_steps  : number of Langevin steps (default 5)
        step_size       : Langevin step size (default 1e-2)
        langevin_beta   : inverse temperature in Langevin noise (default 1.0)
        noise_scale     : Langevin noise scale (default 1.0)
        gamma           : localization strength (default 1.0)
        beta_outer      : TRADES KL weight (default 6.0)
        psi_type        : energy function type (default "softplus")
        psi_alpha       : energy smoothing (default 10.0)
        threshold_mode  : "fixed" or "adaptive" (default "fixed")
        t0              : initial/fixed threshold (default 0.0)
        t_floor         : minimum threshold for adaptive (default 0.0)
        scope_mode      : "fixed" or "dynamic" (default "fixed")
        init_method     : particle initialization (default "uniform")
    """

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"PR Train [{epoch}/{total_epochs}]" if epoch else "PR Training", leave=False)

    # Extract config
    norm = pr_config.get("norm", "linf")
    epsilon = pr_config.get("epsilon", 8/255)
    num_particles = pr_config.get("num_particles", 8)
    langevin_steps = pr_config.get("langevin_steps", 5)
    step_size = pr_config.get("step_size", 1e-2)
    langevin_beta = pr_config.get("langevin_beta", 1.0)
    noise_scale = pr_config.get("noise_scale", 1.0)
    gamma = pr_config.get("gamma", 1.0)
    beta_outer = pr_config.get("beta_outer", 6.0)
    psi_type = pr_config.get("psi_type", "softplus")
    psi_alpha = pr_config.get("psi_alpha", 10.0)
    threshold_mode = pr_config.get("threshold_mode", "fixed")
    t0 = pr_config.get("t0", 0.0)
    t_floor = pr_config.get("t_floor", 0.0)
    scope_mode = pr_config.get("scope_mode", "fixed")
    init_method = pr_config.get("init_method", "uniform")

    # Create particle state if not provided
    if particle_state is None:
        particle_state = ParticleState(epsilon=epsilon, norm=norm, num_particles=num_particles)

    # Create configs
    energy_cfg = EnergyConfig(psi_type=psi_type, psi_alpha=psi_alpha)
    langevin_cfg = LangevinConfig(
        steps=langevin_steps,
        step_size=step_size,
        beta=langevin_beta,
        noise_scale=noise_scale,
    )

    # Stats accumulators
    stat_sums = {"avg_margin": 0.0, "threshold": 0.0, "gamma": 0.0}
    stat_count = 0

    # Initialize return values
    train_acc = 0.0
    avg_stats = {"avg_margin": 0.0, "threshold": 0.0, "gamma": 0.0}

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Initialize particles
        particle_state.init_particles(x, method=init_method, warm_start=False)

        # Update threshold
        if threshold_mode == "fixed":
            # For fixed mode, we only need a dummy tensor with correct shape/device/dtype
            # No need to compute actual margins (saves a forward pass)
            B, N = x.shape[0], particle_state.num_particles
            margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
            t_curr = fixed_threshold_update(margins=margins, state=particle_state, t=t0)
        elif threshold_mode == "adaptive":
            # For adaptive mode, we need actual margins
            margins = compute_margins(model=model, x=x, y=y, state=particle_state)
            t_curr = adaptive_threshold_update(
                margins=margins, state=particle_state, t0=t0, t_floor=t_floor
            )
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        # Update scope
        if scope_mode == "fixed":
            gamma_curr = fixed_scope(t_curr=t_curr, gamma=gamma)
        elif scope_mode == "dynamic":
            gamma_curr = dynamic_scope(t_curr=t_curr, t0=t0, t_floor=t_floor)
        else:
            raise ValueError(f"Unknown scope_mode: {scope_mode}")

        # Langevin update
        langevin_update_local_entropy(
            state=particle_state,
            model=model,
            x=x,
            y=y,
            t_curr=t_curr,
            gamma_curr=gamma_curr,
            energy_cfg=energy_cfg,
            cfg=langevin_cfg,
        )

        # Compute TRADES loss; reuse logits_adv for train_acc to avoid
        # an extra (B*N) forward pass.
        loss, logits_adv = local_entropy_trades_loss(
            model=model,
            x=x,
            y=y,
            state=particle_state,
            criterion=criterion,
            beta_outer=beta_outer,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        # Compute training accuracy on adversarial examples
        B, N = particle_state.x_adv.shape[0], particle_state.x_adv.shape[1]
        with torch.no_grad():
            y_rep = y.repeat_interleave(N)
            preds = logits_adv.argmax(dim=1)
            running_correct += (preds == y_rep).sum().item()

        # Accumulate stats
        stat_sums["avg_margin"] += margins.mean().item()
        stat_sums["threshold"] += t_curr.mean().item()
        stat_sums["gamma"] += gamma_curr.mean().item()
        stat_count += 1

        avg_loss = running_loss / total_samples
        avg_stats = {k: stat_sums[k] / stat_count for k in stat_sums}
        train_acc = running_correct / (total_samples * N)

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            margin=f"{avg_stats['avg_margin']:.3f}",
            t=f"{avg_stats['threshold']:.3f}",
            acc=f"{train_acc:.4f}"
        )

    return running_loss / len(loader.dataset), train_acc, avg_stats


# ------------------------------------------------------------------
#  Single-pass per-epoch evaluation: clean / PGD-10 / LocEnt / RandomPR /
#                                    Corruptions
# ------------------------------------------------------------------

def evaluate_per_epoch(
    model, loader, device, criterion,
    pgd_cfg=None, locent_cfg=None, random_cfgs=None,
    corruptions=None, severities=None,
    eval_name="eval",
):
    """Single-pass eval over loader. Each extra metric is gated by its config
    being non-None (and non-empty for corruptions/random_cfgs); disabled
    blocks are skipped entirely so the cost goes to zero.

    `random_cfgs` is a *list* of pr_random_generator kwargs (one entry per
    distribution to evaluate). Result includes one entry per distribution
    in `random_pr_breakdown`.

    Always returns: clean_acc, clean_loss, num_samples.
    Conditionally returns (None when disabled):
        pgd_acc, locent_pr, random_pr_breakdown, corr_acc, corr_breakdown.
    """
    do_pgd    = pgd_cfg is not None
    do_locent = locent_cfg is not None
    do_random = bool(random_cfgs)
    do_corr   = bool(corruptions) and bool(severities)

    model.eval()
    n_total = 0
    n_clean_correct = 0
    clean_loss_sum  = 0.0
    # Initialise unconditionally; the do_* flags gate accumulation and the
    # return value below — keeping these as plain 0 / 0.0 makes them easy to
    # type-check (no Optional arithmetic).
    n_pgd_correct = 0
    sum_locent_pr = 0.0
    random_sum = {cfg["noise_dist"]: 0.0 for cfg in (random_cfgs or [])}
    corr_correct = {(c, s): 0 for c in (corruptions or []) for s in (severities or [])}

    pbar = tqdm(loader, desc=eval_name, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = y.size(0)
        n_total += B

        # 1) Clean — always
        with torch.no_grad():
            clean_logits = model(x)
            n_clean_correct += (clean_logits.argmax(dim=1) == y).sum().item()
            if criterion is not None:
                clean_loss_sum += criterion(clean_logits, y).item() * B

        # 2) PGD — uses autograd.grad on inputs; do NOT wrap in no_grad.
        if do_pgd:
            x_pgd = pgd_attack(model, x, y, **pgd_cfg)
            with torch.no_grad():
                n_pgd_correct += (model(x_pgd).argmax(dim=1) == y).sum().item()

        # 3) LocEnt mean PR
        if do_locent:
            x_locent = local_entropy_generator(model, x, y, **locent_cfg)  # (B, N, C, H, W)
            N_loc = x_locent.shape[1]
            with torch.no_grad():
                preds = model(x_locent.reshape(B * N_loc, *x_locent.shape[2:])
                              ).argmax(dim=1).view(B, N_loc)
                sum_locent_pr += (preds == y.unsqueeze(1)).float().mean(dim=1).sum().item()

        # 4) Random-PR baseline — one pass per distribution
        if do_random:
            for cfg in random_cfgs:
                x_rand, _ = pr_random_generator(model, x, y, **cfg)
                N_rnd = x_rand.shape[1]
                with torch.no_grad():
                    preds = model(x_rand.reshape(B * N_rnd, *x_rand.shape[2:])
                                  ).argmax(dim=1).view(B, N_rnd)
                    random_sum[cfg["noise_dist"]] += (
                        preds == y.unsqueeze(1)
                    ).float().mean(dim=1).sum().item()

        # 5) Corruptions — each (name, severity) is a single forward.
        if do_corr:
            for cname in corruptions:
                for sev in severities:
                    x_corr = apply_corruption_batch(x, cname, sev)
                    with torch.no_grad():
                        corr_correct[(cname, sev)] += (
                            model(x_corr).argmax(dim=1) == y
                        ).sum().item()

        # tqdm postfix — only show enabled metrics
        post = {"clean": f"{n_clean_correct/n_total:.3f}"}
        if do_pgd:    post["pgd"]  = f"{n_pgd_correct/n_total:.3f}"
        if do_locent: post["loc"]  = f"{sum_locent_pr/n_total:.3f}"
        if do_random: post["rnd"]  = f"{sum(random_sum.values())/(n_total*len(random_sum)):.3f}"
        pbar.set_postfix(**post)

    random_breakdown = {d: s / n_total for d, s in random_sum.items()} if do_random else None
    corr_breakdown   = {k: v / n_total for k, v in corr_correct.items()} if do_corr else None
    corr_mean = (sum(corr_breakdown.values()) / max(1, len(corr_breakdown))) if do_corr else None

    return {
        "clean_acc":         n_clean_correct / n_total,
        "clean_loss":        (clean_loss_sum / n_total) if criterion is not None else None,
        "pgd_acc":           (n_pgd_correct / n_total) if do_pgd    else None,
        "locent_pr":         (sum_locent_pr / n_total) if do_locent else None,
        "random_pr_breakdown": random_breakdown,   # {dist: acc} or None
        "corr_acc":          corr_mean,
        "corr_breakdown":    corr_breakdown,
        "num_samples":       n_total,
    }


# ------------------------------------------------------------------
#  AutoAttack evaluation (slow — call only at the final epoch)
# ------------------------------------------------------------------

def evaluate_aa(model, loader, device, norm, epsilon, version="rand", eval_name="eval-AA"):
    """Run AutoAttack over the loader. Returns the adversarial accuracy.

    AA is hugely expensive — APGD-CE + APGD-DLR (rand) or the full standard
    suite (APGD-CE + APGD-DLR + FAB + Square). Use only as a final benchmark.
    """
    import autoattack
    norm_str = "Linf" if norm.lower() == "linf" else "L2"
    model.eval()

    adversary = autoattack.AutoAttack(
        model, norm=norm_str, eps=float(epsilon),
        version=version, verbose=False,
    )

    n_total = 0
    n_correct = 0
    pbar = tqdm(loader, desc=eval_name, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        with torch.no_grad():
            n_correct += (model(x_adv).argmax(dim=1) == y).sum().item()
        n_total += y.size(0)
        pbar.set_postfix(aa=f"{n_correct/n_total:.3f}")

    return n_correct / n_total


# ------------------------------------------------------------------
#                           Main Function
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # ============================================================
    # Dataset & model
    # ============================================================
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", choices=[
            "resnet18", "resnet50", "wide_resnet50_2",
            "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
            "vit_b_16",
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
    ap.add_argument("--training_type", choices=["standard", "loc_entropy"], default="loc_entropy",
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
    ap.add_argument("--num_particles", type=int, default=8,
                    help="Number of perturbation particles per input.")
    ap.add_argument("--init_method", type=str, default="uniform", choices=["zero", "gaussian", "uniform"],
                    help="Particle initialization method.")
    ap.add_argument("--init_scale", type=float, default=None,
                    help="Scale for Gaussian or approximate L2 initialization. If None, use epsilon.")
    ap.add_argument("--warm_start", action="store_true",
                    help="Reuse previous particles when possible. Usually keep False for shuffled dataloaders.")
    # ============================================================
    # Langevin dynamics
    # ============================================================
    ap.add_argument("--langevin_steps", type=int, default=5,
                    help="Number of Langevin steps.")
    ap.add_argument("--step_size", type=float, default=1e-2,
                    help="Langevin step size.")
    ap.add_argument("--langevin_beta", type=float, default=100.0,
                    help="Inverse temperature beta in Langevin noise sqrt(2 eta / beta).")
    ap.add_argument("--noise_scale", type=float, default=1.0,
                    help="Extra multiplier for Langevin noise.")
    # ============================================================
    # Energy function
    # ============================================================
    ap.add_argument("--psi_type", type=str, default="softplus", choices=["softplus", "hinge"],
                    help="Thresholded energy function type.")
    ap.add_argument("--psi_alpha", type=float, default=10.0,
                    help="Softplus smoothing parameter. Larger means closer to hinge.")
    # ============================================================
    # Threshold strategy
    # ============================================================
    ap.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "adaptive"],
                    help="Threshold update mode.")
    ap.add_argument("--t0", type=float, default=-0.05,
                    help="Initial threshold for adaptive mode, or fixed threshold if threshold_mode=fixed.")
    ap.add_argument("--t_floor", type=float, default=0.0,
                    help="Minimum threshold for adaptive mode.")
    ap.add_argument("--threshold_q", type=float, default=0.4,
                    help="Quantile level over particles for adaptive threshold.")
    ap.add_argument("--threshold_delta_min", type=float, default=0.01,
                    help="Minimum threshold decrease per particle-generation call.")
    ap.add_argument("--threshold_decay", type=float, default=0.995,
                    help="Global exponential decay factor for threshold schedule.")
    # ============================================================
    # Scope strategy
    # ============================================================
    ap.add_argument("--scope_mode", type=str, default="fixed", choices=["fixed", "dynamic"], 
                    help="Scope update mode.")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Fixed localization strength when scope_mode=fixed.")
    ap.add_argument("--gamma_min", type=float, default=0.1, 
                    help="Minimum localization strength for dynamic scope.")
    ap.add_argument("--gamma_max", type=float, default=10.0, 
                    help="Maximum localization strength for dynamic scope.")
    # ============================================================
    # Outer TRADES-style loss
    # ============================================================
    ap.add_argument("--beta_outer", type=float, default=12.0,
                    help="TRADES KL regularization weight.")
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

    # Local-Entropy mean-PR (eval-time overrides; other LocEnt params are
    # inherited from the training pr_config so gamma/beta/threshold etc.
    # match training).
    ap.add_argument("--eval_locent", action="store_true",
                    help="Run Local-Entropy mean-PR eval at every eval cycle.")
    ap.add_argument("--locent_n", type=int, default=8,
                    help="Number of LocEnt particles at eval (overrides training --num_particles).")
    ap.add_argument("--locent_steps", type=int, default=10,
                    help="Number of LocEnt Langevin steps at eval (overrides training --langevin_steps).")
    ap.add_argument("--locent_norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for the LocEnt eval attack.")

    # Random-noise PR baseline. --random_dist accepts multiple distributions;
    # one PR score is reported per distribution.
    ap.add_argument("--eval_random", action="store_true",
                    help="Run random-noise PR eval at every eval cycle.")
    ap.add_argument("--random_n", type=int, default=8,
                    help="Number of random draws per sample.")
    ap.add_argument("--random_norm", choices=["linf", "l2"], default="linf",
                    help="Norm for random-PR projection.")
    ap.add_argument("--random_dist", nargs="+",
                    choices=["gaussian", "uniform", "laplace"],
                    default=["gaussian"],
                    help="One or more sampling distributions for random-PR.")

    # Corruptions
    ap.add_argument("--eval_corruptions", action="store_true",
                    help="Run corruption eval at every eval cycle.")
    ap.add_argument("--corruption_names", nargs="+",
                    default=list(CORRUPTION_FNS.keys()),
                    choices=list(CORRUPTION_FNS.keys()),
                    help="Which corruption types to evaluate.")
    ap.add_argument("--corruption_severities", nargs="+", type=int,
                    default=[1, 3, 5],
                    help="Severity levels (1-5) for corruption eval.")

    # AutoAttack (final epoch only)
    ap.add_argument("--eval_aa", action="store_true",
                    help="Run AutoAttack at the final epoch (slow).")
    ap.add_argument("--aa_version", choices=["standard", "plus", "rand"],
                    default="rand",
                    help="AutoAttack version.")
    ap.add_argument("--aa_norm", choices=["linf", "l2"], default="linf",
                    help="Norm for AutoAttack.")
    # ============================================================
    # Misc
    # ============================================================
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/dignoise/local_entropy_training",
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
    elif args.training_type == "loc_entropy":
        logger.info(f"[config] training_type={args.training_type} (local-entropy), epsilon={args.epsilon:.4f}, norm={args.norm}")
        logger.info(f"         gamma={args.gamma}, beta_outer={args.beta_outer}")
        logger.info(f"         num_particles={args.num_particles}, langevin_steps={args.langevin_steps}, step_size={args.step_size}")
        logger.info(f"         noise_scale={args.noise_scale}, psi_type={args.psi_type}, psi_alpha={args.psi_alpha}")
        logger.info(f"         threshold_mode={args.threshold_mode}, t0={args.t0}, t_floor={args.t_floor}")
        logger.info(f"         scope_mode={args.scope_mode}, init_method={args.init_method}")
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

    # PR config (local entropy)
    pr_config = {
        "type": args.training_type,
        "norm": args.norm,
        "epsilon": args.epsilon,
        "num_particles": args.num_particles,
        "langevin_steps": args.langevin_steps,
        "step_size": args.step_size,
        "langevin_beta": args.langevin_beta,
        "noise_scale": args.noise_scale,
        "gamma": args.gamma,
        "beta_outer": args.beta_outer,
        "psi_type": args.psi_type,
        "psi_alpha": args.psi_alpha,
        "threshold_mode": args.threshold_mode,
        "t0": args.t0,
        "t_floor": args.t_floor,
        "scope_mode": args.scope_mode,
        "init_method": args.init_method,
    }

    # Output path
    out_path = os.path.join(args.save_dir, f"{name_tag}.pth")
    info_csv_path = os.path.join(args.save_dir, f"{name_tag}_training_info.csv")
    logger.info(f"[save] checkpoint -> {out_path}")
    logger.info(f"[save] log       -> {log_path}")
    logger.info(f"[save] csv       -> {info_csv_path}")

    # Train
    ep = 0  # Initialize epoch counter
    for ep in range(1, args.epochs + 1):
        start = time.time()
        if args.training_type == "standard":
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                                    epoch=ep, total_epochs=args.epochs)
            avg_stats = {}  # Empty stats for standard training
        elif args.training_type == "loc_entropy":
            train_loss, train_acc, avg_stats = train_one_epoch_pr(model, train_loader, optimizer, device, criterion, pr_config,
                                                                  epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        scheduler.step()

        # Evaluation and checkpointing
        if ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - start

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

            # LocEnt: inherit pr_config, override n/steps/norm for eval.
            locent_cfg = None
            if args.eval_locent:
                locent_cfg = dict(pr_config)
                locent_cfg["num_particles"]  = args.locent_n
                locent_cfg["langevin_steps"] = args.locent_steps
                locent_cfg["norm"]           = args.locent_norm

            # Random-PR baseline — one cfg per requested distribution.
            random_cfgs = None
            if args.eval_random:
                random_cfgs = [
                    {
                        "epsilon":      args.epsilon,
                        "norm":         args.random_norm,
                        "num_samples":  args.random_n,
                        "noise_dist":   d,
                        "return_stats": False,
                    }
                    for d in args.random_dist
                ]

            # Corruptions.
            corr_names = args.corruption_names if args.eval_corruptions else None
            corr_severities = args.corruption_severities if args.eval_corruptions else None

            ## Evaluation on Test set ##
            test_metrics = evaluate_per_epoch(
                model, test_loader, device, criterion,
                pgd_cfg=pgd_cfg, locent_cfg=locent_cfg, random_cfgs=random_cfgs,
                corruptions=corr_names, severities=corr_severities,
                eval_name=f"eval-test [{ep}/{args.epochs}]",
            )

            ## Evaluation on Train subset (same size as test set) ##
            train_metrics = evaluate_per_epoch(
                model, subtrain_loader, device, criterion,
                pgd_cfg=pgd_cfg, locent_cfg=locent_cfg, random_cfgs=random_cfgs,
                corruptions=corr_names, severities=corr_severities,
                eval_name=f"eval-trainS [{ep}/{args.epochs}]",
            )

            # ----- AutoAttack: only at the final epoch (huge cost). -----
            aa_test = aa_train = None
            if args.eval_aa and ep == args.epochs:
                logger.info(f"[AA] running AutoAttack ({args.aa_version}, "
                            f"{args.aa_norm}) on test set and train subset "
                            f"— this can take a while.")
                aa_test = evaluate_aa(
                    model, test_loader, device,
                    norm=args.aa_norm, epsilon=args.epsilon, version=args.aa_version,
                    eval_name=f"AA-test [{ep}/{args.epochs}]",
                )
                aa_train = evaluate_aa(
                    model, subtrain_loader, device,
                    norm=args.aa_norm, epsilon=args.epsilon, version=args.aa_version,
                    eval_name=f"AA-trainS [{ep}/{args.epochs}]",
                )

            current_lr = scheduler.get_last_lr()[0]

            # Build log message — only show enabled metrics.
            def _pct(v): return f"{v*100:.2f}%" if v is not None else None

            def _line(prefix, m, aa):
                parts = []
                if m["clean_loss"] is not None:
                    parts.append(f"loss={m['clean_loss']:.4f}")
                parts.append(f"clean={_pct(m['clean_acc'])}")
                if m["pgd_acc"]    is not None: parts.append(f"pgd{args.pgd_steps}={_pct(m['pgd_acc'])}")
                if m["locent_pr"]  is not None: parts.append(f"locent={_pct(m['locent_pr'])}")
                # Random PR: one per distribution.
                rb = m.get("random_pr_breakdown")
                if rb:
                    if len(rb) == 1:
                        d, v = next(iter(rb.items()))
                        parts.append(f"rand_{d[:1]}={_pct(v)}")
                    else:
                        parts.append("rand=[" + " ".join(
                            f"{d[:1]}={_pct(v)}" for d, v in rb.items()) + "]")
                if m["corr_acc"]   is not None: parts.append(f"corr={_pct(m['corr_acc'])}")
                if aa is not None:              parts.append(f"aa={_pct(aa)}")
                return f"  {prefix}: " + " ".join(parts)

            log_msg = (
                f"[{ep:03d}/{args.epochs}] "
                f"lr={current_lr:.5f} time={elapsed:.1f}s "
                f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%\n"
                + _line("trainS", train_metrics, aa_train) + "\n"
                + _line("val   ", test_metrics,  aa_test)
            )

            # Add PR stats if available
            if args.training_type == "loc_entropy" and avg_stats:
                log_msg += (
                    f"\n  PR stats: margin={avg_stats['avg_margin']:.3f} "
                    f"t={avg_stats['threshold']:.3f} "
                    f"gamma={avg_stats['gamma']:.3f}"
                )

            logger.info(log_msg)

            # CSV: keep a stable schema across epochs even when an eval is
            # disabled — pandas writes None as an empty cell. Random-PR is
            # expanded into one column per distribution so the schema stays
            # flat regardless of which dists were enabled.
            def _r(m, dist):
                rb = m.get("random_pr_breakdown")
                return rb.get(dist) if rb else None

            epoch_info = {
                'arch':                args.arch,
                'dataset':             args.dataset,
                'training_type':       args.training_type,
                'epoch':               ep,
                'lr':                  current_lr,
                'time':                elapsed,
                'train_loss':          train_loss,
                'train_acc':           train_acc,
                # train subset (no augmentation) metrics
                'trainS_loss':         train_metrics['clean_loss'],
                'trainS_acc':          train_metrics['clean_acc'],
                'trainS_pgd':          train_metrics['pgd_acc'],
                'trainS_locent':       train_metrics['locent_pr'],
                'trainS_random_g':     _r(train_metrics, 'gaussian'),
                'trainS_random_u':     _r(train_metrics, 'uniform'),
                'trainS_random_l':     _r(train_metrics, 'laplace'),
                'trainS_corr':         train_metrics['corr_acc'],
                'trainS_aa':           aa_train,
                # test set metrics
                'val_loss':            test_metrics['clean_loss'],
                'val_acc':             test_metrics['clean_acc'],
                'val_pgd':             test_metrics['pgd_acc'],
                'val_locent':          test_metrics['locent_pr'],
                'val_random_g':        _r(test_metrics, 'gaussian'),
                'val_random_u':        _r(test_metrics, 'uniform'),
                'val_random_l':        _r(test_metrics, 'laplace'),
                'val_corr':            test_metrics['corr_acc'],
                'val_aa':              aa_test,
            }
            if args.training_type == "loc_entropy" and avg_stats:
                epoch_info.update(avg_stats)
            training_history.append(epoch_info)

            # overwrite CSV with the full history so far
            pd.DataFrame(training_history).to_csv(info_csv_path, index=False)
            logger.info(f"  -> saved training info to {info_csv_path}")

            model.train()


    # Save last checkpoint
    ckpt = {
        "epoch": ep,
        "arch": args.arch,
        "dataset": args.dataset,
        "img_size": img_size,
        "training_type": args.training_type,
        "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    }
    if args.training_type == "loc_entropy":
        ckpt["pr_config"] = pr_config

    torch.save(ckpt, out_path)
    logger.info(f"  -> saved last checkpoint to {out_path}")


if __name__ == "__main__":
    main()
