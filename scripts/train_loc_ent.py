#!/usr/bin/env python3
"""
Training script: Region-Aware Adversarial Training (Local Entropy / Langevin).

Outer objective per batch:
    L = CE(f(x), y)  +  lambda_kl * KL( p_theta(x) || p_bar )
where p_bar = (1/M) * sum_i p_theta(x + delta_i) is the particle-averaged
predictive distribution produced by region_generator.

Usage:
    python scripts/train_loc_ent.py --arch resnet18 --dataset cifar10 \\
        --epsilon 0.0314 --norm linf --num_particles 8 --inner_steps 3

Example:
    python scripts/train_loc_ent.py \\
        --dataset cifar10 --arch resnet18 \\
        --epsilon 0.0314 --num_particles 8 --inner_steps 3 \\
        --lambda_kl 6.0 --eta_delta 0.01 --t0 0.5 \\
        --epochs 100 --batch_size 128 --lr 0.01 \\
        --save_dir ./ckp/loc_ent
"""

import os
import logging
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pandas as pd

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from src.local_entropy4pr import region_generator


# ------------------------------------------------------------------
# Evaluator adapter
# ------------------------------------------------------------------

def _eval_region_generator(model, x, y, **kwargs):
    """
    Thin wrapper so Evaluator.evaluate_pr() can call region_generator.
    Strips state_out and p_bar; returns (x_adv, stats) as expected by
    the pr_transform adapter in utils/evaluator.py.
    """
    kwargs.pop("state", None)
    x_adv, stats, _, _ = region_generator(model, x, y, **kwargs)
    return x_adv, stats


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_loc_ent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def generate_exp_name(args):
    """
    Auto-generate experiment name.
    Format: {arch}_{dataset}_{norm}(eps{eps255})_M{M}_L{L}_lkl{lkl}_t0{t0}_psi({psi})
    """
    eps_255 = int(round(args.epsilon * 255))
    parts = [
        args.arch,
        args.dataset,
        f"{args.norm}(eps{eps_255})",
        f"M{args.num_particles}",
        f"L{args.inner_steps}",
        f"lkl{args.lambda_kl}",
        f"t0{args.t0}",
        f"psi({args.psi_type})",
    ]
    return "_".join(parts)


# ------------------------------------------------------------------
# Training for one epoch
# ------------------------------------------------------------------

def train_one_epoch_loc_ent(
    model, loader, optimizer, device, criterion,
    loc_ent_config, epoch=None, total_epochs=None,
):
    """
    One epoch of Region-Aware Local-Entropy Adversarial Training.

    Per batch:
      1. Generate M adversarial particles + particle-averaged p_bar
         via region_generator (inner Langevin loop).
      2. Compute outer loss: L_nat + lambda_kl * KL(p_clean || p_bar).
      3. Backward + optimizer step.
      4. Track robust accuracy on the first particle as a proxy.

    loc_ent_config keys:
        All region_generator kwargs, plus:
        lambda_kl   : KL regularisation weight (float)
        detach_p_bar: if True (default), treat p_bar as a fixed reference
                      (TRADES-style); if False, also backprop through p_bar.

    Returns:
        avg_loss, avg_loss_nat, avg_loss_kl, train_rob_acc, avg_stats
    """
    model.train()

    lambda_kl    = loc_ent_config["lambda_kl"]
    detach_p_bar = loc_ent_config.get("detach_p_bar", True)
    # Keys that are not region_generator parameters
    _outer_keys  = {"lambda_kl", "detach_p_bar"}
    gen_kwargs   = {k: v for k, v in loc_ent_config.items() if k not in _outer_keys}

    running_loss     = 0.0
    running_loss_nat = 0.0
    running_loss_kl  = 0.0
    running_correct  = 0
    total_samples    = 0

    # region_generator stat keys we track in the progress bar / history
    _stat_keys = [
        "t_prev_mean", "t_next_mean", "gamma_mean",
        "margin_curr_mean", "margin_final_mean",
        "p_bar_entropy", "spread_anchor", "spread_center",
    ]
    stat_sums  = {k: 0.0 for k in _stat_keys}
    stat_count = 0

    # step_idx ticks once per epoch so the threshold schedule is epoch-level
    step_idx = (epoch - 1) if epoch else 0

    desc = f"LocEnt Train [{epoch}/{total_epochs}]" if epoch else "LocEnt Training"
    pbar = tqdm(loader, desc=desc, leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------
        # Inner loop: generate adversarial particles + p_bar
        # region_generator saves/restores model train/eval state.
        # ----------------------------------------------------------
        x_adv, inner_stats, _, p_bar = region_generator(
            model, x, y,
            step_idx=step_idx,
            return_stats=True,
            **gen_kwargs,
        )
        # x_adv: (B, M, C, H, W)   p_bar: (B, C)

        # ----------------------------------------------------------
        # Outer objective  (model is back in train mode here)
        # ----------------------------------------------------------
        logits_clean = model(x)                              # (B, C)
        loss_nat = criterion(logits_clean, y)

        p_clean = F.softmax(logits_clean, dim=1)             # (B, C)
        p_ref   = p_bar.detach() if detach_p_bar else p_bar
        # KL(p_clean || p_bar) = F.kl_div(log_p_bar, p_clean)
        loss_kl = F.kl_div(
            torch.log(p_ref.clamp_min(1e-12)),
            p_clean,
            reduction="batchmean",
        )

        loss = loss_nat + lambda_kl * loss_kl
        loss.backward()
        optimizer.step()

        # ----------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------
        B = y.size(0)
        running_loss     += loss.item()     * B
        running_loss_nat += loss_nat.item() * B
        running_loss_kl  += loss_kl.item()  * B
        total_samples    += B

        # Robust accuracy: evaluate on the first particle as a cheap proxy
        model.eval()
        with torch.no_grad():
            preds_adv = model(x_adv[:, 0]).argmax(dim=1)
            running_correct += (preds_adv == y).sum().item()
        model.train()

        # Accumulate region_generator diagnostics
        if inner_stats is not None:
            for k in _stat_keys:
                if k in inner_stats:
                    stat_sums[k] += float(inner_stats[k])
            stat_count += 1

        avg_loss      = running_loss     / total_samples
        avg_loss_nat  = running_loss_nat / total_samples
        avg_loss_kl   = running_loss_kl  / total_samples
        train_rob_acc = running_correct  / total_samples
        avg_stats     = {k: stat_sums[k] / max(stat_count, 1) for k in _stat_keys}

        pbar.set_postfix(
            loss=f"{avg_loss:.4e}",
            nat =f"{avg_loss_nat:.4e}",
            kl  =f"{avg_loss_kl:.4e}",
            t   =f"{avg_stats['t_next_mean']:.3f}",
            rob =f"{train_rob_acc:.3f}",
        )

    return avg_loss, avg_loss_nat, avg_loss_kl, train_rob_acc, avg_stats


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Region-Aware Adversarial Training (Local Entropy)"
    )

    # ---- Device / reproducibility ----
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)

    # ---- Dataset ----
    parser.add_argument("--dataset",   choices=["cifar10", "cifar100", "tinyimagenet"],
                        default="cifar10")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--img_size",  type=int, default=None,
                        help="Override image size (uses dataset default if not set)")

    # ---- Model ----
    parser.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large",
        "efficientnet_b0", "vit_b_16",
    ], default="resnet18")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialise from ImageNet pretrained weights")

    # ---- Perturbation budget ----
    parser.add_argument("--norm",    choices=["linf", "l2"], default="linf")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Perturbation budget radius (e.g. 0.0314 for 8/255)")

    # ---- Particle population ----
    parser.add_argument("--num_particles", type=int, default=8,
                        help="Number of Langevin particles M")
    parser.add_argument("--inner_steps",   type=int, default=3,
                        help="Langevin steps per particle per call L")
    parser.add_argument("--init_mode",     type=str, default="uniform",
                        choices=["uniform", "zero", "normal"],
                        help="Particle initialisation strategy")

    # ---- Threshold schedule ----
    parser.add_argument("--t0",        type=float, default=0.5,
                        help="Initial margin threshold")
    parser.add_argument("--q",         type=float, default=0.4,
                        help="Quantile level for adaptive threshold")
    parser.add_argument("--delta_min", type=float, default=0.01,
                        help="Minimum threshold decrease per outer step")
    parser.add_argument("--t_floor",   type=float, default=0.0,
                        help="Hard floor on the threshold")
    parser.add_argument("--tau_decay", type=float, default=0.995,
                        help="Exponential decay rate for the global schedule term")

    # ---- Scope schedule ----
    parser.add_argument("--gamma_min", type=float, default=0.1,
                        help="Minimum localization scope (broadest exploration)")
    parser.add_argument("--gamma_max", type=float, default=10.0,
                        help="Maximum localization scope (tightest refinement)")

    # ---- Thresholded energy ----
    parser.add_argument("--psi_type",  type=str, default="softplus",
                        choices=["softplus", "hinge"],
                        help="Energy penalty shape")
    parser.add_argument("--psi_alpha", type=float, default=10.0,
                        help="Softplus steepness parameter")

    # ---- Langevin dynamics ----
    parser.add_argument("--eta_delta",   type=float, default=1e-2,
                        help="Langevin inner step size")
    parser.add_argument("--beta",        type=float, default=1.0,
                        help="Inverse temperature (higher = less noise)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Noise scaling factor (0 = deterministic MAP)")

    # ---- Outer objective ----
    parser.add_argument("--lambda_kl",      type=float, default=6.0,
                        help="Weight on KL(p_clean || p_bar) regularisation term")
    parser.add_argument("--no_detach_p_bar", action="store_true",
                        help="Backprop through p_bar as well (default: detach, TRADES-style)")

    # ---- Optimiser ----
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # ---- LR scheduler ----
    parser.add_argument("--use_lr_scheduler", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable warmup+cosine LR schedule (use --no-use_lr_scheduler to disable)")
    parser.add_argument("--lr_warmup_epochs", type=int,   default=5,
                        help="Linear warmup length (epochs)")
    parser.add_argument("--lr_min",           type=float, default=1e-6,
                        help="Cosine annealing floor")

    # ---- Evaluation ----
    parser.add_argument("--eval_every",       type=int, default=5,
                        help="Run full evaluation every N epochs")
    parser.add_argument("--num_eval_samples", type=int, default=16,
                        help="Number of particles per input during PR evaluation")

    # ---- Saving / logging ----
    parser.add_argument("--save_dir", type=str, default="./ckp/loc_ent",
                        help="Directory for checkpoints, logs, and CSV history")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")

    args = parser.parse_args()

    # ---- Post-parse fixups ----
    args.detach_p_bar = not args.no_detach_p_bar
    if args.exp_name is None:
        args.exp_name = generate_exp_name(args)

    # ============================================================
    # Setup
    # ============================================================
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"{args.exp_name}.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k:28s}: {v}")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    # ============================================================
    # Dataset
    # ============================================================
    img_size = get_img_size(args.dataset, args.img_size)
    logger.info(f"[data] dataset={args.dataset}  img_size={img_size}")

    train_set, num_classes = get_dataset(
        args.dataset, args.data_root, train=True, img_size=img_size, augment=True
    )
    test_set, _ = get_dataset(
        args.dataset, args.data_root, train=False, img_size=img_size, augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Fixed non-augmented train subset for per-epoch monitoring
    # (same size as test set, reproducibly sampled)
    train_set_noaug, _ = get_dataset(
        args.dataset, args.data_root, train=True, img_size=img_size, augment=False
    )
    rng = np.random.default_rng(seed=args.seed)
    subset_idx = rng.choice(len(train_set_noaug), len(test_set), replace=False)
    subtrain_loader = torch.utils.data.DataLoader(
        Subset(train_set_noaug, subset_idx),
        batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(
        f"[data] train eval subset: {len(subset_idx)}/{len(train_set)} samples "
        f"({len(subset_idx)/len(train_set)*100:.0f}%, fixed seed={args.seed})"
    )

    # ============================================================
    # Model
    # ============================================================
    model = build_model(args.arch, num_classes, args.dataset, pretrained=args.pretrained)
    model = model.to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
        logger.info(f"[model] DataParallel across {torch.cuda.device_count()} GPUs")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[model] {args.arch}  trainable params: {total_params:,}")

    # ============================================================
    # Optimiser & LR scheduler
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = None
    if args.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.lr_warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.lr_warmup_epochs, 1),
            eta_min=args.lr_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.lr_warmup_epochs],
        )
        logger.info(
            f"[sched] warmup={args.lr_warmup_epochs} epochs -> cosine to "
            f"lr_min={args.lr_min}"
        )

    # ============================================================
    # Config dicts
    # ============================================================
    loc_ent_config = {
        # perturbation budget
        "norm":          args.norm,
        "epsilon":       args.epsilon,
        # particle population
        "num_particles": args.num_particles,
        "inner_steps":   args.inner_steps,
        "init_mode":     args.init_mode,
        # threshold schedule
        "t0":            args.t0,
        "q":             args.q,
        "delta_min":     args.delta_min,
        "t_floor":       args.t_floor,
        "tau_decay":     args.tau_decay,
        # scope schedule
        "gamma_min":     args.gamma_min,
        "gamma_max":     args.gamma_max,
        # thresholded energy
        "psi_type":      args.psi_type,
        "psi_alpha":     args.psi_alpha,
        # Langevin dynamics
        "eta_delta":     args.eta_delta,
        "beta":          args.beta,
        "noise_scale":   args.noise_scale,
        # outer objective (handled in train_one_epoch_loc_ent, not forwarded to generator)
        "lambda_kl":     args.lambda_kl,
        "detach_p_bar":  args.detach_p_bar,
    }

    # Kwargs for _eval_region_generator (outer-loss keys excluded; step_idx fixed)
    eval_gen_kwargs = {
        "norm":          args.norm,
        "epsilon":       args.epsilon,
        "num_particles": args.num_eval_samples,
        "inner_steps":   args.inner_steps,
        "init_mode":     args.init_mode,
        "t0":            args.t0,
        "q":             args.q,
        "delta_min":     args.delta_min,
        "t_floor":       args.t_floor,
        "tau_decay":     args.tau_decay,
        "gamma_min":     args.gamma_min,
        "gamma_max":     args.gamma_max,
        "psi_type":      args.psi_type,
        "psi_alpha":     args.psi_alpha,
        "eta_delta":     args.eta_delta,
        "beta":          args.beta,
        "noise_scale":   args.noise_scale,
        "step_idx":      0,
    }

    # ============================================================
    # Paths
    # ============================================================
    out_path = os.path.join(args.save_dir, f"{args.exp_name}.pth")
    csv_path = os.path.join(args.save_dir, f"{args.exp_name}_history.csv")

    logger.info(f"[save] checkpoint -> {out_path}")
    logger.info(f"[save] log        -> {log_path}")
    logger.info(f"[save] csv        -> {csv_path}")

    # ============================================================
    # Training loop
    # ============================================================
    training_history = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        avg_loss, avg_loss_nat, avg_loss_kl, train_rob_acc, avg_stats = \
            train_one_epoch_loc_ent(
                model, train_loader, optimizer, device, criterion,
                loc_ent_config,
                epoch=epoch, total_epochs=args.epochs,
            )

        if scheduler is not None:
            scheduler.step()

        elapsed    = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"lr={current_lr:.5f}  time={elapsed:.1f}s  "
            f"loss={avg_loss:.4f}  nat={avg_loss_nat:.4f}  kl={avg_loss_kl:.4f}  "
            f"train_rob={train_rob_acc*100:.2f}%  "
            f"t_next={avg_stats['t_next_mean']:.3f}  "
            f"gamma={avg_stats['gamma_mean']:.2f}"
        )

        # ---- Full evaluation every eval_every epochs ----
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            from utils.evaluator import Evaluator

            model.eval()

            # Test set
            evaluator = Evaluator(model, test_loader, criterion, device)
            clean_test  = evaluator.evaluate_standard()
            pr_test     = evaluator.evaluate_pr(_eval_region_generator, **eval_gen_kwargs)

            # Train subset
            evaluator.update_loader(subtrain_loader)
            clean_train = evaluator.evaluate_standard()
            pr_train    = evaluator.evaluate_pr(_eval_region_generator, **eval_gen_kwargs)

            model.train()

            logger.info(
                f"  [eval/test]  clean_acc={clean_test['acc']*100:.2f}%"
                f"  clean_loss={clean_test['loss']:.4f}"
                f"  pr_acc={pr_test['pr']*100:.2f}%"
            )
            logger.info(
                f"  [eval/train] clean_acc={clean_train['acc']*100:.2f}%"
                f"  clean_loss={clean_train['loss']:.4f}"
                f"  pr_acc={pr_train['pr']*100:.2f}%"
            )

            epoch_info = {
                "epoch":            epoch,
                "lr":               current_lr,
                "time_s":           elapsed,
                # training losses
                "train_loss":       avg_loss,
                "train_loss_nat":   avg_loss_nat,
                "train_loss_kl":    avg_loss_kl,
                "train_rob_acc":    train_rob_acc,
                # test set
                "test_clean_acc":   clean_test["acc"],
                "test_clean_loss":  clean_test["loss"],
                "test_pr_acc":      pr_test["pr"],
                # train subset
                "trainS_clean_acc":  clean_train["acc"],
                "trainS_clean_loss": clean_train["loss"],
                "trainS_pr_acc":     pr_train["pr"],
            }
            # Append region_generator diagnostics with stat_ prefix
            for k, v in avg_stats.items():
                epoch_info[f"stat_{k}"] = v

            training_history.append(epoch_info)

            # Overwrite CSV incrementally for fault-tolerance
            pd.DataFrame(training_history).to_csv(csv_path, index=False)
            logger.info(f"  [save] history -> {csv_path}")

    # ============================================================
    # Save final checkpoint
    # ============================================================
    ckpt = {
        "epoch":          args.epochs,
        "arch":           args.arch,
        "dataset":        args.dataset,
        "img_size":       img_size,
        "training_type":  "loc_ent",
        "loc_ent_config": loc_ent_config,
        "model_state": (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        "config": vars(args),
    }
    torch.save(ckpt, out_path)
    logger.info(f"\n[save] final checkpoint -> {out_path}")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
