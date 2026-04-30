# train_classifiers_adv.py - Train image classifiers with PGD-AT or TRADES
#                            adversarial training, WITHOUT data augmentation.
#
# Description:
#   This script trains an image classifier on CIFAR-10, CIFAR-100, or
#   TinyImageNet using one of three training methods:
#
#     standard   - Vanilla cross-entropy training on clean images.
#     adv_pgd    - PGD adversarial training (Madry et al.).
#                  Inner-loop PGD attack generates adversarial examples;
#                  the model is trained to classify them correctly.
#     trades     - TRADES adversarial training (Zhang et al.).
#                  Adds a KL-divergence regularisation term between clean
#                  and adversarial logits controlled by --beta.
#
#   Unlike train_classifiers.py, the training set is loaded WITHOUT data
#   augmentation (random crop / horizontal flip are disabled). This is useful
#   when one wants the adversarial perturbations to be the only source of
#   input variability seen by the model during training.
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
#   utils/preprocess_data.py    dataset loading and preprocessing (no-aug variant)
#   src/adv_attacker.py         PGD-AT and TRADES loss functions
#
# Usage:
#   python scripts/train_classifiers_adv.py [options]
#
# Key arguments:
#   --dataset       {cifar10, cifar100, tinyimagenet}  (default: cifar10)
#   --arch          {resnet18, resnet50, wide_resnet50_2, vgg16,
#                    densenet121, mobilenet_v3_large,
#                    efficientnet_b0, vit_b_16}         (default: resnet18)
#   --training_type {standard, adv_pgd, trades}        (default: adv_pgd)
#   --epochs        number of training epochs           (default: 50)
#   --batch_size    mini-batch size                     (default: 1024)
#   --lr            initial learning rate               (default: 0.01)
#   --save_dir      output directory                    (default: ./ckp/dignoise/adv_training)
#   --device        compute device, e.g. cuda or cpu    (default: cuda)
#
#   PGD-AT / TRADES specific:
#   --epsilon       perturbation budget                 (default: 8/255)
#   --alpha         PGD step size                       (default: 2/255)
#   --num_steps     number of PGD steps                 (default: 10)
#   --beta          TRADES regularisation weight        (default: 6.0)
#
# Examples:
#   # PGD adversarial training on CIFAR-10 with ResNet-18 (no augmentation)
#   python scripts/train_classifiers_adv.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type adv_pgd \
#       --epsilon 0.03137 --alpha 0.00784 --num_steps 10 \
#       --epochs 100 --save_dir ./ckp/dignoise/adv_training
#
#   # TRADES adversarial training on CIFAR-100 with WideResNet-50-2 (no augmentation)
#   python scripts/train_classifiers_adv.py \
#       --dataset cifar100 --arch wide_resnet50_2 \
#       --training_type trades \
#       --epsilon 0.03137 --alpha 0.00784 --num_steps 10 --beta 6.0 \
#       --epochs 100 --save_dir ./ckp/dignoise/adv_training

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
from src.adv_attacker import pgd_at_loss, trades_loss, pgd_attack
from src.local_entropy4pr import (
    ParticleState,
    EnergyConfig,
    LangevinConfig,
    fixed_threshold_update,
    fixed_scope,
    langevin_update_local_entropy,
)
from utils.corrupter import apply_corruption_batch, CORRUPTION_FNS
from utils.utils import pr_random_generator

def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("train_classifiers_adv")
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
    """Make training as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

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
#                    Adversarial Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch_adv(model, loader, optimizer, device, criterion,
                        adv_config,
                        epoch=None, total_epochs=None):
    """
    Adversarial training loop (outer loop).
    Inner loop (attack generation) is handled by adv_attacker functions.
    train_acc is measured on adversarial examples (robust training accuracy).
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Adv Train [{epoch}/{total_epochs}]" if epoch else "Adv Training", leave=False)

    adv_type  = adv_config["type"]
    norm      = adv_config["norm"]
    epsilon   = adv_config["epsilon"]
    alpha     = adv_config["alpha"]
    num_steps = adv_config["num_steps"]
    beta      = adv_config.get("beta", 6.0)  # TRADES only

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Compute adversarial loss (inner loop + outer loss)
        if adv_type == "adv_pgd":
            loss, x_adv = pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm=norm)
            x_eval = x_adv
        elif adv_type == "trades":
            loss, x_adv = trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm=norm)
            x_eval = x_adv
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        model.eval()
        with torch.no_grad():
            preds = model(x_eval).argmax(dim=1)
            running_correct += (preds == y).sum().item()
        model.train()

        avg_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}")

    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)


# ------------------------------------------------------------------
#         Local-Entropy adversarial generator (eval-time only)
# ------------------------------------------------------------------

def local_entropy_generator(model, x, y, **kwargs):
    """Run a Local-Entropy Langevin attack to draw N adversarial particles.

    Returns x_adv of shape (B, N, C, H, W). Used for the optional LocEnt
    mean-PR eval; this script does not train with local entropy.

    Hyper-parameters not exposed via CLI (gamma, beta, step_size, energy
    function, threshold) are pinned to sensible defaults.
    """
    norm = kwargs.get("norm", "linf")
    epsilon = kwargs.get("epsilon", 8 / 255)
    num_particles = kwargs.get("num_particles", 8)
    langevin_steps = kwargs.get("langevin_steps", 10)

    state = ParticleState(epsilon=epsilon, norm=norm, num_particles=num_particles)
    energy_cfg = EnergyConfig(psi_type="softplus", psi_alpha=10.0)
    langevin_cfg = LangevinConfig(
        steps=langevin_steps, step_size=1e-2, beta=100.0, noise_scale=1.0,
    )

    state.init_particles(x, method="uniform", warm_start=False)

    B, N = x.shape[0], num_particles
    margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
    t_curr = fixed_threshold_update(margins=margins, state=state, t=-0.05)
    gamma_curr = fixed_scope(t_curr=t_curr, gamma=0.05)

    langevin_update_local_entropy(
        state=state, model=model, x=x, y=y,
        t_curr=t_curr, gamma_curr=gamma_curr,
        energy_cfg=energy_cfg, cfg=langevin_cfg,
    )
    return state.x_adv


# ------------------------------------------------------------------
#  Single-pass per-epoch evaluation: clean / PGD / LocEnt / RandomPR /
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
    """
    do_pgd    = pgd_cfg is not None
    do_locent = locent_cfg is not None
    do_random = bool(random_cfgs)
    do_corr   = bool(corruptions) and bool(severities)

    model.eval()
    n_total = 0
    n_clean_correct = 0
    clean_loss_sum  = 0.0
    n_pgd_correct = 0
    sum_locent_pr = 0.0
    random_sum = {cfg["noise_dist"]: 0.0 for cfg in (random_cfgs or [])}
    corr_correct  = {(c, s): 0 for c in (corruptions or []) for s in (severities or [])}

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

        # 2) PGD — autograd.grad on inputs; no no_grad wrapper.
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

        post = {"clean": f"{n_clean_correct/n_total:.3f}"}
        if do_pgd:    post["pgd"]  = f"{n_pgd_correct/n_total:.3f}"
        if do_locent: post["loc"]  = f"{sum_locent_pr/n_total:.3f}"
        if do_random: post["rnd"]  = f"{sum(random_sum.values())/(n_total*len(random_sum)):.3f}"
        pbar.set_postfix(**post)

    random_breakdown = {d: s / n_total for d, s in random_sum.items()} if do_random else None
    corr_breakdown   = {k: v / n_total for k, v in corr_correct.items()} if do_corr else None
    corr_mean = (sum(corr_breakdown.values()) / max(1, len(corr_breakdown))) if do_corr else None

    return {
        "clean_acc":           n_clean_correct / n_total,
        "clean_loss":          (clean_loss_sum / n_total) if criterion is not None else None,
        "pgd_acc":             (n_pgd_correct / n_total) if do_pgd    else None,
        "locent_pr":           (sum_locent_pr / n_total) if do_locent else None,
        "random_pr_breakdown": random_breakdown,   # {dist: acc} or None
        "corr_acc":            corr_mean,
        "corr_breakdown":      corr_breakdown,
        "num_samples":         n_total,
    }


# ------------------------------------------------------------------
#  AutoAttack evaluation (slow — call only at the final epoch)
# ------------------------------------------------------------------

def evaluate_aa(model, loader, device, norm, epsilon, version="rand", eval_name="eval-AA"):
    """Run AutoAttack over the loader. Returns adversarial accuracy.

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

    # Dataset & model
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", choices=[
        "resnet18", "resnet50", "wide_resnet50_2",
        "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
        "vit_b_16"
    ], default="resnet18")
    ap.add_argument("--pretrained", action="store_true",
                    help="Load ImageNet pretrained weights (recommended: use --lr 0.01)")

    # General Training Settings
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--img_size", type=int, default=None,
                    help="Input image size (will be resized if dataset images are different)")
    # Training Method
    ap.add_argument("--training_type", choices=["standard", "adv_pgd", "trades"], default="adv_pgd",
                    help="Training method: standard, adv_pgd (PGD-AT), trades (TRADES)")
    ap.add_argument("--augment", action="store_true",
                    help="Enable training-set data augmentation (RandomCrop / Flip / "
                         "RandAugment / RandomErasing). When set, output filenames "
                         "are tagged with '_Aug'.")

    # Adversarial Training Settings (for PGD-AT and TRADES)
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Norm for adversarial perturbations (for PGD-AT and TRADES)")
    ap.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation budget")
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD step size")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="Number of PGD steps")
    ap.add_argument("--beta", type=float, default=12.0,
                    help="TRADES KL regularization weight")

    # ============================================================
    # Per-epoch evaluation knobs
    #
    # Each evaluation is opt-in. Clean accuracy is always reported. We expose
    # only the high-level "what level" knobs (steps, num samples, severities,
    # version) and optionally the norm; finer attack hyperparameters are
    # derived from --epsilon and standard ratios.
    # ============================================================

    # PGD
    ap.add_argument("--eval_pgd", action="store_true",
                    help="Run PGD adversarial eval at every eval cycle.")
    ap.add_argument("--pgd_steps", type=int, default=10,
                    help="Number of PGD steps when --eval_pgd is set.")
    ap.add_argument("--pgd_norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for PGD eval.")

    # Local-Entropy mean-PR (sensible defaults; n + steps + norm exposed,
    # other LocEnt internals pinned inside local_entropy_generator).
    ap.add_argument("--eval_locent", action="store_true",
                    help="Run Local-Entropy mean-PR eval at every eval cycle.")
    ap.add_argument("--locent_n", type=int, default=8,
                    help="Number of LocEnt particles at eval.")
    ap.add_argument("--locent_steps", type=int, default=10,
                    help="Number of Langevin steps for the LocEnt eval attack.")
    ap.add_argument("--locent_norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for the LocEnt attack.")

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

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/dignoise/adv_training",
                    help="Directory to save best checkpoint")

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
    elif args.training_type == "adv_pgd":
        logger.info(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
                    f"alpha={args.alpha:.4f}, num_steps={args.num_steps}")
    elif args.training_type == "trades":
        logger.info(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
                    f"alpha={args.alpha:.4f}, num_steps={args.num_steps}, beta={args.beta}")
    else:
        raise ValueError(f"Unknown training_type: {args.training_type}")

    # accumulate one dict per evaluation epoch; written to CSV incrementally
    training_history = []

    # Build datasets/loaders (augmentation governed by --augment; test set is always no-aug)
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True, img_size, augment=args.augment)
    test_set, _ = get_dataset(args.dataset, args.data_root, False, img_size, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
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

    # Adversarial config
    adv_config = {
        "type":      args.training_type,
        "norm":      args.norm,
        "epsilon":   args.epsilon,
        "alpha":     args.alpha,
        "num_steps": args.num_steps,
        "beta":      args.beta,
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
        elif args.training_type in ["adv_pgd", "trades"]:
            train_loss, train_acc = train_one_epoch_adv(model, train_loader, optimizer, device, criterion, adv_config,
                                                        epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        scheduler.step()

        # Evaluation and checkpointing
        if ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - start

            # ----- Build eval configs from the per-evaluation flags. -----
            pgd_cfg = None
            if args.eval_pgd:
                pgd_cfg = {
                    "epsilon":   args.epsilon,
                    "alpha":     args.epsilon / 4.0,
                    "num_steps": args.pgd_steps,
                    "norm":      args.pgd_norm,
                }

            locent_cfg = None
            if args.eval_locent:
                locent_cfg = {
                    "epsilon":        args.epsilon,
                    "norm":           args.locent_norm,
                    "num_particles":  args.locent_n,
                    "langevin_steps": args.locent_steps,
                }

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

            def _pct(v): return f"{v*100:.2f}%" if v is not None else None

            def _line(prefix, m, aa):
                parts = []
                if m["clean_loss"] is not None:
                    parts.append(f"loss={m['clean_loss']:.4f}")
                parts.append(f"clean={_pct(m['clean_acc'])}")
                if m["pgd_acc"]    is not None: parts.append(f"pgd{args.pgd_steps}={_pct(m['pgd_acc'])}")
                if m["locent_pr"]  is not None: parts.append(f"locent={_pct(m['locent_pr'])}")
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
            logger.info(log_msg)

            # CSV: stable schema across epochs even when an eval is disabled.
            # Random-PR is expanded into one column per distribution.
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
    if args.training_type in ["adv_pgd", "trades"]:
        ckpt["adv_config"] = adv_config

    torch.save(ckpt, out_path)
    logger.info(f"  -> saved last checkpoint to {out_path}")


if __name__ == "__main__":
    main()
