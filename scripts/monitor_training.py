# monitor_training.py - Train image classifiers with standard or PGD-AT
#                       adversarial training.
#
# Description:
#   This script trains an image classifier on CIFAR-10, CIFAR-100, or
#   TinyImageNet using one of two training methods:
#
#     standard   - Vanilla cross-entropy training on clean images.
#     adv_pgd    - PGD adversarial training (Madry et al.).
#                  Inner-loop PGD attack generates adversarial examples;
#                  the model is trained to classify them correctly.
#
#   Training-set data augmentation (RandomCrop / horizontal flip /
#   RandAugment / RandomErasing) is controlled by the --augment flag. When
#   disabled (default), the adversarial perturbations are the only source of
#   input variability seen by the model during training.
#
#   For every training run the script saves:
#     <save_dir>/<arch>_<dataset>_<training_type>[_Aug].pth          model checkpoint
#     <save_dir>/<arch>_<dataset>_<training_type>[_Aug].log          training log
#     <save_dir>/<arch>_<dataset>_<training_type>[_Aug]_training_info.csv  per-epoch metrics
#
#   Evaluation is run every 5 epochs on a fixed subset of the training set
#   (same size as the test set) and the full test set, reporting clean
#   accuracy for both splits (plus optional PGD adversarial accuracy when
#   --eval_pgd is set).
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
#   src/adv_attacker.py         PGD-AT loss and PGD attack
#
# Usage:
#   python scripts/monitor_training.py [options]
#
# Key arguments:
#   --dataset       {cifar10, cifar100, tinyimagenet}  (default: cifar10)
#   --arch          {resnet18, resnet50, wide_resnet50_2, vgg16,
#                    densenet121, mobilenet_v3_large,
#                    efficientnet_b0, vit_b_16}         (default: resnet18)
#   --training_type {standard, adv_pgd}                (default: adv_pgd)
#   --augment       enable training-set data augmentation (default: off)
#   --epochs        number of training epochs           (default: 100)
#   --batch_size    mini-batch size                     (default: 512)
#   --lr            initial learning rate               (default: 0.01)
#   --save_dir      output directory                    (default: ./ckp/dignoise/adv_training)
#   --device        compute device, e.g. cuda or cpu    (default: cuda)
#
#   PGD-AT specific:
#   --epsilon       perturbation budget                 (default: 8/255)
#   --alpha         PGD step size                       (default: 2/255)
#   --num_steps     number of PGD steps                 (default: 10)
#
# Examples:
#   # PGD adversarial training on CIFAR-10 with ResNet-18 (no augmentation)
#   python scripts/monitor_training.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type adv_pgd \
#       --epsilon 0.03137 --alpha 0.00784 --num_steps 10 \
#       --epochs 100 --save_dir ./ckp/dignoise/adv_training
#
#   # Standard training on CIFAR-10 with augmentation enabled
#   python scripts/monitor_training.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type standard --augment \
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
from src.adv_attacker import pgd_at_loss, pgd_attack


def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("monitor_training")
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
#                    PGD Adversarial Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch_adv(model, loader, optimizer, device, criterion,
                        adv_config,
                        epoch=None, total_epochs=None):
    """
    PGD adversarial training loop (outer loop).
    Inner loop (attack generation) is handled by pgd_at_loss.
    train_acc is measured on adversarial examples (robust training accuracy).
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"Adv Train [{epoch}/{total_epochs}]" if epoch else "Adv Training", leave=False)

    norm      = adv_config["norm"]
    epsilon   = adv_config["epsilon"]
    alpha     = adv_config["alpha"]
    num_steps = adv_config["num_steps"]

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Compute adversarial loss (inner PGD loop + outer loss)
        loss, x_adv = pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm=norm)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        model.eval()
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
            running_correct += (preds == y).sum().item()
        model.train()

        avg_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}")

    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)


# ------------------------------------------------------------------
#           Single-pass per-epoch evaluation: clean / PGD
# ------------------------------------------------------------------

def evaluate_per_epoch(model, loader, device, criterion, pgd_cfg=None, eval_name="eval"):
    """Single-pass eval over loader. Clean accuracy is always reported; PGD
    adversarial accuracy is reported when `pgd_cfg` is non-None.
    """
    do_pgd = pgd_cfg is not None

    model.eval()
    n_total = 0
    n_clean_correct = 0
    clean_loss_sum  = 0.0
    n_pgd_correct = 0

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

        post = {"clean": f"{n_clean_correct/n_total:.3f}"}
        if do_pgd:
            post["pgd"] = f"{n_pgd_correct/n_total:.3f}"
        pbar.set_postfix(**post)

    return {
        "clean_acc":   n_clean_correct / n_total,
        "clean_loss":  (clean_loss_sum / n_total) if criterion is not None else None,
        "pgd_acc":     (n_pgd_correct / n_total) if do_pgd else None,
        "num_samples": n_total,
    }


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
    ap.add_argument("--training_type", choices=["standard", "adv_pgd"], default="adv_pgd",
                    help="Training method: standard or adv_pgd (PGD-AT)")
    ap.add_argument("--augment", action="store_true",
                    help="Enable training-set data augmentation (RandomCrop / Flip / "
                         "RandAugment / RandomErasing). When set, output filenames "
                         "are tagged with '_Aug'.")

    # PGD Adversarial Training Settings (for adv_pgd)
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Norm for adversarial perturbations")
    ap.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation budget")
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD step size")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="Number of PGD steps")

    # Per-epoch PGD evaluation (opt-in; clean accuracy is always reported)
    ap.add_argument("--eval_pgd", action="store_true",
                    help="Run PGD adversarial eval at every eval cycle.")
    ap.add_argument("--pgd_steps", type=int, default=10,
                    help="Number of PGD steps when --eval_pgd is set.")
    ap.add_argument("--pgd_norm", choices=["linf", "l2"], default="linf",
                    help="Norm constraint for PGD eval.")

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

    # PGD adversarial config
    adv_config = {
        "norm":      args.norm,
        "epsilon":   args.epsilon,
        "alpha":     args.alpha,
        "num_steps": args.num_steps,
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
        elif args.training_type == "adv_pgd":
            train_loss, train_acc = train_one_epoch_adv(model, train_loader, optimizer, device, criterion, adv_config,
                                                        epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        scheduler.step()

        # Evaluation and checkpointing
        if ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - start

            # ----- Build eval config from the per-evaluation flags. -----
            pgd_cfg = None
            if args.eval_pgd:
                pgd_cfg = {
                    "epsilon":   args.epsilon,
                    "alpha":     args.epsilon / 4.0,
                    "num_steps": args.pgd_steps,
                    "norm":      args.pgd_norm,
                }

            ## Evaluation on Test set ##
            test_metrics = evaluate_per_epoch(
                model, test_loader, device, criterion, pgd_cfg=pgd_cfg,
                eval_name=f"eval-test [{ep}/{args.epochs}]",
            )

            ## Evaluation on Train subset (same size as test set) ##
            train_metrics = evaluate_per_epoch(
                model, subtrain_loader, device, criterion, pgd_cfg=pgd_cfg,
                eval_name=f"eval-trainS [{ep}/{args.epochs}]",
            )

            current_lr = scheduler.get_last_lr()[0]

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

            # CSV: stable schema across epochs even when PGD eval is disabled.
            epoch_info = {
                'arch':          args.arch,
                'dataset':       args.dataset,
                'training_type': args.training_type,
                'epoch':         ep,
                'lr':            current_lr,
                'time':          elapsed,
                'train_loss':    train_loss,
                'train_acc':     train_acc,
                # train subset (no augmentation) metrics
                'trainS_loss':   train_metrics['clean_loss'],
                'trainS_acc':    train_metrics['clean_acc'],
                'trainS_pgd':    train_metrics['pgd_acc'],
                # test set metrics
                'val_loss':      test_metrics['clean_loss'],
                'val_acc':       test_metrics['clean_acc'],
                'val_pgd':       test_metrics['pgd_acc'],
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
    if args.training_type == "adv_pgd":
        ckpt["adv_config"] = adv_config

    torch.save(ckpt, out_path)
    logger.info(f"  -> saved last checkpoint to {out_path}")


if __name__ == "__main__":
    main()
