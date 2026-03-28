# train_classifiers.py - Train image classifiers with standard, adversarial, or
#                        probabilistic robustness (PR) training.
#
# Description:
#   This script trains an image classifier on CIFAR-10, CIFAR-100, or
#   TinyImageNet using one of four training methods:
#
#     standard   - Vanilla cross-entropy training on clean images.
#     adv_pgd    - PGD adversarial training (Madry et al.).
#                  Inner-loop PGD attack generates adversarial examples;
#                  the model is trained to classify them correctly.
#     trades     - TRADES adversarial training (Zhang et al.).
#                  Adds a KL-divergence regularisation term between clean
#                  and adversarial logits controlled by --beta.
#     pr         - Probabilistic Robustness training.
#                  Perturbations are sampled from a Langevin-based mixture
#                  of Gaussians (pr_generator); the model is trained over
#                  N perturbation samples per image.
#
#   For every training run the script saves:
#     <save_dir>/<arch>_<dataset>_<training_type>.pth          model checkpoint
#     <save_dir>/<arch>_<dataset>_<training_type>.log          training log
#     <save_dir>/<arch>_<dataset>_<training_type>_training_info.csv  per-epoch metrics
#
#   Evaluation is run every 5 epochs on a fixed subset of the training set
#   (same size as the test set) and the full test set, reporting clean
#   accuracy and PR-attack robust accuracy for both splits.
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
#   src/adv_attacker.py         PGD-AT and TRADES loss functions
#   src/langevin4pr.py          PR perturbation generator
#   configs/train_clf_cfg.py    sigma schedule builder for PR
#
# Usage:
#   python scripts/train_classifiers.py [options]
#
# Key arguments:
#   --dataset       {cifar10, cifar100, tinyimagenet}  (default: cifar10)
#   --arch          {resnet18, resnet50, wide_resnet50_2, vgg16,
#                    densenet121, mobilenet_v3_large,
#                    efficientnet_b0, vit_b_16}         (default: resnet18)
#   --training_type {standard, adv_pgd, trades, pr}    (default: adv_pgd)
#   --epochs        number of training epochs           (default: 50)
#   --batch_size    mini-batch size                     (default: 128)
#   --lr            initial learning rate               (default: 0.1)
#   --save_dir      output directory                    (default: ./ckp/adv_training)
#   --device        compute device, e.g. cuda or cpu    (default: cuda)
#
#   PGD-AT / TRADES specific:
#   --epsilon       perturbation budget                 (default: 8/255)
#   --alpha         PGD step size                       (default: 2/255)
#   --num_steps     number of PGD steps                 (default: 10)
#   --beta          TRADES regularisation weight        (default: 6.0)
#
#   PR specific:
#   --K             number of MoG components            (default: 3)
#   --beta_mix      CE / soft-0-1 interpolation weight  (default: 1)
#   --kappa         margin surrogate softness           (default: 0.02)
#   --num_samples   perturbation samples per image      (default: 32)
#   --fisher_damping diagonal Fisher damping            (default: 1e-7)
#   --tau           Langevin temperature                (default: 1e-4)
#   --noise_scale   posterior sampling noise scale      (default: 1.0)
#   --sigma_dist_type sigma schedule type               (default: geometric)
#
# Examples:
#   # Standard training on CIFAR-10 with ResNet-18
#   python scripts/train_classifiers.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type standard --epochs 100
#
#   # PGD adversarial training on CIFAR-10 with ResNet-18
#   python scripts/train_classifiers.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type adv_pgd \
#       --epsilon 0.03137 --alpha 0.00784 --num_steps 10 \
#       --epochs 100 --save_dir ./ckp/adv_training
#
#   # PR training on CIFAR-100 with WideResNet-50-2
#   python scripts/train_classifiers.py \
#       --dataset cifar100 --arch wide_resnet50_2 \
#       --training_type pr \
#       --K 3 --num_samples 16 --epochs 50 \
#       --save_dir ./ckp/pr_training

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
from src.adv_attacker import pgd_at_loss, trades_loss
from src.langevin4pr import pr_generator
from configs.train_clf_cfg import build_sigma_list

def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("fit_classifiers")
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
    Inner loop (attack generation) is handled by ad_attacker functions.
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
        elif adv_type == "trades":
            loss, x_adv = trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm=norm)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")

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
#                    Probabilistic Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch_pr(model, loader, optimizer, device, criterion,
                        pr_config,
                        epoch=None, total_epochs=None):
    """
    Probabilistic (PR / Bayesian) training loop.

    pr_config keys:
        norm          : "linf" | "l2" (default "linf")
        epsilon       : perturbation budget radius (default 8/255)

        kappa         : float > 0, margin surrogate softness (default 1.0)
        beta_mix      : float in [0,1], interpolation CE <-> soft-0-1 (default 0.5)
        
        K             : number of MoG components (default 2)
        fisher_damping: diagonal Fisher damping (default 1e-4)
       
        tau           : temperature > 0 (default 1.0)
        noise_scale   : posterior sampling noise scale (default 1.0)

        loss = beta_pr_loss(x_adv)
    """

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc=f"PR Train [{epoch}/{total_epochs}]" if epoch else "PR Training", leave=False)

    # Extract generator kwargs (exclude "type")
    generator_kwargs = {k: v for k, v in pr_config.items() if k != "type"}

    # stats accumulators
    stat_sums = {"D_mu": 0.0, "D_sig": 0.0, "D_proxy": 0.0, "pi_entropy": 0.0, "pi_max": 0.0}
    stat_count = 0  # number of batches (or you can weight by batch size)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Adversarial beta-mixed loss only
        x_adv, stats = pr_generator(model, x, y, **generator_kwargs, return_stats=True)

        # x_adv: (B, N, C, H, W)
        B, N = x_adv.shape[0], x_adv.shape[1]

        x_adv_flat = x_adv.view(B * N, *x_adv.shape[2:])      # (B*N, C, H, W)
        y_rep = y.repeat_interleave(N)                         # (B*N,)

        logits = model(x_adv_flat)                             # (B*N, num_classes)
        loss = criterion(logits, y_rep)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)   # criterion averages over B*N; normalise by original batch size B
        total_samples += y.size(0)

        # runing correct
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            running_correct += (preds == y_rep).sum().item()

        # accumulate stats (detach already done)
        for k in stat_sums:
            stat_sums[k] += float(stats[k].item())
        stat_count += 1

        avg_loss = running_loss / total_samples
        avg_stats = {k: stat_sums[k] / stat_count for k in stat_sums}
        train_acc = running_correct / (total_samples * N)
        
        pbar.set_postfix(loss=f"{avg_loss:.4f}", 
                         D=f"{avg_stats['D_proxy']:.3e}", 
                         Hpi=f"{avg_stats['pi_entropy']:.3f}", 
                         acc=f"{train_acc:.4f}")

    return running_loss / len(loader.dataset), train_acc, avg_stats

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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--img_size", type=int, default=None,
                    help="Input image size (will be resized if dataset images are different)")
    # Training Method  
    ap.add_argument("--training_type", choices=["standard", "adv_pgd", "trades", "pr"], default="adv_pgd",
                    help="Training method: standard, adv_pgd (PGD-AT), trades (TRADES), pr (PR)")
    
    # Adversarial Training Settings (for PGD-AT and TRADES)
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf",
                    help="Norm for adversarial perturbations (for PGD-AT, TRADES, and PR)")
    ap.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation budget")
    ap.add_argument("--alpha", type=float, default=2/255,
                    help="PGD step size")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="Number of PGD steps")
    ap.add_argument("--beta", type=float, default=6.0,
                    help="TRADES regularization weight")

    # PR Training Settings (for PR)
    ap.add_argument("--beta_mix", type=float, default=1,
                    help="Beta mix parameter for PR")
    ap.add_argument("--kappa", type=float, default=0.02,
                    help="Kappa parameter for PR")
    
    ap.add_argument("--K", type=int, default=3,
                    help="Number of mixture components for PR")
    ap.add_argument("--sigma_dist_type", type=str, default="geometric",
                    help="Type of sigma distribution for PR")
    ap.add_argument("--fisher_damping", type=float, default=1e-7,
                    help="Fisher diagonal damping for PR")
    ap.add_argument("--tau", type=float, default=1e-4,
                    help="Temperature for PR")

    ap.add_argument("--num_samples", type=int, default=3,
                    help="Number of perturbation samples per input for PR")
    ap.add_argument("--noise_scale", type=float, default=1.0,
                    help="Posterior sampling noise scale for PR")   

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/adv_training",
                    help="Directory to save best checkpoint")

    args = ap.parse_args()

    from utils.evaluator import Evaluator

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = get_img_size(args.dataset, args.img_size)

    # Set up output directory and logger early so config lines are captured
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        f"{args.arch.lower()}_{args.dataset.lower()}_{args.training_type}.log"
    )
    logger = setup_logger(log_path)

    # Log config
    logger.info(f"[config] dataset={args.dataset}, arch={args.arch}, pretrained={args.pretrained}")
    logger.info(f"[config] img_size={img_size}")
    if args.training_type == "standard":
        logger.info(f"[config] training_type={args.training_type}, no adversarial perturbations")
    elif args.training_type == "adv_pgd":
        logger.info(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
                    f"alpha={args.alpha:.4f}, num_steps={args.num_steps}")
    elif args.training_type == "trades":
        logger.info(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
                    f"alpha={args.alpha:.4f}, num_steps={args.num_steps}, beta={args.beta}")
    elif args.training_type == "pr":
        logger.info(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm}")
        logger.info(f"         beta_mix={args.beta_mix}, kappa={args.kappa}")
        logger.info(f"         K={args.K}, sigma_dist_type={args.sigma_dist_type}, fisher_damping={args.fisher_damping}, tau={args.tau}")
        logger.info(f"         num_samples={args.num_samples}, noise_scale={args.noise_scale}")
    else:
        raise ValueError(f"Unknown training_type: {args.training_type}")

    # accumulate one dict per evaluation epoch; written to CSV incrementally
    training_history = []

    # Build datasets/loaders
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True, img_size, augment=True)
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

    # Build model/
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
        "type": args.training_type,
        "norm": args.norm,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "num_steps": args.num_steps,
        "beta": args.beta,
    }

    # PR config
    sigma_list = build_sigma_list(epsilon=args.epsilon, K=args.K, mode_type=args.sigma_dist_type)
    pr_config = {
        "type": args.training_type,
        "norm": args.norm,
        "epsilon": args.epsilon,

        "beta_mix": args.beta_mix,
        "kappa": args.kappa,
       
        "K": args.K,
        "sigma_list": sigma_list,
        "fisher_damping": args.fisher_damping,
        "tau": args.tau,
       
        "noise_scale": args.noise_scale,
        "num_samples": args.num_samples,
    }

    # Output path
    out_path = os.path.join(args.save_dir, f"{args.arch.lower()}_{args.dataset.lower()}_{args.training_type}.pth")
    logger.info(f"[save] checkpoint -> {out_path}")
    logger.info(f"[save] log       -> {log_path}")
    logger.info(f"[save] csv       -> {os.path.join(args.save_dir, f'{args.arch.lower()}_{args.dataset.lower()}_training_info.csv')}")

    # Train
    for ep in range(1, args.epochs + 1):
        start = time.time()
        if args.training_type == "standard":
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                                    epoch=ep, total_epochs=args.epochs)
        elif args.training_type in ["adv_pgd", "trades"]:
            train_loss, train_acc = train_one_epoch_adv(model, train_loader, optimizer, device, criterion, adv_config,
                                                        epoch=ep, total_epochs=args.epochs)
        elif args.training_type == "pr":
            train_loss, train_acc, avg_stats = train_one_epoch_pr(model, train_loader, optimizer, device, criterion, pr_config,
                                                                  epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        scheduler.step()

        # Evaluation and checkpointing
        if ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - start

            model.eval()
            ## Evaluation on Test set (clean and PR attack) ##
            pr_kwargs = {k: v for k, v in pr_config.items() if k != "type"}
            evaluator = Evaluator(model, test_loader, criterion, device)
            clean = evaluator.evaluate_standard()
            val_acc, val_loss = clean["acc"], clean["loss"]
            pr_result = evaluator.evaluate_pr(pr_generator=pr_generator, **pr_kwargs)
            eval_rob_acc = pr_result["pr"]

            ## Evaluation on Train subset (same size as test set) ##
            evaluator.update_loader(subtrain_loader)
            clean_T = evaluator.evaluate_standard()
            val_acc_T, val_loss_T = clean_T["acc"], clean_T["loss"]
            pr_result_T = evaluator.evaluate_pr(pr_generator=pr_generator, **pr_kwargs)
            eval_rob_acc_T = pr_result_T["pr"]

            current_lr = scheduler.get_last_lr()[0]
            log_msg = (
                f"[{ep:03d}/{args.epochs}] "
                f"lr={current_lr:.5f} "
                f"time={elapsed:.1f}s "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc*100:.2f}% "
                f"| trainS_loss={val_loss_T:.4f} "
                f"trainS_acc={val_acc_T*100:.2f}% "
                f"trainS_rob_acc={eval_rob_acc_T*100:.2f}% "
                f"| val_loss={val_loss:.4f} "
                f"val_acc={val_acc*100:.2f}% "
                f"rob_acc={eval_rob_acc*100:.2f}%"
            )
            logger.info(log_msg)

            epoch_info = {
                'arch':            args.arch,
                'dataset':         args.dataset,
                'training_type':   args.training_type,
                'epoch':           ep,
                'lr':              current_lr,
                'time':            elapsed,
                'train_loss':      train_loss,
                'train_acc':       train_acc,
                'trainS_loss':    val_loss_T,
                'trainS_acc':     val_acc_T,
                'trainS_rob_acc': eval_rob_acc_T,
                'val_loss':       val_loss,
                'val_acc':        val_acc,
                'rob_acc':        eval_rob_acc,
            }
            if args.training_type == "pr":
                epoch_info.update(avg_stats)
            training_history.append(epoch_info)

            # overwrite CSV with the full history so far
            info_csv_path = os.path.join(args.save_dir, f"{args.arch.lower()}_{args.dataset.lower()}_{args.training_type}_training_info.csv")
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
    elif args.training_type == "pr":
        ckpt["pr_config"] = pr_config

    torch.save(ckpt, out_path)
    logger.info(f"  -> saved last checkpoint to {out_path}")


if __name__ == "__main__":
    main()
