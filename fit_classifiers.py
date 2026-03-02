# fit_classifiers.py - Train image classifiers from scratch
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15
#   numpy
#   tqdm
#
# This script relies on:
#   - `utils/data_preprocessing.py` for dataset loading and preprocessing utilities
#   - `utils/adv_attacker.py` for adversarial training losses (PGD-AT, TRADES)
#   - `model_zoo/` for building various model architectures

import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model_zoo import build_model
from utils.data_preprocessing import get_dataset, get_img_size
from utils.adv_attacker import pgd_at_loss, trades_loss
from utils.pr_generator import pr_generator

from config_fitting import build_sigma_list

def set_seed(seed: int = 42):
    """Make training as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    """Return accuracy on `loader`."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# ------------------------------------------------------------------
#                       Standard Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, criterion, 
                    epoch=None, total_epochs=None):
    """Standard training loop."""
    model.train()
    running_loss = 0.0
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
        avg_loss = running_loss / total_samples

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / len(loader.dataset)


# ------------------------------------------------------------------
#                    Adversarial Training For One Epoch
# ------------------------------------------------------------------

def train_one_epoch_adv(model, loader, optimizer, device, criterion, 
                        adv_config, 
                        epoch=None, total_epochs=None):
    """
    Adversarial training loop (outer loop).
    Inner loop (attack generation) is handled by ad_attacker functions.
    """
    model.train()
    running_loss = 0.0
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
            loss, _ = pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm=norm)
        elif adv_type == "trades":
            loss, _ = trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm=norm)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        avg_loss = running_loss / total_samples

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / len(loader.dataset)

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
    total_samples = 0
    pbar = tqdm(loader, desc=f"PR Train [{epoch}/{total_epochs}]" if epoch else "PR Training", leave=False)

    # Extract generator kwargs (exclude "type")
    generator_kwargs = {k: v for k, v in pr_config.items() if k != "type"}

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Adversarial beta-mixed loss only
        x_adv = pr_generator(model, x, y, **generator_kwargs)

        # x_adv: (B, N, C, H, W)
        B, N = x_adv.shape[0], x_adv.shape[1]

        x_adv_flat = x_adv.view(B * N, *x_adv.shape[2:])      # (B*N, C, H, W)
        y_rep = y.repeat_interleave(N)                         # (B*N,)

        logits = model(x_adv_flat)                             # (B*N, num_classes)
        loss = criterion(logits, y_rep)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        avg_loss = running_loss / total_samples

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / len(loader.dataset)

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

    # General Training Settings
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--img_size", type=int, default=None,
                    help="Input image size (will be resized if dataset images are different)")
    # Training Method  
    ap.add_argument("--training_type", choices=["standard", "adv_pgd", "trades", "pr"], default="standard",
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
    ap.add_argument("--beta_mix", type=float, default=0.5,
                    help="Beta mix parameter for PR")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="Kappa parameter for PR")
    
    ap.add_argument("--K", type=int, default=2,
                    help="Number of mixture components for PR")
    ap.add_argument("--sigma_dist_type", type=str, default="linear",
                    help="Type of sigma distribution for PR")
    ap.add_argument("--fisher_damping", type=float, default=1e-4,
                    help="Fisher diagonal damping for PR")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Temperature for PR")

    ap.add_argument("--num_samples", type=int, default=10,
                    help="Number of perturbation samples per input for PR")
    ap.add_argument("--noise_scale", type=float, default=1.0,
                    help="Posterior sampling noise scale for PR")   


    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/standard_training",
                    help="Directory to save best checkpoint")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = get_img_size(args.dataset, args.img_size)

    # Print config
    print(f"[config] dataset={args.dataset}, arch={args.arch}")
    print(f"[config] img_size={img_size}")
    if args.training_type == "standard":
        print(f"[config] training_type={args.training_type}, no adversarial perturbations")
    elif args.training_type == "adv_pgd":
        print(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
              f"alpha={args.alpha:.4f}, num_steps={args.num_steps}")
    elif args.training_type == "trades":
        print(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm} "
              f"alpha={args.alpha:.4f}, num_steps={args.num_steps}, beta={args.beta}")
    elif args.training_type == "pr":
        print(f"[config] training_type={args.training_type}, epsilon={args.epsilon:.4f}, norm={args.norm}")
        print(f"         beta_mix={args.beta_mix}, kappa={args.kappa}")
        print(f"         K={args.K}, sigma_dist_type={args.sigma_dist_type}, fisher_damping={args.fisher_damping}, tau={args.tau}")
        print(f"         num_samples={args.num_samples}, noise_scale={args.noise_scale}")   
    else:
        raise ValueError(f"Unknown training_type: {args.training_type}")

    # Build datasets/loaders
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True, img_size)
    test_set, _ = get_dataset(args.dataset, args.data_root, False, img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Build model/
    model = build_model(args.arch, num_classes, args.dataset)
    model.to(device)

    # Optional DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{args.arch.lower()}_{args.dataset.lower()}.pth")
    print(f"[save] best checkpoint will be written to: {out_path}")

    # Train
    start = time.time()

    for ep in range(1, args.epochs + 1):
        if args.training_type == "standard":
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                         epoch=ep, total_epochs=args.epochs)
        elif args.training_type in ["adv_pgd", "trades"]:
            train_loss = train_one_epoch_adv(model, train_loader, optimizer, device, criterion, adv_config,
                                             epoch=ep, total_epochs=args.epochs)
        elif args.training_type == "pr":
            train_loss = train_one_epoch_pr(model, train_loader, optimizer, device, criterion, pr_config,
                                             epoch=ep, total_epochs=args.epochs)
        else:
            raise ValueError(f"Unknown training_type: {args.training_type}")

        acc = evaluate(model, test_loader, device)
        print(f"[{ep:03d}/{args.epochs}] loss={train_loss:.4f}  val_acc={acc*100:.2f}%")

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
    print(f"  -> saved last checkpoint to {out_path}")

    elapsed = time.time() - start
    print(f"Done. last val acc: {acc*100:.2f}%  (time: {elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
