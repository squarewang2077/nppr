#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline evaluation script for robustness testing.
Evaluates:
1. Baseline noise with Gaussian and Uniform distributions (Monte Carlo sampling)
2. PGD (Projected Gradient Descent) attacks
3. CW (Carlini-Wagner) attacks

Compatible with run_evaluation.sh script.

Usage:
    python test.py \
        --dataset cifar10 \
        --arch resnet18 \
        --clf_ckpt ./model_zoo/trained_model/resnet18_cifar10.pth \
        --epsilon 0.062 \
        --norm_type linf \
        --num_samples 500 \
        --attack_steps 20 \
        --step_size 0.00784 \
        --max_batches 100 \
        --log_dir ./logs
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import statistics
import numpy as np
import torch.distributions as dist
from datetime import datetime
import time

# Import model building functions
from fit_classifiers import build_model
from utils.utils import get_dataset


def get_dataset_loader(dataset_name, data_root='./dataset', train=False, batch_size=10, resize=False):
    """Get dataloader for the specified dataset with proper normalization."""
    dataset, _, _ = get_dataset(dataset_name, root=data_root, train=train, resize=resize)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    return loader


def get_num_classes(dataset_name):
    """Get number of classes for the dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return 10
    elif dataset_name == "cifar100":
        return 100
    elif dataset_name == "tinyimagenet":
        return 200
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_img_size(dataset_name):
    """Get image size for the dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name in ["cifar10", "cifar100"]:
        return 32
    elif dataset_name == "tinyimagenet":
        return 64
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def accuracy(logits, target):
    """Compute accuracy from logits and targets."""
    _, pred = torch.max(logits, dim=1)
    correct = (pred == target).sum()
    total = target.size(0)
    acc = (float(correct) / total) * 100
    return acc


# ==================== Baseline Noise Evaluation ====================

def g_ball(delta, gamma, norm_type='linf'):
    """
    Apply g_ball projection to constrain perturbations to a ball.

    Args:
        delta: Perturbation tensor [B, C, H, W]
        gamma: Radius of the ball (epsilon)
        norm_type: Type of norm ('linf' or 'l2')

    Returns:
        Projected perturbation tensor
    """
    if norm_type == 'linf':
        # Clip to [-gamma, gamma] per element
        return torch.clamp(delta, -gamma, gamma)
    elif norm_type == 'l2':
        # Project to L2 ball of radius gamma
        batch_size = delta.size(0)
        delta_flat = delta.view(batch_size, -1)
        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        scale = torch.clamp(norms / gamma, min=1.0)
        delta_flat = delta_flat / scale
        return delta_flat.view_as(delta)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


@torch.no_grad()
def compute_pr_with_baseline_noise(
    model,
    loader,
    out_shape,
    device,
    distribution='gaussian',
    num_samples=100,
    epsilon=16.0/255,
    norm_type='linf',
    batch_indices=None,
    chunk_size=None,
    clip_to_valid_range=False
):
    """
    Compute PR on clean-correct set using baseline noise distributions (Gaussian or Uniform).

    This function samples perturbations directly in the input space from standard distributions
    (no GMM, no decoder needed) and evaluates probabilistic robustness.

    PR = E_{(x,y) ∈ CleanCorrect} E_{delta~Noise} [ 1{ f(x+delta) = y } ]

    Args:
        model: Classifier model (should be in eval mode)
        loader: DataLoader for evaluation
        out_shape: Image shape (C, H, W)
        device: torch device
        distribution: Type of noise distribution to use:
                     - 'gaussian': Standard Gaussian N(0, I) in input space
                     - 'uniform': Uniform distribution U(-1, 1) in input space
        num_samples: Number of samples per image
        epsilon: Perturbation budget (gamma parameter for g_ball)
        norm_type: Type of norm constraint ('linf' or 'l2')
        batch_indices: Optional set/list of batch indices to evaluate
        chunk_size: Maximum samples to process at once. If None, process all at once.
                   Useful for large num_samples to avoid OOM errors.
        clip_to_valid_range: If True, clip perturbed images to [0, 1]. Default False to match
                            GMM behavior (which doesn't clip).

    Returns:
        (pr, n_used, clean_acc, wall_time): PR score, number of clean-correct samples, clean accuracy, wall time in seconds
    """
    start_time = time.time()

    C, H, W = out_shape

    total_used = 0       # number of clean-correct samples
    pr_sum = 0.0         # sum of per-image PR values
    clean_correct = 0    # number of correctly classified clean samples
    total_seen = 0       # total samples seen

    pbar = tqdm(enumerate(loader),
                total=len(batch_indices) if batch_indices is not None else len(loader),
                desc=f"Baseline ({distribution})")

    for it, (x, y, _) in pbar:
        if (batch_indices is not None) and (it not in batch_indices):
            continue

        x, y = x.to(device), y.to(device)
        B = x.size(0)

        # Clean predictions & mask of correct ones
        logits_clean = model(x)
        pred_clean = logits_clean.argmax(1)
        mask = (pred_clean == y)

        clean_correct += mask.sum().item()
        total_seen += B

        if mask.sum().item() == 0:
            continue  # nothing to evaluate in this batch

        x_sel = x[mask]
        y_sel = y[mask]
        n = x_sel.size(0)

        # Determine chunk size for processing
        if chunk_size is None:
            # Process all samples at once
            actual_chunk_size = num_samples
        else:
            actual_chunk_size = min(chunk_size, num_samples)

        # Accumulate success rate for each image
        per_image_success = torch.zeros(n, device=device)

        # Process in chunks to avoid OOM
        num_chunks = (num_samples + actual_chunk_size - 1) // actual_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * actual_chunk_size
            end_idx = min((chunk_idx + 1) * actual_chunk_size, num_samples)
            S_chunk = end_idx - start_idx

            # Sample noise from the specified distribution
            # Shape: [S_chunk, n, C, H, W]
            if distribution == 'gaussian':
                # Standard Gaussian: N(0, I)
                noise = torch.randn(S_chunk, n, C, H, W, device=device)
            elif distribution == 'uniform':
                # Uniform: U(-1, 1)
                noise = torch.rand(S_chunk, n, C, H, W, device=device) * 2 - 1
            else:
                raise ValueError(f"Unknown distribution: {distribution}. Choose 'gaussian' or 'uniform'.")

            # Reshape for g_ball: [S_chunk * n, C, H, W]
            noise_flat = noise.view(S_chunk * n, C, H, W)

            # Apply g_ball to constrain to perturbation budget
            perturbations = g_ball(noise_flat, gamma=epsilon, norm_type=norm_type)

            # Create perturbed images (matching GMM pipeline exactly)
            x_rep = x_sel.unsqueeze(0).expand(S_chunk, -1, -1, -1, -1).reshape(S_chunk * n, C, H, W)
            x_perturbed = x_rep + perturbations

            # Optionally clip to valid image range [0, 1]
            # Note: GMM method does NOT clip by default, so we match that behavior
            if clip_to_valid_range:
                x_perturbed = torch.clamp(x_perturbed, 0.0, 1.0)

            # Evaluate classifier on perturbed images
            y_rep = y_sel.repeat(S_chunk)
            logits = model(x_perturbed)
            pred = logits.argmax(1)

            # Check which predictions match the true label
            # Shape: [S_chunk * n] -> [S_chunk, n]
            correct = (pred == y_rep).float().view(S_chunk, n)

            # Accumulate success rate
            per_image_success += correct.sum(dim=0)

        # Compute per-image PR (average success rate over all samples)
        per_image_pr = per_image_success / num_samples

        # Accumulate
        pr_sum += per_image_pr.sum().item()
        total_used += n

        # Update progress bar
        if total_used > 0:
            avg_pr = pr_sum / total_used
            pbar.set_postfix({
                'clean_acc': f'{clean_correct/total_seen*100:.2f}%',
                'avg_pr': f'{avg_pr*100:.2f}%',
                'samples': total_used
            })

    pr = pr_sum / max(1, total_used)
    clean_acc = clean_correct / max(1, total_seen)
    wall_time = time.time() - start_time

    chunk_info = f", chunk_size={chunk_size}" if chunk_size is not None else ""
    clip_info = " (clipped)" if clip_to_valid_range else " (no clipping)"
    print(f"\n[PR@clean - Baseline] used={total_used} / seen={total_seen} "
          f"(clean acc={clean_acc*100:.2f}%), num_samples={num_samples}{chunk_info} → "
          f"PR={pr:.4f} [distribution: {distribution}, norm: {norm_type}, eps={epsilon:.6f}{clip_info}]")
    print(f"Wall time: {wall_time:.2f}s")

    return pr, total_used, clean_acc, wall_time


# ==================== PGD Attack ====================

def pgd_attack(model, x, y, epsilon, step_size, num_steps, device):
    """
    Perform PGD attack.

    Args:
        model: Classification model
        x: Clean images (batch)
        y: True labels (batch)
        epsilon: Perturbation budget (in 0-1 scale)
        step_size: Step size for gradient ascent (in 0-1 scale)
        num_steps: Number of attack steps
        device: torch device

    Returns:
        Adversarial examples
    """
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        grad = torch.autograd.grad(loss, [x_adv])[0]

        x_adv = x_adv + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()

    return x_adv


def evaluate_pgd(model, loader, epsilon, step_size, num_steps, device, max_batches=None):
    """
    Evaluate model under PGD attack on correctly classified samples only.

    Returns:
        Dictionary with clean accuracy, robust accuracy on clean-correct set, attack success rate, and wall time
    """
    start_time = time.time()

    model.eval()
    clean_correct = 0
    adv_correct_on_clean_correct = 0
    total = 0

    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches) if max_batches else len(loader),
                desc="PGD Attack")

    for batch_idx, (x, y, _) in pbar:
        if max_batches and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        # Clean accuracy - identify correctly classified samples
        with torch.no_grad():
            model.eval()
            logits_clean = model(x)
            pred_clean = logits_clean.argmax(1)
            mask = (pred_clean == y)
            clean_correct += mask.sum().item()

        # Only attack correctly classified samples
        if mask.sum().item() > 0:
            x_correct = x[mask]
            y_correct = y[mask]

            # Generate adversarial examples for correctly classified samples only
            x_adv = pgd_attack(model, x_correct, y_correct, epsilon, step_size, num_steps, device)

            # Check adversarial accuracy on these samples
            with torch.no_grad():
                logits_adv = model(x_adv)
                pred_adv = logits_adv.argmax(1)
                adv_correct_on_clean_correct += (pred_adv == y_correct).sum().item()

        total += y.size(0)

        # Compute metrics
        clean_acc = clean_correct / total * 100 if total > 0 else 0
        robust_acc = adv_correct_on_clean_correct / clean_correct * 100 if clean_correct > 0 else 0
        attack_success_rate = (clean_correct - adv_correct_on_clean_correct) / clean_correct * 100 if clean_correct > 0 else 0

        pbar.set_postfix({
            'clean_acc': f'{clean_acc:.2f}%',
            'robust_acc': f'{robust_acc:.2f}%',
            'attack_success': f'{attack_success_rate:.2f}%'
        })

    wall_time = time.time() - start_time
    print(f"Wall time: {wall_time:.2f}s")

    return {
        'clean_accuracy': clean_correct / total * 100 if total > 0 else 0,
        'robust_accuracy': adv_correct_on_clean_correct / clean_correct * 100 if clean_correct > 0 else 0,
        'attack_success_rate': (clean_correct - adv_correct_on_clean_correct) / clean_correct * 100 if clean_correct > 0 else 0,
        'num_samples': total,
        'num_clean_correct': clean_correct,
        'wall_time': wall_time
    }


# ==================== CW Attack ====================

def cw_loss(logits, targets, num_classes, margin=2):
    """
    Carlini-Wagner loss.
    """
    onehot_targets = F.one_hot(targets, num_classes).float().to(logits.device)
    self_loss = torch.sum(onehot_targets * logits, dim=1)
    other_loss = torch.max(
        (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1
    )[0]

    loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, min=0))
    loss = loss / logits.size(0)
    return loss


def cw_attack(model, x, y, epsilon, step_size, num_steps, num_classes, device):
    """
    Perform CW attack.

    Args:
        model: Classification model
        x: Clean images (batch)
        y: True labels (batch)
        epsilon: Perturbation budget (in 0-1 scale)
        step_size: Step size for gradient ascent (in 0-1 scale)
        num_steps: Number of attack steps
        num_classes: Number of classes
        device: torch device

    Returns:
        Adversarial examples
    """
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = cw_loss(logits, y, num_classes)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv + step_size * torch.sign(grad.detach())

        # Projection to epsilon-ball
        x_adv = x + torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    return x_adv


def evaluate_cw(model, loader, epsilon, step_size, num_steps, num_classes, device, max_batches=None):
    """
    Evaluate model under CW attack on correctly classified samples only.

    Returns:
        Dictionary with clean accuracy, robust accuracy on clean-correct set, attack success rate, and wall time
    """
    start_time = time.time()

    model.eval()
    clean_correct = 0
    adv_correct_on_clean_correct = 0
    total = 0

    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches) if max_batches else len(loader),
                desc="CW Attack")

    for batch_idx, (x, y, _) in pbar:
        if max_batches and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        # Clean accuracy - identify correctly classified samples
        with torch.no_grad():
            model.eval()
            logits_clean = model(x)
            pred_clean = logits_clean.argmax(1)
            mask = (pred_clean == y)
            clean_correct += mask.sum().item()

        # Only attack correctly classified samples
        if mask.sum().item() > 0:
            x_correct = x[mask]
            y_correct = y[mask]

            # Generate adversarial examples for correctly classified samples only
            x_adv = cw_attack(model, x_correct, y_correct, epsilon, step_size, num_steps, num_classes, device)

            # Check adversarial accuracy on these samples
            with torch.no_grad():
                logits_adv = model(x_adv)
                pred_adv = logits_adv.argmax(1)
                adv_correct_on_clean_correct += (pred_adv == y_correct).sum().item()

        total += y.size(0)

        # Compute metrics
        clean_acc = clean_correct / total * 100 if total > 0 else 0
        robust_acc = adv_correct_on_clean_correct / clean_correct * 100 if clean_correct > 0 else 0
        attack_success_rate = (clean_correct - adv_correct_on_clean_correct) / clean_correct * 100 if clean_correct > 0 else 0

        pbar.set_postfix({
            'clean_acc': f'{clean_acc:.2f}%',
            'robust_acc': f'{robust_acc:.2f}%',
            'attack_success': f'{attack_success_rate:.2f}%'
        })

    wall_time = time.time() - start_time
    print(f"Wall time: {wall_time:.2f}s")

    return {
        'clean_accuracy': clean_correct / total * 100 if total > 0 else 0,
        'robust_accuracy': adv_correct_on_clean_correct / clean_correct * 100 if clean_correct > 0 else 0,
        'attack_success_rate': (clean_correct - adv_correct_on_clean_correct) / clean_correct * 100 if clean_correct > 0 else 0,
        'num_samples': total,
        'num_clean_correct': clean_correct,
        'wall_time': wall_time
    }


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description='Baseline Robustness Evaluation')

    # Dataset and model
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'tinyimagenet'],
                       help='Dataset name')
    parser.add_argument('--arch', type=str, required=True,
                       choices=['resnet18', 'resnet50', 'wide_resnet50_2', 'vgg16',
                               'densenet121', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16'],
                       help='Model architecture')
    parser.add_argument('--clf_ckpt', type=str, required=True,
                       help='Path to classifier checkpoint')
    parser.add_argument('--data_root', type=str, default='./dataset',
                       help='Root directory for datasets')

    # Attack parameters
    parser.add_argument('--epsilon', type=float, required=True,
                       help='Perturbation budget (in 0-1 scale, e.g., 0.062 for 16/255)')
    parser.add_argument('--norm_type', type=str, default='linf',
                       choices=['linf', 'l2'],
                       help='Norm type for perturbation')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of Monte Carlo samples for baseline noise evaluation')
    parser.add_argument('--attack_steps', type=int, default=20,
                       help='Number of attack steps for PGD/CW')
    parser.add_argument('--step_size', type=float, default=0.00784,
                       help='Step size for attacks (in 0-1 scale, e.g., 0.00784 for 2/255)')

    # Evaluation settings
    parser.add_argument('--max_batches', type=int, default=10,
                       help='Maximum number of batches to evaluate (None = all)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"evaluation_{timestamp}.txt")

    # Get dataset info
    num_classes = get_num_classes(args.dataset)
    img_size = get_img_size(args.dataset)

    print(f"\n{'='*80}")
    print(f"Baseline Robustness Evaluation")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {args.arch}")
    print(f"Checkpoint: {args.clf_ckpt}")
    print(f"Epsilon: {args.epsilon} ({args.epsilon*255:.1f}/255)")
    print(f"Step size: {args.step_size} ({args.step_size*255:.1f}/255)")
    print(f"Attack steps: {args.attack_steps}")
    print(f"Num MC samples: {args.num_samples}")
    print(f"Max batches: {args.max_batches}")
    print(f"{'='*80}\n")

    # Write config to log
    with open(log_file, 'w') as f:
        f.write(f"Baseline Robustness Evaluation\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Dataset: {args.dataset}\n")
        f.write(f"  Architecture: {args.arch}\n")
        f.write(f"  Checkpoint: {args.clf_ckpt}\n")
        f.write(f"  Epsilon: {args.epsilon} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"  Step size: {args.step_size} ({args.step_size*255:.1f}/255)\n")
        f.write(f"  Attack steps: {args.attack_steps}\n")
        f.write(f"  MC samples: {args.num_samples}\n")
        f.write(f"  Max batches: {args.max_batches}\n")
        f.write(f"  Timestamp: {timestamp}\n")
        f.write(f"\n{'='*80}\n\n")

    # Load model
    print("Loading model...")
    model = build_model(args.arch, num_classes, device, pretrained=False)

    if not os.path.exists(args.clf_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.clf_ckpt}")

    checkpoint = torch.load(args.clf_ckpt, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Detect image size from checkpoint to determine if we need resizing
    checkpoint_img_size = checkpoint.get('img_size', img_size)
    resize = (checkpoint_img_size == 224)  # Resize if model was trained on 224x224
    print(f"Checkpoint image size: {checkpoint_img_size}, Native size: {img_size}, Resize: {resize}\n")

    # Get data loader
    print("Loading test dataset...")
    test_loader = get_dataset_loader(args.dataset, args.data_root, train=False,
                                    batch_size=args.batch_size, resize=resize)
    print(f"Test loader created with {len(test_loader)} batches\n")

    # Define out_shape for the dataset (C, H, W)
    out_shape = (3, img_size, img_size)

    # Convert max_batches to batch_indices if needed
    batch_indices = range(args.max_batches) if args.max_batches is not None else None

    # ==================== Evaluate Baseline Noise (Uniform) ====================
    print(f"\n{'='*80}")
    print("Evaluating Baseline Noise - Uniform Distribution")
    print(f"{'='*80}\n")

    pr_uniform, n_used_uniform, clean_acc_uniform, time_uniform = compute_pr_with_baseline_noise(
        model=model,
        loader=test_loader,
        out_shape=out_shape,
        device=device,
        distribution='uniform',
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        norm_type=args.norm_type,
        batch_indices=batch_indices,
        chunk_size=32
    )

    print(f"\nResults (Uniform):")
    print(f"  Clean Accuracy: {clean_acc_uniform*100:.2f}%")
    print(f"  Avg PR: {pr_uniform*100:.2f}%")
    print(f"  Num evaluated: {n_used_uniform}")
    print(f"  Wall Time: {time_uniform:.2f}s")

    with open(log_file, 'a') as f:
        f.write(f"Baseline Noise - Uniform Distribution\n")
        f.write(f"{'-'*80}\n")
        f.write(f"  Clean Accuracy: {clean_acc_uniform*100:.2f}%\n")
        f.write(f"  Average PR: {pr_uniform*100:.2f}%\n")
        f.write(f"  Num evaluated: {n_used_uniform}\n")
        f.write(f"  Wall Time: {time_uniform:.2f}s\n")
        f.write(f"\n")

    # ==================== Evaluate Baseline Noise (Gaussian) ====================
    print(f"\n{'='*80}")
    print("Evaluating Baseline Noise - Gaussian Distribution")
    print(f"{'='*80}\n")

    pr_gaussian, n_used_gaussian, clean_acc_gaussian, time_gaussian = compute_pr_with_baseline_noise(
        model=model,
        loader=test_loader,
        out_shape=out_shape,
        device=device,
        distribution='gaussian',
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        norm_type=args.norm_type,
        batch_indices=batch_indices,
        chunk_size=32
    )

    print(f"\nResults (Gaussian):")
    print(f"  Clean Accuracy: {clean_acc_gaussian*100:.2f}%")
    print(f"  Avg PR: {pr_gaussian*100:.2f}%")
    print(f"  Num evaluated: {n_used_gaussian}")
    print(f"  Wall Time: {time_gaussian:.2f}s")

    with open(log_file, 'a') as f:
        f.write(f"Baseline Noise - Gaussian Distribution\n")
        f.write(f"{'-'*80}\n")
        f.write(f"  Clean Accuracy: {clean_acc_gaussian*100:.2f}%\n")
        f.write(f"  Average PR: {pr_gaussian*100:.2f}%\n")
        f.write(f"  Num evaluated: {n_used_gaussian}\n")
        f.write(f"  Wall Time: {time_gaussian:.2f}s\n")
        f.write(f"\n")

    # ==================== Evaluate PGD Attack ====================
    print(f"\n{'='*80}")
    print("Evaluating PGD Attack")
    print(f"{'='*80}\n")

    results_pgd = evaluate_pgd(
        model, test_loader, args.epsilon, args.step_size,
        args.attack_steps, device, max_batches=args.max_batches
    )

    print(f"\nResults (PGD):")
    print(f"  Clean Accuracy: {results_pgd['clean_accuracy']:.2f}%")
    print(f"  Robust Accuracy (on clean-correct): {results_pgd['robust_accuracy']:.2f}%")
    print(f"  Attack Success Rate: {results_pgd['attack_success_rate']:.2f}%")
    print(f"  Num samples: {results_pgd['num_samples']}")
    print(f"  Num clean-correct: {results_pgd['num_clean_correct']}")
    print(f"  Wall Time: {results_pgd['wall_time']:.2f}s")

    with open(log_file, 'a') as f:
        f.write(f"PGD Attack\n")
        f.write(f"{'-'*80}\n")
        f.write(f"  Clean Accuracy: {results_pgd['clean_accuracy']:.2f}%\n")
        f.write(f"  Robust Accuracy (on clean-correct): {results_pgd['robust_accuracy']:.2f}%\n")
        f.write(f"  Attack Success Rate: {results_pgd['attack_success_rate']:.2f}%\n")
        f.write(f"  Num samples: {results_pgd['num_samples']}\n")
        f.write(f"  Num clean-correct: {results_pgd['num_clean_correct']}\n")
        f.write(f"  Wall Time: {results_pgd['wall_time']:.2f}s\n")
        f.write(f"\n")

    # ==================== Evaluate CW Attack ====================
    print(f"\n{'='*80}")
    print("Evaluating CW Attack")
    print(f"{'='*80}\n")

    results_cw = evaluate_cw(
        model, test_loader, args.epsilon, args.step_size,
        args.attack_steps, num_classes, device, max_batches=args.max_batches
    )

    print(f"\nResults (CW):")
    print(f"  Clean Accuracy: {results_cw['clean_accuracy']:.2f}%")
    print(f"  Robust Accuracy (on clean-correct): {results_cw['robust_accuracy']:.2f}%")
    print(f"  Attack Success Rate: {results_cw['attack_success_rate']:.2f}%")
    print(f"  Num samples: {results_cw['num_samples']}")
    print(f"  Num clean-correct: {results_cw['num_clean_correct']}")
    print(f"  Wall Time: {results_cw['wall_time']:.2f}s")

    with open(log_file, 'a') as f:
        f.write(f"CW Attack\n")
        f.write(f"{'-'*80}\n")
        f.write(f"  Clean Accuracy: {results_cw['clean_accuracy']:.2f}%\n")
        f.write(f"  Robust Accuracy (on clean-correct): {results_cw['robust_accuracy']:.2f}%\n")
        f.write(f"  Attack Success Rate: {results_cw['attack_success_rate']:.2f}%\n")
        f.write(f"  Num samples: {results_cw['num_samples']}\n")
        f.write(f"  Num clean-correct: {results_cw['num_clean_correct']}\n")
        f.write(f"  Wall Time: {results_cw['wall_time']:.2f}s\n")
        f.write(f"\n")

    # ==================== Summary ====================
    total_wall_time = time_uniform + time_gaussian + results_pgd['wall_time'] + results_cw['wall_time']

    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Baseline Uniform PR:     {pr_uniform*100:.2f}%  (Wall Time: {time_uniform:.2f}s)")
    print(f"  Baseline Gaussian PR:    {pr_gaussian*100:.2f}%  (Wall Time: {time_gaussian:.2f}s)")
    print(f"  PGD Robust Accuracy:     {results_pgd['robust_accuracy']:.2f}%  (Wall Time: {results_pgd['wall_time']:.2f}s)")
    print(f"  PGD Attack Success Rate: {results_pgd['attack_success_rate']:.2f}%")
    print(f"  CW Robust Accuracy:      {results_cw['robust_accuracy']:.2f}%  (Wall Time: {results_cw['wall_time']:.2f}s)")
    print(f"  CW Attack Success Rate:  {results_cw['attack_success_rate']:.2f}%")
    print(f"\n  Total Wall Time:         {total_wall_time:.2f}s")
    print(f"\nResults saved to: {log_file}\n")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"  Baseline Uniform PR:     {pr_uniform*100:.2f}%  (Wall Time: {time_uniform:.2f}s)\n")
        f.write(f"  Baseline Gaussian PR:    {pr_gaussian*100:.2f}%  (Wall Time: {time_gaussian:.2f}s)\n")
        f.write(f"  PGD Robust Accuracy:     {results_pgd['robust_accuracy']:.2f}%  (Wall Time: {results_pgd['wall_time']:.2f}s)\n")
        f.write(f"  PGD Attack Success Rate: {results_pgd['attack_success_rate']:.2f}%\n")
        f.write(f"  CW Robust Accuracy:      {results_cw['robust_accuracy']:.2f}%  (Wall Time: {results_cw['wall_time']:.2f}s)\n")
        f.write(f"  CW Attack Success Rate:  {results_cw['attack_success_rate']:.2f}%\n")
        f.write(f"\n  Total Wall Time:         {total_wall_time:.2f}s\n")
        f.write(f"\n")


if __name__ == "__main__":
    main()
