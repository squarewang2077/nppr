# ad_attacker.py - Adversarial training attack and loss functions
#
# Requirements:
#   torch >= 2.0
#
# Supports:
#   - PGD-AT (Madry et al., 2018)
#   - TRADES (Zhang et al., 2019)

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(model, x, y, epsilon, alpha, num_steps, random_start=True):
    """
    PGD attack (inner loop) for generating adversarial examples.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        epsilon: Maximum perturbation (L-inf)
        alpha: Step size per iteration
        num_steps: Number of PGD steps
        random_start: Whether to start from random perturbation

    Returns:
        x_adv: Adversarial examples (B, C, H, W)
    """
    x_adv = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        # Project back to epsilon ball and valid image range
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv


def pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion):
    """
    PGD-AT loss (outer loop): Train on adversarial examples.

    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon, alpha, num_steps: PGD parameters
        criterion: Loss function (e.g., CrossEntropyLoss)

    Returns:
        loss: Adversarial loss
        x_adv: Generated adversarial examples
    """
    model.eval()  # Use eval mode for attack generation
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps)

    model.train()  # Switch back to train mode for loss computation
    logits_adv = model(x_adv)
    loss = criterion(logits_adv, y)

    return loss, x_adv


def trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion):
    """
    TRADES loss (outer loop): Combines clean loss with robustness regularization.

    Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy", ICML 2019

    Loss = CE(f(x), y) + beta * KL(f(x) || f(x_adv))

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon, alpha, num_steps: PGD parameters
        beta: Trade-off parameter (typically 1.0 ~ 6.0)
        criterion: Loss function for clean examples

    Returns:
        loss: Combined TRADES loss
        x_adv: Generated adversarial examples
    """
    model.eval()

    # Get clean logits for KL target
    with torch.no_grad():
        logits_clean = model(x)
        p_clean = F.softmax(logits_clean, dim=1)

    # Generate adversarial examples using KL divergence
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)

        # KL divergence: KL(p_clean || p_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            p_clean,
            reduction='batchmean'
        )

        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()

    model.train()

    # Compute final TRADES loss
    logits_clean = model(x)
    logits_adv = model(x_adv)

    loss_natural = criterion(logits_clean, y)
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_clean, dim=1),
        reduction='batchmean'
    )

    loss = loss_natural + beta * loss_robust

    return loss, x_adv
