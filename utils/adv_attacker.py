# adv_attacker.py - Adversarial training attack and loss functions
#
# Requirements:
#   torch >= 2.0
#
# Supports:
#   - PGD-AT (Madry et al., 2018)  — L-inf and L2
#   - TRADES (Zhang et al., 2019)  — L-inf and L2

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
#                        Shared helpers
# ------------------------------------------------------------------

def _l2_random_init(x, epsilon):
    """Random start on the L2 sphere of radius epsilon."""
    noise = torch.randn_like(x)
    flat = noise.view(noise.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    noise = (flat / norms * epsilon).view_as(x)
    return torch.clamp(x + noise, 0.0, 1.0)


def _l2_step(x_adv, x, grad, alpha, epsilon):
    """Normalised gradient step + L2 projection + image clamp."""
    B = grad.size(0)
    grad_flat = grad.view(B, -1)
    grad_norms = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    grad_unit = (grad_flat / grad_norms).view_as(grad)

    x_adv = x_adv.detach() + alpha * grad_unit

    # Project delta back onto L2 ball
    delta = x_adv - x
    delta_flat = delta.view(B, -1)
    delta_norms = delta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factors = torch.minimum(torch.ones_like(delta_norms), epsilon / delta_norms)
    delta = (delta_flat * factors).view_as(delta)

    return torch.clamp(x + delta, 0.0, 1.0).detach()


def _linf_step(x_adv, x, grad, alpha, epsilon):
    """Sign gradient step + L-inf projection + image clamp."""
    x_adv = x_adv.detach() + alpha * grad.sign()
    delta = torch.clamp(x_adv - x, -epsilon, epsilon)
    return torch.clamp(x + delta, 0.0, 1.0).detach()


# ------------------------------------------------------------------
#                          PGD attack
# ------------------------------------------------------------------

def pgd_attack(model, x, y, epsilon, alpha, num_steps, norm="linf", random_start=True):
    """
    PGD attack (inner loop) for generating adversarial examples.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)  — expected in [0, 1]
        y: True labels (B,)
        epsilon: Maximum perturbation radius
        alpha: Step size per iteration
        num_steps: Number of PGD steps
        norm: "linf" or "l2"
        random_start: Whether to start from a random perturbation

    Returns:
        x_adv: Adversarial examples (B, C, H, W)
    """
    norm = norm.lower()
    x_adv = x.clone().detach()

    if random_start:
        if norm == "linf":
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif norm == "l2":
            x_adv = _l2_random_init(x_adv, epsilon)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        if norm == "linf":
            x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        else:  # l2
            x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)

    return x_adv


# ------------------------------------------------------------------
#                         PGD-AT loss
# ------------------------------------------------------------------

def pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm="linf"):
    """
    PGD-AT loss (outer loop): Train on adversarial examples.

    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon, alpha, num_steps: PGD parameters
        criterion: Loss function (e.g., CrossEntropyLoss)
        norm: "linf" or "l2"

    Returns:
        loss: Adversarial loss
        x_adv: Generated adversarial examples
    """
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps, norm=norm)

    model.train()
    logits_adv = model(x_adv)
    loss = criterion(logits_adv, y)

    return loss, x_adv


# ------------------------------------------------------------------
#                         TRADES loss
# ------------------------------------------------------------------

def trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm="linf"):
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
        norm: "linf" or "l2"

    Returns:
        loss: Combined TRADES loss
        x_adv: Generated adversarial examples
    """
    norm = norm.lower()
    model.eval()

    # Clean logits (fixed KL target)
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)

    # Random start
    x_adv = x.clone().detach()
    if norm == "linf":
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == "l2":
        x_adv = _l2_random_init(x_adv, epsilon)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    # Inner maximisation loop (maximise KL)
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss_kl = F.kl_div(
            F.log_softmax(model(x_adv), dim=1),
            p_clean,
            reduction="batchmean",
        )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]

        if norm == "linf":
            x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        else:  # l2
            x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)

    model.train()

    # Outer loss
    logits_clean = model(x)
    logits_adv   = model(x_adv)

    loss_natural = criterion(logits_clean, y)
    loss_robust  = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_clean, dim=1),
        reduction="batchmean",
    )

    return loss_natural + beta * loss_robust, x_adv
