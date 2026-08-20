# adv_loss.py - Adversarial training attack and loss functions
#
# Requirements:
#   torch >= 2.0
#
# Supports:
#   - PGD-AT  (Madry et al., 2018)  — L-inf and L2
#   - TRADES  (Zhang et al., 2019)  — L-inf and L2
#   - CW      (Carlini & Wagner, 2017)
#   - FGSM-RS (Wong et al., 2020)   — single-step
#   - MART    (Wang et al., 2020)   — L-inf and L2
#
# Related modules following the same conventions:
#   - src/mma_loss.py    MMA (Ding et al., 2020) — per-example margin radius
#   - src/atpr_loss.py   AT-PR (Zhang et al., 2025) — probabilistic robustness
#   - src/pos_geo_loss.py  margin level-set perturbations
#
# Conventions shared by every function here:
#   * inputs x are raw images in [0, 1]; normalisation lives inside the model
#   * attacks  `*_attack(model, x, y, ...)`       -> x_adv
#   * losses   `*_loss(model, x, y, ...)`         -> (loss, x_adv)
#     so a training loop can evaluate robust accuracy on the returned x_adv.
#   * the inner loop runs under model.eval(), the outer loss under model.train()
#   * everything is device-agnostic — no .cuda() calls

from typing import Literal, Tuple, overload

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

# pgd_attack's return type depends on return_path, so type checkers otherwise
# infer a union at every call site — callers then cannot use the result as a
# plain tensor without narrowing it first. These two stubs pin it down; they
# declare nothing the docstring does not already say.

@overload
def pgd_attack(model, x, y, epsilon, alpha, num_steps, norm=..., random_start=...,
               return_path: Literal[False] = False) -> torch.Tensor: ...

@overload
def pgd_attack(model, x, y, epsilon, alpha, num_steps, norm=..., random_start=...,
               *, return_path: Literal[True]) -> Tuple[torch.Tensor, torch.Tensor]: ...


def pgd_attack(model, x, y, epsilon, alpha, num_steps, norm="linf",
               random_start=True, return_path=False):
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
        return_path: If True, also return the perturbation trajectory
                     Delta = [delta_1, ..., delta_T] with delta_t = x_adv_t - x.

    Returns:
        x_adv: Adversarial examples (B, C, H, W)
        path (only if return_path): tensor (num_steps, B, C, H, W) of per-step
              perturbations delta_t, recorded after each PGD step.
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

    path = [] if return_path else None
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        if norm == "linf":
            x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        else:  # l2
            x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)

        if return_path:
            path.append((x_adv - x).detach())

    if return_path:
        return x_adv, torch.stack(path, dim=0)
    return x_adv


# ------------------------------------------------------------------
#                         PGD-AT loss
# ------------------------------------------------------------------

def pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm="linf",
                random_start=True):
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
        random_start: Whether the inner PGD attack starts from a random perturbation

    Returns:
        loss: Adversarial loss
        x_adv: Generated adversarial examples
    """
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps, norm=norm,
                       random_start=random_start)

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

# ------------------------------------------------------------------
#                         CW loss and attack
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
#                     FGSM-RS attack and loss
# ------------------------------------------------------------------

def fgsm_attack(model, x, y, epsilon, alpha=None, random_start=True):
    """
    FGSM with random start (FGSM-RS): a single-step attack.

    Wong et al., "Fast is Better than Free: Revisiting Adversarial Training", ICLR 2020

    The random start is what keeps single-step training from catastrophic
    overfitting; the step is taken *from* the randomly perturbed point and the
    result is projected back onto the epsilon-ball.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        y: True labels (B,)
        epsilon: Maximum perturbation radius
        alpha: Step size; defaults to 1.25 * epsilon as recommended in the paper
        random_start: Whether to start from a uniform random perturbation

    Returns:
        x_adv: Adversarial examples (B, C, H, W)
    """
    if alpha is None:
        alpha = 1.25 * epsilon

    x_adv = x.clone().detach()
    if random_start:
        x_adv = torch.clamp(x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon),
                            0.0, 1.0)

    x_adv.requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]

    return _linf_step(x_adv, x, grad, alpha, epsilon)


def fgsm_at_loss(model, x, y, epsilon, criterion, alpha=None, random_start=True):
    """
    FGSM-RS adversarial training loss (outer loop).

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon: Maximum perturbation radius
        criterion: Loss function (e.g., CrossEntropyLoss)
        alpha: FGSM step size; defaults to 1.25 * epsilon
        random_start: Whether the inner attack starts from a random perturbation

    Returns:
        loss: Adversarial loss
        x_adv: Generated adversarial examples
    """
    model.eval()
    x_adv = fgsm_attack(model, x, y, epsilon, alpha=alpha, random_start=random_start)

    model.train()
    loss = criterion(model(x_adv), y)

    return loss, x_adv


# ------------------------------------------------------------------
#                            MART loss
# ------------------------------------------------------------------

def mart_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm="linf"):
    """
    MART loss (outer loop): weights the robustness term by how confidently each
    example is classified when clean.

    Wang et al., "Improving Adversarial Robustness Requires Revisiting
    Misclassified Examples", ICLR 2020

    Loss = [ BCE(f(x_adv), y) ] + beta * [ KL(f(x_adv) || f(x)) weighted by (1 - p_y(x)) ]

    where the first term adds a margin penalty on the runner-up class and the
    KL weight (1 - p_y) concentrates the robustness pressure on examples the
    model is *not* already confident about.

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon, alpha, num_steps: PGD parameters for the inner loop
        beta: Trade-off parameter (typically 5.0 ~ 6.0)
        criterion: Unused; accepted so the signature matches the other AT losses
        norm: "linf" or "l2"

    Returns:
        loss: Combined MART loss
        x_adv: Generated adversarial examples
    """
    norm = norm.lower()
    model.eval()

    # MART's reference implementation starts from a small Gaussian ball rather
    # than the uniform epsilon-ball used by PGD-AT / TRADES.
    x_adv = torch.clamp(x.detach() + 0.001 * torch.randn_like(x), 0.0, 1.0)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, x_adv)[0]

        if norm == "linf":
            x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        elif norm == "l2":
            x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    model.train()

    logits_clean = model(x)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    # Boosted CE: penalise the most likely *wrong* class explicitly.
    top2 = torch.argsort(adv_probs, dim=1)[:, -2:]
    runner_up = torch.where(top2[:, -1] == y, top2[:, -2], top2[:, -1])
    loss_adv = (F.cross_entropy(logits_adv, y)
                + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), runner_up))

    # KL term, weighted per example by (1 - p_y(x)).
    nat_probs = F.softmax(logits_clean, dim=1)
    true_probs = nat_probs.gather(1, y.view(-1, 1)).squeeze(1)
    kl = F.kl_div(torch.log(adv_probs + 1e-12), nat_probs, reduction="none").sum(dim=1)
    loss_robust = (kl * (1.0000001 - true_probs)).mean()

    return loss_adv + float(beta) * loss_robust, x_adv
