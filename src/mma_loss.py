# mma_loss.py - MMA: per-example margin maximisation
#
# Requirements:
#   torch >= 2.0
#
# Ding et al., "MMA Training: Direct Input Space Margin Maximization through
# Adversarial Training", ICLR 2020
#
# MMA does not train at one fixed epsilon. Every example gets its own radius —
# an estimate of its distance to the decision boundary — and the loss pushes
# that boundary outwards for the examples that are still fragile.
#
# Conventions shared with src/adv_loss.py:
#   * inputs x are raw images in [0, 1]; normalisation lives inside the model
#   * the inner loop runs under model.eval(), the outer loss under model.train()
#   * everything is device-agnostic — no .cuda() calls
#
# Two deliberate deviations from the adv_loss family, which is why this lives
# in its own file:
#   * epsilon is per-example, a (B,) tensor rather than a scalar, so the shared
#     _linf_step / _l2_step helpers in adv_loss.py do not apply — the
#     projection helpers below take a (B,) radius instead.
#   * mma_loss returns (loss, x_adv, curr_eps), not the usual (loss, x_adv):
#     the per-example radius has to persist across epochs, and MMA cannot be
#     made stateless without discarding the margin estimate.
#     (src/pos_geo_loss.py returns a third value for its own reasons.)
#
# Known overlap, not fixed here: elementwise_margin below and
# pos_geo_loss.margin compute the identical quantity via different
# implementations. Unifying them touches scripts/pr_training/pos_geo_training.py
# and belongs in its own change.

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
#                        Margin utilities
# ------------------------------------------------------------------

def elementwise_margin(logits, y):
    """
    Per-example logit margin: f_y - max_{j != y} f_j.

    Positive means correctly classified; the value is the distance (in logit
    space) to the decision boundary.

    Args:
        logits: Model outputs (B, C)
        y: True labels (B,)

    Returns:
        margin: (B,) tensor, > 0 iff the example is correctly classified
    """
    correct = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.masked_fill(
        F.one_hot(y, logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    return correct - other


def _batch_scale(vec, x):
    """Multiply each sample x[i] by its own scalar vec[i]."""
    return x * vec.view(-1, *([1] * (x.dim() - 1)))


def _project_delta(delta, eps, norm):
    """Project each perturbation onto its *own* epsilon-ball."""
    if norm == "linf":
        eps_b = eps.view(-1, *([1] * (delta.dim() - 1)))
        return torch.min(torch.max(delta, -eps_b), eps_b)

    if norm == "l2":
        B = delta.size(0)
        flat = delta.view(B, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(torch.ones_like(norms), eps.view(B, 1) / norms)
        return (flat * factors).view_as(delta)

    raise ValueError(f"Unsupported norm: {norm}")


def bisection_search(model, x, y, direction, max_eps, num_search_steps,
                     lo=None, hi=None):
    """
    Find, per example, the smallest radius along `direction` at which the model
    flips from correct to incorrect — i.e. the input-space margin.

    The margin is monotonically decreasing in the radius along a fixed
    adversarial direction, so a bisection converges on the crossing point:
    while the example is still correct the radius must grow, once it is
    misclassified the radius must shrink.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        direction: Unit-scale perturbation direction (B, C, H, W)
        max_eps: (B,) per-example upper bound on the search
        num_search_steps: Number of bisection iterations
        lo, hi: Optional (B,) initial brackets; default [0, max_eps]

    Returns:
        eps: (B,) estimated distance to the decision boundary, in [0, max_eps]
    """
    lo = torch.zeros_like(max_eps) if lo is None else lo.clone()
    hi = max_eps.clone() if hi is None else hi.clone()

    eps = (lo + hi) / 2
    for _ in range(num_search_steps):
        x_probe = torch.clamp(x + _batch_scale(eps, direction), 0.0, 1.0)
        margin = elementwise_margin(model(x_probe), y)

        still_correct = margin > 0
        lo = torch.where(still_correct, eps, lo)    # boundary lies further out
        hi = torch.where(still_correct, hi, eps)    # boundary lies closer in
        eps = (lo + hi) / 2

    return eps.clamp(max=max_eps)


# ------------------------------------------------------------------
#                    Per-example PGD attack
# ------------------------------------------------------------------

def mma_pgd(model, x, y, eps, num_steps, norm="linf", eps_iter_scale=2.5):
    """
    PGD with a *per-example* perturbation budget.

    MMA gives every training example its own radius, so adv_loss.pgd_attack
    (single scalar epsilon) cannot be reused here.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        eps: (B,) per-example radius
        num_steps: Number of PGD steps
        norm: "linf" or "l2"
        eps_iter_scale: Step size is eps_iter_scale * eps / num_steps

    Returns:
        x_adv: Adversarial examples (B, C, H, W)
    """
    norm = norm.lower()
    alpha = eps_iter_scale * eps / max(num_steps, 1)

    delta = torch.zeros_like(x).uniform_(-1.0, 1.0)
    delta = _project_delta(_batch_scale(eps, delta), eps, norm)
    x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        if norm == "linf":
            step = _batch_scale(alpha, grad.sign())
        else:  # l2
            B = grad.size(0)
            flat = grad.view(B, -1)
            unit = (flat / flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)).view_as(grad)
            step = _batch_scale(alpha, unit)

        delta = _project_delta(x_adv.detach() + step - x, eps, norm)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv


# ------------------------------------------------------------------
#                            MMA loss
# ------------------------------------------------------------------

def mma_loss(model, x, y, prev_eps, criterion, max_eps=None, d_max=None,
             num_steps=10, num_search_steps=10, norm="linf",
             eps_iter_scale=2.5):
    """
    MMA loss (outer loop): maximise each example's *input-space margin* rather
    than train at one fixed epsilon.

    Ding et al., "MMA Training: Direct Input Space Margin Maximization through
    Adversarial Training", ICLR 2020

    Per example:
      * misclassified when clean  -> plain CE on the clean input (pull it back)
      * correctly classified      -> CE at the point on the decision boundary,
                                     which pushes the boundary outwards; skipped
                                     once the margin already exceeds d_max
                                     (the hinge), so effort goes to the examples
                                     that are still fragile.

    The per-example radius is stateful across epochs: pass the previous
    epoch's `curr_eps` back in as `prev_eps`, keyed by dataset index.

    Note this returns three values, unlike the losses in src/adv_loss.py — MMA
    cannot be made stateless without discarding the margin estimate.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        prev_eps: (B,) radius from the previous epoch; a scalar is broadcast
        criterion: Loss function used for the clean/misclassified branch
        max_eps: Upper bound on the margin search (default: 2 * mean(prev_eps))
        d_max: Hinge — margins beyond this contribute no gradient
               (default: max_eps)
        num_steps: PGD steps used to find the adversarial direction
        num_search_steps: Bisection iterations for the margin estimate
        norm: "linf" or "l2"
        eps_iter_scale: PGD step size scale, see mma_pgd

    Returns:
        loss: Combined MMA loss
        x_adv: Examples at the estimated decision boundary (B, C, H, W)
        curr_eps: (B,) updated per-example margins, to carry into the next epoch
    """
    norm = norm.lower()
    B = x.size(0)

    if not torch.is_tensor(prev_eps):
        prev_eps = torch.full((B,), float(prev_eps), device=x.device, dtype=x.dtype)
    prev_eps = prev_eps.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)

    if max_eps is None:
        max_eps = 2.0 * prev_eps.mean().item()
    if d_max is None:
        d_max = max_eps
    max_eps_vec = torch.full((B,), float(max_eps), device=x.device, dtype=x.dtype)

    model.eval()

    # Which examples are already wrong on clean data?
    with torch.no_grad():
        clean_margin = elementwise_margin(model(x), y)
    is_correct = clean_margin > 0

    # Adversarial direction at the current radius, then bisect for the margin.
    x_adv = mma_pgd(model, x, y, prev_eps.clamp(max=max_eps), num_steps,
                    norm=norm, eps_iter_scale=eps_iter_scale)
    direction = _batch_scale(1.0 / prev_eps, x_adv - x)

    with torch.no_grad():
        curr_eps = bisection_search(model, x, y, direction, max_eps_vec,
                                    num_search_steps)

    # Point on the estimated decision boundary.
    x_margin = torch.clamp(x + _batch_scale(curr_eps, direction), 0.0, 1.0).detach()

    model.train()

    # Correct examples contribute the margin term, but only while inside the
    # hinge; misclassified ones contribute the clean term.
    in_hinge = is_correct & (curr_eps < d_max)
    loss_margin = F.cross_entropy(model(x_margin), y, reduction="none")
    loss_clean = F.cross_entropy(model(x), y, reduction="none")

    per_example = torch.where(in_hinge, loss_margin,
                              torch.where(is_correct,
                                          torch.zeros_like(loss_clean),
                                          loss_clean))
    loss = per_example.mean()

    return loss, x_margin, curr_eps.detach()
