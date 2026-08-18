# pos_geo_loss.py - Margin level-set perturbations and position geometry
#
# Requirements:
#   torch >= 2.0
#
# Studies how *position on a margin level set* affects adversarial training.
# For a target margin level t, N random starts inside the epsilon-ball are each
# driven onto the surface {delta : m(x + delta, y) = t}, and the geometry of
# where they land is measured.
#
# Two pieces make the "different positions, same level" question meaningful:
#   * a *symmetric* penalty on m - t, so perturbations approach the level from
#     either side instead of being pushed past it;
#   * an L2 anchor to each perturbation's own random start, which keeps the
#     landing point attributable to the start it came from.
#
# On the anchor, measured rather than assumed: in input dimensions this large
# (3072 for CIFAR) random starts are already ~19 apart and the solver moves
# each one only ~0.6, so the particles never collapse together even with
# anchor_lambda = 0. What the anchor actually trades off is drift from the
# start against reaching the level set at all — on CIFAR, lambda >= 0.2 pulls
# hard enough that most perturbations no longer land within tol of t. Keep it
# small (~0.01-0.05) and check the valid fraction.
#
# Conventions shared with src/adv_loss.py:
#   * inputs x are raw images in [0, 1]; normalisation lives inside the model
#   * the inner solver runs under model.eval(), the outer loss under model.train()
#   * everything is device-agnostic — no .cuda() calls
#   * pos_geo_loss returns (loss, x_adv, info) and its x_adv is
#     (B, N, C, H, W), not the (B, C, H, W) used elsewhere — there are N
#     perturbations per image here. (adv_loss.mma_loss sets the precedent for
#     a loss returning more than the usual pair.)

import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
#                        Margin and energy
# ------------------------------------------------------------------

def margin(logits, y):
    """
    Multiclass margin: m(x, y) = f_y(x) - max_{j != y} f_j(x).

    Positive means correctly classified; the magnitude is the distance to the
    decision boundary in logit space.

    Args:
        logits: Model outputs (B, C)
        y: True labels (B,)

    Returns:
        margins: (B,)
    """
    f_y = logits.gather(1, y.view(-1, 1)).squeeze(1)

    wrong_logits = logits.clone()
    wrong_logits.scatter_(1, y.view(-1, 1), float("-inf"))
    f_other = wrong_logits.max(dim=1).values

    return f_y - f_other


def top2_margins(logits, y):
    """
    Margins against the two strongest wrong classes.

    m1 = z_y - z_j1 is the usual margin (j1 = the top wrong class); m2 uses the
    runner-up wrong class j2. The gap m2 - m1 says how clearly j1 owns the
    decision — a small gap means the "opponent" can flip between steps, which
    is exactly when a level-set position is unstable.

    Args:
        logits: Model outputs (B, C)
        y: True labels (B,)

    Returns:
        m1, m2: (B,) margins, m1 <= m2
        j1, j2: (B,) the corresponding wrong-class indices
    """
    f_y = logits.gather(1, y.view(-1, 1)).squeeze(1)

    wrong = logits.clone()
    wrong.scatter_(1, y.view(-1, 1), float("-inf"))
    top2 = wrong.topk(2, dim=1)

    j1, j2 = top2.indices[:, 0], top2.indices[:, 1]
    return f_y - top2.values[:, 0], f_y - top2.values[:, 1], j1, j2


def dual_norm(g, norm):
    """
    Dual norm of a per-sample gradient, matching the ball the perturbations
    live in: the dual of an L-inf constraint is L1, and L2 is self-dual.

    Args:
        g: Gradients (B, N, C, H, W)
        norm: "linf" or "l2" — the *primal* ball

    Returns:
        (B, N) dual norms
    """
    flat = g.flatten(2)
    if norm == "linf":
        return flat.abs().sum(dim=-1)
    if norm == "l2":
        return flat.norm(p=2, dim=-1)
    raise ValueError(f"Unsupported norm: {norm}")


def symmetric_softplus(z, alpha=10.0):
    """
    Smooth, symmetric penalty on z — a soft |z|.

        psi(z) = [softplus(alpha z) + softplus(-alpha z)] / alpha - 2 ln2 / alpha

    Its derivative is tanh(alpha z / 2), which is positive for z > 0 and
    negative for z < 0, so minimising it drives z toward 0 *from either side*.
    That is what makes the solver land on the level set m = t rather than
    overshooting past it, as a one-sided hinge would.

    The constant is subtracted only so psi(0) = 0 and the reported value reads
    as a distance; it does not change the gradient.

    Args:
        z: Any shape
        alpha: Sharpness; larger is closer to |z|

    Returns:
        psi: Same shape as z, >= 0
    """
    return (F.softplus(alpha * z) + F.softplus(-alpha * z)) / alpha - 2.0 * math.log(2.0) / alpha


# ------------------------------------------------------------------
#                      Sampling and projection
# ------------------------------------------------------------------

def random_starts(x, epsilon, num_starts, norm="linf"):
    """
    Sample N random perturbations inside the epsilon-ball, one set per image.

    Args:
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        epsilon: Radius of the ball
        num_starts: Number of perturbations per image (N)
        norm: "linf" or "l2"

    Returns:
        delta: (B, N, C, H, W), already projected to be feasible for x
    """
    norm = norm.lower()
    B, C, H, W = x.shape
    shape = (B, num_starts, C, H, W)

    if norm == "linf":
        delta = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        # Direction uniform on the sphere, radius ~ U(0, 1)^(1/d) for a
        # uniform fill of the ball rather than a shell.
        delta = torch.randn(shape, device=x.device, dtype=x.dtype)
        flat = delta.view(B, num_starts, -1)
        d = flat.size(-1)
        unit = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand(B, num_starts, 1, device=x.device, dtype=x.dtype) ** (1.0 / d)
        delta = (unit * radius * epsilon).view_as(delta)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    return project_delta(delta, x, epsilon, norm)


def project_delta(delta, x, epsilon, norm):
    """
    Project perturbations onto the feasible set:
        ||delta||_p <= epsilon   and   0 <= x + delta <= 1

    Args:
        delta: Perturbations (B, N, C, H, W)
        x: Clean inputs (B, C, H, W)
        epsilon: Radius of the ball
        norm: "linf" or "l2"

    Returns:
        delta: Projected perturbations, same shape
    """
    norm = norm.lower()
    x_ = x.unsqueeze(1)

    if norm == "linf":
        delta = delta.clamp(-epsilon, epsilon)
    elif norm == "l2":
        B, N = delta.shape[:2]
        flat = delta.view(B, N, -1)
        norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = (flat * (epsilon / norms).clamp_max(1.0)).view_as(delta)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    # Clamp into the valid image range last, which can only shrink the norm.
    return (x_ + delta).clamp(0.0, 1.0) - x_


# ------------------------------------------------------------------
#                       Level-set solver
# ------------------------------------------------------------------

def solve_level_perturbations(model, x, y, t, epsilon, *,
                              num_starts=8, num_steps=50, step_size=1e-2,
                              anchor_lambda=1.0, alpha=10.0, tol=0.05,
                              norm="linf", delta0=None):
    """
    Drive N random perturbations per image onto the margin level set m = t.

    Minimises, per perturbation:

        J(delta) = symmetric_softplus(m(x + delta, y) - t)
                   + anchor_lambda * ||delta - delta_0||^2

    The anchor keeps each perturbation near the start it came from, so its
    landing point stays attributable to that start. It is a fidelity knob, not
    a diversity one — see the note at the top of this file — and large values
    stop perturbations reaching the level set at all, so watch valid_mask.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        y: True labels (B,)
        t: Target margin level
        epsilon: Radius of the ball the perturbations live in
        num_starts: Number of perturbations per image (N)
        num_steps: Gradient steps; 0 returns the random starts unoptimised
        step_size: Gradient step size
        anchor_lambda: Strength of the L2 pull toward each random start.
                       Keep small (~0.01-0.05); large values prevent the
                       perturbations from reaching the level set.
        alpha: Sharpness of the symmetric penalty
        tol: A perturbation counts as valid when |m - t| <= tol
        norm: "linf" or "l2"
        delta0: Optional starting perturbations (B, N, C, H, W); sampled when None

    Returns:
        delta: (B, N, C, H, W) detached perturbations
        margins: (B, N) final margins
        valid_mask: (B, N) bool, True where |m - t| <= tol
        grad_dual: (B, N) dual norm of the CE gradient at each landing point —
                   the local sharpness of the loss surface there, used to build
                   the sharp / flat weightings.
    """
    was_training = model.training
    model.eval()

    if delta0 is None:
        delta0 = random_starts(x, epsilon, num_starts, norm=norm)
    anchor = delta0.detach()
    delta = anchor.clone()

    B, N = delta.shape[:2]
    x_rep = x.unsqueeze(1)
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

    for _ in range(num_steps):
        delta.requires_grad_(True)
        logits = model((x_rep + delta).reshape(B * N, *x.shape[1:]))
        margins = margin(logits, y_rep).view(B, N)

        energy = symmetric_softplus(margins - t, alpha=alpha)
        anchor_pen = (delta - anchor).flatten(2).pow(2).sum(dim=-1)
        objective = (energy + anchor_lambda * anchor_pen).sum()

        grad = torch.autograd.grad(objective, delta)[0]
        delta = project_delta(delta.detach() - step_size * grad, x, epsilon, norm)

    # One final pass over the perturbations we are about to return: the CE
    # gradient here gives the local sharpness, and its logits give the margins.
    delta = delta.detach().requires_grad_(True)
    logits = model((x_rep + delta).reshape(B * N, *x.shape[1:]))
    ce = F.cross_entropy(logits, y_rep, reduction="sum")
    grad = torch.autograd.grad(ce, delta)[0]

    with torch.no_grad():
        margins = margin(logits, y_rep).view(B, N)
        grad_dual = dual_norm(grad, norm)

    model.train(was_training)

    return delta.detach(), margins, (margins - t).abs() <= tol, grad_dual


# ------------------------------------------------------------------
#                       Position weighting
# ------------------------------------------------------------------

WEIGHT_MODES = ("uniform", "sharp", "flat", "min_norm", "max_norm")


def pos_geo_weights(grad_dual, delta_norm, valid_mask, mode="uniform", tau=1.0):
    """
    Decide how much each landing point contributes to the training loss.

    The five modes are the experiment: they ask whether it matters *where* on
    the level set the training signal comes from.

        uniform   every valid position counts equally
        sharp     favour large ||grad_delta CE||_*  — steep spots
        flat      favour small ||grad_delta CE||_*  — wide-valley spots
        min_norm  all weight on the position closest to the clean image
        max_norm  all weight on the position furthest from it

    Perturbations that never reached the level set get zero weight in every
    mode: the premise is "different positions at the *same* level", so a point
    that is not on the level does not belong in the average. Rows where nothing
    is valid come back all-zero and contribute no gradient — watch valid_rate.

    Args:
        grad_dual: (B, N) dual-norm CE gradient at each position
        delta_norm: (B, N) distance of each perturbation from the origin
        valid_mask: (B, N) bool, which positions reached the level set
        mode: one of WEIGHT_MODES
        tau: softmax temperature for sharp / flat. Large tau flattens them
             back toward uniform.

    Returns:
        weights: (B, N), each row summing to 1 over its valid entries (or all
                 zero if the row has none)
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got {mode!r}")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    valid = valid_mask.to(grad_dual.dtype)
    neg_inf = torch.finfo(grad_dual.dtype).min

    if mode == "uniform":
        w = valid
    elif mode in ("sharp", "flat"):
        sign = 1.0 if mode == "sharp" else -1.0
        scores = sign * grad_dual / tau
        # Centre per row for numerical stability, then mask before softmax so
        # invalid positions cannot take any mass.
        scores = scores - scores.amax(dim=1, keepdim=True)
        w = torch.softmax(scores.masked_fill(~valid_mask, neg_inf), dim=1)
        w = w * valid                      # rows with no valid entry -> all zero
    else:  # min_norm / max_norm
        scores = delta_norm.masked_fill(~valid_mask,
                                        neg_inf if mode == "max_norm" else -neg_inf)
        pick = scores.argmax(dim=1) if mode == "max_norm" else scores.argmin(dim=1)
        w = torch.zeros_like(grad_dual)
        w.scatter_(1, pick.view(-1, 1), 1.0)
        w = w * valid

    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------
#                     Level-set training loss
# ------------------------------------------------------------------

def pos_geo_loss(model, x, y, criterion, t, epsilon, *,
                 weight_mode="uniform", tau=1.0, train_mode=True,
                 **solver_kwargs):
    """
    Weighted PGD-style adversarial training on margin level-set perturbations.

        loss = sum_r w_r * CE(f(x + delta_r), y)

    Both delta and w are detached, so the gradient flows only into the model —
    the positions and their weights are treated as fixed data for the step.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        criterion: Unused; accepted so the signature matches the other AT losses
        t: Target margin level
        epsilon: Radius of the ball the perturbations live in
        weight_mode: one of WEIGHT_MODES — see pos_geo_weights
        tau: softmax temperature for the sharp / flat modes
        train_mode: run the outer forward pass in train mode, as adversarial
                    training requires. Pass False when calling this purely for
                    diagnostics: a train-mode forward updates BatchNorm running
                    statistics, which would let a measurement change the model
                    it is measuring.
        **solver_kwargs: Forwarded to solve_level_perturbations
                         (num_starts, num_steps, step_size, anchor_lambda,
                          alpha, tol, norm, delta0)

    Returns:
        loss: Scalar training loss
        x_adv: (B, N, C, H, W) perturbed inputs
        info: per-position diagnostics, all (B, N) unless noted —
              margin1 / margin2 (top-2 wrong classes), j1 / j2 (their indices),
              ce (per-position cross-entropy), grad_dual, delta_norm, weights,
              valid_mask, and logits_adv (B * N, num_classes) so the caller can
              compute accuracy without a second forward pass.
    """
    norm = solver_kwargs.get("norm", "linf")
    delta, _, valid_mask, grad_dual = solve_level_perturbations(
        model, x, y, t, epsilon, **solver_kwargs
    )

    B, N = delta.shape[:2]
    delta_norm = delta.flatten(2).norm(p=2, dim=-1)
    weights = pos_geo_weights(grad_dual, delta_norm, valid_mask,
                              mode=weight_mode, tau=tau).detach()

    was_training = model.training
    model.train(train_mode)
    x_adv = (x.unsqueeze(1) + delta).clamp(0.0, 1.0)
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

    logits_adv = model(x_adv.reshape(B * N, *x.shape[1:]))
    ce = F.cross_entropy(logits_adv, y_rep, reduction="none").view(B, N)
    loss = (weights * ce).sum(dim=1).mean()

    model.train(was_training)

    with torch.no_grad():
        m1, m2, j1, j2 = top2_margins(logits_adv, y_rep)

    info = {
        "margin1": m1.view(B, N),
        "margin2": m2.view(B, N),
        "j1": j1.view(B, N),
        "j2": j2.view(B, N),
        "ce": ce.detach(),
        "grad_dual": grad_dual,
        "delta_norm": delta_norm,
        "weights": weights,
        "valid_mask": valid_mask,
        "logits_adv": logits_adv,
    }
    return loss, x_adv, info


# ------------------------------------------------------------------
#                       Position geometry
# ------------------------------------------------------------------

def level_geometry(delta, margins):
    """
    Describe where the perturbations landed, relative to each other.

    Pure tensor arithmetic — no model, no forward pass — so it is cheap enough
    to call every step and can be unit-tested on synthetic tensors.

    Args:
        delta: Perturbations (B, N, C, H, W)
        margins: Final margins (B, N)

    Returns:
        dict with
            delta_norm: (B, N)    distance of each perturbation from the origin
            pair_cos:   (B, N, N) cosine of the angle between perturbation pairs
            margins:    (B, N)    passed through, so callers log one object
    """
    flat = delta.flatten(2)
    norms = flat.norm(dim=-1)
    unit = flat / norms.unsqueeze(-1).clamp_min(1e-12)

    return {
        "delta_norm": norms,
        "pair_cos": torch.bmm(unit, unit.transpose(1, 2)).clamp(-1.0, 1.0),
        "margins": margins,
    }


def dispersion(pair_cos):
    """
    How spread out the N perturbations are, from the spectrum of their pairwise
    cosine matrix.

    That matrix is the Gram matrix of the unit perturbations, so it is
    symmetric PSD and `eigvalsh` gives its spectrum directly (cheaper and
    better conditioned than an SVD).

    The headline number is the participation ratio:

        eff_rank = (sum s)^2 / sum s^2       in [1, N]

    It reads directly as "how many independent directions the perturbations
    actually span": all N collapsed onto one direction gives 1, mutually
    orthogonal gives N. Unlike a condition number it stays finite when the
    perturbations become nearly collinear.

    Args:
        pair_cos: (B, N, N) pairwise cosine matrix from level_geometry

    Returns:
        dict of (B,) tensors: eff_rank, sigma_max, sigma_min, anisotropy
    """
    s = torch.linalg.eigvalsh(pair_cos.float()).clamp_min(0.0)   # ascending
    total = s.sum(dim=-1).clamp_min(1e-12)

    return {
        "eff_rank": total.pow(2) / s.pow(2).sum(dim=-1).clamp_min(1e-12),
        "sigma_max": s[:, -1],
        "sigma_min": s[:, 0],
        "anisotropy": s[:, -1] / total,
    }
