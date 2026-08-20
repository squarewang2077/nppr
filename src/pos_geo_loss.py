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
# start against reaching the level set at all.
#
# How much it costs depends on the solver, so the default is per-solver
# (_DEFAULT_ANCHOR_LAMBDA): 0.01 for "energy", **0 for "newton"**. A Newton
# step shrinks to nothing as the level set is approached while the anchor pull
# does not, so any fixed anchor eventually wins and drags the solver back off
# the level — measured, it held |m - t| at 0.20 instead of 0.003 and cost
# three quarters of the valid rate. See solve_level_perturbations.
#
# Conventions shared with src/adv_loss.py:
#   * inputs x are raw images in [0, 1]; normalisation lives inside the model
#   * the whole of pos_geo_loss runs under model.eval() — inner solver and
#     outer loss alike, so the level means the same thing in both
#   * everything is device-agnostic — no .cuda() calls
#   * pos_geo_loss returns (loss, x_adv, info) and its x_adv is
#     (B, N, C, H, W), not the (B, C, H, W) used elsewhere — there are N
#     perturbations per image here. (adv_loss.mma_loss sets the precedent for
#     a loss returning more than the usual pair.)

import math
from typing import NamedTuple

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

def matched_l2_epsilon(epsilon_linf, img_size, channels=3):
    """
    The L2 radius holding perturbations of the same typical size as an L-inf
    ball of radius epsilon_linf.

    Reusing an L-inf radius under norm="l2" gives a ball that is smaller by
    orders of magnitude in high dimension, and the level set then sits mostly
    outside it. A point drawn uniformly from the L-inf ball of radius e has
    expected squared L2 norm d * e^2 / 3, so the matching radius is
    e * sqrt(d / 3) — for square RGB images just e * img_size.

    Measured on an adversarially trained CIFAR-10 ResNet18 (t = half the median
    clean margin, tol = 0.05), against L-inf at 8/255 reaching the level set
    for 46% of perturbations at median gap 0.077:

        eps_2 = 0.50  (RobustBench convention)   valid 0.33   gap 0.187
        eps_2 = 1.00  (this rule)                valid 0.48   gap 0.062
        eps_2 = 1.74  (e * sqrt(d), whole ball)  valid 0.45   gap 0.081
        eps_2 = 3.00 / 5.00                      valid 0.47   gap 0.060

    So this rule matches L-inf and larger radii add nothing; the L-inf solver's
    own perturbations land at median ||delta||_2 = 1.035, right at it.

    Args:
        epsilon_linf: The L-inf radius to match
        img_size: Image side length in pixels
        channels: Image channels (3 for RGB)

    Returns:
        The matched L2 radius (float)
    """
    d = channels * img_size * img_size
    return epsilon_linf * math.sqrt(d / 3.0)


def random_starts(x, epsilon, num_starts, norm="linf"):
    """
    Sample N random perturbations inside the epsilon-ball, one set per image.

    The two norms differ in how exact the sampling is, which matters here
    because position on the level set is the thing being measured — a biased
    start distribution would show up as structure in the geometry:

      * linf: exactly uniform over the feasible set. The set is a box (the
        ball intersected with the image range), so it factorises per pixel and
        can be sampled directly.
      * l2: uniform over the ball, then projected into the image range. The
        feasible set is a ball-box intersection that does not factorise, so
        the projection leaves mass concentrated on the box faces for images
        with pixels near 0 or 1. Accepted for now — see the module header.

    Args:
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        epsilon: Radius of the ball
        num_starts: Number of perturbations per image (N)
        norm: "linf" or "l2"

    Returns:
        delta: (B, N, C, H, W), feasible for x — that is, ||delta|| <= epsilon
               and x + delta stays in [0, 1]
    """
    norm = norm.lower()
    B, C, H, W = x.shape
    shape = (B, num_starts, C, H, W)

    if norm == "linf":
        # Sample uniformly over the *intersection* of the epsilon-ball and the
        # image box, instead of sampling the ball and clamping afterwards.
        # Clamping would collapse everything outside the box onto its faces,
        # putting an atom of probability mass there for every pixel near 0 or 1
        # — exactly the pixels whose feasible interval is shortest. Per pixel
        # that interval is
        #     [max(-epsilon, -x),  min(epsilon, 1 - x)]
        # which is non-empty for any x in [0, 1] (it always contains 0), so the
        # result is feasible by construction and needs no projection.
        x_ = x.unsqueeze(1)
        low  = (-x_).clamp_min(-epsilon)          # max(-epsilon, -x)
        high = (1.0 - x_).clamp_max(epsilon)      # min(epsilon, 1 - x)

        u = torch.rand(shape, device=x.device, dtype=x.dtype)
        return low + (high - low) * u

    elif norm == "l2":
        # Direction uniform on the sphere, radius ~ U(0, 1)^(1/d) for a
        # uniform fill of the ball rather than a shell.
        delta = torch.randn(shape, device=x.device, dtype=x.dtype)
        flat = delta.view(B, num_starts, -1)
        d = flat.size(-1)
        unit = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12) # random direction
        radius = torch.rand(B, num_starts, 1, device=x.device, dtype=x.dtype) ** (1.0 / d) # random radius
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


def reachable_margin_floor(model, x, y, epsilon, norm="linf", num_steps=5):
    """
    Lowest margin reachable inside the epsilon-ball, estimated per sample.

    This is the floor a target level has to sit above to be attainable at all.
    It is not a fixed property of the data: adversarial training raises it, so
    a level that was reachable at epoch 1 can be outside the ball by epoch 6 —
    measured, 6 epochs took the median floor from -1.54 to -0.20, at which
    point a fixed t = -0.5 was unreachable for 89% of samples.

    Estimated by sign-PGD descending the margin. The sign matters: plain
    gradient descent badly under-estimates how far the ball reaches (measured
    -0.42 where sign-PGD found -1.26, a factor of three), which would make the
    floor look tighter than it is.

    Five steps is the recommended budget. Against a 50-step reference:

        model         1st-order   sPGD5    sPGD10   sPGD50
        random-init      -1.99    -1.53    -1.62    -1.69
        energy-6ep       -1.90    -1.34    -1.21    -1.26
        newton-6ep       -0.30    -0.21    -0.22    -0.24
        adv-trained      -7.58    -7.00    -7.30    -7.48

    so five steps gets 87-100% of the 50-step answer. The one-backward
    first-order bound m_clean - epsilon * ||grad m||_1 is also shown: it is
    cheap but optimistic by 17-51% because it ignores curvature and the [0, 1]
    clamp, and the bias moves with the model.

    Args:
        model: Target model, already in the caller's chosen mode
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        epsilon: Radius of the ball
        norm: "linf" or "l2"
        num_steps: sign-PGD steps

    Returns:
        (B,) estimated lowest reachable margin, detached
    """
    norm = norm.lower()
    alpha = 2.5 * epsilon / max(num_steps, 1)

    delta = random_starts(x, epsilon, 1, norm=norm).squeeze(1)
    for _ in range(num_steps):
        delta.requires_grad_(True)
        grad = torch.autograd.grad(margin(model(x + delta), y).sum(), delta)[0]
        # Descend the margin: steepest descent under the primal norm, which is
        # the sign for L-inf and the normalised gradient for L2.
        if norm == "linf":
            step = grad.sign()
        else:
            flat = grad.flatten(1)
            step = (flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)).view_as(grad)
        delta = project_delta((delta.detach() - alpha * step).unsqueeze(1),
                              x, epsilon, norm).squeeze(1)

    with torch.no_grad():
        return margin(model(x + delta), y).detach()


# ------------------------------------------------------------------
#                       Level-set solver
# ------------------------------------------------------------------

SOLVERS = ("energy", "newton")

# Standard step size per solver: 1.0 is the full minimum-norm Newton step,
# 0.2 is what the energy solver needs (see solve_level_perturbations).
_DEFAULT_STEP_SIZE = {"energy": 0.2, "newton": 1.0}

# Anchor strength per solver. Zero for newton is not a tuning preference — an
# anchor of any fixed strength eventually dominates a Newton step, which shrinks
# to nothing as the level set is approached. See solve_level_perturbations.
_DEFAULT_ANCHOR_LAMBDA = {"energy": 0.01, "newton": 0.0}


class LevelSolution(NamedTuple):
    """
    What the level-set solver found, per perturbation.

    Unpacks positionally like a plain tuple, but naming the fields keeps the
    three gradient quantities from being confused with one another — they are
    all (B, N) non-negative reals and nothing about their shape says which is
    which.

    Fields:
        delta:            (B, N, C, H, W) best point reached, detached
        margins:          (B, N) margin at that point
        valid_mask:       (B, N) bool, True where |margin - t| <= tol
        grad_margin_l2:   (B, N) ||grad_delta m||_2, the coarea density
        grad_margin_dual: (B, N) ||grad_delta m||_*, dual to the primal ball —
                          the Lp threat-aware thickness of the level set
        grad_ce_dual:     (B, N) ||grad_delta CE||_*, kept as a baseline and
                          as what the sharp / flat weightings rank by
    """
    delta: torch.Tensor
    margins: torch.Tensor
    valid_mask: torch.Tensor
    grad_margin_l2: torch.Tensor
    grad_margin_dual: torch.Tensor
    grad_ce_dual: torch.Tensor


def solve_level_perturbations(model, x, y, t, epsilon, *,
                              num_starts=8, num_steps=50, step_size=None,
                              anchor_lambda=None, alpha=10.0, tol=0.05,
                              norm="linf", delta0=None, solver="energy"):
    """
    Drive N random perturbations per image onto the margin level set m = t.

    Minimises, per perturbation:

        J(delta) = symmetric_softplus(m(x + delta, y) - t)
                   + anchor_lambda * ||delta - delta_0||^2

    The anchor keeps each perturbation near the start it came from, so its
    landing point stays attributable to that start. It is a fidelity knob, not
    a diversity one — see the note at the top of this file — and large values
    stop perturbations reaching the level set at all, so watch valid_mask.

    Each perturbation is reported at the *best* point it reached — the iterate
    minimising |m - t| — not wherever the fixed step budget left it. The two
    differ: the solver has no line search and can step past the level set and
    drift away again, and that drift would otherwise be recorded as the
    landing position, depressing valid_mask and adding noise to every geometry
    statistic measured downstream.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        y: True labels (B,)
        t: Target margin level
        epsilon: Radius of the ball the perturbations live in
        num_starts: Number of perturbations per image (N)
        num_steps: Solver steps; 0 returns the random starts unoptimised
        solver: how each step is taken.
            "energy" (default, the original): minimise
                symmetric_softplus(m - t) by gradient descent. The step along
                the normal is ss * psi'(m - t) * ||grad m||, and psi' saturates
                to +-1 for |m - t| beyond about 2/alpha — so the step is
                governed by ||grad m|| and barely shrinks as the level set is
                approached. Steep spots therefore get *larger* steps, which is
                backwards, and the useful alpha turns out to depend on the
                margin scale (measured: best alpha 10 at margin scale 0.6, 0.3
                at scale 3.1, with no stable alpha * scale rule).
            "newton": one minimum-norm Newton step for the root problem
                m(delta) = t, moving |m - t| / ||grad m|| along the normal.
                Dividing by ||grad m|| rather than multiplying is the whole
                difference: steep spots take small steps. It is scale free —
                multiply the logits by c and both (m - t) and ||grad m|| scale
                with c, leaving the step unchanged — so it needs no alpha and
                no per-model retuning. Same cost per step (one backward for
                grad m either way).

                Measured on CIFAR-10 ResNet18, 10 steps, one shared setting,
                valid_rate energy -> newton:
                    margin scale 0.6 (random init)  0.893 -> 0.945
                    margin scale 3.1 (trained)      0.220 -> 0.612
                    margin scale 12.2 (x4 logits)   0.045 -> 0.410
                The advantage grows with the margin scale, which is the
                direction training moves in. Not yet validated inside a live
                training loop, where the model itself is moving.

                Caveat: the step blows up where ||grad m|| -> 0. Measured step
                lengths have median 0.003-0.016 but p99 ~1.5, and only
                project_delta bounds them. An explicit trust region made things
                worse (0.945 -> 0.659), so none is applied.
        step_size: Step scale. None picks the solver's own default: 1.0 for
            "newton", which is the full Newton step, and 0.2 for "energy". For
            "energy" this is not the 1e-2 that
            looks natural for a gradient method — this solver is the difference
            between the experiment working and not. Measured on a randomly
            initialised CIFAR-10 ResNet18 (t = median clean margin, tol = 0.05,
            epsilon = 8/255):

                step_size  num_steps   valid_rate   |m - t| median   fwd/batch
                     0.01         10        0.238          0.1234          12
                     0.01         50        0.656          0.0183          52
                     0.20         10        0.844          0.0004          12
                     0.20         50        0.996          0.0002          52

            At 1e-2 the solver simply does not arrive: 10 steps leave |m - t|
            at 0.12, well outside tol, so most perturbations are never on the
            level set and the whole downstream analysis is measuring off-level
            points. At 0.2 ten steps suffice, which is also five times cheaper
            than the old fifty. Raising epsilon does not substitute for this
            (4x epsilon moved valid_rate only 0.203 -> 0.242) — the ball was
            never the binding constraint, the optimisation was.
        anchor_lambda: Strength of the pull toward each random start. None
            picks the solver's own default: 0.01 for "energy", **0 for
            "newton"**.

            The zero is not a preference, it is structural. A Newton step is
            |m - t| / ||grad m|| long, so it shrinks to nothing as the level
            set is approached, while the anchor pull stays at full strength —
            past some point the anchor simply wins. Measured on a trained
            CIFAR-10 ResNet18 (t = 0, 50 steps), |m - t| median by step:

                step      lambda=0.01     lambda=0
                   2           0.323        0.195
                   4           0.189       0.0069
                  59        0.203 (up)     0.0034

            Note the rise at the tail: the anchor does not merely slow the
            solver, it drags it back off the level set. The valid rate at
            tol=0.05 goes 0.228 -> 0.898 on that model, which is the
            reachability ceiling (0.914) — so with lambda=0 nothing is left on
            the table. Drift from the start is barely affected (0.520 vs
            0.453), i.e. at this strength the anchor was never constraining
            anything, only breaking convergence. Consistent with the module
            header: in 3072 dimensions the random starts are already far apart
            and do not collapse.

            "energy" is unaffected (0.094 vs 0.090 on the same model) because
            its gradient saturates to O(1) and stays comparable to the anchor.
            Two things also tried and rejected: projecting the anchor onto the
            level set's tangent space (0.634, and worse as lambda grows —
            the first-order projection breaks down over a long step), and a
            trust region (0.945 -> 0.659).
                       Keep small (~0.01-0.05); large values prevent the
                       perturbations from reaching the level set.
        alpha: Sharpness of the symmetric penalty. Used only by
            solver="energy"; "newton" has no such parameter.
        tol: A perturbation counts as valid when |m - t| <= tol
        norm: "linf" or "l2"
        delta0: Optional starting perturbations (B, N, C, H, W); projected onto
                the feasible set before use, and sampled when None

    Returns:
        LevelSolution — see that class. The two margin gradients are taken from
        the hard margin m = f_y - max_{j != y} f_j, so at an exact top-2 tie
        they are a subgradient (autograd picks whichever wrong class the max
        selected), not a true gradient. top2_margins reports the gap, which is
        how close to such a tie each position sits.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if step_size is None:
        step_size = _DEFAULT_STEP_SIZE[solver]
    if anchor_lambda is None:
        anchor_lambda = _DEFAULT_ANCHOR_LAMBDA[solver]

    was_training = model.training
    model.eval()

    if delta0 is None:
        delta0 = random_starts(x, epsilon, num_starts, norm=norm)
    else:
        # A caller-supplied start need not be feasible. Project it before it
        # becomes the anchor, or the penalty would pull every perturbation
        # toward a point outside the ball.
        delta0 = project_delta(delta0.detach(), x, epsilon, norm)
    anchor = delta0.detach()
    delta = anchor.clone()

    B, N = delta.shape[:2]
    x_rep = x.unsqueeze(1)
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

    # Best point seen so far, per (B, N). The loop runs num_steps + 1 times so
    # that the iterate produced by the final update is evaluated too, rather
    # than being returned unexamined.
    best_delta = delta.clone()
    best_gap = delta.new_full((B, N), float("inf"))

    for step in range(num_steps + 1):
        delta.requires_grad_(True)
        logits = model((x_rep + delta).reshape(B * N, *x.shape[1:]))
        margins = margin(logits, y_rep).view(B, N)

        with torch.no_grad():
            gap = (margins - t).abs()
            improved = gap < best_gap
            best_gap = torch.where(improved, gap, best_gap)
            best_delta = torch.where(improved.view(B, N, 1, 1, 1),
                                     delta.detach(), best_delta)

        if step == num_steps:
            break

        if solver == "newton":
            # Minimum-norm Newton step for the root problem m(delta) = t.
            # Linearising, m + grad_m . D = t has infinitely many solutions in
            # d dimensions for one equation; the least-norm one is
            #     D = -(m - t) * grad_m / ||grad_m||^2
            # which moves along the level set's normal by |m - t| / ||grad_m||,
            # the first-order distance to it. The sign of (m - t) makes this
            # two-sided on its own, which is why no symmetric penalty and no
            # alpha appear here.
            grad_m = torch.autograd.grad(margins.sum(), delta)[0]
            with torch.no_grad():
                gm_sq = grad_m.flatten(2).pow(2).sum(dim=-1).clamp_min(1e-12)
                newton = ((margins - t) / gm_sq).view(B, N, 1, 1, 1) * grad_m
                # Both terms are input-space displacements, so they add
                # directly and anchor_lambda is dimensionless here — unlike in
                # the energy branch, where the two gradients carry different
                # units and the useful lambda drifts with the model's scale.
                update = newton + anchor_lambda * (delta.detach() - anchor)
        else:
            energy = symmetric_softplus(margins - t, alpha=alpha)
            anchor_pen = (delta - anchor).flatten(2).pow(2).sum(dim=-1)
            objective = (energy + anchor_lambda * anchor_pen).sum()
            update = torch.autograd.grad(objective, delta)[0]

        delta = project_delta(delta.detach() - step_size * update, x, epsilon, norm)

    # One final pass at the points we are about to return. Both gradients come
    # off the same forward — two backward passes through one graph, rather than
    # two forwards.
    best_delta = best_delta.detach().requires_grad_(True)
    logits = model((x_rep + best_delta).reshape(B * N, *x.shape[1:]))
    margins = margin(logits, y_rep).view(B, N)
    ce = F.cross_entropy(logits, y_rep, reduction="sum")

    grad_margin = torch.autograd.grad(margins.sum(), best_delta, retain_graph=True)[0]
    grad_ce = torch.autograd.grad(ce, best_delta)[0]

    with torch.no_grad():
        margins = margins.detach()
        grad_margin_l2 = grad_margin.flatten(2).norm(p=2, dim=-1)
        grad_margin_dual = dual_norm(grad_margin, norm)
        grad_ce_dual = dual_norm(grad_ce, norm)
        valid_mask = (margins - t).abs() <= tol

    # Restore only once every forward pass is done, so none of them can touch
    # BatchNorm running statistics when called from a training loop.
    model.train(was_training)

    return LevelSolution(best_delta.detach(), margins, valid_mask,
                         grad_margin_l2, grad_margin_dual, grad_ce_dual)


# ------------------------------------------------------------------
#                       Position weighting
# ------------------------------------------------------------------

WEIGHT_MODES = ("uniform", "sharp", "flat", "min_norm", "max_norm")

GEOMETRY_MODES = ("coarea", "dual", "ce")

T_MODES = ("fixed", "adaptive", "reachable")

# sign-PGD steps used to estimate the reachable margin floor under
# t_mode="reachable". Five is where the accuracy/cost curve flattens — see
# reachable_margin_floor.
_REACHABLE_PGD_STEPS = 5

TOL_MODES = ("absolute", "relative")

# Default tolerance per mode. They are on different scales: "absolute" is a
# margin value, "relative" a fraction of the clean margin IQR.
_DEFAULT_TOL = {"absolute": 0.05, "relative": 0.02}


def pos_geo_weights(grad_score, delta_norm, valid_mask, mode="uniform", tau=1.0,
                    eps=1e-12):
    """
    Decide how much each landing point contributes to the training loss.

    The five modes are the experiment: they ask whether it matters *where* on
    the level set the training signal comes from.

        uniform   every valid position counts equally
        sharp     favour large grad_score — steep spots
        flat      favour small grad_score — wide-valley spots
        min_norm  all weight on the valid position closest to the clean image
        max_norm  all weight on the valid position furthest from it

    Perturbations that never reached the level set get zero weight in every
    mode: the premise is "different positions at the *same* level", so a point
    that is not on the level does not belong in the average. Rows where nothing
    is valid come back all-zero and contribute no gradient — watch
    valid_position_rate and valid_sample_rate in pos_geo_loss's info.

    What grad_score means is the caller's choice, and it decides what the
    sharp / flat modes are measuring. See GEOMETRY_MODES and pos_geo_loss:

      * flat + grad_margin_l2 + tau=1 — coarea-inspired inverse-gradient
        weighting. Under the Euclidean coarea formula the surface measure on
        {m = t} carries a 1/||grad m||_2 factor, which is exactly what this
        reproduces, given a uniform Q and an approximately surface-uniform
        proposal over the candidates.
      * flat + grad_margin_dual — Lp threat-aware first-order thickness: the
        dual norm is what converts a margin gap into a distance in the norm
        the perturbations actually live in.
      * flat + grad_ce_dual — a CE loss-flatness baseline. This is *not* a
        coarea weighting; CE is not the function whose level set is being
        traced, and the two gradients differ in direction as well as scale.

    On the theory, stated plainly: for a non-uniform perturbation distribution
    q, the strict PR weight is w(delta) ∝ q(delta) / ||grad_delta m(delta)||_2.
    This implementation compares only the gradient factor, i.e. it assumes the
    candidates share a common q / proposal density. It is therefore a
    PR-motivated *proxy*, not an unbiased PR estimator.

    Sharp and flat are softmaxes over log(grad_score) rather than over
    grad_score itself, which is what ties them to the theory:

        sharp:  w_n ∝ grad_score_n ** ( 1 / tau)
        flat:   w_n ∝ grad_score_n ** (-1 / tau)

    so tau=1 gives exactly w ∝ g and w ∝ 1/g. Two consequences worth knowing:
    the weights are invariant to rescaling every grad_score by a common c > 0
    (a softmax over raw g is not), and tau -> infinity tends to uniform over
    the valid positions.

    Args:
        grad_score: (B, N) non-negative, finite per-position score — one of the
                    gradient norms from LevelSolution. Only read by sharp/flat,
                    but validated in every mode.
        delta_norm: (B, N) distance of each perturbation from the clean image,
                    in whichever norm min_norm / max_norm should rank by
        valid_mask: (B, N) bool, which positions reached the level set
        mode: one of WEIGHT_MODES
        tau: temperature for sharp / flat; larger flattens toward uniform
        eps: floor for grad_score before the log, raised to the dtype's
             smallest positive value when that is larger

    Returns:
        weights: (B, N), each row summing to 1 over its valid entries, or all
                 zero for a row with no valid entry. Invalid positions are
                 always exactly zero.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got {mode!r}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be bool, got {valid_mask.dtype}")
    if not (grad_score.shape == delta_norm.shape == valid_mask.shape):
        raise ValueError(
            f"grad_score {tuple(grad_score.shape)}, delta_norm "
            f"{tuple(delta_norm.shape)} and valid_mask {tuple(valid_mask.shape)} "
            "must all be (B, N)"
        )
    if grad_score.dim() != 2:
        raise ValueError(f"expected (B, N) inputs, got {grad_score.dim()} dims")

    # One fused check, so this costs a single device sync per call rather than
    # one per condition.
    ok = (torch.isfinite(grad_score) & (grad_score >= 0) & torch.isfinite(delta_norm))
    if not bool(ok.all()):
        raise ValueError(
            "grad_score must be finite and non-negative and delta_norm finite; "
            f"got grad_score in [{grad_score.min()}, {grad_score.max()}]"
        )

    dtype = grad_score.dtype
    safe_eps = max(float(eps), torch.finfo(dtype).tiny)
    valid = valid_mask.to(dtype)
    has_valid = valid_mask.any(dim=1)

    if mode == "uniform":
        w = valid

    elif mode in ("sharp", "flat"):
        sign = 1.0 if mode == "sharp" else -1.0
        log_g = torch.log(grad_score.clamp_min(safe_eps))
        scores = sign * log_g / tau

        # Softmax only over rows that have something to normalise: a row masked
        # to all -inf would come back NaN.
        w = torch.zeros_like(grad_score)
        if bool(has_valid.any()):
            masked = scores.masked_fill(~valid_mask, float("-inf"))
            w[has_valid] = torch.softmax(masked[has_valid], dim=1).to(dtype)
        w = w * valid                      # exact zeros on invalid positions

    else:  # min_norm / max_norm
        # +/-inf on invalid entries keeps them from ever being selected. Ties
        # among valid entries fall to argmin/argmax's default, the lowest index.
        fill = float("inf") if mode == "min_norm" else float("-inf")
        scores = delta_norm.masked_fill(~valid_mask, fill)
        pick = scores.argmin(dim=1) if mode == "min_norm" else scores.argmax(dim=1)

        w = torch.zeros_like(grad_score)
        w.scatter_(1, pick.view(-1, 1), 1.0)
        w = w * valid                      # all-invalid rows -> all zero

    # Rows with no valid entry divide 0 by safe_eps and stay 0 rather than NaN.
    return w / w.sum(dim=1, keepdim=True).clamp_min(safe_eps)


# ------------------------------------------------------------------
#                     Level-set training loss
# ------------------------------------------------------------------

def pos_geo_loss(model, x, y, criterion, t, epsilon, *,
                 t_mode="fixed", t_quantile=0.5, t_frac=0.5, tol_mode="absolute",
                 weight_mode="uniform", geometry_mode="coarea", tau=1.0,
                 **solver_kwargs):
    """
    Weighted PGD-style adversarial training on margin level-set perturbations.

        loss = sum_r w_r * CE(f(x + delta_r), y)

    Both delta and w are detached, so the gradient flows only into the model —
    the positions and their weights are treated as fixed data for the step.

    One model mode throughout
    -------------------------
    The whole function runs under model.eval(), inner solver and outer forward
    alike, restored in a finally block. This is not a detail: the experiment
    compares *positions at the same margin level*, so the level has to mean the
    same thing when the solver validates it and when the outer loss consumes
    it. A train-mode outer forward normalises by batch statistics rather than
    running ones, which moves every margin — the solver would certify
    m = t under one function and the loss would train on another. That
    difference is confounded with position, which is the variable under study.

    eval() does not disable autograd. The outer forward still builds a graph,
    loss.backward() still reaches every parameter, and BatchNorm's affine
    weight/bias still receive gradients; only the running statistics are frozen
    and Dropout is off. Because nothing here updates those statistics, a
    training loop must update them itself — see the example below.

    Example::

        model.train()

        # Update BN running statistics exactly once, on the clean batch. Skip
        # this if the loop already does a train-mode clean forward.
        with torch.no_grad():
            _ = model(x)

        optimizer.zero_grad(set_to_none=True)
        loss, x_adv, info = pos_geo_loss(model, x, y, criterion, t, epsilon,
                                         weight_mode=weight_mode,
                                         geometry_mode=geometry_mode, tau=tau,
                                         **solver_kwargs)
        loss.backward()
        optimizer.step()

    What this loss can and cannot show
    ----------------------------------
    The outer loss is ordinary multiclass CE, which makes this a practical CE
    adversarial-training variant. Equal hard margin does *not* imply equal CE —
    the margin fixes one logit difference, CE depends on all of them — so this
    function alone cannot establish that position matters *at fixed loss*. It
    shows that position matters at a fixed margin level. A strict causal
    control needs either a hard-margin outer loss or a margin-matched *and*
    CE-matched design.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W)
        y: True labels (B,)
        criterion: Unused. Accepted so the signature matches the other AT
                   losses in src/adv_loss.py; the outer loss is always CE.
        t: Target margin level. Used as given when t_mode="fixed"; ignored
           when t_mode="adaptive".
        epsilon: Radius of the ball the perturbations live in
        t_mode: how the target level is chosen.
                "fixed": t is an absolute margin value, the same every batch.
                    This is the cleaner control — the level is literally the
                    same number throughout — but the margin distribution
                    drifts during training (CE pushes margins outward while t
                    stays put), so the valid rate decays and eventually the
                    level sits outside the epsilon-ball entirely. Pair it with
                    a warmup so the distribution settles first.
                "adaptive": t is the t_quantile quantile of the *clean* margin
                    distribution of this batch, recomputed every step. The
                    level follows the model, so the valid rate holds up during
                    training, and it is the more comparable choice across
                    methods whose margin scales differ — the same quantile
                    means the same thing, an absolute t does not. The cost is
                    that the level's absolute value moves, so margin1 is only
                    interpretable next to info["t_used"].
                "reachable": t interpolates between the sample's clean margin
                    and the lowest margin the epsilon-ball can reach for it,
                    t = start + t_frac * (floor - start) where
                    start = min(m_clean, 0) and the floor is estimated per
                    sample by reachable_margin_floor. Anchoring at m_clean is
                    what makes it attainable — the ball pushes the margin down
                    from where the sample already is, not from zero — and the
                    clamp at 0 is what keeps it an adversarial example, since
                    a sample the model already classifies correctly would
                    otherwise get a positive level.

                    The level is then capped at -tol so the whole tolerance
                    band stays below zero. A sample whose floor is above -tol
                    cannot be both adversarial and reachable, so its level
                    becomes unreachable and it drops out of valid_mask rather
                    than contributing a non-adversarial position. This trades
                    a little valid rate for the guarantee that every weighted
                    landing really is an adversarial example. This keeps the level
                    inside the ball as training moves the floor, which neither
                    of the other modes does: measured over six epochs the
                    median floor rose from -1.54 to -0.20, and a fixed t = -0.5
                    went from 96% valid to 15%. With t_frac = 0.5 the same runs
                    held 0.88-1.00. The level is negative whenever the floor
                    is, so the landings stay adversarial examples by
                    construction. Costs _REACHABLE_PGD_STEPS extra backward
                    passes per batch.
        t_quantile: quantile in [0, 1] for t_mode="adaptive"; 0.5 is the median
                    clean margin. Ignored by the other modes.
        t_frac: how far to push, for t_mode="reachable". 0 leaves the level at
                the clean margin, 1 puts it at the reachable floor, 0.5 halfway. Values near 1 sit on the edge of what is
                reachable and lose validity to estimation error. Ignored by the
                other modes.
        tol_mode: how the validity tolerance is interpreted. Pass the value
                itself as solver_kwargs["tol"], or omit it for the mode's
                default (0.05 absolute, 0.02 relative).
                "absolute" (default): tol is a margin value, fixed for the run.
                    Simple, but it silently tightens as training widens the
                    margin distribution — measured on a newton run, the clean
                    margin IQR went 0.42 -> 24.46 over six epochs, so a fixed
                    tol=0.05 went from 8% of the spread to 0.2% and the valid
                    rate fell 0.98 -> 0.21 for that reason alone.
                "relative": tol is a fraction of the clean margin IQR of this
                    batch, so it tracks the distribution. Same two models:
                    0.88 vs 0.80 at tol=0.02, against 0.98 vs 0.21 absolute.
                    IQR rather than std because the distribution is wide and
                    skewed and a few extreme margins should not set the
                    tolerance. Normalising by ||grad m|| instead (a first-order
                    input-space distance) reads as more principled but measured
                    worse — 0.72 vs 0.40 — because ||grad m|| drifts on its own
                    schedule.
        weight_mode: one of WEIGHT_MODES — see pos_geo_weights
        geometry_mode: which gradient the sharp / flat modes rank by —
                       "coarea": grad_margin_l2, the hard-margin L2 gradient
                           and the standard Euclidean coarea factor;
                       "dual": grad_margin_dual, the hard-margin dual norm and
                           the Lp threat-aware local thickness;
                       "ce": grad_ce_dual, the CE input-gradient dual norm,
                           a CE loss-geometry baseline only.
                       Ignored by the other weight modes, which never read it.
        tau: temperature for the sharp / flat modes
        **solver_kwargs: Forwarded to solve_level_perturbations
                         (num_starts, num_steps, step_size, anchor_lambda,
                          alpha, tol, norm, delta0)

    Returns:
        loss: Scalar training loss, the plain batch mean. Samples with no
              valid position have all-zero weights, so they contribute exactly
              0 and no gradient — the mean is diluted by them but never
              polluted, and the gradient scale stays independent of the valid
              rate. When the whole batch has none the loss is a zero still
              attached to the graph.

              Dividing by the valid count instead is tempting, because it
              stops a falling valid rate from shrinking the reported number.
              It also ties the gradient scale to 1 / valid_sample_rate, and
              that closes a feedback loop: valid rate falls, gradients grow,
              margins blow up, valid rate falls further. Measured on this
              codebase it took |margin| from 0.36 to 218 over 30 steps. The
              undiluted number is still worth having, so it is reported as
              info["loss_valid_only"] rather than optimised.
        x_adv: (B, N, C, H, W) perturbed inputs
        info: diagnostics, every tensor detached. Per position (B, N) unless
              noted —
                margin1 / margin2 (top-2 wrong classes), j1 / j2 (their
                indices), ce, weights, valid_mask, grad_ce_dual /
                grad_margin_l2 / grad_margin_dual (see LevelSolution),
                delta_norm (threat norm, what min_norm / max_norm rank by),
                delta_l2_norm (always Euclidean, comparable across threat
                models);
              consistency between solver and outer forward —
                solver_margins, outer_margins, level_mode_drift and its mean
                level_mode_drift_mean, outer_level_gap, outer_valid_mask,
                outer_valid_position_rate. Sharing one model mode should hold
                level_mode_drift at numerical zero; a non-zero mean means
                something reintroduced a mode or state difference between the
                two forwards. The weighting deliberately still uses the
                solver's valid_mask — the outer mask is a diagnostic only;
              per sample (B,) has_valid; scalars valid_position_rate and
              valid_sample_rate; ae_rate and ae_rate_valid, the fraction of
              landings that are genuine adversarial examples (m < 0) over all
              positions and over the valid ones — the premise of the experiment
              if the positions are meant to be AEs, and not guaranteed by
              validity alone; t_used and tol_used, the values actually
              applied this batch — under the adaptive/relative modes these move
              every batch, and margin1 cannot be read without them; the strings
              geometry_mode, t_mode, tol_mode and solver;
              and logits_adv
              (B * N, num_classes) so the caller can compute accuracy without
              a second forward pass.

              Watch valid_sample_rate while training: samples with no valid
              position produce no gradient at all, so if it stays low, adjust
              num_steps, step_size, anchor_lambda, tol or num_starts.
    """
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(
            f"geometry_mode must be one of {GEOMETRY_MODES}, got {geometry_mode!r}"
        )
    if t_mode not in T_MODES:
        raise ValueError(f"t_mode must be one of {T_MODES}, got {t_mode!r}")
    if not 0.0 <= t_quantile <= 1.0:
        raise ValueError(f"t_quantile must be in [0, 1], got {t_quantile}")
    if not 0.0 < t_frac <= 1.0:
        raise ValueError(f"t_frac must be in (0, 1], got {t_frac}")
    if tol_mode not in TOL_MODES:
        raise ValueError(f"tol_mode must be one of {TOL_MODES}, got {tol_mode!r}")

    norm = solver_kwargs.get("norm", "linf").lower()
    solver = solver_kwargs.get("solver", "energy")
    tol = solver_kwargs.pop("tol", None)
    if tol is None:
        tol = _DEFAULT_TOL[tol_mode]

    was_training = model.training
    try:
        # One mode for both forwards. solve_level_perturbations saves and
        # restores the mode itself, but since it is entered from eval it
        # returns to eval, so the outer forward below sees the same model the
        # solver certified against.
        model.eval()

        # Resolve the target level and the tolerance *inside* eval mode, so
        # they, the solver and the outer forward are all defined against the
        # same model state. Letting the caller compute them is what
        # reintroduces mode mismatches. One clean forward serves both.
        if t_mode in ("adaptive", "reachable") or tol_mode == "relative":
            with torch.no_grad():
                clean_margins = margin(model(x), y)

            if t_mode == "adaptive":
                t = clean_margins.quantile(t_quantile).item()

            if tol_mode == "relative":
                # IQR, not std: the margin distribution is wide and skewed, and
                # a few extreme margins should not set the tolerance.
                iqr = (clean_margins.quantile(0.75)
                       - clean_margins.quantile(0.25)).item()
                tol = tol * max(iqr, 1e-12)

            # After the tolerance is resolved: the level is capped relative to
            # tol below, so it has to see the final value.
            if t_mode == "reachable":
                # Interpolate between where the sample already sits and the
                # deepest the ball can push it. Two clamps, each fixing a
                # failure that only shows up at the edges:
                #
                #   start = min(m_clean, 0)
                #     Anchoring at m_clean is what makes the level attainable —
                #     the ball pushes the margin down from where the sample
                #     already is, not from zero, so t_frac * floor can land
                #     *above* the starting point (m_clean = -0.40 with a floor
                #     of -0.41 gives -0.20) and is then unreachable. Clamping
                #     the start at 0 is what keeps the level adversarial: for a
                #     sample the model already gets right, interpolating from a
                #     positive m_clean gives a positive level, which is not an
                #     AE at all (measured, ae_rate_valid 0.70-0.87).
                #
                #   level <= -tol
                #     The tolerance is two-sided, so a level of -0.005 admits
                #     landings up to +0.045. Leaving a tol of headroom keeps the
                #     whole band below zero (measured, this closed the last gap
                #     from 0.86-0.94 to exactly 1.0). Samples whose floor sits
                #     above -tol cannot be both adversarial and reachable; their
                #     level becomes unreachable and they drop out of valid_mask,
                #     which is the right outcome — they have no usable position.
                #
                # Per sample, not per batch: how far the margin can be driven
                # varies a lot between images, and one shared level leaves the
                # hard ones outside the ball. Shaped (B, 1) to broadcast against
                # the (B, N) margins.
                floor = reachable_margin_floor(model, x, y, epsilon, norm=norm,
                                               num_steps=_REACHABLE_PGD_STEPS)
                start = clean_margins.clamp(max=0.0)
                level = start + t_frac * (floor - start)
                t = level.clamp(max=-tol).view(-1, 1)

        t_used = float(t.median()) if torch.is_tensor(t) else float(t)
        tol_used = float(tol)

        sol = solve_level_perturbations(model, x, y, t, epsilon, tol=tol,
                                        **solver_kwargs)
        delta, valid_mask = sol.delta, sol.valid_mask

        grad_score = {
            "coarea": sol.grad_margin_l2,
            "dual": sol.grad_margin_dual,
            "ce": sol.grad_ce_dual,
        }[geometry_mode]

        B, N = delta.shape[:2]
        flat_delta = delta.flatten(2)
        # Distance in the norm the perturbations are constrained by — that is
        # what "closest to / furthest from the clean image" means under this
        # threat model, and what min_norm / max_norm therefore rank by.
        if norm == "linf":
            delta_norm = flat_delta.abs().amax(dim=-1)
        else:
            delta_norm = flat_delta.norm(p=2, dim=-1)
        # Euclidean companion, kept separately because it stays comparable
        # across threat models where delta_norm does not.
        delta_l2_norm = flat_delta.norm(p=2, dim=-1)

        weights = pos_geo_weights(grad_score, delta_norm, valid_mask,
                                  mode=weight_mode, tau=tau).detach()

        x_adv = (x.unsqueeze(1) + delta).clamp(0.0, 1.0)
        y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

        # Still eval, and deliberately not under no_grad: the CE outer loss has
        # to reach the model parameters.
        logits_adv = model(x_adv.reshape(B * N, *x.shape[1:]))
        ce = F.cross_entropy(logits_adv, y_rep, reduction="none").view(B, N)

        # Rows with no valid position have all-zero weights and contribute
        # exactly 0, so a plain batch mean is never polluted by them — only
        # diluted. Optimise the batch mean and report the valid-only average
        # separately: dividing by the valid count instead would tie the
        # gradient scale to 1 / valid_sample_rate, and that feedback loop
        # (valid down -> gradient up -> margins blow up -> valid down) is what
        # made this diverge. See the Returns note.
        per_sample_loss = (weights * ce).sum(dim=1)
        has_valid = valid_mask.any(dim=1)
        valid_count = has_valid.sum().to(per_sample_loss.dtype)
        loss = per_sample_loss.mean()
        loss_valid_only = per_sample_loss.sum() / valid_count.clamp_min(1.0)

        with torch.no_grad():
            m1, m2, j1, j2 = top2_margins(logits_adv, y_rep)
            outer_margins = m1.view(B, N)
            solver_margins = sol.margins
            level_mode_drift = (outer_margins - solver_margins).abs()
            outer_level_gap = (outer_margins - t).abs()
            outer_valid_mask = outer_level_gap <= tol

            # Is a landing point actually an adversarial example? m < 0 means
            # the true class lost, so only then. This is not implied by being
            # on the level set — with t = 0 roughly half the landings come out
            # on the correct side (measured 0.46-0.56), because the tolerance
            # is two-sided. What decides it in practice is the sign of t: the
            # landings cluster near t rather than filling the band, so t = -0.3
            # with tol = 0.4 still gave every landing m < 0. t < 0 is the thing
            # to get right; t <= -tol is merely the conservative version.
            is_ae = solver_margins < 0
            ae_rate = is_ae.float().mean()
            n_valid_pos = valid_mask.sum()
            ae_rate_valid = (torch.where(valid_mask, is_ae, False).sum()
                             / n_valid_pos.clamp_min(1)).to(ae_rate.dtype)

            info = {
                "margin1": outer_margins,
                "margin2": m2.view(B, N),
                "j1": j1.view(B, N),
                "j2": j2.view(B, N),
                "ce": ce.detach(),
                "grad_ce_dual": sol.grad_ce_dual.detach(),
                "grad_margin_l2": sol.grad_margin_l2.detach(),
                "grad_margin_dual": sol.grad_margin_dual.detach(),
                "geometry_mode": geometry_mode,
                "t_mode": t_mode,
                "t_used": t_used,   # median when t is per-sample
                "tol_mode": tol_mode,
                "tol_used": tol_used,
                "solver": solver,
                # Optimised value vs. undiluted value. Their ratio is the
                # amplification a valid-only objective would have applied.
                "loss_batch_mean": loss.detach(),
                "loss_valid_only": loss_valid_only.detach(),
                "delta_norm": delta_norm.detach(),
                "delta_l2_norm": delta_l2_norm.detach(),
                "weights": weights,
                "valid_mask": valid_mask,
                "has_valid": has_valid,
                "valid_position_rate": valid_mask.float().mean(),
                "valid_sample_rate": has_valid.float().mean(),
                # ae_rate_valid is the one that matters: it covers exactly the
                # positions that carry weight in the loss.
                "ae_rate": ae_rate,
                "ae_rate_valid": ae_rate_valid,
                # Solver / outer agreement — see the Returns note.
                "solver_margins": solver_margins.detach(),
                "outer_margins": outer_margins,
                "level_mode_drift": level_mode_drift,
                "level_mode_drift_mean": level_mode_drift.mean(),
                "outer_level_gap": outer_level_gap,
                "outer_valid_mask": outer_valid_mask,
                "outer_valid_position_rate": outer_valid_mask.float().mean(),
                # Detached so holding info cannot pin the backward graph.
                "logits_adv": logits_adv.detach(),
            }
    finally:
        model.train(was_training)

    return loss, x_adv, info


# ------------------------------------------------------------------
#                       Position geometry
# ------------------------------------------------------------------

def level_geometry(delta, margins, valid_mask=None, eps=1e-12):
    """
    Describe where the perturbations landed, relative to each other.

    Pure tensor arithmetic — no model, no forward pass — so it is cheap enough
    to call every step and can be unit-tested on synthetic tensors.

    Only candidates that are *on* the level set take part. The premise is
    "different positions at the same level", so a perturbation that never
    reached the level contributes no direction here; pass the solver's
    valid_mask to enforce that. A zero perturbation is excluded too — it has a
    norm but no direction, and normalising it would manufacture an arbitrary
    one.

    Everything is computed in float32. Under AMP the incoming delta may be
    float16, and a Gram matrix accumulated in half precision carries enough
    error to visibly distort the spectrum downstream.

    Args:
        delta: Perturbations (B, N, C, H, W)
        margins: Margins at those perturbations (B, N)
        valid_mask: (B, N) bool, |margin - t| <= tol. None treats every
                    candidate as valid.
        eps: Norm below which a perturbation counts as zero

    Returns:
        dict, all detached, with
            delta_l2_norm: (B, N)    Euclidean norm of each perturbation.
                           Named in full because delta_norm elsewhere in this
                           file means the *threat* norm, which is L-inf under
                           norm="linf" and a different quantity.
            pair_cos:      (B, N, N) pairwise cosines, symmetrised and clamped
                           to [-1, 1]. Inactive candidates occupy zero rows and
                           columns.
            margins:       (B, N)    passed through, so callers log one object
            active_mask:   (B, N)    valid and non-zero, i.e. what was counted
            active_count:  (B,)      how many that was
    """
    flat = delta.detach().flatten(2).float()
    delta_l2_norm = flat.norm(p=2, dim=-1)

    if valid_mask is None:
        valid_mask = torch.ones_like(delta_l2_norm, dtype=torch.bool)
    nonzero_mask = delta_l2_norm > eps
    active_mask = valid_mask & nonzero_mask

    # Zero out inactive rows *after* normalising, so they become zero vectors
    # rather than arbitrary unit ones. Their Gram rows and columns are then
    # identically zero.
    unit = flat / delta_l2_norm.unsqueeze(-1).clamp_min(eps)
    unit = unit * active_mask.unsqueeze(-1).to(unit.dtype)

    pair_cos = torch.bmm(unit, unit.transpose(1, 2))
    # bmm is not exactly symmetric in floating point, and eigvalsh reads only
    # one triangle — symmetrise so the spectrum cannot depend on which.
    pair_cos = 0.5 * (pair_cos + pair_cos.transpose(1, 2))
    pair_cos = pair_cos.clamp(-1.0, 1.0)

    return {
        "delta_l2_norm": delta_l2_norm,
        "pair_cos": pair_cos,
        "margins": margins.detach(),
        "active_mask": active_mask,
        "active_count": active_mask.sum(dim=1),
    }


def pair_l2_distance(delta_l2_norm, pair_cos):
    """
    Pairwise Euclidean distances between perturbations, from norms and cosines.

        d_ij^2 = |d_i|^2 + |d_j|^2 - 2 |d_i| |d_j| cos_ij

    This measures *positional* spread and is deliberately separate from
    dispersion's eff_rank, which measures directional spread. The two answer
    different questions and can disagree: perturbations at +v and -v are far
    apart in distance while spanning a single direction.

    Entries involving an inactive candidate are meaningless — its cosine row is
    zero, so the formula degenerates to sqrt(|d_i|^2 + |d_j|^2). Mask them with
    active_mask before reducing.

    Args:
        delta_l2_norm: (B, N) from level_geometry
        pair_cos: (B, N, N) from level_geometry

    Returns:
        (B, N, N) distances, zero on the diagonal
    """
    n_i = delta_l2_norm.unsqueeze(2)
    n_j = delta_l2_norm.unsqueeze(1)
    sq = n_i.pow(2) + n_j.pow(2) - 2.0 * n_i * n_j * pair_cos
    return sq.clamp_min(0.0).sqrt()


def dispersion(pair_cos, active_mask=None, eps=1e-12):
    """
    How many independent *directions* the perturbations span, from the spectrum
    of their pairwise cosine matrix.

    That matrix is the Gram matrix of the unit perturbations, so it is
    symmetric PSD and eigvalsh gives its spectrum directly — cheaper and better
    conditioned than an SVD. These are eigenvalues, not singular values, which
    is why they are reported as eig_max / eig_min. For a PSD Gram matrix the
    singular values are their square roots, returned as sigma_max / sigma_min
    for callers that want them.

    The headline number is the participation ratio

        eff_rank = (sum lambda)^2 / sum lambda^2      in [1, active_count]

    computed from eigenvalues, not from their square roots. It reads as the
    effective directional dimensionality:

      * all directions collinear   -> eff_rank = 1, anisotropy = 1
      * orthogonal, equal energy   -> eff_rank = active_count,
                                      anisotropy = 1 / active_count

    Two cautions about what this is not. It measures *direction*, not position:
    perturbations at +v and -v are antipodal but span one dimension, so they
    give eff_rank = 1 — use pair_l2_distance for positional spread. And
    eig_min is not a condition measure for the active subspace: inactive
    candidates sit as zero rows and columns, so a single inactive candidate
    already forces the full matrix's smallest eigenvalue to zero. For the
    active subspace's smallest eigenvalue, extract each sample's active
    submatrix and decompose that separately.

    Args:
        pair_cos: (B, N, N) from level_geometry
        active_mask: (B, N) bool from level_geometry. None treats every
                     candidate as active.
        eps: Floor for the denominators

    Returns:
        dict of (B,) tensors: eff_rank, eig_max, eig_min, sigma_max, sigma_min,
        anisotropy, active_count. Samples with no active candidate get exact
        zeros throughout, rather than the arbitrary small numbers a clamped
        denominator would produce.
    """
    pair_cos = pair_cos.detach().float()
    B, N = pair_cos.shape[:2]

    if active_mask is None:
        active_count = torch.full((B,), float(N), device=pair_cos.device)
        has_active = torch.ones(B, dtype=torch.bool, device=pair_cos.device)
    else:
        active_count = active_mask.sum(dim=1).to(pair_cos.dtype)
        has_active = active_count > 0

    lam = torch.linalg.eigvalsh(pair_cos).clamp_min(0.0)     # ascending
    total = lam.sum(dim=-1)
    sq_total = lam.pow(2).sum(dim=-1)

    eff_rank = total.pow(2) / sq_total.clamp_min(eps)
    eig_max = lam[:, -1]
    eig_min = lam[:, 0]
    anisotropy = eig_max / total.clamp_min(eps)

    # A row with nothing active has an all-zero Gram matrix, whose spectrum is
    # all zeros — every ratio above is 0/eps. Say zero explicitly instead of
    # letting the clamp decide.
    keep = has_active.to(pair_cos.dtype)
    eff_rank = eff_rank * keep
    eig_max = eig_max * keep
    eig_min = eig_min * keep
    anisotropy = anisotropy * keep

    return {
        "eff_rank": eff_rank,
        "eig_max": eig_max,
        "eig_min": eig_min,
        "sigma_max": eig_max.sqrt(),
        "sigma_min": eig_min.sqrt(),
        "anisotropy": anisotropy,
        "active_count": active_count,
    }
