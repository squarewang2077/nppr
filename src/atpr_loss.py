# atpr_loss.py - AT-PR: adversarial training for probabilistic robustness
#
# Requirements:
#   torch >= 2.0
#
# Zhang et al., "Adversarial Training for Probabilistic Robustness", ICCV 2025
#   https://openreview.net/forum?id=eFUSbP7YQa
#   https://github.com/wellzline/AT-PR
#
# The idea in one paragraph:
#   Standard AT targets worst-case robustness — the inner maximisation looks
#   for the single highest-loss point in the epsilon-ball. AT-PR instead
#   targets *probabilistic* robustness, the fraction of the ball that is
#   misclassified, so its inner problem looks for an AE representative of a
#   *large* misclassified region rather than a sharp loss spike. It does so by
#     1. generating K candidate AEs with PGD under jittered hyper-parameters
#        (candidate 0 uses the nominal ones);
#     2. walking each candidate *back* towards the correct class along
#        -sign(grad CE) until the batch is mostly correct again, recording how
#        far it travelled — a proxy for how deep inside the misclassified
#        region it sat;
#     3. keeping, per sample, the candidate with the LARGEST such distance;
#     4. training on cross-entropy at those selected AEs.
#
# Conventions shared with src/adv_loss.py:
#   * inputs x are raw images in [0, 1]; normalisation lives inside the model
#   * attacks  `*_attack(model, x, y, ...)`  -> x_adv
#   * losses   `*_loss(model, x, y, ...)`    -> (loss, x_adv)
#   * everything is device-agnostic — no .cuda() calls
#
# Deviations, all deliberate:
#   1. The inner loop does NOT run under model.eval(), unlike the rest of the
#      family. The AT-PR reference does not switch modes for its own attack
#      (though it does for TRADES/MART), so `attack_in_eval_mode` defaults to
#      False to preserve the published numerics. Set it True for the family
#      convention.
#   2. The reference `pick_best_ae` calls `loss.backward()` on the *model*,
#      leaving stale gradients in `param.grad`; the caller in `main.py` then
#      accumulates the training loss on top, contaminating the update. Every
#      inner step here uses `torch.autograd.grad(..., inputs=x)`, so model
#      gradients are never touched. Reproducing the published numbers
#      bit-for-bit would mean reintroducing that bug.
#   3. The reference refinement loop `while is_ae.sum() > 0.1 * B` has no
#      iteration cap and can spin forever on a weak model; `max_refine_steps`
#      bounds it (default 100).
#   4. The reference `PR()` calls `pgd_loss(...)` without the required
#      `optimizer` argument, so `--attack PR` raises TypeError as published.
#      No optimizer is needed here.
#   5. `final_adv_example = adv_list[0]` in the reference aliases the first
#      candidate and mutates it in place; here it is cloned.
#
# Everything else — the jitter ranges, the 10% stopping rule, the L-inf
# distance, the argmax-per-sample selection — follows the reference exactly.
#
# The probabilistic-robustness *metric* lives in utils/evaluator.py
# (Evaluator.evaluate_pr_random + prob_accuracy), not here, so PR is measured
# in one place across the repo.

import contextlib
import random

import torch
import torch.nn.functional as F

from .adv_loss import pgd_attack


# ------------------------------------------------------------------
#                        Shared helpers
# ------------------------------------------------------------------

@contextlib.contextmanager
def _maybe_eval(model, use_eval):
    """Temporarily switch to eval() so BN running stats are not updated by the
    attack's forward passes. See deviation 1 in the header for why this is
    off by default."""
    if not use_eval:
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _input_grad(model, x, y):
    """d CE(model(x), y) / dx, without touching model parameter gradients."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(x), y, reduction="mean")
    return torch.autograd.grad(loss, x)[0].detach()


def _distance_to_boundary(model, x_adv, y, refine_step_size, ae_ratio=0.1,
                          max_refine_steps=100, distance_norm="linf"):
    """
    Walk an AE back towards the true class and measure how far it travelled.

    Descends the CE loss from `x_adv` until at most `ae_ratio` of the batch is
    still misclassified, then reports the per-sample distance covered. A LARGE
    distance means the AE sat deep inside a misclassified region — which under
    the AT-PR objective makes it a better representative of that region than a
    high-loss point sitting right on the boundary.

    Args:
        model: Target model
        x_adv: Candidate adversarial examples (B, C, H, W)
        y: True labels (B,)
        refine_step_size: Step size for the backwards walk
        ae_ratio: Stop once <= this fraction of the batch is still an AE
        max_refine_steps: Iteration cap (the reference has none)
        distance_norm: "linf" or "l2"

    Returns:
        distance: (B,) distance travelled, detached
    """
    distance_norm = distance_norm.lower()
    if distance_norm == "linf":
        p = float("inf")
    elif distance_norm == "l2":
        p = 2.0
    else:
        raise ValueError(f"Unsupported norm: {distance_norm}")

    batch = y.size(0)
    x_curr = x_adv.detach().clone()

    for _ in range(max_refine_steps):
        with torch.no_grad():
            is_ae = model(x_curr).argmax(dim=1) != y
        if is_ae.sum().item() <= batch * ae_ratio:
            break
        grad = _input_grad(model, x_curr, y)

        # only the still-misclassified rows keep moving
        x_next = x_curr.clone()
        x_next[is_ae] = torch.clamp(
            x_curr[is_ae] - refine_step_size * grad[is_ae].sign(), 0.0, 1.0
        )
        x_curr = x_next.detach()

    distance = torch.norm((x_adv - x_curr).reshape(x_adv.size(0), -1), dim=1, p=p)
    return distance.detach()


# ------------------------------------------------------------------
#                          AT-PR attack
# ------------------------------------------------------------------

def atpr_attack(model, x, y, epsilon, alpha, num_steps, num_candidates=10,
                eps_jitter=0.02, step_jitter=0.003, steps_jitter=(-2, 10),
                ae_ratio=0.1, max_refine_steps=100, distance_norm="linf",
                attack_in_eval_mode=False, rng=None):
    """
    AT-PR inner maximisation (inner loop).

    Generates K PGD candidates under jittered hyper-parameters and keeps, per
    sample, the one sitting deepest inside a misclassified region as measured
    by _distance_to_boundary.

    Args:
        model: Target model
        x: Clean inputs (B, C, H, W) — expected in [0, 1]
        y: True labels (B,)
        epsilon, alpha, num_steps: Nominal PGD parameters, used verbatim for
            candidate 0
        num_candidates: K, total candidates per batch (reference: 10)
        eps_jitter: Candidates 1..K-1 draw eps ~ U(epsilon - eps_jitter, epsilon)
        step_jitter: alpha ~ U(alpha - step_jitter, alpha + step_jitter)
        steps_jitter: steps ~ randint(num_steps + lo, num_steps + hi)
        ae_ratio: Stop the refinement once <= this fraction is still an AE
        max_refine_steps: Iteration cap on the refinement walk
        distance_norm: "linf" (as in the released `attack_algorithms.PR`) or
            "l2" (as in the alternative `attack/pr.py`)
        attack_in_eval_mode: See deviation 1 in the header. False reproduces
            the reference.
        rng: Optional random module/Random instance, for reproducible jitter

    Returns:
        x_adv: Selected adversarial examples (B, C, H, W)

    Cost warning: this is ~K PGD attacks plus up to K * max_refine_steps extra
    forward/backward passes per batch. With the defaults it is roughly 15-25x
    the cost of PGD-10 AT. Reduce `num_candidates` first if that is too slow.
    """
    rng = rng or random
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")

    with _maybe_eval(model, attack_in_eval_mode):
        # Candidate pool: index 0 nominal, the rest jittered.
        candidates = [pgd_attack(model, x, y, epsilon, alpha, num_steps)]
        while len(candidates) < num_candidates:
            eps_k = rng.uniform(epsilon - eps_jitter, epsilon)
            alpha_k = rng.uniform(alpha - step_jitter, alpha + step_jitter)
            steps_k = rng.randint(num_steps + steps_jitter[0],
                                  num_steps + steps_jitter[1])
            candidates.append(
                pgd_attack(model, x, y, max(eps_k, 0.0), max(alpha_k, 0.0),
                           max(steps_k, 1))
            )

        # Per-sample argmax over distance-to-boundary.
        best = candidates[0].clone()
        best_distance = torch.zeros(y.size(0), device=x.device, dtype=x.dtype)
        for x_cand in candidates:
            distance = _distance_to_boundary(
                model, x_cand, y, refine_step_size=alpha, ae_ratio=ae_ratio,
                max_refine_steps=max_refine_steps, distance_norm=distance_norm,
            )
            better = distance > best_distance
            best[better] = x_cand[better]
            best_distance[better] = distance[better]

    return best.detach()


# ------------------------------------------------------------------
#                           AT-PR loss
# ------------------------------------------------------------------

def atpr_loss(model, x, y, epsilon, alpha, num_steps, criterion,
              num_candidates=10, **kwargs):
    """
    AT-PR loss (outer loop): Train on the AEs that best represent a large
    misclassified region, rather than on the single worst-case point.

    Zhang et al., "Adversarial Training for Probabilistic Robustness", ICCV 2025

    Args:
        model: Target model
        x: Clean inputs
        y: True labels
        epsilon, alpha, num_steps: Nominal PGD parameters for the inner loop
        criterion: Loss function (e.g., CrossEntropyLoss)
        num_candidates: K candidates per batch
        **kwargs: Forwarded to atpr_attack (eps_jitter, ae_ratio,
            max_refine_steps, distance_norm, attack_in_eval_mode, rng, ...)

    Returns:
        loss: Adversarial loss
        x_adv: Generated adversarial examples
    """
    x_adv = atpr_attack(model, x, y, epsilon, alpha, num_steps,
                        num_candidates=num_candidates, **kwargs)
    loss = criterion(model(x_adv), y)

    return loss, x_adv
