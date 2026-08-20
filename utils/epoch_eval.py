# utils/epoch_eval.py - Shared per-epoch evaluation suite for training scripts.
#
# Requirements:
#   torch >= 2.0
#
# One pass over a loader produces every requested metric at once:
#   clean accuracy/loss, PGD and FGSM adversarial accuracy, margin level-set
#   PR, and random-noise PR baselines.
# AutoAttack is separate (evaluate_aa) because it needs its own loop.
#
# Every evaluation is opt-in: pass None for the configs you do not want and
# the corresponding return value is None. Sharing one pass over the data is
# the point — the alternative is re-loading the test set once per metric.
#
# Used by scripts/train_classifiers.py, scripts/eval_all.py and the level-set
# trainer. Extracted here so those callers cannot drift apart; note that
# set_seed deliberately stays in each script, since the trainers disagree on
# the cudnn determinism/benchmark trade-off.

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.adv_loss import fgsm_attack, pgd_attack
from src.pos_geo_loss import solve_level_perturbations
from utils.utils import pr_random_generator


def level_generator(model, x, y, **kwargs):
    """
    Perturbation generator for the evaluation interface.

    Thin wrapper over solve_level_perturbations so the evaluator can treat
    level-set perturbations like any other PR generator.

    Returns:
        x_adv: perturbed inputs, shape (B, N, C, H, W)
    """
    sol = solve_level_perturbations(
        model, x, y,
        t=kwargs.get("t", 0.0),
        epsilon=kwargs.get("epsilon", 8 / 255),
        num_starts=kwargs.get("num_starts", 8),
        num_steps=kwargs.get("num_steps", 50),
        step_size=kwargs.get("step_size", 1e-2),
        anchor_lambda=kwargs.get("anchor_lambda", 0.02),
        alpha=kwargs.get("psi_alpha", 10.0),
        tol=kwargs.get("tol", 0.05),
        norm=kwargs.get("norm", "linf"),
    )
    return (x.unsqueeze(1) + sol.delta).clamp(0.0, 1.0)

def evaluate_per_epoch(
    model, loader, device, criterion,
    pgd_cfg=None, fgsm_cfg=None, level_cfg=None, random_cfgs=None,
    eval_name="eval",
):
    """Single-pass eval over loader. Each extra metric is gated by its config
    being non-None (and non-empty for random_cfgs); disabled
    blocks are skipped entirely so the cost goes to zero.

    `random_cfgs` is a *list* of pr_random_generator kwargs (one entry per
    distribution to evaluate). Result includes one entry per distribution
    in `random_pr_breakdown`.

    Always returns: clean_acc, clean_loss, num_samples.
    Conditionally returns (None when disabled):
        pgd_acc, fgsm_acc, level_pr, random_pr_breakdown.
    """
    do_pgd    = pgd_cfg is not None
    do_fgsm   = fgsm_cfg is not None
    do_level = level_cfg is not None
    do_random = bool(random_cfgs)

    model.eval()
    n_total = 0
    n_clean_correct = 0
    clean_loss_sum  = 0.0
    # Initialise unconditionally; the do_* flags gate accumulation and the
    # return value below — keeping these as plain 0 / 0.0 makes them easy to
    # type-check (no Optional arithmetic).
    n_pgd_correct = 0
    n_fgsm_correct = 0
    sum_level_pr = 0.0
    random_sum = {cfg["noise_dist"]: 0.0 for cfg in (random_cfgs or [])}

    pbar = tqdm(loader, desc=eval_name, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = y.size(0)
        n_total += B

        # 1) Clean — always
        with torch.no_grad():
            clean_logits = model(x)
            n_clean_correct += (clean_logits.argmax(dim=1) == y).sum().item()
            if criterion is not None:
                clean_loss_sum += criterion(clean_logits, y).item() * B

        # 2) PGD — uses autograd.grad on inputs; do NOT wrap in no_grad.
        if do_pgd:
            x_pgd = pgd_attack(model, x, y, **pgd_cfg)
            with torch.no_grad():
                n_pgd_correct += (model(x_pgd).argmax(dim=1) == y).sum().item()

        # 3) FGSM — single-step, so it bounds PGD from above; a large gap
        #    between the two is the usual sign of gradient masking.
        if do_fgsm:
            x_fgsm = fgsm_attack(model, x, y, **fgsm_cfg)
            with torch.no_grad():
                n_fgsm_correct += (model(x_fgsm).argmax(dim=1) == y).sum().item()

        # 4) Level mean PR
        if do_level:
            x_level = level_generator(model, x, y, **level_cfg)  # (B, N, C, H, W)
            N_loc = x_level.shape[1]
            with torch.no_grad():
                preds = model(x_level.reshape(B * N_loc, *x_level.shape[2:])
                              ).argmax(dim=1).view(B, N_loc)
                sum_level_pr += (preds == y.unsqueeze(1)).float().mean(dim=1).sum().item()

        # 5) Random-PR baseline — one pass per distribution
        if do_random:
            for cfg in random_cfgs:
                x_rand, _ = pr_random_generator(model, x, y, **cfg)
                N_rnd = x_rand.shape[1]
                with torch.no_grad():
                    preds = model(x_rand.reshape(B * N_rnd, *x_rand.shape[2:])
                                  ).argmax(dim=1).view(B, N_rnd)
                    random_sum[cfg["noise_dist"]] += (
                        preds == y.unsqueeze(1)
                    ).float().mean(dim=1).sum().item()

        # tqdm postfix — only show enabled metrics
        post = {"clean": f"{n_clean_correct/n_total:.3f}"}
        if do_pgd:    post["pgd"]  = f"{n_pgd_correct/n_total:.3f}"
        if do_fgsm:   post["fgsm"] = f"{n_fgsm_correct/n_total:.3f}"
        if do_level: post["loc"]  = f"{sum_level_pr/n_total:.3f}"
        if do_random: post["rnd"]  = f"{sum(random_sum.values())/(n_total*len(random_sum)):.3f}"
        pbar.set_postfix(**post)

    random_breakdown = {d: s / n_total for d, s in random_sum.items()} if do_random else None

    return {
        "clean_acc":         n_clean_correct / n_total,
        "clean_loss":        (clean_loss_sum / n_total) if criterion is not None else None,
        "pgd_acc":           (n_pgd_correct / n_total) if do_pgd    else None,
        "fgsm_acc":          (n_fgsm_correct / n_total) if do_fgsm  else None,
        "level_pr":         (sum_level_pr / n_total) if do_level else None,
        "random_pr_breakdown": random_breakdown,   # {dist: acc} or None
        "num_samples":       n_total,
    }

def evaluate_aa(model, loader, device, norm, epsilon, version="rand", eval_name="eval-AA"):
    """Run AutoAttack over the loader. Returns the adversarial accuracy.

    AA is hugely expensive — APGD-CE + APGD-DLR (rand) or the full standard
    suite (APGD-CE + APGD-DLR + FAB + Square). Use only as a final benchmark.
    """
    import autoattack
    norm_str = "Linf" if norm.lower() == "linf" else "L2"
    model.eval()

    adversary = autoattack.AutoAttack(
        model, norm=norm_str, eps=float(epsilon),
        version=version, verbose=False,
    )

    n_total = 0
    n_correct = 0
    pbar = tqdm(loader, desc=eval_name, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        with torch.no_grad():
            n_correct += (model(x_adv).argmax(dim=1) == y).sum().item()
        n_total += y.size(0)
        pbar.set_postfix(aa=f"{n_correct/n_total:.3f}")

    return n_correct / n_total
