# utils/evaluator.py
#
# Unified evaluation infrastructure for image classifiers.
#
# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
#   PointwiseEvalBatch      – wraps a single (B, C, H, W) batch of inputs,
#                             produced by clean or pointwise-adversarial transforms.
#
#   DistributionEvalBatch   – wraps a (B, N, C, H, W) batch of N perturbed
#                             copies per input, produced by distributional
#                             (PR) transforms.
#
# ─────────────────────────────────────────────────────────────────────────────
# Transform adapters
# ─────────────────────────────────────────────────────────────────────────────
#   identity_transform(model, x, y, **kwargs)
#       Returns PointwiseEvalBatch(x) unchanged — used for standard clean eval.
#
#   adv_transform(model, x, y, attacker, **kwargs)
#       Calls attacker(model, x, y, **kwargs) and wraps the result in a
#       PointwiseEvalBatch. Any pointwise attacker (e.g. PGD) can be plugged in.
#
#   pr_transform(model, x, y, pr_generator, **kwargs)
#       Calls pr_generator(model, x, y, **kwargs) which must return
#       (x_samples, stats), and wraps the result in a DistributionEvalBatch.
#
# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────
#   Evaluator(model, dataloader, criterion=None, device="cuda")
#
#   Public methods:
#       .evaluate_standard(eval_name)
#           Runs clean evaluation. Returns:
#           {"mode": "pointwise", "acc": float, "loss": float|None,
#            "num_samples": int [, "stats": dict]}
#
#       .evaluate_adversarial(attacker, eval_name, **kwargs)
#           Runs pointwise adversarial evaluation using the given attacker.
#           Returns the same dict shape as evaluate_standard.
#
#       .evaluate_pr(pr_generator, eval_name, **kwargs)
#           Runs distributional PR evaluation. Returns:
#           {"mode": "distribution", "pr": float, "num_samples": int,
#            "num_draws": int [, "stats": dict]}
#           where pr = mean fraction of N draws correctly classified per sample.
#
#       .evaluate(transform, eval_name, **kwargs)
#           Generic entry point — the three methods above are thin wrappers
#           around this. Accepts any transform returning a recognised batch type.
#
#       .update_loader(dataloader)
#           Swap the dataloader without rebuilding the Evaluator.
#
# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────
#   _PointwiseAccumulator   – accumulates correct count and loss sum.
#   _DistAccumulator        – accumulates PR sum and draw count.
#   _StatsAccumulator       – batch-size-weighted accumulation of scalar stats.
#   _assert_consistent_mode – raises if batch types are mixed within one run.
#
# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
#
#   from utils.evaluator import Evaluator
#   from src.adv_attacker import pgd_attack
#   from src.langevin4pr import pr_generator
#   from configs.train_clf_cfg import build_sigma_list
#
#   evaluator = Evaluator(model, test_loader, criterion=nn.CrossEntropyLoss(),
#                         device=device)
#
#   # 1. Standard clean evaluation
#   clean = evaluator.evaluate_standard()
#   print(f"clean acc={clean['acc']*100:.2f}%  loss={clean['loss']:.4f}")
#
#   # 2. PGD adversarial evaluation
#   pgd = evaluator.evaluate_adversarial(
#       attacker=pgd_attack,
#       eval_name="PGD-20",
#       epsilon=8/255, alpha=2/255, num_steps=20, norm="linf",
#   )
#   print(f"pgd   acc={pgd['acc']*100:.2f}%  loss={pgd['loss']:.4f}")
#
#   # 3. PR distributional evaluation
#   sigma_list = build_sigma_list(epsilon=8/255, K=3, mode_type="linear")
#   pr = evaluator.evaluate_pr(
#       pr_generator=pr_generator,
#       eval_name="PR",
#       norm="linf", epsilon=8/255,
#       K=3, sigma_list=sigma_list,
#       num_samples=32, beta_mix=1.0, kappa=1.0,
#       fisher_damping=1e-7, tau=1e-4, noise_scale=1.0,
#   )
#   print(f"pr    pr={pr['pr']*100:.2f}%  draws={pr['num_draws']}")
#   print(f"stats: { {k: f'{v:.3e}' for k, v in pr['stats'].items()} }")
#
#   # 4. Switch to train split without rebuilding
#   evaluator.update_loader(train_loader)
#   clean_train = evaluator.evaluate_standard(eval_name="standard-train")

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from utils.utils import pr_random_generator, pr_gmm_generator


# =========================================
# Evaluation batch types
# =========================================

@dataclass
class PointwiseEvalBatch:
    """
    Result of a pointwise transform (clean or adversarial).

    x: (B, C, H, W)
    """
    x: torch.Tensor
    stats: Optional[Dict[str, Any]] = None


@dataclass
class DistributionEvalBatch:
    """
    Result of a distributional transform (PR).

    x: (B, N, C, H, W)
    """
    x: torch.Tensor
    stats: Optional[Dict[str, Any]] = None


# =========================================
# Transform adapters
# =========================================

def identity_transform(_model, x, _y, **_kwargs):
    """No perturbation — standard evaluation."""
    return PointwiseEvalBatch(x=x)


def adv_transform(model, x, y, attacker, **kwargs):
    """
    Pointwise adversarial transform.

    Example:
        result = adv_transform(
            model, x, y,
            attacker=pgd_attack,
            epsilon=8/255, alpha=2/255, num_steps=20, norm="linf",
        )
    """
    x_adv = attacker(model, x, y, **kwargs)
    return PointwiseEvalBatch(x=x_adv)


def pr_transform(model, x, y, pr_generator, **kwargs):
    """
    Distributional PR transform.

    Example:
        result = pr_transform(
            model, x, y,
            pr_generator=pr_generator,
            epsilon=8/255, norm="linf", num_samples=32, ...
        )
    """
    x_samples, stats = pr_generator(model, x, y, **kwargs)
    return DistributionEvalBatch(x=x_samples, stats=stats)


# =========================================
# Evaluator
# =========================================

class Evaluator:
    """
    Unified evaluator supporting:
      - Standard clean accuracy / loss              → evaluate_standard()
      - Pointwise adversarial accuracy / loss       → evaluate_adversarial()
      - Distributional PR evaluation (Langevin)     → evaluate_pr()
      - Distributional PR evaluation (random noise) → evaluate_pr_random()
      - Distributional PR evaluation (trained GMM)  → evaluate_pr_gmm()

    Usage::

        evaluator = Evaluator(model, test_loader, criterion, device)

        # Standard
        results = evaluator.evaluate_standard()

        # PGD adversarial
        results = evaluator.evaluate_adversarial(
            attacker=pgd_attack,
            epsilon=8/255, alpha=2/255, num_steps=20, norm="linf",
        )

        # PR distributional
        results = evaluator.evaluate_pr(
            pr_generator=pr_generator,
            epsilon=8/255, norm="linf", num_samples=32, ...
        )

    All methods return a dict. Keys by mode:

        Pointwise:      {"mode", "acc", "loss", "num_samples" [, "stats"]}
        Distributional: {"mode", "pr",  "num_samples", "num_draws" [, "stats"]}
    """

    def __init__(self, model, dataloader, criterion=None, device="cuda"):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def update_loader(self, dataloader):
        self.dataloader = dataloader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_standard(self, eval_name="standard"):
        """Standard clean evaluation."""
        return self.evaluate(transform=None, eval_name=eval_name)

    def evaluate_adversarial(self, attacker, eval_name="adversarial", **kwargs):
        """
        Pointwise adversarial evaluation using any attacker callable.

        Args:
            attacker: function(model, x, y, **kwargs) -> x_adv
            **kwargs: forwarded to attacker (e.g. epsilon, alpha, num_steps, norm)
        """
        return self.evaluate(
            transform=adv_transform,
            eval_name=eval_name,
            attacker=attacker,
            **kwargs,
        )

    def evaluate_pr(self, pr_generator, eval_name="pr", **kwargs):
        """
        Distributional PR evaluation.

        Args:
            pr_generator: function(model, x, y, **kwargs) -> (x_samples, stats)
                          where x_samples has shape (B, N, C, H, W)
            **kwargs: forwarded to pr_generator
        """
        return self.evaluate(
            transform=pr_transform,
            eval_name=eval_name,
            pr_generator=pr_generator,
            **kwargs,
        )

    def evaluate_pr_gmm(self, gmm, eval_name="pr_gmm", **kwargs):
        """
        Distributional evaluation using a trained GMM4PR model.

        The perturbation budget (epsilon, norm) is taken from the GMM's own
        training configuration — no separate epsilon argument is required.

        Args:
            gmm       : trained GMM4PR instance returned by load_gmm_model().
                        Its internal feature extractor may differ from the
                        classifier stored in self.model — that is intentional.
            eval_name : label shown on the tqdm bar.
            **kwargs  : forwarded to pr_gmm_generator.
                        Key parameters:
                          num_samples – N draws per input.
                          epsilon     – override the GMM's training radius.
                          norm        – override the GMM's training norm
                                        ("linf" or "l2").
        """
        return self.evaluate(
            transform=pr_transform,
            eval_name=eval_name,
            pr_generator=pr_gmm_generator,
            gmm=gmm,
            **kwargs,
        )

    def evaluate_pr_random(self, eval_name="pr_random", **kwargs):
        """
        Distributional evaluation using i.i.d. random noise perturbations.

        Perturbations are sampled from a chosen distribution (Gaussian, Uniform,
        or Laplace), projected onto the epsilon-ball, and evaluated with the same
        (B, N, C, H, W) pipeline as evaluate_pr — making results directly
        comparable.

        Args:
            eval_name : label shown on the tqdm bar.
            **kwargs  : forwarded to pr_random_generator.
                        Key parameters:
                          epsilon    – perturbation budget
                          norm       – "linf" or "l2"
                          num_samples – N draws per input
                          noise_dist  – "gaussian" | "uniform" | "laplace"
        """
        return self.evaluate(
            transform=pr_transform,
            eval_name=eval_name,
            pr_generator=pr_random_generator,
            **kwargs,
        )

    def evaluate(self, transform=None, eval_name="evaluation", **kwargs):
        """
        Generic evaluation loop.

        Args:
            transform: None for clean eval, or a callable returning
                       PointwiseEvalBatch / DistributionEvalBatch.
            eval_name: label shown on the tqdm bar.
            **kwargs:  forwarded verbatim to transform.

        Returns:
            dict — see class docstring for key/value details.
        """
        self.model.eval()

        mode = None
        pw_acc   = _PointwiseAccumulator()
        dist_acc = _DistAccumulator()
        stats_acc = _StatsAccumulator()

        for x, y in tqdm(self.dataloader, desc=eval_name, leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if transform is None:
                batch_out = identity_transform(self.model, x, y)
            else:
                batch_out = transform(self.model, x, y, **kwargs)

            # Some transforms (e.g. pr_generator) may leave the model in
            # train mode — restore eval mode before running inference.
            self.model.eval()

            if isinstance(batch_out, PointwiseEvalBatch):
                mode = _assert_consistent_mode(mode, "pointwise")
                pw_acc.update(self._eval_pointwise_batch(batch_out, y))
                stats_acc.update(batch_out.stats, y.size(0))

            elif isinstance(batch_out, DistributionEvalBatch):
                mode = _assert_consistent_mode(mode, "distribution")
                dist_acc.update(self._eval_dist_batch(batch_out, y))
                stats_acc.update(batch_out.stats, y.size(0))

            else:
                raise TypeError(
                    f"transform must return PointwiseEvalBatch or DistributionEvalBatch, "
                    f"got {type(batch_out).__name__}"
                )

        if mode is None:
            raise ValueError("Empty dataloader: no batches were evaluated.")

        result = pw_acc.result(self.criterion) if mode == "pointwise" else dist_acc.result()

        stats = stats_acc.result()
        if stats:
            result["stats"] = stats

        return result

    # ------------------------------------------------------------------
    # Private batch-level helpers
    # ------------------------------------------------------------------

    def _eval_pointwise_batch(self, batch_out: PointwiseEvalBatch, y: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self.model(batch_out.x)
            preds  = logits.argmax(dim=1)

        correct  = (preds == y).sum().item()
        loss_sum = 0.0
        if self.criterion is not None:
            loss_sum = self.criterion(logits, y).item() * y.size(0)

        return {"correct": correct, "loss_sum": loss_sum, "num_samples": y.size(0)}

    def _eval_dist_batch(self, batch_out: DistributionEvalBatch, y: torch.Tensor) -> dict:
        x_dist = batch_out.x
        if x_dist.dim() < 3:
            raise ValueError(
                f"DistributionEvalBatch.x must be at least 3-D (B, N, ...), got {tuple(x_dist.shape)}"
            )

        B, N   = x_dist.shape[:2]
        x_flat = x_dist.reshape(B * N, *x_dist.shape[2:])
        y_flat = y.unsqueeze(1).expand(B, N).reshape(-1)

        with torch.no_grad():
            preds = self.model(x_flat).argmax(dim=1)

        pr_per_sample = (preds == y_flat).view(B, N).float().mean(dim=1)

        return {
            "pr_sum":      pr_per_sample.sum().item(),
            "num_samples": B,
            "num_draws":   B * N,
        }


# =========================================
# Private accumulator helpers
# =========================================

def _assert_consistent_mode(current: Optional[str], new: str) -> str:
    """Raise if batch output types are mixed within one evaluation run."""
    if current is not None and current != new:
        raise ValueError(
            f"Mixed batch output types within one evaluation run: "
            f"saw '{new}' after '{current}'"
        )
    return new


class _PointwiseAccumulator:
    """Accumulates correct count, loss, and sample count across batches."""

    def __init__(self):
        self.correct     = 0
        self.loss_sum    = 0.0
        self.num_samples = 0

    def update(self, batch: dict):
        self.correct     += batch["correct"]
        self.loss_sum    += batch["loss_sum"]
        self.num_samples += batch["num_samples"]

    def result(self, criterion) -> dict:
        n = max(1, self.num_samples)
        return {
            "mode":        "pointwise",
            "acc":         self.correct / n,
            "loss":        self.loss_sum / n if criterion is not None else None,
            "num_samples": self.num_samples,
        }


class _DistAccumulator:
    """Accumulates PR sum, sample count, and total draws across batches."""

    def __init__(self):
        self.pr_sum      = 0.0
        self.num_samples = 0
        self.num_draws   = 0

    def update(self, batch: dict):
        self.pr_sum      += batch["pr_sum"]
        self.num_samples += batch["num_samples"]
        self.num_draws   += batch["num_draws"]

    def result(self) -> dict:
        n = max(1, self.num_samples)
        return {
            "mode":        "distribution",
            "pr":          self.pr_sum / n,
            "num_samples": self.num_samples,
            "num_draws":   self.num_draws,
        }


class _StatsAccumulator:
    """Batch-size-weighted accumulator for scalar stats dicts."""

    def __init__(self):
        self._sums   = {}
        self._weight = 0

    def update(self, stats: Optional[Dict[str, Any]], batch_size: int):
        if stats is None:
            return
        for k, v in stats.items():
            if torch.is_tensor(v):
                if v.numel() != 1:
                    continue
                v = v.item()
            if isinstance(v, (int, float)):
                self._sums[k] = self._sums.get(k, 0.0) + float(v) * batch_size
        self._weight += batch_size

    def result(self) -> dict:
        if not self._sums or self._weight == 0:
            return {}
        return {k: v / self._weight for k, v in self._sums.items()}
