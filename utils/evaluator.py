# utils/evaluator.py
#
# Unified evaluation infrastructure for image classifiers.
#
# An Evaluator wraps (model, dataloader) and runs one metric per call, each over
# its own pass of the data. A `transform` decides what the model sees, and its
# return type decides how results are reduced:
#
#   PointwiseEvalBatch     (B, C, H, W)     -> {"mode": "pointwise", "acc",
#                                               "loss", "num_samples"}
#   DistributionEvalBatch  (B, N, C, H, W)  -> {"mode": "distribution", "pr",
#                                               "num_draws", "num_samples"}
#
# where pr = mean fraction of the N draws that stay correctly classified. Both
# shapes carry an optional "stats" key holding batch-size-weighted scalars that
# the generator reported. Mixing batch types within one run raises.
#
# Note this is deliberately *not* the same thing as utils/epoch_eval.py:
# evaluate_per_epoch collects every metric in a single pass, which is what
# per-epoch evaluation during training needs. Evaluator trades that for the
# freedom to run one metric at a time. Folding either into the other would
# either slow training down or grow the code without removing anything, so
# they are kept apart on purpose.
#
# Example:
#
#   from utils.evaluator import Evaluator
#   from src.adv_loss import pgd_attack
#
#   ev = Evaluator(model, test_loader, criterion=nn.CrossEntropyLoss(), device=device)
#   clean = ev.evaluate_standard()
#   pgd   = ev.evaluate_adversarial(attacker=pgd_attack, epsilon=8/255,
#                                   alpha=2/255, num_steps=20, norm="linf")
#   pr    = ev.evaluate_pr_random(norm="linf", epsilon=8/255,
#                                 num_samples=32, noise_dist="gaussian")
#   ev.update_loader(train_loader)          # swap split, keep the Evaluator
#
# Per-method details live in the docstrings below.

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

    ``pr_generator`` is any callable returning (x_samples, stats) with
    x_samples of shape (B, N, C, H, W).  Used by evaluate_pr_random and
    evaluate_pr_gmm.
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
        results = evaluator.evaluate_pr_random(
            epsilon=8/255, norm="linf", num_samples=32, noise_dist="gaussian",
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

            # A transform may leave the model in train mode — restore eval
            # mode before running inference.
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
