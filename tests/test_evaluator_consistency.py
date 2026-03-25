"""
Consistency tests: utils/evaluator.py  vs  eval_classifier.py

Run with:
    pytest tests/test_evaluator_consistency.py -v

What is checked:
    - Standard evaluation:    acc and loss must be identical.
    - Adversarial evaluation: acc and loss must be identical given the same attacker.
    - PR evaluation (N=1):    pr must equal adversarial accuracy (metrics coincide at N=1).
    - PR evaluation (N>1):    metrics DIFFER by design (see note below).

Note on PR metric divergence (N > 1):
    eval_classifier: accuracy of the *averaged softmax* over N draws  ("ensemble accuracy")
    evaluator:       mean *fraction* of the N draws that are correct   ("probabilistic robustness")
    These are different quantities. Only at N=1 do they collapse to the same value.
"""

import sys
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_classifier import evaluate, evaluate_with_pgd_attack, evaluate_with_pr_attack
from utils.evaluator import Evaluator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    """Small deterministic linear classifier (flat inputs → 3 classes)."""
    torch.manual_seed(0)
    m = nn.Linear(8, 3)
    m.eval()
    return m


@pytest.fixture
def loader():
    """24 samples, batch size 8 → 3 full batches (no remainder, clean stats)."""
    torch.manual_seed(1)
    x = torch.randn(24, 8)
    y = torch.randint(0, 3, (24,))
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


# =============================================================================
# Mock helpers
# =============================================================================

def make_mock_attacker(delta: float = 0.05):
    """
    Attacker that adds a fixed scalar delta to every input.
    Works regardless of whether it is called positionally
    (eval_classifier style) or via **kwargs (evaluator style).
    """
    def _attack(model, x, y, *args, **kwargs):  # noqa: N802
        return x + delta
    return _attack


def make_mock_pr_generator(N: int, delta: float = 0.05):
    """
    PR generator that returns N copies of x + small random noise.
    Accepts and ignores any extra kwargs (including return_stats).
    Always returns (x_samples, stats) as a tuple.
    """
    _stats_template = {
        "D_mu":       torch.tensor(0.10),
        "D_sig":      torch.tensor(0.20),
        "D_proxy":    torch.tensor(0.30),
        "pi_entropy": torch.tensor(1.00),
        "pi_max":     torch.tensor(0.50),
    }

    def _gen(model, x, y, **kwargs):  # noqa: N802
        torch.manual_seed(42)
        B = x.size(0)
        noise = delta * torch.randn(B, N, *x.shape[1:])
        x_samples = x.unsqueeze(1) + noise          # (B, N, ...)
        # Return fresh copies of the tensors so both callers are independent
        stats = {k: v.clone() for k, v in _stats_template.items()}
        return x_samples, stats

    return _gen


# =============================================================================
# Standard evaluation
# =============================================================================

class TestStandardEval:

    def test_accuracy_matches(self, model, loader, criterion, device):
        acc_ref, _ = evaluate(model, loader, device, criterion)
        result = Evaluator(model, loader, criterion, device).evaluate_standard()
        assert abs(result["acc"] - acc_ref) < 1e-6

    def test_loss_matches(self, model, loader, criterion, device):
        _, loss_ref = evaluate(model, loader, device, criterion)
        result = Evaluator(model, loader, criterion, device).evaluate_standard()
        assert abs(result["loss"] - loss_ref) < 1e-6

    def test_loss_none_without_criterion(self, model, loader, device):
        _, loss_ref = evaluate(model, loader, device, criterion=None)
        result = Evaluator(model, loader, criterion=None, device=device).evaluate_standard()
        assert loss_ref is None
        assert result["loss"] is None

    def test_accuracy_matches_without_criterion(self, model, loader, device):
        acc_ref, _ = evaluate(model, loader, device, criterion=None)
        result = Evaluator(model, loader, criterion=None, device=device).evaluate_standard()
        assert abs(result["acc"] - acc_ref) < 1e-6

    def test_num_samples(self, model, loader, criterion, device):
        result = Evaluator(model, loader, criterion, device).evaluate_standard()
        assert result["num_samples"] == 24

    def test_result_mode(self, model, loader, device):
        result = Evaluator(model, loader, device=device).evaluate_standard()
        assert result["mode"] == "pointwise"


# =============================================================================
# Adversarial evaluation
# =============================================================================

class TestAdversarialEval:

    def test_accuracy_matches(self, model, loader, criterion, device):
        attacker = make_mock_attacker(delta=0.05)

        # eval_classifier hardcodes `pgd_attack` — patch it with our mock.
        with patch("eval_classifier.pgd_attack",
                   side_effect=lambda m, x, y, eps, alp, steps, norm: attacker(m, x, y)):
            acc_ref, _ = evaluate_with_pgd_attack(
                model, loader, device, criterion,
                epsilon=0.03, alpha=0.01, num_steps=3, norm="linf",
            )

        result = Evaluator(model, loader, criterion, device).evaluate_adversarial(
            attacker=attacker,
            epsilon=0.03, alpha=0.01, num_steps=3, norm="linf",
        )
        assert abs(result["acc"] - acc_ref) < 1e-6

    def test_loss_matches(self, model, loader, criterion, device):
        attacker = make_mock_attacker(delta=0.05)

        with patch("eval_classifier.pgd_attack",
                   side_effect=lambda m, x, y, eps, alp, steps, norm: attacker(m, x, y)):
            _, loss_ref = evaluate_with_pgd_attack(
                model, loader, device, criterion,
                epsilon=0.03, alpha=0.01, num_steps=3, norm="linf",
            )

        result = Evaluator(model, loader, criterion, device).evaluate_adversarial(
            attacker=attacker,
            epsilon=0.03, alpha=0.01, num_steps=3, norm="linf",
        )
        assert abs(result["loss"] - loss_ref) < 1e-6

    def test_result_mode(self, model, loader, device):
        attacker = make_mock_attacker()
        result = Evaluator(model, loader, device=device).evaluate_adversarial(attacker=attacker)
        assert result["mode"] == "pointwise"

    def test_identity_attacker_equals_clean(self, model, loader, criterion, device):
        """An attacker that does nothing should reproduce standard-eval results."""
        identity = lambda m, x, y, **kw: x  # noqa: E731
        adv_result = Evaluator(model, loader, criterion, device).evaluate_adversarial(
            attacker=identity
        )
        clean_result = Evaluator(model, loader, criterion, device).evaluate_standard()
        assert abs(adv_result["acc"]  - clean_result["acc"])  < 1e-6
        assert abs(adv_result["loss"] - clean_result["loss"]) < 1e-6


# =============================================================================
# PR evaluation
# =============================================================================

class TestPREval:

    def test_pr_n1_equals_adversarial_accuracy(self, model, loader, criterion, device):
        """
        With N=1 draw, PR == fraction of correctly classified adversarial samples
        == adversarial accuracy.  Both evaluator.evaluate_pr and
        evaluator.evaluate_adversarial must agree.
        """
        attacker = make_mock_attacker(delta=0.05)

        def pr_gen_n1(m, x, y, **kwargs):
            x_adv = attacker(m, x, y).unsqueeze(1)  # (B, 1, ...)
            stats = {
                "D_mu": torch.tensor(0.0), "D_sig": torch.tensor(0.0),
                "D_proxy": torch.tensor(0.0), "pi_entropy": torch.tensor(0.0),
                "pi_max": torch.tensor(1.0),
            }
            return x_adv, stats

        evaluator = Evaluator(model, loader, criterion, device)
        pr_result  = evaluator.evaluate_pr(pr_generator=pr_gen_n1)
        adv_result = evaluator.evaluate_adversarial(attacker=attacker)

        assert abs(pr_result["pr"] - adv_result["acc"]) < 1e-6

    def test_pr_result_mode(self, model, loader, device):
        gen = make_mock_pr_generator(N=4)
        result = Evaluator(model, loader, device=device).evaluate_pr(pr_generator=gen)
        assert result["mode"] == "distribution"

    def test_pr_in_range(self, model, loader, device):
        gen = make_mock_pr_generator(N=4)
        result = Evaluator(model, loader, device=device).evaluate_pr(pr_generator=gen)
        assert 0.0 <= result["pr"] <= 1.0

    def test_num_draws(self, model, loader, device):
        N = 6
        gen = make_mock_pr_generator(N=N)
        result = Evaluator(model, loader, device=device).evaluate_pr(pr_generator=gen)
        assert result["num_draws"] == result["num_samples"] * N
        assert result["num_samples"] == 24

    def test_stats_returned_and_averaged(self, model, loader, device):
        """Stats dict should be present and contain the expected keys."""
        gen = make_mock_pr_generator(N=4)
        result = Evaluator(model, loader, device=device).evaluate_pr(pr_generator=gen)
        assert "stats" in result
        for key in ("D_mu", "D_sig", "D_proxy", "pi_entropy", "pi_max"):
            assert key in result["stats"], f"Missing stat key: {key}"

    def test_pr_metric_is_mean_fraction_not_ensemble_accuracy(self, model, loader, device):
        """
        Document that evaluator.pr is the mean *fraction* of draws correct,
        NOT the accuracy of the averaged softmax (ensemble accuracy).

        We compute both metrics manually on the same draws and check they differ.
        They only coincide at N=1 (handled by test_pr_n1_equals_adversarial_accuracy).
        """
        import torch.nn.functional as F

        N = 8
        delta = 0.20  # large perturbation → varied per-draw predictions
        gen = make_mock_pr_generator(N=N, delta=delta)

        evaluator = Evaluator(model, loader, device=device)
        pr_result = evaluator.evaluate_pr(pr_generator=gen)

        # Recompute ensemble accuracy (accuracy of averaged softmax) manually.
        ensemble_correct = 0
        total = 0
        for x, y in loader:
            x_dist, _ = gen(model, x, y)             # (B, N, 8)
            B = x.size(0)
            x_flat = x_dist.reshape(B * N, *x_dist.shape[2:])
            with torch.no_grad():
                logits = model(x_flat).view(B, N, -1)
            preds = F.softmax(logits, dim=-1).mean(dim=1).argmax(dim=1)
            ensemble_correct += (preds == y).sum().item()
            total += B
        ensemble_acc = ensemble_correct / total

        # Both metrics are valid probabilities.
        assert 0.0 <= pr_result["pr"] <= 1.0
        assert 0.0 <= ensemble_acc <= 1.0
        # They are different quantities — assert they do not agree in general.
        assert abs(pr_result["pr"] - ensemble_acc) > 1e-6, (
            "Mean-fraction-correct and ensemble-accuracy happened to be equal "
            "on this data; try a different seed or larger delta."
        )


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:

    def test_empty_loader_raises(self, model, criterion, device):
        empty_loader = DataLoader(TensorDataset(torch.empty(0, 8), torch.empty(0, dtype=torch.long)),
                                  batch_size=8)
        with pytest.raises(ValueError, match="Empty dataloader"):
            Evaluator(model, empty_loader, criterion, device).evaluate_standard()

    def test_update_loader(self, model, loader, criterion, device):
        """update_loader should change which data the evaluator runs on."""
        torch.manual_seed(99)
        x2 = torch.randn(16, 8)
        y2 = torch.randint(0, 3, (16,))
        loader2 = DataLoader(TensorDataset(x2, y2), batch_size=8, shuffle=False)

        evaluator = Evaluator(model, loader, criterion, device)
        r1 = evaluator.evaluate_standard()

        evaluator.update_loader(loader2)
        r2 = evaluator.evaluate_standard()

        assert r1["num_samples"] == 24
        assert r2["num_samples"] == 16
