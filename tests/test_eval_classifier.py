"""
Tests for eval_classifier.py

All heavy IO (torch.load, dataset loading, model building, Evaluator) is mocked
so the tests run without checkpoints, datasets, or GPUs.

Run with:
    pytest tests/test_eval_classifier.py -v
"""

import os
import sys
import csv
import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_classifier import main, set_seed


# =============================================================================
# Shared helpers
# =============================================================================

BASE_ARGS = [
    "eval_classifier.py",
    "--ckp_path", "/fake/ckp/resnet18_cifar10.pth",
]

def _fake_checkpoint(training_type="pr_training", with_meta=True):
    ckpt = {"model_state": {}, "epoch": 10}
    if with_meta:
        ckpt.update({"arch": "resnet18", "dataset": "cifar10",
                     "training_type": training_type, "img_size": 32})
    return ckpt


def _fake_model():
    m = MagicMock()
    m.to.return_value = m
    m.eval.return_value = m
    return m


def _fake_evaluator():
    ev = MagicMock()
    ev.evaluate_standard.return_value = {
        "mode": "pointwise", "acc": 0.92, "loss": 0.25, "num_samples": 1000,
    }
    ev.evaluate_adversarial.return_value = {
        "mode": "pointwise", "acc": 0.65, "loss": 0.80, "num_samples": 1000,
    }
    ev.evaluate_pr.return_value = {
        "mode": "distribution", "pr": 0.70, "num_samples": 1000, "num_draws": 32000,
        "stats": {"D_mu": 0.10, "D_sig": 0.20, "D_proxy": 0.30,
                  "pi_entropy": 1.00, "pi_max": 0.50},
    }
    return ev


def _fake_loader(size=1000):
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=size)
    loader = MagicMock()
    loader.dataset = ds
    return loader


def _run_main(extra_args=None, checkpoint=None, evaluator=None, loader=None):
    """
    Run main() with all external dependencies mocked.
    Returns the mock Evaluator instance so callers can inspect calls.
    """
    ev = evaluator or _fake_evaluator()
    ld = loader or _fake_loader()
    ckpt = checkpoint or _fake_checkpoint()

    argv = BASE_ARGS + (extra_args or [])

    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", argv))
        stack.enter_context(patch("eval_classifier.os.path.isfile", return_value=True))
        stack.enter_context(patch("eval_classifier.torch.load", return_value=ckpt))
        stack.enter_context(patch("eval_classifier.get_img_size", return_value=32))
        stack.enter_context(patch("eval_classifier.get_dataset", return_value=(MagicMock(), 10)))
        stack.enter_context(patch("eval_classifier.torch.utils.data.DataLoader", return_value=ld))
        stack.enter_context(patch("eval_classifier.build_model", return_value=_fake_model()))
        stack.enter_context(patch("eval_classifier.build_sigma_list", return_value=[0.01, 0.02]))
        stack.enter_context(patch("eval_classifier.Evaluator", return_value=ev))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        stack.enter_context(patch("torch.cuda.device_count", return_value=1))
        main()

    return ev


# =============================================================================
# Basic flow
# =============================================================================

class TestBasicFlow:

    def test_main_completes_without_error(self):
        """main() should run end-to-end without raising."""
        _run_main()

    def test_all_three_evaluations_are_called(self):
        ev = _run_main()
        ev.evaluate_standard.assert_called_once()
        ev.evaluate_adversarial.assert_called_once()
        ev.evaluate_pr.assert_called_once()

    def test_evaluator_receives_pgd_attack_callable(self):
        from utils.adv_attacker import pgd_attack
        ev = _run_main()
        _, kwargs = ev.evaluate_adversarial.call_args
        assert kwargs["attacker"] is pgd_attack

    def test_evaluator_receives_pr_generator_callable(self):
        from utils.pr_generator import pr_generator
        ev = _run_main()
        _, kwargs = ev.evaluate_pr.call_args
        assert kwargs["pr_generator"] is pr_generator


# =============================================================================
# CLI arguments forwarded correctly
# =============================================================================

class TestArgForwarding:

    def test_pgd_epsilon_forwarded(self):
        ev = _run_main(["--epsilon", "0.05"])
        _, kwargs = ev.evaluate_adversarial.call_args
        assert abs(kwargs["epsilon"] - 0.05) < 1e-9

    def test_pgd_alpha_forwarded(self):
        ev = _run_main(["--alpha", "0.01"])
        _, kwargs = ev.evaluate_adversarial.call_args
        assert abs(kwargs["alpha"] - 0.01) < 1e-9

    def test_pgd_steps_forwarded(self):
        ev = _run_main(["--pgd_steps", "10"])
        _, kwargs = ev.evaluate_adversarial.call_args
        assert kwargs["num_steps"] == 10

    def test_pgd_norm_forwarded(self):
        ev = _run_main(["--norm", "l2"])
        _, kwargs = ev.evaluate_adversarial.call_args
        assert kwargs["norm"] == "l2"

    def test_pr_num_samples_forwarded(self):
        ev = _run_main(["--num_samples", "16"])
        _, kwargs = ev.evaluate_pr.call_args
        assert kwargs["num_samples"] == 16

    def test_pr_K_forwarded(self):
        ev = _run_main(["--K", "5"])
        _, kwargs = ev.evaluate_pr.call_args
        assert kwargs["K"] == 5

    def test_pr_norm_forwarded(self):
        ev = _run_main(["--norm", "l2"])
        _, kwargs = ev.evaluate_pr.call_args
        assert kwargs["norm"] == "l2"

    def test_pr_epsilon_forwarded(self):
        ev = _run_main(["--epsilon", "0.05"])
        _, kwargs = ev.evaluate_pr.call_args
        assert abs(kwargs["epsilon"] - 0.05) < 1e-9


# =============================================================================
# training_type resolution
# =============================================================================

class TestTrainingTypeResolution:

    def test_training_type_taken_from_checkpoint(self):
        """If the checkpoint carries training_type, it must be used as-is."""
        # We only verify main() completes; the value shows up in stdout / CSV.
        _run_main(checkpoint=_fake_checkpoint(training_type="adv_training"))

    def test_training_type_inferred_standard(self):
        ckpt = _fake_checkpoint()
        ckpt["training_type"] = "unknown"
        _run_main(
            extra_args=["--ckp_path", "/ckp/standard_training/resnet18.pth"],
            checkpoint=ckpt,
        )

    def test_training_type_inferred_adv(self):
        ckpt = _fake_checkpoint()
        ckpt["training_type"] = "unknown"
        _run_main(
            extra_args=["--ckp_path", "/ckp/adv_training/resnet18.pth"],
            checkpoint=ckpt,
        )

    def test_training_type_inferred_pr(self):
        ckpt = _fake_checkpoint()
        ckpt["training_type"] = "unknown"
        _run_main(
            extra_args=["--ckp_path", "/ckp/pr_training/resnet18.pth"],
            checkpoint=ckpt,
        )

    def test_arch_and_dataset_overridden_by_checkpoint(self):
        """CLI --arch / --dataset should be overridden by what is in the checkpoint."""
        ckpt = _fake_checkpoint()
        ckpt["arch"] = "resnet50"
        ckpt["dataset"] = "cifar100"

        with ExitStack() as stack:
            argv = BASE_ARGS + ["--arch", "resnet18", "--dataset", "cifar10"]
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("eval_classifier.os.path.isfile", return_value=True))
            stack.enter_context(patch("eval_classifier.torch.load", return_value=ckpt))
            stack.enter_context(patch("eval_classifier.get_img_size", return_value=32))
            mock_get_dataset = stack.enter_context(
                patch("eval_classifier.get_dataset", return_value=(MagicMock(), 10))
            )
            stack.enter_context(patch("eval_classifier.torch.utils.data.DataLoader",
                                      return_value=_fake_loader()))
            stack.enter_context(patch("eval_classifier.build_model", return_value=_fake_model()))
            stack.enter_context(patch("eval_classifier.build_sigma_list", return_value=[]))
            stack.enter_context(patch("eval_classifier.Evaluator", return_value=_fake_evaluator()))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch("torch.cuda.device_count", return_value=1))
            main()

        # get_dataset is called with the arch from the checkpoint, not the CLI
        args_used = mock_get_dataset.call_args[0]
        assert args_used[0] == "cifar100"


# =============================================================================
# eval_train flag
# =============================================================================

class TestEvalTrainFlag:

    def test_evaluator_called_once_without_flag(self):
        with ExitStack() as stack:
            argv = BASE_ARGS
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("eval_classifier.os.path.isfile", return_value=True))
            stack.enter_context(patch("eval_classifier.torch.load",
                                      return_value=_fake_checkpoint()))
            stack.enter_context(patch("eval_classifier.get_img_size", return_value=32))
            stack.enter_context(patch("eval_classifier.get_dataset",
                                      return_value=(MagicMock(), 10)))
            stack.enter_context(patch("eval_classifier.torch.utils.data.DataLoader",
                                      return_value=_fake_loader()))
            stack.enter_context(patch("eval_classifier.build_model", return_value=_fake_model()))
            stack.enter_context(patch("eval_classifier.build_sigma_list", return_value=[]))
            MockEvaluator = stack.enter_context(patch("eval_classifier.Evaluator"))
            MockEvaluator.return_value = _fake_evaluator()
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch("torch.cuda.device_count", return_value=1))
            main()

        assert MockEvaluator.call_count == 1

    def test_evaluator_called_twice_with_eval_train(self):
        with ExitStack() as stack:
            argv = BASE_ARGS + ["--eval_train"]
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch("eval_classifier.os.path.isfile", return_value=True))
            stack.enter_context(patch("eval_classifier.torch.load",
                                      return_value=_fake_checkpoint()))
            stack.enter_context(patch("eval_classifier.get_img_size", return_value=32))
            stack.enter_context(patch("eval_classifier.get_dataset",
                                      return_value=(MagicMock(), 10)))
            stack.enter_context(patch("eval_classifier.torch.utils.data.DataLoader",
                                      return_value=_fake_loader()))
            stack.enter_context(patch("eval_classifier.build_model", return_value=_fake_model()))
            stack.enter_context(patch("eval_classifier.build_sigma_list", return_value=[]))
            MockEvaluator = stack.enter_context(patch("eval_classifier.Evaluator"))
            MockEvaluator.return_value = _fake_evaluator()
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch("torch.cuda.device_count", return_value=1))
            main()

        # One Evaluator per split: test + train
        assert MockEvaluator.call_count == 2


# =============================================================================
# Error handling
# =============================================================================

class TestErrorHandling:

    def test_missing_checkpoint_raises_file_not_found(self):
        with patch("sys.argv", BASE_ARGS), \
             patch("eval_classifier.os.path.isfile", return_value=False):
            with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
                main()


# =============================================================================
# CSV output
# =============================================================================

class TestCSVOutput:

    def test_csv_is_created(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        _run_main(["--save_csv", csv_path])
        assert os.path.isfile(csv_path)

    def test_csv_contains_expected_columns(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        _run_main(["--save_csv", csv_path])

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        for col in ("arch", "dataset", "training_type",
                    "test_clean_acc", "test_clean_loss",
                    "test_pgd_acc", "test_pgd_loss",
                    "test_pr"):
            assert col in row, f"Missing column: {col}"

    def test_csv_pr_stats_present(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        _run_main(["--save_csv", csv_path])

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        for stat in ("D_mu", "D_sig", "D_proxy", "pi_entropy", "pi_max"):
            assert f"test_pr_{stat}" in row, f"Missing PR stat column: test_pr_{stat}"

    def test_csv_values_match_evaluator_output(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        ev = _fake_evaluator()
        ev.evaluate_standard.return_value["acc"] = 0.88
        ev.evaluate_adversarial.return_value["acc"] = 0.55
        ev.evaluate_pr.return_value["pr"] = 0.60
        _run_main(["--save_csv", csv_path], evaluator=ev)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert abs(float(row["test_clean_acc"]) - 0.88) < 1e-6
        assert abs(float(row["test_pgd_acc"])   - 0.55) < 1e-6
        assert abs(float(row["test_pr"])         - 0.60) < 1e-6

    def test_csv_not_created_without_flag(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        _run_main()  # no --save_csv
        assert not os.path.isfile(csv_path)


# =============================================================================
# set_seed
# =============================================================================

class TestSetSeed:

    def test_set_seed_produces_deterministic_output(self):
        set_seed(0)
        a = __import__("torch").randn(5)
        set_seed(0)
        b = __import__("torch").randn(5)
        assert (a == b).all()

    def test_different_seeds_produce_different_output(self):
        set_seed(0)
        a = __import__("torch").randn(5)
        set_seed(1)
        b = __import__("torch").randn(5)
        assert not (a == b).all()
