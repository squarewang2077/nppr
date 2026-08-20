# tests/test_pos_geo_loss_mode.py - Model-mode contract for pos_geo_loss.
#
# Requirements:
#   torch >= 2.0, pytest
#
# Run with:
#   PYTHONPATH=. pytest tests/test_pos_geo_loss_mode.py -q
#
# The experiment compares positions at the *same* margin level, so the level
# has to mean the same thing in the inner solver and the outer loss. These
# tests pin that: one eval-mode model throughout, BatchNorm running statistics
# untouched, Dropout off in both forwards, and the caller's mode restored even
# when the call raises. Plus the things eval mode must *not* break — gradients
# still reach the parameters.

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pos_geo_loss import GEOMETRY_MODES, WEIGHT_MODES, margin, pos_geo_loss

SOLVER = dict(num_starts=3, num_steps=4, epsilon=8 / 255)


class BNDropNet(nn.Module):
    """Small net carrying both mode-sensitive layers, so eval/train differ."""

    def __init__(self, p_drop=0.5):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        h = self.drop(h)
        return self.head(h.mean(dim=(2, 3)))


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(6, 3, 8, 8), torch.randint(0, 4, (6,))


@pytest.fixture
def model():
    torch.manual_seed(0)
    net = BNDropNet()
    # Give the BN running stats a non-default value, so "unchanged" is a real
    # assertion rather than one about freshly initialised zeros/ones.
    net.train()
    with torch.no_grad():
        net(torch.rand(16, 3, 8, 8))
    return net


def _bn_stats(m):
    return [(b.running_mean.clone(), b.running_var.clone(), b.num_batches_tracked.clone())
            for b in m.modules() if isinstance(b, nn.BatchNorm2d)]


def _target_level(model, x, y):
    """Median clean margin, computed without disturbing the model's mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        t = margin(model(x), y).median().item()
    model.train(was_training)
    return t


def _call(model, batch, **kw):
    x, y = batch
    t = _target_level(model, x, y)
    return pos_geo_loss(model, x, y, nn.CrossEntropyLoss(), t=t, **SOLVER, **kw)


# ------------------------------------------------------------------
#                          Mode restoration
# ------------------------------------------------------------------

@pytest.mark.parametrize("start_training", [True, False])
def test_caller_mode_is_restored(model, batch, start_training):
    model.train(start_training)
    _call(model, batch)
    assert model.training is start_training


@pytest.mark.parametrize("start_training", [True, False])
def test_mode_restored_even_when_the_call_raises(model, batch, start_training):
    """The finally block, not the happy path, is what this checks."""
    model.train(start_training)
    with pytest.raises(ValueError):
        _call(model, batch, geometry_mode="not-a-mode")
    assert model.training is start_training

    _, y = batch
    with pytest.raises(RuntimeError):
        # Wrong channel count blows up inside the solver's first forward pass,
        # i.e. after model.eval() has already been applied.
        pos_geo_loss(model, torch.rand(6, 1, 8, 8), y, None, t=0.0, **SOLVER)
    assert model.training is start_training


# ------------------------------------------------------------------
#                     BatchNorm and Dropout are frozen
# ------------------------------------------------------------------

@pytest.mark.parametrize("start_training", [True, False])
def test_bn_running_stats_unchanged(model, batch, start_training):
    model.train(start_training)
    before = _bn_stats(model)
    _call(model, batch)
    for (m0, v0, n0), (m1, v1, n1) in zip(before, _bn_stats(model)):
        assert torch.equal(m0, m1), "BN running_mean moved"
        assert torch.equal(v0, v1), "BN running_var moved"
        assert torch.equal(n0, n1), "BN num_batches_tracked moved"


def test_dropout_is_off_in_both_forwards(model, batch):
    """
    With Dropout active the outer logits would differ run to run. Calling twice
    from train mode with identical inputs must give identical logits, which can
    only happen if eval mode held for the outer forward.
    """
    model.train()
    torch.manual_seed(1234)
    _, _, a = _call(model, batch)
    torch.manual_seed(1234)
    _, _, b = _call(model, batch)
    assert torch.allclose(a["logits_adv"], b["logits_adv"], atol=1e-6)


def test_outer_forward_matches_an_explicit_eval_forward(model, batch):
    """The outer logits must equal what an eval-mode forward on x_adv gives."""
    model.train()
    _, x_adv, info = _call(model, batch)
    model.eval()
    with torch.no_grad():
        expected = model(x_adv.reshape(-1, *x_adv.shape[2:]))
    assert torch.allclose(info["logits_adv"], expected, atol=1e-5)


# ------------------------------------------------------------------
#                   Inner / outer level consistency
# ------------------------------------------------------------------

@pytest.mark.parametrize("start_training", [True, False])
def test_level_mode_drift_is_numerically_zero(model, batch, start_training):
    model.train(start_training)
    _, _, info = _call(model, batch)
    assert info["level_mode_drift_mean"].item() < 1e-4, (
        "solver and outer forward disagree on the margin — they are no longer "
        "seeing the same model state"
    )
    assert torch.allclose(info["solver_margins"], info["outer_margins"], atol=1e-4)


def test_drift_would_be_caught_if_modes_differed(model, batch):
    """
    Guard on the guard: with BN in train mode the margins genuinely move, so a
    near-zero drift in the test above is evidence and not a tautology.
    """
    x, y = batch
    model.eval()
    with torch.no_grad():
        eval_m = margin(model(x), y)
    model.train()
    with torch.no_grad():
        train_m = margin(model(x), y)
    assert (eval_m - train_m).abs().mean() > 1e-4


# ------------------------------------------------------------------
#                    eval() must not break learning
# ------------------------------------------------------------------

def test_backward_reaches_parameters_including_bn_affine(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _, _ = _call(model, batch)
    loss.backward()

    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is not None for g in grads.values()), "some parameter got no grad"
    assert any(g.abs().sum() > 0 for g in grads.values()), "all grads are zero"
    # eval() freezes BN's running stats, not its learnable affine parameters.
    assert grads["bn.weight"].abs().sum() > 0
    assert grads["bn.bias"].abs().sum() > 0


def test_logits_in_info_are_detached(model, batch):
    loss, _, info = _call(model, batch)
    assert loss.requires_grad, "loss must stay attached to the graph"
    for k, v in info.items():
        if torch.is_tensor(v):
            assert not v.requires_grad, f"info[{k!r}] still requires grad"
    assert info["logits_adv"].grad_fn is None


# ------------------------------------------------------------------
#                      Empty-valid-set behaviour
# ------------------------------------------------------------------

def test_no_valid_candidate_gives_backpropagatable_zero(model, batch):
    """An unreachable target level leaves every row invalid."""
    x, y = batch
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _, info = pos_geo_loss(model, x, y, None, t=1e6, **SOLVER, tol=1e-9)

    assert info["valid_sample_rate"].item() == 0.0
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()          # must not raise
    assert model.training is True


def test_loss_is_finite_across_all_mode_combinations(model, batch):
    for wm in WEIGHT_MODES:
        for gm in GEOMETRY_MODES:
            loss, _, info = _call(model, batch, weight_mode=wm, geometry_mode=gm)
            assert torch.isfinite(loss), f"{wm}/{gm} produced {loss}"

            w, valid = info["weights"], info["valid_mask"]
            has_valid = info["has_valid"]
            assert (w[~valid] == 0).all(), f"{wm}/{gm} weighted an invalid position"
            assert torch.allclose(w.sum(dim=1)[has_valid],
                                  torch.ones(int(has_valid.sum())), atol=1e-5)
            assert (w.sum(dim=1)[~has_valid] == 0).all()
