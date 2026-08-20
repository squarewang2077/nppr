# tests/test_pos_geo_loss_target_and_scale.py
#
# Requirements:
#   torch >= 2.0, pytest
#
# Run with:
#   PYTHONPATH=. pytest tests/test_pos_geo_loss_target_and_scale.py -q
#
# Regression tests for the divergence fix. Two things are pinned here:
#
#   * how the target level t is chosen (fixed vs adaptive), including that the
#     adaptive clean forward does not disturb BatchNorm or the model mode;
#   * that the gradient scale does NOT track 1 / valid_sample_rate. That
#     coupling is what closed the feedback loop — valid rate down, gradients
#     up, margins blow up, valid rate down — and took |margin| from 0.36 to 218
#     over 30 steps. test_gradient_scale_is_independent_of_valid_rate is the
#     one that would catch it coming back.

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pos_geo_loss import T_MODES, margin, pos_geo_loss

SOLVER = dict(num_starts=4, num_steps=5, epsilon=8 / 255)

# A configuration that actually puts perturbations on the level set for this
# tiny random net. The default SOLVER above gives valid_rate = 0 here, which
# would silently skip the gradient-scale tests — exactly the failure mode they
# exist to catch — so those tests use this and assert the rate is non-zero
# rather than skipping.
LIVE = dict(num_starts=4, num_steps=20, epsilon=8 / 255, tol=0.2,
            t_mode="adaptive")


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return self.head(h.mean(dim=(2, 3)))


@pytest.fixture
def model():
    torch.manual_seed(0)
    net = BNNet()
    net.train()
    with torch.no_grad():                      # move BN stats off their defaults
        net(torch.rand(16, 3, 8, 8))
    return net


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(8, 3, 8, 8), torch.randint(0, 4, (8,))


def _clean_margins(model, x, y):
    was = model.training
    model.eval()
    with torch.no_grad():
        m = margin(model(x), y)
    model.train(was)
    return m


# ------------------------------------------------------------------
#                        How t is chosen
# ------------------------------------------------------------------

def test_fixed_mode_uses_t_verbatim(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.37, **SOLVER, t_mode="fixed")
    assert info["t_mode"] == "fixed"
    assert info["t_used"] == pytest.approx(0.37)


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_adaptive_mode_uses_the_clean_margin_quantile(model, batch, q):
    x, y = batch
    expected = _clean_margins(model, x, y).quantile(q).item()

    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                              t_mode="adaptive", t_quantile=q)
    assert info["t_used"] == pytest.approx(expected, abs=1e-5)


def test_adaptive_mode_ignores_the_passed_t(model, batch):
    x, y = batch
    a = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode="adaptive")[2]
    b = pos_geo_loss(model, x, y, None, t=999.0, **SOLVER, t_mode="adaptive")[2]
    assert a["t_used"] == pytest.approx(b["t_used"], abs=1e-6)


def test_adaptive_t_tracks_a_shifting_margin_distribution(model, batch):
    """The point of adaptive mode: t follows the model instead of staying put."""
    x, y = batch
    before = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode="adaptive")[2]

    with torch.no_grad():                      # shift every logit's scale
        model.head.weight.mul_(5.0)
        model.head.bias.mul_(5.0)
    after = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode="adaptive")[2]

    assert abs(after["t_used"]) > abs(before["t_used"]) * 2


@pytest.mark.parametrize("start_training", [True, False])
def test_adaptive_clean_forward_leaves_bn_and_mode_alone(model, batch,
                                                         start_training):
    x, y = batch
    model.train(start_training)
    before = [(b.running_mean.clone(), b.running_var.clone())
              for b in model.modules() if isinstance(b, nn.BatchNorm2d)]

    pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode="adaptive")

    after = [(b.running_mean, b.running_var)
             for b in model.modules() if isinstance(b, nn.BatchNorm2d)]
    assert model.training is start_training
    for (m0, v0), (m1, v1) in zip(before, after):
        assert torch.equal(m0, m1) and torch.equal(v0, v1)


def test_rejects_bad_t_arguments(model, batch):
    x, y = batch
    with pytest.raises(ValueError, match="t_mode must be one of"):
        pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode="nope")
    with pytest.raises(ValueError, match="t_quantile must be in"):
        pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_quantile=1.5)


@pytest.mark.parametrize("mode", T_MODES)
def test_both_modes_run_and_report_t(model, batch, mode):
    x, y = batch
    loss, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_mode=mode)
    assert torch.isfinite(loss)
    assert isinstance(info["t_used"], float)
    assert info["t_mode"] == mode


# ------------------------------------------------------------------
#                    Loss: optimise mean, report both
# ------------------------------------------------------------------

def test_loss_is_the_batch_mean(model, batch):
    x, y = batch
    loss, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER)

    per_sample = (info["weights"] * info["ce"]).sum(dim=1)
    assert loss.item() == pytest.approx(per_sample.mean().item(), abs=1e-6)
    assert info["loss_batch_mean"].item() == pytest.approx(loss.item(), abs=1e-6)


def test_valid_only_is_reported_and_never_smaller(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER)
    # Same numerator, a denominator that is <= the batch size.
    assert info["loss_valid_only"].item() >= info["loss_batch_mean"].item() - 1e-6


def test_reported_metrics_are_detached_scalars(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER)
    for k in ("loss_batch_mean", "loss_valid_only"):
        assert not info[k].requires_grad
        assert info[k].dim() == 0


def test_empty_batch_gives_zero_for_both_and_still_backprops(model, batch):
    x, y = batch
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _, info = pos_geo_loss(model, x, y, None, t=1e6, **SOLVER, tol=1e-9)

    assert info["valid_sample_rate"].item() == 0.0
    assert info["loss_batch_mean"].item() == 0.0
    assert info["loss_valid_only"].item() == 0.0
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()


# ------------------------------------------------------------------
#          The regression this whole change exists to prevent
# ------------------------------------------------------------------

def _grad_norm(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.sqrt(sum((p.grad ** 2).sum()
                          for p in model.parameters() if p.grad is not None)).item()


@pytest.mark.parametrize("n_valid", [1, 2, 4])
def test_valid_only_denominator_would_amplify_the_gradient(model, batch, n_valid):
    """
    The two normalisations on *identical* data, so nothing but the denominator
    differs. Comparing different sample subsets instead would confound this
    with between-sample variation: gradients are vectors, and averaging more of
    them can shrink the norm through cancellation.

    The valid-only gradient is larger by exactly B / n_valid — that factor is
    the feedback loop, and it grows without bound as the valid rate falls.
    """
    x, y = batch
    B, N = x.shape[0], SOLVER["num_starts"]
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)
    model.eval()

    valid = torch.zeros(B, N, dtype=torch.bool)
    valid[:n_valid] = True
    w = valid.float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def per_sample():
        logits = model(x.unsqueeze(1).expand(B, N, *x.shape[1:])
                       .reshape(B * N, *x.shape[1:]))
        ce = nn.functional.cross_entropy(logits, y_rep, reduction="none").view(B, N)
        return (w * ce).sum(dim=1)

    g_batch = _grad_norm(model, per_sample().mean())
    g_valid = _grad_norm(model, per_sample().sum() / float(n_valid))

    assert g_valid / g_batch == pytest.approx(B / n_valid, rel=1e-3)


def test_pos_geo_loss_uses_the_unamplified_scale(model, batch):
    """
    The gradient pos_geo_loss produces must match the batch-mean gradient on
    the same weights, not the valid-only one.

    Everything is derived from a *single* call: the solver starts from random
    perturbations, so calling it twice would give different deltas and the
    comparison would drift by a fraction of a percent for reasons that have
    nothing to do with the scale.
    """
    x, y = batch
    model.eval()
    loss, x_adv, info = pos_geo_loss(model, x, y, None, t=0.0, **LIVE)

    rate = info["valid_sample_rate"].item()
    assert 0.0 < rate < 1.0, (
        f"fixture produced valid_sample_rate={rate}; this test is meaningless "
        "unless some but not all samples are valid"
    )

    B, N = x_adv.shape[:2]
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)
    w = info["weights"]

    def per_sample():
        logits = model(x_adv.reshape(B * N, *x.shape[1:]))
        ce = nn.functional.cross_entropy(logits, y_rep, reduction="none").view(B, N)
        return (w * ce).sum(dim=1)

    got = _grad_norm(model, loss)
    ref_batch = _grad_norm(model, per_sample().mean())
    n_v = max(round(rate * B), 1)
    ref_valid = _grad_norm(model, per_sample().sum() / float(n_v))

    assert ref_batch > 0.0
    assert got == pytest.approx(ref_batch, rel=1e-4), (
        f"gradient scale is {got / ref_batch:.3f}x the batch mean "
        f"(valid-only would be {ref_valid / ref_batch:.2f}x) — the "
        "1/valid_sample_rate coupling is back"
    )
    # And the alternative really is different, so the assertion has teeth.
    assert ref_valid / ref_batch == pytest.approx(B / n_v, rel=1e-3)


def test_loss_ratio_exposes_the_amplification(model, batch):
    """loss_valid_only / loss_batch_mean is the factor a valid-only objective
    would have multiplied the gradient by, so it is worth being able to read."""
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **LIVE)
    rate = info["valid_sample_rate"].item()
    assert rate > 0.0 and info["loss_batch_mean"].item() > 0.0, (
        "fixture produced no valid samples; the ratio would be vacuous"
    )

    ratio = info["loss_valid_only"].item() / info["loss_batch_mean"].item()
    assert ratio == pytest.approx(1.0 / rate, rel=1e-4)


# ------------------------------------------------------------------
#                       Solver selection
# ------------------------------------------------------------------

from src.pos_geo_loss import (SOLVERS, _DEFAULT_STEP_SIZE, margin as _margin,
                              solve_level_perturbations)


@pytest.mark.parametrize("solver", SOLVERS)
def test_both_solvers_run_and_stay_feasible(model, batch, solver):
    x, y = batch
    sol = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255,
                                    num_starts=4, num_steps=5, solver=solver)
    d = sol.delta
    assert d.flatten(2).abs().amax(-1).max() <= 8/255 + 1e-6
    xp = x.unsqueeze(1) + d
    assert xp.min() >= -1e-6 and xp.max() <= 1 + 1e-6
    assert torch.isfinite(sol.margins).all()


@pytest.mark.parametrize("solver", SOLVERS)
def test_step_size_defaults_per_solver(model, batch, solver):
    """step_size=None must resolve to the solver's own scale, not a shared one."""
    x, y = batch
    torch.manual_seed(0)
    auto = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255,
                                     num_starts=4, num_steps=5, solver=solver)
    torch.manual_seed(0)
    explicit = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255,
                                         num_starts=4, num_steps=5, solver=solver,
                                         step_size=_DEFAULT_STEP_SIZE[solver])
    assert torch.allclose(auto.delta, explicit.delta, atol=1e-6)


def test_solver_defaults_differ():
    assert _DEFAULT_STEP_SIZE["newton"] != _DEFAULT_STEP_SIZE["energy"]


def test_rejects_unknown_solver(model, batch):
    x, y = batch
    with pytest.raises(ValueError, match="solver must be one of"):
        solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255,
                                  num_steps=2, solver="bogus")


def test_newton_ignores_psi_alpha(model, batch):
    """alpha shapes the energy penalty; the Newton branch must not read it."""
    x, y = batch
    torch.manual_seed(0)
    a = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="newton", alpha=10.0)
    torch.manual_seed(0)
    b = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="newton", alpha=0.01)
    assert torch.allclose(a.delta, b.delta, atol=1e-6)

    # ... whereas the energy branch must be sensitive to it.
    torch.manual_seed(0)
    c = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="energy", alpha=10.0)
    torch.manual_seed(0)
    d = solve_level_perturbations(model, x, y, t=0.0, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="energy", alpha=0.01)
    assert not torch.allclose(c.delta, d.delta, atol=1e-6)


def test_newton_step_is_two_sided(model, batch):
    """
    The Newton step must move m toward t from either side — that symmetry is
    the reason symmetric_softplus is not needed here. One step, no anchor.
    """
    x, y = batch
    model.eval()
    with torch.no_grad():
        m0 = _margin(model(x), y)
    # Target above every clean margin, and below every clean margin.
    for t, expect in ((m0.max().item() + 0.5, "up"), (m0.min().item() - 0.5, "down")):
        sol = solve_level_perturbations(model, x, y, t=t, epsilon=8/255,
                                        num_starts=1, num_steps=1,
                                        solver="newton", anchor_lambda=0.0)
        moved = sol.margins.mean().item() - m0.mean().item()
        if expect == "up":
            assert moved > 0, f"target above, margins moved {moved:+.4f}"
        else:
            assert moved < 0, f"target below, margins moved {moved:+.4f}"


def test_solver_is_reported_in_info(model, batch):
    x, y = batch
    for solver in SOLVERS:
        _, _, info = pos_geo_loss(model, x, y, None, t=0.0, epsilon=8/255,
                                  num_starts=4, num_steps=5, solver=solver)
        assert info["solver"] == solver


# ------------------------------------------------------------------
#                     Tolerance mode
# ------------------------------------------------------------------

from src.pos_geo_loss import TOL_MODES, _DEFAULT_TOL


def test_absolute_mode_uses_tol_verbatim(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                              tol_mode="absolute", tol=0.11)
    assert info["tol_mode"] == "absolute"
    assert info["tol_used"] == pytest.approx(0.11)


def test_relative_mode_scales_by_the_clean_margin_iqr(model, batch):
    x, y = batch
    m = _clean_margins(model, x, y)
    iqr = (m.quantile(0.75) - m.quantile(0.25)).item()

    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                              tol_mode="relative", tol=0.02)
    assert info["tol_used"] == pytest.approx(0.02 * iqr, rel=1e-5)


def test_relative_tol_tracks_a_widening_margin_distribution(model, batch):
    """
    The whole point: as training widens the margins, an absolute tol silently
    tightens in relative terms while a relative one keeps pace.
    """
    x, y = batch
    before = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                          tol_mode="relative", tol=0.02)[2]["tol_used"]
    with torch.no_grad():                       # widen every margin 5x
        model.head.weight.mul_(5.0)
        model.head.bias.mul_(5.0)
    after = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                         tol_mode="relative", tol=0.02)[2]["tol_used"]

    assert after == pytest.approx(before * 5.0, rel=0.05)

    # An absolute tol does not move — that is the failure being fixed.
    abs_tol = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                           tol_mode="absolute", tol=0.05)[2]["tol_used"]
    assert abs_tol == pytest.approx(0.05)


@pytest.mark.parametrize("mode", TOL_MODES)
def test_tol_defaults_per_mode(model, batch, mode):
    """tol=None must resolve to the mode's own scale, not a shared one."""
    x, y = batch
    auto = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, tol_mode=mode)[2]
    explicit = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, tol_mode=mode,
                            tol=_DEFAULT_TOL[mode])[2]
    assert auto["tol_used"] == pytest.approx(explicit["tol_used"], rel=1e-6)


def test_tol_defaults_differ_between_modes():
    assert _DEFAULT_TOL["absolute"] != _DEFAULT_TOL["relative"]


def test_rejects_unknown_tol_mode(model, batch):
    x, y = batch
    with pytest.raises(ValueError, match="tol_mode must be one of"):
        pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, tol_mode="bogus")


def test_tol_used_actually_gates_validity(model, batch):
    """tol_used must be the number the valid mask was computed against."""
    x, y = batch
    # LIVE already carries tol=0.2; passing it again would collide.
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **LIVE, tol_mode="absolute")
    assert info["tol_used"] == pytest.approx(LIVE["tol"])
    gap = (info["solver_margins"] - info["t_used"]).abs()
    assert torch.equal(info["valid_mask"], gap <= info["tol_used"])


# ------------------------------------------------------------------
#            anchor_lambda per solver, and the AE premise
# ------------------------------------------------------------------

from src.pos_geo_loss import _DEFAULT_ANCHOR_LAMBDA


@pytest.mark.parametrize("solver", SOLVERS)
def test_anchor_lambda_defaults_per_solver(model, batch, solver):
    """anchor_lambda=None must resolve to the solver's own value."""
    x, y = batch
    torch.manual_seed(0)
    auto = solve_level_perturbations(model, x, y, t=-0.5, epsilon=8/255,
                                     num_starts=4, num_steps=5, solver=solver)
    torch.manual_seed(0)
    explicit = solve_level_perturbations(model, x, y, t=-0.5, epsilon=8/255,
                                         num_starts=4, num_steps=5, solver=solver,
                                         anchor_lambda=_DEFAULT_ANCHOR_LAMBDA[solver])
    assert torch.allclose(auto.delta, explicit.delta, atol=1e-6)


def test_anchor_defaults_differ_and_newton_is_zero():
    assert _DEFAULT_ANCHOR_LAMBDA["newton"] == 0.0
    assert _DEFAULT_ANCHOR_LAMBDA["energy"] != _DEFAULT_ANCHOR_LAMBDA["newton"]


def test_anchor_still_overridable(model, batch):
    """The old behaviour must remain reachable for reproduction."""
    x, y = batch
    torch.manual_seed(0)
    a = solve_level_perturbations(model, x, y, t=-0.5, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="newton", anchor_lambda=0.0)
    torch.manual_seed(0)
    b = solve_level_perturbations(model, x, y, t=-0.5, epsilon=8/255, num_starts=4,
                                  num_steps=5, solver="newton", anchor_lambda=0.5)
    assert not torch.allclose(a.delta, b.delta, atol=1e-6)


class LinearNet(nn.Module):
    """
    Linear model: the margin is linear in delta, so a Newton step is exact and
    the solver reaches machine precision. That is what the anchor test needs —
    the anchor only overtakes the Newton step once the step has shrunk, so any
    setup where the solver stalls early (a tiny conv net, or an epsilon-ball
    tight enough that the projection binds) cannot show the effect at all.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_anchor_free_newton_converges_much_closer():
    """
    The regression this fix exists to prevent: a fixed anchor eventually
    dominates the shrinking Newton step and holds the solver off the level set.

    epsilon is deliberately loose (0.1, not 8/255). Under a tight ball the
    projection caps the step before the anchor ever matters and both settings
    stall at the same place — measured, 0.24426 vs 0.24436, a 0.004% gap that
    would make this test pass for the wrong reason.
    """
    model = LinearNet().eval()
    torch.manual_seed(0)
    x, y = torch.rand(8, 3, 8, 8), torch.randint(0, 4, (8,))
    kw = dict(t=-0.5, epsilon=0.1, num_starts=8, num_steps=30, solver="newton")

    torch.manual_seed(0)
    free = solve_level_perturbations(model, x, y, anchor_lambda=0.0, **kw)
    torch.manual_seed(0)
    held = solve_level_perturbations(model, x, y, anchor_lambda=0.01, **kw)

    gap_free = (free.margins - kw["t"]).abs().median().item()
    gap_held = (held.margins - kw["t"]).abs().median().item()

    assert gap_free < 1e-5, f"anchor-free solver did not converge: {gap_free:.2e}"
    assert gap_held > 100 * gap_free, (
        f"anchor no longer degrades convergence ({gap_held:.2e} vs "
        f"{gap_free:.2e}) — either the default changed or the fixture stalls"
    )


# ------------------------------------------------------------------
#                         AE rate
# ------------------------------------------------------------------

def test_ae_rate_matches_the_sign_of_the_margins(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=-0.5, **LIVE)
    expected = (info["solver_margins"] < 0).float().mean()
    assert info["ae_rate"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_ae_rate_valid_covers_only_the_weighted_positions(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=-0.5, **LIVE)
    valid = info["valid_mask"]
    if valid.sum() == 0:
        pytest.fail("fixture produced no valid positions; the test is vacuous")

    expected = (info["solver_margins"][valid] < 0).float().mean()
    assert info["ae_rate_valid"].item() == pytest.approx(expected.item(), abs=1e-6)
    # Everything carrying weight must be an AE at a sufficiently negative t.
    assert (info["weights"][~(info["solver_margins"] < 0)] == 0).all() or \
           info["ae_rate_valid"].item() == pytest.approx(1.0)


def test_negative_t_makes_every_weighted_landing_an_ae(model, batch):
    """t below -tol puts the whole tolerance band on the misclassified side."""
    x, y = batch
    kw = dict(num_starts=8, num_steps=20, epsilon=8/255, tol=0.05, solver="newton")
    _, _, info = pos_geo_loss(model, x, y, None, t=-0.5, **kw)
    assert info["valid_position_rate"] > 0, "no valid positions; test is vacuous"
    assert info["ae_rate_valid"].item() == pytest.approx(1.0)


def test_positive_t_gives_non_ae_landings():
    """
    The failure the AE diagnostic exists to expose. A positive level sits on
    the correctly-classified side, so the landings are not adversarial examples
    however well the solver converges.

    Note this is about the sign of t, not about t + tol straddling zero:
    measured, t=-0.30 with tol=0.4 still gave ae_rate_valid = 1.0, because the
    landings cluster near t rather than filling the tolerance band.
    """
    model = LinearNet().eval()
    torch.manual_seed(0)
    x, y = torch.rand(8, 3, 8, 8), torch.randint(0, 4, (8,))
    kw = dict(num_starts=8, num_steps=30, epsilon=8 / 255, tol=0.05, solver="newton")

    _, _, neg = pos_geo_loss(model, x, y, None, t=-0.1, **kw)
    _, _, pos = pos_geo_loss(model, x, y, None, t=+0.1, **kw)

    assert neg["valid_position_rate"] > 0 and pos["valid_position_rate"] > 0, \
        "fixture produced no valid positions; the comparison is vacuous"
    assert neg["ae_rate_valid"].item() == pytest.approx(1.0)
    assert pos["ae_rate_valid"].item() == pytest.approx(0.0)


def test_ae_rates_are_detached_scalars(model, batch):
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=-0.5, **LIVE)
    for k in ("ae_rate", "ae_rate_valid"):
        assert not info[k].requires_grad and info[k].dim() == 0
        assert 0.0 <= info[k].item() <= 1.0


# ------------------------------------------------------------------
#                  t_mode = "reachable"
# ------------------------------------------------------------------

from src.pos_geo_loss import reachable_margin_floor


def test_floor_is_below_the_clean_margin(model, batch):
    """The ball can only push the margin down from where it starts."""
    x, y = batch
    model.eval()
    clean = _clean_margins(model, x, y)
    floor = reachable_margin_floor(model, x, y, epsilon=8/255)

    assert floor.shape == clean.shape
    assert (floor <= clean + 1e-4).all(), "floor above the clean margin"
    assert torch.isfinite(floor).all()


def test_floor_drops_as_the_ball_grows(model, batch):
    x, y = batch
    model.eval()
    tight = reachable_margin_floor(model, x, y, epsilon=1/255)
    loose = reachable_margin_floor(model, x, y, epsilon=32/255)
    assert loose.median() < tight.median()


def test_sign_pgd_beats_plain_gradient_descent(model, batch):
    """
    Why the floor uses sign-PGD. Plain gradient descent under-estimates how far
    an L-inf ball reaches — measured on a real model, -0.42 where sign-PGD
    found -1.26 — which would make the floor look tighter than it is.
    """
    x, y = batch
    model.eval()
    eps = 8/255
    sign_floor = reachable_margin_floor(model, x, y, epsilon=eps, num_steps=10)

    delta = torch.zeros_like(x)
    for _ in range(10):
        delta.requires_grad_(True)
        g = torch.autograd.grad(margin(model(x + delta), y).sum(), delta)[0]
        delta = ((x + (delta.detach() - 0.01 * g).clamp(-eps, eps)).clamp(0, 1) - x)
    with torch.no_grad():
        plain_floor = margin(model(x + delta), y)

    assert sign_floor.median() <= plain_floor.median()


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
def test_reachable_interpolates_between_clean_margin_and_floor(model, batch, frac):
    """
    t = m_clean + frac * (floor - m_clean), not frac * floor. Scaling the floor
    alone assumes the margin starts at zero; when a sample already sits at
    m_clean = -0.40 and the ball only reaches -0.41, frac * floor lands at
    -0.20 — above the start, and the ball only pushes downward.
    """
    x, y = batch
    model.eval()
    clean = _clean_margins(model, x, y)
    floor = reachable_margin_floor(model, x, y, epsilon=SOLVER["epsilon"])
    expected = (clean + frac * (floor - clean)).median().item()

    _, _, info = pos_geo_loss(model, x, y, None, t=99.0, **SOLVER,
                              t_mode="reachable", t_frac=max(frac, 1e-6))
    assert info["t_mode"] == "reachable"
    # t is derived, so the passed-in 99.0 is irrelevant.
    assert info["t_used"] == pytest.approx(expected, abs=0.05)


def test_reachable_t_always_lies_within_the_attainable_band(model, batch):
    """Whatever t_frac, the level must sit between the floor and the start."""
    x, y = batch
    model.eval()
    clean = _clean_margins(model, x, y)
    floor = reachable_margin_floor(model, x, y, epsilon=SOLVER["epsilon"])

    for frac in (0.1, 0.5, 0.9, 1.0):
        _, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                                  t_mode="reachable", t_frac=frac)
        t = info["t_used"]
        assert floor.median().item() - 1e-4 <= t <= clean.median().item() + 1e-4, (
            f"t_frac={frac} put t={t:.4f} outside [{floor.median():.4f}, "
            f"{clean.median():.4f}]"
        )


@pytest.mark.parametrize("frac", [0.3, 0.5, 0.9])
def test_reachable_t_is_negative_so_landings_are_aes(model, batch, frac):
    """The floor is below zero wherever an AE exists, so t inherits the sign."""
    x, y = batch
    _, _, info = pos_geo_loss(model, x, y, None, t=0.0, num_starts=8,
                              num_steps=20, epsilon=8/255, tol=0.05,
                              solver="newton", t_mode="reachable", t_frac=frac)
    assert info["t_used"] < 0


def test_reachable_t_is_per_sample(model, batch):
    """
    A single batch level leaves the hard samples outside the ball; the whole
    point is that each sample gets its own.
    """
    x, y = batch
    floor = reachable_margin_floor(model, x, y, epsilon=8/255)
    assert floor.std() > 0, "fixture has a degenerate floor; test is vacuous"

    # Broadcasting a (B, 1) t against (B, N) margins must work end to end.
    loss, _, info = pos_geo_loss(model, x, y, None, t=0.0, **SOLVER,
                                 t_mode="reachable")
    assert torch.isfinite(loss)
    assert info["valid_mask"].shape == info["solver_margins"].shape


def test_reachable_holds_validity_when_the_floor_rises(model, batch):
    """
    The regression this mode exists to prevent. Harden the model so the floor
    rises toward zero; a fixed t is left outside the ball while reachable
    follows it down.
    """
    x, y = batch
    kw = dict(num_starts=8, num_steps=20, epsilon=8/255, tol=0.05, solver="newton")
    before = reachable_margin_floor(model, x, y, epsilon=8/255).median().item()

    with torch.no_grad():                     # shrink the input sensitivity
        model.conv.weight.mul_(0.1)
        model.conv.bias.mul_(0.1)
    after = reachable_margin_floor(model, x, y, epsilon=8/255).median().item()
    assert after > before, "fixture did not raise the floor; test is vacuous"

    _, _, reach = pos_geo_loss(model, x, y, None, t=0.0, **kw, t_mode="reachable")
    _, _, fixed = pos_geo_loss(model, x, y, None, t=before, **kw, t_mode="fixed")
    assert reach["valid_position_rate"] >= fixed["valid_position_rate"]


def test_rejects_bad_t_frac(model, batch):
    x, y = batch
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="t_frac must be in"):
            pos_geo_loss(model, x, y, None, t=0.0, **SOLVER, t_frac=bad)


def test_reachable_level_stays_non_positive_for_correct_samples():
    """
    The AE premise under t_mode="reachable". For a sample the model already
    gets right (m_clean > 0), interpolating from m_clean would put the level on
    the correctly-classified side — measured end to end, that held
    ae_rate_valid at 0.70-0.87 instead of 1.0. The start is clamped to 0.
    """
    model = LinearNet().eval()
    torch.manual_seed(0)
    x, y = torch.rand(8, 3, 8, 8), torch.randint(0, 4, (8,))
    # Bias the head so most clean margins come out positive.
    with torch.no_grad():
        logits = model(x)
        model.fc.bias.add_(
            torch.nn.functional.one_hot(y, 4).float().mean(0) * 20.0)
    clean = _clean_margins(model, x, y)
    assert (clean > 0).any(), "fixture produced no correctly-classified samples"

    for frac in (0.2, 0.5, 1.0):
        _, _, info = pos_geo_loss(model, x, y, None, t=0.0, num_starts=4,
                                  num_steps=10, epsilon=8/255, tol=0.05,
                                  solver="newton", t_mode="reachable", t_frac=frac)
        assert info["t_used"] <= 1e-6, (
            f"t_frac={frac} gave a positive level t={info['t_used']:.4f}; the "
            "landings would not be adversarial examples"
        )


def test_reachable_guarantees_every_weighted_landing_is_an_ae(model, batch):
    """
    The full AE guarantee under t_mode="reachable": the level is capped at
    -tol so the whole two-sided band stays below zero. Without that cap a
    sample whose floor sits just under zero contributes landings on the
    correct side — measured end to end, ae_rate_valid stalled at 0.86-0.94.
    """
    x, y = batch
    for frac in (0.3, 0.5, 1.0):
        _, _, info = pos_geo_loss(model, x, y, None, t=0.0, num_starts=8,
                                  num_steps=20, epsilon=8/255, tol=0.05,
                                  solver="newton", t_mode="reachable", t_frac=frac)
        if info["valid_position_rate"] == 0:
            continue                       # nothing reachable; nothing to check
        assert info["ae_rate_valid"].item() == pytest.approx(1.0), (
            f"t_frac={frac}: ae_rate_valid={info['ae_rate_valid'].item():.3f}"
        )
