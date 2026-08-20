# tests/test_pos_geo_weights.py - Contract tests for pos_geo_weights.
#
# Requirements:
#   torch >= 2.0, pytest
#
# Run with:
#   PYTHONPATH=. pytest tests/test_pos_geo_weights.py -q
#
# These pin the invariants the level-set experiment relies on: invalid
# positions never carry weight, rows with nothing valid stay zero rather than
# NaN, and sharp / flat are the power laws the coarea argument calls for
# (w ∝ g and w ∝ 1/g at tau=1) rather than an exponential heuristic.

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pos_geo_loss import WEIGHT_MODES, pos_geo_weights


def _mask(rows):
    return torch.tensor(rows, dtype=torch.bool)


# ------------------------------------------------------------------
#                        Validity and masking
# ------------------------------------------------------------------

@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_all_valid_rows_sum_to_one(mode):
    g = torch.tensor([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    dn = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    w = pos_geo_weights(g, dn, _mask([[True] * 3, [True] * 3]), mode=mode)

    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-6)
    assert (w >= 0).all()


def test_uniform_is_uniform_over_valid():
    g = torch.tensor([[1.0, 5.0, 9.0, 2.0]])
    dn = torch.ones(1, 4)
    w = pos_geo_weights(g, dn, _mask([[True] * 4]), mode="uniform")
    assert torch.allclose(w, torch.full((1, 4), 0.25), atol=1e-6)


@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_invalid_positions_get_exactly_zero(mode):
    g = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    dn = torch.tensor([[0.4, 0.1, 0.3, 0.2]])
    valid = _mask([[True, False, True, False]])
    w = pos_geo_weights(g, dn, valid, mode=mode)

    # Exact zero, not merely small — these must not leak gradient.
    assert (w[~valid] == 0).all()
    assert torch.allclose(w.sum(dim=1), torch.ones(1), atol=1e-6)


@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_row_with_no_valid_candidate_is_all_zero_and_finite(mode):
    g = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])
    dn = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    valid = _mask([[False, False, False], [True, True, False]])
    w = pos_geo_weights(g, dn, valid, mode=mode)

    assert torch.isfinite(w).all(), "empty row produced NaN/Inf"
    assert w[0].sum() == 0.0
    assert torch.allclose(w[1].sum(), torch.ones(()), atol=1e-6)


def test_every_row_empty_is_all_zero():
    g = torch.rand(3, 4) + 0.1
    w = pos_geo_weights(g, torch.rand(3, 4), torch.zeros(3, 4, dtype=torch.bool),
                        mode="sharp")
    assert torch.isfinite(w).all() and (w == 0).all()


# ------------------------------------------------------------------
#                    Sharp / flat are power laws
# ------------------------------------------------------------------

def test_sharp_is_proportional_to_g_at_tau_one():
    g = torch.tensor([[1.0, 2.0, 4.0]])
    w = pos_geo_weights(g, torch.ones(1, 3), _mask([[True] * 3]), mode="sharp", tau=1.0)
    assert torch.allclose(w, (g / g.sum()), atol=1e-6)


def test_flat_is_proportional_to_inverse_g_at_tau_one():
    g = torch.tensor([[1.0, 2.0, 4.0]])
    inv = 1.0 / g
    w = pos_geo_weights(g, torch.ones(1, 3), _mask([[True] * 3]), mode="flat", tau=1.0)
    assert torch.allclose(w, inv / inv.sum(), atol=1e-6)


@pytest.mark.parametrize("mode", ["sharp", "flat"])
@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_power_law_form(mode, tau):
    g = torch.tensor([[1.0, 2.0, 4.0, 7.0]])
    power = (1.0 / tau) if mode == "sharp" else (-1.0 / tau)
    expected = g ** power
    expected = expected / expected.sum()

    w = pos_geo_weights(g, torch.ones(1, 4), _mask([[True] * 4]), mode=mode, tau=tau)
    assert torch.allclose(w, expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["sharp", "flat"])
@pytest.mark.parametrize("c", [1e-4, 0.5, 3.0, 1e4])
def test_common_rescaling_leaves_weights_unchanged(mode, c):
    """A softmax over raw g would fail this; over log(g) it holds exactly."""
    g = torch.tensor([[1.0, 2.0, 4.0, 7.0]])
    valid = _mask([[True] * 4])
    base = pos_geo_weights(g, torch.ones(1, 4), valid, mode=mode)
    scaled = pos_geo_weights(g * c, torch.ones(1, 4), valid, mode=mode)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_large_tau_tends_to_uniform():
    g = torch.tensor([[1.0, 10.0, 100.0]])
    w = pos_geo_weights(g, torch.ones(1, 3), _mask([[True] * 3]), mode="flat", tau=1e6)
    assert torch.allclose(w, torch.full((1, 3), 1 / 3), atol=1e-4)


def test_zero_grad_score_is_floored_not_nan():
    g = torch.tensor([[0.0, 1.0, 2.0]])
    for mode in ("sharp", "flat"):
        w = pos_geo_weights(g, torch.ones(1, 3), _mask([[True] * 3]), mode=mode)
        assert torch.isfinite(w).all()
        assert torch.allclose(w.sum(), torch.ones(()), atol=1e-6)


# ------------------------------------------------------------------
#                         min_norm / max_norm
# ------------------------------------------------------------------

def test_min_and_max_norm_pick_the_right_valid_candidate():
    dn = torch.tensor([[0.5, 0.1, 0.9, 0.3]])
    g = torch.ones(1, 4)
    # The global min (0.1) and max (0.9) are both invalid, so they must be
    # skipped in favour of the best *valid* entries: 0.3 and 0.5.
    valid = _mask([[True, False, False, True]])

    w_min = pos_geo_weights(g, dn, valid, mode="min_norm")
    w_max = pos_geo_weights(g, dn, valid, mode="max_norm")

    assert w_min.argmax(dim=1).item() == 3 and torch.allclose(w_min[0, 3], torch.ones(()))
    assert w_max.argmax(dim=1).item() == 0 and torch.allclose(w_max[0, 0], torch.ones(()))
    assert (w_min[~valid] == 0).all() and (w_max[~valid] == 0).all()


def test_min_max_norm_are_one_hot():
    dn = torch.rand(5, 6)
    w = pos_geo_weights(torch.ones(5, 6), dn, torch.ones(5, 6, dtype=torch.bool),
                        mode="min_norm")
    assert ((w == 0) | (w == 1)).all()
    assert (w.sum(dim=1) == 1).all()


def test_min_norm_tie_breaks_to_lowest_index():
    """Documented behaviour: argmin keeps PyTorch's first-element default."""
    dn = torch.tensor([[0.2, 0.2, 0.5]])
    w = pos_geo_weights(torch.ones(1, 3), dn, _mask([[True] * 3]), mode="min_norm")
    assert w[0, 0] == 1.0


# ------------------------------------------------------------------
#                        Input validation
# ------------------------------------------------------------------

def test_rejects_bad_inputs():
    g, dn = torch.ones(2, 3), torch.ones(2, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="mode must be one of"):
        pos_geo_weights(g, dn, valid, mode="nope")
    with pytest.raises(ValueError, match="tau must be > 0"):
        pos_geo_weights(g, dn, valid, tau=0.0)
    with pytest.raises(TypeError, match="valid_mask must be bool"):
        pos_geo_weights(g, dn, torch.ones(2, 3))
    with pytest.raises(ValueError, match="must all be"):
        pos_geo_weights(g, torch.ones(2, 4), valid)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pos_geo_weights(-g, dn, valid)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pos_geo_weights(g * float("nan"), dn, valid)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pos_geo_weights(g * float("inf"), dn, valid)


# ------------------------------------------------------------------
#                        dtype / precision
# ------------------------------------------------------------------

@pytest.mark.parametrize("mode", WEIGHT_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes_stay_finite(mode, dtype):
    torch.manual_seed(0)
    g = (torch.rand(8, 5, dtype=dtype) * 1e-3)
    dn = torch.rand(8, 5, dtype=dtype)
    valid = torch.rand(8, 5) > 0.5
    w = pos_geo_weights(g, dn, valid, mode=mode)

    assert w.dtype == dtype
    assert torch.isfinite(w).all()
    has_valid = valid.any(dim=1)
    assert torch.allclose(w.sum(dim=1)[has_valid],
                          torch.ones(int(has_valid.sum()), dtype=dtype), atol=1e-5)
    assert (w.sum(dim=1)[~has_valid] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_autocast_stays_finite(mode):
    torch.manual_seed(0)
    g = torch.rand(8, 5, device="cuda") * 1e-4
    dn = torch.rand(8, 5, device="cuda")
    valid = torch.rand(8, 5, device="cuda") > 0.4

    with torch.autocast("cuda", dtype=torch.float16):
        w = pos_geo_weights(g, dn, valid, mode=mode)

    assert torch.isfinite(w).all()
    has_valid = valid.any(dim=1)
    assert torch.allclose(w.sum(dim=1)[has_valid],
                          torch.ones(int(has_valid.sum()), device="cuda"), atol=1e-3)


# ------------------------------------------------------------------
#              The coarea claim, stated as a test
# ------------------------------------------------------------------

def test_flat_reproduces_inverse_gradient_coarea_factor():
    """
    flat + grad_margin_l2 + tau=1 must give w ∝ 1 / ||grad m||_2, which is the
    factor the Euclidean coarea formula puts on the level-set surface measure.
    """
    grad_margin_l2 = torch.tensor([[0.5, 1.0, 2.0, 4.0]])
    w = pos_geo_weights(grad_margin_l2, torch.ones(1, 4), _mask([[True] * 4]),
                        mode="flat", tau=1.0)

    inv = 1.0 / grad_margin_l2
    assert torch.allclose(w, inv / inv.sum(), atol=1e-6)
    # Doubling the gradient must halve the relative weight.
    assert math.isclose((w[0, 0] / w[0, 1]).item(), 2.0, rel_tol=1e-5)
    assert math.isclose((w[0, 1] / w[0, 2]).item(), 2.0, rel_tol=1e-5)
