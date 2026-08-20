# tests/test_level_geometry.py - Contract tests for level_geometry / dispersion.
#
# Requirements:
#   torch >= 2.0, pytest
#
# Run with:
#   PYTHONPATH=. pytest tests/test_level_geometry.py -q
#
# These pin what the geometry statistics are allowed to count: only candidates
# that actually reached the level set and have a direction at all. eff_rank is
# a *directional* dimensionality, so the antipodal case (+v, -v) must give 1
# despite the two points being far apart — pair_l2_distance is what separates
# those two questions.

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pos_geo_loss import dispersion, level_geometry, pair_l2_distance


def _as_delta(rows):
    """(N, D) directions -> (1, N, 1, 1, D), the flattened dim inferred."""
    if isinstance(rows, (list, tuple)):
        rows = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in rows])
    t = torch.as_tensor(rows, dtype=torch.float32)
    n, d = t.shape
    return t.view(1, n, 1, 1, d)


def _geo(delta, valid=None):
    n = delta.shape[1]
    margins = torch.zeros(1, n)
    vm = None if valid is None else torch.tensor([valid], dtype=torch.bool)
    g = level_geometry(delta, margins, valid_mask=vm)
    return g, dispersion(g["pair_cos"], g["active_mask"])


# ------------------------------------------------------------------
#                    eff_rank counts directions
# ------------------------------------------------------------------

def test_orthogonal_directions_give_eff_rank_n():
    g, d = _geo(_as_delta(torch.eye(4)))
    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 4.0
    assert pytest.approx(d["anisotropy"].item(), abs=1e-4) == 0.25


def test_identical_directions_give_eff_rank_one():
    v = [1.0, 2.0, 3.0, 4.0]
    g, d = _geo(_as_delta([v, v, v, v]))
    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 1.0
    assert pytest.approx(d["anisotropy"].item(), abs=1e-4) == 1.0


def test_scaled_but_collinear_directions_give_eff_rank_one():
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    g, d = _geo(_as_delta([v, 2 * v, 7 * v]))
    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 1.0


def test_antipodal_directions_span_one_dimension():
    """+v and -v are far apart in space but span a single direction."""
    v = torch.tensor([1.0, 1.0, 0.0, 0.0])
    g, d = _geo(_as_delta([v, -v]))

    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 1.0
    assert pytest.approx(g["pair_cos"][0, 0, 1].item(), abs=1e-5) == -1.0
    # ... while the positional statistic sees them as maximally separated.
    dist = pair_l2_distance(g["delta_l2_norm"], g["pair_cos"])
    assert pytest.approx(dist[0, 0, 1].item(), rel=1e-5) == 2 * v.norm().item()


def test_eff_rank_is_bounded_by_active_count():
    torch.manual_seed(0)
    delta = torch.randn(6, 5, 3, 4, 4)
    margins = torch.zeros(6, 5)
    valid = torch.rand(6, 5) > 0.4
    g = level_geometry(delta, margins, valid_mask=valid)
    d = dispersion(g["pair_cos"], g["active_mask"])

    active = g["active_count"].float()
    has = active > 0
    assert (d["eff_rank"][has] >= 1.0 - 1e-4).all()
    assert (d["eff_rank"][has] <= active[has] + 1e-4).all()


# ------------------------------------------------------------------
#                     Masking: valid and non-zero
# ------------------------------------------------------------------

def test_invalid_candidates_do_not_affect_the_result():
    e = torch.eye(4)
    # Two orthogonal valid directions, two invalid ones that would raise the
    # rank to 4 if they were counted.
    delta = _as_delta([e[0], e[1], e[2], e[3]])
    _, d_all = _geo(delta)
    _, d_two = _geo(delta, valid=[True, True, False, False])

    assert pytest.approx(d_all["eff_rank"].item(), abs=1e-4) == 4.0
    assert pytest.approx(d_two["eff_rank"].item(), abs=1e-4) == 2.0
    assert d_two["active_count"].item() == 2


def test_invalid_candidates_occupy_zero_rows_and_columns():
    g, _ = _geo(_as_delta(torch.eye(4)), valid=[True, False, True, True])
    pc = g["pair_cos"][0]
    assert (pc[1, :] == 0).all() and (pc[:, 1] == 0).all()
    assert pc[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_zero_perturbation_is_excluded():
    e = torch.eye(4)
    delta = _as_delta([e[0], torch.zeros(4), e[1]])
    g, d = _geo(delta)

    assert g["active_mask"].tolist() == [[True, False, True]]
    assert g["active_count"].item() == 2
    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 2.0
    # A zero vector must not be handed a manufactured unit direction.
    assert (g["pair_cos"][0, 1, :] == 0).all()


def test_zero_perturbation_still_reports_its_norm():
    delta = _as_delta([[1.0, 0, 0, 0], [0.0, 0, 0, 0]])
    g, _ = _geo(delta)
    assert g["delta_l2_norm"][0, 1].item() == 0.0


# ------------------------------------------------------------------
#                    Empty active set is all zeros
# ------------------------------------------------------------------

def test_no_active_candidate_gives_exact_zeros_no_nan():
    g, d = _geo(_as_delta(torch.eye(3)), valid=[False, False, False])

    assert g["active_count"].item() == 0
    assert (g["pair_cos"] == 0).all()
    for k, v in d.items():
        assert torch.isfinite(v).all(), f"{k} not finite"
        assert v.item() == 0.0, f"{k} should be exactly 0, got {v.item()}"


def test_all_zero_perturbations_give_zeros():
    g, d = _geo(_as_delta(torch.zeros(3, 4)))
    assert g["active_count"].item() == 0
    assert d["eff_rank"].item() == 0.0 and torch.isfinite(d["eff_rank"]).all()


def test_mixed_rows_empty_and_populated():
    delta = torch.zeros(2, 3, 1, 1, 4)
    delta[1] = torch.eye(3, 4).view(3, 1, 1, 4)
    g = level_geometry(delta, torch.zeros(2, 3))
    d = dispersion(g["pair_cos"], g["active_mask"])

    assert d["eff_rank"][0].item() == 0.0
    assert pytest.approx(d["eff_rank"][1].item(), abs=1e-4) == 3.0
    assert torch.isfinite(d["eff_rank"]).all()


# ------------------------------------------------------------------
#                    Numerical properties
# ------------------------------------------------------------------

def test_gram_matrix_is_exactly_symmetric():
    torch.manual_seed(0)
    g = level_geometry(torch.randn(4, 6, 3, 8, 8), torch.zeros(4, 6))
    pc = g["pair_cos"]
    assert torch.equal(pc, pc.transpose(1, 2)), "Gram matrix is not symmetric"


def test_cosines_are_within_range_and_diagonal_is_one():
    torch.manual_seed(0)
    g = level_geometry(torch.randn(3, 5, 3, 4, 4), torch.zeros(3, 5))
    pc = g["pair_cos"]
    assert (pc >= -1.0).all() and (pc <= 1.0).all()
    diag = torch.diagonal(pc, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_eigenvalues_are_non_negative():
    torch.manual_seed(0)
    g = level_geometry(torch.randn(4, 6, 3, 4, 4), torch.zeros(4, 6))
    d = dispersion(g["pair_cos"], g["active_mask"])
    assert (d["eig_min"] >= 0).all() and (d["eig_max"] >= 0).all()
    assert (d["eig_max"] >= d["eig_min"] - 1e-5).all()


def test_sigma_is_the_square_root_of_eig():
    torch.manual_seed(0)
    g = level_geometry(torch.randn(3, 4, 3, 4, 4), torch.zeros(3, 4))
    d = dispersion(g["pair_cos"], g["active_mask"])
    assert torch.allclose(d["sigma_max"], d["eig_max"].sqrt(), atol=1e-6)
    assert torch.allclose(d["sigma_min"], d["eig_min"].sqrt(), atol=1e-6)


def test_eig_min_is_zero_whenever_anything_is_inactive():
    """
    Documented limitation: zero padding forces a zero eigenvalue, so eig_min
    says nothing about the conditioning of the active subspace.
    """
    g, d = _geo(_as_delta(torch.eye(4)), valid=[True, True, True, False])
    assert d["eig_min"].item() == pytest.approx(0.0, abs=1e-6)
    assert pytest.approx(d["eff_rank"].item(), abs=1e-4) == 3.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_input_is_computed_in_float32(dtype):
    torch.manual_seed(0)
    delta = torch.randn(3, 5, 3, 8, 8).to(dtype)
    g = level_geometry(delta, torch.zeros(3, 5))
    d = dispersion(g["pair_cos"], g["active_mask"])

    assert g["pair_cos"].dtype == torch.float32
    assert g["delta_l2_norm"].dtype == torch.float32
    assert torch.isfinite(g["pair_cos"]).all()
    for k, v in d.items():
        assert torch.isfinite(v).all(), f"{k} not finite in {dtype}"


def test_float16_orthogonal_case_stays_accurate():
    """The reason for the float32 cast: half-precision Gram accumulation drifts."""
    delta = _as_delta(torch.eye(4)).half()
    _, d = _geo(delta)
    assert pytest.approx(d["eff_rank"].item(), abs=1e-3) == 4.0


# ------------------------------------------------------------------
#                        Detachment
# ------------------------------------------------------------------

def test_outputs_are_detached():
    delta = torch.randn(2, 3, 3, 4, 4, requires_grad=True)
    margins = torch.zeros(2, 3, requires_grad=True)
    g = level_geometry(delta, margins)
    d = dispersion(g["pair_cos"], g["active_mask"])

    for k, v in {**g, **d}.items():
        if torch.is_tensor(v) and v.is_floating_point():
            assert not v.requires_grad, f"{k} still requires grad"


# ------------------------------------------------------------------
#                      Degenerate shapes
# ------------------------------------------------------------------

def test_single_candidate_gives_eff_rank_one():
    """N=1 is the ablation's control run, so it has to survive the geometry.

    With one perturbation the Gram matrix is (B, 1, 1) and every dispersion
    statistic is degenerate — the risk is a 0/0 producing NaN rather than the
    single direction the run actually has.
    """
    g, d = _geo(_as_delta([[1.0, 2.0, -3.0, 0.5]]))

    assert g["pair_cos"].shape == (1, 1, 1)
    assert pytest.approx(g["pair_cos"].item(), abs=1e-5) == 1.0
    assert d["active_count"].item() == 1.0
    assert pytest.approx(d["eff_rank"].item(), abs=1e-5) == 1.0
    for k, v in d.items():
        assert torch.isfinite(v).all(), f"{k} not finite at N=1"


def test_single_invalid_candidate_stays_finite():
    """N=1 that never reached the level set: nothing active, still no NaN."""
    _, d = _geo(_as_delta([[1.0, 0.0, 0.0, 0.0]]), valid=[False])

    assert d["active_count"].item() == 0.0
    for k, v in d.items():
        assert torch.isfinite(v).all(), f"{k} not finite with no active candidate"
