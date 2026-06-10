"""
Correctness tests for B-spline / NURBS basis math against geomdl ground truth.

Run:  .venv/bin/python -m pytest tests/test_basis_math.py -v
  or: .venv/bin/python tests/test_basis_math.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tests.conftest_stubs  # noqa: F401  (must precede modules.* imports)

import numpy as np
import torch

from geomdl import BSpline, NURBS

from modules.basis import (
    compute_bases_uv_diff,
    compute_all_derivatives_2d,
    make_clamped_uniform_knots,
)

torch.manual_seed(0)
np.random.seed(0)

DEGREE = 3
H, W = 8, 9          # control grid
US, VS = 17, 13      # sample resolution
ATOL = 1e-4


def _geomdl_surface(ctrl, knots_u, knots_v, weights=None):
    if weights is None:
        surf = BSpline.Surface()
    else:
        surf = NURBS.Surface()
    surf.degree_u = surf.degree_v = DEGREE
    surf.ctrlpts_size_u, surf.ctrlpts_size_v = H, W
    pts = ctrl.reshape(H * W, 3).tolist()
    if weights is None:
        surf.ctrlpts = pts
    else:
        wflat = weights.reshape(H * W).tolist()
        surf.ctrlpts = pts
        surf.weights = wflat
    surf.knotvector_u = knots_u.tolist()
    surf.knotvector_v = knots_v.tolist()
    return surf


def _eval_with_basis(bu, bv, ctrl):
    """S(u_i, v_j) = sum_h sum_w bu[i,h] bv[j,w] P[h,w]"""
    return torch.einsum("uh,hwc,vw->uvc", bu, ctrl, bv)


def _nonuniform_knots(n_ctrl, degree):
    """Clamped, non-uniform interior knots."""
    n_internal = n_ctrl + degree + 1 - 2 * (degree + 1)
    interior = np.sort(np.random.uniform(0.15, 0.85, n_internal))
    return torch.tensor(
        [0.0] * (degree + 1) + interior.tolist() + [1.0] * (degree + 1),
        dtype=torch.float64,
    )


def _setup(nonuniform=False, rational=False):
    if nonuniform:
        ku = _nonuniform_knots(H, DEGREE)
        kv = _nonuniform_knots(W, DEGREE)
    else:
        ku = make_clamped_uniform_knots(H, DEGREE, device="cpu").double()
        kv = make_clamped_uniform_knots(W, DEGREE, device="cpu").double()
    ctrl = torch.rand(H, W, 3, dtype=torch.float64)
    weights = (
        torch.rand(H, W, dtype=torch.float64) * 1.5 + 0.5 if rational else None
    )
    u = torch.linspace(0.01, 0.99, US, dtype=torch.float64)
    v = torch.linspace(0.01, 0.99, VS, dtype=torch.float64)
    return ku, kv, ctrl, weights, u, v


def _geomdl_derivs(surf, u, v, order=2):
    """Returns S, Su, Sv, Suu, Svv arrays of shape [US, VS, 3]."""
    S = np.zeros((len(u), len(v), 3))
    Su = np.zeros_like(S)
    Sv = np.zeros_like(S)
    Suu = np.zeros_like(S)
    Svv = np.zeros_like(S)
    for i, uu in enumerate(u):
        for j, vv in enumerate(v):
            d = surf.derivatives(float(uu), float(vv), order=order)
            S[i, j] = d[0][0]
            Su[i, j] = d[1][0]
            Sv[i, j] = d[0][1]
            if order >= 2:
                Suu[i, j] = d[2][0]
                Svv[i, j] = d[0][2]
    return S, Su, Sv, Suu, Svv


def _check(name, got, want, atol):
    got = np.asarray(got)
    err = np.abs(got - want).max()
    status = "PASS" if err < atol else "FAIL"
    print(f"  [{status}] {name:42s} max_abs_err = {err:.3e}")
    return err < atol


def run_case(title, basis_fn, nonuniform):
    """basis_fn(u, v, ku, kv) -> (bu, dbu, dbuu, bv, dbv, dbvv)"""
    print(f"\n=== {title} | {'non-uniform' if nonuniform else 'uniform'} knots ===")
    ku, kv, ctrl, _, u, v = _setup(nonuniform=nonuniform)
    bu, dbu, dbuu, bv, dbv, dbvv = basis_fn(u, v, ku, kv)
    bu_dtype = bu.dtype
    bu, dbu, dbuu = bu.double(), dbu.double(), dbuu.double()
    bv, dbv, dbvv = bv.double(), dbv.double(), dbvv.double()

    surf = _geomdl_surface(ctrl.numpy(), ku.numpy(), kv.numpy())
    S, Su, Sv, Suu, Svv = _geomdl_derivs(surf, u.numpy(), v.numpy())

    # fp32 paths can only be accurate to ~1e-6
    unity_tol = 1e-9 if bu_dtype == torch.float64 else 5e-6
    ok = True
    ok &= _check("partition of unity (bu)", bu.sum(-1), 1.0, unity_tol)
    ok &= _check("partition of unity (bv)", bv.sum(-1), 1.0, unity_tol)
    ok &= _check("derivative basis sums to 0 (dbu)", dbu.sum(-1), 0.0, unity_tol * 10)
    ok &= _check("surface point S", _eval_with_basis(bu, bv, ctrl), S, ATOL)
    ok &= _check("first derivative Su", _eval_with_basis(dbu, bv, ctrl), Su, ATOL * 10)
    ok &= _check("first derivative Sv", _eval_with_basis(bu, dbv, ctrl), Sv, ATOL * 10)
    ok &= _check("second derivative Suu", _eval_with_basis(dbuu, bv, ctrl), Suu, ATOL * 100)
    ok &= _check("second derivative Svv", _eval_with_basis(bu, dbvv, ctrl), Svv, ATOL * 100)
    return ok


def basis_live_path(u, v, ku, kv):
    """The path used by BasisFunction.recompute() in training."""
    bf = compute_bases_uv_diff(
        u.float(), v.float(), ku.float(), kv.float(), H, W,
        degree=DEGREE, device="cpu",
    )
    return bf.bu, bf.dbu, bf.dbuu, bf.bv, bf.dbv, bf.dbvv


def basis_triangular_table(u, v, ku, kv):
    """Exact triangular-table algorithm (compute_all_derivatives_2d)."""
    uv = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1).reshape(-1, 2)
    (bu, dbu, dbuu), (bv, dbv, dbvv) = compute_all_derivatives_2d(
        uv, ku, kv, DEGREE, DEGREE, max_deriv=2
    )
    # 2d helper evaluates per uv pair; reduce back to separable 1-D matrices
    bu = bu.reshape(US, VS, H)[:, 0, :]
    dbu = dbu.reshape(US, VS, H)[:, 0, :]
    dbuu = dbuu.reshape(US, VS, H)[:, 0, :]
    bv = bv.reshape(US, VS, W)[0, :, :]
    dbv = dbv.reshape(US, VS, W)[0, :, :]
    dbvv = dbvv.reshape(US, VS, W)[0, :, :]
    return bu, dbu, dbuu, bv, dbv, dbvv


def test_rational_quotient_rule():
    """NURBS: dS = (A' - S W') / W, where A = sum N (w P), W = sum N w."""
    print("\n=== rational (NURBS) derivative — quotient rule ===")
    ku, kv, ctrl, weights, u, v = _setup(nonuniform=True, rational=True)

    uv = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1).reshape(-1, 2)
    (bu, dbu), (bv, dbv) = compute_all_derivatives_2d(
        uv, ku, kv, DEGREE, DEGREE, max_deriv=1
    )
    bu = bu.reshape(US, VS, H)[:, 0, :]
    dbu = dbu.reshape(US, VS, H)[:, 0, :]
    bv = bv.reshape(US, VS, W)[0, :, :]
    dbv = dbv.reshape(US, VS, W)[0, :, :]

    wP = ctrl * weights[..., None]
    A = _eval_with_basis(bu, bv, wP)
    Wd = torch.einsum("uh,hw,vw->uv", bu, weights, bv)[..., None]
    Au = _eval_with_basis(dbu, bv, wP)
    Wu = torch.einsum("uh,hw,vw->uv", dbu, weights, bv)[..., None]
    S = A / Wd
    Su_correct = (Au - S * Wu) / Wd
    Su_buggy = Au / Wd  # what position.py::_deriv_interpolate currently does

    surf = _geomdl_surface(ctrl.numpy(), ku.numpy(), kv.numpy(), weights.numpy())
    _, Su_ref, _, _, _ = _geomdl_derivs(surf, u.numpy(), v.numpy(), order=1)

    ok = _check("quotient rule (A' - S W')/W", Su_correct, Su_ref, 1e-6)
    bad = not _check("buggy A'/W (expected FAIL)", Su_buggy, Su_ref, 1e-6)
    print(f"  → buggy path max err vs correct: "
          f"{np.abs(Su_buggy.numpy() - Su_ref).max():.3e}")
    return ok and bad


if __name__ == "__main__":
    results = {}
    for nonuni in (False, True):
        results[f"triangular_table nonuni={nonuni}"] = run_case(
            "triangular table (exact)", basis_triangular_table, nonuni
        )
        results[f"live_path nonuni={nonuni}"] = run_case(
            "LIVE training path (compute_bases_uv_diff)", basis_live_path, nonuni
        )
    results["rational_quotient"] = test_rational_quotient_rule()

    print("\n================ SUMMARY ================")
    for k, vv in results.items():
        print(f"  {'PASS' if vv else 'FAIL'}  {k}")
    sys.exit(0 if all(results.values()) else 1)
