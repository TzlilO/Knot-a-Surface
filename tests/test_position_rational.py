"""
End-to-end test of PositionControl: NURBS surface evaluation and rational
derivatives (quotient rule) against geomdl, plus gradient flow through the
interpolation cache.

Run:  .venv/bin/python tests/test_position_rational.py
"""
import sys
import pathlib
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tests.conftest_stubs  # noqa: F401

import numpy as np
import torch

from geomdl import NURBS

from modules.basis import compute_bases_uv_diff, make_clamped_uniform_knots
from modules.control_feature.position import PositionControl

torch.manual_seed(0)

DEGREE = 3
H, W = 7, 8
US, VS = 11, 9


class MockState:
    def __init__(self):
        self.H, self.W = H, W
        self.Us, self.Vs = US, VS
        self.device = 'cpu'
        self.use_bmm = False
        self.full_basis = False
        self.flatten_uv = False


class MockBasis:
    """Exposes bu/dbu/dbuu/bv/dbv/dbvv + contract paths like BasisFunction."""

    def __init__(self, funcs):
        self._f = funcs
        self.contract_path = 'uh,hwc,vw->uvc'
        self.optimal_path = 'auto'

    def __getattr__(self, name):
        if name in ('bu', 'dbu', 'dbuu', 'bv', 'dbv', 'dbvv'):
            return getattr(self._f, name)
        raise AttributeError(name)

    def recompute(self):
        pass


class MockWeights:
    def __init__(self, logits):
        self.logits = logits

    @property
    def features(self):
        return torch.sigmoid(self.logits).view(H, W, 1)


def main():
    ku = make_clamped_uniform_knots(H, DEGREE, device='cpu')
    kv = make_clamped_uniform_knots(W, DEGREE, device='cpu')
    u = torch.linspace(0.02, 0.98, US)
    v = torch.linspace(0.02, 0.98, VS)
    funcs = compute_bases_uv_diff(u, v, ku, kv, H, W, degree=DEGREE, device='cpu')

    ctrl = torch.rand(H, W, 3)
    weight_logits = torch.randn(H * W, 1) * 0.7
    weight_logits.requires_grad_(True)

    state = MockState()
    basis = MockBasis(funcs)
    pos = PositionControl(state, ctrl.clone(), basis)
    pos.set_weights(MockWeights(weight_logits))

    S = pos.forward().reshape(US, VS, 3)
    Su = pos.dSu.reshape(US, VS, 3)
    Sv = pos.dSv.reshape(US, VS, 3)

    # geomdl reference
    surf = NURBS.Surface()
    surf.degree_u = surf.degree_v = DEGREE
    surf.ctrlpts_size_u, surf.ctrlpts_size_v = H, W
    surf.ctrlpts = ctrl.reshape(H * W, 3).tolist()
    surf.weights = torch.sigmoid(weight_logits).detach().reshape(-1).tolist()
    surf.knotvector_u = ku.tolist()
    surf.knotvector_v = kv.tolist()

    S_ref = np.zeros((US, VS, 3))
    Su_ref = np.zeros_like(S_ref)
    Sv_ref = np.zeros_like(S_ref)
    for i, uu in enumerate(u.tolist()):
        for j, vv in enumerate(v.tolist()):
            d = surf.derivatives(uu, vv, order=1)
            S_ref[i, j] = d[0][0]
            Su_ref[i, j] = d[1][0]
            Sv_ref[i, j] = d[0][1]

    def check(name, got, ref, tol):
        err = np.abs(got.detach().numpy() - ref).max()
        ok = err < tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max_abs_err = {err:.3e}")
        return ok

    ok = True
    ok &= check("NURBS surface point S", S, S_ref, 5e-6)
    ok &= check("rational derivative Su", Su, Su_ref, 5e-5)
    ok &= check("rational derivative Sv", Sv, Sv_ref, 5e-5)

    # Gradient flow: through forward cache AND through derivatives
    loss = S.sum() + Su.sum()
    loss.backward()
    grad_ok = (
        pos.control_features.grad is not None
        and pos.control_features.grad.abs().sum().item() > 0
        and weight_logits.grad is not None
        and weight_logits.grad.abs().sum().item() > 0
    )
    print(f"  [{'PASS' if grad_ok else 'FAIL'}] gradients reach control points AND weights")
    ok &= grad_ok

    # Cacheless: repeated forward recomputes and keeps autograd graph
    S2 = pos.forward()
    recompute_ok = S2.requires_grad
    print(f"  [{'PASS' if recompute_ok else 'FAIL'}] repeated forward keeps autograd graph")
    ok &= recompute_ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
