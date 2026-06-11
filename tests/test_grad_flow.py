"""
Gradient-flow probe: verifies that a scalar loss over every DERIVED splat
attribute (position, scale from Su/Sv, rotation from tangent frame,
opacity, SH) produces finite, nonzero gradients on the underlying control
grids — the exact paths the renderer uses.

Run:  .venv/bin/python tests/test_grad_flow.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tests.conftest_stubs  # noqa: F401

import torch

from modules.basis import compute_bases_uv_diff, make_clamped_uniform_knots
from modules.control_feature.position import PositionControl
from modules.spline_formulas import uv_tangent

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


def make_model():
    state = MockState()
    u = torch.linspace(0.02, 0.98, US)
    v = torch.linspace(0.02, 0.98, VS)
    ku = make_clamped_uniform_knots(H, DEGREE, device='cpu')
    kv = make_clamped_uniform_knots(W, DEGREE, device='cpu')
    funcs = compute_bases_uv_diff(u, v, ku, kv, H, W, degree=DEGREE)
    basis = MockBasis(funcs)
    # a curved control grid (not flat: flat surfaces have degenerate normals)
    gx, gy = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W),
                            indexing='ij')
    cpts = torch.stack([gx, gy, 0.3 * torch.sin(4 * gx) * torch.cos(3 * gy)],
                       dim=-1)
    cpts = cpts + 0.01 * torch.randn_like(cpts)
    pos = PositionControl(state, cpts, basis, name='xyz')
    return state, basis, pos


def check(name, param, loss):
    if param.grad is not None:
        param.grad = None
    # retain_graph: evaluate() is memoized per parameter version, so all
    # checks legitimately share one evaluation graph (as training does).
    loss.backward(retain_graph=True)
    g = param.grad
    assert g is not None, f"{name}: grad is None (graph broken)"
    assert torch.isfinite(g).all(), f"{name}: non-finite grads"
    frac = (g.abs() > 0).float().mean().item()
    print(f"  {name:28s} |grad| mean={g.abs().mean():.3e}  "
          f"nonzero={frac * 100:.1f}%")
    assert frac > 0.5, f"{name}: too many zero grads ({frac * 100:.1f}%)"


def main():
    state, basis, pos = make_model()
    p = pos.control_features

    print("Derived-attribute gradient flow:")
    check("position S", p, pos.forward().square().mean())

    # scale path: ||Su||*du, ||Sv||*dv  (deltas constant, as in ablation)
    du, dv = 1.0 / US, 1.0 / VS
    Su, Sv = pos.dSu, pos.dSv
    scale = torch.stack([Su.norm(dim=-1) * du, Sv.norm(dim=-1) * dv], dim=-1)
    check("derived scale", p, scale.mean())

    # rotation path: tangent frame -> quaternion
    rots = uv_tangent(pos.dSu.reshape(US, VS, 3), pos.dSv.reshape(US, VS, 3))
    target = torch.randn_like(rots)
    check("derived rotation", p, (rots * target).sum())

    # full composite (as the renderer consumes all at once)
    S = pos.forward()
    Su, Sv = pos.dSu, pos.dSv
    rots = uv_tangent(Su.reshape(US, VS, 3), Sv.reshape(US, VS, 3))
    scale = torch.stack(
        [Su.norm(dim=-1) * du, Sv.norm(dim=-1) * dv], dim=-1)
    composite = S.square().mean() + scale.mean() + rots.square().mean()
    check("composite (render-like)", p, composite)

    # opacity-style: sigmoid activate-then-interpolate via base ControlFeature
    from modules.control_feature.base import ControlFeature

    class Sig(ControlFeature):
        @property
        def activation(self):
            return torch.sigmoid

    op = Sig(state, torch.randn(H, W, 1), basis, name='opacity')
    check("opacity (sigmoid interp)", op.control_features,
          op.forward().mean())

    # ── evaluate() memoization semantics ──────────────────────────────
    s1 = pos.evaluate()[0]
    s2 = pos.evaluate()[0]
    assert s1 is s2, "memo: same version must reuse the same graph"

    with torch.no_grad():
        sng = pos.evaluate()[0]
    assert sng is not s1, "memo: no_grad eval must not share grad-mode memo"
    s3 = pos.evaluate()[0]
    assert s3.grad_fn is not None, "memo: no_grad result poisoned grad mode"

    with torch.no_grad():
        p.add_(0.01 * torch.randn_like(p))  # simulate optimizer.step()
    s4 = pos.evaluate()[0]
    assert s4 is not s3, "memo: in-place param update must bust the memo"
    check("post-step position S", p, s4.square().mean())
    print("  memoization semantics OK (reuse / no_grad isolation / step-bust)")

    print("ALL GRADIENT-FLOW CHECKS PASSED")


if __name__ == '__main__':
    main()
