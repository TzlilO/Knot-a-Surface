"""
Validate the fused CUDA B-spline evaluation (bspline_eval) against the
einsum reference path: values, gradients (control points, weights, knots),
and speed. Requires a CUDA GPU with the extension built.

Run:  python tests/test_fused_eval.py
"""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

assert torch.cuda.is_available(), "needs CUDA"
import bspline_eval as bse

from modules.basis import compute_bases_uv_diff, make_clamped_uniform_knots

torch.manual_seed(0)
DEGREE = 3


def reference(grid, bu, dbu, dbuu, bv, dbv, dbvv):
    """Dense einsum contraction of all 5 basis pairs."""
    def c(a, b):
        return torch.einsum("uh,hwc,vw->uvc", a, grid, b)
    return torch.stack([
        c(bu, bv), c(dbu, bv), c(bu, dbv), c(dbuu, bv), c(bu, dbvv)
    ])


def main():
    H, W, C = 95, 108, 4         # rational layout: xyz*w | w
    Us, Vs = 350, 380
    dev = "cuda"

    ku = make_clamped_uniform_knots(H, DEGREE, device=dev)
    kv = make_clamped_uniform_knots(W, DEGREE, device=dev)
    u = torch.rand(Us, device=dev).sort().values * 0.98 + 0.01
    v = torch.rand(Vs, device=dev).sort().values * 0.98 + 0.01
    funcs = compute_bases_uv_diff(u, v, ku, kv, H, W, degree=DEGREE, device=dev)
    bu, dbu, dbuu = funcs.bu, funcs.dbu, funcs.dbuu
    bv, dbv, dbvv = funcs.bv, funcs.dbv, funcs.dbvv

    grid = torch.rand(H, W, C, device=dev, requires_grad=True)
    grid_ref = grid.detach().clone().requires_grad_(True)

    # compact representation (same helper the live integration uses)
    from modules.control_feature.position import PositionControl
    cbu, cdbu, cdbuu, su = PositionControl._compact(bu, dbu, dbuu, H)
    cbv, cdbv, cdbvv, sv = PositionControl._compact(bv, dbv, dbvv, W)

    # ---- forward correctness ----
    out = bse.tp_contract(grid, cbu, cdbu, cdbuu, cbv, cdbv, cdbvv, su, sv)
    ref = reference(grid_ref, bu, dbu, dbuu, bv, dbv, dbvv)
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item()
    ok = err < 1e-4 * max(1.0, scale)
    print(f"[{'PASS' if ok else 'FAIL'}] forward  max_abs_err={err:.3e} (ref scale {scale:.2f})")
    all_ok = ok

    # ---- backward correctness (grid + basis values) ----
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    gerr = (grid.grad - grid_ref.grad).abs().max().item()
    gscale = grid_ref.grad.abs().max().item()
    ok = gerr < 1e-4 * max(1.0, gscale)
    print(f"[{'PASS' if ok else 'FAIL'}] backward grid  max_abs_err={gerr:.3e} (scale {gscale:.2f})")
    all_ok &= ok

    # gradient w.r.t. basis values (knot-optimization path): compare against
    # autograd on the dense reference by perturbing compact values directly.
    cbu2 = cbu.detach().clone().requires_grad_(True)
    out2 = bse.tp_contract(grid.detach(), cbu2, cdbu, cdbuu, cbv, cdbv, cdbvv, su, sv)
    out2.backward(g)
    # finite-difference check on a few entries
    eps = 1e-3
    idx = (3, 1)
    cbu_p = cbu.detach().clone(); cbu_p[idx] += eps
    cbu_m = cbu.detach().clone(); cbu_m[idx] -= eps
    f_p = (bse.tp_contract(grid.detach(), cbu_p, cdbu, cdbuu, cbv, cdbv, cdbvv, su, sv) * g).sum()
    f_m = (bse.tp_contract(grid.detach(), cbu_m, cdbu, cdbuu, cbv, cdbv, cdbvv, su, sv) * g).sum()
    fd = ((f_p - f_m) / (2 * eps)).item()
    an = cbu2.grad[idx].item()
    ok = abs(fd - an) < 5e-2 * max(1.0, abs(fd))
    print(f"[{'PASS' if ok else 'FAIL'}] backward basis  fd={fd:.4f} analytic={an:.4f}")
    all_ok &= ok

    # ---- benchmark ----
    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    gd = grid.detach()
    t_fused = bench(lambda: bse.tp_contract(gd, cbu, cdbu, cdbuu, cbv, cdbv, cdbvv, su, sv))
    t_ref = bench(lambda: reference(gd, bu, dbu, dbuu, bv, dbv, dbvv))
    print(f"[BENCH] fused {t_fused:.3f} ms  vs  einsum {t_ref:.3f} ms  "
          f"(speedup x{t_ref / t_fused:.1f})")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
