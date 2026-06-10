"""
Multilevel B-spline Approximation (MBA) surface fitting.

Lee, Wolberg, Shin, "Scattered Data Interpolation with Multilevel
B-Splines" (TVCG 1997) — the standard method for fitting a B-spline
surface to scattered data:

  level 0: coarse grid fits the global shape (closed-form, local)
  level k: grid at 2x resolution fits the RESIDUALS of the sum so far
  ...
  final:   the summed surface is projected onto the finest basis by
           collocation at Greville abscissae (well-conditioned by
           Schoenberg-Whitney; no smoothing needed)

Each level is a per-control-point closed-form update (no global solve):
    phi_c = sum_p w_pc^2 * (w_pc z_p / s_p) / sum_p w_pc^2,  s_p = sum w^2
so empty cells are simply inherited from coarser levels — holes and
irregular sampling are handled by construction. Fully vectorized torch
(CPU or CUDA); typical clouds fit in milliseconds.
"""

from typing import Optional, Tuple

import torch

from modules.basis import bspline_basis_and_derivs_1d
from modules.knotvector import make_clamped_uniform_knots

DEGREE = 3
ORD = DEGREE + 1


def _compact_window(params: torch.Tensor, knots: torch.Tensor, n_ctrl: int):
    """Per-sample 4 basis values + first control index. [N,4], [N]."""
    (b,) = bspline_basis_and_derivs_1d(params, knots, DEGREE, max_deriv=0)
    spans = (b.abs() > 1e-12).to(torch.int64).argmax(dim=1).clamp(max=n_ctrl - ORD)
    cols = spans.unsqueeze(1) + torch.arange(ORD, device=b.device).unsqueeze(0)
    return torch.gather(b, 1, cols), spans


def _mba_level(uv, z, H, W):
    """One MBA level: closed-form local fit of z over uv. -> [H, W, C]."""
    device, n, C = uv.device, uv.shape[0], z.shape[1]
    ku = make_clamped_uniform_knots(H, DEGREE, device=device)
    kv = make_clamped_uniform_knots(W, DEGREE, device=device)
    bu, su = _compact_window(uv[:, 0].contiguous(), ku, H)
    bv, sv = _compact_window(uv[:, 1].contiguous(), kv, W)

    # w[p, i, j] = bu[p,i] * bv[p,j] over each point's 4x4 window
    w = (bu.unsqueeze(2) * bv.unsqueeze(1)).reshape(n, ORD * ORD)        # [N,16]
    s = (w * w).sum(dim=1, keepdim=True).clamp_min(1e-12)                # [N,1]
    phi = (w / s).unsqueeze(-1) * z.unsqueeze(1)                         # [N,16,C]

    # flat ctrl indices of each window cell
    gu = su.unsqueeze(1) + torch.arange(ORD, device=device).unsqueeze(0)  # [N,4]
    gv = sv.unsqueeze(1) + torch.arange(ORD, device=device).unsqueeze(0)
    idx = (gu.unsqueeze(2) * W + gv.unsqueeze(1)).reshape(n, ORD * ORD)   # [N,16]

    w2 = (w * w).reshape(-1)
    num = torch.zeros(H * W, C, device=device).index_add_(
        0, idx.reshape(-1), w2.unsqueeze(-1) * phi.reshape(-1, C))
    den = torch.zeros(H * W, device=device).index_add_(0, idx.reshape(-1), w2)
    ctrl = num / den.clamp_min(1e-12).unsqueeze(-1)
    ctrl[den < 1e-12] = 0.0                       # empty cells: defer to coarser levels
    return ctrl.reshape(H, W, C), (bu, su, bv, sv)


def _eval_at_points(ctrl, bu, su, bv, sv):
    """Evaluate surface at the points' own params via their windows."""
    H, W, C = ctrl.shape
    gu = su.unsqueeze(1) + torch.arange(ORD, device=ctrl.device).unsqueeze(0)
    gv = sv.unsqueeze(1) + torch.arange(ORD, device=ctrl.device).unsqueeze(0)
    win = ctrl.reshape(H * W, C)[(gu.unsqueeze(2) * W + gv.unsqueeze(1)).reshape(-1)]
    win = win.reshape(-1, ORD, ORD, C)
    return torch.einsum('ni,nijc,nj->nc', bu, win, bv)


def _greville(knots: torch.Tensor, n_ctrl: int) -> torch.Tensor:
    """Greville abscissae: mean of degree consecutive interior knots."""
    g = torch.stack([knots[i + 1: i + 1 + DEGREE] for i in range(n_ctrl)]).mean(1)
    return g.clamp(1e-6, 1.0 - 1e-6)


def mba_fit_surface(
    uv: torch.Tensor,          # [N, 2] in [0,1]^2
    values: torch.Tensor,      # [N, C] (xyz, or xyz+rgb)
    target_h: int,
    target_w: int,
    base: int = 8,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hierarchical MBA fit; returns a SINGLE control grid at the target
    resolution plus its clamped-uniform knot vectors.

    Returns: ctrl [target_h, target_w, C], knots_u, knots_v
    """
    if device is not None:
        uv, values = uv.to(device), values.to(device)
    uv = uv.clamp(1e-6, 1.0 - 1e-6).float()
    z = values.float()

    # --- coarse-to-fine residual fitting ---
    levels = []
    H = W = base
    residual = z
    while True:
        h, w = min(H, target_h), min(W, target_w)
        ctrl, (bu, su, bv, sv) = _mba_level(uv, residual, h, w)
        levels.append((ctrl, h, w))
        residual = residual - _eval_at_points(ctrl, bu, su, bv, sv)
        if h >= target_h and w >= target_w:
            break
        H, W = H * 2, W * 2

    # --- project the summed surface onto the finest basis ---
    # Evaluate the sum on the finest Greville grid, then solve the banded
    # collocation system (well-conditioned; Schoenberg-Whitney holds).
    dev = uv.device
    ku = make_clamped_uniform_knots(target_h, DEGREE, device=dev)
    kv = make_clamped_uniform_knots(target_w, DEGREE, device=dev)
    gu, gv = _greville(ku, target_h), _greville(kv, target_w)

    (Bu,) = bspline_basis_and_derivs_1d(gu, ku, DEGREE, max_deriv=0)   # [H,H]
    (Bv,) = bspline_basis_and_derivs_1d(gv, kv, DEGREE, max_deriv=0)   # [W,W]

    samples = torch.zeros(target_h, target_w, z.shape[1], device=dev)
    for ctrl, h, w in levels:
        lku = make_clamped_uniform_knots(h, DEGREE, device=dev)
        lkv = make_clamped_uniform_knots(w, DEGREE, device=dev)
        (lu,) = bspline_basis_and_derivs_1d(gu, lku, DEGREE, max_deriv=0)
        (lv,) = bspline_basis_and_derivs_1d(gv, lkv, DEGREE, max_deriv=0)
        samples = samples + torch.einsum('uh,hwc,vw->uvc', lu, ctrl, lv)

    # ctrl = Bu^-1 @ samples @ Bv^-T  (separable collocation solve)
    step = torch.linalg.solve(Bu, samples.reshape(target_h, -1))
    step = step.reshape(target_h, target_w, -1).permute(1, 0, 2)
    ctrl = torch.linalg.solve(Bv, step.reshape(target_w, -1))
    ctrl = ctrl.reshape(target_w, target_h, -1).permute(1, 0, 2).contiguous()

    return ctrl, ku, kv
