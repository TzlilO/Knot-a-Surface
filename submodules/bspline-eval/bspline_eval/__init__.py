"""
Fused local-support B-spline surface evaluation (CUDA).

`tp_contract(grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v)` returns a
[5, Mu, Mv, C] tensor with the tensor-product contractions
(S, dS/du, dS/dv, d2S/du2, d2S/dv2) of `grid` against compact per-sample
basis values. Differentiable w.r.t. grid AND the basis values (so knot
optimization — which produces basis values through autograd in Python —
keeps its gradient path).

`compact_basis(dense, spans)` converts a dense [M, n_ctrl] basis matrix to
the compact [M, 4] window representation given the span (first nonzero
column) of each row.
"""
import torch

from . import _C


class _TpContract(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v):
        out = _C.forward(grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v)
        ctx.save_for_backward(grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v = ctx.saved_tensors
        d_grid, d_bu, d_dbu, d_d2bu, d_bv, d_dbv, d_d2bv = _C.backward(
            grad_out.contiguous(), grid, bu, dbu, d2bu, bv, dbv, d2bv,
            span_u, span_v,
        )
        return d_grid, d_bu, d_dbu, d_d2bu, d_bv, d_dbv, d_d2bv, None, None


def tp_contract(grid, bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v):
    """[H,W,C] x compact bases -> [5, Mu, Mv, C] (S, Su, Sv, Suu, Svv sums)."""
    return _TpContract.apply(
        grid.contiguous(), bu, dbu, d2bu, bv, dbv, d2bv, span_u, span_v
    )


def find_spans(samples: torch.Tensor, knots: torch.Tensor, degree: int,
               n_ctrl: int) -> torch.Tensor:
    """First control index of each sample's (degree+1) window. [M] int64."""
    # span s satisfies knots[s] <= u < knots[s+1], degree <= s <= n_ctrl-1;
    # window of ctrl indices is [s-degree, s].
    s = torch.searchsorted(knots, samples.contiguous(), right=True) - 1
    s = s.clamp(min=degree, max=n_ctrl - 1)
    return (s - degree).to(torch.int64)


def compact_basis(dense: torch.Tensor, spans: torch.Tensor,
                  degree: int = 3) -> torch.Tensor:
    """Gather the (degree+1) active columns of each row: [M, n] -> [M, p+1]."""
    cols = spans.unsqueeze(1) + torch.arange(
        degree + 1, device=dense.device
    ).unsqueeze(0)
    return torch.gather(dense, 1, cols)
