"""Position control features (XYZ coordinates) with optional NURBS weights."""

from typing import TYPE_CHECKING, Optional, Tuple

import opt_einsum as oe
import torch

from .base import ControlFeature, fused_available, fused_contract

if TYPE_CHECKING:
    from modules.ModelState import ModelState
    from modules.basis import BasisFunction


class PositionControl(ControlFeature):
    """
    Learnable position control grid.

    Supports rational (NURBS) mode via an attached WeightControl.
    Provides analytic surface derivatives (dSu, dSv, dSuu, dSvv).

    No caching: `evaluate()` computes the surface point and all four
    derivatives in one fused CUDA kernel call (einsum fallback); each
    accessor calls it directly.
    """

    weights: Optional['WeightControl'] = None

    def __init__(self, state, control_grid, basis, **kwargs):
        super().__init__(state, control_grid, basis, **kwargs)

    def set_weights(self, weights: 'WeightControl'):
        self.weights = weights

    # ------------------------------------------------------------------
    # Evaluation: one fused call -> (S, Su, Sv, Suu, Svv)
    # ------------------------------------------------------------------

    def evaluate(self):
        """Return (S, Su, Sv, Suu, Svv), each [Us, Vs, 3]."""
        cpts = self.features
        rational = self.weights is not None

        if fused_available(self.state, self.basis, cpts):
            if rational:
                w = self.weights.features
                if w.dim() == 2:
                    w = w.unsqueeze(-1)
                grid = torch.cat([cpts * w, w], dim=-1)   # [H, W, 4]
            else:
                grid = cpts
            out = fused_contract(grid, self.basis, self.state.H, self.state.W)
            if rational:
                A, Wd = out[..., :3], out[..., 3:]
                Wn = Wd[0].clamp(min=1e-6)
                S = A[0] / Wn
                Su = (A[1] - S * Wd[1]) / Wn
                Sv = (A[2] - S * Wd[2]) / Wn
                Suu = (A[3] - 2.0 * Su * Wd[1] - S * Wd[3]) / Wn
                Svv = (A[4] - 2.0 * Sv * Wd[2] - S * Wd[4]) / Wn
                return S, Su, Sv, Suu, Svv
            return out[0], out[1], out[2], out[3], out[4]

        # einsum fallback (CPU / non-grid modes)
        b = self.basis
        if rational:
            w = self.weights.features
            if w.dim() == 2:
                w = w.unsqueeze(-1)
            wp = cpts * w
            W0 = self._contract(b.bu, b.bv, w).clamp(min=1e-6)
            S = self._contract(b.bu, b.bv, wp) / W0
            def deriv(bu_, bv_):
                A1 = self._contract(bu_, bv_, wp)
                W1 = self._contract(bu_, bv_, w)
                return (A1 - S * W1) / W0, W1
            Su, Wu = deriv(b.dbu, b.bv)
            Sv, Wv = deriv(b.bu, b.dbv)
            Auu = self._contract(b.dbuu, b.bv, wp)
            Wuu = self._contract(b.dbuu, b.bv, w)
            Avv = self._contract(b.bu, b.dbvv, wp)
            Wvv = self._contract(b.bu, b.dbvv, w)
            Suu = (Auu - 2.0 * Su * Wu - S * Wuu) / W0
            Svv = (Avv - 2.0 * Sv * Wv - S * Wvv) / W0
            return S, Su, Sv, Suu, Svv

        S = self._contract(b.bu, b.bv, cpts)
        Su = self._contract(b.dbu, b.bv, cpts)
        Sv = self._contract(b.bu, b.dbv, cpts)
        Suu = self._contract(b.dbuu, b.bv, cpts)
        Svv = self._contract(b.bu, b.dbvv, cpts)
        return S, Su, Sv, Suu, Svv

    def forward(self) -> torch.Tensor:
        """Surface points [Us*Vs, 3]."""
        cpts = self.features
        rational = self.weights is not None
        if fused_available(self.state, self.basis, cpts):
            return self.evaluate()[0].reshape(-1, self.feature_channels)
        prod = (
            self._contract(self.basis.bu, self.basis.bv, cpts)
            if not rational else None
        )
        if rational:
            w = self.weights.features
            if w.dim() == 2:
                w = w.unsqueeze(-1)
            num = self._contract(self.basis.bu, self.basis.bv, cpts * w)
            den = self._contract(self.basis.bu, self.basis.bv, w).clamp(min=1e-6)
            prod = num / den
        return prod.reshape(-1, self.feature_channels)

    @property
    def dSu(self):
        return self.evaluate()[1]

    @property
    def dSv(self):
        return self.evaluate()[2]

    @property
    def dSuu(self):
        return self.evaluate()[3]

    @property
    def dSvv(self):
        return self.evaluate()[4]

    # ------------------------------------------------------------------
    # Features (no activation for positions)
    # ------------------------------------------------------------------

    @property
    def features(self):
        return self.control_features.view(
            self.state.H, self.state.W, self.control_features.shape[-1]
        )

    # ------------------------------------------------------------------
    # Finite differences over the sample grid
    # ------------------------------------------------------------------

    @property
    def du(self):
        grid = self.forward().reshape(self.state.sampling_layout)
        d = grid[1:, :, :] - grid[:-1, :, :]
        return torch.cat([d[:1, :, :], d], dim=0)

    @property
    def dv(self):
        grid = self.forward().reshape(self.state.sampling_layout)
        d = grid[:, 1:, :] - grid[:, :-1, :]
        return torch.cat([d[:, :1, :], d], dim=1)

    @property
    def diff_u(self):
        return self.du.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @property
    def diff_v(self):
        return self.dv.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Control-grid-level finite differences
    # ------------------------------------------------------------------

    @property
    def scale_u_ctrl(self):
        return self.dsdu().norm(dim=-1, keepdim=True)

    def dsdu(self):
        grid = self.features.reshape(self.state.H, self.state.W, 3)
        d = grid[1:, :, :] - grid[:-1, :, :]
        return torch.cat([d[:1, :, :], d], dim=0)

    @property
    def scale_v_ctrl(self):
        return self.dsdv().norm(dim=-1, keepdim=True)

    def dsdv(self):
        grid = self.features.reshape(self.state.H, self.state.W, 3)
        d = grid[:, 1:, :] - grid[:, :-1, :]
        return torch.cat([d[:, :1, :], d], dim=1)

    # ------------------------------------------------------------------
    # Shared einsum contraction
    # ------------------------------------------------------------------

    def _contract(self, bu, bv, grid):
        if self.state.use_bmm:
            return self._interpolate_bmm(bu, bv, grid).contiguous()
        return oe.contract(
            self.basis.contract_path, bu, grid, bv,
            optimize=self.basis.optimal_path,
        ).contiguous()

    # ------------------------------------------------------------------
    # Invalidation hook: only triggers basis recompute (no caches exist)
    # ------------------------------------------------------------------

    def invalidate(self, hard=False):
        self.basis.recompute()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda', **kwargs):
        instance = super().from_state(state, model_state, basis, device=device, **kwargs)
        weights_state = state.get('weights_state', None)
        if weights_state is not None:
            from .weights import WeightControl
            instance.weights = WeightControl.from_state(
                weights_state, model_state, basis, device=device,
            )
        return instance
