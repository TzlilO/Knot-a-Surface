"""Position control features (XYZ coordinates) with optional NURBS weights."""

from typing import TYPE_CHECKING, Optional, Tuple

import opt_einsum as oe
import torch

from .base import ControlFeature

if TYPE_CHECKING:
    from modules.ModelState import ModelState
    from modules.basis import BasisFunction


class PositionControl(ControlFeature):
    """
    Learnable position control grid.

    Supports rational (NURBS) mode via an attached WeightControl.
    Provides analytic surface derivatives (dSu, dSv, dSuu, dSvv)
    and finite-difference approximations (du, dv) from cached samples.
    """

    weights: Optional['WeightControl'] = None

    def __init__(self, state, control_grid, basis, **kwargs):
        super().__init__(state, control_grid, basis, **kwargs)
        self.dsu_cache = None
        self.dsv_cache = None
        self.dsuu_cache = None
        self.dsvv_cache = None

    def set_weights(self, weights: 'WeightControl'):
        self.weights = weights

    # ------------------------------------------------------------------
    # Interpolation (overrides base to handle rational denominator)
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        if self.cache_valid:
            return self.cache.reshape(-1, self.feature_channels)

        cpts = self.features
        if self.weights is not None:
            w = self.weights.features
            cpts = cpts * w

        prod = self._contract(self.basis.bu, self.basis.bv, cpts)

        if self.weights is not None:
            denom = self._contract(self.basis.bu, self.basis.bv, w).clamp(min=1e-6)
            prod = prod / denom

        self.set_cache(prod)
        return prod.reshape(-1, self.feature_channels)


    # ------------------------------------------------------------------
    # Features (no activation for positions)
    # ------------------------------------------------------------------

    @property
    def features(self):
        return self.control_features.view(
            self.state.H, self.state.W, self.control_features.shape[-1]
        )

    # ------------------------------------------------------------------
    # Surface point access
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Finite-difference derivatives from cached samples
    # ------------------------------------------------------------------

    @property
    def du(self):
        grid = self._cache.clone().reshape(self.state.sampling_layout)
        d = grid[1:, :, :] - grid[:-1, :, :]
        return torch.cat([d[:1, :, :], d], dim=0)

    @property
    def dv(self):
        grid = self._cache.clone().reshape(self.state.sampling_layout)
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
    # Analytic B-spline derivatives (using derivative basis functions)
    # ------------------------------------------------------------------

    def _contract(self, bu, bv, grid):
        if self.state.use_bmm:
            return self._interpolate_bmm(bu, bv, grid).contiguous()
        return oe.contract(
            self.basis.contract_path,
            bu, grid, bv,
            optimize=self.basis.optimal_path,
        ).contiguous()

    def _deriv_interpolate(self, bu, bv, cache_attr, d1_pair=None):
        """
        Surface partial derivative for the basis pair (bu, bv), where exactly
        one of the two is a derivative basis.

        B-spline:  S' = Σ N' P
        NURBS:     S = A/W with A = Σ N (w P), W = Σ N w. Quotient rule:
                   S'  = (A' − S·W') / W
                   S'' = (A'' − 2·S'·W' − S·W'') / W   (d1_pair gives the
                   first-derivative basis pair for the same direction)
        """
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        cpts = self.features
        if self.weights is not None:
            w = self.weights.features
            cpts = cpts * w

        prod = self._contract(bu, bv, cpts)

        if self.weights is not None:
            W = self._contract(self.basis.bu, self.basis.bv, w.unsqueeze(-1) if w.dim() == 2 else w).clamp(min=1e-6)
            W_prime = self._contract(bu, bv, w.unsqueeze(-1) if w.dim() == 2 else w)
            S = self._contract(self.basis.bu, self.basis.bv, cpts) / W
            if d1_pair is None:
                # First derivative
                prod = (prod - S * W_prime) / W
            else:
                # Second derivative
                b1u, b1v = d1_pair
                A1 = self._contract(b1u, b1v, cpts)
                W1 = self._contract(b1u, b1v, w.unsqueeze(-1) if w.dim() == 2 else w)
                S1 = (A1 - S * W1) / W
                prod = (prod - 2.0 * S1 * W1 - S * W_prime) / W

        setattr(self, cache_attr, prod)
        return prod

    @property
    def dSu(self):
        return self._deriv_interpolate(self.basis.dbu, self.basis.bv, 'dsu_cache')

    @property
    def dSv(self):
        return self._deriv_interpolate(self.basis.bu, self.basis.dbv, 'dsv_cache')

    @property
    def dSuu(self):
        return self._deriv_interpolate(
            self.basis.dbuu, self.basis.bv, 'dsuu_cache',
            d1_pair=(self.basis.dbu, self.basis.bv),
        )

    @property
    def dSvv(self):
        return self._deriv_interpolate(
            self.basis.bu, self.basis.dbvv, 'dsvv_cache',
            d1_pair=(self.basis.bu, self.basis.dbv),
        )

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def invalidate(self, hard=False):
        super().invalidate(hard)
        self.basis.recompute()
        self.dsu_cache = None
        self.dsv_cache = None
        self.dsuu_cache = None
        self.dsvv_cache = None

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