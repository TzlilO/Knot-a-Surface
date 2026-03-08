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

    def interpolate_samples(self, cache=True) -> torch.Tensor:
        if self.cache_valid:
            return self.cache

        cpts = self.features
        if self.weights is not None:
            cpts = cpts * self.weights.features
            denom = self.weights.interpolate_samples()

        if self.state.use_bmm:
            prod = self._interpolate_bmm(
                self.basis.bu, self.basis.bv, cpts
            ).contiguous()
        else:
            prod = oe.contract(
                self.basis.contract_path,
                self.basis.bu, cpts, self.basis.bv,
                optimize=self.basis.optimal_path,
            ).contiguous()

        if self.weights is not None:
            prod = prod / (denom + 1e-8)

        self._cache = prod
        return prod.reshape(-1, self.feature_channels)

    forward = interpolate_samples

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

    @property
    def S(self):
        return self._cache.view(-1, 3)

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
        grid = self.features.reshape(self.state.H, self.state.W, 3).clone()
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

    def _deriv_interpolate(self, bu, bv, cache_attr):
        """Helper for derivative computations."""
        cpts = self.features
        if self.weights is not None:
            cpts = cpts * self.weights.features
            denom = self.weights.interpolate_samples()

        if self.state.use_bmm:
            prod = self._interpolate_bmm(bu, bv, cpts).contiguous()
        else:
            prod = oe.contract(
                self.basis.contract_path,
                bu, cpts, bv,
                optimize=self.basis.optimal_path,
            ).contiguous()

        if self.weights is not None:
            prod = prod / denom.clamp(min=1e-6)

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
        return self._deriv_interpolate(self.basis.dbuu, self.basis.bv, 'dsuu_cache')

    @property
    def dSvv(self):
        return self._deriv_interpolate(self.basis.bu, self.basis.dbvv, 'dsvv_cache')

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def invalidate(self, hard=False):
        super().invalidate(hard)
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