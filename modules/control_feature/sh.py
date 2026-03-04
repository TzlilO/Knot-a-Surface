"""Spherical Harmonics control features (DC + rest) with wrapper."""

import torch
import torch.nn as nn
import opt_einsum as oe

from .base import ControlFeature


class SHControl(ControlFeature):
    """
    Learnable SH coefficient control grid (either DC or higher-order).

    DC component:   [H, W, 3]         -> interpolated [Us*Vs, 1, 3]
    Rest component: [H, W, (shc-1)*3] -> interpolated [Us*Vs, shc-1, 3]
    """

    def __init__(self, state, control_grid, basis, sh_component='dc', *args, **kwargs):
        super().__init__(state, control_grid, basis, *args, **kwargs)
        self.sh_channels = kwargs.get('sh_channels', control_grid.shape[-1])
        self._original_channels = self.sh_channels
        self.sh_component = sh_component

        if sh_component == 'dc':
            self.sh_channels = 3
            self.num_sh_coeffs = 1
        else:
            self.sh_channels = 3
            self.num_sh_coeffs = state.shc - 1

        self._expected_feature_dim = self.sh_channels * self.num_sh_coeffs

    @property
    def features(self):
        return self.control_features.view(
            self.state.H, self.state.W, self._expected_feature_dim,
        )

    @property
    def feature_channels(self):
        return self._expected_feature_dim

    def interpolate_samples(self) -> torch.Tensor:
        if self.cache_valid:
            return self.cache

        cpts = self.features

        interpolated = oe.contract(
            self.basis.contract_path,
            self.basis.bu, cpts, self.basis.bv,
            optimize=self.basis.optimal_path,
        ).contiguous()

        if self.sh_component == 'dc':
            reshaped = interpolated.view(-1, 1, 3)
        else:
            reshaped = interpolated.view(-1, self.num_sh_coeffs, 3)

        self._cache = reshaped
        return reshaped

    forward = interpolate_samples

    def compute_removed_grid(
        self, direction, remove_idx, blend_radius=3,
        blend_strength=0.5, use_blend=False,
    ) -> torch.Tensor:
        blend_radius = blend_radius if blend_radius is not None else self.state.degree
        return super().compute_removed_grid(
            direction, remove_idx,
            blend_radius=blend_radius,
            blend_strength=blend_strength,
            use_blend=use_blend,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def capture_state(self) -> dict:
        state = super().capture_state()
        state['sh_component'] = self.sh_component
        state['sh_channels'] = self.sh_channels
        state['num_sh_coeffs'] = self.num_sh_coeffs
        return state

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        sh_component = state.get('sh_component')
        instance = super(SHControl, cls).from_state(
            state, model_state, basis, device,
        )
        instance.sh_component = sh_component
        instance.sh_channels = state.get('sh_channels', 3)
        instance.num_sh_coeffs = state.get('num_sh_coeffs', 1)
        instance._expected_feature_dim = instance.sh_channels * instance.num_sh_coeffs
        return instance


class SHControlWrapper(nn.Module):
    """
    Combines DC and higher-order SH components into a single interface.

    Provides unified forward(), features, cache, and serialization
    while keeping DC and rest as separately optimizable parameter groups.
    """

    def __init__(self, state, sh_dc, sh_rest, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.sh_degree = state.max_sh_degree
        self.sh_dc = sh_dc
        self.sh_rest = sh_rest

    def invalidate_all(self):
        self.sh_dc._cache = None
        self.sh_rest._cache = None

    @property
    def features(self):
        """Combined SH features [H, W, shc, 3]."""
        H, W = self.state.H, self.state.W
        shc = self.state.shc
        dc = self.sh_dc.features.view(H, W, 3).unsqueeze(2)
        rest = self.sh_rest.features.view(H, W, shc - 1, 3)
        return torch.cat([dc, rest], dim=2)

    @property
    def feature_channels(self):
        return self.sh_dc.feature_channels + self.sh_rest.feature_channels

    def interpolate_samples(self) -> torch.Tensor:
        return torch.cat([
            self.sh_dc.interpolate_samples(),
            self.sh_rest.interpolate_samples(),
        ], dim=1)

    forward = interpolate_samples

    @property
    def cache(self):
        dc = self.sh_dc.cache.reshape(-1, self.sh_dc.feature_channels)
        rest = self.sh_rest.cache.reshape(-1, self.sh_rest.feature_channels)
        combined = torch.cat([dc, rest], dim=-1)
        shc = dc.shape[-1] // 3 + rest.shape[-1] // 3
        return combined.reshape(-1, shc, 3)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def capture_state(self) -> dict:
        return {
            'sh_dc': self.sh_dc.capture_state(),
            'sh_rest': self.sh_rest.capture_state(),
            'sh_degree': self.sh_degree,
        }

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.state = model_state
        instance.basis = basis
        instance.sh_degree = state.get('sh_degree', model_state.max_sh_degree)
        instance.sh_dc = SHControl.from_state(state['sh_dc'], model_state, basis, device)
        instance.sh_rest = SHControl.from_state(state['sh_rest'], model_state, basis, device)
        return instance