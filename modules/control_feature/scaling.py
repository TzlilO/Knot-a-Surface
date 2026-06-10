"""Scaling control features (log-space Gaussian scales)."""

from typing import Tuple

import torch

from .base import ControlFeature


from simple_knn._C import distCUDA2

class ScalingControl(ControlFeature):
    """
    Learnable per-control-point scaling in log-space.

    Stores log(scale) as parameters; activation = exp, inverse = log.
    Handles density correction when sampling resolution changes, and
    compensates neighboring scales after knot removal.
    """

    def __init__(self, state, control_grid, basis, activation=torch.exp, **kwargs):
        super().__init__(state, control_grid, basis, activation=activation, **kwargs)
        self._activation = torch.exp
        self._inverse_activation = torch.log
        self.subdivision_scale_factor = 0.8
        self.subdivision_count = 2
        if position := kwargs.get('position'):
            self.set_position(position)

    @property
    def activation(self):
        return torch.exp

    @property
    def inverse_activation(self):
        return torch.log

    @property
    def features(self):
        """Exponentiated control grid (world-space scales)."""
        return self.activation(self.control_features.view(
            self.state.H, self.state.W, self.control_features.shape[-1]
        ))

    @property
    def density_factor(self):
        return 1.0 / (0.8 * self.state.sampling_density)

    # ------------------------------------------------------------------
    # Interpolation with density correction + normal padding
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        scales = super().forward()

        # Pad the flat normal axis BEFORE density correction (which indexes
        # all three axes). `scales` is already activated (exp-space); the
        # tiny positive value is exp(-9) ≈ 1.234e-4 in activated space.
        if self.state.scaling_dims == 2:
            scaling_n = torch.full(
                (scales.shape[0], 1), fill_value=1.234e-4, device=scales.device,
            )
            scales = torch.cat([scales, scaling_n], dim=-1)

        return scales#self._apply_density_correction(scales)

    def _apply_density_correction(self, scales: torch.Tensor) -> torch.Tensor:
        """
        Correct tangential scales for current sampling density.

        When Us, Vs change (via density or adaptive tessellation), the
        inter-sample spacing changes. We scale σ_u, σ_v proportionally
        so Gaussians neither overlap excessively nor leave gaps.
        """
        H, W = self.state.H, self.state.W
        delta_u_ref = 1.0 / (H - 1)
        delta_v_ref = 1.0 / (W - 1)

        delta_u_current = self.basis.uv_sampler.delta_u
        delta_v_current = self.basis.uv_sampler.delta_v

        correction_u = (delta_u_current / delta_u_ref).reshape(-1)
        correction_v = (delta_v_current / delta_v_ref).reshape(-1)

        s_u = scales[:, 0] * correction_u
        s_v = scales[:, 1] * correction_v
        s_n = scales[:, 2]
        return torch.stack([s_u, s_v, s_n], dim=-1)

    # ------------------------------------------------------------------
    # Insertion: shrink new control point scales
    # ------------------------------------------------------------------
    def compute_inserted_grid(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=None, blend_strength=0.3, use_blend=False, old_H=None, old_W=None
    ) -> Tuple[torch.Tensor, int]:
        new_grid, insert_idx = super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )
        self.position.invalidate()
        # old_H, old_W = self.state.H, self.state.W
        H, W = self.state.H, self.state.W
        xyz = self.position.features.detach().reshape(H, W, -1)
        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)
            xyz = xyz.permute(1, 0, 2)
        new_slice_pos = xyz[insert_idx: insert_idx + degree + 1]

        dist = torch.sqrt(
            distCUDA2(new_slice_pos.reshape(-1, 3)).clamp_min(1e-20)) * 0.5
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        # scaling_init = (dist)
        scaling_init = torch.log(
            (torch.stack([dist, dist, torch.ones_like(dist) * 1e-6], dim=-1)))[insert_idx: insert_idx + degree + 1].reshape(degree+1, -1,
                                                                                       3).detach().clone().contiguous()

        new_grid[insert_idx: insert_idx + degree + 1] = scaling_init.contiguous()

        # Shrink UV scales of newly inserted control points.
        # We're now in log-space directly, so subtract log(shrink_factor).
        # Use span-ratio-derived factor instead of magic 0.8*2.
        shrink_factor = self.subdivision_scale_factor * self.subdivision_count  # 0.8 * 2 = 1.6
        log_shrink = torch.tensor(shrink_factor, device=new_grid.device).log()
        new_slice = new_grid[insert_idx: insert_idx + degree + 1]
        new_slice[..., :2] = new_slice[..., :2] - log_shrink
        new_grid[insert_idx: insert_idx + degree + 1] = new_slice

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid, insert_idx
    def compute_inserted_grid2(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=None, blend_strength=0.3, use_blend=False,
    ) -> Tuple[torch.Tensor, int]:
        new_grid, insert_idx = super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        # # Shrink UV scales of newly inserted control points
        # new_slice = new_grid[insert_idx: insert_idx + degree + 1]
        # new_slice_uv = new_slice[..., :2].exp() / (0.8 * 2)
        # new_slice[..., :2] = new_slice_uv.log()
        # new_grid[insert_idx: insert_idx + degree + 1] = new_slice

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid, insert_idx

    # ------------------------------------------------------------------
    # Removal: expand neighbor scales to fill gap
    # ------------------------------------------------------------------

    def compute_removed_grid(
        self, direction, remove_idx, blend_radius=3,
        blend_strength=0.5, use_blend=False,
    ) -> torch.Tensor:
        new_ctrl = super().compute_removed_grid(
            direction, remove_idx,
            blend_radius=blend_radius,
            blend_strength=blend_strength,
            use_blend=False,
        )
        new_ctrl = self.apply_removal_compensation(
            new_ctrl, direction, remove_idx,
            blend_radius, expansion_factor=1.0 + blend_strength,
        )
        return new_ctrl

    def apply_removal_compensation(
        self,
        new_ctrl: torch.Tensor,
        direction: str,
        remove_idx: int,
        blend_radius: int = None,
        expansion_factor: float = 1.5,
    ) -> torch.Tensor:
        """
        Expand neighboring scales after knot removal to maintain surface coverage.

        Applies Gaussian-falloff expansion in log-space along the removal
        direction, analogous to 3DGS expanding Gaussian scales after pruning.
        """
        blend_radius = blend_radius if blend_radius is not None else self.state.degree

        if direction == 'u':
            new_H, new_W = self.state._H - 1, self.state._W
        else:
            new_H, new_W = self.state._H, self.state._W - 1

        ch = self.feature_channels
        device = self.state.device

        scales = new_ctrl.view(new_H, new_W, ch) if new_ctrl.ndim == 2 else new_ctrl.clone()
        scale_dim = 0 if direction == 'u' else 1

        for offset in range(blend_radius + 1):
            local_exp = 1.0 + (expansion_factor - 1.0) * torch.exp(
                torch.tensor(-0.5 * (offset / max(blend_radius / 2, 1)) ** 2, device=device)
            )
            log_exp = torch.log(local_exp)

            if direction == 'u':
                idx_before = remove_idx - 1 - offset
                idx_after = remove_idx - 1 + offset
                if 0 <= idx_before < new_H:
                    scales[idx_before, :, scale_dim] += log_exp
                if 0 <= idx_after < new_H:
                    scales[idx_after, :, scale_dim] += log_exp
            else:
                idx_before = remove_idx - 1 - offset
                idx_after = remove_idx - 1 + offset
                if 0 <= idx_before < new_W:
                    scales[:, idx_before, scale_dim] += log_exp
                if 0 <= idx_after < new_W:
                    scales[:, idx_after, scale_dim] += log_exp

        return scales.reshape(-1, ch) if new_ctrl.ndim == 2 else scales

    def _get_direction_weights(self, direction: str) -> torch.Tensor:
        """Per-axis scale reduction weights for subdivision."""
        if direction == 'u':
            weights = torch.tensor([1.0, 0.5, 0.3], device=self.control_features.device)
        else:
            weights = torch.tensor([0.5, 1.0, 0.3], device=self.control_features.device)
        return weights[:self.state.scaling_dims]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def capture_state(self) -> dict:
        state = super().capture_state()
        state['scaling_dims'] = self.state.scaling_dims
        return state

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        return super(ScalingControl, cls).from_state(
            state, model_state, basis, device,
        )