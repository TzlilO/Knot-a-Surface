"""
Base control feature module for B-spline surface properties.

ControlFeature is the abstract base for all learnable surface attributes
(position, rotation, scaling, opacity, SH coefficients, NURBS weights).
It handles:
  - B-spline interpolation (separable einsum or BMM)
  - Cache management for interpolated values
  - Knot insertion/removal with neighbor blending
  - Serialization (capture/restore)
"""

from typing import TYPE_CHECKING, Tuple

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.basis.basis_matrix import SparseBasis

if TYPE_CHECKING:
    from modules.ModelState import ModelState
    from modules.basis import BasisFunction


class ControlFeature(nn.Module):
    """
    Base module for control-feature B-spline elements.

    Each subclass stores a grid of learnable parameters (control_features)
    of shape [H*W, C] that are interpolated via shared basis functions
    to produce per-sample values of shape [Us*Vs, C].
    """

    def __init__(
        self,
        state: 'ModelState',
        control_grid: torch.Tensor,
        basis: 'BasisFunction',
        *args,
        **kwargs,
    ):
        # === PE Configuration (kept for backward compat, rarely used) ===
        self.use_pe = kwargs.get('use_pe', False)
        self.pe_type = kwargs.get('pe_type', 'improved')
        self.pe_levels = kwargs.get('pe_levels', 4)
        self.pe_include_input = kwargs.get('pe_include_input', True)
        self.pe_log_sampling = kwargs.get('pe_log_sampling', True)
        self.pe_learnable_freqs = kwargs.get('pe_learnable_freqs', False)
        self.pe_use_residual = kwargs.get('pe_use_residual', True)
        self.pe_freq_scale = kwargs.get('pe_freq_scale', 0.5)
        self.pe_max_freq = kwargs.get('pe_max_freq', None)
        self.pe_num_features = kwargs.get('pe_num_features', 128)
        self.pe_sigma = kwargs.get('pe_sigma', 1.0)

        super().__init__()
        self.state = state
        self.name = kwargs.get('name', None)
        self.basis = basis
        self.is_rational = False
        self.device = state.device

        self.initialize_control_feature(control_grid)

        self._cache = None
        self.__blending_alpha = 0.0
        self._previous_cache = (
            torch.zeros(
                self.state.Us, self.state.Vs, control_grid.shape[-1],
                device=self.device,
            )
            if control_grid is not None
            else 0.0
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_control_feature(self, control_features):
        if control_features is None:
            self.control_features = None
            return

        self._original_channels = control_features.shape[-1]
        control_features = (
            control_features
            .reshape(-1, self._original_channels)
            .detach()
            .clone()
            .contiguous()
        )
        self.control_features = nn.Parameter(control_features, requires_grad=True)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @property
    def cache_valid(self):
        return self._cache is not None

    @property
    def cache(self):
        return self._cache

    def invalidate(self, hard: bool = False):
        self._cache = None
        if hard:
            self.__blending_alpha = 0.0
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------

    @property
    def features(self):
        """Return activated control grid [H, W, C] for interpolation."""
        return self.activation(
            self.control_features.view(
                self.state.H, self.state.W, self.control_features.shape[-1]
            )
        )

    @property
    def feature_channels(self):
        if self.use_pe:
            return self._original_channels
        return 0 if self.control_features is None else self.control_features.shape[-1]

    # ------------------------------------------------------------------
    # Activation (override in subclasses)
    # ------------------------------------------------------------------

    @property
    def activation(self):
        return torch.nn.Identity()

    @property
    def inverse_activation(self):
        return torch.nn.Identity()

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate_samples(self) -> torch.Tensor:
        """
        Interpolate control features via B-spline basis functions.

        Returns:
            Interpolated values [Us*Vs, C].
        """
        if self.cache_valid:
            return self._cache.reshape(-1, self.feature_channels)

        bu, bv = self.basis.bu, self.basis.bv

        if isinstance(bu, SparseBasis) and isinstance(bv, SparseBasis):
            prod = self._interpolate_gather(bu, bv, self.features).contiguous()
        elif self.state.use_bmm:
            prod = self._interpolate_bmm(bu, bv, self.features).contiguous()
        else:
            # oe.contract requires dense tensors
            bu_dense = bu.to_dense() if isinstance(bu, SparseBasis) else bu
            bv_dense = bv.to_dense() if isinstance(bv, SparseBasis) else bv
            prod = oe.contract(
                self.basis.contract_path,
                bu_dense,
                self.features,
                bv_dense,
                optimize=self.basis.optimal_path,
            ).contiguous()

        self._cache = prod
        return prod.reshape(-1, self.feature_channels)

    # Alias for backward compatibility with version 1 of the file
    forward = interpolate_samples

    def _interpolate_bmm(
        self,
        Bu: torch.Tensor,
        Bv: torch.Tensor,
        ctrl_points: torch.Tensor,
    ) -> torch.Tensor:
        """BMM-based separable interpolation: Bu @ P @ Bv^T."""
        H, W, C = ctrl_points.shape
        Us = Bu.shape[0]

        P_2d = ctrl_points.reshape(H, W * C)
        step1 = torch.mm(Bu, P_2d)                          # [Us, W*C]
        step1 = step1.reshape(Us, W, C).permute(0, 2, 1).reshape(Us * C, W)
        step2 = torch.mm(step1, Bv.T)                       # [Us*C, Vs]
        return step2.reshape(Us, C, -1).permute(0, 2, 1).contiguous()

    def _interpolate_gather(
        self,
        sbu: 'SparseBasis',
        sbv: 'SparseBasis',
        ctrl_points: torch.Tensor,
    ) -> torch.Tensor:
        """Gather-based separable interpolation exploiting B-spline local support.

        Avoids dense [N, num_control] matrix allocation by gathering only the
        p+1 non-zero control points for each sample.

        Args:
            sbu: SparseBasis for U — values/indices shape [Us, p+1].
            sbv: SparseBasis for V — values/indices shape [Vs, p+1].
            ctrl_points: Control-point grid [H, W, C].

        Returns:
            Interpolated surface [Us, Vs, C].
        """
        # Step 1: Gather along u — for each u sample gather p+1 rows from P.
        # ctrl_points[sbu.indices]: [Us, p+1, W, C]
        P_gathered_u = ctrl_points[sbu.indices]

        # Weighted sum along local u support: [Us, W, C]
        step1 = (sbu.values[:, :, None, None] * P_gathered_u).sum(dim=1)

        # Step 2: Gather along v — for each v sample gather p+1 columns.
        # step1[:, sbv.indices]: [Us, Vs, p+1, C]
        step1_gathered = step1[:, sbv.indices]

        # Weighted sum along local v support: [Us, Vs, C]
        result = (sbv.values[None, :, :, None] * step1_gathered).sum(dim=2)

        return result

    # ------------------------------------------------------------------
    # Blending (alpha for temporal smoothing after subdivision)
    # ------------------------------------------------------------------

    @property
    def blending_alpha(self):
        return self.__blending_alpha if self._previous_cache.sum() > 0.0 else 0.0

    @property
    def blending_beta(self):
        return 1 - self.blending_alpha

    def set_alpha(self, new_alpha):
        self.__blending_alpha = new_alpha
        self.invalidate(hard=True)

    # ------------------------------------------------------------------
    # Knot insertion
    # ------------------------------------------------------------------

    def compute_inserted_grid(
        self,
        direction: str,
        knots: torch.Tensor,
        degree: int,
        val: float,
        insert_idx: int,
        insertion_fn: callable,
        blend_radius: int = None,
        blend_strength: float = 0.3,
        use_blend: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute new control grid after Boehm knot insertion.

        Args:
            direction: 'u' or 'v'.
            knots: Current knot vector for the insertion direction.
            degree: B-spline degree.
            val: Parameter value of the new knot.
            insert_idx: Insertion index in the control grid.
            insertion_fn: Boehm insertion function (e.g. insert_knot_u).
            blend_radius: Neighbor blending radius.
            blend_strength: Blending intensity in [0, 1].
            use_blend: Whether to apply post-insertion blending.

        Returns:
            (new_grid, insert_idx) tuple.
        """
        H, W = self.state._H, self.state._W
        blend_radius = blend_radius if blend_radius is not None else degree

        if self.use_pe:
            decoded_grid = self.features.view(H, W, self._original_channels)
        else:
            decoded_grid = self.features.view(H, W, self.feature_channels)

        if direction == 'v':
            decoded_grid = decoded_grid.permute(1, 0, 2)

        new_grid, _ = insertion_fn(decoded_grid, knots, degree, val)
        new_grid = self.inverse_activation(new_grid)

        if use_blend:
            new_grid = self._apply_insertion_blending(
                new_grid, insert_idx, degree,
                blend_radius, blend_strength, direction='ortho',
            )

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid, insert_idx

    # ------------------------------------------------------------------
    # Knot removal
    # ------------------------------------------------------------------

    def compute_removed_grid(
        self,
        direction: str,
        remove_idx: int,
        blend_radius: int = None,
        blend_strength: float = 0.5,
        use_blend=False,
    ) -> torch.Tensor:
        """Compute control grid after removing a row/column."""
        blend_radius = blend_radius if blend_radius is not None else self.state.degree

        if self.use_pe:
            ch = self._original_channels
            ctrl_grid = self.features.view(self.H, self.W, ch)
        else:
            ch = self.feature_channels
            ctrl_grid = self.control_features.view(self.H, self.W, ch)

        if direction == 'v':
            ctrl_grid = ctrl_grid.permute(1, 0, 2)

        removed_row = ctrl_grid[remove_idx].clone()
        new_ctrl = torch.cat([
            ctrl_grid[:remove_idx],
            ctrl_grid[remove_idx + 1:],
        ], dim=0)

        if use_blend:
            new_ctrl = self._apply_removal_blending(
                new_ctrl, removed_row, remove_idx,
                blend_radius, blend_strength,
                direction=direction, original_size=self.H,
            )

        if self.use_pe:
            new_ctrl_flat = new_ctrl.view(-1, ch)
            return self._encode_to_pe(new_ctrl_flat)

        if direction == 'v':
            new_ctrl = new_ctrl.permute(1, 0, 2)
        return new_ctrl.reshape(-1, ch)

    # ------------------------------------------------------------------
    # Blending helpers (insertion / removal)
    # ------------------------------------------------------------------

    def _apply_insertion_blending(
        self,
        grid: torch.Tensor,
        insert_idx: int,
        degree: int,
        blend_radius: int,
        blend_strength: float,
        direction: str = 'ortho',
    ) -> torch.Tensor:
        """
        Smooth control points near an insertion site.

        Uses orthogonal Gaussian smoothing + neighbor interpolation
        to prevent tangent discontinuities after knot insertion.
        """
        if blend_strength <= 0 or blend_radius <= 0:
            return grid

        new_dim, other_dim, ch = grid.shape
        insert_start = insert_idx
        insert_end = min(insert_idx + degree + 1, new_dim)
        blended_grid = grid.clone()

        # --- Strategy 1: Orthogonal smoothing ---
        if direction == 'ortho' and other_dim > 1:
            kernel_size = min(5, other_dim)
            smoothing_zone = range(
                max(0, insert_start - blend_radius),
                min(new_dim, insert_end + blend_radius),
            )

            for idx in smoothing_zone:
                dist = min(abs(idx - insert_start), abs(idx - (insert_end - 1)))
                exponent = torch.tensor(
                    -0.5 * (dist / max(blend_radius / 2, 1)) ** 2,
                    device=self.device, dtype=grid.dtype,
                )
                local_weight = blend_strength * torch.exp(exponent)
                if local_weight > 0.01:
                    row = blended_grid[idx]
                    smoothed = self._smooth_along_dim(
                        row.unsqueeze(0), kernel_size=kernel_size
                    ).squeeze(0)
                    blended_grid[idx] = (1 - local_weight) * row + local_weight * smoothed

        # --- Strategy 2: Neighbor pull toward inserted region mean ---
        inserted_region = blended_grid[insert_start:insert_end]
        inserted_mean = inserted_region.mean(dim=0, keepdim=True)

        offsets = torch.arange(1, blend_radius + 1, device=self.device, dtype=grid.dtype)
        exponents = -0.5 * (offsets / max(blend_radius / 2, 1)) ** 2
        weights = blend_strength * 0.3 * torch.exp(exponents)

        for i, offset in enumerate(range(1, blend_radius + 1)):
            idx_before = insert_start - offset
            if 0 <= idx_before:
                neighbor = blended_grid[idx_before]
                blended_grid[idx_before] = (
                    (1 - weights[i]) * neighbor + weights[i] * inserted_mean.squeeze(0)
                )

            idx_after = insert_end - 1 + offset
            if idx_after < new_dim:
                neighbor = blended_grid[idx_after]
                blended_grid[idx_after] = (
                    (1 - weights[i]) * neighbor + weights[i] * inserted_mean.squeeze(0)
                )

        # --- Strategy 3: Feature-specific adjustments ---
        feature_type = getattr(self, 'type', 'generic')

        if feature_type == 'position':
            if insert_start > 0 and insert_end < new_dim:
                tangent_before = blended_grid[insert_start] - blended_grid[insert_start - 1]
                tangent_after = blended_grid[insert_end] - blended_grid[insert_end - 1]
                blended_grid[insert_start] = blended_grid[insert_start - 1] + (
                    0.7 * tangent_before + 0.3 * tangent_after
                )
        elif feature_type == 'scaling':
            scale_smooth_weight = blend_strength * 0.5
            for idx in range(insert_start, insert_end):
                if 0 < idx < new_dim - 1:
                    avg = (blended_grid[idx - 1] + blended_grid[idx + 1]) / 2
                    blended_grid[idx] = (
                        (1 - scale_smooth_weight) * blended_grid[idx]
                        + scale_smooth_weight * avg
                    )

        return blended_grid

    def _apply_removal_blending(
        self,
        grid: torch.Tensor,
        removed_slice: torch.Tensor,
        remove_idx: int,
        blend_radius: int,
        blend_strength: float,
        direction: str,
        original_size: int,
    ) -> torch.Tensor:
        """
        Pull neighbors toward the gap after row/column removal.

        Uses Gaussian-weighted interpolation to prevent surface cracks.
        """
        if blend_strength <= 0 or blend_radius <= 0:
            return grid

        device = grid.device
        blended_grid = grid.clone()

        if direction == 'u':
            new_H, W, ch = grid.shape
            gap_reference = removed_slice

            # Blend BEFORE gap
            for offset in range(1, blend_radius + 1):
                idx = remove_idx - offset
                if idx < 0 or idx >= new_H:
                    continue
                exp = torch.tensor(
                    -0.5 * (offset / max(blend_radius / 2, 1)) ** 2,
                    device=device, dtype=grid.dtype,
                )
                weight = blend_strength * torch.exp(exp) * 0.5
                target = (
                    blended_grid[min(remove_idx, new_H - 1)]
                    if remove_idx < new_H else gap_reference
                )
                blended_grid[idx] = (1 - weight) * blended_grid[idx] + weight * target

            # Blend AFTER gap
            for offset in range(1, blend_radius + 1):
                idx = remove_idx - 1 + offset
                if idx < 0 or idx >= new_H or idx == remove_idx - 1:
                    continue
                exp = torch.tensor(
                    -0.5 * (offset / max(blend_radius / 2, 1)) ** 2,
                    device=device, dtype=grid.dtype,
                )
                weight = blend_strength * torch.exp(exp) * 0.5
                target = (
                    blended_grid[max(remove_idx - 1, 0)]
                    if remove_idx - 1 >= 0 else gap_reference
                )
                blended_grid[idx] = (1 - weight) * blended_grid[idx] + weight * target

            # Close the gap between directly adjacent rows
            if remove_idx > 0 and remove_idx <= new_H:
                bi = remove_idx - 1
                ai = remove_idx if remove_idx < new_H else new_H - 1
                if 0 <= bi < new_H and 0 <= ai < new_H and bi != ai:
                    gw = blend_strength * 0.4
                    rb, ra = blended_grid[bi].clone(), blended_grid[ai].clone()
                    blended_grid[bi] = (1 - gw) * rb + gw * ra
                    blended_grid[ai] = (1 - gw) * ra + gw * rb

        else:  # direction == 'v'
            H, new_W, ch = grid.shape
            gap_reference = removed_slice

            for offset in range(1, blend_radius + 1):
                idx = remove_idx - offset
                if idx < 0 or idx >= new_W:
                    continue
                exp = torch.tensor(
                    -0.5 * (offset / max(blend_radius / 2, 1)) ** 2,
                    device=device, dtype=grid.dtype,
                )
                weight = blend_strength * torch.exp(exp) * 0.5
                target = (
                    blended_grid[:, min(remove_idx, new_W - 1)]
                    if remove_idx < new_W else gap_reference
                )
                blended_grid[:, idx] = (1 - weight) * blended_grid[:, idx] + weight * target

            for offset in range(1, blend_radius + 1):
                idx = remove_idx - 1 + offset
                if idx < 0 or idx >= new_W or idx == remove_idx - 1:
                    continue
                exp = torch.tensor(
                    -0.5 * (offset / max(blend_radius / 2, 1)) ** 2,
                    device=device, dtype=grid.dtype,
                )
                weight = blend_strength * torch.exp(exp) * 0.5
                target = (
                    blended_grid[:, max(remove_idx - 1, 0)]
                    if remove_idx - 1 >= 0 else gap_reference
                )
                blended_grid[:, idx] = (1 - weight) * blended_grid[:, idx] + weight * target

            if remove_idx > 0 and remove_idx <= new_W:
                bi = remove_idx - 1
                ai = remove_idx if remove_idx < new_W else new_W - 1
                if 0 <= bi < new_W and 0 <= ai < new_W and bi != ai:
                    gw = blend_strength * 0.4
                    cb, ca = blended_grid[:, bi].clone(), blended_grid[:, ai].clone()
                    blended_grid[:, bi] = (1 - gw) * cb + gw * ca
                    blended_grid[:, ai] = (1 - gw) * ca + gw * cb

        return blended_grid

    def _smooth_along_dim(
        self,
        tensor: torch.Tensor,
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """1D Gaussian smoothing along last spatial dimension."""
        if tensor.shape[1] < kernel_size:
            return tensor

        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        pad = kernel_size // 2

        sigma = kernel_size / 6.0
        x = torch.arange(kernel_size, device=tensor.device, dtype=torch.float32) - pad
        weights = torch.exp(-0.5 * (x / sigma) ** 2)
        weights = weights / weights.sum()

        batch, dim, ch = tensor.shape
        reshaped = tensor.permute(0, 2, 1).reshape(batch * ch, 1, dim)
        padded = F.pad(reshaped, (pad, pad), mode='replicate')
        kernel = weights.view(1, 1, kernel_size)
        smoothed = F.conv1d(padded, kernel)
        return smoothed.view(batch, ch, dim).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Shape / dimension helpers
    # ------------------------------------------------------------------

    @property
    def total_parameters(self):
        with torch.no_grad():
            return self.control_features.view(-1).shape[0]

    @property
    def H(self):
        return self.state.H

    @property
    def W(self):
        return self.state.W

    @property
    def control_grid_shape(self):
        return self.state.H, self.state.W, -1

    @property
    def sampling_grid_shape(self):
        return self.Us, self.Vs, -1

    @property
    def num_patches_u(self):
        return self.state.num_patches_u

    @property
    def num_patches_v(self):
        return self.state.num_patches_v

    @property
    def Us(self):
        return self.state.Us

    @property
    def Vs(self):
        return self.state.Vs

    @property
    def uv2xyz(self):
        return self.state.uv2xyz

    def to_grid_view(self) -> torch.Tensor:
        return self.control_features.view(self.state.H, self.state.W, -1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def capture_state(self) -> dict:
        state = {
            'control_features': (
                None if self.control_features is None
                else self.control_features.data.clone().cpu()
            ),
            'name': self.name,
            'use_pe': getattr(self, 'use_pe', False),
            'is_rational': self.is_rational,
            'cache': None,
            'cache_valid': self.cache_valid,
            'blending_alpha': self.__blending_alpha,
        }
        if self._cache is not None and self.cache_valid:
            state['cache'] = self._cache.clone().cpu()
        return state

    @classmethod
    def from_state(
        cls,
        state: dict,
        model_state: 'ModelState',
        basis: 'BasisFunction',
        device: str = 'cuda',
        **kwargs,
    ) -> 'ControlFeature':
        control_features = state['control_features']
        if control_features is not None:
            control_features = control_features.detach().to(device)

        instance = cls(
            model_state, control_features, basis,
            name=state.get('name'), use_pe=False, **kwargs,
        )
        instance.is_rational = state.get('is_rational', False)
        instance.__blending_alpha = state.get('blending_alpha', 0.0)

        cache = state.get('cache')
        instance._cache = cache.to(device) if cache is not None else None
        instance._previous_cache = None
        return instance