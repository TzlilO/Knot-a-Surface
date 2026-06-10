"""
Base control feature module for B-spline surface properties.

ControlFeature is the abstract base for all learnable surface attributes
(position, rotation, scaling, opacity, SH coefficients, NURBS weights).
It handles:
  - B-spline interpolation (fused CUDA kernel, einsum fallback)
  - Knot insertion/removal with neighbor blending
  - Serialization (capture/restore)

No lazy caching: every access recomputes from the current parameters.
The fused kernel makes recomputation cheap, and stale-cache bugs are
impossible by construction.
"""

import os
from typing import TYPE_CHECKING, Tuple

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from modules.ModelState import ModelState
    from modules.basis import BasisFunction

# Fused CUDA local-support evaluation (optional; einsum fallback).
try:
    import bspline_eval as _bse
    _FUSED_OK = os.environ.get("KNOTS_FUSED", "1") != "0"
except ImportError:
    print("Warning: bspline_eval not found, falling back to slower PyTorch evaluation.")
    _bse = None
    _FUSED_OK = False


def compact_basis_windows(b, db, d2b, n_ctrl):
    """
    Dense [M, n_ctrl] basis value/derivative rows -> compact [M, 4] windows
    sharing ONE span per sample (span from the union support of all three
    arrays: at a knot the VALUE of the leftmost basis vanishes while its
    DERIVATIVE does not, so per-array spans would disagree).
    """
    support = b.abs() + db.abs() + d2b.abs()
    spans = (
        (support > 1e-12).to(torch.int64).argmax(dim=1).clamp(max=n_ctrl - 4)
    )
    cols = spans.unsqueeze(1) + torch.arange(4, device=b.device).unsqueeze(0)
    return (
        torch.gather(b, 1, cols),
        torch.gather(db, 1, cols),
        torch.gather(d2b, 1, cols),
        spans,
    )


def fused_available(state, basis, tensor) -> bool:
    """True when the fused CUDA kernel applies to this evaluation."""
    return (
        _FUSED_OK
        and tensor.is_cuda
        and basis.bu.is_cuda
        and state.H >= 4 and state.W >= 4          # degree-3 kernel
        and not getattr(state, 'flatten_uv', False)  # outer-product grid only
        and not getattr(state, 'full_basis', False)
        and getattr(basis, 'contract_path', 'uh,hwc,vw->uvc') == 'uh,hwc,vw->uvc'
    )


def fused_contract(grid, basis, H, W):
    """[H,W,C] -> [5, Us, Vs, C] contraction sums (S, Su, Sv, Suu, Svv)."""
    bu, dbu, dbuu, su = compact_basis_windows(basis.bu, basis.dbu, basis.dbuu, H)
    bv, dbv, dbvv, sv = compact_basis_windows(basis.bv, basis.dbv, basis.dbvv, W)
    return _bse.tp_contract(grid, bu, dbu, dbuu, bv, dbv, dbvv, su, sv)


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
        self.pe_type = kwargs.get('pe_type', None)#'improved')
        self.pe_levels = kwargs.get('pe_levels', 4)
        self.pe_include_input = kwargs.get('pe_include_input', False)
        self.pe_log_sampling = kwargs.get('pe_log_sampling', False)
        self.pe_learnable_freqs = kwargs.get('pe_learnable_freqs', False)
        self.pe_use_residual = kwargs.get('pe_use_residual', False)
        self.pe_freq_scale = kwargs.get('pe_freq_scale', 0.5)
        self.pe_max_freq = kwargs.get('pe_max_freq', None)
        self.pe_num_features = kwargs.get('pe_num_features', 1)
        self.pe_sigma = kwargs.get('pe_sigma', 1.0)

        super().__init__()
        self.state = state
        self.name = kwargs.get('name', None)
        self.basis = basis
        self.is_rational = False
        self.device = state.device

        self.initialize_control_feature(control_grid)

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

    def invalidate(self, hard: bool = False):
        """No caches exist; kept for API compatibility (subclasses hook it)."""
        pass


    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------

    @property
    def features(self):
        """Return activated control grid [H, W, C]."""
        return self.activation(
            self.control_features.view(
                self.state.H, self.state.W, self.control_features.shape[-1]
            )
        )

    @property
    def raw_features(self):
        """Raw (parameter-space) control grid [H, W, C] for interpolation."""
        return self.control_features.view(
            self.state.H, self.state.W, self.control_features.shape[-1]
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

    def forward(self) -> torch.Tensor:
        """
        Interpolate control features via B-spline basis functions.

        Interpolation happens in RAW parameter space, activation afterwards
        (paper Eq. 7: σ(Σ N·x̃), not Σ N·σ(x̃)). This also matches Boehm
        knot insertion, which operates on raw parameters.

        Returns:
            Interpolated values [Us*Vs, C].
        """
        grid = self.raw_features
        if fused_available(self.state, self.basis, grid):
            # Fused local-support CUDA kernel; [0] = value contraction.
            raw = fused_contract(grid, self.basis, self.state.H, self.state.W)[0]
        # elif self.state.use_bmm:
        #     raw = self._interpolate_bmm(self.basis.bu, self.basis.bv, grid)
        else:
            raw = oe.contract(
                self.basis.contract_path,
                self.basis.bu,
                grid,
                self.basis.bv,
            )

        prod = self.activation(raw.contiguous())
        return prod.reshape(-1, self.feature_channels)
    #
    # def _interpolate_bmm(
    #     self,
    #     Bu: torch.Tensor,
    #     Bv: torch.Tensor,
    #     ctrl_points: torch.Tensor,
    # ) -> torch.Tensor:
    #     """BMM-based separable interpolation: Bu @ P @ Bv^T."""
    #     H, W, C = ctrl_points.shape
    #     Us = Bu.shape[0]
    #
    #     P_2d = ctrl_points.reshape(H, W * C)
    #     step1 = torch.mm(Bu, P_2d)                          # [Us, W*C]
    #     step1 = step1.reshape(Us, W, C).permute(0, 2, 1).reshape(Us * C, W)
    #     step2 = torch.mm(step1, Bv.T)                       # [Us*C, Vs]
    #     return step2.reshape(Us, C, -1).permute(0, 2, 1).contiguous()

    # ------------------------------------------------------------------
    # Blending (alpha for temporal smoothing after subdivision)
    # ------------------------------------------------------------------

    blending_alpha = 0.0
    blending_beta = 1.0

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
            old_H=None, old_W=None

    ) -> Tuple[torch.Tensor, int]:
        """
        Compute new control grid after Boehm knot insertion.

        Operates in RAW PARAMETER SPACE to preserve mathematical exactness
        of Boehm's algorithm. For identity-activated features (position, SH),
        this is equivalent to the previous activated-space insertion.
        For non-linear activations (exp, sigmoid), this avoids the
        systematic bias introduced by activation→insertion→inverse_activation.
        """
        H, W = self.state._H, self.state._W
        blend_radius = blend_radius if blend_radius is not None else degree

        # KEY FIX: Use raw control_features (parameter space), NOT self.features (activated space)
        if self.use_pe:
            raw_grid = self.features.view(H, W, self._original_channels)
        else:
            ch = self.feature_channels
            raw_grid = self.control_features.view(H, W, ch)

        if direction == 'v':
            raw_grid = raw_grid.permute(1, 0, 2)

        # Boehm insertion in parameter space — exact for affine activations,
        # correct convex combination in parameter space for non-linear ones.
        new_grid, _ = insertion_fn(raw_grid, knots, degree, val)
        # No inverse_activation needed: we're already in parameter space.

        if use_blend:
            new_grid = self._apply_insertion_blending(
                new_grid, insert_idx, degree,
                blend_radius, blend_strength, direction='ortho',
            )

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid.contiguous(), insert_idx
    def set_position(self, position):
        setattr(self, 'position', position)

    def compute_inserted_grid2(
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
        }
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
        return instance