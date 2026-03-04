"""
Mapping utilities between sampling grid [Us, Vs] and control grid [H, W].

Provides principled aggregation of sample-space statistics to control-space
using Greville abscissae and B-spline basis weights.
"""

import torch
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MappingResult:
    """Result of mapping samples to control points."""
    # Aggregated values at control points
    values: torch.Tensor  # [H, W, C]
    # Per-control contribution counts (for averaging)
    counts: torch.Tensor  # [H, W]
    # Per-control visibility fraction
    visibility: Optional[torch.Tensor] = None  # [H, W]
    # Mapping indices for debugging
    u_indices: Optional[torch.Tensor] = None  # [Us, Vs]
    v_indices: Optional[torch.Tensor] = None  # [Us, Vs]


def compute_greville_abscissae(knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute Greville abscissae (knot averages) for control point UV locations.

    The Greville abscissa for control point i is:
        g_i = (t_{i+1} + t_{i+2} + ... + t_{i+p}) / p

    where p is the degree and t are knots.

    Args:
        knots: [n_knots] full knot vector (clamped)
        degree: B-spline degree

    Returns:
        greville:  [n_ctrl] UV coordinates for each control point
    """
    knots = knots.squeeze()
    n_knots = len(knots)
    n_ctrl = n_knots - degree - 1

    # Create index matrix for vectorized computation
    # For each control point i, sum knots[i+1] through knots[i+degree]
    indices = torch.arange(n_ctrl, device=knots.device).unsqueeze(1) + \
              torch.arange(1, degree + 1, device=knots.device).unsqueeze(0)

    # Gather and average
    greville = knots[indices].sum(dim=1) / degree

    return greville


class SamplingToControlMapper:
    """
    Maps statistics from sampling grid [Us, Vs] to control grid [H, W].

    Uses Greville abscissae to determine canonical UV positions of control points,
    then assigns each sample to nearby controls with distance-based weights.

    Key features:
    - Respects B-spline structure (not naive grid reduction)
    - Handles non-uniform sampling (adaptive UV intervals)
    - Provides weighted aggregation with visibility awareness
    - Supports both hard assignment (nearest) and soft assignment (weighted)
    """

    def __init__(
            self,
            state,  # ModelState
            knot_u: torch.Tensor,
            knot_v: torch.Tensor,
            degree: int = 3,
            assignment_mode: str = 'soft',  # 'nearest', 'bilinear', 'soft'
            soft_radius: float = 1.5,  # In units of average control spacing
    ):
        """
        Args:
            state: ModelState with H, W, Us, Vs
            knot_u, knot_v: Full knot vectors
            degree: B-spline degree
            assignment_mode: How to assign samples to controls
                - 'nearest': Each sample maps to single nearest control
                - 'bilinear': Each sample maps to 4 surrounding controls
                - 'soft':  Gaussian-weighted assignment within radius
            soft_radius: For 'soft' mode, radius in control-spacing units
        """
        self.state = state
        self.degree = degree
        self.assignment_mode = assignment_mode
        self.soft_radius = soft_radius
        self.device = state.device

        # Compute Greville abscissae for control points
        self._greville_u = compute_greville_abscissae(knot_u, degree)
        self._greville_v = compute_greville_abscissae(knot_v, degree)

        # Cache for mapping indices (invalidate when sampling changes)
        self._cached_mapping = None
        self._cache_valid = False

    @property
    def greville_u(self) -> torch.Tensor:
        """UV coordinates of control points in U direction."""
        return self._greville_u

    @property
    def greville_v(self) -> torch.Tensor:
        """UV coordinates of control points in V direction."""
        return self._greville_v

    @property
    def H(self) -> int:
        return self.state.H

    @property
    def W(self) -> int:
        return self.state.W

    @property
    def Us(self) -> int:
        return self.state.Us

    @property
    def Vs(self) -> int:
        return self.state.Vs

    def invalidate_cache(self):
        """Call when sampling intervals or knots change."""
        self._cache_valid = False
        self._cached_mapping = None

    def update_knots(self, knot_u: torch.Tensor, knot_v: torch.Tensor):
        """Update Greville abscissae after knot insertion/removal."""
        self._greville_u = compute_greville_abscissae(knot_u, self.degree)
        self._greville_v = compute_greville_abscissae(knot_v, self.degree)
        self.invalidate_cache()

    def compute_mapping(
            self,
            sample_u: torch.Tensor,  # [Us, Vs] or [Us]
            sample_v: torch.Tensor,  # [Us, Vs] or [Vs]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mapping from samples to control points.

        Returns dict with:
            - 'u_indices': [Us, Vs] nearest control index in U
            - 'v_indices':  [Us, Vs] nearest control index in V
            - 'weights': [Us, Vs, max_neighbors] assignment weights
            - 'neighbor_u':  [Us, Vs, max_neighbors] U indices of neighbors
            - 'neighbor_v': [Us, Vs, max_neighbors] V indices of neighbors
        """
        Us, Vs = self.Us, self.Vs
        H, W = self.H, self.W

        # Handle 1D vs 2D sample coordinates
        if sample_u.dim() == 1:
            sample_u = sample_u.unsqueeze(1).expand(Us, Vs)
        if sample_v.dim() == 1:
            sample_v = sample_v.unsqueeze(0).expand(Us, Vs)

        # Ensure same shape
        assert sample_u.shape == (Us, Vs), f"sample_u shape {sample_u.shape} != ({Us}, {Vs})"
        assert sample_v.shape == (Us, Vs), f"sample_v shape {sample_v.shape} != ({Us}, {Vs})"

        if self.assignment_mode == 'nearest':
            return self._compute_nearest_mapping(sample_u, sample_v)
        elif self.assignment_mode == 'bilinear':
            return self._compute_bilinear_mapping(sample_u, sample_v)
        elif self.assignment_mode == 'soft':
            return self._compute_soft_mapping(sample_u, sample_v)
        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

    def _compute_nearest_mapping(
            self,
            sample_u: torch.Tensor,
            sample_v: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Each sample maps to single nearest control point."""
        Us, Vs = self.Us, self.Vs
        H, W = self.H, self.W

        # Find nearest Greville index for each sample
        # sample_u: [Us, Vs], greville_u: [H]
        dist_u = (sample_u.unsqueeze(-1) - self._greville_u.unsqueeze(0).unsqueeze(0)).abs()
        u_indices = dist_u.argmin(dim=-1).clamp(0, H - 1)  # [Us, Vs]

        dist_v = (sample_v.unsqueeze(-1) - self._greville_v.unsqueeze(0).unsqueeze(0)).abs()
        v_indices = dist_v.argmin(dim=-1).clamp(0, W - 1)  # [Us, Vs]

        # Weights are all 1 for nearest
        weights = torch.ones(Us, Vs, 1, device=self.device)

        return {
            'u_indices': u_indices,
            'v_indices': v_indices,
            'weights': weights,
            'neighbor_u': u_indices.unsqueeze(-1),  # [Us, Vs, 1]
            'neighbor_v': v_indices.unsqueeze(-1),  # [Us, Vs, 1]
            'num_neighbors': 1
        }

    def _compute_bilinear_mapping(
            self,
            sample_u: torch.Tensor,
            sample_v: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Each sample maps to 4 surrounding control points with bilinear weights."""
        Us, Vs = self.Us, self.Vs
        H, W = self.H, self.W

        # Find surrounding indices
        # For each sample, find the interval [g_i, g_{i+1}] it falls into

        # U direction
        u_idx_low = torch.searchsorted(self._greville_u, sample_u.flatten(), side='right') - 1
        u_idx_low = u_idx_low.clamp(0, H - 2).view(Us, Vs)
        u_idx_high = (u_idx_low + 1).clamp(0, H - 1)

        # V direction
        v_idx_low = torch.searchsorted(self._greville_v, sample_v.flatten(), side='right') - 1
        v_idx_low = v_idx_low.clamp(0, W - 2).view(Us, Vs)
        v_idx_high = (v_idx_low + 1).clamp(0, W - 1)

        # Compute interpolation weights
        g_u_low = self._greville_u[u_idx_low]
        g_u_high = self._greville_u[u_idx_high]

        alpha_u = (sample_u - g_u_low) / (g_u_high - g_u_low + 1e-8)
        alpha_u = alpha_u.clamp(0, 1)

        g_v_low = self._greville_v[v_idx_low]
        g_v_high = self._greville_v[v_idx_high]
        alpha_v = (sample_v - g_v_low) / (g_v_high - g_v_low + 1e-8)
        alpha_v = alpha_v.clamp(0, 1)

        # Bilinear weights for 4 corners:  (1-au)(1-av), au(1-av), (1-au)av, au*av
        w00 = (1 - alpha_u) * (1 - alpha_v)
        w10 = alpha_u * (1 - alpha_v)
        w01 = (1 - alpha_u) * alpha_v
        w11 = alpha_u * alpha_v

        weights = torch.stack([w00, w10, w01, w11], dim=-1)  # [Us, Vs, 4]

        # Neighbor indices
        neighbor_u = torch.stack([u_idx_low, u_idx_high, u_idx_low, u_idx_high], dim=-1)
        neighbor_v = torch.stack([v_idx_low, v_idx_low, v_idx_high, v_idx_high], dim=-1)

        return {
            'u_indices': u_idx_low,  # Primary index
            'v_indices': v_idx_low,
            'weights': weights,
            'neighbor_u': neighbor_u,  # [Us, Vs, 4]
            'neighbor_v': neighbor_v,  # [Us, Vs, 4]
            'num_neighbors': 4
        }

    def _compute_soft_mapping(
            self,
            sample_u: torch.Tensor,
            sample_v: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Each sample maps to multiple controls within radius using Gaussian weights.
        More expensive but smoother gradients.
        """
        Us, Vs = self.Us, self.Vs
        H, W = self.H, self.W

        # Compute average spacing for radius scaling
        avg_spacing_u = 1.0 / (H - 1) if H > 1 else 1.0
        avg_spacing_v = 1.0 / (W - 1) if W > 1 else 1.0

        radius_u = self.soft_radius * avg_spacing_u
        radius_v = self.soft_radius * avg_spacing_v

        # For efficiency, use fixed max neighbors (e.g., 3x3 = 9)
        max_neighbors = 9

        # Find center index
        dist_u_all = (sample_u.unsqueeze(-1) - self._greville_u.view(1, 1, H)).abs()
        center_u = dist_u_all.argmin(dim=-1)  # [Us, Vs]

        dist_v_all = (sample_v.unsqueeze(-1) - self._greville_v.view(1, 1, W)).abs()
        center_v = dist_v_all.argmin(dim=-1)  # [Us, Vs]

        # Get 3x3 neighborhood around center
        offsets = torch.tensor([-1, 0, 1], device=self.device)

        neighbor_u = center_u.unsqueeze(-1) + offsets.view(1, 1, 3)  # [Us, Vs, 3]
        neighbor_u = neighbor_u.clamp(0, H - 1)

        neighbor_v = center_v.unsqueeze(-1) + offsets.view(1, 1, 3)  # [Us, Vs, 3]
        neighbor_v = neighbor_v.clamp(0, W - 1)

        # Expand to 3x3 grid
        neighbor_u_grid = neighbor_u.unsqueeze(-1).expand(Us, Vs, 3, 3).reshape(Us, Vs, 9)
        neighbor_v_grid = neighbor_v.unsqueeze(-2).expand(Us, Vs, 3, 3).reshape(Us, Vs, 9)

        # Compute Gaussian weights
        greville_u_neighbors = self._greville_u[neighbor_u_grid]  # [Us, Vs, 9]
        greville_v_neighbors = self._greville_v[neighbor_v_grid]

        dist_u = (sample_u.unsqueeze(-1) - greville_u_neighbors) / radius_u
        dist_v = (sample_v.unsqueeze(-1) - greville_v_neighbors) / radius_v

        dist_sq = dist_u ** 2 + dist_v ** 2
        weights = torch.exp(-0.5 * dist_sq)  # Gaussian

        # Normalize weights per sample
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            'u_indices': center_u,
            'v_indices': center_v,
            'weights': weights,  # [Us, Vs, 9]
            'neighbor_u': neighbor_u_grid,  # [Us, Vs, 9]
            'neighbor_v': neighbor_v_grid,
            'num_neighbors': 9
        }

    def aggregate_to_control(
            self,
            sample_values: torch.Tensor,  # [Us, Vs, C] or [Us*Vs, C]
            sample_u: torch.Tensor,
            sample_v: torch.Tensor,
            visibility: Optional[torch.Tensor] = None,  # [Us, Vs] or [Us*Vs]
            reduction: str = 'sum', # 'mean', 'sum', 'max',
            vis_thresh = 0.,


    ) -> MappingResult:
        """
        Aggregate sample-space values to control-space.

        Args:
            sample_values: Values at sample points [Us, Vs, C]
            sample_u, sample_v: Sample UV coordinates
            visibility: Optional visibility mask [Us, Vs]
            reduction: How to combine contributions ('mean', 'sum', 'max')

        Returns:
            MappingResult with aggregated values [H, W, C]
        """
        Us, Vs = self.Us, self.Vs
        H, W = self.H, self.W

        # Reshape inputs if needed
        if sample_values.dim() == 2:
            sample_values = sample_values.view(Us, Vs, -1)
        C = sample_values.shape[-1]

        if visibility is not None and visibility.dim() == 1:
            visibility = visibility.view(Us, Vs)

        # Get mapping
        mapping = self.compute_mapping(sample_u, sample_v)
        weights = mapping['weights']  # [Us, Vs, num_neighbors]
        neighbor_u = mapping['neighbor_u']  # [Us, Vs, num_neighbors]
        neighbor_v = mapping['neighbor_v']
        num_neighbors = mapping['num_neighbors']

        # Apply visibility to weights
        if visibility is not None:
            vis_weights = weights * visibility
        else:
            vis_weights = weights

        # Scatter-add to control grid
        control_values = torch.zeros(H, W, C, device=self.device)
        control_counts = torch.zeros(H, W, device=self.device)
        control_visibility = torch.zeros(H, W, device=self.device) if visibility is not None else None

        # Flatten for scatter
        flat_idx = neighbor_u * W + neighbor_v  # [Us, Vs, num_neighbors]

        for k in range(num_neighbors):
            idx_k = flat_idx[..., k].view(-1)  # [Us*Vs]
            weight_k = vis_weights[..., k].view(-1, 1)  # [Us*Vs, 1]

            # Weighted values
            weighted_vals = sample_values.view(-1, C) * weight_k

            # Scatter add
            control_values.view(-1, C).scatter_add_(0, idx_k.unsqueeze(-1).expand(-1, C), weighted_vals)
            control_counts.view(-1).scatter_add_(0, idx_k, weight_k.squeeze(-1))

            if visibility is not None:
                vis_k = (visibility.view(-1) * weights[..., k].view(-1))
                control_visibility.view(-1).scatter_add_(0, idx_k, vis_k)

        # Apply reduction
        if reduction == 'mean':
            control_values = control_values / (control_counts.unsqueeze(-1) + 1e-8)
        elif reduction == 'max':
            # For max, we'd need different logic - use mean as fallback
            control_values = control_values / (control_counts.unsqueeze(-1) + 1e-8)
        # 'sum' keeps values as-is

        # Visibility fraction
        if control_visibility is not None:
            total_possible = torch.zeros(H, W, device=self.device)
            for k in range(num_neighbors):
                idx_k = flat_idx[..., k].view(-1)
                total_possible.view(-1).scatter_add_(0, idx_k, weights[..., k].view(-1))
            control_visibility = control_visibility / (total_possible + 1e-8)
            control_visibility = control_visibility > vis_thresh
        return MappingResult(
            values=control_values.reshape(-1, C),
            counts=control_counts.reshape(-1, 1),
            visibility=control_visibility.reshape(-1, 1),
            u_indices=mapping['u_indices'],
            v_indices=mapping['v_indices']
        )

    def get_control_density(
            self,
            sample_u: torch.Tensor,
            sample_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute how many samples map to each control point.

        Returns:
            density: [H, W] count of samples per control
        """
        mapping = self.compute_mapping(sample_u, sample_v)

        H, W = self.H, self.W
        density = torch.zeros(H, W, device=self.device)

        flat_idx = mapping['neighbor_u'] * W + mapping['neighbor_v']
        weights = mapping['weights']

        for k in range(mapping['num_neighbors']):
            idx_k = flat_idx[..., k].view(-1)
            weight_k = weights[..., k].view(-1)
            density.view(-1).scatter_add_(0, idx_k, weight_k)

        return density


