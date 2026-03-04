"""
View-Dependent Adaptive Tessellation for B-Spline/NURBS Surfaces

Based on: Chhugani & Kumar, "View-dependent Adaptive Tessellation of Spline
Surfaces" (2001), adapted for Gaussian Splatting sample placement.

Key adaptations from the paper:
1. Flatness criterion (§3.1) adapted for B-spline control polygons
2. Screen-space projected error bounds for view-dependent density
3. Separable 1D interval output (compatible with existing BasisFunction)
4. Optional full 2D quadtree for uv_grid mode

The core idea: sample densely where the surface curves (high 2nd derivative)
and where those curves face the camera (high projected curvature), sample
sparsely on flat regions or regions facing away.

ARCHITECTURAL CONSTRAINT:
    This module produces 1D separable intervals (u_intervals, v_intervals)
    that must have EXACTLY state.Us and state.Vs elements respectively.
    This is non-negotiable — the BasisFunction, ControlFeature.interpolate_samples(),
    and all downstream einsum paths depend on these exact dimensions.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum
import math


class RefinementCriterion(Enum):
    """Which criterion drives the adaptive tessellation."""
    FLATNESS = "flatness"           # Chhugani §3.1: deviation from tangent plane
    SCREEN_SPACE = "screen_space"   # Projected curvature × solid angle
    HYBRID = "hybrid"               # Both combined (recommended)


@dataclass
class TessellationParams:
    """
    Configuration for view-dependent tessellation.

    These map to the paper's parameters but are renamed for clarity
    in the Gaussian splatting context.
    """
    # --- Core criterion ---
    criterion: RefinementCriterion = RefinementCriterion.HYBRID

    # --- Flatness threshold (Chhugani §3.1) ---
    # Maximum allowed deviation (in world units) of the surface from its
    # tangent plane before requiring denser sampling.
    # Lower = more samples in curved regions. Typical: 0.01-0.1 × scene_extent
    flatness_threshold: float = 0.01

    # --- Screen-space threshold ---
    # Maximum allowed projected error in pixels before requiring denser sampling.
    screen_pixel_threshold: float = 2.0

    # --- Density bounds ---
    # Minimum samples per knot span (prevents degenerate under-sampling)
    min_samples_per_span: int = 1
    # Maximum samples per knot span (prevents OOM on high-curvature regions)
    max_samples_per_span: int = 32

    # --- Temporal smoothing ---
    # Smoothing factor for density between frames.
    # 0.0 = no smoothing (instant), 1.0 = never change.
    temporal_alpha: float = 0.3

    # --- Frustum culling ---
    # Margin (in NDC) outside the frustum before a UV region is considered
    # invisible. Negative = aggressive culling, Positive = conservative.
    frustum_margin: float = 0.05

    # --- Backface handling ---
    # Weight factor for backfacing regions. 0.0 = cull completely,
    # 1.0 = treat same as frontfacing.
    backface_density_factor: float = 0.1

    # --- Curvature weighting ---
    # Exponent for curvature-to-density mapping.
    # density ∝ curvature^exponent. Higher = more aggressive concentration.
    curvature_exponent: float = 0.5

    # --- Floor for minimum density in visible regions ---
    min_visible_density_ratio: float = 0.2


class ViewDependentTessellator:
    """
    Computes adaptive UV sampling intervals based on view-dependent
    surface analysis.

    This is NOT a neural module — it runs @torch.no_grad() and produces
    deterministic sample placements. Tessellation decisions are discrete
    (how many samples where) and should not be part of the gradient
    computation.

    CRITICAL INVARIANT:
        The output intervals always have EXACTLY (Us, Vs) samples.
        The adaptivity is in WHERE those samples are placed, not HOW MANY.
        This is because the entire downstream pipeline (BasisFunction shapes,
        einsum paths, ControlFeature caches) is hardcoded to (Us, Vs).

    Architecture:
        1. Evaluate surface curvature at control point level
        2. Project curvature into screen space using camera parameters
        3. Compute per-span density requirements
        4. Generate non-uniform UV intervals that concentrate the fixed
           budget of Us/Vs samples where the projected curvature is highest
    """

    def __init__(self, config: Optional[TessellationParams] = None):
        self.config = config or TessellationParams()
        # Cache for temporal smoothing
        self._prev_density_u: Optional[torch.Tensor] = None
        self._prev_density_v: Optional[torch.Tensor] = None

    def reset(self):
        """Reset temporal caches. Call after subdivision/pruning."""
        self._prev_density_u = None
        self._prev_density_v = None


    @torch.no_grad()
    def compute_adaptive_intervals(
        self,
        surface,
        camera,
        Us: Optional[int] = None,
        Vs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main entry point: compute view-dependent UV intervals.

        Implements a simplified version of Chhugani's algorithm:
        1. Compute per-span flatness (world-space deviation from tangent plane)
        2. Project deviation into screen space using camera intrinsics
        3. Determine per-span relative density
        4. Generate non-uniform intervals that redistribute the FIXED budget
           of Us/Vs samples according to density

        Args:
            surface: SplineModel instance (must have valid control points)
            camera: Viewpoint camera with projection parameters
            Us: Number of U samples (default: surface.state.Us)
            Vs: Number of V samples (default: surface.state.Vs)

        Returns:
            (u_intervals, v_intervals): each a sorted 1D tensor in [0, 1],
            with EXACTLY Us and Vs elements respectively.
        """
        Us = Us if Us is not None else surface.state.Us
        Vs = Vs if Vs is not None else surface.state.Vs

        device = surface.position.control_features.device
        H, W = surface.state.H, surface.state.W

        knots_u = surface.knot_u.forward().detach()
        knots_v = surface.knot_v.forward().detach()

        # =====================================================================
        # Step 1: Compute per-span flatness in control polygon
        # (Chhugani §3.1 — adapted for B-spline control net)
        # =====================================================================
        ctrl_xyz = surface.position() #.control_features.detach().view(H, W, 3)

        flatness_u = self._control_polygon_flatness(ctrl_xyz, direction='u')
        flatness_v = self._control_polygon_flatness(ctrl_xyz, direction='v')

        # Map to knot spans
        unique_u = torch.unique_consecutive(knots_u)
        unique_v = torch.unique_consecutive(knots_v)
        num_spans_u = max(len(unique_u) - 1, 1)
        num_spans_v = max(len(unique_v) - 1, 1)

        span_flatness_u = self._map_flatness_to_spans(
            flatness_u, num_spans_u, H, direction='u', device=device
        )
        span_flatness_v = self._map_flatness_to_spans(
            flatness_v, num_spans_v, W, direction='v', device=device
        )

        # =====================================================================
        # Step 2: View-dependent weighting
        # =====================================================================
        if self.config.criterion in (
            RefinementCriterion.SCREEN_SPACE,
            RefinementCriterion.HYBRID
        ):
            view_weight_u, view_weight_v = self._compute_view_weights(
                ctrl_xyz, camera, num_spans_u, num_spans_v, H, W
            )
        else:
            view_weight_u = torch.ones(num_spans_u, device=device)
            view_weight_v = torch.ones(num_spans_v, device=device)

        # =====================================================================
        # Step 3: Combine into per-span density requirement
        # =====================================================================
        if self.config.criterion == RefinementCriterion.FLATNESS:
            density_u = span_flatness_u / (self.config.flatness_threshold + 1e-8)
            density_v = span_flatness_v / (self.config.flatness_threshold + 1e-8)
        elif self.config.criterion == RefinementCriterion.SCREEN_SPACE:
            density_u = view_weight_u
            density_v = view_weight_v
        else:  # HYBRID
            density_u = (
                span_flatness_u / (self.config.flatness_threshold + 1e-8)
            ) * view_weight_u
            density_v = (
                span_flatness_v / (self.config.flatness_threshold + 1e-8)
            ) * view_weight_v

        # Apply curvature exponent
        density_u = density_u.clamp(min=1e-8).pow(self.config.curvature_exponent)
        density_v = density_v.clamp(min=1e-8).pow(self.config.curvature_exponent)

        # Floor: ensure minimum density in visible regions
        density_u = density_u.clamp(min=self.config.min_visible_density_ratio)
        density_v = density_v.clamp(min=self.config.min_visible_density_ratio)

        # =====================================================================
        # Step 4: Temporal smoothing
        # =====================================================================
        if (self._prev_density_u is not None and
                self._prev_density_u.shape == density_u.shape):
            alpha = self.config.temporal_alpha
            density_u = alpha * self._prev_density_u + (1 - alpha) * density_u
            density_v = alpha * self._prev_density_v + (1 - alpha) * density_v

        self._prev_density_u = density_u.clone()
        self._prev_density_v = density_v.clone()

        # =====================================================================
        # Step 5: Generate non-uniform intervals from density
        # Output has EXACTLY Us and Vs samples.
        # =====================================================================
        u_intervals = self._density_to_intervals(
            density_u, unique_u, Us, device
        )
        v_intervals = self._density_to_intervals(
            density_v, unique_v, Vs, device
        )

        return u_intervals, v_intervals

    # =========================================================================
    # Flatness computation (Chhugani §3.1)
    # =========================================================================

    def _control_polygon_flatness(
        self,
        ctrl_xyz: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        """
        Compute flatness deviation of the control polygon.

        From Chhugani §3.1: The flatness of a B-spline patch is measured by
        the maximum distance of interior control points from the line
        connecting their neighbors.

        For a row of control points P_{i-1}, P_i, P_{i+1}, the deviation is:
            |P_i - (P_{i-1} + P_{i+1}) / 2|

        This is a second-difference measure that bounds actual surface
        deviation via the convex hull property of B-splines.

        Args:
            ctrl_xyz: [H, W, 3] control point positions
            direction: 'u' or 'v'

        Returns:
            Deviation values. For 'u': [H-2, W], for 'v': [H, W-2]
        """
        if direction == 'u':
            if ctrl_xyz.shape[0] < 3:
                return torch.zeros(0, ctrl_xyz.shape[1], device=ctrl_xyz.device)
            mid = (ctrl_xyz[:-2] + ctrl_xyz[2:]) / 2.0
            deviation = (ctrl_xyz[1:-1] - mid).norm(dim=-1)
            return deviation
        else:
            if ctrl_xyz.shape[1] < 3:
                return torch.zeros(ctrl_xyz.shape[0], 0, device=ctrl_xyz.device)
            mid = (ctrl_xyz[:, :-2] + ctrl_xyz[:, 2:]) / 2.0
            deviation = (ctrl_xyz[:, 1:-1] - mid).norm(dim=-1)
            return deviation

    def _map_flatness_to_spans(
        self,
        flatness: torch.Tensor,
        num_spans: int,
        ctrl_dim: int,
        direction: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Map control-point-level flatness to knot span level.

        Each knot span is influenced by (degree+1) control points.
        We take the max flatness among control points that map to each span.

        For the separable approximation, we first max-pool over the
        orthogonal direction, then distribute to spans.

        Args:
            flatness: [H-2, W] or [H, W-2] depending on direction
            num_spans: Number of knot spans
            ctrl_dim: H or W (number of control points in this direction)
            direction: 'u' or 'v'
            device: torch device

        Returns:
            [num_spans] per-span maximum flatness
        """
        if flatness.numel() == 0 or num_spans == 0:
            return torch.ones(max(num_spans, 1), device=device)

        # Max-pool over orthogonal direction
        if direction == 'u':
            flatness_1d = flatness.max(dim=1).values  # [H-2]
        else:
            flatness_1d = flatness.max(dim=0).values  # [W-2]

        n_flat = flatness_1d.shape[0]
        if n_flat == 0:
            return torch.ones(num_spans, device=device)

        # Map flatness indices to span indices
        # flatness[i] corresponds to control point i+1 (since we skip boundaries)
        # Use proportional mapping
        span_flatness = torch.zeros(num_spans, device=device)
        ratio = n_flat / max(num_spans, 1)

        for s in range(num_spans):
            start = max(0, int(s * ratio))
            end = min(n_flat, int((s + 1) * ratio) + 1)
            if start < end:
                span_flatness[s] = flatness_1d[start:end].max()
            else:
                span_flatness[s] = flatness_1d[min(start, n_flat - 1)]

        return span_flatness

    # =========================================================================
    # View-dependent weighting
    # =========================================================================

    def _compute_view_weights(
        self,
        ctrl_xyz: torch.Tensor,
        camera,
        num_spans_u: int,
        num_spans_v: int,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute view-dependent weight for each knot span.

        Combines:
        1. Distance attenuation: closer regions need more samples
        2. Foreshortening: regions facing the camera need more samples
        3. Frustum visibility: regions outside frustum get low weight
        """
        device = ctrl_xyz.device

        # Camera position
        cam_pos = camera.camera_center.to(device)  # [3]

        # View directions from each control point to camera
        view_dirs = cam_pos.unsqueeze(0).unsqueeze(0) - ctrl_xyz  # [H, W, 3]
        distances = view_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        view_dirs_normalized = view_dirs / distances

        # Approximate surface normals from control polygon
        normals = self._control_polygon_normals(ctrl_xyz)

        # Foreshortening: |cos(angle between normal and view direction)|
        cos_angle = (normals * view_dirs_normalized).sum(dim=-1).abs()

        # Distance attenuation: 1/d² (normalized)
        dist_weight = 1.0 / (distances.squeeze(-1) ** 2 + 1e-8)
        dist_weight = dist_weight / (dist_weight.max() + 1e-8)

        # Frustum check
        frustum_mask = self._approximate_frustum_check(ctrl_xyz, camera)

        # Backface handling
        backface_weight = torch.where(
            cos_angle > 0.0,
            torch.ones_like(cos_angle),
            torch.full_like(cos_angle, self.config.backface_density_factor)
        )

        # Combined per-control-point weight
        combined = cos_angle * dist_weight * frustum_mask * backface_weight

        # Project to span weights (separable approximation via max-pooling)
        weight_u = self._project_weights_to_spans(combined, num_spans_u, H, 'u')
        weight_v = self._project_weights_to_spans(combined, num_spans_v, W, 'v')

        return weight_u, weight_v

    def _control_polygon_normals(
        self, ctrl_xyz: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate surface normals from control polygon via finite differences.
        """
        H, W, _ = ctrl_xyz.shape

        # Tangent in U (forward difference, replicate boundary)
        du = torch.zeros_like(ctrl_xyz)
        du[:-1] = ctrl_xyz[1:] - ctrl_xyz[:-1]
        du[-1] = du[-2] if H > 1 else du[-1]

        # Tangent in V
        dv = torch.zeros_like(ctrl_xyz)
        dv[:, :-1] = ctrl_xyz[:, 1:] - ctrl_xyz[:, :-1]
        dv[:, -1] = dv[:, -2] if W > 1 else dv[:, -1]

        normals = torch.cross(du, dv, dim=-1)
        return F.normalize(normals, dim=-1, eps=1e-8)

    def _approximate_frustum_check(
        self,
        ctrl_xyz: torch.Tensor,
        camera,
    ) -> torch.Tensor:
        """
        Approximate frustum visibility for control points.

        Returns a soft mask: 1.0 = fully inside, 0.0 = fully outside.
        Uses a sigmoid transition at the boundary.
        """
        device = ctrl_xyz.device
        H, W, _ = ctrl_xyz.shape

        try:
            # Get camera transforms — handle both tensor and numpy R/T
            R = camera.R
            T = camera.T
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R, device=device, dtype=torch.float32)
            else:
                R = R.to(device).float()
            if not isinstance(T, torch.Tensor):
                T = torch.tensor(T, device=device, dtype=torch.float32)
            else:
                T = T.to(device).float()

            # Transform to camera space
            pts_flat = ctrl_xyz.reshape(-1, 3)
            pts_cam = pts_flat @ R.T + T.unsqueeze(0)

            # In front of camera
            in_front = (pts_cam[:, 2] > 0.01).float()

            # Project to image coordinates
            # Handle different camera attribute names
            fx = getattr(camera, 'Fx', None)
            if fx is None:
                fx = getattr(camera, 'focal_x',
                             getattr(camera, 'image_width', 800) / 2)
            fy = getattr(camera, 'Fy', None)
            if fy is None:
                fy = getattr(camera, 'focal_y',
                             getattr(camera, 'image_height', 600) / 2)
            cx = getattr(camera, 'Cx', None)
            if cx is None:
                cx = getattr(camera, 'image_width', 800) / 2
            cy = getattr(camera, 'Cy', None)
            if cy is None:
                cy = getattr(camera, 'image_height', 600) / 2

            img_w = getattr(camera, 'image_width', 800)
            img_h = getattr(camera, 'image_height', 600)

            z = pts_cam[:, 2].clamp(min=0.01)
            px = pts_cam[:, 0] * fx / z + cx
            py = pts_cam[:, 1] * fy / z + cy

            # NDC coordinates [0, 1]
            ndc_x = px / img_w
            ndc_y = py / img_h

            # Soft frustum mask
            margin = self.config.frustum_margin
            inside_x = torch.sigmoid(20.0 * (ndc_x + margin)) * \
                        torch.sigmoid(20.0 * (1.0 + margin - ndc_x))
            inside_y = torch.sigmoid(20.0 * (ndc_y + margin)) * \
                        torch.sigmoid(20.0 * (1.0 + margin - ndc_y))

            visibility = in_front * inside_x * inside_y
            return visibility.reshape(H, W)

        except Exception:
            # Fallback: assume everything visible
            return torch.ones(H, W, device=device)

    def _project_weights_to_spans(
        self,
        weights: torch.Tensor,
        num_spans: int,
        ctrl_dim: int,
        direction: str,
    ) -> torch.Tensor:
        """
        Project 2D control-point weights to 1D span weights via max-pooling
        over the orthogonal direction.
        """
        device = weights.device

        if direction == 'u':
            weights_1d = weights.max(dim=1).values  # [H]
        else:
            weights_1d = weights.max(dim=0).values  # [W]

        n = weights_1d.shape[0]
        if n == 0 or num_spans == 0:
            return torch.ones(max(num_spans, 1), device=device)

        span_weights = torch.zeros(num_spans, device=device)
        ratio = n / max(num_spans, 1)

        for s in range(num_spans):
            start = max(0, int(s * ratio))
            end = min(n, int((s + 1) * ratio) + 1)
            if start < end:
                span_weights[s] = weights_1d[start:end].max()
            else:
                span_weights[s] = weights_1d[min(start, n - 1)]

        return span_weights

    # =========================================================================
    # Density → Fixed-count interval generation
    # =========================================================================

    def _density_to_intervals(
        self,
        density: torch.Tensor,
        knot_boundaries: torch.Tensor,
        target_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert per-span density to exactly `target_count` non-uniform samples.

        This is the KEY function that makes the tessellator compatible with
        the existing pipeline. It always returns exactly `target_count` samples.

        Strategy:
        1. Normalize density to get per-span sample allocation
        2. Allocate integer samples per span (summing to target_count)
        3. Place samples uniformly within each span
        4. Concatenate and sort

        Args:
            density: [num_spans] relative density values
            knot_boundaries: [num_spans+1] unique knot values (span boundaries)
            target_count: EXACT number of samples to produce
            device: torch device

        Returns:
            [target_count] sorted tensor of UV positions in [0, 1]
        """
        num_spans = len(density)

        if num_spans == 0 or target_count <= 0:
            return torch.linspace(0, 1, max(target_count, 2), device=device)

        # Normalize density to allocate samples proportionally
        density_sum = density.sum().clamp(min=1e-8)
        # Continuous allocation
        continuous_alloc = density / density_sum * target_count

        # Clamp per-span allocation
        min_s = self.config.min_samples_per_span
        max_s = self.config.max_samples_per_span
        continuous_alloc = continuous_alloc.clamp(min=min_s, max=max_s)

        # Re-normalize to sum to target_count after clamping
        alloc_sum = continuous_alloc.sum().clamp(min=1e-8)
        continuous_alloc = continuous_alloc / alloc_sum * target_count

        # Integer allocation using largest-remainder method
        floor_alloc = continuous_alloc.floor().long()
        remainders = continuous_alloc - floor_alloc.float()

        current_total = floor_alloc.sum().item()
        deficit = target_count - current_total

        if deficit > 0:
            # Give extra samples to spans with largest remainders
            _, indices = remainders.sort(descending=True)
            n_to_add = min(int(deficit), num_spans)
            floor_alloc[indices[:n_to_add]] += 1
        elif deficit < 0:
            # Remove from spans with smallest remainders (respecting minimum)
            _, indices = remainders.sort()
            remaining = int(-deficit)
            for idx in indices:
                if remaining <= 0:
                    break
                if floor_alloc[idx] > min_s:
                    can_remove = min(floor_alloc[idx].item() - min_s, remaining)
                    floor_alloc[idx] -= can_remove
                    remaining -= can_remove

        # Final adjustment if still not matching (edge case)
        final_total = floor_alloc.sum().item()
        if final_total != target_count:
            # Force-adjust the span with largest allocation
            diff = target_count - final_total
            max_span = floor_alloc.argmax()
            floor_alloc[max_span] += diff

        # Ensure non-negative
        floor_alloc = floor_alloc.clamp(min=0)

        # Generate samples within each span
        intervals = []
        for span_idx in range(num_spans):
            n_samples = floor_alloc[span_idx].item()
            if n_samples <= 0:
                continue

            t_start = knot_boundaries[span_idx].item()
            t_end = knot_boundaries[min(span_idx + 1, len(knot_boundaries) - 1)].item()

            if t_start >= t_end:
                # Degenerate span — place all at midpoint
                intervals.append(
                    torch.full((n_samples,), (t_start + t_end) / 2, device=device)
                )
                continue

            # Uniform within span, offset from boundaries to avoid
            # exact knot values (which can cause numerical issues)
            eps = (t_end - t_start) * 0.01
            span_samples = torch.linspace(
                t_start + eps, t_end - eps, n_samples, device=device
            )
            intervals.append(span_samples)

        if not intervals:
            return torch.linspace(0, 1, target_count, device=device)

        result = torch.cat(intervals)

        # Safety: ensure exactly target_count elements
        if result.shape[0] != target_count:
            # Resample to exact count (preserves distribution shape)
            result = self._resample_to_count(result, target_count, device)

        # Sort and clamp
        result = torch.sort(result)[0].clamp(0.0, 1.0)

        return result

    def _resample_to_count(
        self,
        intervals: torch.Tensor,
        target_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Resample intervals to exactly `target_count` points while
        preserving the non-uniform distribution shape.

        Uses CDF-based interpolation.
        """
        if len(intervals) == target_count:
            return intervals
        if len(intervals) == 0:
            return torch.linspace(0, 1, target_count, device=device)

        intervals_sorted = torch.sort(intervals)[0]

        # Create CDF of current distribution
        current_cdf = torch.linspace(0, 1, len(intervals_sorted), device=device)
        target_cdf = torch.linspace(0, 1, target_count, device=device)

        # Interpolate: find interval values at target CDF positions
        indices = torch.searchsorted(
            current_cdf, target_cdf.clamp(current_cdf[0], current_cdf[-1])
        )
        indices = indices.clamp(1, len(current_cdf) - 1)

        x_lo = current_cdf[indices - 1]
        x_hi = current_cdf[indices]
        y_lo = intervals_sorted[indices - 1]
        y_hi = intervals_sorted[indices]

        t = (target_cdf.clamp(current_cdf[0], current_cdf[-1]) - x_lo) / \
            (x_hi - x_lo + 1e-10)
        t = t.clamp(0, 1)

        resampled = y_lo + t * (y_hi - y_lo)
        return resampled.clamp(0.0, 1.0)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_density_map(
        self,
        surface,
        camera,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagnostic density maps without generating intervals.
        Useful for visualization and debugging.
        """
        device = surface.position.control_features.device
        H, W = surface.state.H, surface.state.W
        ctrl_xyz = surface.position.control_features.detach().view(H, W, 3)

        knots_u = surface.knot_u.forward().detach()
        knots_v = surface.knot_v.forward().detach()
        unique_u = torch.unique_consecutive(knots_u)
        unique_v = torch.unique_consecutive(knots_v)
        num_spans_u = max(len(unique_u) - 1, 1)
        num_spans_v = max(len(unique_v) - 1, 1)

        flatness_u = self._control_polygon_flatness(ctrl_xyz, 'u')
        flatness_v = self._control_polygon_flatness(ctrl_xyz, 'v')
        span_flat_u = self._map_flatness_to_spans(
            flatness_u, num_spans_u, H, 'u', device)
        span_flat_v = self._map_flatness_to_spans(
            flatness_v, num_spans_v, W, 'v', device)

        view_w_u, view_w_v = self._compute_view_weights(
            ctrl_xyz, camera, num_spans_u, num_spans_v, H, W
        )

        return {
            'flatness_u': span_flat_u,
            'flatness_v': span_flat_v,
            'view_weight_u': view_w_u,
            'view_weight_v': view_w_v,
            'combined_u': span_flat_u * view_w_u,
            'combined_v': span_flat_v * view_w_v,
        }


    @torch.no_grad()
    def compute_adaptive_intervals(
            self, surface, camera, Us: int = None, Vs: int = None, mode: str = 'quadtree'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute vd adaptive UV samples.
        - mode='separable': Original 1D (for baseline).
        - mode='hybrid_separable': 2D density → modulated 1D marginals.
        - mode='quadtree': Full 2D irregular points (N,2).
        """
        Us = Us or surface.state.Us
        Vs = Vs or surface.state.Vs
        total_samples = Us * Vs
        device = surface.position.control_features.device
        surface.invalidate_control_features()
        # Compute 2D density (shared for hybrid/quadtree)
        density_2d = self._compute_2d_density(surface, camera)  # (H, W)

        if mode == 'separable':
            # Original: project to 1D
            density_u = density_2d.max(dim=1).values  # (H,)
            density_v = density_2d.max(dim=0).values  # (W,)
            u_int = self._density_to_intervals(density_u, surface.knot_u.unique(), Us, device)
            v_int = self._density_to_intervals(density_v, surface.knot_v.unique(), Vs, device)
            return u_int, v_int

        elif mode == 'x':
            # Modulated marginals from 2D
            density_u = density_2d.sum(dim=1) ** self.config.curvature_exponent  # (H,) integrated over V
            density_v = density_2d.sum(dim=0) ** self.config.curvature_exponent  # (W,)
            density_u = self._smooth_temporal(density_u, '_prev_density_u')
            density_v = self._smooth_temporal(density_v, '_prev_density_v')
            knots_u = torch.unique_consecutive(surface.knot_u.forward().detach())
            knots_v = torch.unique_consecutive(surface.knot_v.forward().detach())
            u_int = self._density_to_intervals(density_u, knots_u, Us, device)
            v_int = self._density_to_intervals(density_v, knots_v, Vs, device)
            return u_int, v_int

        elif mode == 'quadtree':
            # Full 2D: build quadtree, allocate, generate points
            root = self._build_quadtree(density_2d, device)
            leaves, allocations = self._allocate_samples_quadtree(root, total_samples, device)
            uv_points = self._generate_2d_points(leaves, allocations, device)
            uv_points = torch.sort(uv_points[:, 0] + uv_points[:, 1] * 1e6)[0]  # Sort by U then V
            return uv_points  # (total_samples, 2)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_2d_density2(self, surface, camera):
        ctrl_xyz = surface.position.control_features.detach().view(surface.state.H, surface.state.W, 3)

        # 2D flatness (curvature approx)
        du = ctrl_xyz[1:, :] - ctrl_xyz[:-1, :]  # (H-1, W, 3)
        dv = ctrl_xyz[:, 1:] - ctrl_xyz[:, :-1]  # (H, W-1, 3)
        cross = torch.cross(du[:, :-1], dv[:-1, :], dim=-1)  # (H-1, W-1, 3)
        flatness_2d = torch.norm(cross, dim=-1)  # (H-1, W-1)

        # Screen projection (adapt _project_to_ndc or similar; assume helper)
        proj = self._project_to_screen(ctrl_xyz, camera)  # (H, W, 2) screen coords
        dproj_u = proj[1:, :] - proj[:-1, :]  # (H-1, W, 2)
        dproj_v = proj[:, 1:] - proj[:, :-1]  # (H, W-1, 2)
        proj_cross = dproj_u[:, :-1] * dproj_v[:-1, :, 1] - dproj_u[:, :-1, 1] * dproj_v[
                                                                                                             :-1, :,
                                                                                                             0]  # Approx area (det)
        proj_error_2d = torch.abs(proj_cross.squeeze(-1))  # (H-1, W-1)

        # Combine, exponent, backface/frustum (adapt existing)
        density_2d = (flatness_2d * proj_error_2d).clamp(min=self.config.min_visible_density_ratio)
        density_2d = density_2d ** self.config.curvature_exponent

        # Upsample to control res
        density_2d = F.interpolate(density_2d.unsqueeze(0).unsqueeze(0), size=(surface.state.H, surface.state.W),
                                   mode='bilinear').squeeze()

        # Apply backface (normals dot view >0)
        normals = surface.get_normal(camera).view(surface.state.H, surface.state.W, 3)  # Assume available
        view_dirs = (ctrl_xyz - camera.camera_center).normalize()
        backface_mask = (normals * view_dirs).sum(-1) < 0
        density_2d[backface_mask] *= self.config.backface_density_factor

        return density_2d

    # def _compute_2d_density2(self, surface, camera):
    #     ctrl_xyz = surface.position().detach().view(surface.state.H, surface.state.W, 3)
    #
    #     # 2D flatness (curvature approx)
    #     du = ctrl_xyz[1:, :] - ctrl_xyz[:-1, :]  # (H-1, W, 3)
    #     dv = ctrl_xyz[:, 1:] - ctrl_xyz[:, :-1]  # (H, W-1, 3)
    #     cross = torch.cross(du[:, :-1], dv[:-1, :], dim=-1)  # (H-1, W-1, 3)
    #     flatness_2d = torch.norm(cross, dim=-1)  # (H-1, W-1)
    #
    #     # Screen projection
    #     proj = self._project_to_screen(ctrl_xyz, camera)  # (H, W, 2)
    #     dproj_u = proj[1:, :] - proj[:-1, :]  # (H-1, W, 2)
    #     dproj_v = proj[:, 1:] - proj[:, :-1]  # (H, W-1, 2)
    #
    #     # Component-wise 2D det for projected area (fix: no unsqueeze, direct scalars)
    #     ux = dproj_u[:, :-1, 0]  # (H-1, W-1)
    #     uy = dproj_u[:, :-1, 1]  # (H-1, W-1)
    #     vx = dproj_v[:-1, :, 0]  # (H-1, W-1)
    #     vy = dproj_v[:-1, :, 1]  # (H-1, W-1)
    #     proj_cross = ux * vy - uy * vx  # (H-1, W-1), scalar det
    #
    #     proj_error_2d = torch.abs(proj_cross)  # Positive areas
    #
    #     # Combine, exponent, backface/frustum (adapt existing)
    #     density_2d = (flatness_2d * proj_error_2d).clamp(min=self.config.min_visible_density_ratio)
    #     density_2d = density_2d ** self.config.curvature_exponent
    #
    #     # Upsample to control res
    #     density_2d = F.interpolate(density_2d.unsqueeze(0).unsqueeze(0), size=(surface.state.H, surface.state.W),
    #                                mode='bilinear').squeeze()
    #
    #     # Apply backface (normals dot view >0)
    #     normals = surface.get_normal(camera).view_as(ctrl_xyz)  # Assume available
    #
    #     view_dirs = F.normalize(ctrl_xyz - camera.camera_center, dim=-1)
    #     backface_mask = torch.sum(normals * view_dirs, dim=-1) < 0
    #     density_2d[backface_mask] *= self.config.backface_density_factor
    #
    #     return density_2d
    def _compute_2d_density(self, surface, camera):
        ctrl_xyz = surface.position().view(surface.state.Us, surface.state.Vs, 3) #.control_features.detach().view(surface.state.H, surface.state.W, 3)


        # 2D flatness (curvature approx)
        du = ctrl_xyz[1:, :] - ctrl_xyz[:-1, :]  # (H-1, W, 3)
        dv = ctrl_xyz[:, 1:] - ctrl_xyz[:, :-1]  # (H, W-1, 3)
        cross = torch.cross(du[:, :-1], dv[:-1, :], dim=-1)  # (H-1, W-1, 3)
        flatness_2d = torch.norm(cross, dim=-1)  # (H-1, W-1)

        # Screen projection
        proj = self._project_to_screen(ctrl_xyz, camera)  # (H, W, 2)
        dproj_u = proj[1:, :] - proj[:-1, :]  # (H-1, W, 2)
        dproj_v = proj[:, 1:] - proj[:, :-1]  # (H, W-1, 2)

        # Component-wise 2D det for projected area (fix: no unsqueeze, direct scalars)
        ux = dproj_u[:, :-1, 0]  # (H-1, W-1)
        uy = dproj_u[:, :-1, 1]  # (H-1, W-1)
        vx = dproj_v[:-1, :, 0]  # (H-1, W-1)
        vy = dproj_v[:-1, :, 1]  # (H-1, W-1)
        proj_cross = ux * vy - uy * vx  # (H-1, W-1), scalar det

        proj_error_2d = torch.abs(proj_cross)  # Positive areas

        # Combine, exponent, backface/frustum (adapt existing)
        density_2d = (flatness_2d * proj_error_2d).clamp(min=self.config.min_visible_density_ratio)
        density_2d = density_2d ** self.config.curvature_exponent

        # Upsample to control res
        density_2d = F.interpolate(density_2d.unsqueeze(0).unsqueeze(0), size=(surface.state.Us, surface.state.Vs),
                                   mode='bilinear').squeeze()

        # Apply backface (normals dot view >0)
        normals = surface.get_normal(camera).view_as(ctrl_xyz)  # Assume available
        view_dirs = F.normalize(ctrl_xyz - camera.camera_center, dim=-1)
        backface_mask = torch.sum(normals * view_dirs, dim=-1) < 0
        density_2d[backface_mask] *= self.config.backface_density_factor

        return density_2d
    # def _build_quadtree(self, density_2d, device, depth=0, bounds=torch.tensor([0., 1., 0., 1.], device='cuda')):
    #     quad = Quad(bounds=bounds)
    #     quad_idx_u = torch.linspace(bounds[0], bounds[1], density_2d.shape[0], device=device).long().clamp(0,
    #                                                                                                        density_2d.shape[
    #                                                                                                            0] - 1)
    #     quad_idx_v = torch.linspace(bounds[2], bounds[3], density_2d.shape[1], device=device).long().clamp(0,
    #                                                                                                        density_2d.shape[
    #                                                                                                            1] - 1)
    #     quad_density = density_2d[quad_idx_u[1:-1], :][:, quad_idx_v[1:-1]].max()  # Max error in quad

    def _build_quadtree(self, density_2d, device, depth=0, bounds=torch.tensor([0., 1., 0., 1.], device='cuda')):
        quad = Quad(bounds=bounds)
        H, W = density_2d.shape
        start_u = int(bounds[0] * (H - 1))
        end_u = int(bounds[1] * (H - 1)) + 1
        start_v = int(bounds[2] * (W - 1))
        end_v = int(bounds[3] * (W - 1)) + 1
        sub_density = density_2d[start_u:end_u, start_v:end_v]
        quad_density = sub_density.max() if sub_density.numel() > 0 else 0.0

        if quad_density > self.config.flatness_threshold and depth < self.config.quadtree_max_depth and (
                bounds[1] - bounds[0]) > self.config.quadtree_min_size:
            u_mid, v_mid = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2
            quad.children = [
                self._build_quadtree(density_2d, device, depth + 1,
                                     torch.tensor([bounds[0], u_mid, bounds[2], v_mid], device=device)),
                self._build_quadtree(density_2d, device, depth + 1,
                                     torch.tensor([u_mid, bounds[1], bounds[2], v_mid], device=device)),
                self._build_quadtree(density_2d, device, depth + 1,
                                     torch.tensor([bounds[0], u_mid, v_mid, bounds[3]], device = device)),
                self._build_quadtree(density_2d, device, depth + 1,
                                     torch.tensor([u_mid, bounds[1], v_mid, bounds[3]], device = device))
            ]
            quad.density = quad_density * (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])  # Area-weighted
            return quad
    def _allocate_samples_quadtree(self, root, total_samples, device):
        leaves = []

        def collect(quad):
            if quad.children:
                for c in quad.children: collect(c)
            else:
                leaves.append(quad)

        collect(root)

        densities = torch.tensor([q.density for q in leaves], device=device)
        areas = torch.tensor([(q.bounds[1] - q.bounds[0]) * (q.bounds[3] - q.bounds[2]) for q in leaves],
                             device=device)
        weighted = densities * areas
        norm = weighted / weighted.sum()
        allocs = (norm * total_samples).floor().long()

        # Largest-remainder for exact
        remain = total_samples - allocs.sum().item()
        if remain > 0:
            _, idx = torch.sort(norm - allocs.float(), descending=True)
            allocs[idx[:remain]] += 1
        return leaves, allocs.tolist()

    def _generate_2d_points(self, leaves, allocations, device):
        from torch.quasirandom import SobolEngine  # Low-discrepancy
        points = []
        for leaf, alloc in zip(leaves, allocations):
            if alloc == 0: continue
            sobol = SobolEngine(dimension=2, scramble=True)
            offsets = sobol.draw(alloc).to(device)  # (alloc, 2) [0,1]
            scales = leaf.bounds[1] - leaf.bounds[0], leaf.bounds[3] - leaf.bounds[2]
            mins = leaf.bounds[0], leaf.bounds[2]
            quad_points = torch.tensor([mins[0], mins[1]], device=device) + offsets * torch.tensor(
                [scales[0], scales[1]], device=device)
            points.append(quad_points)
        return torch.cat(points, dim=0).clamp(0, 1)

    def _smooth_temporal(self, density, attr):
        prev = getattr(self, attr)
        if prev is not None and prev.shape == density.shape:
            density = (1 - self.config.temporal_alpha) * density + self.config.temporal_alpha * prev
        setattr(self, attr, density)
        return density

    # Existing _density_to_intervals, _project_to_screen (assume implemented or from original)
    # ... (include original methods like _control_polygon_flatness if needed for completeness)

    def _project_to_screen(self, ctrl_xyz, camera):
        # Placeholder: Project to NDC/screen (adapt from original view_w computation)
        homo = torch.cat([ctrl_xyz, torch.ones_like(ctrl_xyz[..., :1])], -1) @ camera.full_proj_transform.T
        ndc = homo[..., :3] / homo[..., 3:4].clamp(min=1e-6)
        screen = (ndc[..., :2] + 1) * 0.5 * torch.tensor([camera.image_width, camera.image_height],
                                                         device=ctrl_xyz.device)
        return screen  # (H, W, 2)

