"""
View-Dependent Adaptive Tessellation (Chhugani et al., 2001)

Faithful implementation of:
  "View-dependent Adaptive Tessellation of Spline Surfaces"
  https://www.cs.jhu.edu/graphics/papers/Chhugani01.pdf

ADAPTATION NOTES:
  The original algorithm is a recursive subdivision scheme that splits
  patches until a screen-space flatness criterion is met. Our architecture
  uses separable 1D UV intervals feeding into pre-computed B-spline basis
  matrices, so recursive 2D subdivision is not directly applicable.

  We faithfully implement the paper's THREE key components and convert
  their output into our 1D interval format:

  1. FLATNESS TEST (Section 3.1):
     Uses second differences of control points projected to screen space.
     This gives a guaranteed upper bound on the screen-space deviation.

  2. FACE CLASSIFICATION (Section 3.2):
     Classifies each patch as front-facing, back-facing, or silhouette
     using consistent normal orientation.
     - Back-facing: minimal tessellation
     - Silhouette: threshold reduced by 1/cos(θ) factor
     - Front-facing: standard threshold

  3. ADAPTIVE DENSITY (Section 3.3):
     Subdivision depth: n = ceil(log2(deviation / threshold))
     Converted to importance weights for inverse-CDF sampling.

  BATCHED AGGREGATION (Extension for training):
     Pre-computes importance maps for multiple training views and
     aggregates them into a global importance profile. Per-view intervals
     are then derived by blending the global profile with view-specific
     adjustments. This amortizes the expensive flatness test across
     the training loop.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

from modules.spline_utils import BasisFuncs

if TYPE_CHECKING:
    from modules.KnotSurface import SplineModel



# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class ForwardContext:
    """Per-forward-pass context storing view-dependent UV intervals."""
    warped_us: Dict[int, torch.Tensor] = field(default_factory=dict)
    warped_vs: Dict[int, torch.Tensor] = field(default_factory=dict)
    warped_uvs: Dict[int, torch.Tensor] = field(default_factory=dict)
    uv_basis: Dict[int, BasisFuncs] = field(default_factory=dict)
    camera_uid: Optional[int] = None


@dataclass
class ChhuganiParams:
    """
    Parameters mapping directly to Chhugani's algorithm.

    - pixel_threshold: ε in the flatness test (pixels)
    - silhouette_threshold: cos(θ) below which a front-face is near-silhouette
    - backface_factor: density multiplier for back-facing patches [0, 1]
    - temporal_alpha: EMA blending with previous frame [0, 1]
    - min_density: floor on relative density to prevent zero-sample regions
    - coarse_u/v: resolution of the evaluation grid for face classification
    """
    pixel_threshold: float = 2.0
    silhouette_threshold: float = 0.1
    backface_factor: float = 0.0
    temporal_alpha: float = 0.0
    min_density: float = 0.0
    coarse_u: int = 32
    coarse_v: int = 32
    global_weight: float = 0.1


@dataclass
class AggregationState:
    """
    Stores aggregated importance information across multiple training views.

    This is the key data structure for batched training. Instead of computing
    the full Chhugani pipeline per view per iteration, we pre-compute and
    cache aggregated statistics that can be efficiently queried.
    """
    # Global importance profiles (aggregated across views)
    global_importance_u: Optional[torch.Tensor] = None  # [coarse_u]
    global_importance_v: Optional[torch.Tensor] = None  # [coarse_v]

    # Global intervals derived from aggregated importance
    global_intervals_u: Optional[torch.Tensor] = None  # [Us]
    global_intervals_v: Optional[torch.Tensor] = None  # [Vs]

    # Per-view importance deltas (deviation from global)
    # Maps camera_uid -> (delta_u, delta_v) where delta is the
    # difference between view-specific and global importance
    view_deltas_u: Dict[int, torch.Tensor] = field(default_factory=dict)
    view_deltas_v: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-view flatness values (cached for incremental updates)
    view_flatness_u: Dict[int, torch.Tensor] = field(default_factory=dict)
    view_flatness_v: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-view face classification (cached)
    view_cos_u: Dict[int, torch.Tensor] = field(default_factory=dict)
    view_cos_v: Dict[int, torch.Tensor] = field(default_factory=dict)
    view_backface_u: Dict[int, torch.Tensor] = field(default_factory=dict)
    view_backface_v: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Tracking
    num_views_aggregated: int = 0
    last_aggregation_iteration: int = -1
    stale: bool = True  # True if control points changed since last aggregation

    def clear(self):
        """Reset all cached state."""
        self.global_importance_u = None
        self.global_importance_v = None
        self.global_intervals_u = None
        self.global_intervals_v = None
        self.view_deltas_u.clear()
        self.view_deltas_v.clear()
        self.view_flatness_u.clear()
        self.view_flatness_v.clear()
        self.view_cos_u.clear()
        self.view_cos_v.clear()
        self.view_backface_u.clear()
        self.view_backface_v.clear()
        self.num_views_aggregated = 0
        self.stale = True


# =========================================================================
# Step 1: Screen-Space Flatness Test (Chhugani Section 3.1)
# =========================================================================

def _build_mvp(camera) -> torch.Tensor:
    """
    Build column-major MVP matrix: clip = MVP @ [x,y,z,1]^T.

    Handles the 3DGS convention where full_proj_transform is row-major
    (point_row @ M), so we transpose.
    """
    device = camera.camera_center.device

    if hasattr(camera, 'full_proj_transform'):
        mvp = camera.full_proj_transform.to(device)
        if mvp.shape == (4, 4):
            return mvp.T  # row-major -> column-major
        return mvp

    # Fallback: compose view and projection
    if hasattr(camera, 'world_view_transform'):
        view = camera.world_view_transform.to(device)
    else:
        R = torch.tensor(camera.R, device=device, dtype=torch.float32)
        T = torch.tensor(camera.T, device=device, dtype=torch.float32)
        view = torch.eye(4, device=device)
        view[:3, :3] = R.T
        view[:3, 3] = T

    if hasattr(camera, 'projection_matrix'):
        proj = camera.projection_matrix.to(device)
    else:
        fx = getattr(camera, 'Fx', getattr(camera, 'focal_x', None))
        fy = getattr(camera, 'Fy', getattr(camera, 'focal_y', None))
        W = int(camera.image_width)
        H = int(camera.image_height)
        near, far = 0.01, 100.0
        proj = torch.zeros(4, 4, device=device)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1.0

    return proj @ view


def _project_to_screen(
    points_3d: torch.Tensor,  # [N, 3]
    mvp: torch.Tensor,        # [4, 4]
    img_w: int,
    img_h: int,
) -> torch.Tensor:
    """
    Project 3D points to 2D screen coordinates (pixels).
    Returns: [N, 2] pixel coordinates.
    """
    device = points_3d.device
    N = points_3d.shape[0]
    ones = torch.ones(N, 1, device=device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_3d, ones], dim=-1)  # [N, 4]
    clip = (mvp @ pts_h.T).T  # [N, 4]

    w = clip[:, 3:4].abs().clamp(min=1e-6)
    ndc = clip[:, :2] / w  # [-1, 1]

    screen = torch.stack([
        (ndc[:, 0] + 1.0) * 0.5 * img_w,
        (ndc[:, 1] + 1.0) * 0.5 * img_h,
    ], dim=-1)  # [N, 2]

    return screen


def _compute_flatness_map(
    surface: 'SplineModel',
    camera,
    params: ChhuganiParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chhugani's flatness test (Section 3.1).

    For each knot span, compute the maximum screen-space deviation of
    the surface from its linear approximation. This is bounded by the
    screen-space length of the control point second differences.

    The bound for a cubic B-spline patch in span [i, i+1] is:
      δ_screen ≤ (Δu² / 8) * max_j(||Δ²P_screen_j||)

    where Δ²P_j are the second differences of the influencing control
    points, and 1/8 comes from the maximum of the cubic B-spline
    second derivative basis function.

    Returns:
        flatness_u: [num_spans_u] max screen-space deviation per U-span
        flatness_v: [num_spans_v] max screen-space deviation per V-span
    """
    device = camera.camera_center.device
    H, W = surface.state.H, surface.state.W
    degree = surface.state.degree

    # Get control points [H, W, 3]
    ctrl_xyz = surface.position.control_features.detach().view(H, W, -1)[..., :3]

    # Handle rational (NURBS) case: work in projective space
    has_weights = (
        surface.position.weights is not None
        and surface.position.weights.control_features is not None
    )
    if has_weights:
        w_ctrl = surface.position.weights.activation(
            surface.position.weights.control_features.detach()
        ).view(H, W, 1)
        ctrl_xyz = ctrl_xyz * w_ctrl

    # Project ALL control points to screen space
    mvp = _build_mvp(camera)
    img_w = int(camera.image_width)
    img_h = int(camera.image_height)

    ctrl_screen = _project_to_screen(
        ctrl_xyz.reshape(-1, 3), mvp, img_w, img_h
    ).reshape(H, W, 2)

    # Compute screen-space second differences
    # These directly bound the screen-space deviation (Chhugani Eq. 3)
    # d2P_u[i,j] = ctrl_screen[i-1,j] - 2*ctrl_screen[i,j] + ctrl_screen[i+1,j]
    if H >= 3:
        d2_screen_u = (
            ctrl_screen[:-2, :, :] - 2.0 * ctrl_screen[1:-1, :, :] + ctrl_screen[2:, :, :]
        )  # [H-2, W, 2]
        d2_mag_u = d2_screen_u.norm(dim=-1)  # [H-2, W]
    else:
        d2_mag_u = torch.zeros(0, W, device=device)

    if W >= 3:
        d2_screen_v = (
            ctrl_screen[:, :-2, :] - 2.0 * ctrl_screen[:, 1:-1, :] + ctrl_screen[:, 2:, :]
        )  # [H, W-2, 2]
        d2_mag_v = d2_screen_v.norm(dim=-1)  # [H, W-2]
    else:
        d2_mag_v = torch.zeros(H, 0, device=device)

    # Get knot vectors for span width computation
    knots_u = surface.knot_u().detach()
    knots_v = surface.knot_v().detach()

    # Number of interior spans
    num_spans_u = H - degree
    num_spans_v = W - degree

    # Vectorized flatness computation for U spans
    flatness_u = torch.zeros(num_spans_u, device=device)
    if num_spans_u > 0 and d2_mag_u.shape[0] > 0:
        for span_i in range(num_spans_u):
            span_width = (knots_u[span_i + degree + 1] - knots_u[span_i + degree]).abs()
            if span_width < 1e-10:
                continue

            # Second differences influencing this span:
            # For cubic (degree=3), control points i..i+3, second diffs at i and i+1
            sd_start = max(0, span_i)
            sd_end = min(span_i + degree - 1, d2_mag_u.shape[0])
            if sd_start >= sd_end:
                continue

            # Max across all V columns (worst case for this U-span)
            max_d2 = d2_mag_u[sd_start:sd_end, :].max()

            # Chhugani bound: deviation ≤ (Δu² / 8) * max(||Δ²P_screen||)
            flatness_u[span_i] = (span_width ** 2 / 8.0) * max_d2

    # Vectorized flatness computation for V spans
    flatness_v = torch.zeros(num_spans_v, device=device)
    if num_spans_v > 0 and d2_mag_v.shape[1] > 0:
        for span_j in range(num_spans_v):
            span_width = (knots_v[span_j + degree + 1] - knots_v[span_j + degree]).abs()
            if span_width < 1e-10:
                continue

            sd_start = max(0, span_j)
            sd_end = min(span_j + degree - 1, d2_mag_v.shape[1])
            if sd_start >= sd_end:
                continue

            max_d2 = d2_mag_v[:, sd_start:sd_end].max()
            flatness_v[span_j] = (span_width ** 2 / 8.0) * max_d2

    return flatness_u, flatness_v


# =========================================================================
# Step 2: Face Classification (Chhugani Section 3.2)
# =========================================================================

def _classify_patches(
    surface: 'SplineModel',
    camera,
    params: ChhuganiParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classify patches as front-facing, back-facing, or silhouette.

    Uses a coarse evaluation grid. Normal orientation is made CONSISTENT
    by checking if the majority of normals point toward the camera.

    Returns:
        cos_angles_u: [coarse_u] per-U-row min |cos(angle)|
        cos_angles_v: [coarse_v] per-V-col min |cos(angle)|
        backface_u: [coarse_u] fraction of backfacing samples per U-row
        backface_v: [coarse_v] fraction of backfacing samples per V-col
    """
    from modules.basis import generate_bspline_basis_matrices
    import opt_einsum as oe

    device = camera.camera_center.device
    Nu, Nv = params.coarse_u, params.coarse_v
    H, W = surface.state.H, surface.state.W

    u_coarse = torch.linspace(0.0, 1.0, Nu, device=device)
    v_coarse = torch.linspace(0.0, 1.0, Nv, device=device)

    knot_u = surface.knot_u().detach()
    knot_v = surface.knot_v().detach()

    bu, dbu, _ = generate_bspline_basis_matrices(
        u_coarse, knots=knot_u, num_control_points=H, degree=3, device=device
    )
    bv, dbv, _ = generate_bspline_basis_matrices(
        v_coarse, knots=knot_v, num_control_points=W, degree=3, device=device
    )

    ctrl_xyz = surface.position.control_features.detach().view(H, W, -1)[..., :3]
    path = 'uh,hwc,vw->uvc'

    S = oe.contract(path, bu, ctrl_xyz, bv)     # [Nu, Nv, 3]
    dSu = oe.contract(path, dbu, ctrl_xyz, bv)  # [Nu, Nv, 3]
    dSv = oe.contract(path, bu, ctrl_xyz, dbv)  # [Nu, Nv, 3]

    # Compute normals via cross product
    normals = torch.cross(dSu, dSv, dim=-1)  # [Nu, Nv, 3]
    normals = F.normalize(normals, dim=-1, eps=1e-8)

    # View direction: surface -> camera
    cam_center = camera.camera_center.to(device)
    view_dirs = F.normalize(
        cam_center.unsqueeze(0).unsqueeze(0) - S, dim=-1, eps=1e-8
    )

    # Signed cos(angle): positive = front-facing
    cos_raw = (normals * view_dirs).sum(dim=-1)  # [Nu, Nv]

    # CONSISTENT ORIENTATION (Chhugani Section 3.2):
    # B-spline normals from cross(dSu, dSv) may have arbitrary orientation
    # depending on parameterization direction. If majority point away from
    # camera, the parameterization is flipped — negate all.
    backface_ratio = (cos_raw < 0).float().mean()
    if backface_ratio > 0.6:
        cos_raw = -cos_raw

    # Classification
    is_backface = cos_raw < 0  # [Nu, Nv]
    abs_cos = cos_raw.abs()

    # Per U-row: minimum |cos| across V (if ANY v is near-silhouette, flag row)
    cos_angles_u = abs_cos.min(dim=1).values  # [Nu]
    cos_angles_v = abs_cos.min(dim=0).values  # [Nv]

    # Backface fraction per row/col
    backface_u = is_backface.float().mean(dim=1)  # [Nu]
    backface_v = is_backface.float().mean(dim=0)  # [Nv]

    return cos_angles_u, cos_angles_v, backface_u, backface_v


# =========================================================================
# Step 3: Adaptive Density (Chhugani Section 3.3)
# =========================================================================

def _compute_subdivision_depth(
    flatness: torch.Tensor,  # [num_spans]
    threshold: float,
) -> torch.Tensor:
    """
    Number of subdivisions needed per span.

    From Chhugani: if deviation δ > threshold ε, the span needs
    ceil(log2(δ/ε)) subdivisions. Each subdivision halves the
    deviation (quadratic convergence for cubic splines).

    Returns: [num_spans] float subdivision depths.
    """
    ratio = flatness / max(threshold, 1e-6)
    depth = torch.log2(ratio.clamp(min=1.0))  # 0 if already flat enough
    return depth


def _spans_to_importance(
    subdiv_depth: torch.Tensor,  # [num_spans]
    knots: torch.Tensor,
    degree: int,
    num_coarse: int,
    cos_angles: torch.Tensor,    # [num_coarse]
    backface_frac: torch.Tensor,  # [num_coarse]
    params: ChhuganiParams,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert per-span subdivision depths into a per-sample importance
    profile, incorporating face classification.

    1. Map span depths to fine 1D importance profile
    2. Silhouette: reduce threshold by 1/cos(θ) → multiply importance
    3. Backface: reduce importance
    4. Floor at min_density

    Returns: [num_coarse] importance profile.
    """
    num_spans = subdiv_depth.shape[0]
    if num_spans == 0:
        return torch.ones(num_coarse, device=device) * params.min_density

    # Map coarse sample positions to spans
    unique_knots = torch.unique(knots)
    span_boundaries = unique_knots[degree:-degree] if len(unique_knots) > 2 * degree else unique_knots

    if len(span_boundaries) < 2:
        return torch.ones(num_coarse, device=device) * params.min_density

    coarse_positions = torch.linspace(0.0, 1.0, num_coarse, device=device)

    # Find which span each coarse sample falls into
    span_idx = torch.searchsorted(span_boundaries, coarse_positions, right=True) - 1
    span_idx = span_idx.clamp(0, num_spans - 1)

    # Base importance = 2^depth (relative density needed)
    base_importance = (2.0 ** subdiv_depth).clamp(min=1.0)

    # Map to coarse grid
    importance = base_importance[span_idx]  # [num_coarse]

    # --- Silhouette boost (Chhugani Section 3.2) ---
    # Near-silhouette patches need MORE tessellation because the outline
    # accuracy is critical. The paper reduces threshold by 1/cos(θ),
    # equivalent to multiplying importance by 1/cos(θ).
    near_silhouette = cos_angles < params.silhouette_threshold
    if near_silhouette.any():
        boost = 1.0 / cos_angles[near_silhouette].clamp(min=0.01)
        silhouette_factor = torch.ones(num_coarse, device=device)
        silhouette_factor[near_silhouette] = boost.clamp(max=20.0)
        importance = importance * silhouette_factor

    # --- Backface reduction (Chhugani Section 3.2) ---
    bf_reduction = 1.0 - backface_frac * (1.0 - params.backface_factor)
    importance = importance * bf_reduction

    # --- Floor ---
    importance = importance.clamp(min=params.min_density)

    return importance


def _importance_to_intervals(
    importance: torch.Tensor,  # [N] importance profile
    target_count: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert importance profile to non-uniform sample positions via
    inverse-CDF sampling (Chhugani Section 4 adaptation).

    Returns: [target_count] sorted sample positions in [0, 1].
    """
    N = importance.shape[0]

    total = importance.sum()
    if total < 1e-10:
        return torch.linspace(0.0, 1.0, target_count, device=device)

    pdf = importance / total
    cdf = torch.cumsum(pdf, dim=0)

    # Build CDF with proper left boundary
    cdf_full = torch.cat([torch.zeros(1, device=device), cdf])
    cdf_full[-1] = 1.0

    coarse_positions = torch.linspace(0.0, 1.0, N, device=device)
    half_step = 0.5 / max(N - 1, 1) if N > 1 else 0.0
    pos_full = torch.cat([
        torch.tensor([max(0.0, coarse_positions[0].item() - half_step)], device=device),
        coarse_positions,
    ])

    # Uniform quantiles at target resolution
    quantiles = torch.linspace(0.0, 1.0, target_count, device=device)

    # Inverse CDF via searchsorted + linear interpolation
    indices = torch.searchsorted(cdf_full, quantiles.clamp(0.0, 1.0))
    indices = indices.clamp(1, len(cdf_full) - 1)

    cdf_lo = cdf_full[indices - 1]
    cdf_hi = cdf_full[indices]
    pos_lo = pos_full[indices - 1]
    pos_hi = pos_full[indices]

    denom = (cdf_hi - cdf_lo).clamp(min=1e-10)
    t = ((quantiles - cdf_lo) / denom).clamp(0.0, 1.0)

    intervals = pos_lo + t * (pos_hi - pos_lo)
    intervals = intervals.clamp(0.0, 1.0)
    intervals = torch.sort(intervals)[0]

    return intervals


# =========================================================================
# Batched Aggregation (Extension for training efficiency)
# =========================================================================

@torch.no_grad()
def aggregate_views(
    surface: 'SplineModel',
    cameras: List,
    params: Optional[ChhuganiParams] = None,
    max_views: int = 0,
    aggregation_mode: str = 'max',
) -> AggregationState:
    """
    Pre-compute and aggregate importance maps across multiple training views.

    This is the key function for batched training efficiency. Instead of
    running the full Chhugani pipeline per view per iteration, we:

    1. Compute flatness test ONCE (geometry-dependent, view-independent
       for the control point second differences part).
    2. Compute face classification per view (view-dependent).
    3. Aggregate importance profiles across views.
    4. Store per-view deltas for efficient per-view interval recovery.

    Args:
        surface: SplineModel to analyze
        cameras: List of training cameras
        params: Tessellation parameters
        max_views: Maximum views to process (0 = all)
        aggregation_mode: How to combine per-view importance:
            'max': Take element-wise maximum (conservative — ensures
                   all views get adequate tessellation)
            'mean': Average importance (balanced)
            'percentile_90': 90th percentile (robust to outliers)

    Returns:
        AggregationState with global and per-view importance data
    """
    if params is None:
        params = _get_or_create_params(surface)

    device = surface.position.control_features.device
    Us, Vs = surface.state.Us, surface.state.Vs

    agg = AggregationState()

    # --- Step 1: Flatness test (view-independent geometry part) ---
    # The control point second differences don't change between views.
    # Only the screen-space projection changes. But for aggregation,
    # we compute per-view flatness and aggregate.

    # Limit number of views if requested
    view_list = cameras
    if max_views > 0 and len(cameras) > max_views:
        import random
        view_list = random.sample(cameras, max_views)

    all_importance_u = []
    all_importance_v = []

    for cam in view_list:
        uid = cam.uid

        # Flatness test (Section 3.1) - view-dependent via screen projection
        flatness_u, flatness_v = _compute_flatness_map(surface, cam, params)

        # Face classification (Section 3.2) - view-dependent
        cos_u, cos_v, bf_u, bf_v = _classify_patches(surface, cam, params)

        # Cache per-view data
        agg.view_flatness_u[uid] = flatness_u
        agg.view_flatness_v[uid] = flatness_v
        agg.view_cos_u[uid] = cos_u
        agg.view_cos_v[uid] = cos_v
        agg.view_backface_u[uid] = bf_u
        agg.view_backface_v[uid] = bf_v

        # Compute per-view importance
        subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
        subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

        imp_u = _spans_to_importance(
            subdiv_u, surface.knot_u().detach(), surface.state.degree,
            params.coarse_u, cos_u, bf_u, params, device
        )
        imp_v = _spans_to_importance(
            subdiv_v, surface.knot_v().detach(), surface.state.degree,
            params.coarse_v, cos_v, bf_v, params, device
        )

        all_importance_u.append(imp_u)
        all_importance_v.append(imp_v)

    if not all_importance_u:
        agg.stale = True
        return agg

    # --- Step 2: Aggregate importance across views ---
    stacked_u = torch.stack(all_importance_u, dim=0)  # [num_views, coarse_u]
    stacked_v = torch.stack(all_importance_v, dim=0)  # [num_views, coarse_v]

    if aggregation_mode == 'max':
        agg.global_importance_u = stacked_u.max(dim=0).values
        agg.global_importance_v = stacked_v.max(dim=0).values
    elif aggregation_mode == 'mean':
        agg.global_importance_u = stacked_u.mean(dim=0)
        agg.global_importance_v = stacked_v.mean(dim=0)
    elif aggregation_mode == 'percentile_90':
        k = max(1, int(0.9 * stacked_u.shape[0]))
        agg.global_importance_u = stacked_u.kthvalue(k, dim=0).values
        agg.global_importance_v = stacked_v.kthvalue(k, dim=0).values
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

    # --- Step 3: Compute global intervals ---
    agg.global_intervals_u = _importance_to_intervals(
        agg.global_importance_u, Us, device
    )
    agg.global_intervals_v = _importance_to_intervals(
        agg.global_importance_v, Vs, device
    )

    # --- Step 4: Store per-view deltas ---
    for i, cam in enumerate(view_list):
        uid = cam.uid
        agg.view_deltas_u[uid] = all_importance_u[i] - agg.global_importance_u
        agg.view_deltas_v[uid] = all_importance_v[i] - agg.global_importance_v

    agg.num_views_aggregated = len(view_list)
    agg.stale = False

    return agg


@torch.no_grad()
def get_view_intervals_from_aggregation(
    surface: 'SplineModel',
    camera,
    agg: AggregationState,
    params: Optional[ChhuganiParams] = None,
    view_blend_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficiently derive per-view intervals from pre-computed aggregation.

    This is the FAST PATH used during training. Instead of running the
    full Chhugani pipeline, we:

    1. Start from the global intervals (pre-computed).
    2. If we have a cached delta for this view, blend it in.
    3. If this is an unseen view, compute its delta on-the-fly
       (still faster because we skip the aggregation step).

    Args:
        surface: SplineModel
        camera: Current viewpoint
        agg: Pre-computed aggregation state
        params: Tessellation parameters
        view_blend_weight: How much to weight view-specific adjustments
            vs. the global intervals. 0.0 = pure global, 1.0 = pure per-view.

    Returns:
        (intervals_u, intervals_v): [Us], [Vs] sorted sample positions
    """
    if params is None:
        params = _get_or_create_params(surface)

    device = surface.position.control_features.device
    Us, Vs = surface.state.Us, surface.state.Vs
    uid = camera.uid

    # If aggregation is stale or empty, fall back to full per-view computation
    if agg.stale or agg.global_importance_u is None:
        return _compute_view_intervals_full(surface, camera, params)

    # Fast path: blend global + view-specific delta
    if uid in agg.view_deltas_u:
        # Known view: use cached delta
        view_importance_u = agg.global_importance_u + view_blend_weight * agg.view_deltas_u[uid]
        view_importance_v = agg.global_importance_v + view_blend_weight * agg.view_deltas_v[uid]
    else:
        # Unknown view: compute delta on-the-fly
        flatness_u, flatness_v = _compute_flatness_map(surface, camera, params)
        cos_u, cos_v, bf_u, bf_v = _classify_patches(surface, camera, params)

        subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
        subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

        view_imp_u = _spans_to_importance(
            subdiv_u, surface.knot_u().detach(), surface.state.degree,
            params.coarse_u, cos_u, bf_u, params, device
        )
        view_imp_v = _spans_to_importance(
            subdiv_v, surface.knot_v().detach(), surface.state.degree,
            params.coarse_v, cos_v, bf_v, params, device
        )

        view_importance_u = (1 - view_blend_weight) * agg.global_importance_u + view_blend_weight * view_imp_u
        view_importance_v = (1 - view_blend_weight) * agg.global_importance_v + view_blend_weight * view_imp_v

        # Cache the delta for future use
        agg.view_deltas_u[uid] = view_imp_u - agg.global_importance_u
        agg.view_deltas_v[uid] = view_imp_v - agg.global_importance_v

    # Clamp importance (deltas can make it negative)
    view_importance_u = view_importance_u.clamp(min=params.min_density)
    view_importance_v = view_importance_v.clamp(min=params.min_density)

    # Convert to intervals
    new_u = _importance_to_intervals(view_importance_u, Us, device)
    new_v = _importance_to_intervals(view_importance_v, Vs, device)

    return new_u, new_v


def _compute_view_intervals_full(
    surface: 'SplineModel',
    camera,
    params: ChhuganiParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full per-view interval computation (fallback when no aggregation available).
    """
    device = camera.camera_center.device
    Us, Vs = surface.state.Us, surface.state.Vs

    flatness_u, flatness_v = _compute_flatness_map(surface, camera, params)
    cos_u, cos_v, bf_u, bf_v = _classify_patches(surface, camera, params)

    subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
    subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

    imp_u = _spans_to_importance(
        subdiv_u, surface.knot_u().detach(), surface.state.degree,
        params.coarse_u, cos_u, bf_u, params, device
    )
    imp_v = _spans_to_importance(
        subdiv_v, surface.knot_v().detach(), surface.state.degree,
        params.coarse_v, cos_v, bf_v, params, device
    )

    new_u = _importance_to_intervals(imp_u, Us, device)
    new_v = _importance_to_intervals(imp_v, Vs, device)

    return new_u, new_v


# =========================================================================
# Main Entry Point (single-view, backward compatible)
# =========================================================================

# @torch.no_grad()
def update_uv_distribution_chhugani(
    surface: 'SplineModel',
    camera,
    neighbor_cameras: Optional[List] = None,
    params: Optional[ChhuganiParams] = None,
    blend_neighbors: bool = False,
    neighbor_weight: float = 0.,
):
    """
    Update UV sampling intervals using Chhugani's view-dependent
    adaptive tessellation algorithm.

    Faithful to the paper's three-step process:
      1. Flatness test via control point second differences (Section 3.1)
      2. Face classification: front/back/silhouette (Section 3.2)
      3. Density computation via inverse-CDF (Section 3.3)

    BATCHED MODE: If surface._chhugani_aggregation exists and is fresh,
    uses the fast path via get_view_intervals_from_aggregation().
    Otherwise, falls back to full per-view computation.
    """
    params = _get_or_create_params(surface, params)
    global_weight = params.global_weight
    ctx = _ensure_forward_context(surface)
    uid = camera.uid
    device = camera.camera_center.device
    Us, Vs = surface.state.Us, surface.state.Vs

    # =====================================================================
    # Fast path: use pre-computed aggregation if available
    # =====================================================================
    agg = getattr(surface, '_chhugani_aggregation', None)
    if agg is not None and not agg.stale:
        new_u, new_v = get_view_intervals_from_aggregation(
            surface, camera, agg, params,
            view_blend_weight=0.3,
        )
    else:
        # =====================================================================
        # Full path: compute from scratch
        # =====================================================================

        # Step 1: Flatness test (Section 3.1)
        flatness_u, flatness_v = _compute_flatness_map(surface, camera, params)

        # Step 2: Face classification (Section 3.2)
        cos_u, cos_v, bf_u, bf_v = _classify_patches(surface, camera, params)

        # Step 3: Compute adaptive density (Section 3.3)
        subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
        subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

        imp_u = _spans_to_importance(
            subdiv_u, surface.knot_u().detach(), surface.state.degree,
            params.coarse_u, cos_u, bf_u, params, device
        )
        imp_v = _spans_to_importance(
            subdiv_v, surface.knot_v().detach(), surface.state.degree,
            params.coarse_v, cos_v, bf_v, params, device
        )

        new_u = _importance_to_intervals(imp_u, Us, device)
        new_v = _importance_to_intervals(imp_v, Vs, device)

    # =====================================================================
    # Step 4: Temporal blending
    # =====================================================================
    if params.temporal_alpha > 0 and uid in ctx.warped_us:
        alpha = params.temporal_alpha
        new_u = (1 - alpha) * new_u + alpha * ctx.warped_us[uid]
        new_v = (1 - alpha) * new_v + alpha * ctx.warped_vs[uid]

    # =================================================================
    # Step 4: Blend with global intervals (if available)
    # =================================================================
    if hasattr(surface.uv_sampler, '_global_intervals') and surface._global_intervals is not None:
        global_u, global_v = surface.uv_sampler._interval_u_global, surface.uv_sampler._interval_v_global
        # Chhugani intervals refine ON TOP of global intervals
        # Strategy: weighted blend (global provides the base, per-view refines)
        new_u = (1 - global_weight) * new_u + global_weight * global_u.to(device)
        new_v = (1 - global_weight) * new_v + global_weight * global_v.to(device)

    # Step 5: Neighbor blending (not in original paper)
    if blend_neighbors and neighbor_cameras:
        for ncam in neighbor_cameras:
            if agg is not None and not agg.stale:
                n_u, n_v = get_view_intervals_from_aggregation(
                    surface, ncam, agg, params
                )
            else:
                n_u, n_v = _compute_view_intervals_full(surface, ncam, params)
            new_u = (1 - neighbor_weight) * new_u + neighbor_weight * n_u
            new_v = (1 - neighbor_weight) * new_v + neighbor_weight * n_v

    # Ensure sorted
    new_u = torch.sort(new_u)[0]
    new_v = torch.sort(new_v)[0]

    # =====================================================================
    # Step 6: Store and recompute basis
    # =====================================================================
    ctx.warped_us[uid] = new_u
    ctx.warped_vs[uid] = new_v

    # Recompute basis
    if surface.state.full_basis:
        u_grid = new_u.unsqueeze(1).expand(-1, Vs)
        v_grid = new_v.unsqueeze(0).expand(Us, -1)
        ctx.warped_uvs[uid] = torch.stack([u_grid, v_grid], dim=-1)

        ctx.uv_basis[uid] = surface.basis.forward(
            ctx.warped_uvs[uid],
            surface.knot_u(),
            surface.knot_v()
        )
    else:
        surface.uv_sampler.update_intervals(new_u, new_v)
        ctx.uv_basis[uid] = surface.basis.forward(
            (new_u, new_v),
            surface.knot_u(),
            surface.knot_v()
        )

    surface.invalidate_control_features()


# =========================================================================
# Batched Training Integration
# =========================================================================

@torch.no_grad()
def refresh_aggregation(
    surface: 'SplineModel',
    cameras: List,
    params: Optional[ChhuganiParams] = None,
    max_views: int = 0,
    aggregation_mode: str = 'max',
):
    """
    Refresh the aggregated importance state for a surface.

    Call this periodically during training (e.g., every N iterations or
    after subdivision/pruning changes the control grid).

    Usage in training loop:
        if iteration % aggregation_interval == 0:
            for surface in surfaces:
                refresh_aggregation(surface, training_cameras)
    """
    params = _get_or_create_params(surface, params)

    agg = aggregate_views(
        surface, cameras, params,
        max_views=max_views,
        aggregation_mode=aggregation_mode,
    )
    surface._chhugani_aggregation = agg
    agg.last_aggregation_iteration = getattr(surface, 'iteration', 0)


def mark_aggregation_stale(surface: 'SplineModel'):
    """
    Mark aggregation as stale after geometry changes (subdivision/pruning).

    Call after apply_subdivision() or apply_pruning().
    """
    if hasattr(surface, '_chhugani_aggregation'):
        surface._chhugani_aggregation.stale = True


# =========================================================================
# Global Interval Optimization (renders-based refinement)
# =========================================================================

@torch.no_grad()
def optimize_global_intervals(
    surface: 'SplineModel',
    cameras: List,
    render_fn,
    pipe,
    background,
    num_steps: int = 30,
    chhugani_weight: float = 0.1,
    num_render_views: int = 4,
    app_model=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Refine global intervals using rendering feedback.

    Combines Chhugani geometric importance with rendering-based
    gradient information to find globally optimal intervals.

    Args:
        surface: SplineModel
        cameras: All training cameras
        render_fn: Rendering function
        pipe: Pipeline parameters
        background: Background tensor
        num_steps: Optimization iterations
        chhugani_weight: Weight for Chhugani prior vs rendering gradient
        num_render_views: Views to render per optimization step
        app_model: Optional appearance model

    Returns:
        (global_u, global_v): Optimized global intervals
    """
    import random

    params = _get_or_create_params(surface)
    device = surface.position.control_features.device
    Us, Vs = surface.state.Us, surface.state.Vs

    # Start from current aggregated intervals or uniform
    agg = getattr(surface, '_chhugani_aggregation', None)
    if agg is not None and agg.global_intervals_u is not None:
        best_u = agg.global_intervals_u.clone()
        best_v = agg.global_intervals_v.clone()
    else:
        best_u = torch.linspace(0.0, 1.0, Us, device=device)
        best_v = torch.linspace(0.0, 1.0, Vs, device=device)

    # Compute Chhugani-based importance as a prior
    if agg is not None and agg.global_importance_u is not None:
        chhugani_imp_u = agg.global_importance_u
        chhugani_imp_v = agg.global_importance_v
    else:
        # Quick single-view approximation
        cam = cameras[0] if cameras else None
        if cam is not None:
            _, _, _, imp_u, imp_v = _compute_view_importance_components(
                surface, cam, params
            )
            chhugani_imp_u = imp_u
            chhugani_imp_v = imp_v
        else:
            chhugani_imp_u = torch.ones(params.coarse_u, device=device)
            chhugani_imp_v = torch.ones(params.coarse_v, device=device)

    best_loss = float('inf')

    for step in range(num_steps):
        # Select random subset of views
        view_subset = random.sample(cameras, min(num_render_views, len(cameras)))

        # Accumulate gradient-based importance from rendering
        render_importance_u = torch.zeros(Us, device=device)
        render_importance_v = torch.zeros(Vs, device=device)

        for cam in view_subset:
            # Set intervals and render
            surface.basis.forward(
                (best_u, best_v),
                surface.knot_u(),
                surface.knot_v()
            )
            surface.invalidate_control_features()

            render_pkg = render_fn(
                cam, surface, pipe, background,
                app_model=app_model,
                return_plane=False,
                return_depth_normal=False
            )

            # Use visibility and alpha as importance signal
            if 'visibility_filter' in render_pkg:
                vis = render_pkg['visibility_filter'].float()
                vis_grid = vis[:Us * Vs].reshape(Us, Vs)
                render_importance_u += vis_grid.mean(dim=1)
                render_importance_v += vis_grid.mean(dim=0)

        # Normalize
        render_importance_u = render_importance_u / max(len(view_subset), 1)
        render_importance_v = render_importance_v / max(len(view_subset), 1)

        # Combine Chhugani prior with rendering signal
        # Resample Chhugani importance to match output resolution
        combined_u = (1 - chhugani_weight) * render_importance_u + \
                     chhugani_weight * _resample_importance(chhugani_imp_u, Us, device)
        combined_v = (1 - chhugani_weight) * render_importance_v + \
                     chhugani_weight * _resample_importance(chhugani_imp_v, Vs, device)

        combined_u = combined_u.clamp(min=params.min_density)
        combined_v = combined_v.clamp(min=params.min_density)

        # Convert to intervals
        candidate_u = _importance_to_intervals(combined_u, Us, device)
        candidate_v = _importance_to_intervals(combined_v, Vs, device)

        # Blend toward candidate (conservative update)
        best_u = 0.7 * best_u + 0.3 * candidate_u
        best_v = 0.7 * best_v + 0.3 * candidate_v

        # Keep sorted
        best_u = torch.sort(best_u)[0]
        best_v = torch.sort(best_v)[0]

    # Store as global intervals
    if not hasattr(surface, '_global_intervals') or surface._global_intervals is None:
        surface._global_intervals = (best_u, best_v)
    else:
        # Blend with existing global intervals for stability
        old_u, old_v = surface._global_intervals
        surface._global_intervals = (
            0.5 * old_u + 0.5 * best_u,
            0.5 * old_v + 0.5 * best_v,
        )

    return surface._global_intervals


def _compute_view_importance_components(
    surface: 'SplineModel',
    camera,
    params: ChhuganiParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper: compute all importance components for a single view."""
    device = camera.camera_center.device

    flatness_u, flatness_v = _compute_flatness_map(surface, camera, params)
    cos_u, cos_v, bf_u, bf_v = _classify_patches(surface, camera, params)

    subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
    subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

    imp_u = _spans_to_importance(
        subdiv_u, surface.knot_u().detach(), surface.state.degree,
        params.coarse_u, cos_u, bf_u, params, device
    )
    imp_v = _spans_to_importance(
        subdiv_v, surface.knot_v().detach(), surface.state.degree,
        params.coarse_v, cos_v, bf_v, params, device
    )

    return flatness_u, flatness_v, cos_u, imp_u, imp_v


def _resample_importance(
    importance: torch.Tensor,  # [N_coarse]
    target_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Resample importance profile to different resolution via interpolation."""
    if importance.shape[0] == target_count:
        return importance

    return F.interpolate(
        importance.unsqueeze(0).unsqueeze(0),
        size=target_count,
        mode='linear',
        align_corners=True,
    ).squeeze()


# =========================================================================
# Helpers
# =========================================================================

def _get_or_create_params(
    surface: 'SplineModel',
    params: Optional[ChhuganiParams] = None,
) -> ChhuganiParams:
    """Get or create ChhuganiParams, caching on the surface."""
    if params is not None:
        return params
    if not hasattr(surface, '_chhugani_params'):
        surface._chhugani_params = ChhuganiParams(
            pixel_threshold=2.0,
            silhouette_threshold=0.2,
            backface_factor=0.05,
            temporal_alpha=0.0,
            min_density=0.01,
            coarse_u=max(surface.state.H * 4, 64),
            coarse_v=max(surface.state.W * 4, 64),
        )
    return surface._chhugani_params


def _ensure_forward_context(surface: 'SplineModel') -> ForwardContext:
    """Ensure the surface has a _forward_context attribute."""
    if not hasattr(surface, '_forward_context') or surface._forward_context is None:
        surface._forward_context = ForwardContext()
    return surface._forward_context


def blend_with_global_intervals(surface, new_u, new_v, global_weight=0.3):
    """
    If the surface has _global_intervals (from BatchedIntervalOptimizer),
    blend the per-view Chhugani intervals with the global ones.

    Insert this into update_uv_distribution_chhugani RIGHT BEFORE:
        ctx.warped_us[uid] = new_u
        ctx.warped_vs[uid] = new_v

    Example integration in chhugani.py:

        # ... after computing new_u, new_v and temporal/neighbor blending ...

        # Blend with global intervals if available
        if hasattr(surface, '_global_intervals') and surface._global_intervals is not None:
            global_u, global_v = surface._global_intervals
            gw = 0.3  # Global weight
            new_u = (1 - gw) * new_u + gw * global_u
            new_v = (1 - gw) * new_v + gw * global_v
            new_u = torch.sort(new_u)[0]
            new_v = torch.sort(new_v)[0]

        # Store and apply
        ctx.warped_us[uid] = new_u
        ctx.warped_vs[uid] = new_v
    """
    import torch

    if not hasattr(surface, '_global_intervals') or surface._global_intervals is None:
        return new_u, new_v

    global_u, global_v = surface._global_intervals

    # Ensure same device
    global_u = global_u.to(new_u.device)
    global_v = global_v.to(new_v.device)

    # Ensure same size (global may have been computed at different density)
    if global_u.shape[0] != new_u.shape[0]:
        global_u = torch.nn.functional.interpolate(
            global_u.unsqueeze(0).unsqueeze(0),
            size=new_u.shape[0],
            mode='linear',
            align_corners=True,
        ).squeeze()

    if global_v.shape[0] != new_v.shape[0]:
        global_v = torch.nn.functional.interpolate(
            global_v.unsqueeze(0).unsqueeze(0),
            size=new_v.shape[0],
            mode='linear',
            align_corners=True,
        ).squeeze()

    # Weighted blend
    blended_u = (1 - global_weight) * new_u + global_weight * global_u
    blended_v = (1 - global_weight) * new_v + global_weight * global_v

    # Re-sort for monotonicity
    blended_u = torch.sort(blended_u)[0]
    blended_v = torch.sort(blended_v)[0]

    return blended_u, blended_v

# =========================================================================
# Attach to SplineModel
# =========================================================================

def attach_chhugani_to_surface(surface: 'SplineModel'):
    """
    Attach Chhugani methods to a SplineModel instance.

    Call in Spline__init__():
        from modules.tessellation.chhugani import attach_chhugani_to_surface
        attach_chhugani_to_surface(self)
    """
    import types

    surface.update_uv_distribution_chhugani = types.MethodType(
        lambda self, *args, **kwargs: update_uv_distribution_chhugani(self, *args, **kwargs),
        surface
    )
    surface._forward_context = ForwardContext()
    surface._chhugani_aggregation = AggregationState()