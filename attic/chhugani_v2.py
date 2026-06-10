"""
View-Dependent Adaptive Tessellation (Chhugani et al., 2001)

Faithful adaptation of:
  "View-dependent Adaptive Tessellation of Spline Surfaces"
  https://www.cs.jhu.edu/graphics/papers/Chhugani01.pdf

ADAPTATION NOTES:
  The original algorithm is a recursive subdivision scheme that splits
  patches until a screen-space flatness criterion is met. Our architecture
  uses separable 1D UV intervals feeding into pre-computed B-spline basis
  matrices, so recursive 2D subdivision is not directly applicable.

  Instead, we faithfully implement the paper's THREE key components and
  convert their output into our 1D interval format:

  1. FLATNESS TEST (Section 3.1):
     Uses second differences of control points (NOT surface derivatives)
     projected to screen space. This gives a guaranteed upper bound on
     the screen-space deviation of the surface from its linear approx.

  2. FACE CLASSIFICATION (Section 3.2):
     Classifies each patch as front-facing, back-facing, or silhouette
     using a CONSISTENT normal orientation (not raw cross products).
     - Back-facing patches: minimal tessellation
     - Silhouette patches: threshold reduced by 1/cos(θ) factor
     - Front-facing patches: standard threshold

  3. ADAPTIVE DENSITY (Section 3.3):
     The number of subdivisions a patch needs is:
       n_subdiv = ceil(log2(max_deviation / threshold))
     which translates to a local sampling density of 2^n_subdiv.
     We convert this to importance weights for inverse-CDF sampling.

Integration:
  Output is always exactly (Us,) and (Vs,) sorted 1D interval arrays
  consumed by BasisFunction.forward() via the separable path.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

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
    uv_basis: Dict[int, 'BasisFuncs'] = field(default_factory=dict)
    camera_uid: Optional[int] = None


@dataclass
class ChhuganiParams:
    """
    Parameters for Chhugani's algorithm.

    These map directly to the paper's parameters:
    - pixel_threshold: ε in the flatness test (pixels)
    - silhouette_threshold: cos(θ) below which a front-face is "near-silhouette"
    - backface_factor: density multiplier for back-facing patches [0, 1]
    - temporal_alpha: EMA blending with previous frame [0, 1]
    - min_density: floor on relative density to prevent zero-sample regions
    - coarse_u/v: resolution of the evaluation grid for the flatness test
    """
    pixel_threshold: float = 2.0
    silhouette_threshold: float = 0.2
    backface_factor: float = 0.05
    temporal_alpha: float = 0.0
    min_density: float = 0.01
    coarse_u: int = 32
    coarse_v: int = 32
    global_weight: int = 0.5


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
            return mvp.T  # row-major → column-major
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

    return (proj @ view)


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


def _compute_control_point_second_differences(
    ctrl_xyz: torch.Tensor,  # [H, W, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute second differences of control points in U and V directions.

    This is the key quantity in Chhugani's flatness test. The second
    difference of control points BOUNDS the surface deviation from its
    linear approximation (convex hull property of B-splines).

    For a row of control points P_0, P_1, ..., P_n:
      Δ²P_i = P_{i-1} - 2*P_i + P_{i+1}

    The maximum screen-space length of these second differences, scaled
    by the appropriate basis function coefficients, gives an upper bound
    on the screen-space error of a linear approximation.

    Returns:
        d2P_u: [H-2, W, 3] second differences in U
        d2P_v: [H, W-2, 3] second differences in V
        d2P_uv: [H-2, W-2, 3] mixed second differences (cross term)
    """
    H, W, _ = ctrl_xyz.shape

    # Second differences in U: P[i-1] - 2*P[i] + P[i+1]
    if H >= 3:
        d2P_u = ctrl_xyz[:-2, :, :] - 2.0 * ctrl_xyz[1:-1, :, :] + ctrl_xyz[2:, :, :]
    else:
        d2P_u = torch.zeros(0, W, 3, device=ctrl_xyz.device)

    # Second differences in V: P[j-1] - 2*P[j] + P[j+1]
    if W >= 3:
        d2P_v = ctrl_xyz[:, :-2, :] - 2.0 * ctrl_xyz[:, 1:-1, :] + ctrl_xyz[:, 2:, :]
    else:
        d2P_v = torch.zeros(H, 0, 3, device=ctrl_xyz.device)

    # Mixed second differences (for the cross term)
    if H >= 3 and W >= 3:
        d2P_uv = (
            ctrl_xyz[:-2, :-2, :] - ctrl_xyz[:-2, 2:, :]
            - ctrl_xyz[2:, :-2, :] + ctrl_xyz[2:, 2:, :]
        ) / 4.0  # Central difference for mixed partial
    else:
        d2P_uv = torch.zeros(0, 0, 3, device=ctrl_xyz.device)

    return d2P_u, d2P_v, d2P_uv


def _compute_flatness_map(
    surface: 'SplineModel',
    camera,
    params: ChhuganiParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chhugani's flatness test (Section 3.1).

    For each knot span, compute the maximum screen-space deviation of
    the surface from its linear (bilinear) approximation. This is bounded
    by the screen-space length of the control point second differences.

    The bound for a cubic B-spline patch in span [i, i+1] is:
      δ_screen ≤ (1/8) * max_j(||P_screen(Δ²P_j)||)

    where Δ²P_j are the second differences of the 4 control points
    influencing that span, and the 1/8 factor comes from the maximum
    of the cubic B-spline second derivative basis function.

    Returns:
        flatness_u: [num_spans_u] max screen-space deviation per U-span
        flatness_v: [num_spans_v] max screen-space deviation per V-span
    """
    device = camera.camera_center.device
    H, W = surface.state.H, surface.state.W

    # Get control points
    ctrl_xyz = surface.position.control_features.detach().view(H, W, -1)[..., :3]

    # Handle rational (NURBS) case
    has_weights = (
        surface.position.weights is not None
        and surface.position.weights.control_features is not None
    )
    if has_weights:
        w_ctrl = surface.position.weights.activation(
            surface.position.weights.control_features.detach()
        ).view(H, W, 1)
        # For rational case, work with weighted points (projective space)
        ctrl_xyz = ctrl_xyz * w_ctrl

    # Compute second differences
    d2P_u, d2P_v, d2P_uv = _compute_control_point_second_differences(ctrl_xyz)

    # Project control points AND their second differences to screen space
    mvp = _build_mvp(camera)
    img_w = int(camera.image_width)
    img_h = int(camera.image_height)

    # Project all control points to screen
    ctrl_screen = _project_to_screen(
        ctrl_xyz.reshape(-1, 3), mvp, img_w, img_h
    ).reshape(H, W, 2)

    # Screen-space second differences (the actual quantity we need)
    # These bound the screen-space deviation of the surface from linear
    if d2P_u.shape[0] > 0:
        d2_screen_u = (
            ctrl_screen[:-2, :, :] - 2.0 * ctrl_screen[1:-1, :, :] + ctrl_screen[2:, :, :]
        )  # [H-2, W, 2]
        # Magnitude of screen-space second differences
        d2_mag_u = d2_screen_u.norm(dim=-1)  # [H-2, W]
    else:
        d2_mag_u = torch.zeros(0, W, device=device)

    if d2P_v.shape[0] > 0:
        d2_screen_v = (
            ctrl_screen[:, :-2, :] - 2.0 * ctrl_screen[:, 1:-1, :] + ctrl_screen[:, 2:, :]
        )  # [H, W-2, 2]
        d2_mag_v = d2_screen_v.norm(dim=-1)  # [H, W-2]
    else:
        d2_mag_v = torch.zeros(H, 0, device=device)

    # For cubic B-splines, the maximum of the second derivative basis
    # function within a span is 1/6 * 6 = 1 (the B3'' peak), and the
    # deviation bound is (Δu²/8) * max(||Δ²P||_screen).
    # Since we're working in the [0,1] parameter domain with uniform
    # knots, Δu ≈ 1/(H-degree) per span. But for non-uniform knots,
    # the span width varies.

    knots_u = surface.knot_u().detach()
    knots_v = surface.knot_v().detach()
    degree = surface.state.degree

    # Number of spans
    num_spans_u = H - degree  # interior spans
    num_spans_v = W - degree

    # Compute flatness per span in U direction
    # For span i (knot interval [knots[i+degree], knots[i+degree+1]]):
    # The influencing control points are P[i], P[i+1], ..., P[i+degree]
    # The second differences among these are d2P_u[i], ..., d2P_u[i+degree-2]
    flatness_u = torch.zeros(num_spans_u, device=device)
    for span_i in range(num_spans_u):
        # Span width in parameter space
        span_width = (knots_u[span_i + degree + 1] - knots_u[span_i + degree]).abs()
        if span_width < 1e-10:
            continue

        # Second differences influencing this span
        # For cubic (degree=3), control points i..i+3, second diffs i..i+1
        sd_start = max(0, span_i)
        sd_end = min(span_i + degree - 1, d2_mag_u.shape[0])
        if sd_start >= sd_end:
            continue

        # Max screen-space second difference across all V columns
        max_d2 = d2_mag_u[sd_start:sd_end, :].max()

        # Chhugani's bound: deviation ≤ (Δu² / 8) * max(||Δ²P_screen||)
        # The 1/8 comes from the maximum of the normalized cubic B-spline
        # second derivative
        flatness_u[span_i] = (span_width ** 2 / 8.0) * max_d2

    # Same for V direction
    flatness_v = torch.zeros(num_spans_v, device=device)
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

    Uses a coarse evaluation grid to determine face orientation.
    Normal orientation is made CONSISTENT by orienting all normals
    to point toward the camera (since we're doing view-dependent
    tessellation, this is the correct convention).

    Returns:
        cos_angles_u: [coarse_u] per-U-row min |cos(angle)| (for silhouette detection)
        cos_angles_v: [coarse_v] per-V-col min |cos(angle)|
        backface_u: [coarse_u] fraction of backfacing samples per U-row
        backface_v: [coarse_v] fraction of backfacing samples per V-col
    """
    from model.modules.basis import generate_bspline_basis_matrices
    import opt_einsum as oe

    device = camera.camera_center.device
    Nu, Nv = params.coarse_u, params.coarse_v
    H, W = surface.state.H, surface.state.W

    u_coarse = torch.linspace(0.0, 1.0, Nu, device=device)
    v_coarse = torch.linspace(0.0, 1.0, Nv, device=device)

    knot_u = surface.knot_u().detach()
    knot_v = surface.knot_v().detach()

    # Compute basis and first derivatives at coarse grid
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

    # Compute normals
    normals = torch.cross(dSu, dSv, dim=-1)  # [Nu, Nv, 3]
    normals = F.normalize(normals, dim=-1, eps=1e-8)

    # View direction: surface → camera
    cam_center = camera.camera_center.to(device)
    view_dirs = F.normalize(cam_center.unsqueeze(0).unsqueeze(0) - S, dim=-1, eps=1e-8)

    # Signed cos(angle)
    # Positive = front-facing, Negative = back-facing
    cos_raw = (normals * view_dirs).sum(dim=-1)  # [Nu, Nv]

    # CONSISTENT ORIENTATION: For view-dependent tessellation, we care
    # about the absolute orientation w.r.t. the camera. If the normal
    # points away (cos < 0), the patch is genuinely back-facing.
    # But B-spline normals from cross(dSu, dSv) may have arbitrary
    # orientation depending on parameterization. We detect this by
    # checking if the MAJORITY of normals point away — if so, the
    # surface parameterization is flipped and we should negate.

    # Heuristic: if more than 60% of samples have cos < 0, flip all normals
    backface_ratio = (cos_raw < 0).float().mean()
    if backface_ratio > 0.6:
        cos_raw = -cos_raw

    # Now cos_raw > 0 means front-facing (toward camera)
    is_backface = cos_raw < 0  # [Nu, Nv]

    # Per-row/col statistics for separable decomposition
    # For silhouette: minimum |cos| along the orthogonal direction
    # (if ANY point in a row is near-silhouette, the whole row needs care)
    abs_cos = cos_raw.abs()

    # Per U-row: minimum |cos| across V (worst case for silhouette)
    cos_angles_u = abs_cos.min(dim=1).values  # [Nu]
    # Per V-col: minimum |cos| across U
    cos_angles_v = abs_cos.min(dim=0).values  # [Nv]

    # Backface fraction per row/col
    backface_u = is_backface.float().mean(dim=1)  # [Nu]
    backface_v = is_backface.float().mean(dim=0)  # [Nv]

    return cos_angles_u, cos_angles_v, backface_u, backface_v


# =========================================================================
# Step 3: Adaptive Density Computation (Chhugani Section 3.3)
# =========================================================================

def _compute_subdivision_depth(
    flatness: torch.Tensor,    # [num_spans] screen-space deviation per span
    threshold: float,          # pixel threshold ε
) -> torch.Tensor:
    """
    Compute how many times each span needs to be subdivided.

    From Chhugani: if the flatness test value δ exceeds threshold ε,
    the span needs ceil(log2(δ/ε)) subdivisions. Each subdivision
    halves the deviation (quadratic convergence for cubic splines).

    Returns:
        [num_spans] number of required subdivisions (float, can be fractional)
    """
    # Ratio of deviation to threshold
    ratio = flatness / max(threshold, 1e-6)

    # Number of subdivisions: log2(ratio), clamped to [0, max_depth]
    # We keep it as float for smooth density computation
    depth = torch.log2(ratio.clamp(min=1.0))  # 0 if already flat enough

    return depth


def _spans_to_importance(
    subdiv_depth: torch.Tensor,  # [num_spans] subdivision depth per span
    knots: torch.Tensor,         # full knot vector
    degree: int,
    num_coarse: int,             # coarse grid size for face classification
    cos_angles: torch.Tensor,    # [num_coarse] min |cos(angle)| per row/col
    backface_frac: torch.Tensor, # [num_coarse] backface fraction per row/col
    params: ChhuganiParams,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert per-span subdivision depths into a per-sample importance
    profile, incorporating face classification.

    Strategy:
    1. Map span depths to a fine 1D importance profile
    2. Apply silhouette boost (reduce threshold → increase depth)
    3. Apply backface reduction
    4. Floor at min_density

    Returns:
        [num_coarse] importance profile
    """
    num_spans = subdiv_depth.shape[0]
    if num_spans == 0:
        return torch.ones(num_coarse, device=device) * params.min_density

    # Create fine importance profile by mapping coarse samples to spans
    # First, get span boundaries in [0, 1] parameter space
    unique_knots = torch.unique(knots)
    # Interior knot boundaries define spans
    span_boundaries = unique_knots[degree:-degree] if len(unique_knots) > 2 * degree else unique_knots

    if len(span_boundaries) < 2:
        return torch.ones(num_coarse, device=device) * params.min_density

    # Map coarse sample positions to spans
    coarse_positions = torch.linspace(0.0, 1.0, num_coarse, device=device)

    # Find which span each coarse sample falls into
    span_idx = torch.searchsorted(span_boundaries, coarse_positions, right=True) - 1
    span_idx = span_idx.clamp(0, num_spans - 1)

    # Base importance = 2^depth (relative density needed)
    base_importance = (2.0 ** subdiv_depth).clamp(min=1.0)

    # Map to coarse grid
    importance = base_importance[span_idx]  # [num_coarse]

    # --- Silhouette modification (Chhugani Section 3.2) ---
    # Near-silhouette patches need MORE tessellation because the outline
    # accuracy is critical. The paper reduces the threshold by 1/cos(θ),
    # which is equivalent to multiplying importance by 1/cos(θ).
    silhouette_factor = torch.ones(num_coarse, device=device)
    near_silhouette = cos_angles < params.silhouette_threshold
    if near_silhouette.any():
        # 1/cos(θ) boost, clamped to prevent infinity
        boost = 1.0 / cos_angles[near_silhouette].clamp(min=0.01)
        silhouette_factor[near_silhouette] = boost.clamp(max=20.0)

    importance = importance * silhouette_factor

    # --- Backface reduction (Chhugani Section 3.2) ---
    # Back-facing patches need minimal tessellation.
    # We reduce importance proportional to backface fraction.
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
    inverse-CDF sampling.

    Regions with higher importance get denser sampling.

    Returns: [target_count] sorted sample positions in [0, 1].
    """
    N = importance.shape[0]

    total = importance.sum()
    if total < 1e-10:
        return torch.linspace(0.0, 1.0, target_count, device=device)

    pdf = importance / total
    cdf = torch.cumsum(pdf, dim=0)

    # Build CDF with proper boundaries
    cdf_full = torch.cat([torch.zeros(1, device=device), cdf])
    cdf_full[-1] = 1.0

    coarse_positions = torch.linspace(0.0, 1.0, N, device=device)
    half_step = 0.5 / max(N - 1, 1) if N > 1 else 0.0
    pos_full = torch.cat([
        torch.tensor([max(0.0, coarse_positions[0].item() - half_step)], device=device),
        coarse_positions,
    ])

    # Uniform quantiles
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
# Main Entry Point
# =========================================================================

@torch.no_grad()
def update_uv_distribution_chhugani(
    surface: 'SplineModel',
    camera,
    neighbor_cameras: Optional[List] = None,
    params: Optional[ChhuganiParams] = None,
    blend_neighbors: bool = False,
    neighbor_weight: float = 0.1,
):
    """
    Update UV sampling intervals using Chhugani's view-dependent
    adaptive tessellation algorithm.

    Faithful to the paper's three-step process:
      1. Flatness test: Compute screen-space deviation bound per span
         using control point second differences.
      2. Face classification: Identify front/back/silhouette patches.
         - Back-facing → reduce density
         - Silhouette → reduce threshold (increase density)
      3. Density computation: Convert flatness + classification into
         non-uniform 1D intervals via inverse-CDF sampling.
    """
    if params is None:
        if not hasattr(surface, '_chhugani_params'):
            surface._chhugani_params = ChhuganiParams(
                pixel_threshold=2.0,
                silhouette_threshold=0.2,
                backface_factor=0.05,
                temporal_alpha=0.0,
                min_density=0.01,
                coarse_u=max(surface.state.H * 2, 64),
                coarse_v=max(surface.state.W * 2, 64),
                global_weight=0.3  # How much to trust the global optimization

            )
        params = surface._chhugani_params
    global_weight = params.global_weight
    ctx = _ensure_forward_context(surface)
    uid = camera.uid
    device = camera.camera_center.device
    Us, Vs = surface.state.Us, surface.state.Vs

    # =====================================================================
    # Step 1: Flatness test (Section 3.1)
    # =====================================================================
    flatness_u, flatness_v = _compute_flatness_map(surface, camera, params)

    # =====================================================================
    # Step 2: Face classification (Section 3.2)
    # =====================================================================
    cos_angles_u, cos_angles_v, backface_u, backface_v = _classify_patches(
        surface, camera, params
    )

    # =====================================================================
    # Step 3: Compute adaptive density (Section 3.3)
    # =====================================================================

    # 3a. Subdivision depth from flatness test
    subdiv_u = _compute_subdivision_depth(flatness_u, params.pixel_threshold)
    subdiv_v = _compute_subdivision_depth(flatness_v, params.pixel_threshold)

    # 3b. Convert to importance with face classification
    importance_u = _spans_to_importance(
        subdiv_u, surface.knot_u().detach(), surface.state.degree,
        params.coarse_u, cos_angles_u, backface_u, params, device
    )
    importance_v = _spans_to_importance(
        subdiv_v, surface.knot_v().detach(), surface.state.degree,
        params.coarse_v, cos_angles_v, backface_v, params, device
    )

    # 3c. Convert importance to intervals
    new_u = _importance_to_intervals(importance_u, Us, device)
    new_v = _importance_to_intervals(importance_v, Vs, device)

    # =====================================================================
    # Step 4: Temporal blending (not in original paper, added for stability)
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

    # =================================================================
    # Step 5: Temporal blending (not in paper, for stability)
    # =================================================================
    if params.temporal_alpha > 0 and uid in ctx.warped_us:
        alpha = params.temporal_alpha
        new_u = (1 - alpha) * new_u + alpha * ctx.warped_us[uid]
        new_v = (1 - alpha) * new_v + alpha * ctx.warped_vs[uid]

    # Neighbor blending (not in original paper)
    if blend_neighbors and neighbor_cameras:
        for ncam in neighbor_cameras:
            f_u, f_v = _compute_flatness_map(surface, ncam, params)
            c_u, c_v, b_u, b_v = _classify_patches(surface, ncam, params)
            sd_u = _compute_subdivision_depth(f_u, params.pixel_threshold)
            sd_v = _compute_subdivision_depth(f_v, params.pixel_threshold)
            imp_u = _spans_to_importance(
                sd_u, surface.knot_u().detach(), surface.state.degree,
                params.coarse_u, c_u, b_u, params, device
            )
            imp_v = _spans_to_importance(
                sd_v, surface.knot_v().detach(), surface.state.degree,
                params.coarse_v, c_v, b_v, params, device
            )
            n_u = _importance_to_intervals(imp_u, Us, device)
            n_v = _importance_to_intervals(imp_v, Vs, device)
            new_u = (1 - neighbor_weight) * new_u + neighbor_weight * n_u
            new_v = (1 - neighbor_weight) * new_v + neighbor_weight * n_v

    # Ensure sorted
    new_u = torch.sort(new_u)[0]
    new_v = torch.sort(new_v)[0]

    # =====================================================================
    # Step 5: Store and recompute basis
    # =====================================================================
    ctx.warped_us[uid] = new_u
    ctx.warped_vs[uid] = new_v

    # Store UV grid if needed
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
        ctx.uv_basis[uid] = surface.basis.forward(
            (new_u, new_v),
            surface.knot_u(),
            surface.knot_v()
        )

    surface.invalidate_control_features()


# =========================================================================
# Helpers
# =========================================================================

def _ensure_forward_context(surface: 'SplineModel') -> ForwardContext:
    """Ensure the surface has a _forward_context attribute."""
    if not hasattr(surface, '_forward_context') or surface._forward_context is None:
        surface._forward_context = ForwardContext()
    return surface._forward_context


# =========================================================================
# Attach to SplineModel
# =========================================================================

def attach_chhugani_to_surface(surface: 'SplineModel'):
    """
    Attach update_uv_distribution_chhugani as a bound method.

    Call in SplineModel.__init__():
        from model.modules.tessellation.chhugani import attach_chhugani_to_surface
        attach_chhugani_to_surface(self)
    """
    import types
    surface.update_uv_distribution_chhugani = types.MethodType(
        lambda self, *args, **kwargs: update_uv_distribution_chhugani(self, *args, **kwargs),
        surface
    )
    surface._forward_context = ForwardContext()