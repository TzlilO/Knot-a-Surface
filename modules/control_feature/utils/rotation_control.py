"""
Bridge utility: compute per-control-point quaternion initialization
from TSDF mesh normals, to be injected into the Stage 2 SplineModel.

This replaces the naive identity-quaternion init (or the tangent-derived
init from noisy early-stage spline tangents) with a geometrically
grounded rotation prior extracted from the mesh reconstructed in Stage 1.
"""

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
def bridge_mesh_normals_to_control_quaternions(
    mesh: o3d.geometry.TriangleMesh,
    fit_result: dict,
    smoothing_iters: int = 3,
    orientation_reference: str = 'z_positive',
) -> torch.Tensor:
    """
    Compute per-control-point quaternion initialization from mesh normals.

    The pipeline:
        1. Ensure mesh has consistent vertex normals.
        2. Sample the mesh densely (Poisson disk) to get (point, normal) pairs.
        3. Use the UV parameterization from `fit_result` to map normals
           onto the regular control grid via griddata interpolation.
        4. At each control grid cell, average the normals → single direction.
        5. Convert per-cell normal vectors to unit quaternions that rotate
           [0, 0, 1] to align with the normal (compatible with the
           quaternion_to_normal / normals_to_quaternions2 conventions
           already in the codebase).
        6. Optionally smooth quaternions spatially on the grid via iterative
           geodesic averaging to suppress noise.

    Args:
        mesh: Open3D TriangleMesh with computed vertex normals.
        fit_result: dict from bridge_fit_spline_to_mesh, must contain:
            - 'control_points': [H, W, 3] fitted B-spline control grid
            - 'uv': [N, 2] UV coordinates of the sampled points
            - 'points': [N, 3] 3D positions of the sampled points
        smoothing_iters: Number of Laplacian-style quaternion smoothing
            passes on the control grid. 0 = no smoothing.
        orientation_reference: How to orient normals consistently.
            'z_positive': flip normals so z-component >= 0 (matches
                          uv_tangent convention in spline_formulas.py)
            'centroid_outward': flip normals to point away from mesh centroid

    Returns:
        quaternions: [H, W, 4] tensor of unit quaternions in (w, x, y, z)
            format, ready to be injected into the fit_result dict or
            directly into RotationControl.
    """
    # ── 0. Validate inputs ──────────────────────────────────────────
    assert mesh.has_vertex_normals(), (
        "Mesh must have vertex normals. Call mesh.compute_vertex_normals() first."
    )
    assert 'control_points' in fit_result, "fit_result must contain 'control_points'"
    assert 'uv' in fit_result, "fit_result must contain 'uv'"
    assert 'points' in fit_result, "fit_result must contain 'points'"

    ctrl_pts = fit_result['control_points']  # [H, W, 3]
    uv = fit_result['uv']                    # [N, 2]
    sample_points = fit_result['points']     # [N, 3]
    H, W, _ = ctrl_pts.shape

    # ── 1. Extract mesh vertices + normals ──────────────────────────
    mesh_verts = np.asarray(mesh.vertices, dtype=np.float32)
    # mesh_normals = np.asarray(mesh.v, dtype=np.float32)
    mesh_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    # ── 2. Transfer normals to the sampled points via nearest-neighbor
    #        (sample_points came from Poisson disk sampling of this mesh,
    #         so they're on the surface but not at vertex positions) ──
    tree = cKDTree(mesh_verts)
    _, nn_idx = tree.query(sample_points, k=1)
    sample_normals = mesh_normals[nn_idx]  # [N, 3]

    # Normalize (mesh normals should already be unit, but be safe)
    norms = np.linalg.norm(sample_normals, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    sample_normals = sample_normals / norms

    # ── 3. Orient normals consistently ──────────────────────────────
    if orientation_reference == 'z_positive':
        # Flip normals whose z < 0 (matches uv_tangent convention:
        # normals[vec <= 0] *= -1 where vec = normals[..., 2])
        flip_mask = sample_normals[:, 2] < 0
        sample_normals[flip_mask] *= -1
    elif orientation_reference == 'centroid_outward':
        centroid = mesh_verts.mean(axis=0)
        to_point = sample_points - centroid[None, :]
        dots = np.sum(sample_normals * to_point, axis=-1)
        flip_mask = dots < 0
        sample_normals[flip_mask] *= -1
    else:
        raise ValueError(f"Unknown orientation_reference: {orientation_reference}")

    # ── 4. Interpolate normals onto control grid ────────────────────
    #    Build a regular UV grid matching [H, W], then use griddata
    #    to scatter-interpolate the per-sample normals.
    u_grid = np.linspace(0.001, 0.999, H)
    v_grid = np.linspace(0.001, 0.999, W)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing='ij')
    grid_uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)  # [H*W, 2]

    grid_normals = np.zeros((H * W, 3), dtype=np.float32)
    for dim in range(3):
        grid_normals[:, dim] = griddata(
            uv, sample_normals[:, dim], grid_uv,
            method='linear',
            fill_value=0.0,
        )

    # Fill NaN holes with nearest-neighbor
    nan_mask = np.isnan(grid_normals).any(axis=1)
    if nan_mask.any():
        for dim in range(3):
            grid_normals[nan_mask, dim] = griddata(
                uv, sample_normals[:, dim], grid_uv[nan_mask],
                method='nearest',
            )

    # Re-normalize after interpolation (linear interp of unit vectors
    # doesn't yield unit vectors)
    grid_normals_norms = np.linalg.norm(grid_normals, axis=-1, keepdims=True)
    grid_normals_norms = np.clip(grid_normals_norms, 1e-8, None)
    grid_normals = grid_normals / grid_normals_norms

    grid_normals = grid_normals.reshape(H, W, 3)

    # ── 5. Convert normals → quaternions ────────────────────────────
    #    We want the quaternion that rotates [0, 0, 1] to align with
    #    the normal. This is the same convention as normals_to_quaternions2
    #    and vectors_to_quaternions in modules/spline_utils.py.
    normals_t = torch.from_numpy(grid_normals).float().reshape(-1, 3).to('cuda')  # [H*W, 3]
    quats = _normals_to_quaternions_robust(normals_t)  # [H*W, 4]
    quats = quats.reshape(H, W, 4)

    # ── 6. Spatial smoothing on the quaternion grid ─────────────────
    #    Iterative 3×3 geodesic mean to suppress high-frequency noise
    #    from mesh discretization artifacts.
    for _ in range(smoothing_iters):
        quats = _smooth_quaternion_grid(quats)

    # Final normalization
    quats = F.normalize(quats, dim=-1)

    print(f"[Bridge] Computed control quaternions: [{H}, {W}, 4], "
          f"smoothing_iters={smoothing_iters}")

    return quats


def _normals_to_quaternions_robust(
    normals: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Convert (N, 3) unit normals to (N, 4) quaternions (w, x, y, z)
    representing rotation from [0, 0, 1] to the given normal.

    Handles the degenerate case (normal ≈ [0, 0, -1]) explicitly.
    This is a cleaned-up version of normals_to_quaternions2 from
    modules/spline_utils.py with better numerical stability.
    """
    N = normals.shape[0]
    normals = F.normalize(normals, dim=-1)

    ref = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
    ref = ref.expand(N, 3)

    dot = (ref * normals).sum(dim=-1, keepdim=True)    # [N, 1]
    cross = torch.cross(ref, normals, dim=1)             # [N, 3]

    # Standard case: q = [1 + dot, cross], then normalize
    w = dot + 1.0  # [N, 1]
    quat = torch.cat([w, cross], dim=-1)  # [N, 4]

    # Degenerate case: normals ≈ -ref (anti-parallel)
    # Pick an arbitrary perpendicular axis for 180° rotation
    anti_mask = (w.abs() < eps).squeeze(-1)  # [N]
    if anti_mask.any():
        # Use the axis with smallest |component| to find perpendicular
        anti_normals = normals[anti_mask]
        abs_n = anti_normals.abs()
        min_idx = abs_n.argmin(dim=1)

        # Build unit vector along min axis
        perp = torch.zeros_like(anti_normals)
        perp[torch.arange(perp.shape[0]), min_idx] = 1.0

        # Gram-Schmidt: make perpendicular to anti_normals
        ortho = perp - (perp * anti_normals).sum(dim=-1, keepdim=True) * anti_normals
        ortho = F.normalize(ortho, dim=-1)

        # 180° rotation: q = [0, ortho]
        quat[anti_mask, 0] = 0.0
        quat[anti_mask, 1:] = ortho

    # Normalize to unit quaternion
    quat = F.normalize(quat, dim=-1)

    # Standardize: ensure w >= 0 (hemisphere consistency)
    flip = quat[:, 0] < 0
    quat[flip] = -quat[flip]

    return quat


def _smooth_quaternion_grid(
    quats: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    One pass of 3×3 geodesic-aware quaternion smoothing on [H, W, 4] grid.

    Uses sign-corrected linear averaging (fast approximation to Riemannian
    mean on SO(3) that's sufficient for smooth grids).

    The key insight: for quaternions that are already close to each other
    (as they should be on a smooth surface), linear averaging + renorm
    approximates the geodesic (Karcher) mean to first order.
    """
    H, W, _ = quats.shape
    pad = kernel_size // 2

    # Ensure hemisphere consistency: flip neighbors to same hemisphere
    # as center before averaging (prevents cancellation at q / -q boundary)
    smoothed = quats.clone()

    for i in range(H):
        for j in range(W):
            center = quats[i, j]  # [4]
            acc = torch.zeros(4, device=quats.device, dtype=quats.dtype)
            count = 0

            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor = quats[ni, nj]
                        # Flip to same hemisphere as center
                        if (center * neighbor).sum() < 0:
                            neighbor = -neighbor
                        acc += neighbor
                        count += 1

            smoothed[i, j] = F.normalize(acc.unsqueeze(0), dim=-1).squeeze(0)

    return smoothed


def inject_quaternions_into_fit_result(
    fit_result: dict,
    quats: torch.Tensor,
) -> dict:
    """
    Convenience: add the computed quaternions to the fit_result dict
    so that stage2_full_training can pass them through to SplineModel.

    Args:
        fit_result: dict from bridge_fit_spline_to_mesh
        quats: [H, W, 4] from bridge_mesh_normals_to_control_quaternions

    Returns:
        Updated fit_result with 'control_quaternions' key added.
    """
    fit_result['control_quaternions'] = quats
    return fit_result