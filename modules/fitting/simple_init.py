"""
Lean, semi-autonomous initial B-spline surface estimation.

One call, three user-facing knobs:

    surf_data = fit_initial_surface(points, colors, cameras, quality='balanced')

Everything else (control resolution, LS smoothing, parameterization,
Chamfer post-fit budget) is derived from the data:

  * resolution   — from point count and spatial anisotropy of the cloud
                   (PCA aspect ratio splits the budget between u and v)
  * smoothing    — from the estimated noise level (median nn-distance
                   relative to the bounding-box diagonal)
  * post-fit     — iteration budget by quality preset; the refinement is
                   already self-guarding (EMA best-tracking + dense-CD
                   no-harm check), so a generous budget cannot hurt

The heavy lifting reuses the tested fitting stack (least-squares fit +
BSplinePostFitter); this module only removes the configuration burden.
"""

import numpy as np
import torch

QUALITY_PRESETS = {
    #          target_res  post_fit_iters
    'raw':      (128,       0),      # MBA fit only, no Chamfer refinement
    'fast':     (128,       300),
    'balanced': (192,       1500),
    'fine':     (256,       3000),
}


def _auto_resolution(points: np.ndarray, cap: int):
    """Control-grid (H, W) from point count + cloud anisotropy."""
    n = len(points)
    # ~8 points per control coefficient keeps LS well-determined.
    base = int(np.clip(np.sqrt(n / 8.0), 32, cap))

    # Split the budget by the cloud's tangential aspect ratio.
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(n - 1, 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    aspect = float(np.sqrt(max(eigvals[0], 1e-12) / max(eigvals[1], 1e-12)))
    aspect = float(np.clip(aspect, 1.0, 2.0))

    H = int(np.clip(base * np.sqrt(aspect), 16, cap))
    W = int(np.clip(base / np.sqrt(aspect), 16, cap))
    return H, W


def _auto_smoothing(points: np.ndarray) -> float:
    """LS regularization from the cloud's relative noise level."""
    from scipy.spatial import cKDTree
    n = len(points)
    sample = points[:: max(1, n // 4000)]
    d, _ = cKDTree(points).query(sample, k=2)
    nn = float(np.median(d[:, 1]))
    diag = float(np.linalg.norm(points.max(0) - points.min(0))) + 1e-12
    rel_noise = nn / diag
    # Sparse/noisy clouds (large nn spacing) need more smoothing.
    return float(np.clip(rel_noise * 5.0, 0.005, 0.1))


def fit_initial_surface(
    points,
    colors=None,
    cameras=None,
    quality: str = 'balanced',
    parameterization: str = 'spherical',
):
    """
    Fit ONE B-spline surface to a point cloud with auto-derived settings.

    Args:
        points:  [N, 3] numpy array or tensor.
        colors:  [N, 3] optional.
        cameras: optional training cameras (observation-weighted fitting).
        quality: 'fast' | 'balanced' | 'fine'.
        parameterization: 'spherical' | 'geodesic' | 'pca'.

    Returns:
        MultiSurfaceResult with exactly one fitted surface.
    """
    from modules.fitting.nurbs_from_pointcloud import (
        create_nurbs_from_pointcloud, DecompositionMode,
    )

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"quality must be one of {list(QUALITY_PRESETS)}, got {quality!r}"
        )
    target_res, post_iters = QUALITY_PRESETS[quality]

    from modules.fitting.nurbs_from_pointcloud import (
        NURBSSurfaceFitter, SurfaceConfig, NURBSSurfaceData,
        MultiSurfaceResult, DecompositionMode, BSplinePostFitter,
        PostFitConfig, PointCloudProcessor,
    )
    from modules.fitting.mba import mba_fit_surface

    # --- parameterize (reuse the tested parameterizations) ---
    fitter = NURBSSurfaceFitter(SurfaceConfig())
    if parameterization == 'spherical':
        uv = fitter._parameterize_spherical(points)
    elif parameterization == 'geodesic':
        uv = fitter._parameterize_geodesic(points)
    else:
        uv = fitter._parameterize_pca(points)

    # --- resolution: preset target, split by cloud anisotropy ---
    # (MBA needs no points-per-coefficient floor: sparse cells inherit
    # from coarser levels by construction.)
    centered = points - points.mean(axis=0)
    eig = np.sort(np.linalg.eigvalsh(centered.T @ centered))[::-1]
    aspect = float(np.clip(np.sqrt(max(eig[0], 1e-12) / max(eig[1], 1e-12)), 1.0, 1.5))
    H = int(np.clip(target_res * np.sqrt(aspect), 32, 256))
    W = int(np.clip(target_res / np.sqrt(aspect), 32, 256))

    # --- Multilevel B-spline Approximation (coarse-to-fine residuals) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vals = np.concatenate(
        [points, colors if colors is not None else np.full_like(points, 0.5)],
        axis=1,
    )
    ctrl, ku, kv = mba_fit_surface(
        torch.tensor(uv, dtype=torch.float32),
        torch.tensor(vals, dtype=torch.float32),
        H, W, device=device,
    )
    ctrl = ctrl.cpu().numpy()
    print(
        f"[SimpleInit] quality={quality}: MBA grid {H}x{W} "
        f"({len(points)} pts), post-fit {post_iters} iters"
    )

    surface = NURBSSurfaceData(
        control_points=ctrl[..., :3].astype(np.float32),
        control_colors=np.clip(ctrl[..., 3:6], 0, 1).astype(np.float32),
        knots_u=ku.cpu().numpy().astype(np.float32),
        knots_v=kv.cpu().numpy().astype(np.float32),
        degree_u=3, degree_v=3,
        label='main',
    )
    surface.point_indices = np.arange(len(points))
    surface.bounds = {
        'min': points.min(axis=0), 'max': points.max(axis=0),
        'center': points.mean(axis=0),
    }

    # --- optional Chamfer post-fit (self-guarding: no-harm check) ---
    if post_iters > 0:
        proc = PointCloudProcessor(points, colors)
        normals = proc.estimate_normals()
        surface = BSplinePostFitter(PostFitConfig(
            num_iterations=post_iters, verbose=True,
        )).refine(surface, target_points=points, target_normals=normals)

    labels = np.zeros(len(points), dtype=np.int32)
    return MultiSurfaceResult(
        surfaces=[surface],
        decomposition_mode=DecompositionMode.SINGLE,
        labels=labels,
    )
