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
    #          res_cap  post_fit_iters
    'raw':      (144,    0),       # LS fit only, no Chamfer refinement
    'fast':     (96,     300),
    'balanced': (144,    1500),
    'fine':     (192,    3000),
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
    res_cap, post_iters = QUALITY_PRESETS[quality]

    H, W = _auto_resolution(points, res_cap)
    smoothing = _auto_smoothing(points)
    print(
        f"[SimpleInit] quality={quality}: grid {H}x{W}, "
        f"smoothing {smoothing:.4f}, post-fit {post_iters} iters"
    )

    return create_nurbs_from_pointcloud(
        points, colors,
        mode=DecompositionMode.SINGLE,
        generate_adaptive_samples=False,
        cameras=cameras,
        smoothing=smoothing,
        parameterization=parameterization,
        # pin the auto resolution (bypass the internal calculator)
        bg_resolution=None,
        object_resolution=None,
        min_resolution=min(H, W),
        max_resolution=max(H, W),
        resolution=(H, W),
        post_fit_enabled=post_iters > 0,
        post_fit_iterations=post_iters,
    )
