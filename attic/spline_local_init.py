"""
Multi-view depth fusion via TSDF → single consistent point cloud.

Replaces the naive per-view concatenation in `depth_maps_to_pointcloud`
with a proper fusion that:
  1. Removes per-view depth inconsistencies
  2. Produces a single canonical UV parameterization
  3. Returns per-point observation counts for downstream weighting

Requires: open3d (already a dependency — used in spline_render.py)
"""

import numpy as np
import torch
import warnings
from typing import List, Tuple, Optional, Dict

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def fused_depth_to_pointcloud(
    cameras: list,
    depth_maps: list,
    voxel_size: float = 0.005,
    depth_trunc: float = 5.0,
    sdf_trunc_factor: float = 4.0,
    max_depth_quantile: float = 0.95,
    grazing_angle_deg: float = 80.0,
    downsample_voxel: Optional[float] = None,
    color_images: Optional[list] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Fuse multi-view depth maps into a single consistent point cloud
    via TSDF integration, then compute a global UV parameterization.

    This eliminates the fundamental problem with `depth_maps_to_pointcloud`:
    each view contributes its own pixels with its own image-plane UV,
    producing overlapping points with inconsistent parameterizations.
    TSDF fusion merges redundant observations and outputs one canonical
    point per surface location.

    Parameters
    ----------
    cameras : list of Camera (scene.cameras.Camera)
        Training cameras with .R, .T, .Fx, .Fy, .Cx, .Cy,
        .image_width, .image_height, .FoVx, .FoVy attributes.
    depth_maps : list of torch.Tensor or np.ndarray
        Per-view depth maps, each (H, W) in world meters.
        Must be same length as `cameras`.
    voxel_size : float
        TSDF voxel size in meters. Controls output point density.
        Smaller = denser but slower. 0.005 is good for room-scale.
    depth_trunc : float
        Maximum depth to integrate (meters). Points beyond this are sky/noise.
    sdf_trunc_factor : float
        SDF truncation distance = sdf_trunc_factor * voxel_size.
        4.0 is standard for room-scale scenes.
    max_depth_quantile : float
        Per-view adaptive depth truncation: ignore depths beyond this
        quantile of each view's depth distribution. Prevents integrating
        SfM failures at extreme depth.
    grazing_angle_deg : float
        Filter depth pixels where the view ray hits the surface at
        a grazing angle (> this many degrees from the normal).
        These pixels have high depth uncertainty.
    downsample_voxel : float or None
        If set, further downsample the output cloud to this voxel size.
        Useful for very large scenes.
    color_images : list of torch.Tensor or None
        Per-view RGB images, each (3, H, W) float in [0, 1].
        If None, output colors will be uniform grey.
    device : str
        Output tensor device.

    Returns
    -------
    dict with keys:
        "xyz"    : (N, 3) torch.Tensor — world-space point positions
        "colors" : (N, 3) torch.Tensor — RGB in [0, 1]
        "normals": (N, 3) torch.Tensor — estimated surface normals
        "uv"     : (N, 2) torch.Tensor — global UV parameterization in [0, 1]²
        "obs_count" : (N,) torch.Tensor — per-point observation count
    """
    if not HAS_O3D:
        raise ImportError(
            "open3d is required for TSDF fusion. "
            "Install with: pip install open3d"
        )

    n_views = len(cameras)
    assert len(depth_maps) == n_views, \
        f"Got {n_views} cameras but {len(depth_maps)} depth maps"

    # ------------------------------------------------------------------
    # 1. Build TSDF volume
    # ------------------------------------------------------------------
    sdf_trunc = sdf_trunc_factor * voxel_size
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # ------------------------------------------------------------------
    # 2. Integrate each view
    # ------------------------------------------------------------------
    integrated_count = 0
    for i in range(n_views):
        cam = cameras[i]
        depth = depth_maps[i]

        # --- Convert depth to numpy (H, W) float32 ---
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            depth_np = np.asarray(depth, dtype=np.float32).squeeze()

        H, W = depth_np.shape

        # --- Per-view adaptive depth clipping ---
        valid_depth = depth_np[depth_np > 0]
        if len(valid_depth) < 100:
            warnings.warn(f"[TSDF] View {i} has <100 valid depth pixels, skipping")
            continue

        view_depth_max = min(
            depth_trunc,
            float(np.quantile(valid_depth, max_depth_quantile)),
        )

        # Zero out invalid depths
        depth_clean = depth_np.copy()
        depth_clean[depth_clean <= 0] = 0
        depth_clean[depth_clean > view_depth_max] = 0

        # --- Optional: grazing angle filter ---
        if grazing_angle_deg < 90.0:
            depth_clean = _filter_grazing_angles(
                depth_clean, cam, grazing_angle_deg
            )

        # --- Build Open3D intrinsic ---
        fx = float(cam.Fx)
        fy = float(cam.Fy)
        cx = float(cam.Cx)
        cy = float(cam.Cy)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # --- Build W2C extrinsic (4x4) ---
        #  cam.R is stored such that R^T is the W2C rotation
        #  cam.T is the translation in the W2C [R^T | T] matrix
        #  This matches your spline_render.py convention exactly
        R_np = np.asarray(cam.R, dtype=np.float64)
        T_np = np.asarray(cam.T, dtype=np.float64)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R_np.T  # W2C rotation
        extrinsic[:3, 3] = T_np     # W2C translation

        # --- Build RGBD image ---
        depth_o3d = o3d.geometry.Image(
            (depth_clean * 1000.0).astype(np.uint16)
        )

        if color_images is not None and i < len(color_images):
            color_img = color_images[i]
            if isinstance(color_img, torch.Tensor):
                # (3, H, W) → (H, W, 3) uint8
                color_np = (
                    color_img.detach().cpu().permute(1, 2, 0)
                    .clamp(0, 1).mul(255).byte().numpy()
                )
            else:
                color_np = (np.asarray(color_img) * 255).astype(np.uint8)
            # Resize if needed
            if color_np.shape[0] != H or color_np.shape[1] != W:
                import cv2
                color_np = cv2.resize(color_np, (W, H))
            color_o3d = o3d.geometry.Image(color_np)
        else:
            # No color → uniform grey
            color_np = np.full((H, W, 3), 128, dtype=np.uint8)
            color_o3d = o3d.geometry.Image(color_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=view_depth_max,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated_count += 1

    if integrated_count == 0:
        raise RuntimeError("[TSDF] No views were integrated — check depth maps")

    print(f"[TSDF] Integrated {integrated_count}/{n_views} views")

    # ------------------------------------------------------------------
    # 3. Extract point cloud from TSDF
    # ------------------------------------------------------------------
    pcd = volume.extract_point_cloud()

    if len(pcd.points) == 0:
        raise RuntimeError(
            "[TSDF] Extracted 0 points — try increasing depth_trunc "
            "or decreasing voxel_size"
        )

    print(f"[TSDF] Extracted {len(pcd.points)} points from volume")

    # Optional further downsampling
    if downsample_voxel is not None and downsample_voxel > voxel_size:
        pcd = pcd.voxel_down_sample(downsample_voxel)
        print(f"[TSDF] Downsampled to {len(pcd.points)} points "
              f"(voxel={downsample_voxel:.4f})")

    # Estimate normals if not already present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )

    # Orient normals toward cameras (use the first camera center)
    cam0_center = _get_camera_center(cameras[0])
    pcd.orient_normals_towards_camera_location(cam0_center)

    # ------------------------------------------------------------------
    # 4. Compute per-point observation counts
    # ------------------------------------------------------------------
    points_np = np.asarray(pcd.points, dtype=np.float64)
    obs_count = _compute_observation_counts(
        points_np, cameras, depth_maps, depth_trunc,
        depth_tolerance=3.0 * voxel_size,
    )

    # ------------------------------------------------------------------
    # 5. Global UV parameterization via PCA
    #    (conformal_parameterize can be swapped in here later)
    # ------------------------------------------------------------------
    uv_np = _compute_global_uv(points_np)

    # ------------------------------------------------------------------
    # 6. Pack into tensors
    # ------------------------------------------------------------------
    xyz = torch.from_numpy(points_np.astype(np.float32)).to(device)
    colors = torch.from_numpy(
        np.asarray(pcd.colors, dtype=np.float32)
    ).to(device)
    normals = torch.from_numpy(
        np.asarray(pcd.normals, dtype=np.float32)
    ).to(device)
    uv = torch.from_numpy(uv_np.astype(np.float32)).to(device)
    obs = torch.from_numpy(obs_count.astype(np.float32)).to(device)

    print(f"[TSDF] Final point cloud: {len(xyz)} pts, "
          f"mean obs={obs.mean():.1f}, "
          f"UV range=[{uv.min():.3f}, {uv.max():.3f}]")

    return {
        "xyz": xyz,
        "colors": colors,
        "normals": normals,
        "uv": uv,
        "obs_count": obs,
    }


# ======================================================================
#  Internal helpers
# ======================================================================

def _get_camera_center(cam) -> np.ndarray:
    """
    Extract camera center in world coordinates.

    Convention: W2C is [R^T | T], so C2W is its inverse,
    and camera center = -R @ T  (since T = -R^T @ center).
    """
    R = np.asarray(cam.R, dtype=np.float64)   # R such that R^T is W2C rot
    T = np.asarray(cam.T, dtype=np.float64)   # T in W2C
    # W2C: [R^T | T] → camera_center = -(R^T)^{-1} @ T = -R @ T
    center = -R @ T
    return center


def _filter_grazing_angles(
    depth: np.ndarray,
    cam,
    max_angle_deg: float,
) -> np.ndarray:
    """
    Zero out depth pixels where the view ray is nearly tangent to the
    surface (estimated from local depth gradients).

    Grazing-angle pixels have very high depth uncertainty and produce
    noisy surface estimates. Filtering them before TSDF integration
    avoids phantom surfaces.
    """
    H, W = depth.shape

    # Compute local surface normals from depth via finite differences
    # These are approximate but sufficient for filtering
    fx = float(cam.Fx)
    fy = float(cam.Fy)
    cx = float(cam.Cx)
    cy = float(cam.Cy)

    # Pixel grid
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera space
    z = depth.copy()
    z[z <= 0] = np.nan
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy

    # Finite-difference normals in camera space
    # dP/du and dP/dv, then cross product
    dxdu = np.gradient(x, axis=1)
    dxdv = np.gradient(x, axis=0)
    dydu = np.gradient(y, axis=1)
    dydv = np.gradient(y, axis=0)
    dzdu = np.gradient(z, axis=1)
    dzdv = np.gradient(z, axis=0)

    # Normal = (dP/du) × (dP/dv)
    nx = dydu * dzdv - dzdu * dydv
    ny = dzdu * dxdv - dxdu * dzdv
    nz = dxdu * dydv - dydu * dxdv

    norm_len = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
    nx /= norm_len
    ny /= norm_len
    nz /= norm_len

    # View direction for each pixel (camera space: ray = normalize([x, y, z]))
    ray_len = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    vx = x / ray_len
    vy = y / ray_len
    vz = z / ray_len

    # Angle between normal and view ray
    cos_angle = np.abs(nx * vx + ny * vy + nz * vz)
    cos_threshold = np.cos(np.radians(max_angle_deg))

    # Zero out grazing-angle pixels
    depth_out = depth.copy()
    grazing_mask = cos_angle < cos_threshold
    grazing_mask |= np.isnan(cos_angle)
    depth_out[grazing_mask] = 0

    n_filtered = int(grazing_mask.sum())
    n_valid = int((depth > 0).sum())
    if n_filtered > 0.5 * n_valid:
        # If we'd filter >50% of pixels, the threshold is too aggressive
        # — likely a forward-facing scene. Skip filtering for this view.
        return depth

    return depth_out


def _compute_observation_counts(
    points: np.ndarray,
    cameras: list,
    depth_maps: list,
    depth_trunc: float,
    depth_tolerance: float = 0.015,
) -> np.ndarray:
    """
    For each fused 3D point, count how many views observe it consistently.

    A point is "observed" by a view if:
      1. It projects inside the image
      2. It is in front of the camera
      3. Its depth matches the view's depth map within tolerance

    This gives a per-point confidence weight: points seen from many
    views are reliable; points from a single view may be noise.

    Parameters
    ----------
    points : (N, 3) world-space positions
    cameras : list of Camera
    depth_maps : list of depth tensors/arrays
    depth_trunc : max depth
    depth_tolerance : absolute depth difference tolerance (meters)

    Returns
    -------
    obs_count : (N,) float, per-point observation count (>= 1)
    """
    N = len(points)
    obs_count = np.zeros(N, dtype=np.float32)

    for i, cam in enumerate(cameras):
        depth = depth_maps[i]
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            depth_np = np.asarray(depth, dtype=np.float32).squeeze()

        H, W = depth_np.shape
        fx = float(cam.Fx)
        fy = float(cam.Fy)
        cx = float(cam.Cx)
        cy = float(cam.Cy)

        # Build W2C transform
        R = np.asarray(cam.R, dtype=np.float64)
        T = np.asarray(cam.T, dtype=np.float64)
        # W2C: x_cam = R^T @ x_world + T
        pts_cam = (R.T @ points.T).T + T[None, :]

        z = pts_cam[:, 2]
        valid_z = z > 0.01

        # Project to pixel
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy

        # Bounds check
        u_int = np.round(u).astype(np.int64)
        v_int = np.round(v).astype(np.int64)
        in_bounds = (
            valid_z &
            (u_int >= 0) & (u_int < W) &
            (v_int >= 0) & (v_int < H)
        )

        # Depth consistency check
        indices = np.where(in_bounds)[0]
        if len(indices) == 0:
            continue

        u_valid = u_int[indices]
        v_valid = v_int[indices]
        z_valid = z[indices]

        depth_at_proj = depth_np[v_valid, u_valid]

        # A point is consistently observed if:
        # - The depth map has valid depth at this pixel
        # - The point's depth matches within tolerance (relative)
        has_depth = depth_at_proj > 0
        depth_diff = np.abs(depth_at_proj - z_valid)
        # Use relative + absolute tolerance: max(absolute, relative * depth)
        tol = np.maximum(depth_tolerance, 0.02 * z_valid)
        consistent = has_depth & (depth_diff < tol)

        obs_count[indices[consistent]] += 1.0

    # Ensure minimum count of 1 (every point was seen by at least the
    # view that created it in the TSDF)
    obs_count = np.maximum(obs_count, 1.0)

    return obs_count


def _compute_global_uv(points: np.ndarray) -> np.ndarray:
    """
    Compute a global UV parameterization for the fused point cloud.

    Uses PCA projection as the default. This is the plug-in point for
    upgrading to conformal parameterization (Fix 4) later.

    Returns
    -------
    uv : (N, 2) in [0.001, 0.999]
    """
    from sklearn.decomposition import PCA

    N = len(points)
    if N < 3:
        return np.full((N, 2), 0.5)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(points - points.mean(axis=0))

    uv = proj - proj.min(axis=0)
    rng = uv.max(axis=0)
    rng[rng < 1e-10] = 1.0
    uv = uv / rng

    return np.clip(uv, 0.001, 0.999)