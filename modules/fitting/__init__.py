"""
NURBS surface fitting from point cloud data.

This module provides robust fitting of NURBS surfaces to 3D point clouds,
suitable for initializing SplineModel.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from sklearn.decomposition import PCA


@dataclass
class NURBSSurface:
    """Container for NURBS surface data compatible with SplineModel initialization."""
    control_points: np.ndarray  # [H, W, 3] control point grid
    control_colors: np.ndarray  # [H, W, 3] RGB colors (0-1 range)
    weights: Optional[np.ndarray] = None  # [H, W] weights for rational surfaces
    knots_u: Optional[np.ndarray] = None
    knots_v: Optional[np.ndarray] = None
    degree: int = 3

    def to_torch(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Convert to torch tensors for SplineModel."""
        result = {
            'control_points': torch.tensor(self.control_points, dtype=torch.float32, device=device),
            'control_colors': torch.tensor(self.control_colors, dtype=torch.float32, device=device),
        }
        if self.weights is not None:
            result['weights'] = torch.tensor(self.weights, dtype=torch.float32, device=device)
        if self.knots_u is not None:
            result['knots_u'] = torch.tensor(self.knots_u, dtype=torch.float32, device=device)
        if self.knots_v is not None:
            result['knots_v'] = torch.tensor(self.knots_v, dtype=torch.float32, device=device)
        return result


class PointCloudToNURBS:
    """
    Fits a NURBS surface to a 3D point cloud.

    Pipeline:
    1. PCA-based orientation alignment
    2. Projection to approximate UV coordinates
    3. Grid-based interpolation
    4. B-spline surface fitting
    5. Optional refinement
    """

    def __init__(
            self,
            target_resolution: Tuple[int, int] = (32, 32),
            degree: int = 3,
            smoothing: float = 0.1,
            use_weights: bool = False,
            outlier_threshold: float = 2.5
    ):
        """
        Args:
            target_resolution: (H, W) control point grid resolution
            degree:  NURBS polynomial degree
            smoothing: Smoothing factor for fitting (0 = interpolation, >0 = approximation)
            use_weights:  If True, create rational NURBS
            outlier_threshold: Number of std devs for outlier rejection
        """
        self.target_H, self.target_W = target_resolution
        self.degree = degree
        self.smoothing = smoothing
        self.use_weights = use_weights
        self.outlier_threshold = outlier_threshold

    def fit(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            normals: Optional[np.ndarray] = None
    ) -> NURBSSurface:
        """
        Fit NURBS surface to point cloud.

        Args:
            points: [N, 3] point positions
            colors: [N, 3] RGB colors (0-255 or 0-1)
            normals: [N, 3] surface normals (optional, for orientation)

        Returns:
            NURBSSurface object
        """
        points = np.asarray(points, dtype=np.float64)
        N = len(points)

        if colors is None:
            colors = np.ones((N, 3)) * 0.5
        colors = np.asarray(colors, dtype=np.float64)

        if colors.max() > 1.0:
            colors = colors / 255.0

        print(f"[NURBS Fitting] Input:  {N} points")

        # Step 1: Remove outliers
        points_clean, colors_clean, inlier_mask = self._remove_outliers(points, colors)
        print(f"[NURBS Fitting] After outlier removal: {len(points_clean)} points")

        # Step 2: PCA-based alignment and UV parameterization
        uv_coords, aligned_points, transform_info = self._compute_uv_parameterization(
            points_clean, normals[inlier_mask] if normals is not None else None
        )

        # Step 3: Create regular grid and interpolate
        control_points, control_colors = self._grid_interpolation(
            uv_coords, aligned_points, colors_clean
        )

        # Step 4: Fit B-spline surface
        control_points_smooth = self._fit_bspline(control_points)

        # Step 5: Transform back to original coordinate system
        control_points_final = self._inverse_transform(control_points_smooth, transform_info)

        # Step 6: Create knot vectors
        knots_u = self._create_knot_vector(self.target_H, self.degree)
        knots_v = self._create_knot_vector(self.target_W, self.degree)

        # Step 7: Compute weights if needed
        weights = None
        if self.use_weights:
            weights = self._compute_weights(control_points_final, points_clean, uv_coords)

        surface = NURBSSurface(
            control_points=control_points_final.astype(np.float32),
            control_colors=control_colors.astype(np.float32),
            weights=weights,
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree=self.degree
        )

        print(f"[NURBS Fitting] Output: {self.target_H}x{self.target_W} control grid")

        return surface

    def _remove_outliers(
            self,
            points: np.ndarray,
            colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove statistical outliers using local density estimation."""
        # Use KD-tree for efficient neighbor queries
        tree = KDTree(points)

        # Compute average distance to k nearest neighbors
        k = min(20, len(points) // 10)
        distances, _ = tree.query(points, k=k)
        avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self

        # Statistical outlier removal
        mean_dist = avg_distances.mean()
        std_dist = avg_distances.std()
        threshold = mean_dist + self.outlier_threshold * std_dist

        inlier_mask = avg_distances < threshold

        return points[inlier_mask], colors[inlier_mask], inlier_mask

    def _compute_uv_parameterization(
            self,
            points: np.ndarray,
            normals: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Compute UV parameterization using PCA-based projection.

        Returns:
            uv_coords:  [N, 2] UV coordinates in [0, 1]
            aligned_points: [N, 3] points in aligned coordinate system
            transform_info:  dict with transformation parameters
        """
        # Center points
        centroid = points.mean(axis=0)
        centered = points - centroid

        # PCA to find principal directions
        pca = PCA(n_components=3)
        pca.fit(centered)

        # Transform to PCA space
        aligned = pca.transform(centered)

        # Use first two principal components as UV
        # (assumes surface is roughly planar or can be unfolded)
        uv_raw = aligned[:, :2]

        # Normalize to [0, 1]
        uv_min = uv_raw.min(axis=0)
        uv_max = uv_raw.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range < 1e-10] = 1.

        uv_coords = (uv_raw - uv_min) / uv_range

        # Clamp to valid range
        uv_coords = np.clip(uv_coords, 0.001, 0.999)

        transform_info = {
            'centroid': centroid,
            'pca_components': pca.components_,
            'pca_mean': pca.mean_
        }

        return uv_coords, aligned, transform_info

    def _grid_interpolation(
            self,
            uv_coords: np.ndarray,
            points: np.ndarray,
            colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate scattered points to regular grid.
        """
        H, W = self.target_H, self.target_W

        # Create regular grid
        u_grid = np.linspace(0, 1, H)
        v_grid = np.linspace(0, 1, W)
        uu, vv = np.meshgrid(u_grid, v_grid, indexing='ij')
        grid_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

        # Interpolate XYZ
        control_points = np.zeros((H * W, 3))
        for dim in range(3):
            control_points[:, dim] = griddata(
                uv_coords, points[:, dim], grid_uv,
                method='linear', fill_value=points[:, dim].mean()
            )

        # Fill any NaN values with nearest neighbor
        nan_mask = np.isnan(control_points).any(axis=1)
        if nan_mask.any():
            for dim in range(3):
                control_points[nan_mask, dim] = griddata(
                    uv_coords, points[:, dim], grid_uv[nan_mask],
                    method='nearest'
                )

        # Interpolate colors
        control_colors = np.zeros((H * W, 3))
        for dim in range(3):
            control_colors[:, dim] = griddata(
                uv_coords, colors[:, dim], grid_uv,
                method='linear', fill_value=colors[:, dim].mean()
            )

        # Fill NaN colors
        nan_mask = np.isnan(control_colors).any(axis=1)
        if nan_mask.any():
            for dim in range(3):
                control_colors[nan_mask, dim] = griddata(
                    uv_coords, colors[:, dim], grid_uv[nan_mask],
                    method='nearest'
                )

        # Clip colors to valid range
        control_colors = np.clip(control_colors, 0, 1)

        return control_points.reshape(H, W, 3), control_colors.reshape(H, W, 3)

    def _fit_bspline(self, control_points: np.ndarray) -> np.ndarray:
        """
        Apply B-spline smoothing to control points.
        Uses separable 1D smoothing in U and V directions.
        """
        from scipy.ndimage import gaussian_filter1d

        if self.smoothing <= 0:
            return control_points

        H, W, _ = control_points.shape
        smoothed = control_points.copy()

        # Smooth along U direction
        sigma = self.smoothing * H / 10
        for dim in range(3):
            for w in range(W):
                smoothed[:, w, dim] = gaussian_filter1d(
                    control_points[:, w, dim], sigma, mode='reflect'
                )

        # Smooth along V direction
        sigma = self.smoothing * W / 10
        temp = smoothed.copy()
        for dim in range(3):
            for h in range(H):
                smoothed[h, :, dim] = gaussian_filter1d(
                    temp[h, :, dim], sigma, mode='reflect'
                )

        return smoothed

    def _inverse_transform(
            self,
            aligned_points: np.ndarray,
            transform_info: Dict
    ) -> np.ndarray:
        """Transform points back to original coordinate system."""
        H, W, _ = aligned_points.shape
        points_flat = aligned_points.reshape(-1, 3)

        # Inverse PCA transform
        # aligned = (original - centroid) @ components. T
        # original = aligned @ components + centroid

        original = points_flat @ transform_info['pca_components'] + transform_info['centroid']

        return original.reshape(H, W, 3)

    def _create_knot_vector(self, n_ctrl: int, degree: int) -> np.ndarray:
        """Create clamped uniform knot vector."""
        n_knots = n_ctrl + degree + 1
        n_internal = n_knots - 2 * (degree + 1)

        knots = np.concatenate([
            np.zeros(degree + 1),
            np.linspace(0, 1, n_internal + 2)[1:-1] if n_internal > 0 else np.array([]),
            np.ones(degree + 1)
        ])

        return knots

    def _compute_weights(
            self,
            control_points: np.ndarray,
            original_points: np.ndarray,
            uv_coords: np.ndarray
    ) -> np.ndarray:
        """
        Compute NURBS weights based on local point density.
        Higher density regions get higher weights.
        """
        H, W, _ = control_points.shape

        # Create grid for weight computation
        u_grid = np.linspace(0, 1, H)
        v_grid = np.linspace(0, 1, W)

        weights = np.ones((H, W))

        # Build KD-tree of UV coordinates
        tree = KDTree(uv_coords)

        for i in range(H):
            for j in range(W):
                # Query nearby points
                query_point = np.array([[u_grid[i], v_grid[j]]])
                radius = 2.0 / max(H, W)  # Adaptive radius
                indices = tree.query_ball_point(query_point, radius)[0]

                # Weight based on local density
                density = len(indices) / (np.pi * radius ** 2 + 1e-10)
                weights[i, j] = 1.0 + 0.5 * np.tanh(density / 100 - 1)

        # Normalize weights
        weights = weights / weights.mean()

        return weights.astype(np.float32)


class AdvancedNURBSFitter:
    """
    Advanced NURBS fitting with iterative refinement.
    Uses scipy's NURBS implementation for accurate fitting.
    """

    def __init__(
            self,
            initial_resolution: Tuple[int, int] = (8, 8),
            max_resolution: Tuple[int, int] = (256, 256),
            degree: int = 3,
            max_iterations: int = 100,
            error_threshold: float = 0.0002,
            adaptive_refinement: bool = True
    ):
        initial_resolution = max(8, initial_resolution[0]), max(8, initial_resolution[1])
        max_resolution = min(256, max_resolution[0]), min(256, max_resolution[1])
        self.initial_resolution = initial_resolution
        self.max_resolution = max_resolution
        self.degree = degree
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold
        self.adaptive_refinement = adaptive_refinement

    def fit(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            verbose: bool = True
    ) -> NURBSSurface:
        """
        Iteratively fit NURBS surface with adaptive refinement.
        """
        # Start with coarse fit
        current_res = self.initial_resolution

        fitter = PointCloudToNURBS(
            target_resolution=current_res,
            degree=self.degree,
            smoothing=0.1
        )
        surface = fitter.fit(points, colors)

        for iteration in range(self.max_iterations):
            # Evaluate fitting error
            error = self._compute_fitting_error(surface, points)

            if verbose:
                print(f"[NURBS Fit] Iteration {iteration + 1}:  "
                      f"resolution={current_res}, error={error:.6f}")

            if error < self.error_threshold:
                print("[NURBS Fit] Converged.")
                print(f"[NURBS Fit] Final resolution: {current_res}, error: {error:.6f}")
                break

            # Check if we can refine further
            new_res = (
                min(current_res[0] * 2, self.max_resolution[0]),
                min(current_res[1] * 2, self.max_resolution[1])
            )

            if new_res == current_res:
                break

            current_res = new_res

            # Refit with higher resolution
            fitter = PointCloudToNURBS(
                target_resolution=current_res,
                degree=self.degree,
                smoothing=max(0.05, 0.2 - iteration * 0.05)  # Reduce smoothing
            )
            surface = fitter.fit(points, colors)

        return surface

    def _compute_fitting_error(
            self,
            surface: NURBSSurface,
            points: np.ndarray
    ) -> float:
        """Compute average distance from points to surface."""
        # Sample surface densely
        H, W, _ = surface.control_points.shape

        # Simple:  use control points as proxy for surface
        # For better accuracy, evaluate actual B-spline surface
        surface_points = surface.control_points.reshape(-1, 3)

        # Build KD-tree
        tree = KDTree(surface_points)

        # Query distances
        distances, _ = tree.query(points)

        return float(distances.mean())


def create_nurbs_from_pointcloud_basic(
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        resolution: Tuple[int, int] = (32, 32),
        degree: int = 3,
        smoothing: float = 0.01,
        use_advanced: bool = False
) -> NURBSSurface:
    """
    Convenience function to create NURBS surface from point cloud.

    Args:
        points: [N, 3] point positions
        colors: [N, 3] RGB colors
        resolution: (H, W) control grid resolution
        degree: NURBS degree
        smoothing:  Smoothing factor
        use_advanced: Use iterative refinement

    Returns:
        NURBSSurface object
    """
    if use_advanced:
        fitter = AdvancedNURBSFitter(
            initial_resolution=(resolution[0] // 16, resolution[1] // 16),
            max_resolution=resolution,
            degree=degree
        )
    else:
        fitter = PointCloudToNURBS(
            target_resolution=resolution,
            degree=degree,
            smoothing=smoothing
        )

    return fitter.fit(points, colors)


# Integration with SplineModel
def initialize_spline_model_from_pointcloud(
        points: np.ndarray,
        colors: np.ndarray,
        config,
        args,
        resolution: Tuple[int, int] = (32, 32),
        **kwargs
) -> Dict[str, Any]:
    """
    Create SplineModel initialization data from point cloud.

    Returns a dictionary that can be passed to SplineModel constructor.
    """
    # Fit NURBS surface
    surface = create_nurbs_from_pointcloud(
        points, colors, resolution,
        degree=config.spline_degree[0],
        use_advanced=True,
        smoothing=0.01
    )

    # Convert to SplineModel-compatible format
    torch_data = surface.to_torch()

    class MockSurface:
        def __init__(self, ctrl_pts, knots_u, knots_v, degree):
            self.ctrlpts2d = ctrl_pts.cpu().numpy().tolist()
            self.knotvector_u = knots_u.cpu().numpy().tolist()
            self.knotvector_v = knots_v.cpu().numpy().tolist()
            self.degree_u = degree
            self.degree_v = degree

    mock_surf = MockSurface(
        torch_data['control_points'],
        torch_data['knots_u'],
        torch_data['knots_v'],
        surface.degree
    )

    mock_surf_rgb = MockSurface(
        torch_data['control_colors'],
        torch_data['knots_u'],
        torch_data['knots_v'],
        surface.degree
    )

    return {
        'surf': [mock_surf],
        'surf_rgb': [mock_surf_rgb],
        'config': config,
        'args': args,
        **kwargs
    }


