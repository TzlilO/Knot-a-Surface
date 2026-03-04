
from dataclasses import dataclass, field
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import cKDTree


@dataclass
class CameraObject:
    """Minimal camera representation for preprocessing.

    Extracted from the project's Camera class (cameras.py) to avoid
    importing the full nn.Module during preprocessing.
    """
    R: np.ndarray  # [3, 3] rotation (world-to-camera)
    T: np.ndarray  # [3,] translation (world-to-camera)
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    uid: int = 0

    @property
    def position(self) -> np.ndarray:
        """Camera center in world coordinates."""
        # C = -R^T @ T
        return -self.R.T @ self.T

    @property
    def forward_dir(self) -> np.ndarray:
        """Camera forward direction in world coordinates (negative Z)."""
        c2w = np.eye(4)
        c2w[:3, :3] = self.R.T
        c2w[:3, 3] = self.position
        forward = -c2w[:3, 2]
        return forward / (np.linalg.norm(forward) + 1e-8)

    @classmethod
    def from_camera(cls, cam) -> "CameraObject":
        """Extract from the project's Camera nn.Module."""
        R = cam.R if isinstance(cam.R, np.ndarray) else cam.R.cpu().numpy()
        T = cam.T if isinstance(cam.T, np.ndarray) else cam.T.cpu().numpy()
        return cls(
            R=R, T=T,
            FoVx=cam.FoVx, FoVy=cam.FoVy,
            image_width=cam.image_width,
            image_height=cam.image_height,
            uid=cam.uid,
        )


def compute_observation_weights(
        points: np.ndarray,
        cameras: List["CameraObject"],
        normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-point observation quality weights from training cameras.

    A point's weight is high when it is:
    1. Visible from many cameras (observation count)
    2. Front-facing w.r.t. those cameras (normal-view dot product)
    3. Close to image center (not at periphery)

    Args:
        points: [N, 3] world-space points
        cameras: List of CameraObject
        normals: [N, 3] optional surface normals (estimated if None)

    Returns:
        weights: [N] quality weights in [0, 1]
    """
    n_points = len(points)
    if not cameras:
        return np.ones(n_points, dtype=np.float64)

    observation_count = np.zeros(n_points, dtype=np.float64)
    facing_score = np.zeros(n_points, dtype=np.float64)

    for cam in cameras:
        cam_pos = cam.position
        view_dirs = cam_pos - points  # [N, 3] pointing toward camera
        view_dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs_norm = view_dirs / (view_dists + 1e-8)

        # Front-facing check
        if normals is not None:
            dots = np.sum(normals * view_dirs_norm, axis=1)
            front_facing = dots > 0.0
        else:
            front_facing = np.ones(n_points, dtype=bool)

        # Project points into camera image to check visibility
        # Transform to camera space: p_cam = R @ p_world + T
        p_cam = (cam.R @ points.T).T + cam.T  # [N, 3]
        in_front = p_cam[:, 2] > 0.01  # Must be in front of camera

        # Project to pixel coordinates
        fx = cam.image_width / (2.0 * np.tan(cam.FoVx / 2.0))
        fy = cam.image_height / (2.0 * np.tan(cam.FoVy / 2.0))
        cx, cy = cam.image_width / 2.0, cam.image_height / 2.0

        z = np.maximum(p_cam[:, 2], 1e-6)
        u = fx * p_cam[:, 0] / z + cx
        v = fy * p_cam[:, 1] / z + cy

        # Check if within image bounds (with margin)
        margin = 0.05
        in_image = (
                (u > -margin * cam.image_width)
                & (u < (1 + margin) * cam.image_width)
                & (v > -margin * cam.image_height)
                & (v < (1 + margin) * cam.image_height)
        )

        # Visible = in front + in image + front-facing
        visible = in_front & in_image & front_facing
        observation_count[visible] += 1.0

        if normals is not None:
            facing_score[visible] += np.clip(dots[visible], 0, 1)

    # Normalize
    max_obs = max(observation_count.max(), 1.0)
    obs_weight = observation_count / max_obs

    if normals is not None and observation_count.max() > 0:
        safe_count = np.maximum(observation_count, 1.0)
        avg_facing = facing_score / safe_count
        weight = 0.7 * obs_weight + 0.3 * avg_facing
    else:
        weight = obs_weight

    # Clamp to [0, 1] with minimum floor for unobserved points
    weight = np.clip(weight, 0.05, 1.0)
    return weight


def compute_camera_consistent_normals(
        points: np.ndarray,
        cameras: List["CameraObject"],
        k: int = 30,
) -> np.ndarray:
    """
    Estimate normals oriented consistently using camera positions.

    Unlike PointCloudProcessor.estimate_normals which orients toward
    scene center (wrong for concave regions), this uses actual camera
    positions as the "outside" reference.

    Args:
        points: [N, 3]
        cameras: List of CameraObject
        k: Neighborhood size for PCA normal estimation

    Returns:
        normals: [N, 3] consistently oriented surface normals
    """
    n_points = len(points)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(k, n_points - 1))

    normals = np.zeros_like(points)
    for i in range(n_points):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            normals[i] = Vt[-1]
        except np.linalg.LinAlgError:
            normals[i] = np.array([0, 0, 1])

    # Orient using camera positions (majority vote)
    if cameras:
        cam_positions = np.array([c.position for c in cameras])  # [K, 3]
        # For each point, compute average view direction from all cameras
        # [N, 1, 3] - [1, K, 3] -> [N, K, 3]
        view_dirs = cam_positions[None, :, :] - points[:, None, :]
        # Average view direction weighted by inverse distance
        view_dists = np.linalg.norm(view_dirs, axis=2, keepdims=True)
        view_dirs_norm = view_dirs / (view_dists + 1e-8)
        avg_view_dir = view_dirs_norm.mean(axis=1)  # [N, 3]

        # Flip normals that point away from cameras
        dots = np.sum(normals * avg_view_dir, axis=1)
        flip_mask = dots < 0
        normals[flip_mask] *= -1
    else:
        # Fallback: orient toward centroid
        center = points.mean(axis=0)
        center_vecs = points - center
        flip_mask = np.sum(normals * center_vecs, axis=1) < 0
        normals[flip_mask] *= -1

    return normals


# =============================================================================
# BSpline Post-Fit Optimizer (Chamfer Refinement)
# =============================================================================

# =============================================================================
# Camera-Aware Utilities
# =============================================================================


# import numpy as np
# import torch
# from typing import Tuple, List, Optional, Dict, Union, Any
# from dataclasses import dataclass, field
# from enum import Enum
# from scipy.spatial import cKDTree
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import  shortest_path
# from scipy.ndimage import gaussian_filter, binary_dilation
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.neighbors import kneighbors_graph
# from sklearn.preprocessing import StandardScaler
#
# # NURBS library
# from geomdl import BSpline
#
#
# import numpy as np
# from scipy.spatial import cKDTree
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components
# from sklearn.cluster import AgglomerativeClustering
# from typing import List, Tuple, Optional
# import warnings
#
#
#
# # =============================================================================
# # Enums and Configuration
# # =============================================================================
#
#
# class DecompositionMode(Enum):
#     SINGLE = "single"
#     BACKGROUND_OBJECT = "background"
#     K_COMPONENTS = "k_components"
#     ADAPTIVE = "adaptive"  # NEW: Automatically determine K
#
#
# @dataclass
# class SurfaceConfig:
#     """Configuration for NURBS surface fitting."""
#
#     # --- Grid Resolution ---
#     adaptive_resolution: bool = True
#     base_resolution: int = 256
#     min_resolution: int = 32
#     max_resolution: int = 512
#     resolution_complexity_sensitivity: float = 1.0
#
#     # Fallback fixed resolution
#     resolution_u: int = 32
#     resolution_v: int = 32
#
#     # --- NURBS Parameters ---
#     degree_u: int = 3
#     degree_v: int = 3
#     smoothing: float = 0.05
#
#     # --- Decomposition Parameters ---
#     decomposition_mode: DecompositionMode = DecompositionMode.K_COMPONENTS
#     n_components: int = 4
#
#     # --- Feature Weights for Segmentation ---
#     weight_spatial: float = 1
#     weight_normal: float = 0.
#     weight_color: float = 1.
#     weight_complexity: float = 0.
#
#     # --- Connectivity Graph ---
#     connectivity_k: int = 4
#     min_component_size: int = 1
#     use_soft_edges: bool = False
#     edge_color_adaptive: bool = False
#
#     # --- Background/Object (Legacy) ---
#     bg_detection_method: str = "hybrid"
#     bg_distance_threshold: float = 0.05
#     bg_min_size_ratio: float = 0.5
#     bg_max_size_ratio: float = 0.8
#
#     # --- UV Parameterization ---
#     use_geodesic_uv: bool = True
#     parameterization: bool = 'spherical'
#     generate_adaptive_samples: bool = True  # NEW parameter
#     sampling_resolution_factor: float =1.0  # NEW parameter
#
#     # --- Decomposition Parameters ---
#     decomposition_mode: DecompositionMode = DecompositionMode.K_COMPONENTS
#     n_components: int = 4
#     base_resolution: int = 64
#     # --- Connectivity Constraints (NEW) ---
#     max_normal_angle_diff: float = 60.0  # Max angle (degrees) between normals for edge validity
#     # --- Quality Parameters ---
#     outlier_removal: bool = False
#     outlier_std_ratio: float = 2.0
#     normal_estimation_k: int = 30
#
#     bg_resolution: Optional[Tuple[int, int]] = 16, 16  # (u, v) for background
#     object_resolution: Optional[Tuple[int, int]] = 64, 64  # (u, v) for foreground object
#
#     bg_resolution_scale: float = .5  # Scale factor relative to base_resolution
#     object_resolution_scale: float = 2.0  # Scale factor relative to base_resolution
#
#
# @dataclass
# class AdaptiveSamplingResult:
#     """Result from adaptive sampling generation."""
#     # 1D intervals (separable, memory-efficient)
#     intervals_u: np.ndarray  # [Us] - sorted U coordinates
#     intervals_v: np.ndarray  # [Vs] - sorted V coordinates
#
#     # 2D grids (full flexibility, for view-dependent use)
#     grid_u: np.ndarray  # [Us, Vs] - U coordinate at each sample point
#     grid_v: np.ndarray  # [Us, Vs] - V coordinate at each sample point
#
#     # Metadata
#     complexity_map: np.ndarray  # [H_ctrl, W_ctrl] - complexity at control points
#     density_map: np.ndarray  # [Us, Vs] - target density used
#
#     def get_1d(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Get separable 1D intervals."""
#         return self.intervals_u, self.intervals_v
#
#     def get_2d(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Get full 2D grids."""
#         return self.grid_u, self.grid_v
#
#     def to_uv_grid(self) -> np.ndarray:
#         """Get stacked [Us, Vs, 2] grid."""
#         return np.stack([self.grid_u, self.grid_v], axis=-1)
#
# @dataclass
# class DecompositionResult:
#     """Result of scene decomposition."""
#     components: List[np.ndarray]  # List of point index arrays
#     labels: np.ndarray  # Per-point labels
#     mode: DecompositionMode
#     metadata: Dict[str, Any] = field(default_factory=dict)
#
#     @property
#     def num_components(self) -> int:
#         return len(self.components)
#
#     def get_component_sizes(self) -> List[int]:
#         return [len(c) for c in self.components]
#
#
# # =============================================================================
# # Data Containers
# # =============================================================================
# @dataclass
# class NURBSSurfaceData:
#     """Container for a single NURBS surface."""
#     control_points: np.ndarray  # [H, W, 3] XYZ
#     control_colors: np.ndarray  # [H, W, 3] RGB
#     knots_u: np.ndarray
#     knots_v: np.ndarray
#     degree_u: int
#     degree_v: int
#     label: str = "surface"
#     point_indices: Optional[np.ndarray] = None
#     bounds: Optional[Dict[str, np.ndarray]] = None
#     sampling_u_1D: Optional[np.ndarray] = None  # [Us, Vs] or [Us]
#     sampling_v_1D: Optional[np.ndarray] = None  # [Us, Vs] or [Vs]
#     grid_samplings_u: Optional[np.ndarray] = None  # [N, 3] sampled points on surface
#     grid_samplings_v: Optional[np.ndarray] = None  # [N, 3] sampled points on surface
#     complexity_map: Optional[np.ndarray] = None  # [H, W] complexity at control points
#     sampling_density_map: Optional[np.ndarray] = None  # [Us, Vs] target density
#     interval_u_1D: Optional[np.ndarray] = None  # [num_intervals_u + degree_u + 1]
#     interval_v_1D: Optional[np.ndarray] = None  # [num_intervals_v + degree_v + 1]
#     def to_torch(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
#         """Convert to torch tensors for SplineModel."""
#         result = {
#             'control_points': torch.tensor(self.control_points, dtype=torch.float32, device=device),
#             'control_colors': torch.tensor(self.control_colors, dtype=torch.float32, device=device),
#         }
#         if self.weights is not None:
#             result['weights'] = torch.tensor(self.weights, dtype=torch.float32, device=device)
#         if self.knots_u is not None:
#             result['knots_u'] = torch.tensor(self.knots_u, dtype=torch.float32, device=device)
#         if self.knots_v is not None:
#             result['knots_v'] = torch.tensor(self.knots_v, dtype=torch.float32, device=device)
#         return result
#
#
#
# @dataclass
# class AdaptiveSamplingConfig:
#     """Configuration for adaptive UV sampling generation."""
#     # Target sampling resolution
#     samples_u: int = 16
#     samples_v: int = 16
#     sampling_resolution_factor: float = 1.0
#     # Complexity weights
#     weight_curvature: float = 1.0
#     weight_color_variance: float = 0.5
#     weight_normal_variance: float = 0.8
#     weight_edge_proximity: float = 0.3
#
#     # Density control
#     min_density_ratio: float = 0.3  # Minimum relative density in low-complexity regions
#     max_density_ratio: float = 3.0  # Maximum relative density in high-complexity regions
#     smoothing_sigma: float = 2.0  # Gaussian smoothing of density map
#
#     # Monotonicity enforcement
#     enforce_monotonic: bool = True
#     min_spacing_ratio: float = 0.1  # Minimum spacing as fraction of uniform spacing
#
# @dataclass
# class MultiSurfaceResult:
#     """Result container for multi-surface decomposition."""
#     surfaces: List[NURBSSurfaceData]
#     decomposition_mode: DecompositionMode
#     labels: np.ndarray
#     metadata: Dict = field(default_factory=dict)
#
#
#
# # =============================================================================
# # Point Cloud Processor
# # =============================================================================
#
# class PointCloudProcessor:
#     """Handles point cloud preprocessing, normal estimation, and feature extraction."""
#
#     def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
#         self.points = np.asarray(points, dtype=np.float64)
#         self.colors = np.asarray(colors, dtype=np.float64) if colors is not None else None
#         self.n_points = len(points)
#
#         # Lazy computed properties
#         self._normals = None
#         self._kdtree = None
#         self._bounds = None
#         self._extent = None
#         self._complexity = None
#         self._curvature = None
#
#     @property
#     def kdtree(self) -> cKDTree:
#         if self._kdtree is None:
#             self._kdtree = cKDTree(self.points)
#         return self._kdtree
#
#     @property
#     def bounds(self) -> Dict[str, np.ndarray]:
#         if self._bounds is None:
#             self._bounds = {
#                 'min': self.points.min(axis=0),
#                 'max': self.points.max(axis=0),
#                 'center': self.points.mean(axis=0)
#             }
#         return self._bounds
#
#     @property
#     def extent(self) -> float:
#         if self._extent is None:
#             self._extent = np.linalg.norm(self.bounds['max'] - self.bounds['min'])
#         return self._extent
#
#     def estimate_normals(self, k: int = 30) -> np.ndarray:
#         """Estimate consistent surface normals using PCA."""
#         if self._normals is not None:
#             return self._normals
#
#         normals = np.zeros_like(self.points)
#         _, indices = self.kdtree.query(self.points, k=k)
#
#         for i in range(self.n_points):
#             neighbors = self.points[indices[i]]
#             centered = neighbors - neighbors.mean(axis=0)
#             try:
#                 _, _, Vt = np.linalg.svd(centered, full_matrices=False)
#                 normals[i] = Vt[-1]
#             except np.linalg.LinAlgError:
#                 normals[i] = np.array([0, 0, 1])
#
#         # Orient normals toward scene center
#         center_vecs = self.points - self.bounds['center']
#         flip_mask = np.sum(normals * center_vecs, axis=1) < 0
#         normals[flip_mask] *= -1
#
#         self._normals = normals
#         return normals
#
#     def estimate_curvature(self, k: int = 16) -> np.ndarray:
#         """Estimate surface curvature using normal variation in local neighborhood."""
#         if self._curvature is not None:
#             return self._curvature
#
#         normals = self.estimate_normals(k=k)
#         _, indices = self.kdtree.query(self.points, k=k)
#
#         curvatures = np.zeros(self.n_points)
#         for i in range(self.n_points):
#             neighbor_normals = normals[indices[i]]
#             cov = np.cov(neighbor_normals.T)
#             try:
#                 eigenvalues = np.linalg.eigvalsh(cov)
#                 curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
#             except np.linalg.LinAlgError:
#                 curvatures[i] = 0.0
#
#         self._curvature = curvatures
#         return curvatures
#
#     def compute_local_complexity(self, k: int = 16) -> np.ndarray:
#         """
#         Computes a scalar complexity score [0, 1] per point.
#         Combines geometric roughness (curvature) and color variation.
#         """
#         if self._complexity is not None:
#             return self._complexity
#
#         _, indices = self.kdtree.query(self.points, k=k)
#         normals = self.estimate_normals(k=k)
#
#         # Geometric complexity via normal variance
#         neighbor_normals = normals[indices]
#         geo_complexity = np.var(neighbor_normals, axis=1).sum(axis=1)
#
#         # Normalize (robust)
#         p95_geo = np.percentile(geo_complexity, 95)
#         geo_complexity = np.clip(geo_complexity / (p95_geo + 1e-8), 0, 1)
#
#         # Color complexity
#         if self.colors is not None:
#             neighbor_colors = self.colors[indices]
#             color_complexity = np.var(neighbor_colors, axis=1).sum(axis=1)
#             p95_col = np.percentile(color_complexity, 95)
#             color_complexity = np.clip(color_complexity / (p95_col + 1e-8), 0, 1)
#         else:
#             color_complexity = np.zeros_like(geo_complexity)
#
#         complexity = 0.7 * geo_complexity + 0.3 * color_complexity
#         complexity = (complexity - complexity.min()) / (complexity.ptp() + 1e-8)
#         self._complexity = complexity
#
#         return self._complexity
#
#     def compute_normal_variance(self, k: int = 16) -> np.ndarray:
#         """Compute local normal variance (high at edges/corners)."""
#         normals = self.estimate_normals()
#         _, indices = self.kdtree.query(self.points, k=k)
#         neighbor_normals = normals[indices]
#         return np.var(neighbor_normals, axis=1).sum(axis=1)
#
#     def compute_color_variance(self, k: int = 16) -> np.ndarray:
#         """Compute local color variance (high at texture boundaries)."""
#         if self.colors is None:
#             return np.zeros(self.n_points)
#         _, indices = self.kdtree.query(self.points, k=k)
#         neighbor_colors = self.colors[indices]
#         return np.var(neighbor_colors, axis=1).sum(axis=1)
#
#     def remove_outliers(self, std_ratio: float = 2.0, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
#         """Remove statistical outliers."""
#         distances, _ = self.kdtree.query(self.points, k=k)
#         mean_distances = distances[:, 1:].mean(axis=1)
#
#         global_mean = mean_distances.mean()
#         global_std = mean_distances.std()
#         threshold = global_mean + std_ratio * global_std
#
#         outlier_mask = mean_distances > threshold
#         clean_indices = np.where(~outlier_mask)[0]
#
#         return clean_indices, outlier_mask
#
#
# # =============================================================================
# # Scene Decomposer
# # =============================================================================
#
# class SceneDecomposer:
#     """
#     Advanced scene decomposition with feature-aware clustering
#     and soft-weighted connectivity graphs.
#     """
#
#     def __init__(self, processor: PointCloudProcessor, config: SurfaceConfig):
#         self.processor = processor
#         self.config = config
#
#         self._surface_graph = None
#     def decompose_background_object(self) -> Tuple[np.ndarray, np.ndarray]:
#         """
#             Semantic background/foreground separation.
#
#             Strategy:
#             1. Compute multiple BG indicators (distance, planarity, normal consistency)
#             2. Fuse indicators into soft probability
#             3. Graph-cut or spectral partitioning respecting surface topology
#             4. Validate and fix connectivity
#             """
#
#         if self._surface_graph is None:
#             self._build_surface_graph()
#         method = self.config.bg_detection_method
#
#         if method == "distance":
#             bg_scores = self._compute_distance_bg_scores()
#         elif method == "normal":
#             bg_scores = self._compute_normal_bg_scores()
#         elif method == "density":
#             bg_scores = self._compute_density_bg_scores()
#         elif method == "hybrid":
#             bg_scores = self._compute_hybrid_bg_scores()
#         else:
#             raise ValueError(f"Unknown bg_detection_method: {method}")
#         # bg_scores = self._compute_density_bg_scores()
#         # bg_scores = self._compute_hybrid_bg_scores()
#
#         # Convert scores to binary labels using graph-aware partitioning
#         bg_indices, fg_indices = self._partition_with_graph_cut(bg_scores)
#         # bg_indices, fg_indices = np.where(bg_scores)[0], np.where(~bg_scores < )[0]
#
#         # Validate connectivity for each partition
#         # bg_indices = self._ensure_connected_partition(bg_indices, "background")
#         # fg_indices = self._ensure_connected_partition(fg_indices, "foreground")
#
#         # Handle edge cases
#         if len(bg_indices) < self.config.min_component_size:
#             warnings.warn("Background too small, returning single surface")
#             return self._decompose_single()
#
#         if len(fg_indices) < self.config.min_component_size:
#             warnings.warn("Foreground too small, returning single surface")
#             return self._decompose_single()
#
#         # Build labels
#         # labels = np.full(n_points, -1, dtype=np.int32)
#         # labels[bg_indices] = 0
#         # labels[fg_indices] = 1
#         #
#         # return DecompositionResult(
#         #     components=[bg_indices, fg_indices],
#         #     labels=labels,
#         #     mode=DecompositionMode.BACKGROUND_OBJECT,
#         #     metadata={
#         #         'bg_score_range': (bg_scores.min(), bg_scores.max()),
#         #         'bg_size': len(bg_indices),
#         #         'fg_size': len(fg_indices),
#         #         'method': method
#         #     }
#         # )
#         #
#         # points = self.processor.points
#         # center = self.processor.bounds['center']
#         # distances = np.linalg.norm(points - center, axis=1)
#         # threshold_percentile = 100 * (1 - self.config.bg_distance_threshold)
#         # bg_mask = distances > np.percentile(distances, threshold_percentile)
#         return bg_indices, fg_indices#np.where(bg_mask)[0], np.where(~bg_mask)[0]
#
#     def _build_surface_graph(self) -> csr_matrix:
#         """
#         Build connectivity graph respecting surface topology.
#         """
#         points = self.processor.points
#         normals = self.processor.estimate_normals()
#         n_points = len(points)
#
#         # Adaptive radius based on local density
#         tree = cKDTree(points)
#         k_density = min(16, n_points - 1)
#         distances, _ = tree.query(points, k=k_density)
#
#         median_spacing = np.median(distances[:, 1])
#         local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
#
#         # Adaptive radius per point
#         adaptive_radius = median_spacing * 2.5 / (local_density / local_density.mean() + 0.5)
#         adaptive_radius = np.clip(adaptive_radius, median_spacing, median_spacing * 5)
#
#         # Build edges
#         rows, cols, weights = [], [], []
#         normal_threshold = np.cos(np.radians(self.config.max_normal_angle_diff))
#
#         for i in range(n_points):
#             neighbors = tree.query_ball_point(points[i], adaptive_radius[i])
#
#             for j in neighbors:
#                 if j <= i:
#                     continue
#
#                 # Normal consistency check
#                 normal_dot = np.dot(normals[i], normals[j])
#                 if normal_dot < normal_threshold:
#                     continue
#
#                 # Edge direction should be in tangent plane
#                 edge_dir = points[j] - points[i]
#                 edge_len = np.linalg.norm(edge_dir)
#                 if edge_len < 1e-8:
#                     continue
#                 edge_dir /= edge_len
#
#                 tangent_dev_i = abs(np.dot(edge_dir, normals[i]))
#                 tangent_dev_j = abs(np.dot(edge_dir, normals[j]))
#
#                 if tangent_dev_i > 0.7 or tangent_dev_j > 0.7:
#                     continue
#
#                 # Compute edge weight
#                 dist_weight = np.exp(-edge_len / (self.processor.extent * 0.1))
#                 normal_weight = (normal_dot + 1) / 2
#                 tangent_weight = 1.0 - 0.5 * (tangent_dev_i + tangent_dev_j)
#
#                 weight = dist_weight * normal_weight * tangent_weight
#
#                 rows.extend([i, j])
#                 cols.extend([j, i])
#                 weights.extend([weight, weight])
#
#         self._surface_graph = csr_matrix(
#             (weights, (rows, cols)),
#             shape=(n_points, n_points)
#         )
#
#         return self._surface_graph
#     def _partition_with_graph_cut(
#             self,
#             bg_scores: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Partition points using normalized graph cut.
#
#         This respects surface topology while using bg_scores as prior.
#         """
#         n_points = len(bg_scores)
#
#         # Build affinity matrix from surface graph
#         # Weight edges by similarity (1 - score difference)
#         graph = self._surface_graph.copy()
#         rows, cols = graph.nonzero()
#
#         # Edge weights: combine topology and score similarity
#         topo_weights = np.array(graph[rows, cols]).flatten()
#         score_diff = np.abs(bg_scores[rows] - bg_scores[cols])
#         score_sim = np.exp(-score_diff ** 2 / 0.1)  # Gaussian similarity
#
#         combined_weights = topo_weights * score_sim
#
#         # Create weighted graph
#         weighted_graph = csr_matrix(
#             (combined_weights, (rows, cols)),
#             shape=(n_points, n_points)
#         )
#         weighted_graph = 0.5 * (weighted_graph + weighted_graph.T)
#
#         try:
#             clustering = SpectralClustering(
#                 n_clusters=2,
#                 affinity='precompute',
#                 assign_labels='kmeans',
#                 random_state=42
#             )
#             # Convert to dense affinity (spectral needs it)
#             affinity = weighted_graph.toarray()
#             np.fill_diagonal(affinity, 0)  # No self-loops
#             labels = clustering.fit_predict(affinity)
#         except Exception as e:
#             warnings.warn(f"Spectral clustering failed: {e}. Using threshold.")
#             # Fallback:  simple threshold
#             threshold = np.percentile(bg_scores, 100 * (1 - self.config.bg_distance_threshold))
#             labels = (bg_scores > threshold).astype(int)
#
#         # Determine which cluster is background (higher mean bg_score)
#         cluster_0_score = bg_scores[labels == 0].mean()
#         cluster_1_score = bg_scores[labels == 1].mean()
#
#         if cluster_0_score > cluster_1_score:
#             bg_mask = labels == 0
#         else:
#             bg_mask = labels == 1
#
#         fg_mask = ~bg_mask
#
#         # Validate size constraints
#         bg_ratio = bg_mask.sum() / n_points
#         # if bg_ratio < self.config.bg_min_size_ratio:
#         #     # BG too small, adjust threshold
#         #     target_size = int(n_points * self.config.bg_min_size_ratio)
#         #     threshold_idx = np.argsort(bg_scores)[-target_size]
#         #     bg_mask = bg_scores >= bg_scores[threshold_idx]
#         #     fg_mask = ~bg_mask
#         #
#         # elif bg_ratio > self.config.bg_max_size_ratio:
#         #     # BG too large, adjust threshold
#         #     target_size = int(n_points * self.config.bg_max_size_ratio)
#         #     threshold_idx = np.argsort(bg_scores)[-target_size]
#         #     bg_mask = bg_scores >= bg_scores[threshold_idx]
#         #     fg_mask = ~bg_mask
#
#         return bg_mask, fg_mask
#     def _ensure_connected_partition(
#             self,
#             mask: np.ndarray,
#             name: str
#     ) -> np.ndarray:
#         """
#         Ensure a partition mask forms a single connected component.
#
#         If multiple components exist, return the largest one.
#         """
#         indices = np.where(mask)[0]
#
#         if len(indices) < self.config.min_component_size:
#             return indices
#
#         # Extract subgraph
#         subgraph = self._surface_graph[mask][:, mask]
#
#         n_cc, cc_labels = connected_components(subgraph, directed=False)
#
#         if n_cc == 1:
#             return indices
#
#         # Find largest connected component
#         sizes = [(cc_labels == i).sum() for i in range(n_cc)]
#         largest_cc = np.argmax(sizes)
#
#         # Map back to global indices
#         local_mask = cc_labels == largest_cc
#         connected_indices = indices[local_mask]
#
#         discarded = len(indices) - len(connected_indices)
#         if discarded > 0:
#             warnings.warn(
#                 f"{name}:  Had {n_cc} components, keeping largest "
#                 f"({len(connected_indices)} points, discarding {discarded})"
#             )
#
#         return connected_indices
#     def _compute_distance_bg_scores(self) -> np.ndarray:
#         """
#         Distance-based BG score (improved).
#
#         Instead of simple distance from center, uses:
#         - Distance from centroid
#         - Distance from principal plane
#         - Boundary proximity in UV-like projection
#         """
#         points = self.processor.points
#         n_points = len(points)
#
#         # 1. Distance from centroid
#         centroid = points.mean(axis=0)
#         dist_from_center = np.linalg.norm(points - centroid, axis=1)
#         dist_norm = (dist_from_center - dist_from_center.min()) / (dist_from_center.ptp() + 1e-8)
#
#         # 2. Distance from principal plane (points far from plane = BG)
#         pca = PCA(n_components=3)
#         pca.fit(points - centroid)
#
#         # Project onto smallest variance direction (normal to main plane)
#         normal_dir = pca.components_[2]  # Smallest variance
#         dist_from_plane = np.abs(np.dot(points - centroid, normal_dir))
#         plane_norm = (dist_from_plane - dist_from_plane.min()) / (dist_from_plane.ptp() + 1e-8)
#
#         # 3. Boundary score in UV projection
#         uv_proj = pca.transform(points - centroid)[:, :2]
#         uv_min, uv_max = uv_proj.min(axis=0), uv_proj.max(axis=0)
#         uv_center = (uv_min + uv_max) / 2
#         uv_extent = (uv_max - uv_min) / 2 + 1e-8
#
#         # Distance from UV center (normalized)
#         uv_dist = np.linalg.norm((uv_proj - uv_center) / uv_extent, axis=1)
#         boundary_score = np.clip(uv_dist, 0, 1)
#
#         # Combine:  high score = likely background
#         bg_score = 0.3 * dist_norm + 0.3 * plane_norm + 0.4 * boundary_score
#
#         return bg_score
#
#     def _compute_normal_bg_scores(self) -> np.ndarray:
#         """
#         Normal-based BG score.
#
#         Background typically has:
#         - More consistent normals (low local variance)
#         - Normals aligned with dominant direction
#         """
#         normals = self.processor.estimate_normals()
#         points = self.processor.points
#         tree = cKDTree(points)
#
#         k = min(20, len(points) - 1)
#         _, indices = tree.query(points, k=k)
#
#         # 1. Local normal consistency
#         normal_consistency = np.zeros(len(points))
#         for i in range(len(points)):
#             neighbor_normals = normals[indices[i]]
#             # Variance of dot products with center normal
#             dots = np.abs(neighbor_normals @ normals[i])
#             normal_consistency[i] = dots.mean()  # High = consistent
#
#         # 2. Alignment with dominant normal direction
#         # Find dominant normal using PCA on normals
#         normal_pca = PCA(n_components=1)
#         normal_pca.fit(normals)
#         dominant_normal = normal_pca.components_[0]
#
#         alignment = np.abs(normals @ dominant_normal)
#
#         # Combine: high consistency + high alignment = BG
#         bg_score = 0.5 * normal_consistency + 0.5 * alignment
#
#         return bg_score
#
#     def _compute_surface_features(self) -> np.ndarray:
#         """
#         Compute rich features for clustering.
#         """
#         points = self.processor.points
#         normals = self.processor.estimate_normals()
#         colors = self.processor.colors if self.processor.colors is not None else np.zeros_like(points)
#         curvature = self.processor.estimate_curvature()
#
#         # Standardized spatial
#         scaler = StandardScaler()
#         feat_spatial = scaler.fit_transform(points) * self.config.weight_spatial
#
#         # Normals
#         feat_normal = normals * self.config.weight_normal
#
#         # Colors
#         feat_color = colors * self.config.weight_color
#
#         # Curvature
#         curv_norm = (curvature - curvature.min()) / (curvature.ptp() + 1e-8)
#         feat_curv = curv_norm.reshape(-1, 1) * self.config.weight_complexity
#
#         return np.hstack([feat_spatial, feat_normal, feat_color, feat_curv])
#
#     def _compute_density_bg_scores(self) -> np.ndarray:
#         """
#         Density-based BG score.
#
#         Background often has:
#         - More uniform density
#         - Lower local complexity
#         """
#         complexity = self.processor.compute_local_complexity()
#
#         # Low complexity = likely background
#         # bg_score = 1.0 -
#
#         return 1-complexity
#     def _compute_hybrid_bg_scores(self) -> np.ndarray:
#         """
#         Combine all indicators with adaptive weighting.
#         """
#         # Compute individual scores
#         dist_score = self._compute_distance_bg_scores()
#         normal_score = self._compute_normal_bg_scores()
#         density_score = self._compute_density_bg_scores()
#         offset_score = self.offset_from_center()
#         color_score = self._compute_surface_features()
#         # Adaptive weighting based on score confidence
#         dist_conf = 1.0 / (dist_score.std() + 0.1)
#         normal_conf = 1.0 / (normal_score.std() + 0.1)
#         density_conf = 1.0 / (density_score.std() + 0.1) #* 0
#         offset_conf = 1.0 / (offset_score.std() + 0.1) #* 0
#         color_conf = 1.0 / (color_score.std() + 0.1)
#
#         total_conf = density_conf + offset_conf #+dist_conf + normal_conf +  1e-10
#
#         # Weighted combination
#         bg_score = (
#                 # dist_score * (dist_conf / total_conf +
#                 # normal_score * normal_conf / total_conf +
#                 density_score * density_conf / total_conf +
#                 offset_score * offset_conf / total_conf
#                 # color_score.mean(axis=-1)# * color_conf / total_conf
#         )
#
#         return bg_score
#
#     def offset_from_center(self) -> np.ndarray:
#         """Separate scene into background and foreground."""
#         points = self.processor.points
#         center = self.processor.bounds['center']
#         distances = np.linalg.norm(points - center, axis=1)
#         distances = (distances - distances.min()) / (distances.ptp() + 1e-8)
#         # threshold_percentile = 100 * (1 - self.config.bg_distance_threshold)
#         # bg_mask = distances > np.percentile(distances, threshold_percentile)
#         return distances #, np.where(bg_mask)[0], np.where(~bg_mask)[0]
#     def decompose_k_components(self, k: Optional[int] = None) -> List[np.ndarray]:
#         """
#         Decompose scene into K components using spatially-constrained
#         agglomerative clustering with enhanced features.
#         """
#         if k is None:
#             k = self.config.n_components
#
#         points = self.processor.points
#         n_points = len(points)
#
#         if n_points < self.config.min_component_size * k:
#             warnings.warn("Too few points for K-components.  Returning single component.")
#             return [np.arange(n_points)]
#
#         # Build feature matrix
#         features = self._compute_enhanced_features()
#
#         # Build connectivity graph
#         if self.config.use_soft_edges:
#             connectivity = self._build_weighted_connectivity()
#         else:
#             connectivity = kneighbors_graph(
#                 points,
#                 n_neighbors=self.config.connectivity_k,
#                 include_self=False,
#                 n_jobs=-1
#             )
#             5 * (connectivity + connectivity.T)
#
#         # Clustering
#         try:
#             model = AgglomerativeClustering(
#                 n_clusters=k,
#                 metric='euclidean',
#                 linkage='ward',
#                 connectivity=connectivity
#             )
#             labels = model.fit_predict(features)
#         except Exception as e:
#             warnings.warn(f"Agglomerative clustering failed ({e}), falling back to K-Means")
#             kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
#             labels = kmeans.fit_predict(features)
#
#         # Filter small components
#         component_indices = []
#         for i in range(k):
#             idx = np.where(labels == i)[0]
#             if len(idx) >= self.config.min_component_size:
#                 component_indices.append(idx)
#
#         if len(component_indices) == 0:
#             return [np.arange(n_points)]
#
#         return component_indices
#
#     def _compute_enhanced_features(self) -> np.ndarray:
#         """
#         Compute rich feature vectors for segmentation.
#         Includes spatial, geometric, appearance, and local consistency features.
#         """
#         points = self.processor.points
#         normals = self.processor.estimate_normals(k=self.config.normal_estimation_k)
#         colors = self.processor.colors if self.processor.colors is not None else np.zeros_like(points)
#
#         # Spatial features (standardized)
#         scaler = StandardScaler()
#         feat_xyz = scaler.fit_transform(points) * self.config.weight_spatial
#
#         # Normal features
#         feat_normal = normals * self.config.weight_normal
#
#         # Color features - convert to LAB if possible for perceptual uniformity
#         try:
#             from skimage.color import rgb2lab
#             colors_rgb = np.clip(colors, 0, 1)
#             colors_lab = rgb2lab(colors_rgb.reshape(1, -1, 3)).reshape(-1, 3)
#             colors_lab[:, 0] /= 100.
#             0
#             colors_lab[:, 1:] /= 128.0
#             feat_color = colors_lab * self.config.weight_color
#         except ImportError:
#             feat_color = colors * self.config.weight_color
#
#         # Curvature feature
#         curvature = self.processor.estimate_curvature(k=8)
#         curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
#         feat_curvature = curvature_norm.reshape(-1, 1) * self.config.weight_complexity
#
#         # Local normal variance (edge detector)
#         normal_variance = self.processor.compute_normal_variance(k=8)
#         nv_norm = (normal_variance - normal_variance.min()) / (normal_variance.max() - normal_variance.min() + 1e-8)
#         feat_edge = nv_norm.reshape(-1, 1) * self.config.weight_normal * 0.5
#
#         # Local color variance (texture boundary detector)
#         color_variance = self.processor.compute_color_variance(k=8)
#         cv_norm = (color_variance - color_variance.min()) / (color_variance.max() - color_variance.min() + 1e-8)
#         feat_texture = cv_norm.reshape(-1, 1) * self.config.weight_color * 0.5
#
#         return np.hstack([
#             feat_xyz,
#             feat_normal,
#             feat_color,
#             feat_curvature,
#             feat_edge,
#             feat_texture
#         ])
#
#     def _build_weighted_connectivity(self) -> csr_matrix:
#         """
#         Build connectivity graph with soft edge weights instead of hard pruning.
#         """
#         points = self.processor.points
#         normals = self.processor.estimate_normals()
#         complexity = self.processor.compute_local_complexity()
#         colors = self.processor.colors if self.processor.colors is not None else np.zeros_like(points)
#
#         # Base connectivity
#         adjacency = kneighbors_graph(
#             points,
#             n_neighbors=self.config.connectivity_k,
#             mode='distance',
#             include_self=False
#         )
#
#         rows, cols = adjacency.nonzero()
#         base_weights = np.array(adjacency[rows, cols]).flatten()
#
#         # Distance weights (closer = higher weight)
#         max_dist = np.percentile(base_weights, 95)
#         dist_weights = 1.0 - np.clip(base_weights / max_dist, 0, 1)
#
#         # Normal consistency (soft falloff)
#         norm_i = normals[rows]
#         norm_j = normals[cols]
#         dot_products = np.clip(np.sum(norm_i * norm_j, axis=1), -1.0, 1.0)
#         normal_weights = (dot_products + 1) / 2  # Map [-1, 1] to [0, 1]
#
#         # Complexity consistency
#         comp_i = complexity[rows]
#         comp_j = complexity[cols]
#         comp_diff = np.abs(comp_i - comp_j)
#         sigma_complexity = 0.3
#         complexity_weights = np.exp(-comp_diff ** 2 / sigma_complexity ** 2)
#
#         # Color consistency
#         col_i = colors[rows]
#         col_j = colors[cols]
#         color_diff = np.linalg.norm(col_i - col_j, axis=1)
#         if self.config.edge_color_adaptive:
#             sigma_color = np.percentile(color_diff, 5) + 1e-8
#         else:
#             sigma_color = 0.5
#         color_weights = np.exp(-color_diff ** 2 / sigma_color ** 2)
#
#         combined_weights = (
#                 dist_weights ** 0.3 *
#                 normal_weights ** 0.4 *
#                 complexity_weights ** 0.2 *
#                 color_weights ** 0.1
#         )
#
#         # Build sparse matrix
#         weighted_adjacency = csr_matrix(
#             (combined_weights, (rows, cols)),
#             shape=adjacency.shape
#         )
#
#         # Make symmetric
#         weighted_adjacency = 0.5 * (weighted_adjacency + weighted_adjacency.T)
#
#         return weighted_adjacency
#
#
# # =============================================================================
# # Adaptive Resolution Calculator
# # =============================================================================
#
# class AdaptiveResolutionCalculator:
#     """Calculates optimal grid resolution for a point cluster."""
#
#     def __init__(self, config: SurfaceConfig):
#         self.config = config
#
#     def calculate(
#             self,
#             points: np.ndarray,
#             complexity: Optional[np.ndarray] = None
#     ) -> Tuple[int, int]:
#         """
#         Compute resolution based on aspect ratio, point density, and complexity.
#         """
#         if not self.config.adaptive_resolution:
#             return self.config.resolution_u, self.config.resolution_v
#
#         n_points = len(points)
#         if n_points < 12:
#             return self.config.min_resolution, self.config.min_resolution
#
#         # PCA for aspect ratio
#         try:
#             pca = PCA(n_components=2)
#             projected = pca.fit_transform(points - points.mean(axis=0))
#             extents = projected.max(axis=0) - projected.min(axis=0)
#             u_extent, v_extent = max(extents[0], 1e-6), max(extents[1], 1e-6)
#         except Exception:
#             u_extent = v_extent = 1.0
#
#         # Aspect ratio (clamped)
#         aspect = np.clip(u_extent / v_extent, 0.25, 4.0)
#
#         # Base resolution from point count
#         base_from_points = int(np.sqrt(n_points) * 0.5)
#         base_res = np.clip(base_from_points, self.config.min_resolution, self.config.base_resolution)
#
#         # Complexity boost
#         if complexity is not None and len(complexity) > 0:
#             avg_complexity = np.percentile(complexity, 80)
#             complexity_factor = 1.0 + self.config.resolution_complexity_sensitivity * avg_complexity
#         else:
#             complexity_factor = 1.0
#         print(f"[Resolution] Base: {base_res}, Complexity factor: {complexity_factor:.2f}, Aspect: {aspect:.2f}")
#         # Final resolution
#         total_res = base_res * complexity_factor
#
#         if aspect >= 1.0:
#             res_u = int(total_res)
#             res_v = int(total_res / aspect)
#         else:
#             res_u = int(total_res * aspect)
#             res_v = int(total_res)
#
#         # Clamp
#         res_u = int(np.clip(res_u, self.config.min_resolution, self.config.max_resolution))
#         res_v = int(np.clip(res_v, self.config.min_resolution, self.config.max_resolution))
#
#         return res_u, res_v
#
#
# # =============================================================================
# # NURBS Surface Fitter
# # =============================================================================
#
# class NURBSSurfaceFitter:
#     """Fits NURBS surfaces to point cloud regions with adaptive resolution."""
#
#     def __init__(self, config: SurfaceConfig):
#         self.config = config
#         self.res_calculator = AdaptiveResolutionCalculator(config)
#         self.sampling_generator = AdaptiveSamplingGenerator(
#             AdaptiveSamplingConfig(
#                 sampling_resolution_factor=config.sampling_resolution_factor,
#                 weight_curvature=config.weight_complexity,
#                 weight_color_variance=config.weight_color,
#                 weight_normal_variance=config.weight_normal,
#             )
#         )
#
#     def _generate_adaptive_samples(
#             self,
#             surface_data: NURBSSurfaceData,
#             sampling_resolution: Optional[Tuple[int, int]] = None
#     ) -> AdaptiveSamplingResult: #Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         if sampling_resolution is not None:
#             self.sampling_generator.config.samples_u = sampling_resolution[0]
#             self.sampling_generator.config.samples_v = sampling_resolution[1]
#
#         sampling_result = self.sampling_generator.generate(
#             control_points=surface_data.control_points,
#             control_colors=surface_data.control_colors,
#             knots_u=surface_data.knots_u,
#             knots_v=surface_data.knots_v,
#             degree=surface_data.degree_u
#         )
#         return sampling_result
#
#     def fit_surface(
#             self,
#             points: np.ndarray,
#             colors: Optional[np.ndarray] = None,
#             label: str = "surface",
#             complexity: Optional[np.ndarray] = None,
#             generate_adaptive_samples: bool = True,
#             sampling_resolution: Optional[Tuple[int, int]] = None,
#             override_resolution: Optional[Tuple[int, int]] = None,
#             parameterization: Optional[str] = None  # Allow explicit override
#     ) -> NURBSSurfaceData:
#         """Fit a NURBS surface to a point cloud region."""
#         if len(points) < 16:
#             raise ValueError(f"Need at least 16 points, got {len(points)}")
#
#         # Use override resolution if provided, otherwise use adaptive
#         if override_resolution is not None:
#             res_u, res_v = override_resolution
#             res_u = int(np.clip(res_u, self.config.min_resolution, self.config.max_resolution))
#             res_v = int(np.clip(res_v, self.config.min_resolution, self.config.max_resolution))
#             sampling_resolution = (res_u, res_v)
#         else:
#             res_u, res_v = self.res_calculator.calculate(points, complexity)
#
#         # Parameterize
#         method = parameterization if parameterization else self.config.parameterization
#
#         # Backward compatibility logic
#         if method == 'pca' and self.config.use_geodesic_uv and len(points) > 12:
#             method = 'geodesic'
#         if method == 'spherical':
#             uv_coords = self._parameterize_spherical(points)
#         elif method == 'geodesic':
#             uv_coords = self._parameterize_geodesic(points)
#         else:
#             uv_coords = self._parameterize_pca(points)
#
#         print(f"[Fit] Label: {label}, Res: {res_u}x{res_v}, Method: {method}")
#
#         # Create grid samples with hole filling
#         grid_xyz, grid_rgb = self._create_grid_samples(
#             points, colors, uv_coords, res_u, res_v
#         )
#
#         # Fit B-spline with awareness of parameterization method for wrapping
#         is_spherical = (method == 'spherical')
#         surface_data = self._fit_bspline_to_grid(grid_xyz, grid_rgb, label, res_u, res_v, is_spherical)
#
#         # Metadata
#         surface_data.bounds = {
#             'min': points.min(axis=0),
#             'max': points.max(axis=0),
#             'center': points.mean(axis=0)
#         }
#
#         if self.config.generate_adaptive_samples:
#             sampling_result = self._generate_adaptive_samples(
#                 surface_data, sampling_resolution
#             )
#             surface_data.sampling_u_1D = sampling_result.intervals_u
#             surface_data.sampling_v_1D = sampling_result.intervals_v
#             surface_data.grid_samplings_u = sampling_result.grid_u
#             surface_data.grid_samplings_v = sampling_result.grid_v
#             surface_data.complexity_map = sampling_result.complexity_map
#
#         return surface_data
#
#     def encode_sphere(self, interval: torch.Tensor) -> torch.Tensor:
#         centroid = interval.mean(dim=0)
#         centered = interval - centroid
#         r= torch.norm(centered, dim=1)
#         r = torch.clamp(r, min=1e-8)
#         phi = torch.acos(torch.clamp(centered[:, 2] / r, -1, 1))
#         theta = torch.atan2(centered[:, 1], centered[:, 0])
#         u = (theta + np.pi) / (2 * np.pi)
#         v = phi / np.pi
#
#         # radius = (interval.max(dim=0).values - interval.min(dim=0).values).max() / 2.0
#
#     def _parameterize_spherical(self, points: np.ndarray) -> np.ndarray:
#         """
#         Spherical UV parameterization for 360/Object-centric scenes.
#         Maps 3D points to UVs based on spherical angles.
#         """
#         centroid = points.mean(axis=0)
#         centered = points - centroid
#
#         r = np.linalg.norm(centered, axis=1)
#         r = np.maximum(r, 1e-8)
#
#         # Phi: [0, pi] (angle from Z-axis)
#         phi = np.arccos(np.clip(centered[:, 2] / r, -1, 1))
#
#         # Theta: [-pi, pi] (angle in XY plane)
#         theta = np.arctan2(centered[:, 1], centered[:, 0])
#
#         # Map to UV [0, 1]
#         # u -> theta / 2pi  (0 to 1 wrapping around Z)
#         # v -> phi / pi     (0 to 1 top to bottom)
#         u = (theta + np.pi) / (2 * np.pi)
#         v = phi / np.pi
#
#         uv_coords = np.stack([u, v], axis=1)
#         # Clip to avoid exact 0/1 issues in some indices, though mostly safe
#         return np.clip(uv_coords, 0.001, 0.999)
#
#     def _inverse_spherical_parameterization(
#             self,
#             uv_coords: np.ndarray,
#             centroid: np.ndarray,
#             radius: float = 1.0
#     ) -> np.ndarray:
#         """
#         Inverse of spherical UV parameterization.
#         Maps UV coordinates back to 3D points on a sphere.
#
#         Args:
#             uv_coords: [N, 2] array of UV coordinates in [0, 1]
#             centroid: [3,] center point of the sphere
#             radius: Scalar or [N,] array of radii (distance from centroid)
#
#         Returns:
#             points: [N, 3] array of 3D Cartesian coordinates
#         """
#         u = uv_coords[:, 0]
#         v = uv_coords[:, 1]
#
#         # Inverse mapping from UV to spherical angles
#         # u -> theta: [0, 1] -> [-pi, pi]
#         theta = u * (2 * np.pi) - np.pi
#
#         # v -> phi: [0, 1] -> [0, pi]
#         phi = v * np.pi
#
#         # Convert spherical to Cartesian
#         # x = r * sin(phi) * cos(theta)
#         # y = r * sin(phi) * sin(theta)
#         # z = r * cos(phi)
#
#         if np.isscalar(radius):
#             r = radius
#         else:
#             r = radius  # Per-point radii
#
#         x = r * np.sin(phi) * np.cos(theta)
#         y = r * np.sin(phi) * np.sin(theta)
#         z = r * np.cos(phi)
#
#         # Stack and translate back to original centroid
#         centered = np.stack([x, y, z], axis=1)
#         points = centered + centroid
#
#         return points
#     def _parameterize_pca(self, points: np.ndarray) -> np.ndarray:
#         """Simple PCA-based UV parameterization."""
#         center = points.mean(axis=0)
#         centered = points - center
#
#         pca = PCA(n_components=3)
#         pca.fit(centered)
#         projected = pca.transform(centered)[:, :2]
#
#         uv_min = projected.min(axis=0)
#         uv_max = projected.max(axis=0)
#         uv_range = uv_max - uv_min
#         uv_range[uv_range < 1e-6] = 1.0
#
#         uv_coords = (projected - uv_min) / uv_range
#         return np.clip(uv_coords, 0.01, 0.99)
#
#     def _parameterize_geodesic(self, points: np.ndarray) -> np.ndarray:
#         """
#         Improved UV parameterization using geodesic distances.
#         Better handles curved surfaces.
#         """
#         n_points = len(points)
#
#         if n_points < 12:#self.config.min_component_size:
#             return self._parameterize_pca(points)
#
#         # Build local neighborhood graph
#         tree = cKDTree(points)
#         k = min(12, n_points - 1)
#         distances, indices = tree.query(points, k=k)
#
#         # Create weighted graph
#         rows = np.repeat(np.arange(n_points), k)
#         cols = indices.flatten()
#         weights = distances.flatten()
#
#         graph = csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))
#         graph = 0.5 * (graph + graph.T)
#
#         # Find anchor points along principal axis
#         pca = PCA(n_components=1)
#         proj = pca.fit_transform(points - points.mean(axis=0)).flatten()
#         anchor1 = int(np.argmin(proj))
#         anchor2 = int(np.argmax(proj))
#
#         # Geodesic distances from anchors
#         try:
#             dist_from_1 = shortest_path(graph, indices=[anchor1], directed=False)[0]
#             dist_from_2 = shortest_path(graph, indices=[anchor2], directed=False)[0]
#         except Exception:
#             return self._parameterize_pca(points)
#
#         # Check for infinite distances (disconnected graph)
#         if np.isinf(dist_from_1).any() or np.isinf(dist_from_2).any():
#             return self._parameterize_pca(points)
#
#         # U from geodesic distance ratio
#         total_dist = dist_from_1[anchor2]
#         if total_dist < 1e-8:
#             return self._parameterize_pca(points)
#
#         u_coords = dist_from_1 / total_dist
#
#         # V from perpendicular spread
#         axis = points[anchor2] - points[anchor1]
#         axis = axis / (np.linalg.norm(axis) + 1e-8)
#
#         centered = points - points[anchor1]
#         along_axis = np.dot(centered, axis)[:, None] * axis
#         perpendicular = centered - along_axis
#
#         if np.std(perpendicular) > 1e-6:
#             pca_perp = PCA(n_components=1)
#             v_coords = pca_perp.fit_transform(perpendicular).flatten()
#             v_coords = (v_coords - v_coords.min()) / (v_coords.max() - v_coords.min() + 1e-8)
#         else:
#             v_coords = np.zeros(n_points)
#
#         uv_coords = np.stack([u_coords, v_coords], axis=1)
#         return np.clip(uv_coords, 0.001, 0.999)
#
#     def _create_knot_vector(self, n_ctrl: int, degree: int) -> np.ndarray:
#         """Create clamped uniform knot vector."""
#         n_knots = n_ctrl + degree + 1
#         n_internal = n_knots - 2 * (degree + 1)
#
#         knots = np.zeros(n_knots)
#         knots[-degree - 1:] = 1.0
#
#         if n_internal > 0:
#             internal = np.linspace(0, 1, n_internal + 2)[1:-1]
#             knots[degree + 1:degree + 1 + n_internal] = internal
#
#         return knots
#
#     def _create_grid_samples(
#             self,
#             points: np.ndarray,
#             colors: Optional[np.ndarray],
#             uv_coords: np.ndarray,
#             res_u: int,
#             res_v: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Resample points onto grid with robust hole filling.
#         """
#         u_vals = np.linspace(0, 1, res_u)
#         v_vals = np.linspace(0, 1, res_v)
#
#         grid_xyz = np.full((res_u, res_v, 3), np.nan)
#         grid_rgb = np.full((res_u, res_v, 3), np.nan)
#
#         # KD-tree in UV space
#         uv_tree = cKDTree(uv_coords)
#         k = min(12, len(points))
#
#         # Determine hole threshold
#         valid_u_range = uv_coords[:, 0].max() - uv_coords[:, 0].min()
#         valid_v_range = uv_coords[:, 1].max() - uv_coords[:, 1].min()
#         expected_spacing = max(valid_u_range / res_u, valid_v_range / res_v)
#         hole_threshold = expected_spacing * 2.5
#
#
#         is_hole = np.zeros((res_u, res_v), dtype=bool)
#
#         for i, u in enumerate(u_vals):
#             for j, v in enumerate(v_vals):
#                 query = np.array([u, v])
#                 dists, idxs = uv_tree.query(query, k=k)
#
#                 if dists[0] > hole_threshold:
#                     is_hole[i, j] = True
#                 else:
#                     # Inverse distance weighting
#                     if dists[0] < 1e-10:
#                         weights = np.zeros(k)
#                         weights[0] = 1.0
#                     else:
#                         weights = 1.0 / (dists + 1e-8)
#                         weights /= weights.sum()
#
#                     grid_xyz[i, j] = np.sum(points[idxs] * weights[:, None], axis=0)
#                     if colors is not None:
#                         grid_rgb[i, j] = np.sum(colors[idxs] * weights[:, None], axis=0)
#                     else:
#                         grid_rgb[i, j] = 0.5
#
#         # Inpaint holes
#         grid_xyz = self._inpaint_holes(grid_xyz, is_hole)
#         grid_rgb = self._inpaint_holes(grid_rgb, is_hole)
#
#         # Clip colors
#         grid_rgb = np.clip(grid_rgb, 0, 1)
#
#         return grid_xyz, grid_rgb
#
#     def _inpaint_holes(self, grid: np.ndarray, is_hole: np.ndarray) -> np.ndarray:
#         """Fill holes by iteratively averaging valid neighbors."""
#         filled = grid.copy()
#         remaining_holes = is_hole.copy()
#
#         max_iterations = max(grid.shape[0], grid.shape[1])
#
#         for _ in range(max_iterations):
#             if not remaining_holes.any():
#                 break
#
#             valid_mask = ~np.isnan(filled[..., 0])
#             dilated_valid = binary_dilation(valid_mask)
#             fillable = remaining_holes & dilated_valid
#
#             if not fillable.any():
#                 # Fill remaining with global mean
#                 global_mean = np.nanmean(filled, axis=(0, 1))
#                 for i, j in zip(*np.where(remaining_holes)):
#                     filled[i, j] = global_mean
#                 break
#
#             for i, j in zip(*np.where(fillable)):
#                 neighbors = []
#                 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                     ni, nj = i + di, j + dj
#                     if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
#                         if valid_mask[ni, nj]:
#                             neighbors.append(filled[ni, nj])
#
#                 if neighbors:
#                     filled[i, j] = np.mean(neighbors, axis=0)
#                     remaining_holes[i, j] = False
#
#         return filled
#
#     def _fit_bspline_to_grid2(
#             self,
#             grid_xyz: np.ndarray,
#             grid_rgb: np.ndarray,
#             label: str,
#             res_u: int,
#             res_v: int
#     ) -> NURBSSurfaceData:
#         """Create B-spline surface from grid with smoothing."""
#         degree_u = min(self.config.degree_u, res_u - 1)
#         degree_v = min(self.config.degree_v, res_v - 1)
#
#         # Smooth
#         if self.config.smoothing > 0:
#             sigma = self.config.smoothing * min(res_u, res_v) / 10.0
#             for c in range(3):
#                 grid_xyz[..., c] = gaussian_filter(grid_xyz[..., c], sigma=sigma, mode='nearest')
#                 grid_rgb[..., c] = gaussian_filter(grid_rgb[..., c], sigma=sigma, mode='nearest')
#
#         knots_u = self._create_knot_vector(res_u, degree_u)
#         knots_v = self._create_knot_vector(res_v, degree_v)
#
#         return NURBSSurfaceData(
#             control_points=grid_xyz.astype(np.float32),
#             control_colors=np.clip(grid_rgb, 0, 1).astype(np.float32),
#             knots_u=knots_u.astype(np.float32),
#             knots_v=knots_v.astype(np.float32),
#             degree_u=degree_u,
#             degree_v=degree_v,
#             label=label
#
#         )
#
#
#     def _fit_bspline_to_grid(
#             self,
#             grid_xyz: np.ndarray,
#             grid_rgb: np.ndarray,
#             label: str,
#             res_u: int,
#             res_v: int,
#             is_spherical: bool = False
#     ) -> NURBSSurfaceData:
#         """Create B-spline surface from grid with smoothing."""
#         degree_u = min(self.config.degree_u, res_u - 1)
#         degree_v = min(self.config.degree_v, res_v - 1)
#
#         # Smooth
#         if self.config.smoothing > 0:
#             sigma = self.config.smoothing * min(res_u, res_v) / 10.0
#
#             # Use 'wrap' mode for U direction if spherical to ensure continuity at the seam
#             mode_u = 'wrap' if is_spherical else 'nearest'
#             mode_v = 'nearest'  # Poles usually don't wrap in V (phi)
#
#             # Smooth U
#             for c in range(3):
#                 grid_xyz[..., c] = gaussian_filter(grid_xyz[..., c], sigma=(0, sigma), mode=(mode_v, mode_u))
#                 grid_rgb[..., c] = gaussian_filter(grid_rgb[..., c], sigma=(0, sigma), mode=(mode_v, mode_u))
#
#             # Smooth V (usually 'nearest' or 'reflect')
#             for c in range(3):
#                 # Note: gaussian_filter handles multiple axes, but here split for clarity or specific sigmas
#                 # Applying V smoothing separately or combined:
#                 pass
#
#                 # Actually simpler to just call gaussian_filter once with mode sequence
#             for c in range(3):
#                 grid_xyz[..., c] = gaussian_filter(grid_xyz[..., c], sigma=sigma, mode=(mode_v, mode_u))
#                 grid_rgb[..., c] = gaussian_filter(grid_rgb[..., c], sigma=sigma, mode=(mode_v, mode_u))
#
#         knots_u = self._create_knot_vector(res_u, degree_u)
#         knots_v = self._create_knot_vector(res_v, degree_v)
#         return NURBSSurfaceData(
#             control_points=grid_xyz.astype(np.float32),
#             control_colors=np.clip(grid_rgb, 0, 1).astype(np.float32),
#             knots_u=knots_u.astype(np.float32),
#             knots_v=knots_v.astype(np.float32),
#             degree_u=degree_u,
#             degree_v=degree_v,
#             label=label
#
#         )
#
#
#     def _create_knot_vector(self, n_ctrl: int, degree: int) -> np.ndarray:
#         """Create clamped uniform knot vector."""
#         n_knots = n_ctrl + degree + 1
#         n_internal = n_knots - 2 * (degree + 1)
#
#         knots = np.zeros(n_knots)
#         knots[-degree - 1:] = 1.0
#
#         if n_internal > 0:
#             internal = np.linspace(0, 1, n_internal + 2)[1:-1]
#             knots[degree + 1:degree + 1 + n_internal] = internal
#
#         return knots
#
#
# # =============================================================================
# # Main Interface
# # =============================================================================
#
# class NURBSFromPointCloud:
#     """Main interface for creating NURBS surfaces from point clouds."""
#
#     def __init__(self, config: Optional[SurfaceConfig] = None):
#         self.config = config or SurfaceConfig()
#
#     def create_surfaces(
#             self,
#             points: Union[np.ndarray, torch.Tensor],
#             colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
#             mode: Optional[DecompositionMode] = None,
#             **kwargs
#     ) -> MultiSurfaceResult:
#         """
#         Create NURBS surface(s) from point cloud.
#         """
#         # Convert tensors
#         if isinstance(points, torch.Tensor):
#             points = points.detach().cpu().numpy()
#         if colors is not None and isinstance(colors, torch.Tensor):
#             colors = colors.detach().cpu().numpy()
#
#         # Update config
#         config = self._update_config(**kwargs)
#         if mode is not None:
#             config.decomposition_mode = mode
#
#         # Preprocess
#         processor = PointCloudProcessor(points, colors)
#
#         if config.outlier_removal:
#             raw_n = processor.n_points
#             clean_idx, _ = processor.remove_outliers(config.outlier_std_ratio, config.connectivity_k)
#             print(f"[NURBS] Preparing for outlier detection...\nTotal points before: {raw_n} points remain")
#             points = points[clean_idx]
#             colors = colors[clean_idx] if colors is not None else None
#             processor = PointCloudProcessor(points, colors)
#             print(f"[NURBS] Removed outliers: {len(clean_idx)} points remain")
#
#         print(f"[NURBS] Processing {processor.n_points} points")
#
#         # Decompose and fit
#         decomposer = SceneDecomposer(processor, config)
#         # decomposer = SurfaceAwareDecomposer(processor, config)
#         fitter = NURBSSurfaceFitter(config)
#
#         if config.decomposition_mode == DecompositionMode.SINGLE:
#             surfaces, labels = self._create_single_surface(processor, fitter)
#
#         elif config.decomposition_mode == DecompositionMode.BACKGROUND_OBJECT:
#             surfaces, labels = self._create_bg_object_surfaces(processor, decomposer, fitter)
#
#         # elif config.decomposition_mode == DecompositionMode.K_COMPONENTS:
#         #     surfaces, labels = self._create_k_component_surfaces(processor, decomposer, fitter)
#         elif config.decomposition_mode == DecompositionMode.K_COMPONENTS:
#             # decomposer = SurfaceAwareDecomposer(processor, config)
#             surfaces, labels = self._create_k_component_surfaces(
#                 processor, decomposer, fitter
#             )
#         elif config.decomposition_mode == DecompositionMode.ADAPTIVE:
#             surfaces, labels = decomposer._decompose_adaptive(processor, decomposer, fitter)
#         else:
#             raise ValueError(f"Unknown decomposition mode: {config.decomposition_mode}")
#
#         return MultiSurfaceResult(
#             surfaces=surfaces,
#             decomposition_mode=config.decomposition_mode,
#             labels=labels,
#             metadata={
#                 'n_input_points': len(points),
#                 'config': config
#             }
#         )
#
#     def _update_config(self, **kwargs) -> SurfaceConfig:
#         """Create updated config with overrides."""
#         config_dict = {}
#         for field_name in self.config.__dataclass_fields__:
#             config_dict[field_name] = kwargs.get(field_name, getattr(self.config, field_name))
#         return SurfaceConfig(**config_dict)
#
#     def _create_single_surface(
#             self,
#             processor: PointCloudProcessor,
#             fitter: NURBSSurfaceFitter,
#             sampling_resolution: Optional[Tuple[int, int]] = None
#     ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
#         """Create single surface for entire scene."""
#         complexity = processor.compute_local_complexity()
#         surface = fitter.fit_surface(
#             processor.points, processor.colors,
#             label="main", complexity=complexity
#         )
#         surface.point_indices = np.arange(processor.n_points)
#         labels = np.zeros(processor.n_points, dtype=np.int32)
#         return [surface], labels
#
#     def _compute_component_boundary(
#             self,
#             indices: np.ndarray,
#             graph: csr_matrix
#     ) -> np.ndarray:
#         """
#         Identify boundary points of a component.
#
#         These are points that have neighbors in other components.
#         Useful for:
#         - Adaptive sampling near boundaries
#         - Ensuring smooth transitions between surfaces
#         """
#         boundary_mask = np.zeros(len(indices), dtype=bool)
#
#         for local_idx, global_idx in enumerate(indices):
#             # Get all neighbors from full graph
#             neighbors = graph[global_idx].indices
#
#             # Check if any neighbor is NOT in this component
#             neighbor_set = set(neighbors)
#             component_set = set(indices)
#
#             if neighbor_set - component_set:  # Has external neighbors
#                 boundary_mask[local_idx] = True
#
#         return boundary_mask
#
#     def _get_component_resolution(
#             self,
#             component_type: str,  # 'background' or 'object'
#             base_resolution: Tuple[int, int],
#             config: SurfaceConfig
#     ) -> Tuple[int, int]:
#         """
#         Get resolution for a specific component type.
#
#         Priority:
#         1. Explicit resolution (bg_resolution / object_resolution)
#         2. Scale factor applied to base_resolution
#         3. Base resolution as fallback
#         """
#         if component_type == 'background':
#             if config.bg_resolution is not None:
#                 return config.bg_resolution
#             scale = config.bg_resolution_scale
#         elif component_type == 'object':
#             if config.object_resolution is not None:
#                 return config.object_resolution
#             scale = config.object_resolution_scale
#         else:
#             scale = 1.0
#
#         # Apply scale factor
#         res_u = int(base_resolution[0] * scale)
#         res_v = int(base_resolution[1] * scale)
#
#         # Clamp to valid range
#         res_u = int(np.clip(res_u, config.min_resolution, config.max_resolution))
#         res_v = int(np.clip(res_v, config.min_resolution, config.max_resolution))
#
#         return (res_u, res_v)
#
#     def _create_bg_object_surfaces(
#             self,
#             processor: PointCloudProcessor,
#             decomposer: SceneDecomposer,
#             fitter: NURBSSurfaceFitter,
#             base_resolution: Tuple[int, int] = None
#     ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
#         """Create separate surfaces for background and object with independent resolutions."""
#
#         bg_indices, obj_indices = decomposer.decompose_background_object()
#
#         surfaces = []
#         labels = np.zeros(processor.n_points, dtype=np.int32)
#
#         # Define per-component resolution
#         component_resolutions = {
#             "background": self.config.bg_resolution,
#             "object": self.config.object_resolution
#         }
#         component_scales = {
#             "background": self.config.bg_resolution_scale,
#             "object": self.config.object_resolution_scale
#         }
#
#         base_res = (self.config.base_resolution, self.config.base_resolution)
#
#         for idx_array, label_val, name in [
#             (bg_indices, 0, "background"),
#             (obj_indices, 1, "object")
#         ]:
#             if len(idx_array) >= 16:
#                 pts = processor.points[idx_array]
#                 cols = processor.colors[idx_array] if processor.colors is not None else None
#
#                 # Get component-specific resolution
#                 override_res = component_resolutions.get(name)
#                 if override_res is None:
#                     scale = component_scales.get(name, 1.0)
#                     override_res = (int(base_res[0] * scale), int(base_res[1] * scale))
#                 print(f"Resolution for {name}: {override_res[0]}x{override_res[1]}")
#                 try:
#                     surface = fitter.fit_surface(
#                         pts, cols,
#                         label=name,
#                         override_resolution=override_res  # Pass the override!
#                     )
#                     surface.point_indices = idx_array
#                     surfaces.append(surface)
#                     labels[idx_array] = label_val
#
#                     print(f"[NURBS] {name.capitalize()}: Final grid "
#                           f"{surface.control_points.shape[0]}x{surface.control_points.shape[1]}")
#
#                 except Exception as e:
#                     warnings.warn(f"Failed to fit {name} surface: {e}")
#
#         if len(surfaces) == 0:
#             warnings.warn("Failed to create separate surfaces, falling back to single")
#             return self._create_single_surface(processor, fitter)
#
#         return surfaces, labels
#
#     def _create_k_component_surfaces(
#             self,
#             processor: PointCloudProcessor,
#             decomposer: SceneDecomposer,
#             fitter: NURBSSurfaceFitter
#     ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
#         """Create separate surfaces for K connected components."""
#         component_indices = decomposer.decompose_k_components()
#         complexity = processor.compute_local_complexity()
#
#         surfaces = []
#         labels = np.full(processor.n_points, -1, dtype=np.int32)
#
#         for i, indices in enumerate(component_indices):
#             if len(indices) < 16:
#                 warnings.warn(f"Component {i} has too few points ({len(indices)}), skipping")
#                 continue
#
#             pts = processor.points[indices]
#             cols = processor.colors[indices] if processor.colors is not None else None
#             comp_complexity = complexity[indices]
#
#             try:
#                 surface = fitter.fit_surface(
#                     pts, cols,
#                     label=f"component_{i}",
#                     complexity=comp_complexity
#                 )
#                 surface.point_indices = indices
#                 surfaces.append(surface)
#                 labels[indices] = len(surfaces) - 1
#
#                 print(f"[NURBS] Component {i}:  {len(indices)} points -> "
#                       f"{surface.control_points.shape[0]}x{surface.control_points.shape[1]} grid")
#
#             except Exception as e:
#                 warnings.warn(f"Failed to fit surface for component {i}: {e}")
#
#         if len(surfaces) == 0:
#             warnings.warn("Failed to create any component surfaces, falling back to single")
#             return self._create_single_surface(processor, fitter)
#
#         return surfaces, labels
#
#
#
#
#
# # =============================================================================
# # Conversion Utilities
# # =============================================================================
#
# def nurbs_to_geomdl(surface_data: NURBSSurfaceData) -> BSpline.Surface:
#     """Convert NURBSSurfaceData to geomdl BSpline. Surface."""
#     surf = BSpline.Surface()
#     surf.degree_u = surface_data.degree_u
#     surf.degree_v = surface_data.degree_v
#
#     H, W, _ = surface_data.control_points.shape
#     ctrlpts = surface_data.control_points.reshape(-1, 3).tolist()
#
#     surf.set_ctrlpts(ctrlpts, H, W)
#     surf.knotvector_u = surface_data.knots_u.tolist()
#     surf.knotvector_v = surface_data.knots_v.tolist()
#
#     return surf
#
#
# def surfaces_to_torch(
#         result: MultiSurfaceResult,
#         device: str = 'cuda'
# ) -> Dict[str, Any]:
#     """
#     Convert MultiSurfaceResult to torch tensors for SplineModel initialization.
#     Returns lists since surfaces may have different resolutions.
#     """
#     surfaces = result.surfaces
#
#     cp_list = [torch.tensor(s.control_points, dtype=torch.float32, device=device) for s in surfaces]
#     cc_list = [torch.tensor(s.control_colors, dtype=torch.float32, device=device) for s in surfaces]
#     ku_list = [torch.tensor(s.knots_u, dtype=torch.float32, device=device) for s in surfaces]
#     kv_list = [torch.tensor(s.knots_v, dtype=torch.float32, device=device) for s in surfaces]
#
#     # NEW: Adaptive sampling
#     adaptive_u_list = []
#     adaptive_v_list = []
#     complexity_list = []
#
#     for s in surfaces:
#         if s.sampling_u_1D is not None:
#             adaptive_u_list.append(torch.tensor(s.sampling_u_1D, dtype=torch.float32, device=device))
#             adaptive_v_list.append(torch.tensor(s.sampling_v_1D, dtype=torch.float32, device=device))
#         else:
#             adaptive_u_list.append(None)
#             adaptive_v_list.append(None)
#
#         if s.complexity_map is not None:
#             complexity_list.append(torch.tensor(s.complexity_map, dtype=torch.float32, device=device))
#         else:
#             complexity_list.append(None)
#
#     return {
#         'control_points': cp_list,
#         'control_colors': cc_list,
#         'knots_u': ku_list,
#         'knots_v': kv_list,
#         'labels': torch.tensor(result.labels, dtype=torch.long, device=device),
#         'surface_labels': [s.label for s in surfaces],
#         'num_surfaces': len(surfaces),
#         # NEW
#         'adaptive_samples_u': adaptive_u_list,
#         'adaptive_samples_v': adaptive_v_list,
#         'complexity_maps': complexity_list,
#     }
#
# # =============================================================================
# # Convenience Functions
# # =============================================================================
#
# def create_nurbs_from_pointcloud(
#         points: Union[np.ndarray, torch.Tensor],
#         colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
#         resolution: Tuple[int, int] = (64, 64),
#         mode: DecompositionMode = DecompositionMode.SINGLE,
#         smoothing:  float = 0.05,
#         generate_adaptive_samples: bool = True,
#         sampling_resolution_factor: float = 1.0,
#         translate=None,
#         bg_resolution: Optional[Tuple[int, int]] =  None,
#         object_resolution:  Optional[Tuple[int, int]] =  None,
#         bg_resolution_scale: float = .5,
#         object_resolution_scale: float = 2.0,
#         parameterization: str = 'spherical', # Added explicit parameter
#         **kwargs
# ) -> MultiSurfaceResult:
#     """
#     Convenience function to create NURBS surface(s) from point cloud.
#     """
#     config = SurfaceConfig(
#         base_resolution=resolution[0],
#         resolution_u=resolution[0],
#         resolution_v=resolution[1],
#         smoothing=smoothing,
#         decomposition_mode=mode,
#         sampling_resolution_factor=sampling_resolution_factor,
#         bg_resolution=bg_resolution,
#         object_resolution=object_resolution,
#         bg_resolution_scale=bg_resolution_scale,
#         object_resolution_scale=object_resolution_scale,
#         parameterization=parameterization,  # Pass to config
#
#         **{k: v for k, v in kwargs.items() if hasattr(SurfaceConfig, k)}
#     )
#
#     creator = NURBSFromPointCloud(config)
#
#     return creator.create_surfaces(
#         points, colors, mode,
#         generate_adaptive_samples=generate_adaptive_samples,
#     )
#
#
# def initialize_spline_model_from_pointcloud(
#         points: Union[np.ndarray, torch.Tensor],
#         colors: Optional[Union[np.ndarray, torch.Tensor]],
#         config,  # NurbsOptimizationParams
#         args,  # Training args
#         resolution: Tuple[int, int] = (64, 64),
#         mode: DecompositionMode = DecompositionMode.SINGLE,
#         **kwargs
# ) -> Dict[str, Any]:
#     """
#     Create SplineModel initialization data from point cloud.
#
#     Returns dictionary suitable for SplineModel constructor.
#     """
#     result = create_nurbs_from_pointcloud(
#         points, colors,
#         resolution=resolution,
#         mode=mode,
#         degree=config.spline_degree[0] if hasattr(config, 'spline_degree') else 3,
#         **kwargs
#     )
#
#     # Convert to geomdl format
#     geomdl_surfaces = []
#     geomdl_surfaces_rgb = []
#
#     for surf_data in result.surfaces:
#         geo_surf = nurbs_to_geomdl(surf_data)
#         geomdl_surfaces.append(geo_surf)
#
#         # Create color surface
#         rgb_data = NURBSSurfaceData(
#             control_points=surf_data.control_colors,
#             control_colors=surf_data.control_colors,
#             knots_u=surf_data.knots_u,
#             knots_v=surf_data.knots_v,
#             degree_u=surf_data.degree_u,
#             degree_v=surf_data.degree_v,
#             label=surf_data.label + "_rgb"
#         )
#         rgb_surf = nurbs_to_geomdl(rgb_data)
#         geomdl_surfaces_rgb.append(rgb_surf)
#
#     torch_data = surfaces_to_torch(result)
#
#     return {
#         'surf': geomdl_surfaces,
#         'surf_rgb': geomdl_surfaces_rgb,
#         'torch_data': torch_data,
#         'result': result,
#         'num_surfaces': len(result.surfaces),
#         'decomposition_mode': mode.value,
#         'config': config,
#         'args': args,
#         **kwargs
#     }
#
# class AdaptiveSamplingGenerator:
#     """
#     Generates adaptive UV sampling coordinates based on surface complexity.
#
#     The generator analyzes the fitted NURBS control grid to identify regions
#     requiring higher sampling density and produces non-uniform UV coordinates
#     that concentrate samples in complex regions.
#     """
#
#     def __init__(self, config: AdaptiveSamplingConfig = None):
#         self.config = config or AdaptiveSamplingConfig()
#
#     def generate(
#             self,
#             control_points: np.ndarray,
#             control_colors: np.ndarray,
#             knots_u: np.ndarray,
#             knots_v: np.ndarray,
#             degree: int = 3
#     ) -> AdaptiveSamplingResult:
#         """
#         Generate adaptive UV sampling coordinates.
#
#         Returns both 1D intervals (separable) and 2D grids (full flexibility).
#         """
#         H, W, _ = control_points.shape
#         Us = int(self.config.sampling_resolution_factor * H)
#         Vs = int(self.config.sampling_resolution_factor * W)
#
#         # Step 1: Compute complexity map at control points
#         complexity_map = self._compute_complexity_map(control_points, control_colors)
#
#         # Step 2: Compute target density from complexity
#         density_map = self._complexity_to_density(complexity_map)
#
#         # Step 3: Upsample density to sampling resolution
#         density_samples = self._upsample_density(density_map, Us, Vs)
#
#         # Step 4a: Generate 1D intervals (marginal densities)
#         intervals_u, intervals_v = self._density_to_1d_intervals(
#             density_samples, knots_u, knots_v, degree
#         )
#
#         # Step 4b: Generate 2D grids (full density)
#         # grid_u, grid_v = self._density_to_2d_grids(
#         #     density_samples, knots_u, knots_v, degree
#         # )
#         grid_u, grid_v = None, None
#         # Step 5: Enforce monotonicity
#         if self.config.enforce_monotonic:
#             intervals_u = np.sort(intervals_u)
#             intervals_v = np.sort(intervals_v)
#             if grid_u is not None and grid_v is not None:
#                 grid_u, grid_v = self._enforce_2d_monotonicity(grid_u, grid_v)
#                 grid_u, grid_v = grid_u.astype(np.float32), grid_v.astype(np.float32)
#
#
#         return AdaptiveSamplingResult(
#             intervals_u=intervals_u.astype(np.float32),
#             intervals_v=intervals_v.astype(np.float32),
#             grid_u=grid_u,
#             grid_v=grid_v,
#             complexity_map=complexity_map.astype(np.float32),
#             density_map=density_samples.astype(np.float32)
#         )
#
#     def _density_to_1d_intervals(
#             self,
#             density: np.ndarray,  # [Us, Vs]
#             knots_u: np.ndarray,
#             knots_v: np.ndarray,
#             degree: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Generate 1D intervals by marginalizing the 2D density.
#
#         This creates separable sampling that concentrates in high-complexity
#         rows/columns.
#         """
#         Us, Vs = density.shape
#
#         # Get valid UV domain
#         u_min, u_max = knots_u[degree], knots_u[-(degree + 1)]
#         v_min, v_max = knots_v[degree], knots_v[-(degree + 1)]
#
#         # Marginal densities:  sum along the other dimension
#         # High-complexity rows -> higher marginal U density
#         marginal_u = density.sum(axis=1)  # [Us]
#         marginal_v = density.sum(axis=0)  # [Vs]
#
#         # Generate 1D samples using inverse CDF
#         intervals_u = self._inverse_cdf_sample(marginal_u, u_min, u_max)
#         intervals_v = self._inverse_cdf_sample(marginal_v, v_min, v_max)
#
#         return intervals_u, intervals_v
#
#     def _density_to_2d_grids(
#             self,
#             density: np.ndarray,  # [Us, Vs]
#             knots_u: np.ndarray,
#             knots_v: np.ndarray,
#             degree: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Generate full 2D grids using row-wise and column-wise inverse CDF.
#
#         This allows non-separable warping while maintaining row/column monotonicity.
#         """
#         Us, Vs = density.shape
#
#         # Get valid UV domain
#         u_min, u_max = knots_u[degree], knots_u[-(degree + 1)]
#         v_min, v_max = knots_v[degree], knots_v[-(degree + 1)]
#
#         grid_u = np.zeros((Us, Vs))
#         grid_v = np.zeros((Us, Vs))
#
#         # For U:  sample each column based on its density profile
#         for j in range(Vs):
#             col_density = density[:, j]
#             grid_u[:, j] = self._inverse_cdf_sample(col_density, u_min, u_max)
#
#         # For V: sample each row based on its density profile
#         for i in range(Us):
#             row_density = density[i, :]
#             grid_v[i, :] = self._inverse_cdf_sample(row_density, v_min, v_max)
#
#         return grid_u, grid_v
#
#     def _enforce_2d_monotonicity(
#             self,
#             grid_u: np.ndarray,
#             grid_v: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Enforce monotonicity in 2D grids.
#
#         U must be monotonic along axis 0 (rows increase in U)
#         V must be monotonic along axis 1 (cols increase in V)
#         """
#         Us, Vs = grid_u.shape
#         cfg = self.config
#
#         # Sort U along rows (axis 0)
#         grid_u_sorted = np.sort(grid_u, axis=0)
#
#         # Sort V along cols (axis 1)
#         grid_v_sorted = np.sort(grid_v, axis=1)
#
#         # Enforce minimum spacing
#         u_range = grid_u_sorted.max() - grid_u_sorted.min()
#         v_range = grid_v_sorted.max() - grid_v_sorted.min()
#         min_du = cfg.min_spacing_ratio * u_range / (Us - 1)
#         min_dv = cfg.min_spacing_ratio * v_range / (Vs - 1)
#
#         # Enforce spacing for U (along axis 0)
#         for j in range(Vs):
#             for i in range(1, Us):
#                 if grid_u_sorted[i, j] - grid_u_sorted[i - 1, j] < min_du:
#                     grid_u_sorted[i, j] = grid_u_sorted[i - 1, j] + min_du
#
#         # Enforce spacing for V (along axis 1)
#         for i in range(Us):
#             for j in range(1, Vs):
#                 if grid_v_sorted[i, j] - grid_v_sorted[i, j - 1] < min_dv:
#                     grid_v_sorted[i, j] = grid_v_sorted[i, j - 1] + min_dv
#
#         # Renormalize to valid domain
#         grid_u_sorted = self._renormalize_to_domain(grid_u_sorted)
#         grid_v_sorted = self._renormalize_to_domain(grid_v_sorted)
#
#         return grid_u_sorted, grid_v_sorted
#     def _compute_complexity_map(
#             self,
#             control_points: np.ndarray,
#             control_colors: np.ndarray
#     ) -> np.ndarray:
#         """
#         Compute per-control-point complexity combining multiple metrics.
#
#         Returns:
#             complexity:  [H, W] normalized complexity in [0, 1]
#         """
#         H, W, _ = control_points.shape
#
#         # Initialize component maps
#         curvature = np.zeros((H, W))
#         color_var = np.zeros((H, W))
#         normal_var = np.zeros((H, W))
#         edge_prox = np.zeros((H, W))
#
#         # 1. Discrete curvature via second differences
#         if H > 2 and W > 2:
#             # U direction second derivative
#             d2u = np.zeros_like(control_points)
#             d2u[1:-1] = control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
#
#             # V direction second derivative
#             d2v = np.zeros_like(control_points)
#             d2v[:, 1:-1] = control_points[:, 2:] - 2 * control_points[:, 1:-1] + control_points[:, :-2]
#
#             # Mixed derivative
#             d2uv = np.zeros_like(control_points)
#             d2uv[1:-1, 1:-1] = (
#                                        control_points[2:, 2:] - control_points[2:, :-2] -
#                                        control_points[:-2, 2:] + control_points[:-2, :-2]
#                                ) / 4.
#             0
#
#             # Approximate Gaussian curvature magnitude
#             curvature = np.linalg.norm(d2u, axis=-1) + np.linalg.norm(d2v, axis=-1)
#             curvature += 0.5 * np.linalg.norm(d2uv, axis=-1)
#
#         # 2. Color variance in local neighborhood
#         color_var = self._compute_local_variance(control_colors, kernel_size=3)
#
#         # 3. Normal variance (from tangent cross products)
#         normal_var = self._compute_normal_variance(control_points)
#
#         # 4. Edge proximity (distance to boundary in UV space)
#         edge_prox = self._compute_edge_proximity(H, W)
#
#         # Normalize each component to [0, 1]
#         def safe_normalize(x):
#             x_min, x_max = x.min(), x.max()
#             if x_max - x_min < 1e-8:
#                 return np.zeros_like(x)
#             return (x - x_min) / (x_max - x_min)
#
#         curvature = safe_normalize(curvature)
#         color_var = safe_normalize(color_var)
#         normal_var = safe_normalize(normal_var)
#         edge_prox = safe_normalize(edge_prox)
#
#         # Weighted combination
#         cfg = self.config
#         complexity = (
#                 cfg.weight_curvature * curvature +
#                 cfg.weight_color_variance * color_var +
#                 cfg.weight_normal_variance * normal_var +
#                 cfg.weight_edge_proximity * edge_prox
#         )
#
#         # Final normalization
#         total_weight = (
#                 cfg.weight_curvature + cfg.weight_color_variance +
#                 cfg.weight_normal_variance + cfg.weight_edge_proximity
#         )
#         complexity /= (total_weight + 1e-8)
#
#         return complexity.astype(np.float32)
#
#     def _compute_local_variance(
#             self,
#             data: np.ndarray,  # [H, W, C]
#             kernel_size: int = 3
#     ) -> np.ndarray:
#         """Compute local variance using a sliding window."""
#         H, W, C = data.shape
#         pad = kernel_size // 2
#
#         # Pad data
#         padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
#
#         variance = np.zeros((H, W))
#         for i in range(H):
#             for j in range(W):
#                 patch = padded[i:i + kernel_size, j:j + kernel_size, :]
#                 variance[i, j] = np.var(patch)
#
#         return variance
#
#     def _compute_normal_variance(self, control_points: np.ndarray) -> np.ndarray:
#         """Compute variance of surface normals in local neighborhood."""
#         H, W, _ = control_points.shape
#
#         # Compute tangent vectors
#         du = np.zeros_like(control_points)
#         dv = np.zeros_like(control_points)
#
#         du[:-1] = control_points[1:] - control_points[:-1]
#         du[-1] = du[-2]
#
#         dv[:, :-1] = control_points[:, 1:] - control_points[:, :-1]
#         dv[:, -1] = dv[:, -2]
#
#         # Compute normals via cross product
#         normals = np.cross(du, dv)
#         norms = np.linalg.norm(normals, axis=-1, keepdims=True)
#         norms = np.maximum(norms, 1e-8)
#         normals = normals / norms
#
#         # Compute variance of normals in 3x3 neighborhood
#         variance = self._compute_local_variance(normals, kernel_size=3)
#
#         return variance
#
#     def _compute_edge_proximity(self, H: int, W: int) -> np.ndarray:
#         """
#         Compute proximity to edges (higher at boundaries).
#         This helps ensure adequate sampling at surface boundaries.
#         """
#         # Distance from each edge
#         u_dist = np.minimum(
#             np.arange(H)[:, None],
#             np.arange(H - 1, -1, -1)[:, None]
#         )
#         v_dist = np.minimum(
#             np.arange(W)[None, :],
#             np.arange(W - 1, -1, -1)[None, :]
#         )
#
#         # Minimum distance to any edge
#         edge_dist = np.minimum(u_dist, v_dist).astype(float)
#
#         # Invert so boundary has high value
#         max_dist = min(H, W) / 2
#         edge_prox = 1.0 - (edge_dist / max_dist)
#
#         return np.broadcast_to(edge_prox, (H, W)).copy()
#
#     def _complexity_to_density(self, complexity: np.ndarray) -> np.ndarray:
#         """
#         Convert complexity map to target sampling density.
#
#         Higher complexity -> higher density (more samples per unit UV area).
#         """
#         cfg = self.config
#
#         # Smooth complexity to avoid abrupt density changes
#         if cfg.smoothing_sigma > 0:
#             complexity = gaussian_filter(complexity, sigma=cfg.smoothing_sigma)
#
#         # Map complexity [0, 1] to density ratio [min_ratio, max_ratio]
#         density_ratio = (
#                 cfg.min_density_ratio +
#                 complexity * (cfg.max_density_ratio - cfg.min_density_ratio)
#         )
#
#         return density_ratio.astype(np.float32)
#
#     def _upsample_density(
#             self,
#             density_ctrl: np.ndarray,  # [H, W]
#             Us: int,
#             Vs: int
#     ) -> np.ndarray:
#         """Upsample density map to sampling resolution."""
#         from scipy.ndimage import zoom
#
#         H, W = density_ctrl.shape
#         zoom_factors = (Us / H, Vs / W)
#
#         density_samples = zoom(density_ctrl, zoom_factors, order=1, mode='nearest')
#
#         # Ensure exact shape
#         density_samples = density_samples[: Us, :Vs]
#
#         return density_samples
#
#     def _density_to_uv(
#             self,
#             density: np.ndarray,  # [Us, Vs]
#             knots_u: np.ndarray,
#             knots_v: np.ndarray,
#             degree: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Convert density map to non-uniform UV coordinates.
#
#         Uses inverse CDF sampling:  higher density regions get more closely
#         spaced UV coordinates.
#         """
#         Us, Vs = density.shape
#
#         # Get valid UV domain from knots
#         u_min = knots_u[degree]
#         u_max = knots_u[-(degree + 1)]
#         v_min = knots_v[degree]
#         v_max = knots_v[-(degree + 1)]
#
#         # Generate samples along each direction using density-weighted CDF
#         samples_u = np.zeros((Us, Vs))
#         samples_v = np.zeros((Us, Vs))
#
#         # For U direction:  integrate density along rows
#         for j in range(Vs):
#             density_1d = density[:, j]
#             samples_u[:, j] = self._inverse_cdf_sample(density_1d, u_min, u_max)
#
#         # For V direction: integrate density along columns
#         for i in range(Us):
#             density_1d = density[i, :]
#             samples_v[i, :] = self._inverse_cdf_sample(density_1d, v_min, v_max)
#
#         return samples_u.astype(np.float32), samples_v.astype(np.float32)
#
#     def _inverse_cdf_sample(
#             self,
#             density: np.ndarray,  # [N]
#             val_min: float,
#             val_max: float
#     ) -> np.ndarray:
#         """
#         Generate non-uniform samples using inverse CDF of density.
#
#         Higher density -> samples closer together.
#         """
#         N = len(density)
#
#         # Normalize density to get PDF
#         density = density + 1e-8  # Avoid zeros
#         pdf = density / density.sum()
#
#         # Compute CDF
#         cdf = np.cumsum(pdf)
#         cdf = np.concatenate([[0], cdf])
#
#         # Uniform samples in [0, 1]
#         uniform = np.linspace(0, 1, N)
#
#         # Inverse CDF:  find positions where CDF equals uniform values
#         # This concentrates samples where density is high
#         positions = np.zeros(N)
#         for i, u in enumerate(uniform):
#             # Find interval in CDF
#             idx = np.searchsorted(cdf, u, side='right') - 1
#             idx = np.clip(idx, 0, N - 1)
#
#             # Linear interpolation within interval
#             if idx < N - 1:
#                 t = (u - cdf[idx]) / (cdf[idx + 1] - cdf[idx] + 1e-8)
#                 positions[i] = idx + t
#             else:
#                 positions[i] = N - 1
#
#         # Map to [val_min, val_max]
#         samples = val_min + (positions / (N - 1)) * (val_max - val_min)
#
#         return samples
#
#     def _enforce_monotonicity(
#             self,
#             samples_u: np.ndarray,
#             samples_v: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Ensure UV samples are strictly monotonic.
#
#         This is required for valid parameterization where:
#         - samples_u[: , j] is increasing for each column j
#         - samples_v[i, :] is increasing for each row i
#         """
#         Us, Vs = samples_u.shape
#         cfg = self.config
#
#         # Compute minimum spacing
#         u_range = samples_u.max() - samples_u.min()
#         v_range = samples_v.max() - samples_v.min()
#         min_du = cfg.min_spacing_ratio * u_range / (Us - 1)
#         min_dv = cfg.min_spacing_ratio * v_range / (Vs - 1)
#
#         # Sort and enforce minimum spacing for U
#         for j in range(Vs):
#             col = np.sort(samples_u[:, j])
#             for i in range(1, Us):
#                 if col[i] - col[i - 1] < min_du:
#                     col[i] = col[i - 1] + min_du
#             samples_u[:, j] = col
#
#         # Sort and enforce minimum spacing for V
#         for i in range(Us):
#             row = np.sort(samples_v[i, :])
#             for j in range(1, Vs):
#                 if row[j] - row[j - 1] < min_dv:
#                     row[j] = row[j - 1] + min_dv
#             samples_v[i, :] = row
#
#         # Renormalize to valid domain
#         samples_u = self._renormalize_to_domain(samples_u)
#         samples_v = self._renormalize_to_domain(samples_v)
#
#         return samples_u, samples_v
#
#     def _renormalize_to_domain(
#             self,
#             samples: np.ndarray,
#             eps: float = 1e-4
#     ) -> np.ndarray:
#         """Renormalize samples to [eps, 1-eps] domain."""
#         s_min, s_max = samples.min(), samples.max()
#         if s_max - s_min < 1e-8:
#             return np.linspace(eps, 1 - eps, samples.shape[0])[:, None] * np.ones_like(samples)
#
#         normalized = (samples - s_min) / (s_max - s_min)
#         return eps + normalized * (1 - 2 * eps)
