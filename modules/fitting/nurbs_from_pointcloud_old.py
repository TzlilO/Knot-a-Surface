

# """
# NURBS Surface Initialization from Point Clouds (Semantic & Geometric)
# 
# This module provides functionality to:
# 1. Create NURBS surfaces from unstructured point clouds
# 2. Separate scenes using SAM 2 semantic masks (Fruit A vs Fruit B)
# 3. Decompose scenes using Geometric heuristics (K-Means/DBSCAN)
# 
# Dependencies:
# - numpy, scipy, sklearn, torch
# - geomdl
# - sam2 (segment-anything-2)
# """
# 
# import numpy as np
# import torch
# from typing import Tuple, List, Optional, Dict, Union, Any
# from dataclasses import dataclass, field
# from enum import Enum
# import warnings
# import os
# 
# # Core dependencies
# from scipy.spatial.ckdtree import cKDTree
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
# 
# # NURBS library
# from geomdl import BSpline
# 
# # Internal Imports (Assumes these exist in your project structure)
# # from model.semantic_clustering import SemanticClusteringModule  # We will integrate the logic directly below to be self-contained
# from utils.graphics_utils import getWorld2View2, getProjectionMatrix # standard Gaussian Splatting utils
# 
# class DecompositionMode(Enum):
#     """Surface decomposition strategies."""
#     SINGLE = "single"  # Single surface for entire scene
#     BACKGROUND_OBJECT = "bg_object"  # Two surfaces: background + foreground
#     K_COMPONENTS = "k_components"  # K separate surfaces based on geometry
#     SEMANTIC = "semantic" # SAM 2 based clustering (Apple, Banana, etc.)
# 
# 
# class DecompositionMode(Enum):
#     SINGLE = "single"
#     BACKGROUND_OBJECT = "bg_object"
#     K_COMPONENTS = "k_components"
#     SEMANTIC = "semantic"
# 
# 
# @dataclass
# class SurfaceConfig:
#     """Configuration for NURBS surface fitting and Scene Decomposition."""
#     # --- Grid resolution limits ---
#     min_resolution: int = 64
#     max_resolution: int = 512
#     base_density: float = 200.0
# 
# 
#     # Clustering weights
#     weight_spatial: float = 1.0
#     weight_normal: float = 0.5
#     weight_color: float = 0.2
#     weight_visibility: float = 0.8  # Importance of camera visibility
# 
#     # --- NURBS parameters ---
#     degree_u: int = 3
#     degree_v: int = 3
#     smoothing: float = 0.05
#     overlap_margin: float = 0.02
# 
#     # --- Decomposition Mode ---
#     decomposition_mode: DecompositionMode = DecompositionMode.K_COMPONENTS
#     n_components: int = 4
# 
#     # --- SAM 2 Configuration (Flexible) ---
#     sam_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
# 
#     # Note: Use internal package name for config
#     sam_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
# 
#     # SAM 2 Generator Parameters
#     # Higher = more detailed masks, Slower
#     sam_points_per_side: int = 32
# 
#     # Higher = stricter quality required to keep a mask (0.0 to 1.0)
#     # Increase this if you get too many "messy" blobs.
#     sam_pred_iou_thresh: float = 0.88
# 
#     # Stability: Higher = requires mask to be stable across thresholding
#     sam_stability_score_thresh: float = 0.95
# 
#     # Minimum size of a mask (in pixels) to consider
#     sam_min_mask_region_area: int = 512
# 
#     # 0 = process whole image. 1 = process crops (slower but better for small objects)
#     sam_crop_n_layers: int = 0
#     downsample_rate = 0.5,  # 0.5 is faster, 1.0 is more accurate
# 
#     # --- Quality parameters ---
#     outlier_removal: bool = True
#     outlier_std_ratio: float = 2.0
# 
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
# 
# @dataclass
# class MultiSurfaceResult:
#     surfaces: List[NURBSSurfaceData]
#     decomposition_mode: DecompositionMode
#     labels: np.ndarray  # Per-point labels
#     metadata: Dict = field(default_factory=dict)
# 
# 
# # =============================================================================
# # Visibility Analysis
# # =============================================================================
# 
# class VisibilityAnalyzer:
#     """
#     Analyzes point cloud visibility against a set of cameras.
#     Returns a visibility feature vector for clustering.
#     """
# 
#     def __init__(self, cameras: List, device='cuda'):
#         self.cameras = cameras
#         self.device = device
# 
#     def compute_visibility_features(self, points: np.ndarray) -> np.ndarray:
#         """
#         Projects points to all cameras.
#         Returns [N, n_cams] binary visibility vector (or soft weights).
#         """
#         if not self.cameras:
#             return np.zeros((len(points), 1))
# 
#         n_points = len(points)
#         n_cams = len(self.cameras)
# 
#         # We process in batches to save VRAM
#         points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
#         vis_matrix = torch.zeros((n_points, n_cams), device=self.device)
# 
#         batch_size = 4
# 
#         with torch.no_grad():
#             for i in range(0, n_cams, batch_size):
#                 batch_cams = self.cameras[i:i + batch_size]
# 
#                 for j, cam in enumerate(batch_cams):
#                     # World to Camera
#                     pts_cam = cam.world_to_camera(points_tensor)
# 
#                     # Check Z > near plane
#                     valid_z = pts_cam[..., 2] > cam.znear
# 
#                     # Project to Image
#                     pts_uv = cam.project_points(pts_cam)
# 
#                     # Check bounds
#                     valid_x = (pts_uv[..., 0] >= 0) & (pts_uv[..., 0] < cam.image_width)
#                     valid_y = (pts_uv[..., 1] >= 0) & (pts_uv[..., 1] < cam.image_height)
# 
#                     visible = valid_z & valid_x & valid_y
#                     vis_matrix[:, i + j] = visible.float()
# 
#         return vis_matrix.cpu().numpy()
# 
# 
# class AdvancedDecomposer:
#     def __init__(self, processor: PointCloudProcessor, config: SurfaceConfig, cameras: List = None):
#         self.processor = processor
#         self.config = config
#         self.cameras = cameras
# 
#     def decompose(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#         """
#         Returns list of point indices for each cluster and their resolution suggestions.
#         """
#         points = self.processor.points
#         colors = self.processor.colors
#         normals = self.processor.estimate_normals()
# 
#         # 1. Construct High-Dim Feature Vector
#         # Normalize spatial coords
#         spatial_scale = np.linalg.norm(points.max(0) - points.min(0)) + 1e-6
#         feat_spatial = points / spatial_scale
# 
#         # Color in [0,1]
#         feat_color = colors
# 
#         # Normals are already unit
#         feat_normal = normals
# 
#         # Visibility features (Multi-view consistency)
#         if self.cameras:
#             vis_analyzer = VisibilityAnalyzer(self.cameras)
#             feat_vis = vis_analyzer.compute_visibility_features(points)
#             # Reduce dimensionality of vis vector if too many cameras
#             if feat_vis.shape[1] > 5:
#                 pca = PCA(n_components=5)
#                 feat_vis = pca.fit_transform(feat_vis)
#             # Normalize
#             feat_vis = feat_vis / (np.linalg.norm(feat_vis, axis=1, keepdims=True) + 1e-6)
#         else:
#             feat_vis = np.zeros((len(points), 1))
# 
#         # Weighted concatenation
#         features = np.concatenate([
#             feat_spatial * self.config.weight_spatial,
#             feat_normal * self.config.weight_normal,
#             feat_color * self.config.weight_color,
#             feat_vis * self.config.weight_visibility
#         ], axis=1)
# 
#         # 2. Perform Clustering
#         # Using KMeans for hard K counts, or DBSCAN/Agglomerative for automatic
#         if self.config.decomposition_mode == DecompositionMode.K_COMPONENTS:
#             print(f"Clustering into {self.config.n_components} components using multi-modal features...")
#             clusterer = KMeans(n_clusters=self.config.n_components, n_init=10, random_state=42)
#             labels = clusterer.fit_predict(features)
#         else:
#             # Fallback to simple DBSCAN on spatial only if mode not specified well
#             # But here we use the features
#             clusterer = KMeans(n_clusters=self.config.n_components, n_init=10)
#             labels = clusterer.fit_predict(features)
# 
#         # 3. Process Clusters & Infer Resolution
#         cluster_indices = []
#         resolutions = []
# 
#         for k in range(self.config.n_components):
#             mask = labels == k
#             if mask.sum() < 16: continue  # Skip noise
# 
#             indices = np.where(mask)[0]
# 
#             # --- Resolution Inference ---
#             sub_points = points[indices]
#             sub_normals = normals[indices]
#             sub_colors = colors[indices]
# 
#             # A. Geometric Complexity (PCA on points)
#             pca_geo = PCA(n_components=3)
#             pts_local = pca_geo.fit_transform(sub_points)
#             extent = pts_local.max(0) - pts_local.min(0)
# 
#             # B. Variance Complexity (Normals & Color)
#             # Higher variance in normals -> needs more control points
#             var_normal = np.var(sub_normals, axis=0).sum()
#             var_color = np.var(sub_colors, axis=0).sum()
#             complexity_score = 1.0 + (var_normal * 2.0) + (var_color * 0.5)
# 
#             # C. Determine H x W based on aspect ratio
#             # Assume U follows principal axis 0, V follows axis 1
#             aspect_ratio = extent[1] / (extent[0] + 1e-6)
# 
#             # Base count based on density
#             # Area approx
#             area = extent[0] * extent[1]
#             target_params = len(indices) / self.config.base_density
#             target_params = target_params * complexity_score
# 
#             target_dim = int(np.sqrt(target_params))
# 
#             res_u = int(target_dim)
#             res_v = int(target_dim * aspect_ratio)
# 
#             # Clamp
#             res_u = np.clip(res_u, self.config.min_resolution, self.config.max_resolution)
#             res_v = np.clip(res_v, self.config.min_resolution, self.config.max_resolution)
# 
#             cluster_indices.append(indices)
#             resolutions.append((res_u, res_v))
# 
#             print(f"  Cluster {k}: {len(indices)} pts | Complexity: {complexity_score:.2f} | Res: {res_u}x{res_v}")
# 
#         return cluster_indices, resolutions
# 
#     def expand_boundaries(self, cluster_indices: List[np.ndarray]) -> List[np.ndarray]:
#         """
#         Adds overlapping points to clusters to ensure continuity.
#         Finds k-nearest neighbors of boundary points in cluster A that belong to cluster B,
#         and adds them to A (temporarily for fitting).
#         """
#         if self.config.overlap_margin <= 0:
#             return cluster_indices
# 
#         points = self.processor.points
#         kdtree = self.processor.kdtree
# 
#         expanded_indices = []
# 
#         # Identify all points used
#         all_indices = np.concatenate(cluster_indices)
#         all_labels = np.full(len(points), -1)
#         for i, idx in enumerate(cluster_indices):
#             all_labels[idx] = i
# 
#         for i, indices in enumerate(cluster_indices):
#             # Find neighbors of these points in the global set
#             # Query range slightly larger than average density
#             # This is heuristic. A better way:
# 
#             # Simply include points that are spatially close but labeled differently
#             pts = points[indices]
# 
#             # Find neighbors within a small radius
#             # Radius approx: extent / 50
#             extent = pts.max(0) - pts.min(0)
#             radius = np.linalg.norm(extent) * self.config.overlap_margin
# 
#             neighbor_inds = kdtree.query_ball_point(pts, r=radius)
# 
#             # Flatten
#             candidates = np.unique(np.concatenate(neighbor_inds))
# 
#             # Keep indices that are EITHER in the current cluster OR in neighbor clusters
#             # (We don't want to add outlier noise, only points belonging to valid surfaces)
#             valid_mask = all_labels[candidates] != -1
#             final_indices = candidates[valid_mask]
# 
#             expanded_indices.append(final_indices.astype(np.int64))
# 
#         return expanded_indices
# 
# # =============================================================================
# # 1. Point Cloud Processor
# # =============================================================================
# 
# class PointCloudProcessor:
#     """Handles point cloud preprocessing and normal estimation."""
# 
#     def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
#         self.points = np.asarray(points, dtype=np.float64)
#         self.colors = np.asarray(colors, dtype=np.float64) if colors is not None else np.zeros_like(points)
#         self.n_points = len(points)
#         self._kdtree = None
# 
#     @property
#     def kdtree(self) -> cKDTree:
#         if self._kdtree is None:
#             self._kdtree = cKDTree(self.points)
#         return self._kdtree
# 
#     def estimate_normals(self, k=30):
#         if self.normals is not None:
#             return self.normals
# 
#         if self._kdtree is not None and self._kdtree.n != len(self.points):
#             self._kdtree = None
# 
#         normals = np.zeros_like(self.points)
#         dists, idx = self.kdtree.query(self.points, k=k)
# 
#         for i in range(len(self.points)):
#             neighbors = self.points[idx[i]]
#             # PCA
#             centered = neighbors - neighbors.mean(axis=0)
#             u, s, vh = np.linalg.svd(centered)
#             normals[i] = vh[2, :]  # Normal is eigenvector of smallest eigenvalue
# 
#         # Orient normals (simple check against center)
#         center = self.points.mean(axis=0)
#         dirs = self.points - center
#         flip = np.sum(normals * dirs, axis=1) < 0
#         normals[flip] = -normals[flip]
# 
#         self.normals = normals
#         return normals
#     def remove_outliers(self, std_ratio: float = 2.0, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
#         """Statistical outlier removal."""
#         distances, _ = self.kdtree.query(self.points, k=k)
#         mean_distances = distances.mean(axis=1)
#         global_mean = mean_distances.mean()
#         global_std = mean_distances.std()
# 
#         threshold = global_mean + std_ratio * global_std
#         mask = mean_distances < threshold
#         return np.where(mask)[0], mask
# 
# # =============================================================================
# # 2. Semantic Clustering (The "Brain" of the operation)
# # =============================================================================
# 
# 
# # =============================================================================
# # Semantic Analyzer (SAM 2 Integration)
# # =============================================================================
# 
# class SemanticAnalyzer:
#     """
#     Wraps SAM 2 to generate masks for cameras and aggregate 3D point labels.
#     """
# 
#     def __init__(self, cameras: List, config: SurfaceConfig, device='cuda'):
#         self.cameras = cameras
#         self.config = config
#         self.device = device
#         self.predictor = None  # Holds the SAM2ImagePredictor or Model
#         self.mask_generator = None
# 
#     def _init_sam(self):
#         """Lazy initialization of SAM 2 with Flexible Config."""
#         if self.mask_generator is not None:
#             return
# 
#         try:
#             from sam2.build_sam import build_sam2
#             from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# 
#             # --- Path Resolution Logic ---
#             import os
#             checkpoint_path = self.config.sam_checkpoint
#             if not os.path.exists(checkpoint_path):
#                 # Fallback: Check relative to this script file
#                 current_dir = os.path.dirname(os.path.abspath(__file__))
#                 alt_path = os.path.join(current_dir, "checkpoints", os.path.basename(checkpoint_path))
#                 if os.path.exists(alt_path):
#                     checkpoint_path = alt_path
# 
#             if not os.path.exists(checkpoint_path):
#                 raise FileNotFoundError(f"SAM 2 Checkpoint not found at {checkpoint_path}")
# 
#             print(f"[Semantic] Loading SAM 2 from {checkpoint_path}...")
#             print(f"  -> Points/Side: {self.config.sam_points_per_side}")
#             print(f"  -> IoU Thresh:  {self.config.sam_pred_iou_thresh}")
# 
#             # Load Model
#             sam2_model = build_sam2(self.config.sam_config, checkpoint_path, device=self.device,
#                                     apply_postprocessing=False)
# 
#             # Initialize Generator with User Config
#             self.mask_generator = SAM2AutomaticMaskGenerator(
#                 model=sam2_model,
#                 points_per_side=self.config.sam_points_per_side,
#                 pred_iou_thresh=self.config.sam_pred_iou_thresh,
#                 stability_score_thresh=self.config.sam_stability_score_thresh,
#                 downsample_rate=self.config.downsample_rate,
#                 crop_n_layers=self.config.sam_crop_n_layers,
#                 min_mask_region_area=self.config.sam_min_mask_region_area,
#             )
#         except ImportError:
#             raise ImportError("SAM 2 not installed. Please install facebook/segment-anything-2.")
# 
#     def clear(self):
#         """Releases GPU memory immediately."""
#         if self.mask_generator is not None:
#             print("[Semantic] Clearing SAM 2 from VRAM...")
#             del self.mask_generator
#             del self.predictor
#             self.mask_generator = None
#             self.predictor = None
#             import gc
#             gc.collect()
#             torch.cuda.empty_cache()
# 
#     def compute_semantic_labels(self, points: np.ndarray) -> np.ndarray:
#         self._init_sam()
#         n_points = len(points)
#         points_torch = torch.tensor(points, dtype=torch.float32, device=self.device)
# 
#         # We will collect "features" for every point.
#         # Feature vector = [Mask_ID_View_1, Mask_ID_View_2, ..., Mask_ID_View_K]
#         point_features = []
# 
#         # Subsample views for speed (every 10th frame)
#         stride = max(1, len(self.cameras) // 10)
#         selected_cameras = self.cameras[::stride]
# 
#         print(f"[Semantic] Extracting masks from {len(selected_cameras)} views...")
# 
#         for i, cam in enumerate(selected_cameras):
#             # 1. Get Image
#             img_tensor, _ = cam.get_image()
#             img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#             H, W = img_np.shape[:2]
# 
#             # 2. Run SAM 2 (Automatic Generation)
#             with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
#                 masks_data = self.mask_generator.generate(img_np)
# 
#             # Sort masks by area so small detailed masks (like fruit stems) are processed last/first
#             # (Here we just need unique IDs)
# 
#             # Create a "Label Map" for this 2D view.
#             # view_label_map[y, x] = Mask_ID
#             view_label_map = torch.full((H, W), -1, device=self.device, dtype=torch.int16)
# 
#             # Fill map: Smaller masks overwrite larger masks (to capture details)
#             masks_data.sort(key=lambda x: x['area'], reverse=True)
#             for mask_idx, mask_dict in enumerate(masks_data):
#                 m = torch.from_numpy(mask_dict['segmentation']).to(self.device)
#                 view_label_map[m] = mask_idx  # Assign integer ID
# 
#             # 3. Project 3D Points -> 2D Pixel Coordinates
#             pts_cam = cam.world_to_camera(points_torch)
#             z_vals = pts_cam[..., 2]
#             pts_uv = cam.project_points(pts_cam)
# 
#             u, v = pts_uv[:, 0].long(), pts_uv[:, 1].long()
# 
#             # Filter points visible in this camera
#             valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_vals > cam.znear)
# 
#             # 4. Sample the Mask ID for every visible point
#             # If point P projects to pixel (u,v), and that pixel belongs to Mask #5,
#             # then point P gets feature value "5" for this view.
#             point_labels_in_view = torch.full((n_points,), -1, device=self.device, dtype=torch.int16)
# 
#             if valid_mask.any():
#                 sampled_ids = view_label_map[v[valid_mask], u[valid_mask]]
#                 point_labels_in_view[valid_mask] = sampled_ids
# 
#             # Store this column of features
#             point_features.append(point_labels_in_view.float().unsqueeze(1))
# 
#         # ---------------------------------------------------------
#         # EXPLANATION OF THE "PRODUCT" PART
#         # ---------------------------------------------------------
#         # At this stage, 'point_features' is a list of N arrays.
#         # Combined, we have a matrix of shape [N_points, N_views].
#         # Row 'i' represents the "Semantic Signature" of point 'i'.
#         # Example: Point A: [Mask 1, Mask 4, Mask 2]
#         #          Point B: [Mask 1, Mask 4, Mask 2]
#         #          Point C: [Mask 9, Mask 8, Mask 7]
#         # Since A and B have the same signature (they fell into the same object masks across views),
#         # they should be clustered together.
# 
#         feature_stack = torch.cat(point_features, dim=1)  # [N, n_views]
# 
#         # Normalize features so ID magnitude doesn't matter (simple heuristic)
#         # or better, use them as categorical features.
#         # Here we normalize to 0-1 range to treat them as signals.
#         feature_stack = feature_stack / (feature_stack.max(dim=0)[0] + 1e-6)
# 
#         # Add spatial coordinates [X,Y,Z] to ensure points are also spatially connected
#         spatial = points_torch / (points_torch.max() - points_torch.min())
# 
#         # Final Feature Matrix for Clustering: [N, N_views + 3]
#         final_features = torch.cat([spatial * 1.5, feature_stack], dim=1).cpu().numpy()
# 
#         # ---------------------------------------------------------
#         # DBSCAN: Density-Based Spatial Clustering of Applications with Noise
#         # ---------------------------------------------------------
#         # What is the product?
#         # DBSCAN returns an array 'labels' of size [N_points].
#         # - Values are integers: -1, 0, 1, 2, ...
#         # - '-1' means the point is NOISE (didn't fit any cluster).
#         # - '0', '1'... are the Cluster IDs.
#         #
#         # Why DBSCAN? It doesn't require us to specify 'K' (number of fruits).
#         # It just finds dense regions in our "Semantic+Spatial" feature space.
# 
#         print("[Semantic] Clustering aggregated features (DBSCAN)...")
#         clusterer = DBSCAN(eps=0.05, min_samples=32, metric='euclidean', n_jobs=-1)
#         labels = clusterer.fit_predict(final_features)
# 
#         # Cleanup: Assign noise (-1) to nearest valid cluster using KNN
#         if -1 in labels:
#             known_mask = labels != -1
#             if known_mask.sum() > 0:
#                 from sklearn.neighbors import KNeighborsClassifier
#                 knn = KNeighborsClassifier(n_neighbors=1)
#                 knn.fit(points[known_mask], labels[known_mask])
# 
#                 noise_mask = labels == -1
#                 labels[noise_mask] = knn.predict(points[noise_mask])
# 
#         return labels
# 
# # =============================================================================
# # 3. Scene Decomposer
# # =============================================================================
# #
# # class SceneDecomposer:
# #     """Orchestrates the splitting of the scene."""
# #
# #     def __init__(self, processor: PointCloudProcessor, config: SurfaceConfig, cameras: List = None):
# #         self.processor = processor
# #         self.config = config
# #         self.cameras = cameras
# #
# #
# #     def decompose(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
# #         """
# #         Returns:
# #             List of point indices per cluster
# #             List of resolution tuples (res_u, res_v) per cluster
# #         """
# #         mode = self.config.decomposition_mode
# #
# #         if mode == DecompositionMode.SINGLE:
# #             labels = np.zeros(self.processor.n_points, dtype=int)
# #
# #         elif mode == DecompositionMode.SEMANTIC:
# #             if not self.cameras:
# #                 warnings.warn("Semantic mode requested but no cameras provided. Falling back to K-Components.")
# #                 return self._decompose_geometric()
# #
# #             analyzer = SemanticAnalyzer(self.cameras, self.config)
# #
# #             try:
# #                 labels = analyzer.compute_semantic_labels(self.processor.points)
# #             finally:
# #                 # CRITICAL: Always unload SAM 2 immediately after use
# #                 analyzer.clear()
# #
# #         elif mode == DecompositionMode.K_COMPONENTS:
# #             labels = self._decompose_geometric()
# #
# #         else: # Background/Object
# #             # Simple distance-based
# #             center = np.mean(self.processor.points, axis=0)
# #             dists = np.linalg.norm(self.processor.points - center, axis=1)
# #             threshold = np.percentile(dists, 80)
# #             labels = (dists > threshold).astype(int)
# #
# #
# #         # ---------------------------------------------------------
# #         # NEW STEP: Merge Oversegmented Clusters
# #         # ---------------------------------------------------------
# #         if self.config.n_components > 0 and mode == DecompositionMode.SEMANTIC:
# #             print(f"[Decomposer] reducing clusters (Target ~{self.config.n_components})...")
# #             labels = self._merge_clusters(self.processor.points, self.processor.colors, labels, target_k=self.config.n_components)
# #         # ---------------------------------------------------------
# #
# #         # Post-process labels to lists (Existing code)
# #         # Post-process labels to lists
# #         return self._labels_to_indices_and_res(labels)
# 
# # ... (Insert PointCloudProcessor class here) ...
# 
# class SceneDecomposer:
#     def __init__(self, processor, config: SurfaceConfig, cameras: List = None):
#         self.processor = processor
#         self.config = config
#         self.cameras = cameras
# 
#     def decompose(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
#         mode = self.config.decomposition_mode
#         points = self.processor.points
# 
#         if mode == DecompositionMode.SEMANTIC:
#             if not self.cameras:
#                 return self._decompose_geometric()
# 
#             print(f"[Decomposer] Starting SAM 2 Semantic Clustering...")
# 
#             # Init analyzer
#             analyzer = SemanticAnalyzer(self.cameras, self.config)
# 
#             try:
#                 # Get labels from DBSCAN
#                 labels = analyzer.compute_semantic_labels(points)
# 
#                 # OPTIONAL: Merge oversegmented clusters
#                 if self.config.n_components > 0:
#                     labels = self._merge_clusters(points, self.processor.colors, labels,
#                                                   target_k=self.config.n_components)
# 
#             except Exception as e:
#                 print(f"[Error] Semantic clustering failed: {e}. Falling back to geometric.")
#                 labels = self._decompose_geometric()
#             finally:
#                 # IMPORTANT: Free memory
#                 analyzer.clear()
# 
#         elif mode == DecompositionMode.K_COMPONENTS:
#             labels = self._decompose_geometric()
#         else:
#             labels = np.zeros(len(points), dtype=int)
# 
#         return self._labels_to_indices_and_res(labels)
# 
#         # ... (Include
# 
#     def _merge_clusters(self, points, colors, labels, target_k=8):
#         """
#         Iteratively merges closest clusters until target_k is reached
#         or no valid merges remain.
#         """
#         unique_labels = np.unique(labels)
#         unique_labels = unique_labels[unique_labels != -1]  # Ignore noise
# 
#         current_k = len(unique_labels)
#         print(f"  -> Initial cluster count: {current_k}")
# 
#         if current_k <= target_k:
#             return labels
# 
#         # Limit iterations to prevent infinite loops
#         max_iter = current_k - target_k + 50
# 
#         for i in range(max_iter):
#             unique_labels = np.unique(labels)
#             unique_labels = unique_labels[unique_labels != -1]
#             if len(unique_labels) <= target_k:
#                 break
# 
#             # 1. Compute Cluster Stats (Centroids & Color Means)
#             centroids = []
#             color_means = []
#             valid_lbls = []
# 
#             for lbl in unique_labels:
#                 mask = labels == lbl
#                 centroids.append(points[mask].mean(0))
#                 color_means.append(colors[mask].mean(0))
#                 valid_lbls.append(lbl)
# 
#             centroids = np.array(centroids)
#             color_means = np.array(color_means)
# 
#             # 2. Find Nearest Pair of Clusters
#             # We compute a distance matrix: D = SpatialDist + lambda * ColorDist
#             from sklearn.metrics.pairwise import euclidean_distances
#             d_spatial = euclidean_distances(centroids)
#             d_color = euclidean_distances(color_means)
# 
#             # Normalize distances
#             d_spatial /= (d_spatial.max() + 1e-6)
#             d_color /= (d_color.max() + 1e-6)
# 
#             # Cost function: mostly spatial (adjacency), but color prevents merging distinct objects
#             # High value = don't merge (diagonal is infinite)
#             cost_matrix = d_spatial + 0.5 * d_color
#             np.fill_diagonal(cost_matrix, np.inf)
# 
#             # Find min index
#             min_idx = np.argmin(cost_matrix)
#             r, c = np.unravel_index(min_idx, cost_matrix.shape)
# 
#             # 3. Merge
#             label_a = valid_lbls[r]
#             label_b = valid_lbls[c]
# 
#             # Update all points with label_b to have label_a
#             labels[labels == label_b] = label_a
# 
#             # Optimization: If the cost is too high, stop merging even if we haven't hit target_k
#             # (Prevent merging things that are far apart)
#             # The normalized cost threshold is heuristic (e.g. 0.2)
#             # if cost_matrix[r, c] > 0.5:
#             #     print(f"  -> Stopping early: Next merge cost {cost_matrix[r, c]:.3f} is too high.")
#             #     break
# 
#         print(f"  -> Final cluster count: {len(np.unique(labels)[np.unique(labels) != -1])}")
#         return labels
#     def _decompose_geometric(self):
#         print(f"[Decomposer] Geometric Clustering (K={self.config.n_components})...")
#         kmeans = KMeans(n_clusters=self.config.n_components, n_init=10, random_state=42)
#         return kmeans.fit_predict(self.processor.points)
# 
#     def _labels_to_indices_and_res(self, labels: np.ndarray):
#         unique_labels = np.unique(labels)
#         cluster_indices = []
#         resolutions = []
# 
#         for label in unique_labels:
#             if label == -1: continue # Skip noise if DBSCAN left it
# 
#             indices = np.where(labels == label)[0]
#             if len(indices) < 50: continue # Skip tiny clusters
# 
#             cluster_indices.append(indices)
# 
#             # Adaptive Resolution based on PCA aspect ratio
#             pts = self.processor.points[indices]
#             pca = PCA(n_components=2)
#             pca.fit(pts)
#             extents = pca.explained_variance_ratio_
#             aspect = extents[1] / (extents[0] + 1e-6)
# 
#             # Complexity heuristic: more points = higher res
#             base = int(np.sqrt(len(indices) / 2)) # e.g. 2000 pts -> 31
# 
#             res_u = np.clip(base, self.config.min_resolution, self.config.max_resolution)
#             res_v = np.clip(int(base * aspect), self.config.min_resolution, self.config.max_resolution)
# 
#             resolutions.append((res_u, res_v))
# 
#         return cluster_indices, resolutions
# 
#     def expand_boundaries(self, cluster_indices: List[np.ndarray]) -> List[np.ndarray]:
#         """Ensures overlapping boundaries for continuity."""
#         if self.config.overlap_margin <= 0:
#             return cluster_indices
# 
#         # Implementation using KDTree query for neighbors
#         # (Simplified for brevity, assumes kdtree is fast)
#         kdtree = self.processor.kdtree
#         points = self.processor.points
# 
#         expanded = []
#         for idx in cluster_indices:
#             pts_cluster = points[idx]
#             # Search slightly outside hull
#             # Here we just take the union of neighbors
#             # For a real robust impl, we'd convex hull or alpha shape.
#             # Naive: Query neighbors of boundary points? Expensive.
#             # Fast approx: Just return as is for now, relies on NURBS smoothing
#             expanded.append(idx)
# 
#         return expanded
# 
# # =============================================================================
# # 4. NURBS Fitter
# # =============================================================================
# 
# class NURBSSurfaceFitter:
#     def __init__(self, config: SurfaceConfig):
#         self.config = config
# 
#     def fit(self, points: np.ndarray, colors: np.ndarray, res: Tuple[int, int], label: str) -> NURBSSurfaceData:
#         # 1. Local Parameterization via PCA
#         pca = PCA(n_components=3)
#         pts_local = pca.fit_transform(points)
# 
#         # Normalize UV [0, 1]
#         uv_raw = pts_local[:, :2]
#         uv_min, uv_max = uv_raw.min(0), uv_raw.max(0)
#         uv = (uv_raw - uv_min) / (uv_max - uv_min + 1e-6)
#         uv = np.clip(uv, 0.01, 0.99) # Safe margins
# 
#         # 2. Inverse Distance Weighting to fill Grid
#         res_u, res_v = res
#         grid_u, grid_v = np.meshgrid(np.linspace(0, 1, res_u), np.linspace(0, 1, res_v), indexing='ij')
#         grid_uv = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 2)
# 
#         from scipy.interpolate import griddata
#         # Linear interpolation is faster/cleaner than IDW loop for griddata
#         # Note: griddata can define convex hull issues, fill with nearest
#         grid_xyz = griddata(uv, points, grid_uv, method='linear')
# 
#         # Fill NaNs (holes) with nearest
#         mask = np.isnan(grid_xyz).any(axis=1)
#         if mask.any():
#             from scipy.interpolate import NearestNDInterpolator
#             interp_near = NearestNDInterpolator(uv, points)
#             grid_xyz[mask] = interp_near(grid_uv[mask])
# 
#         grid_xyz = grid_xyz.reshape(res_u, res_v, 3)
# 
#         # Colors
#         grid_rgb = griddata(uv, colors, grid_uv, method='nearest').reshape(res_u, res_v, 3)
# 
#         # 3. Smoothing
#         if self.config.smoothing > 0:
#             from scipy.ndimage import gaussian_filter
#             for c in range(3):
#                 grid_xyz[..., c] = gaussian_filter(grid_xyz[..., c], sigma=self.config.smoothing)
#                 grid_rgb[..., c] = gaussian_filter(grid_rgb[..., c], sigma=self.config.smoothing)
# 
#         # 4. Knots
#         knots_u = self._generate_knots(res_u, self.config.degree_u)
#         knots_v = self._generate_knots(res_v, self.config.degree_v)
# 
#         return NURBSSurfaceData(
#             control_points=grid_xyz,
#             control_colors=np.clip(grid_rgb, 0, 1),
#             knots_u=knots_u,
#             knots_v=knots_v,
#             degree_u=min(self.config.degree_u, res_u - 1),
#             degree_v=min(self.config.degree_v, res_v - 1),
#             label=label,
#             point_indices=None # assigned later
#         )
# 
#     def _generate_knots(self, n_ctrl, degree):
#         n_knots = n_ctrl + degree + 1
#         knots = np.zeros(n_knots)
#         knots[-degree - 1:] = 1.0
#         n_mid = n_knots - 2 * (degree + 1)
#         if n_mid > 0:
#             knots[degree + 1: degree + 1 + n_mid] = np.linspace(0, 1, n_mid + 2)[1:-1]
#         return knots
# 
# # =============================================================================
# # 5. Main Interface
# # =============================================================================
# 
# class NURBSFromPointCloud:
#     def __init__(self, config: Optional[SurfaceConfig] = None):
#         self.config = config or SurfaceConfig()
# 
#     def create_surfaces(
#             self,
#             points: Union[np.ndarray, torch.Tensor],
#             colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
#             cameras: List = None,
#             mode: Optional[DecompositionMode] = None,
#             **kwargs
#     ) -> MultiSurfaceResult:
# 
#         # Standardize Inputs
#         if isinstance(points, torch.Tensor): points = points.detach().cpu().numpy()
#         if isinstance(colors, torch.Tensor): colors = colors.detach().cpu().numpy()
# 
#         # Override Config
#         if mode: self.config.decomposition_mode = mode
#         for k, v in kwargs.items():
#             if hasattr(self.config, k):
#                 setattr(self.config, k, v)
# 
#         print(f"[NURBS] Input Points: {len(points)}. Mode: {self.config.decomposition_mode.value}")
# 
#         # 1. Processing
#         processor = PointCloudProcessor(points, colors)
#         if self.config.outlier_removal:
#             clean_idx, _ = processor.remove_outliers(self.config.outlier_std_ratio)
#             processor.points = processor.points[clean_idx]
#             processor.colors = processor.colors[clean_idx]
#             print(f"[NURBS] Outliers removed. Remaining: {len(processor.points)}")
# 
#         # 2. Decomposition
#         decomposer = AdvancedDecomposer(processor, self.config, cameras)
# 
#         # decomposer = SceneDecomposer(processor, self.config, cameras)
#         idx_list, res_list = decomposer.decompose()
#         idx_list = decomposer.expand_boundaries(idx_list) # Continuity
# 
#         # 3. Fitting
#         fitter = NURBSSurfaceFitter(self.config)
#         surfaces = []
# 
#         point_labels = np.full(len(points), -1, dtype=int)
# 
#         # Map cleaned indices back to original if needed, but for now we just label clean ones
#         # This mapping is approximate if outliers were removed.
# 
#         for i, (indices, resolution) in enumerate(zip(idx_list, res_list)):
#             label_name = f"cluster_{i}"
# 
#             # Extract data
#             sub_points = processor.points[indices]
#             sub_colors = processor.colors[indices]
# 
#             try:
#                 surf = fitter.fit(sub_points, sub_colors, resolution, label_name)
#                 surf.point_indices = indices # Local clean indices
#                 surfaces.append(surf)
#                 print(f"[NURBS] Fitted Surface {i}: Res {resolution}, Pts {len(sub_points)}")
#             except Exception as e:
#                 print(f"[NURBS] Failed to fit cluster {i}: {e}")
# 
#         return MultiSurfaceResult(
#             surfaces=surfaces,
#             decomposition_mode=self.config.decomposition_mode,
#             labels=point_labels # Placeholder
#         )
# 
# # =============================================================================
# # Convenience Function
# # =============================================================================
# 
# def create_nurbs_from_pointcloud(
#         points: Union[np.ndarray, torch.Tensor],
#         colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
#         cameras: List = None,
#         mode: DecompositionMode = DecompositionMode.SEMANTIC,
#         **kwargs
# ) -> MultiSurfaceResult:
#     """Entry point for external calls."""
#     config = SurfaceConfig(
#     sam_points_per_side=128,   # Higher = finer granularity (fixes "mishmash" if objects are small)
#     sam_pred_iou_thresh=0.95, # Higher = stricter (ignores fuzzy/bad masks)
#     sam_stability_score_thresh=0.96,
# )
# 
# 
#     creator = NURBSFromPointCloud(config)
#     return creator.create_surfaces(points, colors, cameras, mode, **kwargs)