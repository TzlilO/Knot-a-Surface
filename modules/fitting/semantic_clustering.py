"""
Semantic Clustering Module using SAM 2

This module leverages the Segment Anything Model 2 (SAM 2) to segment scene objects
in 2D views and back-projects this semantic information to 3D point clouds.
It uses an affinity-based spectral clustering approach to ensure that points 
belonging to the same semantic object (mask) across multiple views are grouped together.
"""
import os

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import SpectralClustering, DBSCAN
from scipy.sparse import lil_matrix, csr_matrix

# SAM 2 Imports
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    raise ImportError(
        "SAM 2 not found. Please install via: pip install git+https://github.com/facebookresearch/sam2.git")


class SemanticClusteringModule:
    def __init__(
            self,
            model_cfg: str,
            checkpoint_path: str,
            device: str = "cuda",
            downsample_rate: float = 0.5,
            points_per_side: int = 32,
            pred_iou_thresh: float = 0.95,
            stability_score_thresh: float = 0.95
    ):
        self.device = device
        self.downsample_rate = downsample_rate

        print(f"[Semantic] Initializing SAM 2...")
        print(f"  - Checkpoint: {checkpoint_path}")

        # --- FIX FOR HYDRA CONFIG PATHS ---
        # If the user provides a full file path to the config, SAM 2's build_sam2 will fail
        # because it expects a package-relative path (e.g. "configs/sam2.1/sam2.1_hiera_l.yaml").

        # We try to deduce the correct internal name if it looks like a standard config
        if os.path.exists(model_cfg) or "/" in model_cfg:
            filename = os.path.basename(model_cfg)
            if "sam2.1" in filename or "sam2.1" in model_cfg:
                # Force the string that the library expects
                print(f"  - Detected local config path '{model_cfg}', converting to package reference.")
                if "l.yaml" in filename:
                    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                elif "b+.yaml" in filename:
                    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                elif "s.yaml" in filename:
                    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
                elif "t.yaml" in filename:
                    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        print(f"  - Config Name: {model_cfg}")
        # ----------------------------------

        # self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=points_per_side,  # Lower resolution (was 32), faster
            pred_iou_thresh=pred_iou_thresh,  # Only very confident masks
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=0,
            min_mask_region_area=2000,  # Ignore small details (was 100)

            use_m2m=True  # Disable mask-to-mask refinement for speed
        )


        print(f"[SemanticClustering] Loading SAM 2 from {checkpoint_path}...")
        # self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        #
        # # Configure the automatic mask generator
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     model=self.sam2_model,
        #     points_per_side=points_per_side,
        #     pred_iou_thresh=pred_iou_thresh,
        #     stability_score_thresh=stability_score_thresh,
        #     crop_n_layers=0,
        #     min_mask_region_area=100,
        # )
        # print("[SemanticClustering] SAM 2 Loaded Successfully.")

    def compute_affinity_matrix(
            self,
            points: np.ndarray,
            cameras: List,
            max_views: int = 30
    ) -> csr_matrix:
        """
        Constructs a sparse affinity matrix (N_points x N_points).
        Entry (i, j) is incremented if point i and point j share the same mask in a view.
        """
        n_points = len(points)
        # Use sparse matrix to handle large point clouds (LIL is fast for construction)
        affinity = lil_matrix((n_points, n_points), dtype=np.float32)

        # Select a subset of cameras to save time, preferably distributed around the scene
        if len(cameras) > max_views:
            step = len(cameras) // max_views
            selected_cameras = cameras[::step][:max_views]
        else:
            selected_cameras = cameras

        points_torch = torch.tensor(points, dtype=torch.float32, device=self.device)

        print(f"[SemanticClustering] Processing {len(selected_cameras)} views for semantic features...")

        for cam_idx, cam in enumerate(tqdm(selected_cameras)):
            # 1. Get Image
            # Assuming cam.original_image is [3, H, W] tensor in range [0, 1]
            if hasattr(cam, 'original_image'):
                img_tensor = cam.original_image
                # Convert to numpy uint8 [H, W, 3] for SAM
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                # Fallback if image not loaded in RAM
                pil_img = Image.open(cam.image_path)
                img_np = np.array(pil_img)

            # Resize if needed
            h, w = img_np.shape[:2]
            if self.downsample_rate != 1.0:
                new_h, new_w = int(h * self.downsample_rate), int(w * self.downsample_rate)
                img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                new_h, new_w = h, w

            # 2. Generate Masks with SAM 2
            # masks is a list of dicts: {'segmentation': [H, W] bool, 'area': int, ...}
            masks_data = self.mask_generator.generate(img_np)

            if len(masks_data) == 0:
                continue

            # Stack masks into a tensor: [N_masks, H, W]
            masks_stack = np.stack([m['segmentation'] for m in masks_data], axis=0)
            masks_tensor = torch.tensor(masks_stack, device=self.device, dtype=torch.bool)  # [K, H, W]

            # 3. Project 3D Points to 2D
            pts_cam = cam.world_to_camera(points_torch)  # [N, 3]

            # Check depth
            valid_z = pts_cam[..., 2] > cam.znear

            # Project
            pts_uv = cam.project_points(pts_cam)  # [N, 2] in original resolution

            # Adjust UV for downsampling
            if self.downsample_rate != 1.0:
                pts_uv = pts_uv * self.downsample_rate

            # Discretize
            u = pts_uv[..., 0].long()
            v = pts_uv[..., 1].long()

            # Check bounds
            valid_x = (u >= 0) & (u < new_w)
            valid_y = (v >= 0) & (v < new_h)
            valid_points = valid_z & valid_x & valid_y  # [N] boolean

            # Indices of valid points
            valid_indices = torch.nonzero(valid_points).squeeze()

            if valid_indices.numel() == 0:
                continue

            # Sample masks at point locations
            # u is x (col), v is y (row)
            # masks_tensor is [K, H, W] -> access via [:, v, u]

            u_valid = u[valid_indices]
            v_valid = v[valid_indices]

            # Extract mask ownership: [K, Num_Valid_Points]
            # point_mask_hits[k, p] = True if point p falls in mask k
            point_mask_hits = masks_tensor[:, v_valid, u_valid]

            # 4. Update Affinity
            # We iterate over each mask. If a mask covers P points, 
            # all P points are semantically related.

            # Convert to CPU for sparse matrix accumulation (SciPy is CPU based)
            point_mask_hits_cpu = point_mask_hits.cpu().numpy()  # [K, P]
            valid_indices_cpu = valid_indices.cpu().numpy()

            for k in range(point_mask_hits_cpu.shape[0]):
                mask_k_indices = np.where(point_mask_hits_cpu[k])[0]  # Indices local to valid_indices

                if len(mask_k_indices) < 2:
                    continue  # Single point mask provides no relation info

                # Get global point indices
                global_indices = valid_indices_cpu[mask_k_indices]

                # Heuristic: Downsample large masks (e.g. background) to avoid O(N^2) explosion
                if len(global_indices) > 500:
                    np.random.shuffle(global_indices)
                    global_indices = global_indices[:500]

                # Create all pairs (fully connected clique within the mask)
                # This explicitly links points inside the fruit
                # Using itertools.combinations is slow, use meshgrid
                g1, g2 = np.meshgrid(global_indices, global_indices)

                # We only take upper triangle to save time, or just dump into sparse
                # Flatten
                rows = g1.flatten()
                cols = g2.flatten()

                # Increment affinity
                # We interpret "being in the same mask" as +1 vote for similarity
                # Lil_matrix assignment is reasonably fast for batch updates if logic allows, 
                # but direct indexing one by one is slow. 
                # Better: Accumulate coordinates and create a temp CSR to add.

                data = np.ones(len(rows), dtype=np.float32)
                temp_csr = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

                # Add to main matrix (expensive but necessary)
                affinity += temp_csr

        return affinity.tocsr()

    def cluster_points(
            self,
            points: np.ndarray,
            cameras: List,
            n_clusters: int = 4
    ) -> np.ndarray:
        """
        Main entry point. Returns integer labels for each point.
        """
        # 1. Build Semantic Graph
        affinity_matrix = self.compute_affinity_matrix(points, cameras)

        # 2. Normalize Affinity (Symmetric)
        # Often helps Spectral Clustering
        # We can also add spatial proximity here if we want to enforce locality
        # affinity = alpha * semantic_affinity + (1-alpha) * geometric_affinity

        print(f"[SemanticClustering] Running Spectral Clustering for {n_clusters} classes...")

        # Note: Spectral Clustering is O(N^3) or O(N^2) depending on solver. 
        # For > 50k points, this will crash RAM.
        # Strategy for large clouds: 
        # 1. K-Means to get superpixels/superpoints (~1000-5000 clusters).
        # 2. Compute affinity between superpoints.
        # 3. Spectral cluster the superpoints.
        # 4. Project back.

        if len(points) > 10000:
            print("[SemanticClustering] Point cloud too large, using Superpoint reduction...")
            labels = self._cluster_large_cloud(points, affinity_matrix, n_clusters)
        else:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42,
                n_jobs=-1
            )
            labels = sc.fit_predict(affinity_matrix)

        return labels

    def _cluster_large_cloud(self, points, affinity_matrix, n_clusters):
        """
        Two-stage clustering for efficiency.
        1. Oversegment spatially (KMeans/MiniBatchKMeans).
        2. Semantic cluster the centroids.
        """
        from sklearn.cluster import MiniBatchKMeans

        # 1. Superpixels (Spatial only)
        n_super = 2000
        print(f"  -> Generating {n_super} spatial superpoints...")
        pre_clusterer = MiniBatchKMeans(n_clusters=n_super, batch_size=2048, random_state=42)
        super_labels = pre_clusterer.fit_predict(points)

        # 2. Aggregate Affinity Matrix
        # We need to shrink (N, N) -> (K, K)
        # A_super[i, j] = average affinity between points in superpoint i and superpoint j

        # Create a projection matrix P of shape (K, N) where P[k, i] = 1 if point i is in super k
        # Then A_super = P @ A @ P.T

        row = super_labels
        col = np.arange(len(points))
        data = np.ones(len(points))
        P = csr_matrix((data, (row, col)), shape=(n_super, len(points)))

        # Normalize P rows to compute average connection strength
        # P_norm = P / count
        counts = np.bincount(super_labels, minlength=n_super)
        counts[counts == 0] = 1
        P_norm = P.multiply(1.0 / counts[:, np.newaxis])

        print("  -> Compressing semantic graph...")
        # A_super = P_norm @ affinity_matrix @ P_norm.T
        # Note: affinity_matrix is sparse.
        A_super = P_norm.dot(affinity_matrix).dot(P_norm.T)

        # 3. Spectral Clustering on Superpoints
        print("  -> Spectral Clustering on superpoints...")
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        super_meta_labels = sc.fit_predict(A_super)

        # 4. Map back to points
        final_labels = super_meta_labels[super_labels]

        return final_labels


# ... (Previous content of SemanticClusteringModule class) ...

# =============================================================================
# Wrapper Function (Place this at the bottom of the file)
# =============================================================================

def semantic_decomposition(points, cameras, n_components, **kwargs):
    """
    Wrapper to be called from AdvancedDecomposer in nurbs_from_pointcloud.py
    """
    import torch

    # Extract config paths from kwargs, or use defaults
    # We use the kwargs passed from SurfaceConfig
    sam_config = kwargs.get('sam_config', "configs/sam2.1/sam2.1_hiera_l.yaml")
    sam_checkpoint = kwargs.get('sam_checkpoint', "checkpoints/sam2.1_hiera_large.pt")

    # Initialize the module
    clusterer = SemanticClusteringModule(
        model_cfg=sam_config,
        checkpoint_path=sam_checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # Adjusting these parameters helps with speed vs quality
        downsample_rate=0.5,  # 0.5 is faster, 1.0 is more accurate
        points_per_side=32,  # 32 is standard, 64+ for very fine details
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92
    )

    try:
        # Run the clustering pipeline
        labels = clusterer.cluster_points(points, cameras, n_clusters=n_components)
        return labels
    except Exception as e:
        print(f"[Semantic] Error during decomposition: {e}")
        # Fallback: return zeros (single cluster) if SAM fails
        import numpy as np
        return np.zeros(len(points), dtype=int)
    finally:
        # CRITICAL: Clean up GPU memory immediately
        del clusterer.sam2_model
        del clusterer.mask_generator
        del clusterer
        torch.cuda.empty_cache()
        import gc
        gc.collect()