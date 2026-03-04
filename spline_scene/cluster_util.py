import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Tuple


# --- 1. DINOv2 Model Loading ---

def load_dino_model(device: torch.device) -> Tuple[torch.nn.Module, int]:
    """
    Loads the DINOv2 model (ViT-Small) and moves it to the device.
    """
    print("Loading DINOv2 model (vits14)...")

    # Use ViT-Small (vits14) for a good balance of speed and power
    # Patch size is 14
    patch_size = 14
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    print("DINOv2 model loaded.")
    return model, patch_size


# --- 2. 2D Feature Extraction ---

def extract_2d_features(
        model: torch.nn.Module,
        image_paths: List[str],
        patch_size: int,
        device: torch.device
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """
    Runs DINOv2 on each image to get patch-based feature maps.

    Returns:
        - A list of (H_feat, W_feat, D) feature tensors.
        - A list of (H_new, W_new) resized image dimensions.
    """

    # DINOv2 normalization (standard ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    feature_maps = []
    resized_dims = []

    print(f"Extracting 2D features from {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in image_paths:
            pil_img = Image.open(img_path).convert('RGB')
            W_orig, H_orig = pil_img.size

            # DINOv2 needs image dimensions to be a multiple of patch_size
            H_new = int(np.ceil(H_orig / patch_size) * patch_size)
            W_new = int(np.ceil(W_orig / patch_size) * patch_size)

            # Create the transform for this specific image
            transform = transforms.Compose([
                transforms.Resize((H_new, W_new)),
                transforms.ToTensor(),
                normalize,
            ])

            img_tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, H_new, W_new)

            # Get patch features: (1, N, D)
            # N = number of patches, D = feature dimension (384 for vits14)
            features = model.forward_features(img_tensor)

            # (1, N, D) -> (1, D, N) -> (1, D, H_feat, W_feat)
            H_feat, W_feat = H_new // patch_size, W_new // patch_size
            features = features['x_norm_patchtokens']  # (1, N, D)

            # Reshape to (H_feat, W_feat, D)
            features = features.reshape(H_feat, W_feat, -1)

            feature_maps.append(features)
            resized_dims.append((H_new, W_new))

    print("2D feature extraction complete.")
    return feature_maps, resized_dims


# --- 3. 2D-to-3D Feature Lifting ---

def lift_features_to_3d(
        pcd: o3d.geometry.PointCloud,
        image_paths: List[str],
        intrinsics: List[o3d.camera.PinholeCameraIntrinsic],
        extrinsics: List[np.ndarray],
        feature_maps: List[torch.Tensor],
        resized_dims: List[Tuple[int, int]],
        patch_size: int,
        device: torch.device
) -> np.ndarray:
    """
    Projects 3D points into 2D views, samples features, and aggregates.

    Returns:
        An (N_points, D_features) numpy array of aggregated features.
    """

    N_points = len(pcd.points)
    D_features = feature_maps[0].shape[-1]  # e.g., 384
    points_np = np.asarray(pcd.points)
    points_hom = np.hstack((points_np, np.ones((N_points, 1))))  # (N, 4)

    # Accumulators on the GPU for efficient aggregation
    feature_sum = torch.zeros((N_points, D_features), device=device, dtype=torch.float32)
    feature_count = torch.zeros((N_points, 1), device=device, dtype=torch.float32)

    print("Lifting 2D features to 3D point cloud...")

    for i in range(len(image_paths)):
        K = intrinsics[i].intrinsic_matrix  # 3x3 intrinsic matrix
        T = np.linalg.inv(extrinsics[i])  # Extrinsic is T_world_cam, we need T_cam_world

        # Build projection matrix P = K @ [R|t]
        P = K @ T[:3, :]  # (3, 3) @ (3, 4) -> (3, 4)

        # --- 3a. Project all N points into this camera view ---

        # (3, 4) @ (4, N) -> (3, N)
        points_cam_hom = P @ points_hom.T

        # (3, N) -> (N, 3)
        points_cam_hom = points_cam_hom.T

        # De-homogenize: (u*z, v*z, z) -> (u, v, z)
        z = points_cam_hom[:, 2]
        u = points_cam_hom[:, 0] / z
        v = points_cam_hom[:, 1] / z

        # --- 3b. Scale UVs to match *resized* image ---

        W_orig, H_orig = intrinsics[i].width, intrinsics[i].height
        H_new, W_new = resized_dims[i]

        u_scaled = u * (W_new / W_orig)
        v_scaled = v * (H_new / H_orig)

        # --- 3c. Filter points that are visible in this view ---

        mask_z = z > 0
        mask_u = (u_scaled >= 0) & (u_scaled < W_new)
        mask_v = (v_scaled >= 0) & (v_scaled < H_new)
        visible_mask = mask_z & mask_u & mask_v

        if np.sum(visible_mask) == 0:
            continue

        # --- 3d. Get feature coordinates for visible points ---

        visible_indices = np.where(visible_mask)[0]
        u_visible = u_scaled[visible_mask]
        v_visible = v_scaled[visible_mask]

        # Convert pixel coordinates to feature map coordinates
        u_feat = (u_visible / patch_size).astype(int)
        v_feat = (v_visible / patch_size).astype(int)

        # --- 3e. Sample and aggregate features ---

        feature_map = feature_maps[i]  # (H_feat, W_feat, D)

        # Advanced indexing: sample features at (v_feat, u_feat) locations
        sampled_features = feature_map[v_feat, u_feat]

        # Convert indices to a tensor for GPU-based accumulation
        visible_indices_tensor = torch.tensor(visible_indices, device=device)

        # Add sampled features to the global sum
        feature_sum.index_add_(0, visible_indices_tensor, sampled_features)
        feature_count.index_add_(0, visible_indices_tensor, torch.ones_like(sampled_features[:, :1]))

    # --- 3f. Final Averaging ---

    # Avoid divide-by-zero for points that were not seen
    feature_count[feature_count == 0] = 1.0

    aggregated_features = feature_sum / feature_count

    print("Feature lifting complete.")
    return aggregated_features.cpu().numpy()


# --- 4. Clustering ---

def cluster_points(
        pcd: o3d.geometry.PointCloud,
        semantic_features: np.ndarray,
        n_clusters: int,
        w_spatial: float,
        w_color: float,
        w_semantic: float
) -> np.ndarray:
    """
    Combines features, normalizes, and runs K-Means clustering.

    Returns:
        An (N_points,) array of cluster labels.
    """
    print("Clustering points...")

    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)  # Assuming RGB [0,1]

    # --- 4a. Combine and Weight Features ---

    # Normalize each feature type independently
    points_scaled = StandardScaler().fit_transform(points_np)
    colors_scaled = StandardScaler().fit_transform(colors_np)
    semantic_scaled = StandardScaler().fit_transform(semantic_features)

    # Create the final weighted feature vector
    combined_features = np.hstack((
        points_scaled * w_spatial,
        colors_scaled * w_color,
        semantic_scaled * w_semantic
    ))

    # --- 4b. Run K-Means ---

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(combined_features)

    print(f"Clustering complete. Found {n_clusters} clusters.")
    return labels


# --- 5. Visualization ---

def visualize_clusters(pcd: o3d.geometry.PointCloud, labels: np.ndarray):
    """
    Colors the point cloud by cluster label and displays it.
    """

    n_clusters = len(np.unique(labels))
    # Get a "tab20" colormap
    colors = plt.get_cmap("tab20")(labels / (n_clusters - 1))

    # Update point cloud colors
    pcd_clustered = o3d.geometry.PointCloud()
    pcd_clustered.points = pcd.points
    pcd_clustered.colors = o3d.utility.Vector3dVector(colors[:, :3])

    print("Displaying original (left) and clustered (right) point clouds.")
    # Show original and clustered side-by-side
    pcd.translate((-2, 0, 0))  # Move original to the left
    o3d.visualization.draw_geometries(
        [pcd, pcd_clustered],
        window_name="Semantic 3D Clustering"
    )


# --- Main Execution ---

def main():
    # --- 0. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 💡 TUNE THESE PARAMETERS ---
    NUM_CLUSTERS = 10  # How many semantic surfaces to find
    NUM_IMAGES_TO_USE = 10  # Use more for better features, fewer for speed

    # Feature weights for clustering:
    W_SPATIAL = 1.0  # Importance of XYZ position (contiguity)
    W_COLOR = 1.0  # Importance of RGB color
    W_SEMANTIC = 3.0  # Importance of DINOv2 features
    # ---------------------------------

    # --- 1. Load Data (Open3D Redwood Dataset) ---
    print("Loading Redwood dataset...")
    # This dataset has aligned RGB, Depth, and Poses
    dataset = o3d.data.RedwoodRGBDDataset()

    image_paths = dataset.rgb_paths[:NUM_IMAGES_TO_USE]
    depth_paths = dataset.depth_paths[:NUM_IMAGES_TO_USE]
    intrinsics = [dataset.intrinsics] * NUM_IMAGES_TO_USE
    extrinsics = dataset.trajectory.poses[:NUM_IMAGES_TO_USE]

    # --- 2. Create Global Point Cloud ---
    # We build a single point cloud from the first image to project onto
    print("Creating global point cloud from first frame...")
    rgb_img = o3d.io.read_image(image_paths[0])
    depth_img = o3d.io.read_image(depth_paths[0])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img, depth_img, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics[0], extrinsics[0]
    )

    # Downsample for speed
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    print(f"Global point cloud created with {len(pcd.points)} points.")

    # --- 3. Run Pipeline ---
    model, patch_size = load_dino_model(device)

    feature_maps, resized_dims = extract_2d_features(
        model, image_paths, patch_size, device
    )

    aggregated_features = lift_features_to_3d(
        pcd,
        image_paths,
        intrinsics,
        extrinsics,
        feature_maps,
        resized_dims,
        patch_size,
        device
    )

    labels = cluster_points(
        pcd,
        aggregated_features,
        n_clusters=NUM_CLUSTERS,
        w_spatial=W_SPATIAL,
        w_color=W_COLOR,
        w_semantic=W_SEMANTIC
    )

    # --- 4. Visualize ---
    visualize_clusters(pcd, labels)


if __name__ == "__main__":
    main()