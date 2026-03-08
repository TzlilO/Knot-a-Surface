#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import gc
import json
import random
import os


from arguments import ModelParams, NurbsOptimizationParams
from modules.multisurf import MultiSurfaceSplineModel
from modules.fitting.nurbs_from_pointcloud import DecompositionMode
# from modules.uv_consistency import MultiViewPrecomputedUVLoss, PrecomputedCorrespondences, get_neighbor_cameras, \
#     CorrespondencePreprocessor

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration


def initialize_nurbs_surface(cameras, scene_radius, num_patches_rows, num_patches_cols):
    """
    Initialize a 3D NURBS surface control net from a list of cameras.
    The procedure unprojects (for each camera) the image center to world coordinates at a distance of scene_radius*0.1.
    Then, using the union of these world–space points, it builds a bounding box. Over that bounding box, a
    uniform grid of control points is created, which will serve as the control net for a cubic NURBS surface.

    For a cubic NURBS surface with clamped knots and degree 3, if you wish to have num_patches_rows x num_patches_cols patches,
    then the underlying control net is of shape ((num_patches_rows+3) x (num_patches_cols+3)).

    Args:
        cameras (list): List of camera objects. For each camera, the following attributes are assumed:
            - image_width, image_height (ints)
            - fx, fy (focal lengths)
            - cx, cy (principal point coordinates)
            - world_view_transform: a 4x4 matrix (torch.Tensor or convertible to numpy array)
        scene_radius (float): A scene-scale parameter. The unprojection depth is computed as scene_radius*0.1.
        num_patches_rows (int): Desired number of patches in the vertical direction.
        num_patches_cols (int): Desired number of patches in the horizontal direction.

    Returns:
        control_points (torch.Tensor): A tensor of shape ((num_patches_rows+3)*(num_patches_cols+3), 3)
            representing the flattened control net.
        control_net_shape (tuple): (num_net_rows, num_net_cols) = (num_patches_rows+3, num_patches_cols+3)
    """

    def control_net_to_patches(control_net, net_shape):
        """
        Convert a control net into patches for a cubic NURBS surface.

        Args:
            control_net (torch.Tensor): A tensor of shape (net_rows * net_cols, 3)
                                        representing the flattened control net.
            net_shape (tuple): (net_rows, net_cols) of the control net.

        Returns:
            patches (torch.Tensor): A tensor of shape ((net_rows-3)*(net_cols-3), 4, 4, 3)
                                    containing the 3D control points for each patch.
        """
        net_rows, net_cols = net_shape
        # Reshape the flat control net into a 2D grid with shape (net_rows, net_cols, 3)
        control_net_grid = control_net.reshape(net_rows, net_cols, 3)

        patches_list = []
        # The number of patches in the vertical direction is (net_rows - 3)
        # and in the horizontal direction is (net_cols - 3)
        for i in range(net_rows - 3):
            for j in range(net_cols - 3):
                # Extract a 4x4 block of control points
                patch = control_net_grid[i:i + 4, j:j + 4, :]  # shape (4, 4, 3)
                patches_list.append(patch)
        # Stack all patches along a new dimension.
        patches = torch.stack(patches_list, dim=0)
        return patches

    depth = scene_radius * 0.1
    camera_points = []
    cam_centers =[]
    for cam in cameras:
        # Compute the image center.
        cx_img = cam.camera_center[0].cpu()
        cy_img = cam.camera_center[1].cpu()
        ctr = [cx_img, cy_img, cam.camera_center[2].cpu()]
        cam_centers.append(ctr)
        # We assume camera intrinsics: fx, fy, cx, cy.
        # Compute normalized image coordinates for the image center.
        # (Typically: x_normalized = (u - cx)/fx)
        x_norm = (cx_img - cam.cx) / cam.fx
        y_norm = (cy_img - cam.cy) / cam.fy

        # In camera coordinates, a point with these normalized coordinates at depth 'depth' is:
        p_cam = np.array([x_norm * depth, y_norm * depth, depth, 1.0])

        # To convert p_cam to world coordinates, we need to apply the inverse of the camera transform.
        # Assume cam.world_view_transform is a 4x4 matrix (if it's a torch.Tensor, convert it to np).
        if isinstance(cam.world_view_transform, torch.Tensor):
            wv = cam.world_view_transform.cpu().numpy()
        else:
            wv = np.asarray(cam.world_view_transform)
        # Typically, world_view_transform transforms world -> camera coordinates.
        # So, we compute camera-to-world transform as the inverse of the transpose:
        c2w = np.linalg.inv(wv)  # adjust if your convention differs
        p_world = c2w @ p_cam
        camera_points.append(p_world[:3] )#/ p_world[3])
    camera_points = np.stack(camera_points, axis=0)  # shape: (num_cameras, 3)
    cam_centers = np.stack(cam_centers, axis=0)  # shape: (num_cameras, 2)
    visualize_unprojection(camera_points, cam_centers)

    # Compute bounding box of these camera points.
    min_pt = camera_points.min(axis=0)
    max_pt = camera_points.max(axis=0)

    # Create a uniform grid over the bounding box.
    # The control net size for a cubic surface with (num_patches_rows x num_patches_cols) patches is:
    num_net_rows = num_patches_rows + 3
    num_net_cols = num_patches_cols + 3

    # Create grid coordinates for x and y.
    xs = np.linspace(min_pt[0], max_pt[0], num_net_cols)
    ys = np.linspace(min_pt[1], max_pt[1], num_net_rows)

    # For z, you could either sample uniformly or, more simply, use the average depth.
    avg_z = np.mean(camera_points[:, 2])
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = np.full_like(grid_x, fill_value=avg_z)

    control_net = np.stack([grid_x, grid_y, grid_z], axis=-1)  # shape: (num_net_rows, num_net_cols, 3)
    control_net_flat = control_net.reshape(-1, 3)

    # Convert to torch.Tensor.
    control_points_tensor = torch.tensor(control_net_flat, dtype=torch.float32)
    patches = control_net_to_patches(control_points_tensor, (num_net_rows, num_net_cols))

    return patches.to('cuda'), control_points_tensor.to('cuda'), (num_net_rows, num_net_cols)

import matplotlib.pyplot as plt


def visualize_unprojection(camera_points, cam_centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cam_centers[:,0], cam_centers[:,1], cam_centers[:,2], c='red', marker='^', label="Cameras")
    ax.scatter(camera_points[:,0], camera_points[:,1], camera_points[:,2], c='blue', marker='o', label="Unprojected")
    ax.legend()
    plt.show()

class SplineScene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, config: NurbsOptimizationParams, load_iteration=None, resolution_scales=[1.0], shuffle=True,
                 late_init=False, **kwargs):

        """b
        :param late_init:
        :param path: Path to colmap scene main folder.
        """

        # === NEW:  Correspondence storage ===
        # self.correspondences: PrecomputedCorrespondences = None
        # self.uv_consistency_loss: MultiViewPrecomputedUVLoss = None

        self.train_cameras = {}
        self.test_cameras = {}
        self.source_path = args.source_path
        self.model_path = args.model_path
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.loaded_iter = None
        self.pcd = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            print("Could not find known scene file types in source path:", args.source_path)
            print("Scene Name: {}")
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.pcd = scene_info.point_cloud
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.multi_view_num = args.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # self.train_cameras[resolution_scale] = train_cams[::2]
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("computing nearest_id")
            self.world_view_transforms = []
            camera_centers = []
            center_rays = []
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                self.world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
                center_ray = center_ray @ R.transpose(-1, -2)
                center_rays.append(center_ray)
            self.world_view_transforms = torch.stack(self.world_view_transforms)
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
            diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
            tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
            angles = torch.arccos(tmp) * 180 / 3.14159
            angles = angles.detach().cpu().numpy()
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                    sorted_indices = np.lexsort((angles[id], diss[id]))
                    # sorted_indices = np.lexsort((diss[id], angles[id]))
                    mask = (angles[id][sorted_indices] < args.multi_view_max_angle) & \
                           (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                           (diss[id][sorted_indices] < args.multi_view_max_dis)
                    sorted_indices = sorted_indices[mask]
                    multi_view_num = min(self.multi_view_num, len(sorted_indices))
                    json_d = {'ref_name': cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        cur_cam.nearest_id.append(index)
                        cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                        json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                    json_str = json.dumps(json_d, separators=(',', ':'))
                    file.write(json_str)
                    file.write('\n')

        DECOMPOSITIONS = {
            "single": DecompositionMode.SINGLE,
            "kcc": DecompositionMode.K_COMPONENTS,
            # "adaptive": DecompositionMode.ADAPTIVE,
            "background": DecompositionMode.BACKGROUND_OBJECT
        }
        # Get point cloud data
        # pcd = self.pcd.points  # ndarray (N,3)
        # pcd_rgb = self.pcd.colors  # ndarray (N,3)
        pcd = self.pcd.points
        rgb = self.pcd.colors
        # dists = distCUDA2(torch.from_numpy(np.asarray(pcd)).float().cuda())

        # lower = np.percentile(dists.cpu().numpy(), q=.)
        # upper = np.percentile(dists.cpu().numpy(), q=99)

        # density_mask = (dists.cpu() < upper) #& (dists.cpu() > lower)
        # pcd = pcd[density_mask.numpy()]
        # Filter outliers out of nerf normalization radius:
        # center = torch.from_numpy(np.asarray(scene_info.nerf_normalization['translate'])).float().cuda()
        # pcd_tensor = torch.from_numpy(pcd).float().cuda()
        # dist_to_center = torch.norm(pcd_tensor - center, dim=-1)
        # radius_mask = dist_to_center < self.cameras_extent * 3
        # pcd = pcd[radius_mask.cpu().numpy()]
        # rgb = rgb[radius_mask.cpu().numpy()]



        decomposition_mode = DECOMPOSITIONS[config.decomposition_mode]
        pcd, pcd_rgb, inlier_mask = remove_outliers_for_nurbs(
            pcd,
            rgb,
            decomposition_mode=decomposition_mode,
            k_neighbors=8,
            std_ratio=2.0,
            min_cluster_size=64,
        )

        # decomposition_mode = DECOMPOSITIONS[config.decomposition_mode]
        connectivity_radius = 0.005
        n_components = 4
        smoothing = 0.01
        base_res=  config.base_res

        DOWN_FACTOR = {
            "single": 1,
            "kcc": 2,
            "adaptive": 1,
            "background": 2
        }
        down_factor = DOWN_FACTOR[config.decomposition_mode]
        res_u, res_v = int(base_res/down_factor), int(base_res/down_factor)
        # --- Scene Scale Awareness (NEW) ---
        print(f"Decomposition Mode: {config.decomposition_mode}")
        self.splines = MultiSurfaceSplineModel.from_pointcloud(
            pcd,
            pcd_rgb,
            config,  # NurbsOptimizationParams
            args,  # Training args
            resolution=(res_u, res_v),
            decomposition_mode=decomposition_mode,
            nerf_radius=self.cameras_extent,
            nerf_translate=scene_info.nerf_normalization['translate'],
            train_cam_uids=list(set([cam.uid for cam in self.getTrainCameras()])),
            cameras=self.getTrainCameras().copy(),
            n_components=n_components,
            connectivity_radius=connectivity_radius,
            smoothing=smoothing,
            sampling_factor=config.sampling_density,
            base_resolution=128,
            min_resolution=64,
            target_density_per_unit=config.target_density_per_unit,
            max_resolution=config.max_res,
            parameterization=config.encode_points,
            post_fit_iterations=config.post_fit_iterations,
            post_fit_enabled=config.post_fit_enabled,

        )
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # NEW: UV Consistency Setup
    # =========================================================================
    #
    # def _setup_uv_consistency(self, config):
    #     """
    #     Setup UV consistency loss using precomputed correspondences.
    #
    #     Workflow:
    #     1. Check if correspondences file exists
    #     2. If not, precompute and save
    #     3. Load correspondences
    #     4. Create loss module
    #     """
    #     # In SplineScene._setup_uv_consistency:
    #     correspondences_path = os.path.join(self.model_path, "correspondences.pt")
    #
    #     # Delete old cache to force reprocessing
    #     if os.path.exists(correspondences_path):
    #         os.remove(correspondences_path)
    #         print("[UV Consistency] Deleted old cache - will reprocess")
    #
    #     # Reprocess with ALL neighbors
    #     self._precompute_correspondences(correspondences_path, config)
    #
    #     # Load correspondences
    #     self.correspondences = CorrespondencePreprocessor.load(correspondences_path)
    #
    #     print(f"[UV Consistency] Loaded {self.correspondences.num_correspondences} correspondences "
    #           f"across {self.correspondences.num_views} views")
    #
    #     # Create loss module
    #     self.uv_consistency_loss = MultiViewPrecomputedUVLoss(
    #         correspondences=self.correspondences,
    #         max_neighbors=config.uv_consistency_max_neighbors,
    #         loss_weight=config.uv_consistency_weight,
    #         compute_every_n_iters=config.uv_consistency_interval,
    #         device='cuda'
    #     )


    #
    #
    # def _precompute_correspondences(self, save_path: str, config):
    #     """
    #     Precompute feature correspondences from GT images.
    #
    #     This is run ONCE and cached for future training runs.
    #     """
    #     # Get training cameras
    #     train_cams = self.getTrainCameras()
    #
    #     # Extract GT images
    #     print(f"[Preprocessing] Extracting {len(train_cams)} GT images...")
    #     images = {}
    #     for cam in train_cams:
    #         # Get original GT image
    #         # Assuming cam.original_image is [C, H, W] tensor in [0, 1]
    #         if hasattr(cam, 'original_image'):
    #             img = cam.original_image
    #             if isinstance(img, torch.Tensor):
    #                 img = img.cpu().numpy()
    #                 if img.shape[0] == 3:  # [C, H, W] -> [H, W, C]
    #                     img = img.transpose(1, 2, 0)
    #                 img = (img * 255).astype(np.uint8)
    #             images[cam.uid] = img
    #         else:
    #             # Load from disk if not cached
    #             img_path = cam.image_path if hasattr(cam, 'image_path') else None
    #             if img_path and os.path.exists(img_path):
    #                 from PIL import Image
    #                 img = np.array(Image.open(img_path).convert('RGB'))
    #                 images[cam.uid] = img
    #             else:
    #                 print(f"  Warning: Could not load image for camera {cam.uid}")
    #
    #     if len(images) < 2:
    #         print("[Preprocessing] Not enough images for correspondence computation.  Skipping.")
    #         # Save empty correspondences
    #         empty_corr = PrecomputedCorrespondences(
    #             correspondences=[],
    #             view_to_correspondences={},
    #             view_pair_to_matches={},
    #             num_views=len(train_cams),
    #             num_correspondences=0
    #         )
    #         torch.save(self._correspondences_to_dict(empty_corr), save_path)
    #         return
    #
    #     # # Create preprocessor
    #     # preprocessor = CorrespondencePreprocessor(
    #     #     feature_type=getattr(config, 'uv_feature_type', 'sift'),
    #     #     max_features=getattr(config, 'uv_max_features', 2000),
    #     #     match_ratio_threshold=getattr(config, 'uv_match_ratio', 0.75),
    #     #     min_matches=getattr(config, 'uv_min_matches', 20),
    #     #     triangulation_reproj_threshold=4.0,
    #     # use_fundamental_filter = True
    #     # )
    #     #
    #     # # Process dataset
    #     # self.correspondences = preprocessor.process_dataset(
    #     #     images=images,
    #     #     cameras=train_cams,
    #     #     max_neighbors=getattr(config, 'uv_preprocess_max_neighbors', 3),
    #     #     verbose=True
    #     # )
    #
    #     # Save
    #     preprocessor.save(self.correspondences, save_path)
    #     print(f"[Preprocessing] Saved correspondences to {save_path}")
    #
    # def _correspondences_to_dict(self, corr: PrecomputedCorrespondences) -> dict:
    #     """Helper to convert empty correspondences to saveable dict."""
    #     return {
    #         'correspondences': [],
    #         'view_to_correspondences': {},
    #         'view_pair_matches': {},
    #         'num_views': corr.num_views,
    #         'num_correspondences': 0
    #     }
    #
    # # =========================================================================
    # # NEW:  Methods for training integration
    # # =========================================================================
    #
    # def compute_uv_consistency_loss(
    #         self,
    #         surface,  # SplineModel
    #         current_cam,
    #         neighbor_cams: list = None
    # ):
    #     """
    #     Compute UV consistency loss for a view.
    #
    #     Call this in your training loop after forward pass.
    #
    #     Args:
    #         surface: The SplineModel surface
    #         current_cam: Current viewpoint camera
    #         neighbor_cams: Optional list of neighbor cameras (uses cam.nearest_id if None)
    #
    #     Returns:
    #         loss: Scalar tensor
    #         stats: Dict with debugging info
    #     """
    #     if self.uv_consistency_loss is None:
    #         return torch.tensor(0.0, device='cuda', requires_grad=True), {'skipped': True}
    #
    #     # Get neighbors if not provided
    #     if neighbor_cams is None:
    #         neighbor_cams = get_neighbor_cameras(
    #             current_cam,
    #             self.getTrainCameras(),
    #             max_neighbors=2
    #         )
    #
    #     return self.uv_consistency_loss(
    #         surface=surface,
    #         current_cam=current_cam,
    #         neighbor_cams=neighbor_cams
    #     )
    #
    # def get_uv_consistency_stats(self) -> dict:
    #     """Get statistics about the precomputed correspondences."""
    #     if self.correspondences is None:
    #         return {'status': 'not_initialized'}
    #
    #     return {
    #         'num_correspondences': self.correspondences.num_correspondences,
    #         'num_views': self.correspondences.num_views,
    #         'num_view_pairs': len(self.correspondences.view_pair_to_matches),
    #         'avg_matches_per_pair': (
    #                 sum(len(m.pixels_1) for m in self.correspondences.view_pair_to_matches.values())
    #                 / max(1, len(self.correspondences.view_pair_to_matches))
    #         )
    #     }
    # def update_camera_neighbors_by_visibility(self, visibility_system, num_neighbors=10):
    #     """
    #     Updates the 'nearest_id' for each camera based on the similarity of
    #     their visibility vectors from the VisibilityMapper.
    #
    #     Args:
    #         visibility_system (VisibilityMapper): The mapper containing the visibility matrix.
    #         num_neighbors (int): The number of nearest neighbors to store for each camera.
    #     """
    #
    #     # Get the visibility matrix [num_cameras, num_points]
    #     visibility_matrix = visibility_system.visibility_matrix.float()
    #     num_cameras = visibility_matrix.shape[0]
    #
    #     # Calculate pairwise intersection and union
    #     # Intersection: (A & B) = A * B
    #     intersection = visibility_matrix @ visibility_matrix.T
    #
    #     # Union: (A | B) = |A| + |B| - (A & B)
    #     # We can calculate the size of each visibility vector first
    #     visibility_counts = visibility_matrix.sum(dim=1, keepdim=True)
    #     # Union is calculated using broadcasting
    #     union = visibility_counts + visibility_counts.T - intersection
    #
    #     # Jaccard Similarity = Intersection / Union
    #     jaccard_similarity = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero
    #
    #     # Exclude self-similarity
    #     jaccard_similarity.fill_diagonal_(0)
    #
    #     # Find the top K most similar cameras for each camera
    #     top_k_similarities, top_k_indices = torch.topk(
    #         jaccard_similarity, k=min(num_neighbors, num_cameras - 1), dim=1
    #     )
    #
    #     # Update the nearest_id for each camera object
    #     all_cameras = self.getTrainCameras()
    #     for i, cam in enumerate(all_cameras):
    #         # Filter out neighbors with zero similarity
    #         valid_mask = top_k_similarities[i] > 0
    #         cam.nearest_id = top_k_indices[i][valid_mask].tolist()
    #
    # def update_camera_neighbors_by_visibility_by_grid(self, visibility_grids, num_neighbors=None):
    #     num_neighbors = self.multi_view_num if num_neighbors is None else num_neighbors
    #     for cam in self.getTrainCameras():
    #         vis_ref = visibility_grids[cam.uid]
    #         ious = []
    #         for other_uid, vis_other in visibility_grids.items():
    #             if other_uid == cam.uid: continue
    #             intersection = (vis_ref & vis_other).sum().float()
    #             union = (vis_ref | vis_other).sum().float()
    #             iou = intersection / (union + 1e-8)
    #             ious.append((other_uid, iou))
    #         cam.nearest_id = [uid for uid, _ in sorted(ious, key=lambda x: x[1], reverse=True)[:num_neighbors]]
    def get_splines(self) -> MultiSurfaceSplineModel:
        return self.splines

    def get_splesh(self):
        return self.splesh
    @property
    def get_source_pcd(self):
        return self.pcd

    def save(self, iteration, mask=None, scan_name=''):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        self.splines.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)

    def getTrainCameras(self, scale=1.0, num_cameras=None):
        if not num_cameras:
            return self.train_cameras[scale]

        return self.train_cameras[scale][:num_cameras]

    def getTrainStereoCameras(self, scale=1.0, num_cameras=None):
        if not num_cameras:
            return self.train_cameras_stereo[scale]

        return self.train_cameras_stereo[scale][:num_cameras]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


import numpy as np

"""
Surface-Response Camera Similarity.

Replaces geometric heuristics (distance + angle) with learned similarity:
how similarly does the trained surface respond to two cameras?

Core idea: project surface points through each camera via compute_ray_info,
extract depth profiles, and measure overlap-weighted depth correlation.
"""


import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple



@dataclass
class SurfaceResponseSimilarity:
    """
    Per-camera-pair similarity score derived from surface depth response.

    Fields
    ------
    overlap_iou       : fraction of surface points visible in both cameras
    depth_correlation : Pearson r of depth profiles over shared visible points
    depth_histogram_sim: χ²-distance between depth histograms (full FOV)
    normal_consistency : mean cosine similarity of normals at co-visible points
    composite         : weighted combination used for neighbor ranking
    """
    overlap_iou: float = 0.0
    depth_correlation: float = 0.0
    depth_histogram_sim: float = 0.0
    normal_consistency: float = 0.0
    composite: float = 0.0


@dataclass
class SurfaceResponseConfig:
    """Weights and thresholds for composite similarity."""
    # Component weights
    overlap_weight: float = 1.0
    depth_corr_weight: float = 1.5      # Depth is the strongest signal
    depth_hist_weight: float = 0.5
    normal_weight: float = 0.8

    # Depth histogram bins
    num_depth_bins: int = 32

    # Minimum co-visibility to bother computing depth metrics
    min_overlap_fraction: float = 0.05

    # Baseline guard: reject pairs with angle < min_baseline_angle degrees
    # even if depth similarity is high (degenerate coplanar case)
    min_baseline_angle_deg: float = 3.0

    # Border margin passed to compute_ray_info
    border_margin: float = 2.0


def compute_surface_response_similarity(
    cam_a,
    cam_b,
    surface_points: torch.Tensor,        # [N, 3] world-space
    surface_normals: Optional[torch.Tensor] = None,  # [N, 3] optional
    config: SurfaceResponseConfig = None,
    depth_min: float = 1e-3,
    depth_max: float = 1e3,
) -> SurfaceResponseSimilarity:
    """
    Compute surface-response similarity between two cameras.

    Uses compute_ray_info to project surface points into each camera,
    then compares depth profiles over co-visible regions.

    Args:
        cam_a, cam_b : Camera objects (same interface as in compute_ray_info)
        surface_points : [N, 3] evaluated NURBS surface points
        surface_normals : [N, 3] optional surface normals for normal consistency
        config : SurfaceResponseConfig
        depth_min, depth_max : valid depth range

    Returns:
        SurfaceResponseSimilarity
    """
    if config is None:
        config = SurfaceResponseConfig()

    result = SurfaceResponseSimilarity()

    # --- Baseline angle guard -----------------------------------
    # Reject degenerate coplanar camera pairs before any computation
    dir_a = F.normalize(cam_a.camera_center - surface_points.mean(dim=0), dim=0)
    dir_b = F.normalize(cam_b.camera_center - surface_points.mean(dim=0), dim=0)
    baseline_cos = (dir_a * dir_b).sum().clamp(-1, 1)
    baseline_angle_deg = torch.acos(baseline_cos).item() * 180.0 / 3.14159
    if baseline_angle_deg < config.min_baseline_angle_deg:
        return result  # All zeros — degenerate pair

    # --- Ray info for both cameras --------------------------------
    from modules.sampling.ray_utils import compute_ray_info
    from modules.sampling.ray_utils import RayInfo
    ray_a: RayInfo = compute_ray_info(
        cam_a, surface_points,
        border_margin=config.border_margin,
        depth_min=depth_min, depth_max=depth_max
    )
    ray_b: RayInfo = compute_ray_info(
        cam_b, surface_points,
        border_margin=config.border_margin,
        depth_min=depth_min, depth_max=depth_max
    )

    vis_a = ray_a.visible_mask   # [N] bool
    vis_b = ray_b.visible_mask   # [N] bool

    # --- Overlap IoU -----------------------------------------------
    intersection = (vis_a & vis_b).sum().float()
    union = (vis_a | vis_b).sum().float().clamp(min=1.0)
    result.overlap_iou = (intersection / union).item()

    # Early exit: not enough co-visible points
    N = surface_points.shape[0]
    if result.overlap_iou < config.min_overlap_fraction:
        result.composite = 0.0
        return result

    co_visible = vis_a & vis_b  # [N] bool

    # --- Depth correlation over co-visible points ------------------
    depths_a = ray_a.depths[co_visible]   # [M]
    depths_b = ray_b.depths[co_visible]   # [M]

    result.depth_correlation = _pearson_correlation(depths_a, depths_b).item()

    # --- Depth histogram similarity (full visible region) ----------
    # Uses both cameras' full visible depth profiles for broader coverage
    result.depth_histogram_sim = _histogram_similarity(
        ray_a.depths[vis_a],
        ray_b.depths[vis_b],
        num_bins=config.num_depth_bins,
        depth_min=depth_min,
        depth_max=min(depth_max, max(
            ray_a.depths[vis_a].max().item() if vis_a.any() else 1.0,
            ray_b.depths[vis_b].max().item() if vis_b.any() else 1.0,
        ))
    ).item()

    # --- Normal consistency at co-visible points -------------------
    if surface_normals is not None:
        normals_a_view = _normals_in_camera_frame(
            surface_normals[co_visible], ray_a.directions[co_visible]
        )
        normals_b_view = _normals_in_camera_frame(
            surface_normals[co_visible], ray_b.directions[co_visible]
        )
        result.normal_consistency = (normals_a_view * normals_b_view).sum(dim=-1).mean().item()
    else:
        result.normal_consistency = result.depth_correlation  # fallback

    # --- Composite score -------------------------------------------
    result.composite = (
        config.overlap_weight      * result.overlap_iou
      + config.depth_corr_weight   * max(result.depth_correlation, 0.0)
      + config.depth_hist_weight   * result.depth_histogram_sim
      + config.normal_weight       * max(result.normal_consistency, 0.0)
    )

    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson r, numerically stable. Returns 0 if degenerate."""
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return (x * y).sum() / denom


def _histogram_similarity(
    depths_a: torch.Tensor,
    depths_b: torch.Tensor,
    num_bins: int,
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """
    Normalised histogram intersection similarity ∈ [0, 1].

    More robust than χ² for sparse histograms; simple and differentiable
    if needed. Returns 0 for empty inputs.
    """
    device = depths_a.device
    if depths_a.numel() == 0 or depths_b.numel() == 0:
        return torch.tensor(0.0, device=device)

    depth_max = max(depth_max, depth_min + 1e-6)
    edges = torch.linspace(depth_min, depth_max, num_bins + 1, device=device)

    def _hist(d: torch.Tensor) -> torch.Tensor:
        d_clamped = d.clamp(depth_min, depth_max)
        # bucketize returns 1-indexed; subtract 1 and clamp
        idx = torch.bucketize(d_clamped, edges[1:-1])  # [M] in [0, num_bins-1]
        h = torch.zeros(num_bins, device=device)
        h.scatter_add_(0, idx, torch.ones_like(d_clamped))
        return h / (h.sum() + 1e-8)

    ha = _hist(depths_a)
    hb = _hist(depths_b)

    # Histogram intersection: sum of min(ha_i, hb_i)
    return torch.minimum(ha, hb).sum()


def _normals_in_camera_frame(
    normals_world: torch.Tensor,   # [M, 3]
    ray_directions: torch.Tensor,  # [M, 3] normalised
) -> torch.Tensor:
    """Project normals onto the image plane (remove view-aligned component)."""
    dot = (normals_world * ray_directions).sum(dim=-1, keepdim=True)
    projected = normals_world - dot * ray_directions
    return F.normalize(projected, dim=-1)


# ---------------------------------------------------------------------------
# Scene-level neighbor update
# ---------------------------------------------------------------------------

def update_neighbors_by_surface_response(
    scene,
    multi_surface_model,
    num_neighbors: int = 5,
    config: SurfaceResponseConfig = None,
    depth_min: float = 1e-3,
    depth_max: float = 50.0,
    use_normals: bool = True,
    verbose: bool = False,
) -> Dict[int, List[int]]:
    """
    Recompute camera.nearest_id for all training cameras using
    surface-response depth similarity.

    Replaces the geometric heuristic in SplineScene.__init__.
    Safe to call at any point during training (e.g., post-warmup).

    Args:
        scene            : SplineScene instance
        multi_surface_model : MultiSurfaceSplineModel (must have run forward())
        num_neighbors    : k nearest neighbors to store
        config           : SurfaceResponseConfig
        depth_min/max    : valid depth range
        use_normals      : include normal consistency in composite score
        verbose          : print progress

    Returns:
        Dict mapping camera uid → list of neighbor uids
    """
    if config is None:
        config = SurfaceResponseConfig()

    train_cams = scene.getTrainCameras()
    N_cams = len(train_cams)

    # Collect surface geometry from all active surfaces
    surface_points_list = multi_surface_model.xyz_grids  # List[[Us, Vs, 3]]
    surface_points = torch.cat(
        [g.reshape(-1, 3) for g in surface_points_list], dim=0
    )  # [total_N, 3]

    surface_normals = None
    if use_normals:
        normal_grids = multi_surface_model.normal_grids  # List[[Us, Vs, 3]]
        surface_normals = torch.cat(
            [g.reshape(-1, 3) for g in normal_grids], dim=0
        )  # [total_N, 3]

    # Build uid → camera index map
    uid_to_idx = {cam.uid: i for i, cam in enumerate(train_cams)}

    # Compute pairwise similarity (upper triangle only)
    similarity_matrix = torch.zeros(N_cams, N_cams)

    if verbose:
        print(f"[SurfaceResponse] Computing {N_cams*(N_cams-1)//2} camera pairs...")

    for i in range(N_cams):
        for j in range(i + 1, N_cams):
            sim = compute_surface_response_similarity(
                train_cams[i],
                train_cams[j],
                surface_points,
                surface_normals,
                config=config,
                depth_min=depth_min,
                depth_max=depth_max,
            )
            similarity_matrix[i, j] = sim.composite
            similarity_matrix[j, i] = sim.composite

            if verbose and (i * N_cams + j) % 200 == 0:
                print(f"  [{i},{j}] overlap={sim.overlap_iou:.3f} "
                      f"depth_corr={sim.depth_correlation:.3f} "
                      f"composite={sim.composite:.3f}")

    # Assign neighbors
    neighbor_map: Dict[int, List[int]] = {}
    for i, cam in enumerate(train_cams):
        scores = similarity_matrix[i]
        scores[i] = -1.0  # Exclude self

        top_k = min(num_neighbors, N_cams - 1)
        _, top_indices = torch.topk(scores, top_k)

        # Filter zero-similarity neighbors (non-overlapping cameras)
        valid = [idx.item() for idx in top_indices if scores[idx] > 0]

        cam.nearest_id = [train_cams[idx].uid for idx in valid]
        neighbor_map[cam.uid] = cam.nearest_id

        if verbose:
            print(f"  Camera {cam.uid}: neighbors = {cam.nearest_id}")

    return neighbor_map
"""
Drop-in integration for surface-response neighbor update in SplineScene.
Call this after warmup instead of relying on the geometric heuristic.
"""

# from model.utils.surface_response_similarity import (
#     update_neighbors_by_surface_response,
#     SurfaceResponseConfig,
# )


def refresh_camera_neighbors_post_warmup(
    scene,
    multi_surface_model,
    iteration: int,
    warmup_iterations: int = 3000,
    refresh_every: int = 2000,
    num_neighbors: int = 5,
    verbose: bool = True,
):
    """
    Conditionally refresh camera neighbors based on surface response.

    Call this inside your training loop. It is a no-op before warmup
    and refreshes every `refresh_every` iterations thereafter.

    Example usage in train.py:
        refresh_camera_neighbors_post_warmup(
            scene, gaussians, iteration,
            warmup_iterations=args.warmup_iters,
        )
    """
    if iteration < warmup_iterations:
        return
    if (iteration - warmup_iterations) % refresh_every != 0:
        return

    if verbose:
        print(f"\n[NeighborRefresh] Updating camera neighbors at iter {iteration} "
              f"using surface-response depth similarity...")

    config = SurfaceResponseConfig(
        overlap_weight=1.0,
        depth_corr_weight=1.5,
        depth_hist_weight=0.5,
        normal_weight=0.8,
        min_overlap_fraction=0.05,
        min_baseline_angle_deg=3.0,
    )

    update_neighbors_by_surface_response(
        scene=scene,
        multi_surface_model=multi_surface_model,
        num_neighbors=num_neighbors,
        config=config,
        verbose=verbose,
    )
def remove_sparse_outliers_advanced(
        pcd: np.ndarray,  # (N, 3)
        pcd_rgb: np.ndarray,  # (N, 3)
        k_neighbors: int = 15,
        std_ratio: float = 3.0,
        use_density_adaptive: bool = True,
        min_cluster_size: int = 64,
        device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advanced statistical outlier removal for NURBS fitting.

    Three-stage filtering:
    1. KNN distance-based statistical filtering (classic SOR)
    2. Density-adaptive thresholding (handles varying point density)
    3. Small cluster removal (eliminates isolated noise clusters)

    Args:
        pcd: Point cloud positions (N, 3)
        pcd_rgb: Point cloud colors (N, 3)
        k_neighbors: Number of neighbors for statistics
        std_ratio: Standard deviation multiplier for threshold
        use_density_adaptive: Use local density for adaptive thresholding
        min_cluster_size: Minimum points to keep a connected component
        device: CUDA device

    Returns:
        filtered_pcd, filtered_rgb, inlier_mask
    """

    points = torch.from_numpy(np.asarray(pcd)).float().to(device)
    N = points.shape[0]

    # =========================================================================
    # Stage 1: KNN-based Statistical Outlier Removal
    # =========================================================================


    # For proper SOR, we need k-nearest neighbor distances
    # Compute full KNN statistics
    knn_dists = _compute_knn_distances(points, k_neighbors)  # (N, k)
    mean_knn_dist = knn_dists.mean(dim=1)  # (N,)

    # Global statistics
    global_mean = mean_knn_dist.mean()
    global_std = mean_knn_dist.std()

    # Classic SOR threshold
    threshold_sor = global_mean + std_ratio * global_std
    inlier_mask_sor = mean_knn_dist < threshold_sor

    # =========================================================================
    # Stage 2: Density-Adaptive Thresholding
    # =========================================================================

    if use_density_adaptive:
        # Local density estimation (inverse of local mean distance)
        local_density = 1.0 / (mean_knn_dist + 1e-8)

        # Compute local statistics in density neighborhood;
        # Points in dense regions should have tighter thresholds
        density_percentile = _compute_percentile_rank(local_density)

        # Adaptive threshold: tighter in dense regions, looser in sparse
        # Dense regions (high percentile): use stricter threshold
        # Sparse regions (low percentile): use looser threshold
        adaptive_factor = 1.0 + (1.0 - density_percentile) * 0.5  # [1.0, 1.5]
        threshold_adaptive = global_mean + std_ratio * global_std * adaptive_factor

        inlier_mask_adaptive = mean_knn_dist < threshold_adaptive

        # Combine: point must pass both tests
        inlier_mask = inlier_mask_sor & inlier_mask_adaptive
    else:
        inlier_mask = inlier_mask_sor

    # =========================================================================
    # Stage 3: Small Cluster Removal
    # =========================================================================

    if min_cluster_size > 1:
        # Find connected components among inliers
        inlier_indices = torch.where(inlier_mask)[0]
        inlier_points = points[inlier_mask]

        if inlier_points.shape[0] > min_cluster_size:
            cluster_labels = _dbscan_clustering(
                inlier_points,
                eps=global_mean.item() * 2.0,  # Connectivity radius
                min_samples=min_cluster_size // 5
            )

            # Count cluster sizes
            unique_labels, counts = torch.unique(cluster_labels, return_counts=True)

            # Keep only clusters larger than min_cluster_size
            # Label -1 is noise in DBSCAN
            valid_clusters = unique_labels[(counts >= min_cluster_size) & (unique_labels >= 0)]

            # Update inlier mask
            cluster_valid = torch.zeros(inlier_points.shape[0], dtype=torch.bool, device=device)
            for label in valid_clusters:
                cluster_valid |= (cluster_labels == label)

            # Map back to original indices
            final_mask = torch.zeros(N, dtype=torch.bool, device=device)
            final_mask[inlier_indices[cluster_valid]] = True
            inlier_mask = final_mask

    # =========================================================================
    # Apply filtering
    # =========================================================================

    inlier_mask_np = inlier_mask.cpu().numpy()
    filtered_pcd = pcd[inlier_mask_np]
    filtered_rgb = pcd_rgb[inlier_mask_np]

    print(f"[Outlier Removal] Removed {N - inlier_mask.sum().item()}/{N} points "
          f"({100 * (1 - inlier_mask.float().mean().item()):.1f}%)")

    return filtered_pcd, filtered_rgb, inlier_mask_np


def _compute_knn_distances(
        points: torch.Tensor,
        k: int,
        chunk_size: int = 4096
) -> torch.Tensor:
    """
    Compute k-nearest neighbor distances efficiently.
    Uses chunking to handle large point clouds.
    """
    N = points.shape[0]
    device = points.device
    k = min(k, N - 1)

    knn_dists = torch.zeros(N, k, device=device)

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk = points[i:end_i]

        # Compute distances from chunk to all points
        # Using squared distances for efficiency
        diff = chunk.unsqueeze(1) - points.unsqueeze(0)  # (chunk, N, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (chunk, N)

        # Get k+1 smallest (includes self), then exclude self
        topk_dists, _ = torch.topk(dist_sq, k + 1, dim=1, largest=False)
        knn_dists[i:end_i] = topk_dists[:, 1:].sqrt()  # Exclude self (distance 0)

    return knn_dists


def _compute_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Compute percentile rank for each value (0 to 1)."""
    sorted_indices = torch.argsort(values)
    ranks = torch.zeros_like(values)
    ranks[sorted_indices] = torch.linspace(0, 1, len(values), device=values.device)
    return ranks


def _dbscan_clustering(
        points: torch.Tensor,
        eps: float,
        min_samples: int
) -> torch.Tensor:
    """
    Simple DBSCAN implementation for GPU.
    For large point clouds, consider using RAPIDS cuML.
    """
    # Fall back to sklearn for reliability
    # In production, use cuML for GPU acceleration
    try:
        from sklearn.cluster import DBSCAN
        points_np = points.cpu().numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points_np)
        return torch.from_numpy(clustering.labels_).to(points.device)
    except ImportError:
        # If sklearn not available, skip clustering stage
        return torch.zeros(points.shape[0], dtype=torch.long, device=points.device)


# =============================================================================
# NURBS-Specific Outlier Handling
# =============================================================================

def remove_outliers_for_nurbs(
        pcd: np.ndarray,
        pcd_rgb: np.ndarray,
        decomposition_mode: str = 'single',  # 'single', 'background_object', 'k_components'
        **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NURBS-aware outlier removal that considers surface decomposition.

    Key insight: For multi-surface fitting, we should be more conservative
    at potential surface boundaries to avoid removing valid edge points.
    """

    if decomposition_mode == DecompositionMode.SINGLE:
        # Standard filtering for single surface
        return remove_sparse_outliers_advanced(
            pcd, pcd_rgb,
            **kwargs
        )

    elif decomposition_mode in [DecompositionMode.BACKGROUND_OBJECT, DecompositionMode.K_COMPONENTS]:
        # More conservative filtering for multi-surface
        # First pass: remove obvious outliers
        pcd_clean, rgb_clean, mask1 = remove_sparse_outliers_advanced(
            pcd, pcd_rgb,
            **kwargs
        )

        return pcd_clean, rgb_clean, mask1

    else:
        raise ValueError(f"Unknown decomposition mode: {decomposition_mode}")