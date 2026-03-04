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
import torch
import json
import random
import os
from arguments import ModelParams, OptimizationParams, NurbsOptimizationParams
from model import Spline
from model.Splesh import Splesh
from model.spline_init import fit_bspline_surface_to_mesh
from model.spline_utils import mask3d, load_3dm_nurbs_patches, from_pcd_to_NURBS, initialize_nurbs_from_colmap, \
    initialize_nurbs_side_from_colmap, load_3dm_nurbs_grids
from model.utils.spline_local_init import depth_maps_to_pointcloud
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
#, rotm2eul, eul2rotm, \
    # calculate_right_camera_pose, intrinsic_from_camera_params, RT_from_rot_pos, sort_camera_coordinates
# from utils.general_utils import PILtoTorch
# from utils.pose_utils import get_tensor_from_camera



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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_unprojection(camera_points, cam_centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cam_centers[:,0], cam_centers[:,1], cam_centers[:,2], c='red', marker='^', label="Cameras")
    ax.scatter(camera_points[:,0], camera_points[:,1], camera_points[:,2], c='blue', marker='o', label="Unprojected")
    ax.legend()
    plt.show()

class SplineScene:
    gaussians: GaussianModel

    def __init__(self, args : ModelParams, config: NurbsOptimizationParams, resolution_scales=[1.0], shuffle=True, scan_id=None):

        """b
        :param path: Path to colmap scene main folder.
        """
        self.train_cameras = {}
        self.test_cameras = {}
        self.source_path = args.source_path
        self.model_path = args.model_path
        self.loaded_iter = None
        self.pcd = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
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
        # Load mesh from PLY
        # mesh_path = "/sci/labs/sagieb/zlilovadia/nurbs/output/surfels/scan24_4/mesh/tsdf_fusion_post.ply"
        # import open3d as o3d
        # # mesh_path = os.path.join(args.source_path, "input.ply")
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # vertices = np.asarray(mesh.vertices)
        # faces = np.asarray(mesh.triangles)
        # vertex_colors = None
        # if mesh.has_vertex_colors():
        #     vertex_colors = np.asarray(mesh.vertex_colors)

        # Fit BSpline surface
        # surf, color_grid = fit_bspline_surface_to_mesh(
        #     mesh_vertices=vertices,
        #     mesh_faces=faces,
        #     num_patches_u=64,  # Or from your config/args
        #     num_patches_v=64,
        #     deg_u=3, deg_v=3,
        #     mesh_vertex_colors=vertex_colors
        # )

        control_pts = load_3dm_nurbs_grids(file_path=f"/sci/labs/sagieb/zlilovadia/KnotSurface/datasets/DTU/nurbs/{scan_id}.3dm")
        colors = torch.ones_like(control_pts) * 0.5

        self.splines = Spline(ctrl_pts=control_pts,
                              args=args,
                              config=config,
                              spatial_lr_scale=self.cameras_extent)
        # del res
        gc.collect()
        torch.cuda.empty_cache()

    # @property
    def get_splines(self):
        return self.splines

    def get_splesh(self):
        return self.splesh
    @property
    def get_source_pcd(self):
        return self.pcd

    def save(self):
        self.splines.save_ply()

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

    # def prepare_stereo_cameras(self):
    #     cam_extrinsics = read_extrinsics_text(os.path.join(self.source_path,'sparse','0','images.txt'))
    #
    #     self.poses = poses_from_file(cam_extrinsics)
    #     poses_inv = [np.linalg.inv(np.vstack((pose, np.array([0, 0, 0, 1])))) for pose in self.poses]
    #     camera_rotations = [rotm2eul(pose[:3,:3]) for pose in poses_inv]
    #     for i in range(len(camera_rotations)):
    #         rotation = eul2rotm(camera_rotations[i])
    #         rotation[:,1:]*=-1
    #         camera_rotations[i] = rotm2eul(rotation)
    #
    #     camera_locations = [pose[:3, 3].tolist() for pose in poses_inv]
    #
    #     camera_params = read_intrinsics_text(os.path.join(self.source_path, 'sparse', '0', 'cameras.txt'))
    #     camera_params_indices = sorted(list(camera_params.keys()))
    #     camera_params = [{'width': camera_params[i].width,
    #                       'height': camera_params[i].height,
    #                       'fx': camera_params[i].params[0],
    #                       'fy': camera_params[i].params[0 if camera_params[i].model == 'SIMPLE_RADIAL' else 1],
    #                       'cx': camera_params[i].params[1 if camera_params[i].model == 'SIMPLE_RADIAL' else 2],
    #                       'cy': camera_params[i].params[2 if camera_params[i].model == 'SIMPLE_RADIAL' else 3]} for i in
    #                      camera_params_indices]
    #
    #     if len(camera_params) != len(camera_locations):
    #         camera_params = [camera_params[0]] * len(camera_locations)
    #
    #     ts = np.array(camera_locations)
    #     radius = (np.median(np.linalg.norm(ts - ts.mean(axis=0), axis=1))) * 2
    #     self.baseline = radius * (7.0 / 100)
    #     cam_infos = []
    #     self.cameras = []
    #     images_folder = os.path.join(self.source_path, "images")
    #
    #     # Create the CameraInfo object (ensure depth is included)
    #     # sorted_camera_indices = sort_camera_coordinates(np.array(camera_locations))
    #     sorted_camera_indices = range(len(camera_locations))
    #     self.left_cameras = []
    #     self.right_cameras = []
    #     for i in range(1, len(camera_locations)+1):
    #         image_name = os.path.basename(cam_extrinsics[i].name)
    #         camera_index = sorted_camera_indices[i]
    #         R_right, T_right = calculate_right_camera_pose(camera_rotations[camera_index],
    #                                                        camera_locations[camera_index], self.baseline)
    # #     for i in range(0, len(camera_locations), 2):
    #         mask_folder = os.path.join(self.source_path, 'mask')
    #         image_path = os.path.join(images_folder, os.path.basename(image_name))
    #         image_name = os.path.basename(image_path).split(".")[0]
    #         image = Image.open(image_path)
    #         orig_w, orig_h = image.size
    #         # if mask_folder is not None:
    #         try:
    #             # mask_path = os.path.join(mask_folder, os.path.basename(extr.name[1:])).split("/")[:-1]
    #             mask_path = os.path.join(mask_folder, cam_extrinsics[i].name[1:])
    #             mask = Image.open(mask_path)
    #             if mask.size != image.size[:2]:
    #                 mask = mask.resize(image.size[:2], resample=Image.NEAREST)
    #
    #             # Ensure mask is in binary format
    #             mask = mask.point(lambda p: 255 if p > 128 else 0).convert('1')
    #
    #             # image = (np.array(image) * np.asarray(mask)[..., None])[..., :3]
    #             # Convert back to PIL Image
    #             # image = Image.fromarray(image)
    #         except FileNotFoundError as e:
    #             print("Could not find Mask folder, continuing without masking...")
    #             mask = None
    #
    #         left_info = {
    #             'rot': tuple(camera_rotations[camera_index].tolist()),
    #             'pos': tuple(camera_locations[camera_index]),
    #             'width': camera_params[camera_index]['width'],
    #             'height': camera_params[camera_index]['height'],
    #             'fx': camera_params[camera_index]['fx'].item(),
    #             'fy': camera_params[camera_index]['fy'].item(),
    #             'cx': camera_params[camera_index]['cx'].item(),
    #             'cy': camera_params[camera_index]['cy'].item(),
    #             'intrinsic': intrinsic_from_camera_params(camera_params[camera_index]),
    #             'extrinsic': RT_from_rot_pos(tuple(camera_rotations[camera_index]),
    #                                          tuple(camera_locations[camera_index])),
    #             'baseline': self.baseline
    #         }
    #
    #         right_info = {
    #             'rot': R_right,
    #             'pos': T_right,
    #             'width': camera_params[camera_index]['width'],
    #             'height': camera_params[camera_index]['height'],
    #             'fx': camera_params[camera_index]['fx'].item(),
    #             'fy': camera_params[camera_index]['fy'].item(),
    #             'cx': camera_params[camera_index]['cx'].item(),
    #             'cy': camera_params[camera_index]['cy'].item(),
    #             'intrinsic': intrinsic_from_camera_params(camera_params[camera_index]),
    #             'extrinsic': RT_from_rot_pos(R_right, T_right)
    #         }
    #
    #         cam_infos.append({'left': left_info, 'right': right_info})
    #
    #         # Initialize Camera objects for left and right cameras
    #         left_camera = Camera(
    #             colmap_id=i,
    #             R=np.array(left_info['rot']),
    #             T=np.array(left_info['pos']),
    #             FoVx=2 * np.arctan(left_info['width'] / (2 * left_info['fx'])),
    #             FoVy=2 * np.arctan(left_info['height'] / (2 * left_info['fy'])),
    #             image=PILtoTorch(image, (int(orig_w / 2.), int(orig_h / 2.))),  # You may need to load the image here if required
    #             gt_alpha_mask=None,  # Set this if you have a mask
    #             image_name=f"left_{i}",
    #             uid=i,
    #             data_device="cuda",  # Adjust as needed
    #             baseline=self.baseline
    #         )
    #
    #         right_camera = Camera(
    #             colmap_id=i + len(camera_locations),  # Unique ID for right camera
    #             R=R_right,
    #             T=T_right,
    #             FoVx=2 * np.arctan(right_info['width'] / (2 * right_info['fx'])),
    #             FoVy=2 * np.arctan(right_info['height'] / (2 * right_info['fy'])),
    #             image=torch.zeros_like(PILtoTorch(image, (int(orig_w / 2.), int(orig_h / 2.)))),  # You may need to load the image here if required
    #             gt_alpha_mask=None,  # Set this if you have a mask
    #             image_name=f"right_{i}",
    #             uid=i + len(camera_locations),
    #             data_device="cuda",  # Adjust as needed
    #             baseline=self.baseline
    #         )
    #         self.left_cameras.append(left_camera)
    #         self.right_cameras.append(right_camera)
    #
    #     print(f"num views: {len(cam_infos)}")
    #     print(f"baseline: {self.baseline}")


    # def getStereoTrainCameras(self):
    #     return self.left_cameras.copy(), self.right_cameras.copy()

