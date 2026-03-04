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

import torch

from modules.multisurf import MultiSurfaceSplineModel
from modules.fitting.serialization import load_model
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np
import cv2
import open3d as o3d
import copy

def plot_render_outputs(render_dict, gt_image, title=None):
    """
    Plot images returned from render(), including:
      - Rendered RGB image ("render")
      - Depth image ("plane_depth")
      - Rendered normals ("rendered_normal")
      - Depth normals ("depth_normal")
      - Optionally, an app image ("app_image") if available.

    Ground truth images (if any) are not plotted.

    Parameters:
      render_dict (dict): Dictionary returned from render() that contains one or more of the keys:
          "render", "plane_depth", "rendered_normal", "depth_normal", "app_image"
    """
    # Build lists of images and titles for plotting
    images = [gt_image]
    titles = ["Ground Truth"]
    vis_normal = lambda normal: np.uint8((normal[..., [1, 2, 0]] + 1) / 2 * 255)

    if "render" in render_dict:
        images.append(render_dict["render"])
        titles.append("Rendered RGB")

    if "plane_depth" in render_dict:
        images.append(render_dict["plane_depth"])
        titles.append("Plane Depth")
    if "Grid Spherical Harmonics" in render_dict:
        from utils.sh_utils import SH2RGB
        im = SH2RGB(render_dict["Grid Spherical Harmonics"][..., :3]).detach().cpu().numpy()
        # im = np.rot90(im, k=-3)
        images.append(im)
        titles.append("Grid Spherical Harmonics")
    if "rendered_normal" in render_dict:
        # images.append(colorize_normal_map(render_dict["rendered_normal"]))
        im = render_dict["rendered_normal"].permute(1, 2, 0).detach().cpu().numpy()
        images.append(vis_normal(im))
        titles.append("Rendered Normal")

    if "depth_normal" in render_dict:
        im = render_dict["depth_normal"].permute(1, 2, 0).detach().cpu().numpy()

        images.append(vis_normal(im))
        titles.append("Depth Normal")

    if "app_image" in render_dict:
        images.append(render_dict["app_image"])
        titles.append("App Image")

    n_images = len(images)
    if n_images == 0:
        print("No images to plot.")
        return

    # Determine grid layout (using 2 columns here)
    cols = 2
    rows = (n_images + cols - 1) // cols

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(title)
    # Ensure axs is a list of Axes objects.
    if isinstance(axs, np.ndarray):
        axs = list(axs.flat)
    else:
        axs = [axs]

    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert PyTorch tensor to numpy array if necessary
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            # If image has shape (3, H, W), change to (H, W, 3)
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            # If image has a singleton channel (1, H, W), squeeze to (H, W)
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            img = img.numpy()

        if title=="Depth Normal":
            axs[i].imshow(img, cmap='jet')

        # Plot the image depending on its shape
        elif img.ndim == 2:
            axs[i].imshow(img, cmap='gray')


        else:
            axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')

    # Turn off any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_set(nurbs: 'MultiSurfaceSplineModel', model_path, name, iteration, views, pipeline, background,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False, use_depth_normal=False):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    trial_index = model_path.split('/')[-1]
    scan_name = model_path.split('/')[-2]
    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # nurbs._invalidate_cache(force=True)

        gt, _ = view.get_image()
        out = render(view, nurbs, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        if use_depth_normal:
            normal = out["depth_normal"].permute(1,2,0)

            # Second stage: Render with surface normals as colors (in [-1,1])
            # surface_normals = nurbs.get_normal(view)  # Already in [-1,1]
            # normal_render_pkg = render(view, nurbs, pipeline, background, override_color=surface_normals,
            #                            app_model=app_model, return_plane=True, return_depth_normal=False)


            # Compute loss between rendered normal map and depth-derived normals
            # normal = normal_render_pkg["render"].permute(1,2,0)  # Blended normals in [-1,1]

        else:
            normal = out["rendered_normal"].permute(1,2,0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)
        # out.update({"Grid Normals": nurbs.get_normal(view).reshape(nurbs.state.Us, nurbs.state.Vs, -1),
        #             "Grid Spherical Harmonics": nurbs.get_features.reshape(nurbs.state.Us, nurbs.state.Vs, -1)[..., :3]
        #             })

        if idx % 32 == 1:
            plot_render_outputs(out, gt, title=f"{scan_name}/{trial_index}, Iteration: {iteration}")
        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (
                        rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
                np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):

            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                max_depth: float, voxel_size: float, num_cluster: int, use_depth_filter: bool,  use_depth_normal:bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # scene = SplineScene(dataset, opt, scan_id=scene_name, background=background, pipe=pipeline)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        iteration = scene.loaded_iter
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        load_path = dataset.model_path + "/chkpnt" + str(iteration) + ".pth"
        nurbs = load_model(load_path)


        if not skip_train:
            render_set(nurbs, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       pipeline, background, app_model=None,
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, use_depth_normal=use_depth_normal)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)

            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh,
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

            mesh = post_process_mesh(mesh, num_cluster)
            print(f"\nwriting mesh to {path}...\n")
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh,
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(nurbs, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians,
                       pipeline, background, )


def depth_render(scene, camera):
    img, depth = scene.renderer.render_view(
        camera, with_normals=False, with_depth=True)
    return depth


if __name__ == "__main__":
    # torch.set_num_threads(4)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--use_depth_normal", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.use_depth_normal = False
    args.use_depth_filter = False

    # print(f"use depth normal: {args.use_depth_normal}")
    # print(f"use depth filter: {args.use_depth_filter}")
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args.use_depth_normal)