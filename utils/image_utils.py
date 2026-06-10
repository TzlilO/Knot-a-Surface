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
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import functional as F

from utils.sh_utils import SH2RGB


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out


def plot_render_outputs(render_dict, nurbs_prop_dict, gt_image, nurbs: 'MultiSurfaceSplineModel', uid=None, title=None,
                        save_path=None, show=True):
    """
    Side-by-side visualization:
    - Left (50%): High-res renders (GT, RGB, normals, depth) → stacked vertically
    - Right (50%): Grid quantities → arranged in a compact, near-square grid
    """
    import matplotlib.gridspec as gridspec

    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    vis_normal = lambda x: np.uint8((x[..., [1, 2, 0]] + 1) / 2 * 255)
    # vis_normal = lambda x: x
    vis_sh     = lambda x: SH2RGB(x[..., :3]).detach().cpu().numpy()
    vis_gray   = lambda x: (x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else x)
    vis_grad   = lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-6)).cpu().numpy() if isinstance(x, torch.Tensor) else x

    # ------------------------------------------------------------------ #
    # Plot definitions
    # ------------------------------------------------------------------ #
    image_plots = [
        {'key': 'gt_image',                     'title': 'Ground Truth',      'vis': None},
        {'key': 'render',                     'title': 'Rendered RGB',      'vis': None},
        {'key': 'rendered_normal',            'title': 'Rendered Normal',   'vis': lambda x: vis_normal(x.permute(1,2,0).cpu().numpy())},
        {'key': 'depth_normal',               'title': 'Depth Normal',      'vis': lambda x: vis_normal(x.permute(1,2,0).cpu().numpy())},
        {'key': 'plane_depth',                'title': 'Plane Depth',       'vis': vis_gray},
        {'key': 'decomposed_final_img',                'title': 'Decomposed Images',       'vis': None},
        {'key': 'object',                'title': 'Decomposed Images',       'vis': None},
        {'key': 'background',                'title': 'Decomposed Images',       'vis': None},
    ]


    grid_plots_all = []
    for i, surf in enumerate(nurbs.surfaces):
        grid_shape = surf.state.sampling_layout
        cp_grid_shape = surf.state.control_layout
        grid_plots_all.extend([
            {'key': f'norm_grid_{i}',               'title': f'Grid Normals {i}',      'vis': lambda x, gs=grid_shape: vis_normal(x.cpu().numpy().reshape(gs))},
            {'key': f'weights_map_per_view{i}',               'title': f'Grid Normals {i}',      'vis': lambda x, gs=grid_shape: vis_gray(x.cpu().numpy().reshape(gs))},
            {'key': f'sh_grid_{i}',                 'title': f'Grid SH {i}',           'vis': lambda x, gs=grid_shape: vis_sh(x).reshape(gs)},
            {'key': f'sh_cpt_{i}',                 'title': f'Grid SH {i}',           'vis': lambda x, gs=grid_shape: vis_sh(x).reshape(gs)},
            {'key': f'visibility_cp_{i}',                 'title': f'Grid SH {i}',           'vis': lambda x, gs=cp_grid_shape: vis_sh(x).reshape(gs)},
            ])



    # (0, grid_shape_0, cp_grid_shape_0),
    # (1, grid_shape_1, cp_grid_shape_1),

    #grid_plots_all = [
    #     {'key': 'Grads XYZ',                  'title': 'Grads XYZ',         'vis': lambda x: vis_grad(x.reshape(nurbs.state.control_layout))},
    #     {'key': 'Grid Normals',               'title': 'Grid Normals',      'vis': lambda x: vis_normal(x.cpu().numpy().reshape(grid_shape))},
    #     {'key': f'sh_grid_{i}',   'title': 'Grid SH',           'vis': lambda x: vis_sh(x).reshape(grid_shape)},
    #     # {'key': 'Grid Opacity',               'title': 'Grid Opacity',      'vis': lambda x: vis_gray(x).reshape(grid_shape)},
    #     # {'key': 'Grid Scale',                 'title': 'Scaling Norm',      'vis': lambda x: vis_gray(x).reshape(grid_shape)},
    #     {'key': 'out_observe',                'title': 'Out Observe',       'vis': lambda x: vis_grad(x.float()).reshape(grid_shape)},
    #     {'key': 'radii',                      'title': 'Radii',             'vis': lambda x: vis_grad(x.float()).reshape(grid_shape)},
    #     {'key': 'Normals',          'title': 'Normals From Depths', 'vis': lambda x: vis_normal(x.float()).reshape(grid_shape)},
    #     {'key': 'Depths',          'title': 'Depths', 'vis': lambda x: vis_grad(x.float()).reshape(cp_grid_shape)},
    #     {'key': 'CP Spherical Harmonics',          'title': 'CP Spherical Harmonics', 'vis': lambda x: vis_sh(x.float()).reshape(cp_grid_shape)},
    #     {'key': 'Backfaces',          'title': 'Backfaces Filter', 'vis': lambda x: vis_gray(x.float()).reshape(grid_shape)},
    # ]
    to_include = []
    for i in range(len(nurbs.surfaces)):
        to_include.extend([
            f'weights_map_per_view{i}',
            f'norm_grid_{i}',
            f'sh_grid_{i}',
                           # f'sh_cpt_{i}',
                           # f'visibility_cp_{i}'
                           ])


    grid_plots = []
    for p in grid_plots_all:
        if p['key'] in to_include:
            grid_plots.append(p)

    # Optional density intensity
    if uid is not None:
        try:
            density_u = getattr(nurbs, 'partition_density_u', None)
            density_v = getattr(nurbs, 'partition_density_v', None)
            if density_u is not None and density_v is not None:
                du = density_u.cpu().numpy()[uid] / (density_u[uid].max() + 1e-6)
                dv = density_v.cpu().numpy()[uid] / (density_v[uid].max() + 1e-6)
                density_2d = np.outer(du, dv)
                grid_plots.append({'img': density_2d, 'title': 'Density Intensity', 'vis': None})
        except:
            pass

    # ------------------------------------------------------------------ #
    # Collect images
    # ------------------------------------------------------------------ #
    def collect(plist, rdict):
        imgs, titles = [], []
        for p in plist:
            if 'key' in p and p['key'] in rdict:
                data = rdict[p['key']]
            elif 'img' in p:
                data = p['img']
            else:
                continue
            img = p['vis'](data) if p['vis'] else data
            imgs.append(img)
            titles.append(p['title'])
        return imgs, titles

    img_images, img_titles   = collect(image_plots, render_dict)
    grid_images, grid_titles = collect(grid_plots,   nurbs_prop_dict)

    if not img_images and not grid_images:
        return

    # ------------------------------------------------------------------ #
    # Preprocessing + resizing
    # ------------------------------------------------------------------ #
    gt_h, gt_w = (gt_image.shape[-2:] if isinstance(gt_image, torch.Tensor) else gt_image.shape[:2])

    def preprocess(img, target_size=None):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()

        img = np.clip(img, 0, 1) if img.dtype != np.uint8 else img.astype(np.float32)/255.0

        if target_size is not None:
            h, w = img.shape[:2]
            if (h, w) != target_size:
                c = 1 if img.ndim == 2 else img.shape[2]
                t = torch.from_numpy(img)
                t = t.permute(2, 0, 1) if c > 1 else t.unsqueeze(0)
                t = F.interpolate(t.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
                img = t.permute(1, 2, 0).numpy() if c > 1 else t[0].numpy()
        return img

    # Preprocess all
    img_processed   = [preprocess(im, (gt_h, gt_w)) for im in img_images]
    grid_processed  = [preprocess(im) for im in grid_images]

    # Find best near-square layout for grid plots
    def best_grid_layout(n):
        if n <= 1: return 1, 1
        best = (1, n)
        min_diff = float('inf')
        for rows in range(1, int(n**0.5) + 2):
            cols = (n + rows - 1) // rows
            diff = abs(cols - rows)
            waste = rows * cols - n
            if diff < min_diff or (diff == min_diff and waste < (best[0] * best[1] - n)):
                min_diff = diff
                best = (rows, cols)
        return best[0], best[1]

    grid_rows, grid_cols = best_grid_layout(len(grid_processed))
    im_rows, im_cols = best_grid_layout(len(img_images))

    # Target uniform size for grid plots (use largest dimension)
    if grid_processed:
        max_h = max(im.shape[0] for im in grid_processed)
        max_w = max(im.shape[1] for im in grid_processed)
        target_grid_size = (max_h, max_w)
        grid_processed = [preprocess(im, target_grid_size) for im in grid_images]  # re-process with final size

    # ------------------------------------------------------------------ #
    # Final figure with GridSpec
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(24, 10))  # Wide, balanced
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, width_ratios=[1, 1])

    # Left: Image renders (vertical stack)
    left_gs = gridspec.GridSpecFromSubplotSpec(im_rows, im_cols, subplot_spec=outer_gs[0],
                                               wspace=0.01, hspace=0.1)
    try:
        for i, (img, ttl) in enumerate(zip(img_processed, img_titles)):

            ax = fig.add_subplot(left_gs[i // im_cols, i % im_cols])
            cmap = 'jet' if 'Depth' in ttl else ('gray' if img.ndim == 2 or img.shape[2] == 1 else None)
            ax.imshow(img.squeeze() if img.ndim == 3 and img.shape[2] == 1 else img, cmap=cmap)
            ax.set_title(ttl, fontsize=13, pad=10)
            ax.axis('off')
    except Exception as e:
        print(f"Error plotting image renders: {e}")
        pass

    # Right: Grid plots in compact grid
    if grid_processed:
        right_gs = gridspec.GridSpecFromSubplotSpec(grid_rows, grid_cols,
                                                    subplot_spec=outer_gs[1],
                                                    wspace=0.08, hspace=0.3)
        for i, (img, ttl) in enumerate(zip(grid_processed, grid_titles)):
            ax = fig.add_subplot(right_gs[i // grid_cols, i % grid_cols])
            cmap = 'gray' if img.ndim == 2 or img.shape[2] == 1 else None
            ax.imshow(img.squeeze() if img.ndim == 3 and img.shape[2] == 1 else img, cmap=cmap)
            ax.set_title(ttl, fontsize=11)
            ax.axis('off')

        # Turn off unused subplots
        total_cells = grid_rows * grid_cols
        for j in range(len(grid_processed), total_cells):
            ax = fig.add_subplot(right_gs[j // grid_cols, j % grid_cols])
            ax.axis('off')

    # ------------------------------------------------------------------ #
    # Final polish
    # ------------------------------------------------------------------ #
    suptitle = title + " — Renders (left) | Grid Fields (right)" if title else "Render vs Grid Diagnostics"
    fig.suptitle(suptitle, fontsize=18, y=0.98)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    # Always release the figure: repeated calls in the training loop
    # otherwise accumulate matplotlib state and leak memory.
    plt.close(fig)
