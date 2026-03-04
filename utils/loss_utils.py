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
from typing import List

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def multi_surf_feature_consistency(feature_vecs: List[torch.Tensor], weight_maps: List[torch.Tensor]=None, radius: int = 1, sigma: float = 1,
                        dist_type: str = 'l2', skip=True) -> torch.Tensor:
    losses = []
    if skip:
        return torch.tensor(0.0, device='cuda')
    if weight_maps is None:
        weight_maps = [None] * len(feature_vecs)
    for f, w in zip(feature_vecs, weight_maps):
        losses.append(feature_consistency(f, w, radius, sigma, dist_type))
    return torch.stack(losses).mean()
def cossim_loss_multisurf(feature_vecs: List[torch.Tensor], weight_maps: List[torch.Tensor]=None, w=1.0) -> torch.Tensor:
    if w <= 0.0:
        return torch.tensor(0.0, device='cuda')
    losses = []
    if weight_maps is None:
        weight_maps = [None] * len(feature_vecs)
    for f, w in zip(feature_vecs, weight_maps):
        losses.append(cosine_similarity_geodesic_loss(f, w))
    return torch.stack(losses).mean()
def param_surf_deviation(learned_vecs: List[torch.Tensor], geo_vecs: List[torch.Tensor], weight_maps: List[torch.Tensor]=None, w=0.0 ) -> torch.Tensor:
    if w <= 0.0:
        return torch.tensor(0.0, device='cuda')
    losses = []
    # if weight_maps is None:
    for l, g, w in zip(learned_vecs, geo_vecs, weight_maps if weight_maps is not None else [None] * len(learned_vecs)):
        losses.append(cosine_similarity_loss(l, g, w))
    return torch.stack(losses).mean()

def scale_surf_deviation(learned_vecs: List[torch.Tensor], grid_vecs: List[torch.Tensor], weight_maps: List[torch.Tensor]=None) -> torch.Tensor:
    losses = []
    # if weight_maps is None:
    for l, g, w in zip(learned_vecs, grid_vecs, weight_maps if weight_maps is not None else [None]*len(learned_vecs)):
        losses.append(cosine_similarity_loss(l, g, w))
    return torch.stack(losses).mean()
def cosine_similarity_loss(vec_a: torch.Tensor, vec_b: torch.Tensor, weight_maps:torch.Tensor = None) -> torch.Tensor:
    vec_a = F.normalize(vec_a, dim=-1)
    vec_b = F.normalize(vec_b, dim=-1)
    weight_maps = F.normalize(weight_maps, dim=-1) if weight_maps is not None else 1.0
    cossim = vec_a * vec_b * weight_maps
    cossim_sum = cossim.sum(dim=-1)
    loss = ((1 - cossim_sum)).mean()
    return loss
def l1_loss1(vec_a: torch.Tensor, vec_b: torch.Tensor=None) -> torch.Tensor:
    loss = (vec_a - vec_b).abs().sum(dim=-1).mean()
    return loss
def l1_geodesic_loss(features_grid: torch.Tensor) -> torch.Tensor:
    loss_u = (features_grid[1:, ...] - features_grid[:-1, ...])
    loss_v = (features_grid[:, 1:, ...] - features_grid[:,:-1, ...])
    return loss_u.abs().sum(-1).mean() + loss_v.abs().sum(-1).mean()
def cosine_similarity_geodesic_loss(features_grid: torch.Tensor, weight_maps: torch.Tensor=None, radius: int = 1, sigma: float = 1) -> torch.Tensor:
    features_grid = F.normalize(features_grid, dim=-1)
    weight_maps = F.normalize(weight_maps, dim=-1) if weight_maps is not None else 1.0
    assert features_grid.dim() == 3, "features_grid must be (H, W, C)"
    cos_u = features_grid[1:, :, :] * features_grid[:-1, :, :]
    cos_v = features_grid[:, 1:, :] * features_grid[:, :-1, :]
    weight_maps = weight_maps.squeeze(-1) if weight_maps.dim() == 3 else weight_maps
    cos_u = cos_u.sum(dim=-1)  # (H-1, W)
    cos_v = cos_v.sum(dim=-1)  # (H, W-1)
    w_u = weight_maps[1:, :] * weight_maps[:-1, :]
    w_v = weight_maps[:, 1:] * weight_maps[:, :-1]
    loss_u = ((1 - cos_u)*w_u).mean()
    loss_v = ((1 - cos_v)*w_v).mean()

    return (loss_u + loss_v) * 0.5
def feature_consistency(feature_vecs: torch.Tensor, weight: torch.Tensor=None, radius: int = 1, sigma: float = 1,
                        dist_type: str = 'l2') -> torch.Tensor:
    """
    Computes local feature smoothness loss on grid using conv2d for 'l2'/'cosine' (efficient, vectorized).
    Falls back to unfold for 'l1' (disclaimer: no pure conv2d without approximation).

    Args:
        feature_vecs: (H, W, C) tensor of features (e.g., normals).
        radius: Neighborhood radius (int).
        sigma: Gaussian std (float; large for uniform weights).
        dist_type: 'l2' (squared), 'cosine' (1 - dot), 'l1' (abs sum).

    Returns:
        Scalar loss tensor.
    """
    if weight is None:
        weight = torch.ones_like(feature_vecs[..., 0:1])
    H, W, C = feature_vecs.shape
    device = feature_vecs.device
    r = radius
    kh = kw = 2 * r + 1

    # Gaussian kernel (kh, kw)
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device))
    dist_grid = xx ** 2 + yy ** 2
    weights = torch.exp(-dist_grid.float() / (2 * sigma ** 2))
    weights[r, r] = 0.0  # Exclude center
    kernel = weights.view(1, 1, kh, kh)  # (1, 1, kh, kw)

    # Constant sums
    W_sum = weights.sum()  # Sum of weights (excludes center)
    N = kh ** 2 - 1  # Number of non-center neighbors (constant)

    if dist_type in ['l2', 'cosine']:
        # Pad and prepare input: (1, C, H_p, W_p)
        padded = F.pad(feature_vecs.permute(2, 0, 1), (r, r, r, r), mode='replicate').unsqueeze(0)  # (1, C, H+2r, W+2r)

        # Grouped kernel: (C, 1, kh, kw)
        kernel_grouped = kernel.repeat(C, 1, 1, 1)

        # conv(features)
        sum_w_neigh = F.conv2d(padded, kernel_grouped, groups=C)  # (1, C, H, W)

        if dist_type == 'l2':
            # conv(features**2)
            features_sq = feature_vecs.pow(2)
            padded_sq = F.pad(features_sq.permute(2, 0, 1), (r, r, r, r), mode='replicate').unsqueeze(0)
            sum_w_neigh_sq = F.conv2d(padded_sq, kernel_grouped, groups=C)  # (1, C, H, W)

            # Terms: reshape features to (1, C, H, W)
            features_resh = feature_vecs.unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)

            term1 = sum_w_neigh_sq
            term2 = -2 * features_resh * sum_w_neigh
            term3 = features_resh.pow(2) * W_sum
            weighted_sum_dists = (term1 + term2 + term3).sum(dim=1)  # (1, H, W)

        elif dist_type == 'cosine':
            # For cosine: W_sum - sum_c center^c * sum_w neigh^c
            features_resh = feature_vecs.unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
            sum_dots = (features_resh * sum_w_neigh).sum(dim=1)  # (1, H, W)
            weighted_sum_dists = W_sum - sum_dots  # Since sum w (1 - dot) = W_sum - sum w dot

        # Per-point avg and loss
        per_point_avg = weighted_sum_dists / N

    elif dist_type == 'l1':
        # Fallback to unfold (disclaimer: pure conv2d not feasible without smooth approx)
        padded = F.pad(feature_vecs.permute(2, 0, 1), (r, r, r, r), mode='replicate')  # (C, H+2r, W+2r)
        unfolded = padded.unfold(1, kh, 1).unfold(2, kh, 1)  # (C, H, W, kh, kw)
        neighborhoods = unfolded.permute(1, 2, 3, 4, 0).reshape(H, W, kh * kw, C)  # (H, W, K, C)
        centers = feature_vecs.unsqueeze(2)  # (H, W, 1, C)

        dists = (neighborhoods - centers).abs().sum(-1)  # (H, W, K)

        # Weights and mask (same as original)
        weights = torch.exp(-dist_grid.flatten() / (2 * sigma ** 2)).unsqueeze(0).unsqueeze(0)  # (1,1,K)
        center_idx = kh * kw // 2
        mask = torch.ones_like(dists, dtype=torch.bool)
        mask[..., center_idx] = False

        dists = dists * weights
        masked_dists = dists.where(mask, torch.zeros_like(dists))
        valid_counts = mask.sum(-1).clamp(min=1)
        per_point_avg = masked_dists.sum(-1) / valid_counts



    else:
        raise ValueError(f"Unsupported dist_type: {dist_type}")
    per_point_avg = per_point_avg * weight.squeeze(-1)  # Apply weight map
    loss = per_point_avg.mean()
    return loss