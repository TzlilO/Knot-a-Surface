import copy
import dataclasses
import random
from typing import NamedTuple, Optional, List

import geomdl.BSpline
from attr import dataclass
from matplotlib import pyplot as plt
import torch
from scipy.special import comb

from joblib import Memory

from scipy.spatial import KDTree
from torch import nn

from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud
memory = Memory(location='cache_directory', verbose=0)
from scipy.interpolate import griddata


def build_global_control_grid(patches, nU, nV):
    """
    Given a tensor of patches of shape (nU*nV, 4, 4, D),
    build a global control grid of shape (nU*3+1, nV*3+1, D)
    such that each patch at grid position (i, j) corresponds to the window:
        rows: [i*3 : i*3+4]
        cols: [j*3 : j*3+4]
    Overlapping regions are averaged.

    Args:
      patches: torch.Tensor of shape (N, 4, 4, D) with N = nU*nV.
      nU, nV: integers defining the grid dimensions.

    Returns:
      global_grid: torch.Tensor of shape (nU*3+1, nV*3+1, D)
    """
    N, H, W, D = patches.shape  # Here H=W=4
    # Initialize accumulators for sum and count.
    global_rows = nU * 3 + 1
    global_cols = nV * 3 + 1
    global_sum = torch.zeros(global_rows, global_cols, D, device=patches.device, dtype=patches.dtype)
    global_count = torch.zeros(global_rows, global_cols, device=patches.device, dtype=patches.dtype)

    # Reshape patches into (nU, nV, 4, 4, D)
    patches = patches.view(nU, nV, H, W, D)
    # For each patch (i,j), its global location is rows: i*3 to i*3+3, cols: j*3 to j*3+3.
    # We can compute global indices for rows and cols in a vectorized way.
    # Create a grid of indices for a patch: rows 0,1,2,3 and cols 0,1,2,3.
    row_idx_patch = torch.arange(H, device=patches.device).reshape(1, H)  # (1,4)
    col_idx_patch = torch.arange(W, device=patches.device).reshape(1, W)  # (1,4)
    # For patches, the global row indices: i*3 + row_idx_patch, for i in [0, nU)
    patch_row_idx = torch.arange(nU, device=patches.device).reshape(nU, 1, 1) * 3  # (nU,1,1)
    patch_col_idx = torch.arange(nV, device=patches.device).reshape(1, nV, 1) * 3  # (1, nV,1)
    # Broadcast to get full indices: For each patch, rows and cols:
    global_rows_idx = patch_row_idx + row_idx_patch.view(1, 1, H)  # shape: (nU, 1, 4)
    global_cols_idx = patch_col_idx + col_idx_patch.view(1, 1, W)  # shape: (1, nV, 4)
    # Now, expand to (nU, nV, 4, 4)
    global_rows_idx = global_rows_idx.expand(nU, nV, H)
    global_cols_idx = global_cols_idx.expand(nU, nV, W)

    # Iterate over patch indices (vectorized using advanced indexing)
    # Flatten patches and indices to shape (N, 4, 4) and (N,)
    patches_flat = patches.view(-1, H, W, D)  # (N, 4, 4, D)
    global_rows_idx = global_rows_idx.view(-1, H)  # (N, 4)
    global_cols_idx = global_cols_idx.view(-1, W)  # (N, 4)

    N = patches_flat.shape[0]
    for i in range(N):
        # Use the computed indices to scatter-add each patch into the global grid.
        r_idx = global_rows_idx[i]  # (4,)
        c_idx = global_cols_idx[i]  # (4,)
        patch = patches_flat[i]  # (4,4,D)
        # Outer product of indices to create 2D index grid.
        for r in r_idx:
            for c in c_idx:
                # Find local index inside the patch:
                # r_local = index of r in r_idx; same for c.
                r_local = (r_idx == r).nonzero(as_tuple=True)[0].item()
                c_local = (c_idx == c).nonzero(as_tuple=True)[0].item()
                global_sum[r, c] += patch[r_local, c_local]
                global_count[r, c] += 1

    # Avoid division by zero and compute the mean.
    global_grid = global_sum / (global_count.unsqueeze(-1) + 1e-8)
    return global_grid


def extract_patch_from_global(global_grid, i, j):
    """
    Given a global control grid of shape (nU*3+1, nV*3+1, D),
    extract the patch at position (i,j) (0-indexed) as the 4x4 window:
        rows: [i*3 : i*3+4]
        cols: [j*3 : j*3+4]
    """
    start_row = i * 3
    start_col = j * 3
    return global_grid[start_row:start_row + 4, start_col:start_col + 4, :]



def soft_morphological_closing(mask_logits: torch.Tensor,
                               kernel_size: int = 3,
                               tau: float = 10.0):
    """
    Differentiable morphological closing (dilation ∘ erosion) applied
    to a *logit* mask.

    Parameters
    ----------
    mask_logits : (H, W) float tensor
        Raw logits (pre-sigmoid) for foreground.
    kernel_size : int, default = 3
        Size of the square structuring element (must be odd).
    tau : float, default = 10
        Temperature for the soft-max/soft-min.  Larger τ → harder morphology.

    Returns
    -------
    closed_logits : (H, W) float tensor
        Logits after (soft) closing.  Still differentiable.
    """
    if mask_logits.ndim == 2:            # allow shape (1,H,W)
        # mask_logits = mask_logits.squeeze(0)
        mask_logits = mask_logits.unsqueeze(0)
    pad = kernel_size // 2
    # Dilation ≈ local *soft-max*
    dil_logits = F.max_pool2d(mask_logits.unsqueeze(0), kernel_size,
                              stride=1, padding=pad)
    # Use temperature to keep gradient alive
    dil_logits = torch.logsumexp(mask_logits / tau,
                                 dim=0, keepdim=True) * tau

    # Erosion ≈ soft-min  ==  soft-max of (-x)
    ero_logits = -torch.logsumexp(-dil_logits / tau, dim=0, keepdim=True) * tau
    return ero_logits.squeeze(0)

def clean_bad_faces(verts: torch.Tensor,
                    faces: torch.LongTensor,
                    min_area: float = 1e-12,
                    max_aspect: float = 50.0):
    """
    Remove faces that
      • touch a vertex with NaN / Inf
      • have area < min_area   (degenerate)
      • have edge-length ratio > max_aspect   (sliver)
    Returns verts' (compact), faces'
    """
    device = verts.device
    v0, v1, v2 = [verts[faces[:, i]] for i in range(3)]

    # 1) finite-vertex mask
    finite_mask = (
        torch.isfinite(v0).all(1) &
        torch.isfinite(v1).all(1) &
        torch.isfinite(v2).all(1)
    )

    # 2) tiny-area mask
    area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
    area_mask = area > min_area

    # 3) sliver-triangle mask
    e01 = (v1 - v0).norm(dim=-1)
    e12 = (v2 - v1).norm(dim=-1)
    e20 = (v0 - v2).norm(dim=-1)
    max_edge = torch.stack([e01, e12, e20], 1).max(1).values
    min_edge = torch.stack([e01, e12, e20], 1).min(1).values
    aspect_mask = (max_edge / (min_edge + 1e-12)) < max_aspect

    keep_face = finite_mask & area_mask & aspect_mask
    faces = faces[keep_face]

    # compact verts
    used, inv = torch.unique(faces, sorted=True, return_inverse=True)
    verts  = verts[used]
    faces  = inv.view(-1, 3)
    return verts, faces

def topk_connectedness(mask_probs: torch.Tensor,
                       k: int = 1,
                       radius: int = 10,
                       tau: float = 10.0):
    """
    Softly keep the k largest “components” using a continuous proxy:
    each pixel votes for its neighbourhood; neighbourhood mass is
    compared against a soft top-k threshold.

    Parameters
    ----------
    mask_probs : (H, W) float tensor in [0,1]
    k          : how many components to keep (softly)
    radius     : defines neighbourhood window (2*radius+1)²
    tau        : temperature for the soft-threshold

    Returns
    -------
    kept_mask_probs : (H, W) float tensor
    """
    # H, W = mask_probs.shape
    mask_probs = mask_probs.squeeze()
    # Sum of probabilities in a (2R+1)×(2R+1) window
    neighbourhood_mass = F.avg_pool2d(mask_probs.unsqueeze(0).unsqueeze(0),
                                      kernel_size=2*radius+1,
                                      stride=1, padding=radius) * (2*radius+1)**2
    # Flatten and find (soft) k-th largest mass
    flat = neighbourhood_mass.flatten()
    topk_vals, _ = torch.topk(flat, k)
    kth = topk_vals[-1]                  # hard value
    # Soft indicator: sigmoid around the k-th value
    gate = torch.sigmoid((neighbourhood_mass - kth) * tau)
    return (mask_probs * gate.squeeze(0).squeeze(0)).clamp(max=1.0)


import numpy as np                      # add near the other imports
import scipy.ndimage as ndi             # make sure SciPy is available
def refine_mask_morphology_diff(mask_logits: torch.Tensor,
                                kernel_size: int = 5,
                                tau_close: float = 8.0,
                                min_cc: int = 16, thresh=0.05):
    """
    1) soft morphological closing in **logit** space
    2) drop connected components with < `min_cc` pixels
    Returns a (H,W) probability map.
    """
    # --- canonical (1,H,W) layout ----------------------------------------
    if mask_logits.ndim == 2:
        logits = mask_logits.unsqueeze(0)          # (1,H,W)
    elif mask_logits.ndim == 3 and mask_logits.shape[0] == 1:
        logits = mask_logits
    else:
        raise ValueError("Expect (H,W) or (1,H,W) mask")

    # --- differentiable closing in logits --------------------------------
    closed = soft_morphological_closing(logits,
                                        kernel_size=kernel_size,
                                        tau=tau_close)        # (1,H,W)

    # --- convert to binary, run CC on CPU --------------------------------
    probs  = torch.sigmoid(closed)               # (H,W)
    binary = (probs > thresh).cpu().numpy()

    labels, n_lab = ndi.label(binary)               # connected comps
    keep = np.zeros_like(labels, dtype=bool)
    for lab in range(1, n_lab + 1):
        if (labels == lab).sum() >= min_cc:
            keep |= (labels == lab)

    keep = torch.from_numpy(keep).to(probs.device).float()  # (H,W)

    # --- mask the *probabilities* not the logits -------------------------
    refined = probs * keep                           # soft mask
    return refined
def refine_mask_morphology(mask_2d: torch.Tensor,
                           k: int = 16,
                           kernel_size: int = 3) -> torch.Tensor:
    """
    Fill small holes + connect close islands with morphological closing,
    then keep the `k` largest connected components.

    Parameters
    ----------
    mask_2d : (H,W) bool / uint8 tensor   – candidate refinement mask
    k       : int   – how many largest CCs to keep
    kernel_size : int – structuring-element size for closing (odd)

    Returns
    -------
    refined_mask : (H,W) bool tensor
    """
    if mask_2d.ndim == 3:
        mask_2d = mask_2d.squeeze()
    device = mask_2d.device
    H, W   = mask_2d.shape
    # ------------------------------------------------------------
    # 1) morphological closing in pure torch
    # ------------------------------------------------------------
    kernel = torch.ones((1, 1, kernel_size, kernel_size),
                        dtype=torch.float32, device=device)
    pad    = kernel_size // 2
    x      = mask_2d.float().view(1, 1, H, W)

    # dilation  ( >0  → 1 )
    dil = (F.conv2d(x, kernel, padding=pad) > 0).float()
    # erosion  ( ==kernel.numel() → 1 )
    clo = (F.conv2d(dil, kernel, padding=pad) == kernel.numel()).float()
    closed_mask = clo[0, 0].bool()          # (H,W)

    # ------------------------------------------------------------
    # 2) connected-component labelling on CPU (union-find)
    # ------------------------------------------------------------
    # We move to CPU for simplicity; cost is tiny (H,W ≲ 256²)
    cm_np = closed_mask.cpu().numpy().astype('uint8')
    import scipy.ndimage as ndi
    labeled, num_cc = ndi.label(cm_np, structure=ndi.generate_binary_structure(2, 2))

    if num_cc == 0:
        return closed_mask          # nothing to keep
    torch.tensor(closed_mask)
    # area of each component (label 0 = background)
    areas = ndi.sum(cm_np, labeled, index=range(1, num_cc + 1))
    # take k largest (indices are 1-based in ´labeled´)
    keep_lbl = np.argsort(areas)[-k:] + 1
    keep_mask = np.isin(labeled, keep_lbl)

    return torch.from_numpy(keep_mask).to(device=device, dtype=torch.bool)


def init_patches_from_depth(depth_map: torch.Tensor, stride: int = 3):
    """
    Convert a depth image into a batch of 4×4 NURBS-control-point grids
    using a sliding window of size 4 and stride 3 (i.e.\ neighbours overlap
    by one row/column, identical to the grid-aggregation step).

    Each control point stores (x, y, z), where (x, y) are pixel coordinates
    (float) and z is the depth value.  The result is shaped **(N, 4, 4, 3)**.

    Parameters
    ----------
    depth_map : torch.Tensor
        Depth with shape (H, W) or (1, H, W) in world units.
    stride : int, default = 3
        Sliding-window stride in pixels.

    Returns
    -------
    torch.Tensor
        Tensor of control grids with shape (N, 4, 4, 3).
    """
    # Ensure a 2-D array
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    if depth_map.dim() != 2:
        raise ValueError("depth_map must have shape (H, W) or (1, H, W)")

    H, W = depth_map.shape
    device = depth_map.device
    dtype  = depth_map.dtype

    # Pre-compute pixel coordinate grids
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    patches = []
    for top in range(0, H - 3, stride):
        for left in range(0, W - 3, stride):
            z = depth_map[top : top + 4, left : left + 4]
            x = xs[top   : top + 4, left : left + 4]
            y = ys[top   : top + 4, left : left + 4]
            patches.append(torch.stack((x, y, z), dim=-1))   # 4×4×3

    if patches:
        return torch.stack(patches, dim=0)                  # N×4×4×3
    # Image was smaller than 4×4 – return an empty tensor
    return torch.empty((0, 4, 4, 3), dtype=dtype, device=device)


def reduce_grads_to_tangent_direction(cp: torch.Tensor, grad: torch.Tensor,
                                      reduction_factor: float = 1.0) -> torch.Tensor:
    """
    Reduces the gradient of control points by projecting out the component in the normal direction.

    The function assumes:
      - cp: the current control points for a set of patches, with shape (N, H, W, 3),
            where H and W are the patch grid dimensions (e.g. 4 for a 4x4 patch).
      - grad: the gradient tensor with the same shape (N, H, W, 3).
      - reduction_factor: scalar in [0, 1] indicating how much of the normal component to remove.
                          A value of 1.0 removes the entire normal component.

    Returns:
      new_grad: The modified gradient tensor (with shape (N, H, W, 3)) that is projected
                onto the approximate tangent plane of each patch.

    Approach:
      1. Convert the control points from shape (N, H, W, 3) to (N, 3, H, W) so that we can
         use standard convolution-like padding.
      2. Pad the tensor using replication (so that boundaries are handled gracefully).
      3. Compute central differences along the horizontal (u) and vertical (v) directions.
      4. Compute the cross product of these two differences to estimate a local normal.
      5. Normalize the normal (avoiding division by zero).
      6. Permute the normal back to (N, H, W, 3) so it can be used to compute the dot product
         with the gradient.
      7. Compute the component of the gradient in the normal direction and subtract (or scale‐reduce)
         it from the gradient.
    """
    # cp: (N, H, W, 3)
    N, H, W, _ = cp.shape

    # Permute to (N, 3, H, W) for padding and central difference computation.
    cp_permuted = cp.permute(0, 3, 1, 2)  # shape: (N, 3, H, W)

    # Pad the tensor by 1 pixel on all sides using replication so that boundaries are handled.
    cp_padded = F.pad(cp_permuted, pad=(1, 1, 1, 1), mode='replicate')  # shape: (N, 3, H+2, W+2)

    # Compute central differences in the u-direction (vertical direction in the grid):
    # Use the difference between the pixel two rows down and two rows up.
    dU = cp_padded[:, :, 2:, 1:-1] - cp_padded[:, :, :-2, 1:-1]  # shape: (N, 3, H, W)

    # Compute central differences in the v-direction (horizontal direction in the grid):
    dV = cp_padded[:, :, 1:-1, 2:] - cp_padded[:, :, 1:-1, :-2]  # shape: (N, 3, H, W)

    # Estimate a local normal as the cross product of dU and dV.
    # The cross product is computed along the channel dimension (dim=1).
    normal = torch.cross(dU, dV, dim=1)  # shape: (N, 3, H, W)

    # Compute the norm (magnitude) of the normal vectors.
    norm_val = normal.norm(dim=1, keepdim=True) + 1e-8  # shape: (N, 1, H, W)

    # Normalize the normal vectors.
    normal_unit = normal / norm_val  # shape: (N, 3, H, W)

    # Permute the normalized normals back to shape (N, H, W, 3)
    normal_unit = normal_unit.permute(0, 2, 3, 1)

    # Ensure the grad is in shape (N, H, W, 3); if not, assume it is already in that shape.
    # Compute the dot product between grad and the normal_unit.
    dot = (grad * normal_unit).sum(dim=-1, keepdim=True)  # shape: (N, H, W, 1)

    # The projection of grad onto the normal direction is: dot * normal_unit.
    # We subtract (or reduce) that component from the original gradient.
    new_grad = grad - reduction_factor * dot * normal_unit

    return new_grad


def precompute_unique_to_patches(unique_control_points, control_point_indices, u_elem=4, v_elem=4):
    num_unique = unique_control_points.shape[0]
    unique_to_patches = [[] for _ in range(num_unique)]

    for i in range(u_elem - 3):
        for j in range(v_elem - 3):
            patch_indices = control_point_indices[i:i + 4, j:j + 4].unique()
            for idx in patch_indices:
                unique_to_patches[idx].append([i, j])

    return unique_to_patches

def bernstein_poly(i, n, t, device='cuda'):
    return torch.tensor(comb(n, i) * (t ** i) * ((1 - t) ** (n - i)), device=device)


def basis_functions(n, t):
    return torch.cat([bernstein_poly(i, n, t).unsqueeze(0) for i in range(n + 1)], dim=0)


def compute_face_points(xyz):
    return (xyz[:, :-1, :-1] + xyz[:, :-1, 1:] + xyz[:, 1:, :-1] + xyz[:, 1:, 1:]) * 0.25


def compute_edge_points(xyz, face_points):
    N, H, W, C = xyz.shape
    edge_points = torch.zeros((N, 2 * H - 1, 2 * W - 1, C), dtype=xyz.dtype, device=xyz.device)
    edge_points[:, 1::2, ::2] = (xyz[:, :-1, :] + xyz[:, 1:, :]) * 0.5
    edge_points[:, ::2, 1::2] = (xyz[:, :, :-1] + xyz[:, :, 1:]) * 0.5
    edge_points[:, 1::2, 1::2] = face_points
    return edge_points


def compute_vertex_points(xyz, face_points, edge_points):
    N, H, W, C = xyz.shape
    vertex_points = torch.zeros((N, 2 * H - 1, 2 * W - 1, C), dtype=xyz.dtype, device=xyz.device)
    vertex_points[:, ::2, ::2] = xyz
    vertex_points[:, 1::2, ::2] = edge_points[:, 1::2, ::2]
    vertex_points[:, ::2, 1::2] = edge_points[:, ::2, 1::2]
    vertex_points[:, 1::2, 1::2] = face_points
    return vertex_points


def catmull_clark_subdivision(xyz: torch.Tensor):
    """
    Given a BSpline patch (N patches in total), where each patch is defined by 4x4 3D control points,
    this function returns 4 new patches for each of the N patches using the Catmull-Clark subdivision algorithm.

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing control points in 3D.

    Returns:
        torch.Tensor: Subdivided 4 new patches of the old patch according to its 3D control points tensor of shape (4N, 4, 4, 3).
    """
    face_points = compute_face_points(xyz)
    edge_points = compute_edge_points(xyz, face_points)
    vertex_points = compute_vertex_points(xyz, face_points, edge_points)

    # Extract 4 new patches from the new grid
    patches = []
    for i in range(2):
        for j in range(2):
            patches.append(vertex_points[:, i * 3:(i + 1) * 3 + 1, j * 3:(j + 1) * 3 + 1])

    return torch.cat(patches, dim=0)


def evaluate_bspline_surface(control_points, u, v):
    """
    Evaluate the B-Spline surface point S(u, v) given the control points.

    Parameters:
        control_points (torch.Tensor): Tensor of shape (4, 4, 3) representing 4x4 control points in 3D.
        u (torch.Tensor): The u parameters in [0, 1] of shape (m,).
        v (torch.Tensor): The v parameters in [0, 1] of shape (n,).

    Returns:
        torch.Tensor: The surface points S(u, v) of shape (m, n, 3).
    """
    m, n = u.shape[0], v.shape[0]
    Bu = torch.stack([basis_functions(3, u_i) for u_i in u], dim=0)  # (m, 4)
    Bv = torch.stack([basis_functions(3, v_i) for v_i in v], dim=0)  # (n, 4)

    surface_points = torch.tensordot(Bu.to('cuda'), torch.tensordot(Bv.to('cuda'), control_points, dims=([1], [1])),
                                     dims=([1], [1]))
    return surface_points


def vectors_to_quaternions(vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of 3D vectors to quaternions that rotate [0, 0, 1] to align with each vector.

    Args:
        vectors: (N, 3) tensor of 3D vectors (surface normals)

    Returns:
        (N, 4) tensor of quaternions in (w, x, y, z) format
    """
    # Ensure input is normalized
    vectors = torch.nn.functional.normalize(vectors, dim=1)

    # Reference vector (z-axis)
    reference = torch.tensor([0.0, 0.0, 1.0], device=vectors.device)

    # Compute dot product between reference and vectors
    dot = torch.sum(vectors * reference.expand_as(vectors), dim=1)

    # Handle special cases where vectors are parallel or anti-parallel to reference
    parallel_mask = torch.abs(dot - 1.0) < 1e-6
    anti_parallel_mask = torch.abs(dot + 1.0) < 1e-6

    # Initialize quaternions tensor
    quaternions = torch.zeros((vectors.shape[0], 4), device=vectors.device)

    # Handle regular cases
    regular_mask = ~(parallel_mask | anti_parallel_mask)
    if torch.any(regular_mask):
        # Compute rotation axis (cross product with reference)
        axis = torch.cross(reference.expand_as(vectors), vectors)
        axis = torch.nn.functional.normalize(axis, dim=1)

        # Compute rotation angle
        angle = torch.acos(torch.clamp(dot[regular_mask], -1.0, 1.0))

        # Convert axis-angle to quaternion
        half_angle = angle * 0.5
        sin_half_angle = torch.sin(half_angle)

        quaternions[regular_mask, 0] = torch.cos(half_angle)  # w
        quaternions[regular_mask, 1:] = axis[regular_mask] * sin_half_angle.unsqueeze(1)

    # Handle parallel vectors (no rotation needed)
    if torch.any(parallel_mask):
        quaternions[parallel_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=vectors.device)

    # Handle anti-parallel vectors (180-degree rotation around any perpendicular axis)
    if torch.any(anti_parallel_mask):
        # Choose x-axis as rotation axis for anti-parallel case
        quaternions[anti_parallel_mask] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=vectors.device)

    return quaternions


import torch
import torch.nn.functional as F


def quaternion_to_normal(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to a corresponding normal vector.
    We assume the quaternion represents a rotation and extract the rotated z-axis.

    Args:
        q (torch.Tensor): Tensor of shape (P, 4) with quaternions in [w, x, y, z] format.

    Returns:
        torch.Tensor: Tensor of shape (P, 3) representing the rotated z-axis.
    """
    # Normalize the quaternion.
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=1)  # each shape (P,)

    # Convert to a rotation matrix using standard formulas.
    # The rotated z-axis (third column) is computed as:
    #   n_x = 2*(x*z + w*y)
    #   n_y = 2*(y*z - w*x)
    #   n_z = 1 - 2*(x*x + y*y)
    n_x = 2 * (x * z + w * y)
    n_y = 2 * (y * z - w * x)
    n_z = 1 - 2 * (x * x + y * y)
    normals = torch.stack([n_x, n_y, n_z], dim=1)
    return normals

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions (N, 4) in [w, x, y, z] format into rotation matrices (N, 3, 3).
    """
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=1)
    N = q.shape[0]
    R = torch.empty((N, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    return R


def normals_to_quaternions2(normals: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """
    Converts a batch of normal vectors to quaternions representing the rotation from the reference vector [0, 0, 1].

    Args:
        normals (torch.Tensor): Tensor of shape (N, 3) containing normal vectors.
        eps (float): Small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: Quaternion tensor of shape (N, 4) in (w, x, y, z) format.
    """
    # Ensure normals are of shape (N, 3)
    normals = normals.view(-1, 3)
    assert normals.dim() == 2 and normals.size(1) == 3, "normals must be of shape (N, 3)"
    # Normalize the input normals
    normals_norm = normals.view(-1, 3).norm(dim=1, keepdim=True)
    normals_normalized = normals / (normals_norm + eps)

    # Reference vector (global up)
    v_ref = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
    v_ref = v_ref.expand(normals.size(0), 3)

    # Compute dot product and cross product
    dot = (v_ref * normals_normalized).sum(dim=1, keepdim=True)  # Shape: (N, 1)
    cross = torch.cross(v_ref, normals_normalized, dim=1)        # Shape: (N, 3)

    # Handle cases where normals are nearly opposite to the reference vector
    w = dot + 1.0  # Shape: (N, 1)

    # Identify cases where w is close to zero (normals are nearly opposite to v_ref)
    small_w_mask = w.abs() < eps

    # Initialize quaternion components
    quat = torch.cat([w, cross], dim=1)  # Shape: (N, 4)

    if small_w_mask.any():
        # For normals nearly opposite to v_ref, find an orthogonal vector
        normals_opposite = normals_normalized[small_w_mask.squeeze()]
        # Choose the axis with the smallest absolute value
        abs_normals = normals_opposite.abs()
        min_indices = torch.argmin(abs_normals, dim=1)
        t = torch.zeros_like(normals_opposite)
        t[torch.arange(normals_opposite.size(0)), min_indices] = 1.0

        # Compute orthogonal vector
        orthogonal = torch.cross(normals_opposite, t, dim=1)
        orthogonal = orthogonal / (orthogonal.norm(dim=1, keepdim=True) + eps)

        # Set quaternion components for these cases
        quat[small_w_mask.squeeze(), 0] = 0.0  # Set w to zero
        quat[small_w_mask.squeeze(), 1:] = orthogonal

    return quat
def quaternion_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Compute the quaternion that rotates v1 to v2
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)

    w = torch.sqrt((1.0 + torch.sum(v1 * v2, dim=1)) / 2.0) + eps
    xyz = torch.cross(v1, v2, dim=-1) / (4.0 * w.unsqueeze(1) + eps)

    return torch.cat([w.unsqueeze(1), xyz], dim=1)



def b_spline_basis_function(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
        coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] != knots[i + 1] else 0
        return coef1 * b_spline_basis_function(i, k - 1, t, knots) + coef2 * b_spline_basis_function(i + 1, k - 1, t,
                                                                                                     knots)

def b_spline_basis_functions_and_derivatives(degree, knots, t_values, device='cuda'):
    """
    Compute B-spline basis functions and their first and second derivatives for a given set of knots and parameter values.

    Parameters:
        degree (int): Degree of the B-spline basis functions.
        knots (torch.Tensor): Knot vector.
        t_values (torch.Tensor): Parameter values at which to evaluate the basis functions.

    Returns:
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the basis function values.
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the first derivative values.
        torch.Tensor: Tensor of shape (len(t_values), len(knots) - degree - 1) containing the second derivative values.
    """
    num_knots = len(knots)
    num_basis = num_knots - degree - 1
    num_t_values = len(t_values)

    # Initialize the table of basis function values
    N = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    dN = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)
    ddN = torch.zeros((num_t_values, num_knots - 1, degree + 1), dtype=torch.float32, device=knots.device)

    # Compute the zeroth-degree basis functions
    for i in range(num_knots - 1):
        N[:, i, 0] = ((knots[i] <= t_values) & (t_values < knots[i + 1])).float()

    # Compute the higher-degree basis functions iteratively
    for k in range(1, degree + 1):
        for i in range(num_knots - k - 1):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            coef1 = (t_values - knots[i]) / denom1 if denom1 != 0 else torch.zeros_like(t_values)
            coef2 = (knots[i + k + 1] - t_values) / denom2 if denom2 != 0 else torch.zeros_like(t_values)

            dcoef1 = 1.0 / denom1 if denom1 != 0 else torch.zeros_like(t_values)
            dcoef2 = -1.0 / denom2 if denom2 != 0 else torch.zeros_like(t_values)

            N[:, i, k] = coef1 * N[:, i, k - 1] + coef2 * N[:, i + 1, k - 1]
            dN[:, i, k] = (dcoef1 * N[:, i, k - 1] + coef1 * dN[:, i, k - 1] + dcoef2 * N[:, i + 1, k - 1]
                           + coef2 * dN[:,  i + 1,  k - 1])




            ddN[:, i, k] = dcoef1 * dN[:, i, k - 1] + coef1 * ddN[:, i, k - 1] + dcoef2 * dN[:, i + 1,
                                                                                          k - 1] + coef2 * ddN[:, i + 1,
                                                                                                           k - 1]

    return N[:, :num_basis, degree].to(device).unsqueeze(0), dN[:, :num_basis, degree].to(device).unsqueeze(0), ddN[:, :num_basis, degree].to(device).unsqueeze(0)


def build_control_point_index_mapping1(mapping):
    """
    Build a Python dictionary that maps each unique control point index (from 0 to num_unique-1)
    to a list of tuples (patch_idx, i, j) indicating where that control point is used.

    Args:
        mapping (torch.Tensor): A tensor of shape (N, 4, 4) of unique control point indices.

    Returns:
        dict: Keys are unique indices (int) and values are lists of (patch_idx, i, j).
    """
    mapping_np = mapping.cpu().numpy()
    index_dict = {}
    N, h, w = mapping_np.shape
    for p in range(N):
        for i in range(h):
            for j in range(w):
                idx = int(mapping_np[p, i, j])
                if idx not in index_dict:
                    index_dict[idx] = []
                index_dict[idx].append((p, i, j))
    return index_dict


def initialize_nurbs_from_colmap2(point_cloud: torch.Tensor, num_u: int=16, num_v: int=16,
                                 shrink_factor: float = .5) -> torch.Tensor:
    """
    Initialize a NURBS surface from a COLMAP scene.

    Given a 3D point cloud (P, 3) and desired numbers of patches along u and v (num_u, num_v),
    this function computes a global grid of control points covering the point cloud’s bounding box.
    Optionally, the bounding box can be shrunk toward its center by the shrink_factor.

    The global grid has shape ((num_u*3 + 1), (num_v*3 + 1), 3) and is then subdivided into patches
    of shape (4, 4, 3) with adjacent patches sharing control points.

    Args:
        point_cloud (torch.Tensor): Tensor of shape (P, 3) with 3D points.
        num_u (int): Number of patches along the u (first) direction.
        num_v (int): Number of patches along the v (second) direction.
        shrink_factor (float): Factor in (0,1] to shrink the bounding box. 1.0 means no shrinkage.

    Returns:
        control_points (torch.Tensor): Tensor of shape (num_u*num_v, 4, 4, 3) containing
            the initial control points for the NURBS surface.
    """
    # Compute the bounding box of the point cloud.
    min_xyz, _ = torch.min(point_cloud, dim=0)
    max_xyz, _ = torch.max(point_cloud, dim=0)

    # Optionally, add a margin if desired (here we add a 5% margin)
    margin = 0.05 * (max_xyz - min_xyz)
    min_xyz = min_xyz - margin
    max_xyz = max_xyz + margin

    # Compute the center of the bounding box.
    center = (min_xyz + max_xyz) / 2.0

    # Apply the shrink factor: new extents will be shrink_factor times the original extents.
    half_extent = (max_xyz - min_xyz) / 2.0
    half_extent = half_extent * shrink_factor  # shrink the half-extents
    new_min = center - half_extent
    new_max = center + half_extent

    # Determine global grid dimensions:
    # Each patch is 4x4 and adjacent patches share 3 control points along boundaries.
    grid_rows = num_u * 3 + 1
    grid_cols = num_v * 3 + 1

    # Create a uniform grid in the new (shrunken) bounding box.
    u_vals = torch.linspace(new_min[0], new_max[0], steps=grid_rows, device=point_cloud.device)
    v_vals = torch.linspace(new_min[1], new_max[1], steps=grid_cols, device=point_cloud.device)
    # For the z coordinate, you might linearly interpolate similarly.
    w_vals = torch.linspace(new_min[2], new_max[2], steps=grid_rows, device=point_cloud.device)

    # Create a 2D meshgrid over u and v.
    U, V = torch.meshgrid(u_vals, v_vals, indexing='ij')  # shape: (grid_rows, grid_cols)
    W, _ = torch.meshgrid(w_vals, v_vals, indexing='ij')

    # Stack into a global grid of shape (grid_rows, grid_cols, 3).
    global_grid = torch.stack([U, V, W], dim=-1)

    # Now, extract the patches. Each patch is a 4×4 block with overlap:
    patches = []
    for i in range(num_u):
        for j in range(num_v):
            row_start = i * 3
            col_start = j * 3
            patch_cp = global_grid[row_start:row_start + 4, col_start:col_start + 4, :]  # (4,4,3)
            patches.append(patch_cp)
    control_points = torch.stack(patches, dim=0)  # (num_u*num_v, 4, 4, 3)
    return control_points


def initialize_nurbs_side_from_colmap(point_cloud: torch.Tensor,
                                      num_u: int = 16,
                                      num_v: int = 16,
                                      side: str = "bottom",
                                      shrink_factor: float = 1.0) -> torch.Tensor:
    """
    Initialize a NURBS surface from a COLMAP scene, but only for one side of the scene's bounding box.
    The default is the "bottom" side. For example, if z is the vertical axis, then the bottom side is
    defined by z = min_z (with some margin). The function constructs a 2D grid over the horizontal (x, y)
    dimensions and fixes the z coordinate to min_z. The grid is then subdivided into patches of shape (4,4,3)
    with adjacent patches sharing control points.

    Args:
        point_cloud (torch.Tensor): Tensor of shape (P, 3) containing 3D points.
        num_u (int): Number of patches along the u (first) direction.
        num_v (int): Number of patches along the v (second) direction.
        side (str): Which side to initialize. Default is "bottom". (Other sides could be added.)
        shrink_factor (float): Factor in (0,1] to shrink the bounding box toward its center. 1.0 means no shrinkage.

    Returns:
        control_points (torch.Tensor): Tensor of shape (num_u*num_v, 4, 4, 3) containing the initial control points.
    """
    # Compute the bounding box.
    min_xyz, _ = torch.min(point_cloud, dim=0)
    max_xyz, _ = torch.max(point_cloud, dim=0)

    # Add a margin (here 5% of the extent) so the grid covers a bit beyond the raw points.
    margin = 0.05 * (max_xyz - min_xyz)
    min_xyz = min_xyz - margin
    max_xyz = max_xyz + margin

    # Compute the center of the bounding box.
    center = (min_xyz + max_xyz) / 2.0

    # Apply shrink factor: each half-extent is scaled.
    half_extent = (max_xyz - min_xyz) / 2.0 * shrink_factor
    new_min = center - half_extent
    new_max = center + half_extent

    # Determine grid dimensions.
    grid_rows = num_u * 3 + 1
    grid_cols = num_v * 3 + 1

    if side.lower() == "bottom":
        # For the bottom side, we assume the vertical axis is z.
        # Use the full range for x and y, but fix z to new_min[2] (i.e. bottom).
        x_vals = torch.linspace(new_min[0], new_max[0], steps=grid_rows, device=point_cloud.device)
        y_vals = torch.linspace(new_min[1], new_max[1], steps=grid_cols, device=point_cloud.device)
        z_val = new_min[2]
        # Create a 2D meshgrid for x and y.
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        # All z values are set to z_val.
        Z = torch.full_like(X, z_val)
        global_grid = torch.stack([X, Y, Z], dim=-1)  # (grid_rows, grid_cols, 3)
    else:
        # If another side is desired, fall back to full bounding-box initialization.
        x_vals = torch.linspace(new_min[0], new_max[0], steps=grid_rows, device=point_cloud.device)
        y_vals = torch.linspace(new_min[1], new_max[1], steps=grid_cols, device=point_cloud.device)
        z_vals = torch.linspace(new_min[2], new_max[2], steps=grid_rows, device=point_cloud.device)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        W, _ = torch.meshgrid(z_vals, y_vals, indexing='ij')
        global_grid = torch.stack([X, Y, W], dim=-1)

    # Extract patches from the global grid.
    patches = []
    for i in range(num_u):
        for j in range(num_v):
            row_start = i * 3
            col_start = j * 3
            patch_cp = global_grid[row_start:row_start + 4, col_start:col_start + 4, :]  # shape: (4, 4, 3)
            patches.append(patch_cp)
    control_points = torch.stack(patches, dim=0)  # shape: (num_u*num_v, 4, 4, 3)
    return control_points


def initialize_nurbs_from_colmap(point_cloud: torch.Tensor, num_u: int, num_v: int,
                                 shrink_factor: float = 1.0) -> torch.Tensor:
    """
    Initialize a NURBS surface from a COLMAP scene.

    Given a 3D point cloud (P, 3) and desired numbers of patches along u and v (num_u, num_v),
    this function computes a global grid of control points covering the point cloud’s bounding box.
    Optionally, the bounding box can be shrunk toward its center by the shrink_factor.

    The global grid has shape ((num_u*3 + 1), (num_v*3 + 1), 3) and is then subdivided into patches
    of shape (4, 4, 3) with adjacent patches sharing control points.

    Args:
        point_cloud (torch.Tensor): Tensor of shape (P, 3) with 3D points.
        num_u (int): Number of patches along the u (first) direction.
        num_v (int): Number of patches along the v (second) direction.
        shrink_factor (float): Factor in (0,1] to shrink the bounding box. 1.0 means no shrinkage.

    Returns:
        control_points (torch.Tensor): Tensor of shape (num_u*num_v, 4, 4, 3) containing
            the initial control points for the NURBS surface.
    """
    # Compute the bounding box of the point cloud.
    min_xyz, _ = torch.min(point_cloud, dim=0)
    max_xyz, _ = torch.max(point_cloud, dim=0)

    # Optionally, add a margin if desired (here we add a 5% margin)
    margin = 0.05 * (max_xyz - min_xyz)
    min_xyz = min_xyz - margin
    max_xyz = max_xyz + margin

    # Compute the center of the bounding box.
    center = (min_xyz + max_xyz) / 2.0

    # Apply the shrink factor: new extents will be shrink_factor times the original extents.
    half_extent = (max_xyz - min_xyz) / 2.0
    half_extent = half_extent * shrink_factor  # shrink the half-extents
    new_min = center - half_extent
    new_max = center + half_extent

    # Determine global grid dimensions:
    # Each patch is 4x4 and adjacent patches share 3 control points along boundaries.
    grid_rows = num_u * 3 + 1
    grid_cols = num_v * 3 + 1

    # Create a uniform grid in the new (shrunken) bounding box.
    u_vals = torch.linspace(new_min[0], new_max[0], steps=grid_rows, device=point_cloud.device)
    v_vals = torch.linspace(new_min[1], new_max[1], steps=grid_cols, device=point_cloud.device)
    # For the z coordinate, you might linearly interpolate similarly.
    w_vals = torch.linspace(new_min[2], new_max[2], steps=grid_rows, device=point_cloud.device)

    # Create a 2D meshgrid over u and v.
    U, V = torch.meshgrid(u_vals, v_vals, indexing='ij')  # shape: (grid_rows, grid_cols)
    W, _ = torch.meshgrid(w_vals, v_vals, indexing='ij')

    # Stack into a global grid of shape (grid_rows, grid_cols, 3).
    global_grid = torch.stack([U, V, W], dim=-1)

    # Now, extract the patches. Each patch is a 4×4 block with overlap:
    patches = []
    for i in range(num_u):
        for j in range(num_v):
            row_start = i * 3
            col_start = j * 3
            patch_cp = global_grid[row_start:row_start + 4, col_start:col_start + 4, :]  # (4,4,3)
            patches.append(patch_cp)
    control_points = torch.stack(patches, dim=0)  # (num_u*num_v, 4, 4, 3)
    return control_points


def SH_interpolation(sh_features, target_shape=(7, 7), mode='bilinear'):
    """
    Interpolate SH features according to the interpolation of 3D points (xyz).

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing 3D points.
        sh_features (torch.Tensor): Tensor of shape (N, 4, 4, 25, 3) representing SH features.
        target_shape (tuple): The target shape (new_height, new_width) for interpolation.

    Returns:
        torch.Tensor: Interpolated SH features tensor of shape (N, new_height, new_width, 25, 3).
        torch.Tensor: Interpolated 3D points tensor of shape (N, new_height, new_width, 3).

                        # Now, combine these to form the uv grids for the 4 subpatches.
        # Each subpatch gets a pair: (u_left, v_left) for top-left, (u_right, v_left) for top-right,
        # (u_left, v_right) for bottom-left, and (u_right, v_right) for bottom-right.
        subdivided_us = torch.cat([u_left, u_right, u_left, u_right], dim=0)  # shape: (4*N_split, res)
        subdivided_vs = torch.cat([v_left, v_left, v_right, v_right], dim=0)  # shape: (4*N_split, res)

    num_patch, udim, vdim, D = xyz.shape
    interpolated_xyz = F.interpolate(xyz.permute(0, 3, 1, 2), size=target_shape, mode=mode, align_corners=True).permute(0, 2, 3, 1)

    return torch.cat([interpolated_xyz[:, :ud//2, :vd//2, :],  #  LL
                    interpolated_xyz[:, ud//2:, :vdim//2, :],
                    interpolated_xyz[:, :ud//2, vd//2:, :],
                    interpolated_xyz[:, ud//2:, ud//2:, :],
                      ], dim=0)
    """

    # N, h, w, D = xyz.shape # Shape: (N, 4, 4, D, 3)
    num_patch, udim, vdim, num_coeffs, num_channels = sh_features.shape
    if num_coeffs == 0:
        return sh_features.repeat_interleave(4, 0)


    sh_features = sh_features.flatten(start_dim=-2)
    sh_features = sh_features.permute(0, 3, 1, 2)
    interpolated_sh = F.interpolate(sh_features, size=target_shape, mode=mode, align_corners=True)
    interpolated_sh = interpolated_sh.permute(0, 2, 3, 1)
    interpolated_sh = interpolated_sh.view(num_patch, target_shape[0], target_shape[1], num_coeffs,
                                           num_channels)  # Shape: (N, new_height, new_width, 25, 3)
    ud, vd = target_shape

    return torch.cat([interpolated_sh[:, :ud//2 + 1, :vd//2 + 1, :, :],
               interpolated_sh[:, ud//2:, :vd//2 + 1, :, :],
               interpolated_sh[:, :ud//2 + 1, vd//2:, :, :],
               interpolated_sh[:, ud//2:, vd//2:, :, :],
               ], dim=0)


def refine_curve2(cp_row: torch.Tensor) -> torch.Tensor:
    """
    Refine a 1D control point sequence for a uniform cubic B-spline curve.

    Args:
        cp_row (torch.Tensor): Tensor of shape (4, D) representing 4 control points.

    Returns:
        torch.Tensor: Refined control points of shape (7, D) computed by inserting midpoints.

    The refined control points are computed as:
        R0 = P0
        R1 = (P0 + P1) / 2
        R2 = (P0 + 4*P1 + P2) / 6
        R3 = (P1 + P2) / 2
        R4 = (P1 + 4*P2 + P3) / 6
        R5 = (P2 + P3) / 2
        R6 = P3
    """
    P0, P1, P2, P3 = cp_row[0], cp_row[1], cp_row[2], cp_row[3]
    R0 = P0
    R1 = (P0 + P1) / 2
    R2 = (P0 + 4 * P1 + P2) / 6
    R3 = (P1 + P2) / 2
    R4 = (P1 + 4 * P2 + P3) / 6
    R5 = (P2 + P3) / 2
    R6 = P3
    return torch.stack([R0, R1, R2, R3, R4, R5, R6], dim=0)



def CP_interpolation(xyz, target_shape=(7, 7), mode='bilinear'):
    """
    Interpolate SH features according to the interpolation of 3D points (xyz).

    Parameters:
        xyz (torch.Tensor): Tensor of shape (N, 4, 4, 3) representing 3D points.
        sh_features (torch.Tensor): Tensor of shape (N, 4, 4, 25, 3) representing SH features.
        target_shape (tuple): The target shape (new_height, new_width) for interpolation.

    Returns:
        torch.Tensor: Interpolated SH features tensor of shape (N, new_height, new_width, 25, 3).
        torch.Tensor: Interpolated 3D points tensor of shape (N, new_height, new_width, 3).

                # Now, combine these to form the uv grids for the 4 subpatches.
        # Each subpatch gets a pair: (u_left, v_left) for top-left, (u_right, v_left) for top-right,
        # (u_left, v_right) for bottom-left, and (u_right, v_right) for bottom-right.
        subdivided_us = torch.cat([u_left, u_right, u_left, u_right], dim=0)  # shape: (4*N_split, res)
        subdivided_vs = torch.cat([v_left, v_left, v_right, v_right], dim=0)  # shape: (4*N_split, res)

    """
    ud, vd = target_shape
    interpolated_xyz = F.interpolate(xyz.permute(0, 3, 1, 2), size=target_shape, mode=mode, align_corners=True).permute(0, 2, 3, 1)

    return torch.cat([interpolated_xyz[:, :ud//2 + 1, :vd//2 + 1, :],  #  LL
                    interpolated_xyz[:, ud//2:, :vd//2 + 1, :],
                    interpolated_xyz[:, :ud//2 + 1, vd//2:, :],
                    interpolated_xyz[:, ud//2:, ud//2:, :],
                      ], dim=0)


    # ============================================================
    # 1. Adaptive Sampling and Refinement for a Single Patch
    # ============================================================

def adaptive_sample_cell(cp, u0, u1, v0, v1, threshold, depth, max_depth):
    """
    Recursively subdivide a patch cell in the (u,v) domain and return sample points
    that cover the patch adaptively. For each cell that meets the error criteria, return
    the cell's center, along with its surface point and normal.

    Args:
        cp (torch.Tensor): Control points for one patch, shape (R, R, 3).
        u0, u1, v0, v1 (float): Parameter bounds.
        threshold (float): Error threshold.
        depth (int): Current recursion depth.
        max_depth (int): Maximum allowed depth.

    Returns:
        samples (list of tuples): Each tuple is (u, v, position, normal) for a cell.
    """

    # Define functions to evaluate surface and normal at (u,v) in a vectorized manner.
    def evaluate_patch_point(cp, u, v):
        # Assume cubic Bezier basis functions; here we use the standard formulas.
        U = torch.tensor([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3],
                         dtype=cp.dtype, device=cp.device)
        V = torch.tensor([(1 - v) ** 3, 3 * v * (1 - v) ** 2, 3 * v ** 2 * (1 - v), v ** 3],
                         dtype=cp.dtype, device=cp.device)
        return torch.einsum('i,ijc,j->c', U, cp, V)

    def evaluate_patch_normal(cp, u, v):
        device = cp.device
        dtype = cp.dtype
        U = torch.tensor([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3],
                         dtype=dtype, device=device)
        V = torch.tensor([(1 - v) ** 3, 3 * v * (1 - v) ** 2, 3 * v ** 2 * (1 - v), v ** 3],
                         dtype=dtype, device=device)
        dU = torch.tensor([-3 * (1 - u) ** 2,
                           3 * (1 - u) ** 2 - 6 * u * (1 - u),
                           6 * u * (1 - u) - 3 * u ** 2,
                           3 * u ** 2],
                          dtype=dtype, device=device)
        dV = torch.tensor([-3 * (1 - v) ** 2,
                           3 * (1 - v) ** 2 - 6 * v * (1 - v),
                           6 * v * (1 - v) - 3 * v ** 2,
                           3 * v ** 2],
                          dtype=dtype, device=device)
        S_u = torch.einsum('i,ijc,j->c', dU, cp, V)
        S_v = torch.einsum('i,ijc,j->c', U, cp, dV)
        n = torch.cross(S_u, S_v)
        norm = torch.norm(n) + 1e-8
        return n / norm

    # Evaluate the four corners.
    p00 = evaluate_patch_point(cp, u0, v0)
    p10 = evaluate_patch_point(cp, u1, v0)
    p01 = evaluate_patch_point(cp, u0, v1)
    p11 = evaluate_patch_point(cp, u1, v1)
    # Evaluate center.
    um = (u0 + u1) / 2.0
    vm = (v0 + v1) / 2.0
    pmid = evaluate_patch_point(cp, um, vm)
    # Bilinear interpolation at center.
    p_bilinear = (p00 + p10 + p01 + p11) / 4.0
    error = torch.norm(pmid - p_bilinear).item()

    if error < threshold or depth >= max_depth:
        # Return the center sample.
        sample = (um, vm, pmid, evaluate_patch_normal(cp, um, vm))
        return [sample]
    else:
        samples = []
        cells = [
            (u0, um, v0, vm),
            (um, u1, v0, vm),
            (u0, um, vm, v1),
            (um, u1, vm, v1)
        ]
        for cell in cells:
            samples.extend(
                adaptive_sample_cell(cp, cell[0], cell[1], cell[2], cell[3], threshold, depth + 1, max_depth))
        return samples

def adaptive_sample_patch(cp, threshold=0.01, max_depth=5):
    """
    Given a patch's control points cp (shape (R,R,3)), return a tensor of sampled (u,v)
    coordinates (and optionally positions and normals) computed adaptively.

    Returns:
        sample_uv: Tensor of shape (M, 2) of (u,v) coordinates.
        sample_xyz: Tensor of shape (M, 3) of corresponding 3D positions.
        sample_normals: Tensor of shape (M, 3) of corresponding normals.
    """
    samples = adaptive_sample_cell(cp, 0.0, 1.0, 0.0, 1.0, threshold, 0, max_depth)
    # Unpack samples.
    uv = torch.tensor([[s[0], s[1]] for s in samples], dtype=cp.dtype, device=cp.device)
    xyz = torch.stack([s[2] for s in samples], dim=0)
    normals = torch.stack([s[3] for s in samples], dim=0)
    return uv, xyz, normals


def refine_patch_features(sh_patch: torch.Tensor) -> torch.Tensor:
    """
    Refine a 2D patch of spherical harmonics features.

    Args:
        sh_patch (torch.Tensor): Tensor of shape (4, 4, C, 3), where C is the number of SH coefficients.

    Returns:
        torch.Tensor: Refined patch of shape (7, 7, C, 3).
    """
    R = sh_patch.shape[0]  # current resolution (e.g., 4 or 8)
    refined_res = 2 * R - 1  # refined resolution (e.g., 7 or 15)

    # First, refine along each row.
    refined_rows = []
    for i in range(R):
        # Get the i-th row, shape: (R, C, 3)
        row = sh_patch[i]
        # Flatten the last two dimensions to a single vector (shape: (R, C*3))
        row_flat = row.reshape(R, -1)
        # Use your existing refine_curve function (assumed to map (R, D) to (2*R - 1, D))
        refined_row_flat = refine_curve(row_flat)  # shape: (refined_res, C*3)
        # Reshape back to (refined_res, C, 3)
        refined_row = refined_row_flat.reshape(refined_res, sh_patch.shape[2], sh_patch.shape[3])
        refined_rows.append(refined_row)
    # Stack rows; shape: (R, refined_res, C, 3)
    refined_rows = torch.stack(refined_rows, dim=0)

    # Next, refine along the columns.
    refined_cols = []
    for j in range(refined_res):
        # For each column j, extract that column from each row (shape: (R, C, 3))
        col = refined_rows[:, j, :, :]
        col_flat = col.reshape(R, -1)  # shape: (R, C*3)
        refined_col_flat = refine_curve(col_flat)  # shape: (refined_res, C*3)
        refined_col = refined_col_flat.reshape(refined_res, sh_patch.shape[2], sh_patch.shape[3])
        refined_cols.append(refined_col)
    # Stack columns; shape: (refined_res, refined_res, C, 3)
    refined_patch = torch.stack(refined_cols, dim=1)
    return refined_patch


def subdivide_patch_features(sh_patch: torch.Tensor) -> torch.Tensor:
    """
    Subdivide a patch's spherical harmonics features into four sub-patches.

    Args:
        sh_patch (torch.Tensor): Tensor of shape (R, R, C, 3), where R is the current resolution.

    Returns:
        torch.Tensor: Tensor of shape (4, R, R, C, 3), i.e. 4 sub-patches each of shape (R, R, C, 3).
    """
    R = sh_patch.shape[0]
    refined_res = 2 * R - 1
    # Refine the patch: shape becomes (refined_res, refined_res, C, 3)
    refined = refine_patch_features(sh_patch)
    # For subdividing, we want each new sub-patch to have resolution R.
    # They are extracted as:
    # - top_left: rows 0 to R, columns 0 to R
    # - top_right: rows 0 to R, columns (R-1) to refined_res
    # - bottom_left: rows (R-1) to refined_res, columns 0 to R
    # - bottom_right: rows (R-1) to refined_res, columns (R-1) to refined_res
    top_left = refined[0:R, 0:R, :, :]
    top_right = refined[0:R, R - 1:refined_res, :, :]
    bottom_left = refined[R - 1:refined_res, 0:R, :, :]
    bottom_right = refined[R - 1:refined_res, R - 1:refined_res, :, :]
    sub_patches = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=0)
    return sub_patches


def subdivide_all_patches_features_naive(per_patch_sh: torch.Tensor) -> torch.Tensor:
    """
    Subdivide each patch's spherical harmonics features in a batch.

    Args:
        per_patch_sh (torch.Tensor): Tensor of shape (N, R, R, C, 3).

    Returns:
        torch.Tensor: Tensor of shape (4*N, R, R, C, 3) where each original patch is replaced by 4 sub-patches.
    """
    subdivided_list = []
    N = per_patch_sh.shape[0]
    for i in range(N):
        sub_patches = subdivide_patch_features(per_patch_sh[i])  # shape: (4, R, R, C, 3)
        for j in range(4):
            subdivided_list.append(sub_patches[j])
    return torch.stack(subdivided_list, dim=0)


def refine_curve(cp_row: torch.Tensor) -> torch.Tensor:
    """
    Refine a 1D control point sequence for a uniform cubic B-spline curve.

    This function generalizes the refinement from an input sequence of R control points
    (of shape (R, D)) to an output sequence of (2R - 1) control points.

    For an input sequence [P0, P1, ..., P_{R-1}], the refined sequence is:
        R0 = P0
        For i = 1 to R-2:
            R_{2i-1} = (P_{i-1} + P_i) / 2
            R_{2i}   = (P_{i-1} + 4*P_i + P_{i+1}) / 6
        R_{2R-3} = (P_{R-2} + P_{R-1}) / 2
        R_{2R-2} = P_{R-1}

    Args:
        cp_row (torch.Tensor): Tensor of shape (R, D) representing R control points.

    Returns:
        torch.Tensor: Refined control points of shape (2R - 1, D).
    """
    R, D = cp_row.shape
    if R == 1:
        return cp_row
    refined = [cp_row[0]]
    # For control points 1 to R-2, add two refined points per original point.
    for i in range(1, R - 1):
        # Insert the midpoint of P_{i-1} and P_i.
        midpoint = (cp_row[i - 1] + cp_row[i]) / 2
        refined.append(midpoint)
        # Insert a weighted average using the previous, current, and next points.
        weighted = (cp_row[i - 1] + 4 * cp_row[i] + cp_row[i + 1]) / 6
        refined.append(weighted)
    # For the last control point, add the midpoint with the previous and then the last point.
    refined.append((cp_row[R - 2] + cp_row[R - 1]) / 2)
    refined.append(cp_row[R - 1])
    refined = torch.stack(refined, dim=0)
    return refined


def refine_patch(cp_patch: torch.Tensor) -> torch.Tensor:
    """
    Refine a 2D patch control net (4x4) into a 7x7 net using tensor-product refinement.

    Args:
        cp_patch (torch.Tensor): Tensor of shape (4, 4, D) representing a patch.

    Returns:
        torch.Tensor: Refined control net of shape (7, 7, D).
    """
    # Refine along each row.
    refined_rows = []
    for i in range(4):
        row = cp_patch[i]  # shape: (4, D)
        refined_row = refine_curve(row)  # shape: (7, D)
        refined_rows.append(refined_row)
    refined_rows = torch.stack(refined_rows, dim=0)  # shape: (4, 7, D)

    # Now refine along the columns.
    refined_cols = []
    for j in range(7):
        col = refined_rows[:, j, :]  # shape: (4, D)
        refined_col = refine_curve(col)  # shape: (7, D)
        refined_cols.append(refined_col)
    refined_patch = torch.stack(refined_cols, dim=1)  # shape: (7, 7, D)
    return refined_patch


def subdivide_patch(cp_patch: torch.Tensor) -> torch.Tensor:
    """
    Subdivide a single patch's control net into four sub-patches using NURBS subdivision.

    Args:
        cp_patch (torch.Tensor): Tensor of shape (4, 4, D) for a single patch.

    Returns:
        torch.Tensor: Tensor of shape (4, 4, 4, D), where there are 4 sub-patches,
                      each of shape (4, 4, D), corresponding to top-left, top-right,
                      bottom-left, and bottom-right sub-patches.
    """
    # Refine the patch to obtain a 7x7 control net.
    refined = refine_patch(cp_patch)  # shape: (7, 7, D)
    # Extract the four sub-patches.
    top_left = refined[:4, :4, :]
    top_right = refined[:4, 3:, :]
    bottom_left = refined[3:, :4, :]
    bottom_right = refined[3:, 3:, :]
    # Stack them along a new dimension.
    sub_patches = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=0)
    return sub_patches


def subdivide_all_patches(per_patch_cp: torch.Tensor) -> torch.Tensor:
    """
    Subdivide each patch in a batch of patches.

    Args:
        per_patch_cp (torch.Tensor): Tensor of shape (N, 4, 4, D) representing N patches.

    Returns:
        torch.Tensor: A tensor of shape (4*N, 4, 4, D) where each original patch is replaced
                      by 4 subdivided sub-patches.
    """
    subdivided_list = []
    N = per_patch_cp.shape[0]
    for i in range(N):
        sub_patches = subdivide_patch(per_patch_cp[i])  # shape: (4, 4, 4, D)
        for j in range(4):
            subdivided_list.append(sub_patches[j])
    return torch.stack(subdivided_list, dim=0)



scripted_subdivide_features = torch.jit.script(subdivide_all_patches_features_naive)
scripted_subdivide = torch.jit.script(subdivide_all_patches)
import torch
# import torch
import torch


def vectorized_refine_curve(cp_row: torch.Tensor) -> torch.Tensor:
    """
    Vectorized refinement of a 1D control point sequence.
    Expects cp_row of shape (..., R, D) and returns refined points of shape (..., 2R - 1, D).
    """
    R = cp_row.shape[-2]
    if R == 1:
        return cp_row
    # Compute midpoints between consecutive points
    mid = (cp_row[..., :-1, :] + cp_row[..., 1:, :]) / 2.0  # shape: (..., R-1, D)
    if R > 2:
        # Compute the weighted averages for interior points
        weighted = (cp_row[..., :-2, :] + 4 * cp_row[..., 1:-1, :] + cp_row[..., 2:, :]) / 6.0  # shape: (..., R-2, D)
        # Create an output tensor to hold 2R-1 points
        out_shape = cp_row.shape[:-2] + (2 * R - 1, cp_row.shape[-1])
        refined = cp_row.new_empty(out_shape)
        # Set the endpoints
        refined[..., 0, :] = cp_row[..., 0, :]
        refined[..., -1, :] = cp_row[..., -1, :]
        # Fill in the intermediate values:
        # Even indices (starting at index 2) get the weighted averages.
        refined[..., 2:-1:2, :] = weighted
        # Odd indices get the midpoints.
        refined[..., 1:-1:2, :] = mid
    else:
        # For R==2, simply concatenate [P0, midpoint, P1]
        refined = torch.cat([cp_row[..., :1, :], mid, cp_row[..., -1:, :]], dim=-2)
    return refined


def vectorized_refine_patch(cp_patch: torch.Tensor) -> torch.Tensor:
    """
    Vectorized refinement of a 2D patch control net.
    Expects cp_patch of shape (..., 4, 4, D) and returns a refined patch of shape (..., 7, 7, D).
    """
    # Refine along each row (i.e. across the second-to-last dimension)
    # cp_patch has shape (..., 4, 4, D); the refinement is applied along the dimension with size 4.
    refined_rows = vectorized_refine_curve(cp_patch)  # Now shape: (..., 4, 7, D)

    # To refine along the columns, swap the row and column axes so that the column axis becomes
    # the one being refined, then swap back after refinement.
    refined_rows = refined_rows.transpose(-3, -2)  # Shape becomes: (..., 7, 4, D)
    refined_patch = vectorized_refine_curve(refined_rows)  # Now: (..., 7, 7, D)
    refined_patch = refined_patch.transpose(-3, -2)  # Swap back to: (..., 7, 7, D)

    return refined_patch


def vectorized_subdivide_all_patches(per_patch_cp: torch.Tensor) -> torch.Tensor:
    """
    Vectorized subdivision over a batch of patches.
    Expects per_patch_cp of shape (N, 4, 4, D) and returns sub-patches.

    The refined patch is first computed (shape: (N, 7, 7, D)), then the four sub-patches
    are extracted. The resulting tensor has shape (N, 4, 4, 4, D) if you want to keep the batch
    dimension per original patch, or you can reshape it to (4*N, 4, 4, D).
    """
    # Compute the refined patch for all patches simultaneously.
    refined = vectorized_refine_patch(per_patch_cp)  # Shape: (N, 7, 7, D)

    # Extract the four subpatches by slicing the refined control net.
    top_left = refined[:, :4, :4, :]
    top_right = refined[:, :4, 3:, :]
    bottom_left = refined[:, 3:, :4, :]
    bottom_right = refined[:, 3:, 3:, :]

    # Stack them along a new dimension. The output shape is (N, 4, 4, 4, D).
    sub_patches = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
    return sub_patches


def subdivide_uv_1d_vectorized(uv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Vectorized version of subdivide_uv_1d.

    For each row (patch) of uv (shape: (N, L)), compute:
      u_min = min(u), u_max = max(u), mid = (u_min + u_max)/2.
    Then compute raw_left as a linear spacing from u_min to mid and raw_right from mid to u_max,
    and finally normalize each row so that they map to [0, 1].

    Args:
      uv (torch.Tensor): Tensor of shape (N, L)

    Returns:
      u_left_norm, u_right_norm: Each of shape (N, L), representing the normalized left and right halves.
    """
    N, L = uv.shape
    device = uv.device
    dtype = uv.dtype

    # Compute per-row minimum, maximum and midpoint.
    u_min = uv.min(dim=1, keepdim=True)[0]  # shape (N, 1)
    u_max = uv.max(dim=1, keepdim=True)[0]  # shape (N, 1)
    mid = (u_min + u_max) / 2  # shape (N, 1)

    # Create a base linspace vector from 0 to 1 of shape (L,)
    t = torch.linspace(0, 1, steps=L, device=device, dtype=dtype).unsqueeze(0).expand(N, L)

    # For the left half, compute raw values: raw_left = u_min + t * (mid - u_min)
    # For the right half, compute raw values: raw_right = mid + t * (u_max - mid)
    raw_left = u_min + t * (mid - u_min)
    raw_right = mid + t * (u_max - mid)

    # Re-normalize each row. For the left, subtract u_min and divide by (mid - u_min),
    # for the right, subtract mid and divide by (u_max - mid). Add a small epsilon to avoid division by zero.
    epsilon = 1e-8
    u_left_norm = (raw_left - u_min) / (mid - u_min + epsilon)
    u_right_norm = (raw_right - mid) / (u_max - mid + epsilon)

    return u_left_norm, u_right_norm

def subdivide_uv_1d(uv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Given a uv tensor of shape (N, L) representing the current uv values for each patch,
    compute new uv coordinates for the left and right halves based on each patch's range.

    For each patch (row), let:
        u_min = uv.min(), u_max = uv.max(), and mid = (u_min+u_max)/2.
    Then compute:
        raw_left  = linspace(u_min, mid, L)
        raw_right = linspace(mid, u_max, L)
    Finally, re-normalize each so that they run from 0 to 1:
        u_left_norm  = (raw_left  - u_min) / (mid - u_min)
        u_right_norm = (raw_right - mid) / (u_max - mid)

    Args:
        uv (torch.Tensor): Tensor of shape (N, L).

    Returns:
        uv_left_norm, uv_right_norm: Each of shape (N, L), now spanning [0,1].
    """
    N, L = uv.shape
    # Compute per-patch minimum and maximum.
    u_min = uv.min(dim=1, keepdim=True)[0]  # (N,1)
    u_max = uv.max(dim=1, keepdim=True)[0]  # (N,1)
    mid = (u_min + u_max) / 2  # (N,1)

    # For each patch, create new linspace values over the original subintervals.
    u_left_list = []
    u_right_list = []
    for i in range(N):
        # Use .item() to extract scalar values for this patch.
        left_line = torch.linspace(u_min[i, 0].item(), mid[i, 0].item(), steps=L,
                                   device=uv.device, dtype=uv.dtype)
        right_line = torch.linspace(mid[i, 0].item(), u_max[i, 0].item(), steps=L,
                                    device=uv.device, dtype=uv.dtype)
        u_left_list.append(left_line)
        u_right_list.append(right_line)
    raw_u_left = torch.stack(u_left_list, dim=0)  # (N, L)
    raw_u_right = torch.stack(u_right_list, dim=0)  # (N, L)

    # Now re-normalize each row so that the left half spans [0,1] and similarly for the right.
    u_left_norm = (raw_u_left - u_min) / (mid - u_min)
    u_right_norm = (raw_u_right - mid) / (u_max - mid)

    return u_left_norm, u_right_norm


def subdivide_uv_params(us: torch.Tensor, vs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Subdivide the uv parameters for each patch into four sets for subpatches.

    For each patch, assume its current u values span some interval [u_min, u_max] (which need not be [0,1]).
    The left half will be reparameterized to [0,1] by computing:
        u_left = subdivide_uv_1d(us) -> normalized left half,
        u_right = subdivide_uv_1d(us) -> normalized right half.
    Similarly for v.

    Then, the four subpatches get:
      - Top-left: (u_left, v_left)
      - Top-right: (u_right, v_left)
      - Bottom-left: (u_left, v_right)
      - Bottom-right: (u_right, v_right)

    Args:
        us (torch.Tensor): Tensor of shape (N, L) for u coordinates.
        vs (torch.Tensor): Tensor of shape (N, L) for v coordinates.

    Returns:
        subdivided_us: Tensor of shape (4*N, L).
        subdivided_vs: Tensor of shape (4*N, L).
    """
    u_left, u_right = subdivide_uv_1d_vectorized(us)
    v_left, v_right = subdivide_uv_1d_vectorized(vs)
    # u_left, u_right = subdivide_uv_1d(us)
    # v_left, v_right = subdivide_uv_1d(vs)

    subdivided_us = torch.cat([u_left, u_right, u_left, u_right], dim=0)
    subdivided_vs = torch.cat([v_left, v_left, v_right, v_right], dim=0)

    return subdivided_us, subdivided_vs


import torch
import torch.nn.functional as F


# def stitch_control_features(patches, patch_size=4, stride=3):
#     """
#     Given control features for each patch (tensor of shape (N, patch_size, patch_size, D)),
#     where N = P_rows * P_cols, stitch them into one global grid of shape (H, W, D).
#
#     Adjacent patches are assumed to share boundaries if placed on a grid with stride = patch_size - 1.
#
#     Args:
#       patches (torch.Tensor): Tensor of shape (N, patch_size, patch_size, D) containing the control features.
#       patch_size (int): Size of each patch (default: 4).
#       stride (int): Stride between patches (default: patch_size - 1, i.e. 3).
#
#     Returns:
#       global_grid (torch.Tensor): Global grid of shape (H, W, D).
#       grid_dims (tuple): (H, W) of the global grid.
#     """
#     N, H_patch, W_patch, D = patches.shape
#
#     # Assume that patches are arranged in a square grid (or else compute rows/cols appropriately).
#     grid_rows = int(N ** 0.5)
#     grid_cols = grid_rows  # Assuming square arrangement.
#
#     # The overall global grid dimensions (with overlapping boundaries)
#     H_global = grid_rows * (patch_size - 1) + 1
#     W_global = grid_cols * (patch_size - 1) + 1
#
#     global_grid = torch.zeros(H_global, W_global, D, device=patches.device, dtype=patches.dtype)
#     count = torch.zeros(H_global, W_global, 1, device=patches.device, dtype=patches.dtype)
#
#     # Reshape patches into grid shape: (grid_rows, grid_cols, patch_size, patch_size, D)
#     patches_grid = patches.view(grid_rows, grid_cols, patch_size, patch_size, D)
#
#     # Place each patch into the global grid.
#     for i in range(grid_rows):
#         for j in range(grid_cols):
#             top = i * (patch_size - 1)
#             left = j * (patch_size - 1)
#             global_grid[top:top + patch_size, left:left + patch_size, :] += patches_grid[i, j]
#             count[top:top + patch_size, left:left + patch_size, :] += 1
#     # Average overlapping regions.
#     global_grid = global_grid / count.clamp(min=1e-8)
#     return global_grid, (H_global, W_global)

import math
import torch
import torch.nn.functional as F
import torch

def insert_knot_surface_u(ctrl_pts, knot_u, u_star, p):
    """
    Insert a knot in the U direction (surface).
    ctrl_pts: (m, n, d)  (e.g. control_points, control_weights, ...)
    knot_u: (m+p+1,)
    u_star: float, the knot to insert
    p: degree in U
    Returns: new_ctrl_pts (m+1, n, d), new_knot_u (m+p+2,)
    """
    m, n, d = ctrl_pts.shape
    # Clamp u_star to valid domain
    u_star_clamped = float(torch.clamp(
        torch.tensor(u_star, device=knot_u.device, dtype=knot_u.dtype),
        min=knot_u[p].item() + 1e-6,
        max=knot_u[-p - 1].item() - 1e-6
    ))

    k = torch.searchsorted(knot_u, torch.tensor(u_star_clamped, device=knot_u.device), right=False) - 1
    k = int(k.item())
    # Find span k such that knot_u[k] <= u_star < knot_u[k+1]
    # k = torch.searchsorted(knot_u, torch.tensor(u_star, device=knot_u.device), right=False) - 1
    # k = int(k.item())

    # Insert the knot into the vector
    new_knot_u = torch.cat([knot_u[:k+1], torch.tensor([u_star], dtype=knot_u.dtype, device=knot_u.device), knot_u[k+1:]])
    new_ctrl_pts = torch.zeros((m+1, n, d), dtype=ctrl_pts.dtype, device=ctrl_pts.device)

    # Unaffected control points
    copy_left = min(k-p+1, ctrl_pts.shape[0])
    new_ctrl_pts[:copy_left,:,:] = ctrl_pts[:copy_left,:,:]

    copy_right_src = k+1
    copy_right_dst = k+2
    num_right = ctrl_pts.shape[0] - copy_right_src
    if num_right > 0:
        new_ctrl_pts[copy_right_dst:copy_right_dst+num_right,:,:] = ctrl_pts[copy_right_src:copy_right_src+num_right,:,:]

    # De Boor update (the affected rows)
    for i in range(k-p+1, k+1):
        alpha = (u_star - knot_u[i]) / (knot_u[i+p] - knot_u[i] + 1e-8)
        new_ctrl_pts[i,:,:] = (1-alpha) * ctrl_pts[i-1,:,:] + alpha * ctrl_pts[i,:,:]
    return new_ctrl_pts, new_knot_u, k+1

def insert_knot_surface_v(ctrl_pts, knot_v, v_star, q):
    """
    Insert a knot in the V direction (surface).
    ctrl_pts: (m, n, d)
    knot_v: (n+q+1,)
    v_star: float, the knot to insert
    q: degree in V
    Returns: new_ctrl_pts (m, n+1, d), new_knot_v (n+q+2,)
    """
    m, n, d = ctrl_pts.shape
    l = torch.searchsorted(knot_v, torch.tensor(v_star, device=knot_v.device), right=False) - 1
    l = int(l.item())

    new_knot_v = torch.cat([knot_v[:l+1], torch.tensor([v_star], dtype=knot_v.dtype, device=knot_v.device), knot_v[l+1:]])
    new_ctrl_pts = torch.zeros((m, n+1, d), dtype=ctrl_pts.dtype, device=ctrl_pts.device)

    # Unaffected control points
    new_ctrl_pts[:,:l-q+1,:] = ctrl_pts[:,:l-q+1,:]
    new_ctrl_pts[:,l+2:,:] = ctrl_pts[:,l+1:,:]

    # De Boor update (the affected columns)
    for j in range(l-q+1, l+1):
        alpha = (v_star - knot_v[j]) / (knot_v[j+q] - knot_v[j] + 1e-8)
        new_ctrl_pts[:,j,:] = (1-alpha) * ctrl_pts[:,j-1,:] + alpha * ctrl_pts[:,j,:]
    return new_ctrl_pts, new_knot_v
def stitch_control_features(patches: torch.Tensor,
                            true_size=None,
                            patch_size: int = 4,
                            stride: int = 1,
                           ) -> torch.Tensor:
    """
    Vectorized, differentiable stitching of overlapping patches.

    Args:
        patches: Tensor of shape (N, P, P, D) where P=patch_size.
        patch_size: spatial size of each patch (default=4).
        stride: sliding‐window stride (default=patch_size-1=3).

    Returns:
        global_grid: Tensor of shape (H, W, D), where
                     H = grid_rows*(P-1)+1, W = grid_cols*(P-1)+1.
    """
    N, P, _, D = patches.shape

    # figure out grid layout
    grid_rows = math.ceil(math.sqrt(N))
    grid_cols = math.ceil(N / grid_rows)
    M = grid_rows * grid_cols

    # if needed, pad extra zero‐patches
    if M != N:
        pad = patches.new_zeros((M - N, P, P, D))
        patches = torch.cat([patches, pad], dim=0)

    # compute output HxW
    H = grid_rows * (stride)
    W = grid_cols * (stride)
    # H = grid_rows - stride #* (P) #- 1) + 1
    # W = grid_cols - stride# (P) #- 1) + 1
    if true_size is not None:
        H = true_size[0]
        W = true_size[1]

    # re‐arrange patches into “columns” for fold:
    #   (M, D, P, P) → (1, D*P*P, M)
    cols = patches.permute(0, 3, 1, 2) \
                  .reshape(M, D * P * P) \
                  .transpose(0, 1) \
                  .unsqueeze(0)  # shape (1, D*P*P, M)

    # fold them back into image, summing overlaps
    global_sum = F.fold(
        cols,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride
    )  # shape (1, D, H, W)

    # build a matching “count” map to average overlaps
    # ones = cols.new_ones((1, 1, M))
    # build a matching “count” map to average overlaps
    ones = patches.new_ones((1, patch_size * patch_size, M), requires_grad=True)
    count = F.fold(
        ones,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride
    )  # shape (1, 1, H, W)

    # average
    global_grid = global_sum / count.clamp(min=1e-3)

    # to (H, W, D)
    return global_grid.squeeze(0).permute(1, 2, 0)
# def stitch_control_features(patches, patch_size=4, stride=3):
#     """
#     Given control features for each patch (tensor of shape (N, patch_size, patch_size, D)),
#     where N = P_rows * P_cols, stitch them into one global grid of shape (H, W, D).
#
#     Adjacent patches are assumed to share boundaries if placed on a grid with stride = patch_size - 1.
#
#     Args:
#       patches (torch.Tensor): Tensor of shape (N, patch_size, patch_size, D) containing the control features.
#       patch_size (int): Size of each patch (default: 4).
#       stride (int): Stride between patches (default: patch_size - 1, i.e. 3).
#
#     Returns:
#       global_grid (torch.Tensor): Global grid of shape (H, W, D).
#       grid_dims (tuple): (H, W) of the global grid.
#     """
#     N, H_patch, W_patch, D = patches.shape
#
#     # Compute grid dimensions based on N.
#     grid_rows = int(math.ceil(math.sqrt(N)))
#     grid_cols = int(math.ceil(N / grid_rows))
#
#     # If the total number of patches is less than grid_rows * grid_cols, pad with zeros.
#     total_required = grid_rows * grid_cols
#     if total_required != N:
#         num_missing = total_required - N
#         padding = torch.zeros(num_missing, patch_size, patch_size, D, dtype=patches.dtype, device=patches.device)
#         patches = torch.cat([patches, padding], dim=0)
#
#     # Now, reshape to a grid.
#
#     patches_grid = patches.view(grid_rows, grid_cols, patch_size, patch_size, D)
#
#     # Compute global grid dimensions.
#     H_global = grid_rows * (patch_size - 1) + 1
#     W_global = grid_cols * (patch_size - 1) + 1
#
#     global_grid = torch.zeros(H_global, W_global, D, device=patches.device, dtype=patches.dtype)
#     count = torch.zeros(H_global, W_global, 1, device=patches.device, dtype=patches.dtype)
#
#     # Stitch patches into the global grid.
#     for i in range(grid_rows):
#         for j in range(grid_cols):
#             top = i * (patch_size - 1)
#             left = j * (patch_size - 1)
#             global_grid[top:top + patch_size, left:left + patch_size, :] += patches_grid[i, j]
#             count[top:top + patch_size, left:left + patch_size, :] += 1
#     # Average overlapping regions.
#     global_grid = global_grid / count.clamp(min=1e-8)
#     return global_grid


def subdivide_grid_region(parent_grid: torch.Tensor, subdivide_indices: torch.Tensor,
                          patch_size: int, stride: int) -> (torch.Tensor, torch.LongTensor):
    """
    Given a parent global grid (H, W, D) and a 1D tensor of linear indices (of sliding-window patches)
    that are chosen to be subdivided, this function extracts the corresponding patches from the parent grid,
    refines them via upsampling, and determines the indices in the child grid where these refined patches are to be placed.

    Args:
        parent_grid (torch.Tensor): Parent grid with shape (H, W, D).
        subdivide_indices (torch.Tensor): 1D tensor of indices (in sliding-window order) of patches to subdivide.
        patch_size (int): Size of the extracted patch (e.g. 4).
        stride (int): Stride used for the sliding window (e.g. patch_size - 1).
    Returns:
        refined_patches: Tensor of shape (num_subdivided, patch_size, patch_size, D) containing new features.
        child_indices: Tensor of shape (num_subdivided, 2) giving (row, col) starting indices in the child grid.
    """
    # First, convert the parent grid into patches via unfolding.
    H, W, D = parent_grid.shape
    # Reshape to (1, D, H, W)
    grid = parent_grid.permute(2, 0, 1).unsqueeze(0)
    # Extract patches using unfold.
    patches_unfold = F.unfold(grid, kernel_size=patch_size, stride=stride)  # shape (1, D*patch_size*patch_size, L)
    L = patches_unfold.shape[-1]
    patches = patches_unfold.squeeze(0).transpose(0, 1).reshape(-1, patch_size, patch_size,
                                                                D)  # (L, patch_size, patch_size, D)

    # Convert subdivide_indices (in range L) into extracted patches
    refined_patches = patches[subdivide_indices]  # (N_sub, patch_size, patch_size, D)

    # Optionally, refine the patches (e.g. bilinear upsampling or a learned refinement)
    # Here we simply upsample to the same patch size for demonstration.
    # For example, if you want a more refined version, you could target (2*patch_size - 1) then extract a 4x4 window.
    refined_patches = F.interpolate(
        refined_patches.permute(0, 3, 1, 2), size=(patch_size, patch_size), mode='bilinear', align_corners=True
    ).permute(0, 2, 3, 1)

    # Determine the top-left indices in the parent grid for each extracted patch.
    # Assume that the patches were extracted in row-major order.
    # The number of patches along the width is: num_cols = floor((W - patch_size) / stride) + 1.
    num_cols = ((W - patch_size) // stride) + 1
    # Compute row, col indices for each patch
    all_indices = torch.arange(L, device=parent_grid.device)
    rows = all_indices // num_cols
    cols = all_indices % num_cols
    # Now, select rows and cols for the subdivided patches:
    child_indices = torch.stack([rows[subdivide_indices] * stride, cols[subdivide_indices] * stride], dim=1)
    return refined_patches, child_indices


def mark_parent_inactive(parent_mask: torch.Tensor, subdivide_indices: torch.Tensor, patch_size: int,
                         stride: int) -> torch.Tensor:
    """
    Given the parent mask (Boolean tensor with shape (H, W)) and the linear indices of sliding-window patches
    chosen for subdivision, mark the corresponding regions in the parent mask as inactive.

    Args:
        parent_mask (torch.Tensor): Boolean tensor of shape (H, W).
        subdivide_indices (torch.Tensor): 1D tensor of indices (in sliding-window order) that are subdivided.
        patch_size (int): Size of the patch window.
        stride (int): Stride used in the sliding window.
    Returns:
        Updated parent_mask (torch.Tensor) with the corresponding areas set to False.
    """
    H, W = parent_mask.shape
    num_cols = ((W - patch_size) // stride) + 1
    all_indices = torch.arange(parent_mask.numel(), device=parent_mask.device)
    # Create a folded view to simulate sliding window extraction.
    mask_grid = parent_mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    patches_unfold = F.unfold(mask_grid.float(), kernel_size=patch_size, stride=stride)  # (1, patch_size*patch_size, L)
    L = patches_unfold.shape[-1]
    # Create a mapping from sliding-window index to the top-left pixel coordinate:
    rows = (torch.arange(L, device=parent_mask.device) // num_cols) * stride
    cols = (torch.arange(L, device=parent_mask.device) % num_cols) * stride
    # For each patch index in subdivide_indices, mark the corresponding region as inactive.
    new_parent_mask = parent_mask.clone()
    for idx in subdivide_indices.tolist():
        r = int(rows[idx].item())
        c = int(cols[idx].item())
        new_parent_mask[r:r + patch_size, c:c + patch_size] = False
    return new_parent_mask

def find_spans_torch(num_ctrl_pts, degree, u_params, knot_vector):
    """
    Finds the knot spans for a batch of u_params.
    Args:
        num_ctrl_pts (int): Number of control points (n+1, where n is max CP index).
        degree (int): Degree of the B-spline.
        u_params (torch.Tensor): 1D tensor of parameter values to find spans for.
        knot_vector (torch.Tensor): 1D global knot vector.
    Returns:
        torch.Tensor: 1D tensor of span indices.
    """
    n_max_cp_idx = num_ctrl_pts - 1
    # For clamped knots, valid u is typically [knots[degree], knots[num_ctrl_pts]]
    # Ensure u_params are within the valid domain to prevent issues with searchsorted at exact boundaries.
    # A small epsilon is used for the upper bound if u_params can reach knot_vector[num_ctrl_pts].
    u_clamped = torch.clamp(u_params, knot_vector[degree], knot_vector[num_ctrl_pts] - 1e-6)
    if knot_vector[num_ctrl_pts] == u_params.max():  # Handle exact upper bound
        u_clamped[u_params == knot_vector[num_ctrl_pts]] = knot_vector[num_ctrl_pts] - 1e-6

    spans = torch.searchsorted(knot_vector, u_clamped, side='right') - 1
    spans = torch.clamp(spans, degree, n_max_cp_idx)  # Max span index is n (num_ctrl_pts - 1)
    return spans.long()


def build_global_basis_matrix_torch(num_ctrl_pts, degree, eval_params, global_knot_vector, derivative=False):
    """
    Builds a dense global basis (or derivative) matrix B[eval_idx, cp_idx].
    Args:
        num_ctrl_pts (int): Total number of control points in this direction.
        degree (int): Degree of the B-spline.
        eval_params (torch.Tensor): 1D tensor of evaluation parameters (u or v values).
        global_knot_vector (torch.Tensor): 1D global knot vector for this direction.
                                           Expected shape: (num_ctrl_pts + degree + 1,).
        derivative (bool): If True, compute derivative basis functions.
    Returns:
        torch.Tensor: Dense basis matrix of shape (num_eval_points, num_ctrl_pts).
    """
    num_eval_points = eval_params.shape[0]

    # Ensure global_knot_vector is (1, num_knots) for cox_de_boor
    if global_knot_vector.ndim == 1:
        knot_vector_batch = global_knot_vector.unsqueeze(0)
    else:
        knot_vector_batch = global_knot_vector

    # 1. Find spans for all eval_params
    spans = find_spans_torch(num_ctrl_pts, degree, eval_params, global_knot_vector)  # (num_eval_points,)

    # 2. Evaluate (degree+1) non-zero basis functions for each eval_param
    if derivative:
        active_basis_values = cox_de_boor_derivative_batch(knot_vector_batch, degree, eval_params.unsqueeze(0))
    else:
        active_basis_values = cox_de_boor_batch(knot_vector_batch, degree, eval_params.unsqueeze(0))
    # active_basis_values shape: (1, num_eval_points, degree + 1)
    active_basis_values = active_basis_values.squeeze(0)  # (num_eval_points, degree + 1)

    # 3. Scatter these values into the dense global basis matrix
    dense_basis_matrix = torch.zeros(num_eval_points, num_ctrl_pts, device=eval_params.device,
                                     dtype=active_basis_values.dtype)

    s_indices = torch.arange(num_eval_points, device=eval_params.device).unsqueeze(1).expand(-1, degree + 1)
    l_indices_range = torch.arange(degree + 1, device=eval_params.device).unsqueeze(0)  # (1, degree+1)

    # Global CP indices for the (degree+1) active basis functions are: span[s] - degree + l
    gcp_indices = spans.unsqueeze(1) - degree + l_indices_range  # (num_eval_points, degree + 1)

    # Ensure gcp_indices are valid (should be if spans and logic are correct)
    gcp_indices = torch.clamp(gcp_indices, 0, num_ctrl_pts - 1)

    dense_basis_matrix.scatter_(dim=1, index=gcp_indices, src=active_basis_values)
    return dense_basis_matrix


# --- End of spline_utils.py additions ---

def grid_to_patches(global_grid, patch_size=4, stride=0):
    """
    Given a global grid of control features (shape (H, W, D)), extract overlapping patches
    using a sliding window. Each window corresponds to a patch of shape (patch_size, patch_size, D).

    Args:
      global_grid (torch.Tensor): Tensor of shape (H, W, D).
      patch_size (int): The spatial size of a patch (default: 4).
      stride (int): The stride (default: 3, so that adjacent patches share boundaries).

    Returns:
      patches (torch.Tensor): Extracted patches of shape (num_patches, patch_size, patch_size, D).
    """

    ##############################
    H, W, D = global_grid.shape
    stride = patch_size - 1 if not stride else stride
    grid_cf = global_grid.permute(2, 0, 1).unsqueeze(0)          # (1, D, H, W)

    patches_unfold = F.unfold(grid_cf, kernel_size=patch_size, stride=stride)
    # (1, D*P*P, L)

    DPP, L = patches_unfold.shape[1], patches_unfold.shape[2]
    patches = (patches_unfold
               .view(1, D, patch_size, patch_size, L)            # (1, D, P, P, L)
               .squeeze(0)                                       # (D, P, P, L)
               .permute(3, 1, 2, 0) )

    return patches

def knot_insert_1d(ctrl: torch.Tensor,
                   knots: torch.Tensor,
                   degree: int,
                   u_new: float):
    """
    Insert a single knot u_new into a 1-D B-spline control polygon.

    ctrl  : (n_ctrl, D)
    knots : (n_ctrl + degree + 1,)
    degree: p
    Returns (ctrl', knots') where ctrl' has n_ctrl+1 points.
    """
    n, D = ctrl.shape
    k = torch.searchsorted(knots, torch.tensor(u_new, device=knots.device)) - 1
    # number of points to recompute = degree
    new_ctrl = torch.zeros(n+1, D, device=ctrl.device, dtype=ctrl.dtype)
    new_ctrl[:k-degree+1] = ctrl[:k-degree+1]          # left unchanged
    new_ctrl[k+2:]       = ctrl[k+1:]                  # right unchanged
    # recompute the middle degree points
    for j in range(k-degree+1, k+1):
        alpha = (u_new - knots[j]) / (knots[j+degree] - knots[j] + 1e-8)
        new_ctrl[j+1] = (1-alpha)*ctrl[j] + alpha*ctrl[j+1]
    # build new knot vector
    new_knots = torch.cat([knots[:k+1],
                           knots[k:k+1].clone().fill_(u_new),
                           knots[k+1:]], dim=0)
    return new_ctrl, new_knots

def subdivide_nurbs_patch(ctrl_net: torch.Tensor,
                          U: torch.Tensor,
                          V: torch.Tensor,
                          p: int,
                          q: int):
    """
    Exact 4-way subdivision of a single NURBS patch via one knot-insertion
    in each param direction.

    ctrl_net : (n_u, n_v, D)   – control points incl. weight & features
    U        : (n_u + p + 1,)  – knot vector in u
    V        : (n_v + q + 1,)  – knot vector in v
    p, q     : spline degrees
    Returns a list of 4 tuples (ctrl_child, U_child, V_child)
    each ctrl_child has shape (n_u, n_v, D) (same order as original).
    """
    n_u, n_v, D = ctrl_net.shape

    # --- 1) choose middle knots in the span where we split
    #   here we split the whole [0,1] domain in half
    u_mid = 0.5*(U[p] + U[-p-1])
    v_mid = 0.5*(V[q] + V[-q-1])

    # --- 2) insert u_mid in every row ------------------------------------
    rows_refined = []
    U_prime = None
    for j in range(n_v):
        new_row, U_prime = knot_insert_1d(ctrl_net[:, j, :], U, p, u_mid)
        rows_refined.append(new_row)
    rows_refined = torch.stack(rows_refined, dim=1)      # (n_u+1, n_v, D)

    # --- 3) insert v_mid in every column ---------------------------------
    cols_refined = []
    V_prime = None
    for i in range(rows_refined.shape[0]):
        new_col, V_prime = knot_insert_1d(rows_refined[i, :, :], V, q, v_mid)
        cols_refined.append(new_col)
    refined = torch.stack(cols_refined, dim=0)           # (n_u+1, n_v+1, D)

    # --- 4) split control net into TL / TR / BL / BR ----------------------
    k = torch.searchsorted(U_prime, torch.tensor(u_mid)) - 1   # span index
    ℓ = torch.searchsorted(V_prime, torch.tensor(v_mid)) - 1

    tl = refined[:k+1, :ℓ+1, :]
    tr = refined[:k+1, ℓ:, :]
    bl = refined[k:,   :ℓ+1, :]
    br = refined[k:,   ℓ:, :]

    # knot vectors for children
    def child_knots_full(parent, deg, s0, s1):
        left  = parent[:s0+deg+1].clone()
        right = parent[s1:]
        mid   = parent[s0:s0+1].clone().fill_(parent[s0])  # repeated knot
        return torch.cat([left, mid, right], dim=0)

    U_tl = U_prime.clone(); U_tr = U_prime.clone()
    U_bl = U_prime.clone(); U_br = U_prime.clone()
    V_tl = V_prime.clone(); V_tr = V_prime.clone()
    V_bl = V_prime.clone(); V_br = V_prime.clone()

    # return list
    return [
        (tl, U_tl, V_tl),
        (tr, U_tr, V_tr),
        (bl, U_bl, V_bl),
        (br, U_br, V_br),
    ]
def subdivide_patch_batch(ctrl_batch, U_list, V_list, p=3, q=3):
    """
    ctrl_batch : (N, p+1, q+1, D)
    U_list, V_list : length-N list[Tensor]
    Returns:
        child_ctrl   : (4*N, p+1, q+1, D)
        child_U_list : length 4N list
        child_V_list : length 4N list
    """
    child_ctrl, child_U, child_V = [], [], []
    for ctrl, U, V in zip(ctrl_batch, U_list, V_list):
        print(ctrl, U, V)
        for sub_ctrl, sub_U, sub_V in subdivide_nurbs_patch(ctrl, U, V, p, q):
            child_ctrl.append(sub_ctrl)
            child_U.append(sub_U)
            child_V.append(sub_V)
    return (torch.stack(child_ctrl, dim=0),
            child_U,
            child_V)
def subdivide_uv_params2(us: torch.Tensor, vs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Subdivide the uv parameters for each patch into four sets for subpatches.

    For each patch, assume its current u values are in some range [u_min, u_max] and
    v values in [v_min, v_max]. Then, compute:
      - Top-left: (u_left, v_left)
      - Top-right: (u_right, v_left)
      - Bottom-left: (u_left, v_right)
      - Bottom-right: (u_right, v_right)
    where u_left,u_right and v_left,v_right are computed via subdivide_uv_1d.

    Args:
        us (torch.Tensor): shape (N, L) for u coordinates.
        vs (torch.Tensor): shape (N, L) for v coordinates.

    Returns:
        subdivided_us (torch.Tensor): shape (4*N, L)
        subdivided_vs (torch.Tensor): shape (4*N, L)
    """
    u_left, u_right = subdivide_uv_1d(us)
    v_left, v_right = subdivide_uv_1d(vs)
    subdivided_us = torch.cat([u_left, u_right, u_left, u_right], dim=0)  # (4*N, L)
    subdivided_vs = torch.cat([v_left, v_left, v_right, v_right], dim=0)  # (4*N, L)
    return subdivided_us, subdivided_vs


def subdivide_uv_d1(uv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Given a uv tensor of shape (N, L) for a set of patches,
    subdivide it into two new uv arrays by splitting the signal into left and right halves.
    The input is unsqueezed to (N, 1, L) and then linearly upsampled to (N, 1, 2*L - 1).

    Returns:
        uv_left: (N, L) corresponding to the left half after re-normalization.
        uv_right: (N, L) corresponding to the right half after re-normalization.
    """
    N, L = uv.shape
    # Unsqueeze to have shape (N, 1, L)
    uv_unsq = uv.unsqueeze(1)  # (N, 1, L)
    # Upsample to (N, 1, 2*L - 1) using linear interpolation.
    uv_upsampled = F.interpolate(uv_unsq, size=2 * L - 1, mode='linear', align_corners=True)
    uv_upsampled = uv_upsampled.squeeze(1)  # (N, 2*L - 1)
    # Now, the left half can be taken as the first L values,
    # and the right half as the last L values.
    uv_left = uv_upsampled[:, :L]
    uv_right = uv_upsampled[:, -L:]
    return uv_left, uv_right


def patch_subdivision(features: torch.Tensor) -> torch.Tensor:
    """
    Vectorized subdivision of patch features.

    Args:
        features: Tensor of shape (N, R, R, C, 3)

    Returns:
        Tensor of shape (4*N, 4, 4, C, 3)
    """
    # Get the input dimensions.
    N, R, _, C, D = features.shape # if features.ndim == 5 else features.unsqueeze(-2).shape
    refined_res = 2 * R - 1  # New grid resolution after refinement.

    # Rearrange to prepare for 2D interpolation.
    # From (N, R, R, C, 3) -> (N, C, 3, R, R)
    features_perm = features.permute(0, 3, 4, 1, 2)
    # Merge the C and 3 dimensions: (N, C*3, R, R)
    features_flat = features_perm.reshape(N, C * D, R, R)

    # Upsample to (refined_res, refined_res) using bilinear interpolation.
    refined_flat = F.interpolate(features_flat, size=(refined_res, refined_res),
                                 mode='bilinear', align_corners=True)

    # Reshape back: from (N, C*3, refined_res, refined_res) to (N, refined_res, refined_res, C, 3)
    refined = refined_flat.reshape(N, C, D, refined_res, refined_res).permute(0, 3, 4, 1, 2)

    # Now, extract four subpatches, each of shape (R, R, C, 3).
    # Note: In a refined grid of size (2R-1) we have indices 0, 1, ..., 2R-2.
    # We take rows 0 to R-1 (total R rows) for the top subpatches and rows (refined_res-R) to (refined_res-1) for the bottom subpatches.
    top_left = refined[:, :R, :R, :, :]
    top_right = refined[:, :R, R-1:, :, :]
    bottom_left = refined[:, R-1:, :R, :, :]
    bottom_right = refined[:, R-1:, R-1:, :, :]


    # Stack the four subpatches along a new dimension and reshape.
    subpatches = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)  # shape (N, 4, R, R, C, 3)
    # subpatches = subpatches.reshape(4 * N, R, R, C, D)
    return subpatches.squeeze(-2)


def subdivide_uv_params2(us: torch.Tensor, vs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Subdivide the uv parameters for each patch into four sets for subpatches.

    Args:
        us (torch.Tensor): shape (N, L) for u coordinates.
        vs (torch.Tensor): shape (N, L) for v coordinates.

    Returns:
        subdivided_us: shape (4*N, L)
        subdivided_vs: shape (4*N, L)

    The new patches get:
      - Top-left: (u_left, v_left)
      - Top-right: (u_right, v_left)
      - Bottom-left: (u_left, v_right)
      - Bottom-right: (u_right, v_right)
    """
    u_left, u_right = subdivide_uv_1d(us)
    v_left, v_right = subdivide_uv_1d(vs)
    subdivided_us = torch.cat([u_left, u_right, u_left, u_right], dim=0)  # (4*N, L)
    subdivided_vs = torch.cat([v_left, v_left, v_right, v_right], dim=0)  # (4*N, L)
    return subdivided_us, subdivided_vs


def _subdivide_all_patches_once(patches: torch.Tensor) -> torch.Tensor:
    """
    Subdivide a batch of patches (shape: (N, 4, 4, D)) into 4 subpatches per patch
    without duplicating boundary indices. The output shape is (4*N, 4, 4, D).

    This implementation upsamples using bilinear interpolation with scale_factor=2 and
    align_corners=False so that the refined grid becomes (8, 8) (instead of (7, 7)
    which is the result when using align_corners=True). Four non-overlapping 4×4 regions
    are then extracted.
    """
    # Assume patches has shape (N, 4, 4, D)
    # Permute to channel-first for interpolation: shape (N, D, 4, 4)
    patches_flat = patches.permute(0, 3, 1, 2)

    # Upsample by a factor of 2 using bilinear interpolation.
    # When align_corners=False, input 4 becomes output 8.
    refined = F.interpolate(patches_flat, scale_factor=2, mode='bilinear', align_corners=False)
    # refined shape: (N, D, 8, 8)

    # Permute back to shape (N, 8, 8, D)
    refined = refined.permute(0, 2, 3, 1)

    # Extract the four non-overlapping subpatches:
    top_left = refined[:, :4, :4, :]  # rows 0 to 3, columns 0 to 3
    top_right = refined[:, :4, 4:, :]  # rows 0 to 3, columns 4 to 7
    bottom_left = refined[:, 4:, :4, :]  # rows 4 to 7, columns 0 to 3
    bottom_right = refined[:, 4:, 4:, :]  # rows 4 to 7, columns 4 to 7

    # Concatenate along the batch dimension to get 4*N subpatches
    subpatches = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)

    return subpatches


def subdivision(patches: torch.Tensor) -> torch.Tensor:
    """
    Subdivide a batch of square‐grid patches (N, r, r, D) into 4 subpatches per patch,
    ordered so that they form a (2H x 2W) grid in row-major order.

    Args:
        patches: Tensor of shape (N, r, r, D), where N = H * W and H, W form a square grid.

    Returns:
        Tensor of shape (4*N, r, r, D), corresponding to a (2H x 2W) patch grid.
    """
    # Infer sizes
    N, r, _, D = patches.shape
    Hc = int(math.sqrt(N))
    assert Hc * Hc == N, "Patches must form a square grid"
    Wc = Hc

    # 1) Upsample each patch from (r x r) to (2r x 2r)
    patches_flat = patches.permute(0, 3, 1, 2)  # (N, D, r, r)
    refined = F.interpolate(patches_flat,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)  # (N, D, 2r, 2r)
    refined = refined.permute(0, 2, 3, 1)       # (N, 2r, 2r, D)

    # 2) Slice out the four children (non‐overlapping r×r blocks)
    tl = refined[:, 0:r,   0:r,   :]  # top-left
    tr = refined[:, 0:r,   r:2*r, :]  # top-right
    bl = refined[:, r:2*r, 0:r,   :]  # bottom-left
    br = refined[:, r:2*r, r:2*r, :]  # bottom-right

    # Stack them in a children tensor: (4, N, r, r, D)
    children = torch.stack([tl, tr, bl, br], dim=0)

    # 3) Compute the new patch order for a (2Hc × 2Wc) grid
    Hf, Wf = 2 * Hc, 2 * Wc
    total = Hf * Wf
    device = patches.device

    idx = torch.arange(total, device=device)
    row = idx // Wf
    col = idx % Wf

    # Which coarse patch does each subdivided slot come from?
    coarse_i = row // 2
    coarse_j = col // 2
    coarse_idx = coarse_i * Wc + coarse_j       # in [0, N)

    # Which child within that coarse patch?
    # (row%2, col%2) → 0:(0,0)=tl, 1:(0,1)=tr, 2:(1,0)=bl, 3:(1,1)=br
    child_idx = (row % 2) * 2 + (col % 2)        # in [0..3]

    # 4) Gather them into the final tensor of shape (4N, r, r, D)
    refined_patches = children[child_idx, coarse_idx]  # (4N, r, r, D)
    return refined_patches

def _subdivide_all_patches_once2(patches: torch.Tensor) -> torch.Tensor:
    """
    Vectorized implementation to subdivide a batch of patches (shape: (N, 4, 4, D))
    into 4 subpatches per patch (output shape: (4*N, 4, 4, D)).
    This function should be written in a fully vectorized manner.
    """
    # Assume patches has shape (N, 4, 4, D)
    # Use F.interpolate on a tensor of shape (N, D, H, W):
    patches_flat = patches.permute(0, 3, 1, 2)  # (N, D, 4, 4)
    refined = F.interpolate(patches_flat, size=(7, 7), mode='bilinear', align_corners=True)
    # refined = F.interpolate(patches_flat, scale_factor=2, mode='bilinear', align_corners=True)

    # refined now has shape (N, D, 7, 7)
    refined = refined.permute(0, 2, 3, 1)  # (N, 7, 7, D)

    # Extract the four subpatches by slicing:
    top_left = refined[:, :4, :4, :]
    top_right = refined[:, :4, 3:, :]
    bottom_left = refined[:, 3:, :4, :]
    bottom_right = refined[:, 3:, 3:, :]

    # Stack along a new dimension and reshape:
    subpatches = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)  # (N, 4, 4, 4, D)
    # Reshape to (4*N, 4, 4, D)
    # subpatches = subpatches.reshape(4 * N, 4, 4, D)
    return subpatches

def normalize_point_cloud(points):
    """
    Normalize the point cloud to be bounded between -1 and 1.

    Parameters:
    points (torch.tensor): A numpy array of shape (N, 3) representing the point cloud.

    Returns:
    torch.tensor: The normalized point cloud.
    """

    true_shape = points.shape
    if not (points.shape[-1] == 3 and len(points.shape) == 2):
        points = points.reshape(-1, 3)


    # Find the minimum and maximum coordinates along each axis
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # uncomment if we want the scene be centered (Must attend also the cameras parameters)
    # center = (min_coords.values + max_coords.values) / 2. # Calculate the center of the bounding box

    # Calculate the scale factor
    scale = (max_coords - min_coords).max() / 2.
    return scale

def gaussian_kernel(kernel_size: int, sigma: float, channels: int, device='cuda') -> torch.Tensor:
    """
    Create a 2D Gaussian kernel to be used in a convolution.

    Args:
        kernel_size (int): Size of the kernel (must be an odd number).
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels (the kernel is repeated for each channel).

    Returns:
        kernel (torch.Tensor): A tensor of shape (channels, 1, kernel_size, kernel_size) suitable for
                               use in a grouped convolution.
    """
    # Create a 1D tensor with values centered at zero.
    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32) - center
    # Create a 2D grid from the 1D tensor.
    x_grid, y_grid = torch.meshgrid(x, x, indexing="ij")
    gaussian = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()  # Normalize to sum to 1.
    # Expand the kernel to have shape (channels, 1, kernel_size, kernel_size)
    kernel = gaussian.expand(channels, 1, kernel_size, kernel_size).clone()
    return kernel.to(device)


def smooth_and_tangent_grad(cp: torch.Tensor,
                            grad: torch.Tensor,
                            kernel_size: int = 3,
                            sigma: float = 1.0,
                            reduction_factor: float = 1.0) -> torch.Tensor:
    """
    Applies spatial smoothing via a Gaussian convolution on the gradients and then projects the smoothed
    gradients onto the local tangent plane (i.e., removes the normal component) for each patch.

    Args:
        cp (torch.Tensor): Control points with shape (N, H, W, 3), where H and W are the patch grid dimensions.
        grad (torch.Tensor): Gradient tensor with shape (N, H, W, 3).
        kernel_size (int): Size of the Gaussian kernel (default: 3).
        sigma (float): Standard deviation of the Gaussian (default: 1.0).
        reduction_factor (float): Factor for subtracting the normal component (1.0 removes it entirely).

    Returns:
        torch.Tensor: Modified gradient tensor with shape (N, H, W, 3).
    """
    N, H, W, C = grad.shape  # Expect C == 3.
    # --- 1. Spatial Smoothing via Gaussian Convolution ---
    # Permute grad to (N, C, H, W) to apply convolution.
    grad_perm = grad.permute(0, 3, 1, 2)  # shape: (N, 3, H, W)
    # Build a Gaussian kernel.
    kernel = gaussian_kernel(kernel_size, sigma, channels=C)
    # Apply convolution with padding to preserve spatial dimensions.
    smoothed_grad_perm = F.conv2d(grad_perm, weight=kernel, padding=kernel_size // 2, groups=C)
    # Permute back to (N, H, W, C)
    smoothed_grad = smoothed_grad_perm.permute(0, 2, 3, 1)  # shape: (N, H, W, 3)

    # --- 2. Compute Local Normals from Control Points ---
    # cp: (N, H, W, 3). Permute to (N, 3, H, W)
    cp_perm = cp.permute(0, 3, 1, 2)
    # Pad using replication so boundaries are handled.
    cp_padded = F.pad(cp_perm, pad=(1, 1, 1, 1), mode='replicate')  # shape: (N, 3, H+2, W+2)
    # Compute central differences along the vertical (u) direction.
    dU = cp_padded[:, :, 2:, 1:-1] - cp_padded[:, :, :-2, 1:-1]  # shape: (N, 3, H, W)
    # Compute central differences along the horizontal (v) direction.
    dV = cp_padded[:, :, 1:-1, 2:] - cp_padded[:, :, 1:-1, :-2]  # shape: (N, 3, H, W)
    # Estimate local normals as the cross product of dU and dV.
    normal = torch.cross(dU, dV, dim=1)  # shape: (N, 3, H, W)
    # Normalize the normals.
    norm_val = normal.norm(dim=1, keepdim=True) + 1e-8  # shape: (N, 1, H, W)
    normal_unit = normal / norm_val  # shape: (N, 3, H, W)
    # Permute back to (N, H, W, 3)
    normal_unit = normal_unit.permute(0, 2, 3, 1)

    # --- 3. Tangent Projection ---
    # Compute dot product of smoothed_grad with the normal_unit.
    dot = (smoothed_grad * normal_unit).sum(dim=-1, keepdim=True)  # shape: (N, H, W, 1)
    # Remove the normal component.
    projected_grad = smoothed_grad - reduction_factor * dot * normal_unit

    return projected_grad

def smooth_and_tangent_grad2(cp: torch.Tensor,
                            grad: torch.Tensor,
                            kernel: torch.Tensor,
                            reduction_factor: float = 1.0) -> torch.Tensor:
    """
    Applies spatial smoothing via convolution on the gradients and then projects the smoothed
    gradients onto the local tangent plane (i.e. removes the normal component) for each patch.

    Args:
        cp (torch.Tensor): Control points with shape (N, H, W, 3), where H and W are the patch grid dimensions.
        grad (torch.Tensor): Gradient tensor with shape (N, H, W, 3).
        kernel (torch.Tensor): Convolution kernel with shape (C, 1, k, k). For example, for 3 channels,
                               a 3×3 averaging kernel has shape (3, 1, 3, 3). The convolution is applied
                               per-channel by setting groups=C.
        reduction_factor (float): Factor for subtracting the normal component (1.0 removes it entirely).

    Returns:
        torch.Tensor: Modified gradient tensor with shape (N, H, W, 3).
    """
    N, H, W, C = grad.shape  # Expect C == 3.
    # --- 1. Spatial Smoothing via Convolution ---
    # Permute grad to (N, C, H, W) to apply convolution.
    grad_perm = grad.permute(0, 3, 1, 2)  # shape: (N, 3, H, W)
    # Apply 2D convolution with padding=1 to preserve spatial size.
    smoothed_grad_perm = F.conv2d(grad_perm, weight=kernel, padding=1, groups=C)
    # Permute back to (N, H, W, C)
    smoothed_grad = smoothed_grad_perm.permute(0, 2, 3, 1)  # shape: (N, H, W, 3)

    # --- 2. Compute Local Normals from Control Points ---
    # cp is (N, H, W, 3). Permute to (N, 3, H, W) for central differences.
    cp_perm = cp.permute(0, 3, 1, 2)  # (N, 3, H, W)
    # Pad using replication (or reflection) so boundaries are handled.
    cp_padded = F.pad(cp_perm, pad=(1, 1, 1, 1), mode='replicate')  # (N, 3, H+2, W+2)
    # Compute central differences along the vertical direction (u axis)
    dU = cp_padded[:, :, 2:, 1:-1] - cp_padded[:, :, :-2, 1:-1]  # (N, 3, H, W)
    # Compute central differences along the horizontal direction (v axis)
    dV = cp_padded[:, :, 1:-1, 2:] - cp_padded[:, :, 1:-1, :-2]  # (N, 3, H, W)
    # Estimate local normal as the cross product (dU x dV).
    normal = torch.cross(dU, dV, dim=1)  # (N, 3, H, W)
    # Normalize (add a small epsilon to avoid division by zero)
    norm_val = normal.norm(dim=1, keepdim=True) + 1e-8  # (N, 1, H, W)
    normal_unit = normal / norm_val  # (N, 3, H, W)
    # Permute back to (N, H, W, 3)
    normal_unit = normal_unit.permute(0, 2, 3, 1)

    # --- 3. Tangent Projection ---
    # Compute dot product of smoothed_grad with the normal.
    dot = (smoothed_grad * normal_unit).sum(dim=-1, keepdim=True)  # (N, H, W, 1)
    # Subtract the normal component from the smoothed gradient.
    projected_grad = smoothed_grad - reduction_factor * dot * normal_unit

    return projected_grad


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class Gaussians(NamedTuple):
    xyz: torch.Tensor
    features: torch.Tensor
    scaling: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor
    normals: torch.Tensor=None
    active_sh_degree: int = 0


class SplineConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)


    def __getattr__(self, name):
        return self.__dict__.get(name, None)




def compute_bounding_boxes(patches):
    """
    Computes the bounding boxes for each patch.

    Args:
        patches (torch.Tensor): Tensor of shape (N, 4, 4, 3).

    Returns:
        torch.Tensor: Bounding boxes of shape (N, 6), where each box is [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    N = patches.shape[0]
    points = patches.view(N, -1, 3)  # Flatten to (N, 16, 3)

    x_min, _ = points[:, :, 0].min(dim=1)
    y_min, _ = points[:, :, 1].min(dim=1)
    z_min, _ = points[:, :, 2].min(dim=1)
    x_max, _ = points[:, :, 0].max(dim=1)
    y_max, _ = points[:, :, 1].max(dim=1)
    z_max, _ = points[:, :, 2].max(dim=1)

    boxes = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)
    return boxes

def compute_volumes(boxes):
    """
    Computes the volume of each bounding box.

    Args:
        boxes (torch.Tensor): Bounding boxes of shape (N, 6).

    Returns:
        torch.Tensor: Volumes of shape (N,).
    """
    x_min, y_min, z_min, x_max, y_max, z_max = boxes.unbind(dim=1)
    volumes = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    return volumes


def boxes_overlap(boxes1, boxes2):
    """
    Determines if bounding boxes overlap.

    Args:
        boxes1 (torch.Tensor): Bounding boxes of shape (N, 6).
        boxes2 (torch.Tensor): Bounding boxes of shape (M, 6).

    Returns:
        torch.Tensor: Boolean tensor of shape (N, M) indicating overlaps.
    """
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = boxes1[:, None, :].unbind(dim=2)
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = boxes2[None, :, :].unbind(dim=2)

    x_overlap = (x1_min <= x2_max) & (x1_max >= x2_min)
    y_overlap = (y1_min <= y2_max) & (y1_max >= y2_min)
    z_overlap = (z1_min <= z2_max) & (z1_max >= z2_min)

    overlaps = x_overlap & y_overlap & z_overlap
    return overlaps


def compute_iou(boxes1, boxes2):
    """
    Computes the Intersection over Union (IoU) between pairs of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Bounding boxes of shape (N, 6).
        boxes2 (torch.Tensor): Bounding boxes of shape (M, 6).

    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
    """
    # Unpack coordinates
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = boxes1.unbind(dim=1)
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = boxes2.unbind(dim=1)

    # Compute coordinates of the intersection volumes
    x_min = torch.max(x1_min[:, None], x2_min[None, :])
    y_min = torch.max(y1_min[:, None], y2_min[None, :])
    z_min = torch.max(z1_min[:, None], z2_min[None, :])
    x_max = torch.min(x1_max[:, None], x2_max[None, :])
    y_max = torch.min(y1_max[:, None], y2_max[None, :])
    z_max = torch.min(z1_max[:, None], z2_max[None, :])

    # Compute intersection dimensions
    inter_dx = (x_max - x_min).clamp(min=0)
    inter_dy = (y_max - y_min).clamp(min=0)
    inter_dz = (z_max - z_min).clamp(min=0)

    # Compute intersection volume
    inter_volume = inter_dx * inter_dy * inter_dz

    # Compute volumes of each box
    volume1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    volume2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

    # Compute union volume
    union_volume = volume1[:, None] + volume2[None, :] - inter_volume

    # Compute IoU
    iou = inter_volume / (union_volume + 1e-8)  # Add small epsilon to avoid division by zero

    return iou

def nms_patches(patches, scores, iou_threshold):
    """
    Applies Non-Maximum Suppression to the patches using IoU threshold.

    Args:
        patches (torch.Tensor): Tensor of shape (N, 4, 4, 3).
        scores (torch.Tensor): Tensor of shape (N,) containing the representativeness score for each patch.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        torch.Tensor: Indices of patches to keep.
    """
    boxes = compute_bounding_boxes(patches)
    indices = torch.argsort(scores, descending=True)
    keep = []

    while indices.numel() > 0:
        current_idx = indices[0]
        keep.append(current_idx.item())

        if indices.numel() == 1:
            break

        rest_indices = indices[1:]
        current_box = boxes[current_idx].unsqueeze(0)
        rest_boxes = boxes[rest_indices]

        # Compute IoU between the current box and the rest
        iou = compute_iou(current_box, rest_boxes).squeeze(0)

        # Keep only indices where IoU is below the threshold
        indices = rest_indices[iou < iou_threshold]

    result = torch.zeros(patches.shape[0], dtype=torch.bool, device=scores.device)
    keep = torch.tensor(keep, dtype=torch.long)
    result[keep] = True

    return result


def perform_nms_on_patches(patches, opacity=None, normals=None, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression on the given patches using an IoU threshold.

    Args:
        patches (torch.Tensor): Tensor of shape (N, 4, 4, 3).
        features (torch.Tensor): Tensor of shape (N, 4, 4, 3).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        torch.Tensor: The filtered patches after NMS.
    """
    boxes = compute_bounding_boxes(patches)
    scores = compute_volumes(boxes)
    if opacity is not None:
        scores = opacity.reshape(patches.shape[0], -1).mean(-1)

    keep_mask = nms_patches(patches, scores, iou_threshold)

    return patches[keep_mask], keep_mask
import torch
import torch.nn.functional as F

def cox_de_boor_matrix(
    knot: torch.Tensor,
    degree: int,
    us: torch.Tensor
) -> torch.Tensor:
    """
    Compute the B-spline basis matrix via Cox–de Boor recursion.

    Args:
        knot: Tensor of shape (K,) giving the knot vector.
        degree: spline degree p (e.g. 3).
        us:     Tensor of shape (P, D) giving sample parameters in each of P spans (D samples each).

    Returns:
        Tensor of shape (P, D, n_ctrl) where n_ctrl = K - degree - 1.
    """
    device = knot.device
    dtype = knot.dtype
    P, D = us.shape
    # Pad the knot vector with its last value (degree times) so that slicing out-of-bounds is safe
    padded_knot = torch.cat([knot, knot.new_full((degree,), knot[-1])], dim=0)

    # Flatten all sample parameters to a single vector
    u_flat = us.reshape(-1)              # shape (P*D,)
    PD = u_flat.shape[0]

    # Number of zero-degree basis functions = number of knot spans = len(knot)-1
    m = knot.shape[0] - 1

    # Build the 0-th degree basis: indicator on each half-open span [knot[i], knot[i+1])
    u = u_flat.unsqueeze(1)              # (PD,1)
    U_left  = padded_knot[:-1 - degree]                  # (m,)
    U_right = padded_knot[1: -degree]                   # (m,)             # (m,)
    N = ((u >= U_left) & (u < U_right)).float()  # (PD, m)

    # Tweak: include the very last knot exactly
    end_mask = (u_flat == knot[-1])
    if end_mask.any():
        N[end_mask, m-1] = 1.0

    # Recursively grow to degree p
    for p in range(1, degree+1):
        # Prepare denominators for both terms
        denom1 = padded_knot[p : p + m] - padded_knot[:m]             # (m,)
        denom2 = padded_knot[p+1 : p+1 + m] - padded_knot[1 : 1 + m]       # (m,)

        # Safely invert, zeroing out any zero spans
        denom1_recip = torch.where(denom1 > 0, 1.0/denom1, torch.zeros_like(denom1))
        denom2_recip = torch.where(denom2 > 0, 1.0/denom2, torch.zeros_like(denom2))

        # Compute the two coefficient matrices a,b of shape (PD, m)
        a = (u - padded_knot[:m].unsqueeze(0)) * denom1_recip.unsqueeze(0)
        b = (padded_knot[p+1 : p+1 + m].unsqueeze(0) - u) * denom2_recip.unsqueeze(0)

        # Pad N on the right so N[:,i+1] is safe
        N_pad = F.pad(N, (0,1), value=0.0)           # (PD, m+1)

        # Cox–de Boor step
        N = a * N + b * N_pad[:, 1:m+1]             # (PD, m)

    # Number of control points is K - p - 1
    n_ctrl = padded_knot.shape[0] - degree - 1
    # Reshape back into (P, D, n_ctrl)
    return N[:, :n_ctrl].view(P, D, n_ctrl)
def get_basis_functions(basis, device='cuda'):
    """
    Returns the basis matrix for the Catmull-Rom spline.
    Catmull-Rom splines are interpolatory, i.e., the surface passes through the control points.
    """
    if basis == 'catmull_rom':
        M_CR = torch.tensor([
            [-1.0,  3.0, -3.0,  1.0],
            [ 2.0, -5.0,  4.0, -1.0],
            [-1.0,  0.0,  1.0,  0.0],
            [ 0.0,  2.0,  0.0,  0.0]
        ], device=device) * 0.5
        return M_CR.unsqueeze(0)

    elif basis == 'bezier':
        M = torch.tensor([
            [-1.0, 3.0, -3.0, 1.0],
            [3.0, -6.0, 3.0, 0.0],
            [-3.0, 0.0, 3.0, 0.0],
            [1.0, 4.0, 1.0, 0.0]
        ], device=device).unsqueeze(0) / 6.0
        return M

    elif basis == 'hermite':
        M_Hermite = torch.tensor([
            [2.0, -2.0, 1.0, 1.0],
            [-3.0, 3.0, -2.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ], device=device)
        return M_Hermite.unsqueeze(0)
def bernstein_basis_4(u: torch.Tensor) -> torch.Tensor:
    """
    Compute the 4 cubic Bernstein basis functions on input u in [0,1].
    Args:
        u: (..., 1) tensor of parametric values.
    Returns:
        B: (..., 4) tensor where B[...,i] = basis_i(u).
    """
    u0 = (1 - u)**3
    u1 = 3 * u * (1 - u)**2
    u2 = 3 * u**2 * (1 - u)
    u3 = u**3
    return torch.cat([u0, u1, u2, u3], dim=-1)

def bernstein_derivative_4(u: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivatives of the 4 cubic Bernstein basis functions wrt u.
    Args:
        u: (..., 1) tensor of parametric values.
    Returns:
        dB: (..., 4) tensor where dB[...,i] = d/du basis_i(u).
    """
    # d/du (1-u)^3 = -3*(1-u)^2
    du0 = -3 * (1 - u)**2
    # d/du [3u(1-u)^2] = 3*(1-u)^2 - 6u*(1-u)
    du1 = 3 * (1 - u)**2 - 6 * u * (1 - u)
    # d/du [3u^2(1-u)] = 6u*(1-u) - 3u^2
    du2 = 6 * u * (1 - u) - 3 * u**2
    # d/du u^3 = 3u^2
    du3 = 3 * u**2
    return torch.cat([du0, du1, du2, du3], dim=-1)

def get_parameter_vectors(u, v, ord=3):
    """
    Creates batched parameter vectors for u and v parameters.

    Parameters:
    u, v: torch.Tensor of shape (num_patches, N) containing parameter values

    Returns:
    U, V, dU, dV: Parameter matrices of shape (num_patches, N, 1, 4)
    """

    u_powers = torch.stack([u.pow(3), u.pow(2), u, torch.ones_like(u)], dim=-1).requires_grad_()# (num_patches, N, 1, 4)
    v_powers = torch.stack([v.pow(3), v.pow(2), v, torch.ones_like(v)], dim=-1).requires_grad_()

    du_powers = torch.stack([3 * u.pow(2), 2 * u, torch.ones_like(u), torch.zeros_like(u)], dim=-1).requires_grad_()
    dv_powers = torch.stack([3 * v.pow(2), 2 * v, torch.ones_like(v), torch.zeros_like(v)], dim=-1).requires_grad_()
    d2u_powers = None
    d2v_powers = None

    if ord == 3:
        d2u_powers = torch.stack([6 * u, 2*torch.ones_like(u), torch.zeros_like(u), torch.zeros_like(u)], dim=-1).requires_grad_()
        d2v_powers = torch.stack([6 * v, 2*torch.ones_like(v), torch.zeros_like(v), torch.zeros_like(v)], dim=-1).requires_grad_()

    return u_powers, v_powers, du_powers, dv_powers, d2u_powers, d2v_powers
def build_adjacency_from_mapping(control_point_mapping):
    """
    Build an adjacency list (or matrix) between patches based on shared control points.

    Args:
        control_point_mapping (torch.Tensor): shape (N, H, W), where each patch i has
                                              a (H x W) grid of unique control-point indices.

    Returns:
        adjacency (list of sets): adjacency[i] is a set of patch indices that share at least
                                  one control point with patch i.
    """
    N = control_point_mapping.shape[0]

    # Convert each patch's control-point mapping to a set for fast intersection checks.
    points_per_patch = []
    for i in range(N):
        # Flatten the (H x W) mapping to get all unique control-point indices for patch i
        patch_points = set(control_point_mapping[i].flatten().tolist())
        points_per_patch.append(patch_points)

    # Initialize adjacency as a list of sets
    adjacency = [set() for _ in range(N)]

    # Compare each pair of patches to see if they share control points.
    # O(N^2 * (H*W)) approach. For large N, consider a more efficient approach (like an inverted index).
    for i in range(N):
        for j in range(i+1, N):
            # Check if the sets intersect
            if points_per_patch[i].intersection(points_per_patch[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


# Example usage:
# Suppose you have self.control_point_mapping of shape (N, 4, 4).
# Then you can do:
# self.adjacency = build_adjacency_from_mapping(self.control_point_mapping)


def build_adjacency_using_shared_cps(mapping):
    r"""
    Build a patch-level adjacency list using the unique control point mapping.

    The input mapping is assumed to be a LongTensor of shape (N, 4, 4), where
    mapping[i, j, k] gives the unique index for the control point at row j, column k of patch i.
    Two patches are considered adjacent if they share exactly 4 control points along an edge.

    Returns:
      adjacency: a list of sets, where adjacency[i] is a set of patch indices adjacent to patch i.
    """
    N = mapping.shape[0]

    # Helper: extract edge as a tuple (preserving order).
    def get_patch_edges(patch_map):
        # patch_map: (4,4)
        top = tuple(patch_map[0, :].tolist())
        bottom = tuple(patch_map[-1, :].tolist())
        left = tuple(patch_map[:, 0].tolist())
        right = tuple(patch_map[:, -1].tolist())
        return [top, bottom, left, right]

    adjacency = [set() for _ in range(N)]
    edge_dict = {}  # key: edge (as a frozenset or tuple), value: (patch index, edge_id)

    for i in range(N):
        # Get the edges for patch i.
        patch_edges = get_patch_edges(mapping[i])
        for edge_id, edge in enumerate(patch_edges):
            # Use a tuple as canonical representation. If the order must be preserved, we keep it.
            # (Assuming that two patches sharing the same ordered edge are adjacent.)
            key = edge  # Alternatively, key = tuple(sorted(edge)) if order doesn't matter.
            if key in edge_dict:
                j, _ = edge_dict[key]
                adjacency[i].add(j)
                adjacency[j].add(i)
            else:
                edge_dict[key] = (i, edge_id)
    return adjacency


def compute_adjacent_pairs(adjacency, device):
    r"""
    Convert an adjacency list into a tensor of unique patch pairs (i, j) with j > i.

    Args:
      adjacency (list of sets): List where adjacency[i] is a set of patch indices adjacent to patch i.
      device: The torch device on which to create the tensor.

    Returns:
      torch.Tensor: Tensor of shape (M, 2) containing unique adjacent patch pairs,
                    or None if no pairs exist.
    """
    pairs = []
    N = len(adjacency)
    for i in range(N):
        for j in adjacency[i]:
            if j > i:
                pairs.append([i, j])
    if len(pairs) == 0:
        return None
    return torch.tensor(pairs, dtype=torch.long, device=device)


def compute_adjacent_pairs1(adjacency, device):
    """
    Convert an adjacency list into a tensor of unique patch pairs (i, j) with j > i.

    Args:
        adjacency (list of sets): List where adjacency[i] is a set of patch indices adjacent to patch i.
        device: The torch device on which to create the tensor.

    Returns:
        torch.Tensor: Tensor of shape (M, 2) containing unique adjacent patch pairs.
                      Returns None if no pairs exist.
    """
    pairs = []
    N = len(adjacency)
    for i in range(N):
        for j in adjacency[i]:
            if j > i:
                pairs.append([i, j])
    if len(pairs) == 0:
        return None
    return torch.tensor(pairs, dtype=torch.long, device=device)


def build_adjacency_using_shared_cps(control_points):
    """
    Build a patch-level adjacency list by detecting shared edges in control_points.
    Assumes control_points has shape (N, 4, 4, 3) and that boundary control points
    are unified (i.e. identical where shared). Two patches are considered adjacent
    if they share exactly 4 control points along an edge.

    Returns:
        adjacency: a list of sets, with adjacency[i] being a set of patch indices adjacent to patch i.
    """
    N = control_points.shape[0]
    # Flatten patches so that each patch has 16 control points
    cp_flat = control_points.view(N * 16, 3)
    # Use torch.unique to assign a unique ID to each control point.
    _, inverse_indices = torch.unique(cp_flat, dim=0, return_inverse=True)

    # Helper function: for a given patch index, return the flat indices of its four edges.
    def get_patch_edge_indices(base_offset):
        # Each patch (4x4 grid) has these edges (using row-major ordering):
        top = [base_offset + 0, base_offset + 1, base_offset + 2, base_offset + 3]
        bottom = [base_offset + 12, base_offset + 13, base_offset + 14, base_offset + 15]
        left = [base_offset + 0, base_offset + 4, base_offset + 8, base_offset + 12]
        right = [base_offset + 3, base_offset + 7, base_offset + 11, base_offset + 15]
        return [top, bottom, left, right]

    adjacency = [set() for _ in range(N)]
    edge_dict = {}  # maps frozenset({pt1,pt2,pt3,pt4}) to (patch_idx, edge_id)

    for i in range(N):
        base_offset = i * 16
        edges = get_patch_edge_indices(base_offset)
        for edge_id, indices in enumerate(edges):
            # Map the flat indices to unique IDs using inverse_indices
            edge_ids = inverse_indices[torch.tensor(indices)].tolist()
            # Use frozenset (or sorted tuple) for canonical representation.
            key = frozenset(edge_ids)
            if key in edge_dict:
                # Found a shared edge with patch j.
                j, _ = edge_dict[key]
                adjacency[i].add(j)
                adjacency[j].add(i)
            else:
                edge_dict[key] = (i, edge_id)
    return adjacency


def find_clusters(patch_indices, adjacency):
    """
    Find connected clusters within a set of patch indices using the supplied adjacency.

    Args:
        patch_indices (set): The set of patch indices we want to cluster (e.g. under‐reconstructed patches).
        adjacency (list of sets): adjacency[i] is the set of neighbors for patch i.

    Returns:
        A list of sets; each set is a connected component of patches from patch_indices.
    """
    clusters = []
    visited = set()

    for idx in patch_indices:
        if idx in visited:
            continue
        # BFS to find a connected component.
        component = set([idx])
        queue = [idx]
        visited.add(idx)
        while queue:
            current = queue.pop(0)
            try:
                for nb in adjacency[current]:

                    if nb in patch_indices and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        component.add(nb)
            except:
                print(f"Error: Invalid neighbor {nb} for patch {current}")
                continue
        clusters.append(component)
    return clusters


def find_triangle_pairs(triangles, vertices):
    """
    Find adjacent triangles that can be merged into quads.
    Uses edge length and angle criteria to find good pairs.
    """
    # Create edge to triangle mapping
    edge_to_triangle = defaultdict(list)
    for i, tri in enumerate(triangles):
        # Add each edge of the triangle
        for j in range(3):
            edge = tuple(sorted([tri[j], tri[(j + 1) % 3]]))
            edge_to_triangle[edge].append(i)

    # Find potential triangle pairs
    pairs = []
    used_triangles = set()

    def get_triangle_normal(tri_vertices):
        v1 = vertices[tri_vertices[1]] - vertices[tri_vertices[0]]
        v2 = vertices[tri_vertices[2]] - vertices[tri_vertices[0]]
        return np.cross(v1, v2)

    def triangle_angle(tri1_idx, tri2_idx):
        normal1 = get_triangle_normal(triangles[tri1_idx])
        normal2 = get_triangle_normal(triangles[tri2_idx])
        cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    for edge, connected_triangles in edge_to_triangle.items():
        if len(connected_triangles) == 2:
            tri1_idx, tri2_idx = connected_triangles

            # Skip if either triangle is already used
            if tri1_idx in used_triangles or tri2_idx in used_triangles:
                continue

            # Check if triangles are coplanar (angle between normals is small)
            angle = triangle_angle(tri1_idx, tri2_idx)
            if angle < np.pi / 18:  # 10 degrees threshold
                pairs.append((tri1_idx, tri2_idx))
                used_triangles.add(tri1_idx)
                used_triangles.add(tri2_idx)

    return pairs, list(used_triangles)


def triangles_to_quad(tri1, tri2, vertices):
    """
    Convert two adjacent triangles into a quad.
    Returns the vertices in correct order for a quad.
    """
    # Find shared edge vertices
    shared = set(tri1) & set(tri2)
    if len(shared) != 2:
        return None

    # Find unique vertices
    unique1 = list(set(tri1) - shared)[0]
    unique2 = list(set(tri2) - shared)[0]

    # Order vertices to form a valid quad
    shared = list(shared)
    # Check if shared edge needs to be flipped
    if np.dot(vertices[shared[1]] - vertices[shared[0]],
              vertices[unique2] - vertices[unique1]) > 0:
        shared = shared[::-1]

    return [unique1, shared[0], unique2, shared[1]]


def create_quad_mesh_from_points(points, min_points_per_quad=4):
    """
    Create a quad-dominant mesh from 3D point cloud.

    Parameters:
    points: np.ndarray (Nx3)
        Input point cloud coordinates
    min_points_per_quad: int
        Minimum number of points to form a quad

    Returns:
    vertices: np.ndarray (Nx3)
        Vertex coordinates
    quads: list of lists
        Quad face indices
    triangles: list of lists
        Remaining triangle face indices
    """
    from scipy.spatial import Delaunay

    # Perform initial Delaunay triangulation
    tri = Delaunay(points[:, :2])  # Project to 2D for triangulation
    initial_triangles = tri.simplices

    # Find triangle pairs that can form quads
    pairs, used_triangles = find_triangle_pairs(initial_triangles, points)

    # Create quads and keep remaining triangles
    quads = []
    remaining_triangles = []

    # Convert triangle pairs to quads
    for tri1_idx, tri2_idx in pairs:
        quad = triangles_to_quad(
            initial_triangles[tri1_idx],
            initial_triangles[tri2_idx],
            points
        )
        if quad is not None:
            quads.append(quad)

    # Keep remaining triangles
    for i, triangle in enumerate(initial_triangles):
        if i not in used_triangles:
            remaining_triangles.append(triangle.tolist())

    return points, quads, remaining_triangles


def plot_quad_mesh(vertices, quads, triangles, point_cloud=None, figsize=(15, 10)):
    """
    Visualize the quad-dominant mesh and optionally the input point cloud.

    Parameters:
    vertices: np.ndarray (Nx3)
        Vertex coordinates
    quads: list of lists
        Quad face indices
    triangles: list of lists
        Triangle face indices
    point_cloud: np.ndarray (Mx3), optional
        Original point cloud coordinates
    figsize: tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot quads
    quad_vertices = [[vertices[idx] for idx in quad] for quad in quads]
    quad_collection = Poly3DCollection(quad_vertices, alpha=0.6)
    quad_collection.set_facecolor('lightblue')
    quad_collection.set_edgecolor('blue')
    ax.add_collection3d(quad_collection)

    # Plot triangles
    tri_vertices = [[vertices[idx] for idx in tri] for tri in triangles]
    tri_collection = Poly3DCollection(tri_vertices, alpha=0.6)
    tri_collection.set_facecolor('lightgreen')
    tri_collection.set_edgecolor('green')
    ax.add_collection3d(tri_collection)

    # Plot original point cloud if provided
    if point_cloud is not None:
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                   c='red', marker='o', s=1, label='Original Points')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    ax.set_xlim([x_min - 0.1, x_max + 0.1])
    ax.set_ylim([y_min - 0.1, y_max + 0.1])
    ax.set_zlim([z_min - 0.1, z_max + 0.1])

    # Add legend
    ax.legend()

    plt.title('Quad-Dominant Mesh')
    plt.tight_layout()
    plt.show()


def save_to_obj(vertices, quads, triangles, filename):
    """
    Save the quad-dominant mesh to OBJ file format.
    """
    with open(filename, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')

        # Write quads
        for quad in quads:
            f.write(f'f {quad[0] + 1} {quad[1] + 1} {quad[2] + 1} {quad[3] + 1}\n')

        # Write remaining triangles
        for tri in triangles:
            f.write(f'f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n')


import numpy as np
from collections import defaultdict


def find_quad_neighbors(quads, vertices, tolerance=1e-6):
    """
    Find adjacent quads for each quad in the mesh.
    Returns a dictionary mapping quad index to its neighbors in each direction.

    Parameters:
    quads: list of lists
        List of quad vertex indices
    vertices: np.ndarray
        Vertex coordinates
    tolerance: float
        Distance tolerance for considering vertices coincident

    Returns:
    dict: {quad_idx: {'north': idx, 'south': idx, 'east': idx, 'west': idx}}
    """

    def get_edge_midpoint(v1, v2):
        return (vertices[v1] + vertices[v2]) / 2

    def get_quad_edges(quad):
        # Returns edges in order: north, east, south, west
        return [
            (quad[0], quad[1]),  # north
            (quad[1], quad[2]),  # east
            (quad[2], quad[3]),  # south
            (quad[3], quad[0])  # west
        ]

    # Create edge to quad mapping
    edge_to_quad = defaultdict(list)
    for quad_idx, quad in enumerate(quads):
        edges = get_quad_edges(quad)
        for edge_idx, (v1, v2) in enumerate(edges):
            mid = tuple(get_edge_midpoint(v1, v2))
            edge_to_quad[mid].append((quad_idx, edge_idx))

    # Find neighbors for each quad
    neighbors = defaultdict(dict)
    directions = ['north', 'east', 'south', 'west']

    for mid, connected in edge_to_quad.items():
        if len(connected) == 2:
            quad1, edge1 = connected[0]
            quad2, edge2 = connected[1]

            # Add bidirectional connections
            neighbors[quad1][directions[(edge1 + 2) % 4]] = quad2
            neighbors[quad2][directions[(edge2 + 2) % 4]] = quad1

    return neighbors


def cluster_quads(quads, vertices, min_patch_size=4):
    """
    Cluster quads into 4x4 patches.

    Parameters:
    quads: list of lists
        List of quad vertex indices
    vertices: np.ndarray
        Vertex coordinates
    min_patch_size: int
        Minimum size of patch in each direction

    Returns:
    list of lists: Each inner list contains 16 quad indices forming a 4x4 patch
    """
    neighbors = find_quad_neighbors(quads, vertices)
    used_quads = set()
    patches = []

    def grow_patch(start_quad):
        """
        Grow a 4x4 patch starting from a quad by walking in consistent directions.
        """
        if start_quad in used_quads:
            return None

        patch = []
        current = start_quad

        # Try to grow in both directions
        for _ in range(2):  # Rows
            row = []
            current_in_row = current

            for _ in range(2):  # Columns
                if current_in_row is None or current_in_row in used_quads:
                    return None

                row.append(current_in_row)
                current_in_row = neighbors.get(current_in_row, {}).get('east')

            patch.extend(row)
            current = neighbors.get(current, {}).get('south')

        # Check if we have a complete 4x4 patch
        if len(patch) == 16 and all(q is not None for q in patch):
            return patch
        return None

    # Try to grow patches starting from each unused quad
    for quad_idx in range(len(quads)):
        if quad_idx in used_quads:
            continue

        patch = grow_patch(quad_idx)
        if patch is not None:
            patches.append(patch)
            used_quads.update(patch)

    return patches

# Example usage
def main2():
    # Generate sample points (on a curved surface)
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * np.sin(0.5 * X) * np.cos(0.5 * Y)

    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create quad-dominant mesh
    vertices, quads, remaining_triangles = create_quad_mesh_from_points(points)

    # Plot the mesh with original points
    plot_quad_mesh(vertices, quads, remaining_triangles, points)

    # Save to OBJ file
    save_to_obj(vertices, quads, remaining_triangles, "quad_mesh.obj")

def create_spline_patches(patches, quads, vertices):
    """
    Create control points for bi-cubic spline patches.

    Parameters:
    patches: list of lists
        List of 4x4 quad clusters
    quads: list of lists
        List of quad vertex indices
    vertices: np.ndarray
        Vertex coordinates

    Returns:
    list of np.ndarray: List of 4x4x3 control point arrays for each patch
    """
    control_points = []

    for patch in patches:
        # Create 4x4x3 array for control points
        patch_points = np.zeros((4, 4, 3))

        # Convert patch quads to 4x4 grid of vertices
        for i in range(4):
            for j in range(4):
                quad_idx = patch[i * 4 + j]
                quad = quads[quad_idx]
                # Use quad vertices as control points
                patch_points[i, j] = vertices[quad[0]]

        control_points.append(patch_points)

    return control_points


def evaluate_spline_patch(control_points, u, v):
    """
    Evaluate a bi-cubic spline patch at parameter values u, v.

    Parameters:
    control_points: np.ndarray (4x4x3)
        Control points for the patch
    u, v: float
        Parameter values in [0, 1]

    Returns:
    np.ndarray: 3D point on the surface
    """

    # Cubic Bernstein basis functions
    def bernstein(t):
        return np.array([
            (1 - t) ** 3,
            3 * t * (1 - t) ** 2,
            3 * t ** 2 * (1 - t),
            t ** 3
        ])

    # Evaluate basis functions
    bu = bernstein(u)
    bv = bernstein(v)

    # Evaluate surface point
    point = np.zeros(3)
    for i in range(4):
        for j in range(4):
            point += bu[i] * bv[j] * control_points[i, j]

    return point

def visualize_patches(quads, vertices):
    """
    Visualize the quad patches and their spline surfaces,
    displaying vertex indices and face indices.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original quad mesh and face indices
    for quad_idx, quad in enumerate(quads):
        vertices_quad = vertices[quad]
        vertices_quad = np.vstack((vertices_quad, vertices_quad[0]))  # Close the loop
        ax.plot(vertices_quad[:, 0], vertices_quad[:, 1], vertices_quad[:, 2], 'b-', alpha=0.3)

        # Calculate center of the face
        face_center = np.mean(vertices_quad[:-1], axis=0)  # Exclude the duplicated last vertex
        # Annotate face index at the center
        ax.text(face_center[0], face_center[1], face_center[2], str(quad_idx),
                color='black', fontsize=8, ha='center', va='center')

    # Plot patches in different colors
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(patches)))

    # for patch, color in zip(patches, colors):
    #     # Plot patch boundaries
    #     for quad_idx in patch:
    #         quad = quads[quad_idx]
    #         vertices_quad = vertices[quad]
    #         vertices_quad = np.vstack((vertices_quad, vertices_quad[0]))
    #         ax.plot(vertices_quad[:, 0], vertices_quad[:, 1], vertices_quad[:, 2],
    #                 '-', color=color, linewidth=2)
    #
    # # Plot vertex indices
    for idx, vertex in enumerate(vertices):
        ax.text(vertex[0], vertex[1], vertex[2], str(idx),
                color='red', fontsize=8, ha='right', va='bottom')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90, azim=-90)

    plt.title('Quad Patches with Vertex and Face Indices')
    plt.show()


import numpy as np


def get_next_quad_index(unused_quads):
    #todo: think if better way to get next quad
    return int(np.array(list(unused_quads)).min())


def probe_quad_patches(vertices, quads, patch_size=(4, 4)):
    """
    Extract 4x4x3 patches from a quad mesh with correct UV parametrization.

    Args:
    - vertices: Numpy array of vertex coordinates (N x 3)
    - quads: List of quad vertex indices

    Returns:
    - Numpy array of patch vertex indices, shape (num_patches, 4, 4, 3)
    - Numpy array of patch vertex coordinates, shape (num_patches, 4, 4, 3)
    """
    POINT_TO_NEXT_V = 1
    POINT_TO_NEXT_U = 3
    # Determine mesh dimensions
    u, v = patch_size

    unused_quads = set()
    unused_quads.update(i for i in range(len(quads)))

    vertices = np.array(vertices)
    quads = np.array(quads)
    patches = []
    # active_quad_index = current_quad_index
    cur_quad_id = get_next_quad_index(unused_quads)
    next_quad_id_v = None
    while len(unused_quads) > 16:

        # Define initial quad before probing in UV directions
        cur_quad_ptrs = quads[cur_quad_id]  # [v1, v2, v3, v4]
        cur_quad_xyz = vertices[cur_quad_ptrs]

        next_quad_id_u = quads[:, 0] == cur_quad_ptrs[POINT_TO_NEXT_U]
        next_quad_id_u = int(np.argwhere(next_quad_id_u))
        next_u_quad_ptrs = quads[next_quad_id_u]
        next_u_quad_xyz = vertices[next_u_quad_ptrs]

        # Init blank patch to occupy quad vertices
        patch = np.zeros((4, 4, 3), dtype=np.float32)
        # Occupy all corodinates (i, j) where 0 <= i,j <= 3
        for u_step in range(u):
            for v_step in range(v):
                patch[u_step, v_step] = cur_quad_xyz[0]
                unused_quads.remove(int(cur_quad_id))
                print(f"quad at index: {cur_quad_id} was removed from pool")

                try:
                    if v_step < 3:
                        vrt_id_in_v_step = cur_quad_ptrs[POINT_TO_NEXT_V]
                        cur_quad_id = get_quad_ID_upon_step(vrt_id_in_v_step, quads)
                        cur_quad_ptrs = quads[cur_quad_id]
                        cur_quad_xyz = vertices[cur_quad_ptrs]

                    if v_step == 3 and u_step == 0 and not next_quad_id_v:
                        vrt_id_in_v_step = cur_quad_ptrs[POINT_TO_NEXT_V]
                        next_quad_id_v = get_quad_ID_upon_step(vrt_id_in_v_step, quads)
                        next_quad_ptrs_v = quads[next_quad_id_v]
                        next_quad_xyz_v = vertices[next_quad_ptrs_v]
                except KeyError as e:
                    print(e)
                    break
                if not next_quad_id_u:
                    break
            print(f"U: quad at index: {next_quad_id_u} was")
            cur_quad_ptrs = next_u_quad_ptrs
            cur_quad_xyz = next_u_quad_xyz
            cur_quad_id = next_quad_id_u

            # Prepare next quad in U direction
            try:
                next_quad_id_u = quads[:, 0] == cur_quad_ptrs[POINT_TO_NEXT_U]
                next_quad_id_u = int(np.argwhere(next_quad_id_u))
                next_u_quad_ptrs = quads[next_quad_id_u]
                next_u_quad_xyz = vertices[next_u_quad_ptrs]

            # When triggered, it means no more U-probing, do V-probe single step
            except TypeError as _:
                next_quad_id_u = None
                cur_quad_id = next_quad_id_v
                cur_quad_ptrs = next_quad_ptrs_v
                cur_quad_xyz = next_quad_xyz_v
                next_quad_id_v = None

            # unused_patches.remove(next_u_index)
            # print(f"quad at index: {int(next_u_index)} was removed from pool")
        # if not next_quad_id_u:

        patches.append(patch)
        # cur_quad_id = unused_quads.pop()

        print(f"quads left: {len(unused_quads)}")

    return patches


def get_quad_ID_upon_step(quad_in_v_step, quads):
    return int(np.argwhere(quads[:, 0] == quad_in_v_step))


# Example usage remains the same as in the previous implementation
def main():
    # Create a sample mesh
    n, m = 8, 8
    x = np.linspace(-5, 5, n + 1)
    y = np.linspace(-5, 5, m + 1)
    X, Y = np.meshgrid(x, y)
    Z = 0.5*np.sin(0.1 * X) * np.cos(0.1 * Y)

    # Create vertices and quads
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    quads = []
    for i in range(n):
        for j in range(m):
            quad = [
                i * (m + 1) + j,
                i * (m + 1) + (j + 1),
                (i + 1) * (m + 1) + (j + 1),
                (i + 1) * (m + 1) + j
            ]
            quads.append(quad)
    visualize_patches(quads, vertices)
    main(vertices, quads, max_patch_size=1)

    # Extract patches
    patches_indices, patches_vertices = probe_quad_patches(vertices, quads)

    print(f"Number of patches: {len(patches_indices)}")
    print(f"Patch indices shape: {patches_indices.shape}")
    print(f"Patch vertices shape: {patches_vertices.shape}")

    # Cluster quads into patches
    patches = cluster_quads(quads, vertices)


    # Create spline patches
    control_points = create_spline_patches(patches, quads, vertices)

    # Visualize patches




def build_edge_to_quads(quads):
    """
    Build a mapping from edges to the quads that contain them.

    Returns:
        edge_to_quads: dict mapping edge (vertex index tuple) to list of quad indices
    """
    edge_to_quads = {}
    for quad_idx, quad in enumerate(quads):
        # Get edges of the quad
        edges = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0])
        ]
        for v1, v2 in edges:
            edge = tuple(sorted((v1, v2)))  # Store edges with sorted vertex indices
            if edge not in edge_to_quads:
                edge_to_quads[edge] = []
            edge_to_quads[edge].append(quad_idx)
    return edge_to_quads
def build_quad_adjacency(quads, edge_to_quads):
    """
    Build a mapping from each quad to its neighboring quads.

    Returns:
        quad_to_neighbors: dict mapping quad index to list of neighboring quad indices
    """
    quad_to_neighbors = {}
    for quad_idx, quad in enumerate(quads):
        neighbors = set()
        # Get edges of the quad
        edges = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0])
        ]
        for v1, v2 in edges:
            edge = tuple(sorted((v1, v2)))
            quads_sharing_edge = edge_to_quads[edge]
            for neighbor_quad_idx in quads_sharing_edge:
                if neighbor_quad_idx != quad_idx:
                    neighbors.add(neighbor_quad_idx)
        quad_to_neighbors[quad_idx] = list(neighbors)
    return quad_to_neighbors
def cluster_quads_into_patches(quads, quad_to_neighbors, max_patch_size=16):
    """
    Cluster quads into patches.

    Args:
        quads: list of quads
        quad_to_neighbors: dict mapping quad index to list of neighboring quad indices
        max_patch_size: maximum number of quads per patch (optional)

    Returns:
        quad_to_patch: dict mapping quad index to patch index
        patches: list of lists of quad indices
    """
    visited = set()
    quad_to_patch = {}
    patches = []
    patch_index = 0

    for quad_idx in range(len(quads)):
        if quad_idx not in visited:
            # Start a new patch
            patch_quads = []
            queue = [quad_idx]
            while queue:
                current_quad = queue.pop(0)
                if current_quad in visited:
                    continue
                visited.add(current_quad)
                quad_to_patch[current_quad] = patch_index
                patch_quads.append(current_quad)
                # Check if we have reached the maximum patch size
                if max_patch_size is not None and len(patch_quads) >= max_patch_size:
                    continue
                # Add unvisited neighbors to the queue
                for neighbor in quad_to_neighbors[current_quad]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            patches.append(patch_quads)
            patch_index += 1
    return quad_to_patch, patches
def build_patch_adjacency(patches, quad_to_neighbors, quad_to_patch):
    """
    Build a mapping from each patch to the patches it is adjacent to.

    Returns:
        patch_to_adjacent_patches: dict mapping patch index to set of adjacent patch indices
    """
    patch_to_adjacent_patches = {}
    for patch_idx, patch_quads in enumerate(patches):
        adjacent_patches = set()
        for quad_idx in patch_quads:
            for neighbor_quad_idx in quad_to_neighbors[quad_idx]:
                neighbor_patch_idx = quad_to_patch[neighbor_quad_idx]
                if neighbor_patch_idx != patch_idx:
                    adjacent_patches.add(neighbor_patch_idx)
        patch_to_adjacent_patches[patch_idx] = adjacent_patches
    return patch_to_adjacent_patches
def identify_shared_edges_between_patches(patches, quads, quad_to_patch):
    """
    Identify shared edges between adjacent patches.

    Returns:
        shared_edges: dict mapping (patch1, patch2) to list of shared edges
    """
    edge_to_quads = build_edge_to_quads(quads)
    shared_edges = {}
    for edge, quads_on_edge in edge_to_quads.items():
        if len(quads_on_edge) == 2:
            quad1, quad2 = quads_on_edge
            patch1 = quad_to_patch[quad1]
            patch2 = quad_to_patch[quad2]
            if patch1 != patch2:
                key = tuple(sorted((patch1, patch2)))
                if key not in shared_edges:
                    shared_edges[key] = []
                shared_edges[key].append(edge)
    return shared_edges


def load_quad_mesh_from_obj(file_path):
    vertices = []
    quads = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = tuple(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.strip().split()
                face_indices = []
                for part in parts[1:]:
                    idx = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                    face_indices.append(idx)
                if len(face_indices) == 4:
                    quads.append(face_indices)
                else:
                    # Handle triangles or other polygons if necessary
                    pass
    return vertices, quads

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
def visual_comparison(gt_image, image, rend_normal, depthmap ,surf_normal, viewpoint_cam, title=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))  # Adjust the figure size as needed
    # Clamp images to the range [0, 1]
    # gt_image.clamp_(min=0, max=1)
    # image.clamp_(min=0, max=1)
    def prepare_image(img):
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            img = torch.clip(img, 0, 1)  # Clip values to [0, 1] range
        else:
            img = img.to(torch.float32) / 255.0  # Convert to float and normalize

        return img


    # Plot Ground Truth Image
    ax[0, 0].imshow(prepare_image(gt_image).cpu().detach().permute(1, 2, 0).numpy())
    ax[0, 0].set_title(f"Ground Truth (ID: {viewpoint_cam.uid})")
    ax[0, 0].axis('off')
    # Plot Splat Image
    ax[0, 1].imshow(prepare_image(image).cpu().detach().permute(1, 2, 0).numpy())
    ax[0, 1].set_title(f"Splat Image (ID: {viewpoint_cam.uid})")
    ax[0, 1].axis('off')
    # Plot Inferred Normals
    rend_normal = np.linalg.norm(prepare_image(rend_normal).cpu().detach().permute(1, 2, 0).numpy(), axis=2) * 0.5 + 0.5  # Normalize to [0, 1] range
    surf_normal = np.linalg.norm(prepare_image(surf_normal).cpu().detach().permute(1, 2, 0).numpy(), axis=2) * 0.5 + 0.5  # Normalize to [0, 1] range
    # rend_normal = rend_normal.cpu().detach().permute(1, 2, 0).numpy()
    depthmap = np.linalg.norm(depthmap.cpu().detach().permute(1, 2, 0).numpy(), axis=2) #* 0.5 + 0.5
    # depthmap = np.linalg.norm(depthmap.cpu().detach().permute(1, 2, 0).numpy(), axis=2) #* 0.5 + 0.5
    ax[1, 0].imshow(rend_normal, cmap='plasma')
    ax[1, 0].set_title(f"Inferred Normals (ID: {viewpoint_cam.uid})")
    ax[1, 0].axis('off')

    # Plot Depth Map
    ax[1, 1].imshow(depthmap, cmap='viridis')
    ax[1, 1].set_title(f"Surface normal (ID: {viewpoint_cam.uid})")
    ax[1, 1].axis('off')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_split_unsplit(render_pkg_split, render_pkg_unsplit, gt_image, viewpoint_cam):
    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    fig.suptitle(f"Split vs Unsplit Comparison - View {viewpoint_cam.image_name}", fontsize=16)

    # Helper function to process and plot images
    def plot_image(ax, img, title, norm=False, cmap=None):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[0] <= 3:  # If channels are first, transpose
            img = img.transpose(1, 2, 0)
        if norm:
            img = np.linalg.norm(img, axis=2)

        # Normalize the image data
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)  # Clip values to [0, 1] range
        # elif img.dtype == np.uint8:
        #     img = img.astype(np.float32) / 255.0  # Convert to float and normalize
        if cmap is not None:
            if cmap == 'plasma':
              ax.imshow(img*0.5 +0.5, cmap=cmap)
            else:
              ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # Row 1: Rendered Images
    plot_image(axes[0, 0], render_pkg_split["render"], "Split Render")
    plot_image(axes[0, 1], render_pkg_unsplit["render"], "Unsplit Render")
    # rend_normal = np.linalg.norm(prepare_image(rend_normal).cpu().detach().permute(1, 2, 0).numpy(),
    #                              axis=2) * 0.5 + 0.5  # Normalize to [0, 1] range
    # rend_normal = rend_normal.cpu().detach().permute(1, 2, 0).numpy()
    # depthmap = np.linalg.norm(depthmap.cpu().detach().permute(1, 2, 0).numpy(), axis=2)  # * 0.5 + 0.5
    # Row 2: Normal Maps
    plot_image(axes[1, 0], render_pkg_split['rend_normal'], "Split Normal", norm=True, cmap='plasma')
    plot_image(axes[1, 1], render_pkg_unsplit['rend_normal'], "Unsplit Normal", norm=True, cmap='plasma')

    # Row 3: Depth Maps
    # plot_image(axes[2, 0], render_pkg_split['surf_depth'], "Split Depth",norm=True, cmap='viridis')
    # plot_image(axes[2, 1], render_pkg_unsplit['surf_depth'], "Unsplit Depth",norm=True, cmap='viridis')

    # Row 4: Ground Truth and Error Map
    # plot_image(axes[3, 0], gt_image, "Ground Truth")
    # Compute and plot error map
    # error_map = torch.abs(render_pkg_split["render"] - render_pkg_unsplit["render"]).mean(dim=0)
    # plot_image(axes[3, 1], error_map, "Error Map (Split - Unsplit)")

    plt.tight_layout()
    plt.show()




def merge_control_points(per_patch_cp: torch.Tensor, tol: float = 1e-6):
    """
    Merge redundant control points from a per-patch representation.

    Args:
        per_patch_cp (torch.Tensor): Tensor of shape (N, 4, 4, 3) containing the control points for each patch.
        tol (float): Tolerance for considering two control points as identical.

    Returns:
        global_cp (torch.Tensor): A tensor of shape (M, 3) of unique control points.
        patch_cp_idx (torch.LongTensor): A tensor of shape (N, 4, 4) such that for each patch and local position,
                                           patch_cp_idx[n, i, j] is the index into global_cp.
    """
    N, H, W, C = per_patch_cp.shape
    # Reshape to a list of control points: shape (N*H*W, 3)
    cp_flat = per_patch_cp.view(-1, C)

    # We'll perform a tolerance-based uniqueness.
    # One simple approach is to round the coordinates to a given tolerance.
    rounded = (cp_flat / tol).round().to(torch.int64)
    # Create a unique key for each control point as a tuple.
    # (Since we want to do this on GPU ideally, we can use torch.unique on the rows.)
    unique_keys, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)

    # unique_keys.shape[0] is the number of unique control points.
    M = unique_keys.shape[0]
    global_cp = cp_flat[inverse_indices.new_tensor(range(M))]  # This won't work directly.
    # Instead, we need to gather one instance per unique key. For that, we use scatter or index_select.
    # A simple approach: for each unique key, pick the first occurrence.
    # We can use inverse_indices to reconstruct the unique control points.
    global_cp = torch.zeros((M, C), dtype=per_patch_cp.dtype, device=per_patch_cp.device)
    # Create an array to store a flag whether we've assigned this unique control point.
    assigned = torch.zeros(M, dtype=torch.bool, device=per_patch_cp.device)
    # Prepare an index map for the flattened control points.
    cp_indices = torch.arange(cp_flat.shape[0], device=per_patch_cp.device)
    # We will build global_cp by scanning through cp_flat and assigning if not yet assigned.
    for i in range(cp_flat.shape[0]):
        idx = inverse_indices[i].item()
        if not assigned[idx]:
            global_cp[idx] = cp_flat[i]
            assigned[idx] = True
    # Build patch_cp_idx by reshaping inverse_indices.
    patch_cp_idx = inverse_indices.view(N, H, W)
    return global_cp, patch_cp_idx


def view_stereo(gt_image, original_rgb, synth_stereo_rgb, original_depth, stereo_depthmap, original_normal, stereo_normal,
                viewpoint_cam, pred_norm, pred_depthmap):
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # Try changing to this backend
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    # Normalize depth maps
    original_depth = (original_depth - original_depth.min()) / (original_depth.max() - original_depth.min())
    stereo_depthmap = (stereo_depthmap - stereo_depthmap.min()) / (stereo_depthmap.max() - stereo_depthmap.min())

    # List of images and their corresponding titles
    images = [
        (original_rgb, f"Original RGB (ID: {viewpoint_cam.uid})"),
        (original_normal, f"Original Normals (ID: {viewpoint_cam.uid})"),
        (original_depth, f"Original Depth Map (ID: {viewpoint_cam.uid})"),
        (synth_stereo_rgb, f"Splat Image (ID: {viewpoint_cam.uid})"),
        (stereo_normal, f"Shifted Normals (ID: {viewpoint_cam.uid})"),
        (stereo_depthmap, f"Shifted Depth Map (ID: {viewpoint_cam.uid})"),
        (gt_image, f"GT RGB (ID: {viewpoint_cam.uid})"),
        (pred_norm, f"Disparity-based Depth Map (ID: {viewpoint_cam.uid})"),
        (pred_depthmap, f"Disparity-based Normal Map (ID: {viewpoint_cam.uid})")
    ]

    for idx, (img, title) in enumerate(images):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img.clamp_(min=0, max=1).cpu().detach().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis('off')

    plt.subplots_adjust(wspace=1, hspace=0)
    plt.tight_layout()


    # Ensure no axes are visible
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def view_stereo(gt_image, original_rgb, synth_stereo_rgb, original_depth, stereo_depthmap, original_normal, stereo_normal,
                viewpoint_cam, pred_norm, pred_depthmap):
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # Try changing to this backend
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    # Normalize depth maps
    original_depth = (original_depth - original_depth.min()) / (original_depth.max() - original_depth.min())
    stereo_depthmap = (stereo_depthmap - stereo_depthmap.min()) / (stereo_depthmap.max() - stereo_depthmap.min())

    # List of images and their corresponding titles
    images = [
        (original_rgb, f"Original RGB (ID: {viewpoint_cam.uid})"),
        (original_normal, f"Original Normals (ID: {viewpoint_cam.uid})"),
        (original_depth, f"Original Depth Map (ID: {viewpoint_cam.uid})"),
        (synth_stereo_rgb, f"Splat Image (ID: {viewpoint_cam.uid})"),
        (stereo_normal, f"Shifted Normals (ID: {viewpoint_cam.uid})"),
        (stereo_depthmap, f"Shifted Depth Map (ID: {viewpoint_cam.uid})"),
        (gt_image, f"GT RGB (ID: {viewpoint_cam.uid})"),
        (pred_norm, f"Disparity-based Depth Map (ID: {viewpoint_cam.uid})"),
        (pred_depthmap, f"Disparity-based Normal Map (ID: {viewpoint_cam.uid})")
    ]

    for idx, (img, title) in enumerate(images):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img.clamp_(min=0, max=1).cpu().detach().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis('off')

    plt.subplots_adjust(wspace=1, hspace=0)
    plt.tight_layout()


    # Ensure no axes are visible
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def clean_pcd(pts3d, upper_quantile=1., lower_quantile=.0):
    from simple_knn._C import distCUDA2
    dists = distCUDA2(torch.tensor(pts3d).cuda()).float()
    print(f"Total points before filtering: {pts3d.shape[0]}")
    max_dist = torch.quantile(dists, q=upper_quantile).item()
    min_dist = torch.quantile(dists, q=lower_quantile).item()
    mask = (dists > min_dist) & (dists < max_dist)
    print(f"Total points after filtering: {mask.sum()}")
    return mask.detach().cpu().numpy()


import torch
import itertools


def find_highest_curvature_track(patch):
    """
    Find the track with the highest curvature in a 4x4 patch.

    Args:
    patch (torch.Tensor): A 4x4x3 tensor representing the control points of the patch.

    Returns:
    tuple: (highest_curvature, best_track)
    """

    def calculate_curvature(points):
        """Calculate the curvature of a set of points."""

        # Simple curvature calculation using the magnitude of the cross product
        v1 = points[1] - points[0]
        v2 = points[2] - points[1]
        v3 = points[3] - points[2]
        cross1 = torch.cross(v1, v2)
        cross2 = torch.cross(v2, v3)
        curvature = (cross1.norm() + cross2.norm()) / 2
        return curvature

    # Define all valid tracks
    valid_tracks = [
        [(0, 0), (1, 1), (2, 2), (3, 3)],  # Main diagonal
        [(0, 3), (1, 2), (2, 1), (3, 0)],  # Anti-diagonal
        [(0, 1), (1, 1), (2, 2), (3, 2)],  # Split vertically
        [(1, 0), (1, 1), (2, 2), (2, 3)],  # Split horizontally
        [(0, 2), (1, 2), (2, 1), (3, 1)],  # Another vertical split
        [(1, 3), (1, 2), (2, 1), (2, 0)]  # Another horizontal split
    ]

    highest_curvature = 0
    best_track = None

    for track in valid_tracks:
        points = torch.stack([patch[i, j] for i, j in track])
        curvature = calculate_curvature(points)
        if curvature > highest_curvature:
            highest_curvature = curvature
            best_track = track

    return highest_curvature, best_track



def from_pcd_to_NURBS(pcd, cameras):
    """
    :param config:
    :param pcd: BasicPointCloud (pcd.points, pcd.colors, pcd.normals).
    :param cameras: List of camera parameters.
    :return: List of transformed point cloud tensors for each camera.
    """
    mask = clean_pcd(pcd.points, upper_quantile=.95, lower_quantile=.0)
    pcd = BasicPointCloud(points=pcd.points[mask].astype(np.float32),
                          colors=pcd.colors[mask].astype(np.float32),
                          normals=np.zeros_like(pcd.points[mask].astype(np.float32)))
    # cameras = cameras[::10]
    cams = []
    for cam in cameras:
        if int(cam.uid) in [13, 41, 45]:
            cams.append(cam)
    # Step 1: Project points onto images using multiprocessing
    cam_UV_list = project_points_to_image(np.float32(pcd.points), pcd.colors, cameras)

    # Step 2: Organize points into patches using multiprocessing
    # with multiprocessing.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
    #     patches = pool.map(process_camera_patches, cam_UV_list)
    # patches = {"points": [], "colors": []}
    patches = []
    for cam in cam_UV_list:
        patch_batch = process_camera_patches(cam)
        # patches["points"].extend(patch_batch["points"])
        # patches["colors"].extend(patch_batch["colors"])
        patches.append(patch_batch)

    points = []
    colors = []
    for p in patches:
        colors.extend(p['colors'])
        points.extend((p['points']))
    print(f"Raw Patches: {len(points)}")
    # exclude redundant patches by applying NMS
    points, keep_mask = perform_nms_on_patches(torch.stack(points), iou_threshold=.95)
    print(f"Afte removing overlaps patches: {len(points)}")
    colors = torch.stack(colors)[keep_mask]
    return {'points': points.to(device='cuda'), 'colors': colors.to(device='cuda')}



def mask3d(pcd, cameras):
    """
    :param config:
    :param pcd: BasicPointCloud (pcd.points, pcd.colors, pcd.normals).
    :param cameras: List of camera parameters, image and mask (Optional).
    :return: List of transformed point cloud tensors for each camera.
    """
    mask = clean_pcd(pcd.points, upper_quantile=.8, lower_quantile=.0)
    pcd = BasicPointCloud(points=pcd.points[mask].astype(np.float32),
                          colors=pcd.colors[mask].astype(np.float32),
                          normals=np.zeros_like(pcd.points[mask].astype(np.float32)))

    # Step 1: Project points onto images using multiprocessing
    cam_UV_list = project_points_to_image(np.float32(pcd.points), pcd.colors, cameras)

    # Step 2: Organize points into patches using multiprocessing
    # with multiprocessing.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
    #     patches = pool.map(process_camera_patches, cam_UV_list)
    # patches = {"points": [], "colors": []}
    patches = []


    points = []
    colors = []
    for p in patches:
        colors.extend(p['colors'])
        points.extend((p['points']))
    print(f"Raw Patches: {len(points)}")
    # exclude redundant patches by applying NMS
    points, keep_mask = perform_nms_on_patches(torch.stack(points), iou_threshold=.95)
    print(f"Afte removing overlaps patches: {len(points)}")
    colors = torch.stack(colors)[keep_mask]
    return {'points': points.tolist(), 'colors': colors.tolist()}


def process_camera_patches(cam):
    patches = organize_points_into_patches(cam['cam_UV'])
    return patches


def project_points_to_image(points, colors, camera_params):
    cam_UV_list = []
    args_list = [(points, colors, cam_id, cam) for cam_id, cam in enumerate(camera_params)]
    for arg in args_list:
        cam_UV_list.append(project_points_for_camera(arg))
    return cam_UV_list

def project_points_to_image(points, colors, camera_params):
    cam_UV_list = []
    args_list = [(points, colors, cam_id, cam) for cam_id, cam in enumerate(camera_params)]
    for arg in args_list:
        cam_UV_list.append(project_points_for_camera(arg))
    return cam_UV_list

def project_points_for_camera(args):
    points, colors, cam_id, cam = args
    try:
        K = cam.K
        R = cam.R
        t = cam.T.reshape(3, 1)
        width, height = cam.width, cam.height
    except TypeError as e:
        print(f"Error processing camera {cam_id}: {e}")
        print(cam)

    # Compute projection matrix
    P = K @ np.hstack((R.transpose(), t))  # Shape: (3, 4)
    # pcd_mask = np.zeros((points.shape[0], 1), dtype=bool)
    # Convert points to homogeneous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # Shape: (N, 4)

    # Project points onto image plane
    projections = (P @ points_hom.T).T  # Shape: (N, 3)

    # Convert to pixel coordinates
    u = projections[:, 0] / projections[:, 2]
    v = projections[:, 1] / projections[:, 2]
    depth = projections[:, 2]

    # Create a mask for points that project within the image bounds
    in_front = depth > 0
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    within_image = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    valid = in_front & within_image

    # Initialize cam_UV array with None
    cam_UV = np.full((height, width), None, dtype=object)
    depth_UV = np.full((height, width), 0, dtype=np.float32)
    depth_Nones = np.full((height, width), None, dtype=object)
    depth_max_TH = np.quantile(depth, q=.5, axis=-1)
    pcd_mask = depth < depth_max_TH
    # Assign valid projections to cam_UV
    visibility_mask = np.where(valid)[0]
    for idx in visibility_mask:
        ui = u_int[idx]
        vi = v_int[idx]
        if cam_UV[vi, ui] is None or depth[idx] < cam_UV[vi, ui]['depth'] and depth[idx] < depth_max_TH:
            cam_UV[vi, ui] = {'point': points[idx],
                              'color': colors[idx],
                              'depth': depth[idx]}
            depth_UV[vi, ui] = depth[idx]

    # cam_UV[~depth_max_TH] = None
    # cam_UV = cam_UV[depth_max_TH]

    gt_alpha_mask = (depth < depth_max_TH)
    return {'cam_UV': cam_UV, 'cam_id': cam_id, 'visibility_mask': visibility_mask, "mask": gt_alpha_mask, "pcd_mask": pcd_mask}

def transform_world_to_camera(xyz: torch.Tensor, cam) -> torch.Tensor:
    """
    Transform 3D points from world coordinates to camera coordinates.

    Args:
        xyz (torch.Tensor): Tensor of shape (N, 3) in world coordinates.
        cam: Camera instance with attributes:
             - R: Rotation matrix of shape (3, 3) as a torch.Tensor.
             - T: Translation vector of shape (3,) or (3,1) as a torch.Tensor.

    Returns:
        torch.Tensor: Transformed points of shape (N, 3) in camera coordinates.
    """
    # Ensure T is of shape (3,)
    # T = cam.T.view(3)
    # R = cam.R.to(xyz.device)
    # K = torch.from_numpy(cam.K).to(points.device)
    xyz_cam = (cam.R_ten @ xyz.T).T + cam.T_ten.view(1, 3)
    return xyz_cam
    # Transform: x_cam = R^T * (x_world - T)
    # xyz_cam = (cam.R_ten.t() @ (xyz - cam.T_ten).T).T
    # return xyz_cam

def project_points(points: torch.Tensor, cam) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Projects 3D points (world coordinates) into image space and converts them into
    normalized coordinates in the range [-1, 1] (for use with grid_sample).
    Also returns a boolean mask indicating valid projections (i.e. points in front
    of the camera and within image bounds) and the raw depth values.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        cam: A camera instance with the following attributes:
             - K_ten: Intrinsic matrix of shape (3, 3) as a torch.Tensor.
             - R_ten: Rotation matrix of shape (3, 3) as a torch.Tensor.
             - T_ten: Translation vector of shape (3,) or (3,1) as a torch.Tensor.
             - image_width: Image width in pixels.
             - image_height: Image height in pixels.

    Returns:
        uv_norm (torch.Tensor): Tensor of shape (N, 2) with normalized image coordinates in [-1, 1].
        valid (torch.Tensor): Boolean tensor of shape (N,) indicating valid projections.
        depth (torch.Tensor): Tensor of shape (N,) containing the depth values from the projection.
    """
    # Ensure camera parameters are on the same device.
    K = cam.K_ten
    R = cam.R_ten
    t = cam.T_ten.view(3, 1)
    width, height = cam.image_width, cam.image_height

    # Compute the extrinsic matrix using the convention: world-to-camera: x_cam = R^T*(x - T)
    extrinsic = torch.cat((R.t(), t), dim=1)  # shape: (3,4)
    P = K @ extrinsic  # shape: (3,4)

    # Convert points to homogeneous coordinates.
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
    points_hom = torch.cat((points, ones), dim=1)  # shape: (N, 4)

    # Project points: (N,3)
    projections = (P @ points_hom.t()).t()
    # Compute pixel coordinates.
    u = projections[:, 0] / projections[:, 2]
    v = projections[:, 1] / projections[:, 2]
    depth = projections[:, 2]

    # Determine which points are in front of the camera.
    in_front = depth > 0

    # Round u and v to integer pixel indices.
    u_int = torch.round(u).to(torch.int64)
    v_int = torch.round(v).to(torch.int64)

    # Create a valid mask: u_int in [0,width) and v_int in [0,height)
    within_image = (u_int > 0) & (u_int < width) & (v_int > 0) & (v_int < height)
    valid = in_front & within_image

    # Convert pixel coordinates to normalized coordinates:
    # normalized x: map [0, width-1] to [-1, 1]
    # normalized y: map [0, height-1] to [-1, 1]
    x_norm = (u / (width)) * 2 - 1
    y_norm = (v / (height)) * 2 - 1
    uv_norm = torch.stack([x_norm, y_norm], dim=1)  # shape: (N, 2)

    return uv_norm, valid, depth

def project_points1(points: torch.Tensor, cam) -> torch.Tensor:
    """
    Projects 3D points into image space and converts them into normalized coordinates
    for use with grid_sample. The normalization maps pixel coordinates to the range [-1, 1].

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points (on GPU).
        cam: A camera instance with the following attributes:
             - K: Intrinsic matrix of shape (3, 3) as a torch.Tensor.
             - R: Rotation matrix of shape (3, 3) as a torch.Tensor.
             - T: Translation vector of shape (3,) or (3, 1) as a torch.Tensor.
             - image_width: Image width in pixels.
             - image_height: Image height in pixels.

    Returns:
        torch.Tensor: Normalized image coordinates of shape (N, 2) in the range [-1, 1].
    """
    # Ensure camera parameters are on the same device as points.
    K = cam.K_ten
    R = cam.R_ten
    t = cam.T_ten.view(3, 1)
    width, height = cam.image_width, cam.image_height

    # Form the extrinsic matrix [R^T | t]. Note: we assume the camera extrinsics are such that
    # the world-to-camera rotation is R^T.
    extrinsic = torch.cat((R.t(), t), dim=1)  # shape: (3, 4)
    P = K @ extrinsic  # shape: (3, 4)

    # Convert points to homogeneous coordinates.
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
    points_hom = torch.cat((points, ones), dim=1)  # shape: (N, 4)

    # Project points onto the image plane.
    projections = (P @ points_hom.t()).t()  # shape: (N, 3)

    # Convert to pixel coordinates.
    u = projections[:, 0] / projections[:, 2]
    v = projections[:, 1] / projections[:, 2]
    depth = projections[:, 2]

    in_front = depth > 0
    # u_int = torch.round(u).astype(int)
    # v_int = torch.round(v).astype(int)
    within_image = (u >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    valid = in_front & within_image
    # Normalize pixel coordinates to [-1, 1] for grid_sample.
    # Pixel coordinate range: u in [0, width-1], v in [0, height-1].
    x_norm = (u / (width - 1)) * 2 - 1
    y_norm = (v / (height - 1)) * 2 - 1

    uv_norm = torch.stack([x_norm, y_norm], dim=1)  # shape: (N, 2)
    return uv_norm

def organize_points_into_patches1(cam_UV, cell_size_u=32, cell_size_v=32, patch_size=(4, 4),
                                 depth_gradient_threshold=0.045):
    """
    Organize points from cam_UV into patches with depth consistency checks using partial derivatives.
    Subsequent patches are initialized with the last row and/or column of control points from the previous patch
    to 'stitch' them together.

    :param cam_UV: 2D array of shape (height, width) containing dicts with 'point', 'color', 'depth'.
    :param cell_size_u: Cell size in the U direction (pixels).
    :param cell_size_v: Cell size in the V direction (pixels).
    :param patch_size: Tuple indicating the size of each patch (height, width).
    :param depth_gradient_threshold: Maximum allowed depth gradient (partial derivative) within a patch.
    :return: Dictionary containing lists of control point patches and color patches.
    """
    # Step 1: Extract valid points and their image coordinates
    height, width = cam_UV.shape
    valid_pixels = np.array([(i, j) for i in range(height) for j in range(width) if cam_UV[i, j] is not None])
    if len(valid_pixels) == 0:
        print("No valid points found in cam_UV.")
        return {'points': [], 'colors': [], 'patch_grid': None}

    valid_points = np.array([cam_UV[i, j] for i, j in valid_pixels])

    # Step 2: Define grid dimensions
    num_cells_v = int(np.ceil(height / cell_size_v))
    num_cells_u = int(np.ceil(width / cell_size_u))

    # Initialize the grid
    grid = [[[] for _ in range(num_cells_u)] for _ in range(num_cells_v)]

    # Assign points to grid cells
    for idx in range(len(valid_points)):
        v, u = valid_pixels[idx]
        iu = int(u // cell_size_u)
        iv = int(v // cell_size_v)
        # Clamp indices
        iu = min(max(iu, 0), num_cells_u - 1)
        iv = min(max(iv, 0), num_cells_v - 1)
        grid[iv][iu].append(valid_points[idx])

    # Step 3: Select representative points for each grid cell
    downsampled_points = [[None for _ in range(num_cells_u)] for _ in range(num_cells_v)]
    for iv in range(num_cells_v):
        for iu in range(num_cells_u):
            cell_points = grid[iv][iu]
            if len(cell_points) == 0:
                continue
            # Select the point with median depth to avoid outliers
            depths = [pt['depth'] for pt in cell_points]
            median_idx = np.argsort(depths)[len(depths) // 2]
            downsampled_points[iv][iu] = cell_points[median_idx]

    # Step 4: Extract patches and organize them in a grid
    patches = []
    patch_height, patch_width = patch_size
    patch_grid_height = num_cells_v - (patch_height - 1)
    patch_grid_width = num_cells_u - (patch_width - 1)
    patch_grid = [[None for _ in range(patch_grid_width)] for _ in range(patch_grid_height)]

    prev_v_patch = None
    for iv in range(patch_grid_height):
        for iu in range(patch_grid_width):
            fully_occupied = True
            patch = np.empty((patch_height, patch_width), dtype=object)

            # Initialize the patch with points from downsampled_points
            for dv in range(patch_height):
                for du in range(patch_width):
                    point = downsampled_points[iv + dv][iu + du]
                    if point is None:
                        fully_occupied = False
                        break
                    patch[dv, du] = point
                if not fully_occupied:
                    break

            if fully_occupied:
                # Adjust the patch to include control points from previous patches
                # Stitching with the left neighbor
                if iu > 0 and patch_grid[iv][iu - 1] is not None:
                    left_patch = patch_grid[iv][iu - 1]
                    # Copy last column from left_patch to first column of current patch
                    patch[:, 0] = left_patch[:, -1]
                # Stitching with the upper neighbor
                if iv > 0 and patch_grid[iv - 1][iu] is not None:
                    above_patch = patch_grid[iv - 1][iu]
                    # Copy last row from above_patch to first row of current patch
                    patch[0, :] = above_patch[-1, :]

                # Recompute depth_grid after stitching
                depth_grid = np.zeros((patch_height, patch_width))
                for dv in range(patch_height):
                    for du in range(patch_width):
                        point = patch[dv, du]
                        depth_grid[dv, du] = point['depth']

                # Compute partial derivatives (finite differences)
                depth_gradients_u = np.zeros_like(depth_grid)
                depth_gradients_v = np.zeros_like(depth_grid)
                for dv in range(patch_height):
                    for du in range(patch_width):
                        # U direction (horizontal)
                        if 0 < du < patch_width - 1:
                            depth_gradients_u[dv, du] = (depth_grid[dv, du + 1] - depth_grid[dv, du - 1]) / 2
                        elif du == 0 and patch_width > 1:
                            depth_gradients_u[dv, du] = depth_grid[dv, du + 1] - depth_grid[dv, du]
                        elif du == patch_width - 1 and patch_width > 1:
                            depth_gradients_u[dv, du] = depth_grid[dv, du] - depth_grid[dv, du - 1]

                        # V direction (vertical)
                        if 0 < dv < patch_height - 1:
                            depth_gradients_v[dv, du] = (depth_grid[dv + 1, du] - depth_grid[dv - 1, du]) / 2
                        elif dv == 0 and patch_height > 1:
                            depth_gradients_v[dv, du] = depth_grid[dv + 1, du] - depth_grid[dv, du]
                        elif dv == patch_height - 1 and patch_height > 1:
                            depth_gradients_v[dv, du] = depth_grid[dv, du] - depth_grid[dv - 1, du]

                # Check if gradients are within the threshold
                gradients_ok = np.all(np.abs(depth_gradients_u) <= depth_gradient_threshold) and \
                               np.all(np.abs(depth_gradients_v) <= depth_gradient_threshold)

                if gradients_ok:
                    # Store the patch in the grid
                    patch_grid[iv][iu] = patch
                    patches.append((iv, iu, patch))

    # Extract control points and colors from patches
    control_pts_patches = []
    color_patches = []
    for iv, iu, patch in patches:
        pp = []
        cc = []
        for row in patch:
            for pt in row:
                pp.append(pt['point'].tolist())
                cc.append(pt['color'].tolist())

        control_pts_patches.append(torch.tensor(pp, dtype=torch.float32).reshape(patch_height, patch_width, 3))
        color_patches.append(torch.tensor(cc, dtype=torch.float32).reshape(patch_height, patch_width, 3))

    return {'points': control_pts_patches, 'colors': color_patches}
def organize_points_into_patches(cam_UV, cell_size_u=24, cell_size_v=24, patch_size=(4, 4),
                                 depth_gradient_threshold=0.1):
    """
    Organize points from cam_UV into patches with depth consistency checks using partial derivatives.
    Instead of relying on immediate neighbors, track the last successful patch in each row and column
    to perform stitching.

    :param cam_UV: 2D array of shape (height, width) containing dicts with 'point', 'color', 'depth'.
    :param cell_size_u: Cell size in the U direction (pixels).
    :param cell_size_v: Cell size in the V direction (pixels).
    :param patch_size: Tuple indicating the size of each patch (height, width).
    :param depth_gradient_threshold: Maximum allowed depth gradient (partial derivative) within a patch.
    :return: Dictionary containing lists of control point patches and color patches.
    """
    height, width = cam_UV.shape
    valid_pixels = np.array([(i, j) for i in range(height) for j in range(width) if cam_UV[i, j] is not None])
    if len(valid_pixels) == 0:
        print("No valid points found in cam_UV.")
        return {'points': [], 'colors': [], 'patch_grid': None}

    valid_points = np.array([cam_UV[i, j] for i, j in valid_pixels])

    # Determine grid dimensions
    num_cells_u = int(np.ceil(width / cell_size_u))
    num_cells_v = int(np.ceil(height / cell_size_v))

    # Initialize the grid
    grid = [[[] for _ in range(num_cells_u)] for _ in range(num_cells_v)]

    # Assign points to grid cells
    for idx in range(len(valid_points)):
        v, u = valid_pixels[idx]
        iu = int(u // cell_size_u)
        iv = int(v // cell_size_v)
        iu = min(max(iu, 0), num_cells_u - 1)
        iv = min(max(iv, 0), num_cells_v - 1)
        grid[iv][iu].append(valid_points[idx])

    # Create downsampled_points from the grid
    downsampled_points = [[None for _ in range(num_cells_u)] for _ in range(num_cells_v)]
    for iv in range(num_cells_v):
        for iu in range(num_cells_u):
            cell_points = grid[iv][iu]
            if len(cell_points) == 0:
                continue
            depths = [pt['depth'] for pt in cell_points]
            median_idx = np.argsort(depths)[len(depths) // 2]
            downsampled_points[iv][iu] = cell_points[median_idx]

    patch_height, patch_width = patch_size
    patch_grid_height = num_cells_v - (patch_height - 1)
    patch_grid_width = num_cells_u - (patch_width - 1)

    patches = []
    # Keep track of the last successful patch in each row and column
    last_patch_in_row = [None] * patch_grid_height
    last_patch_in_col = [None] * patch_grid_width

    for iv in range(patch_grid_height):
        for iu in range(patch_grid_width):
            fully_occupied = True
            patch = np.empty((patch_height, patch_width), dtype=object)

            # Fill the patch grid
            for dv in range(patch_height):
                for du in range(patch_width):
                    global_iv = iv + dv
                    global_iu = iu + du
                    point = downsampled_points[global_iv][global_iu]
                    if point is None:
                        fully_occupied = False
                        break
                    patch[dv, du] = point
                if not fully_occupied:
                    break

            if not fully_occupied:
                # Not a fully occupied patch
                continue

            # Attempt to stitch with last patch in the same row
            if last_patch_in_row[iv] is not None:
                # Stitch horizontally by aligning the left column of current patch with the right column of last row patch
                left_patch = last_patch_in_row[iv]
                # Ensure they have compatible shapes
                if left_patch.shape[0] == patch_height:
                    # Replace first column of current patch with the last column of the left patch
                    patch[:, 0] = left_patch[:, -1]

            # Attempt to stitch with last patch in the same column
            if last_patch_in_col[iu] is not None:
                # Stitch vertically by aligning the top row of the current patch with bottom row of last column patch
                above_patch = last_patch_in_col[iu]
                if above_patch.shape[1] == patch_width:
                    # Replace the top row of current patch with the bottom row of the above patch
                    patch[0, :] = above_patch[-1, :]

            # Recompute depth_grid after potential stitching
            depth_grid = np.zeros((patch_height, patch_width))
            for dv in range(patch_height):
                for du in range(patch_width):
                    point = patch[dv, du]
                    depth_grid[dv, du] = point['depth']

            # Compute partial derivatives (finite differences)
            depth_gradients_u = np.zeros_like(depth_grid)
            depth_gradients_v = np.zeros_like(depth_grid)
            for dv in range(patch_height):
                for du in range(patch_width):
                    # U direction (horizontal)
                    if 0 < du < patch_width - 1:
                        depth_gradients_u[dv, du] = (depth_grid[dv, du + 1] - depth_grid[dv, du - 1]) / 2
                    elif du == 0 and patch_width > 1:
                        depth_gradients_u[dv, du] = depth_grid[dv, du + 1] - depth_grid[dv, du]
                    elif du == patch_width - 1 and patch_width > 1:
                        depth_gradients_u[dv, du] = depth_grid[dv, du] - depth_grid[dv, du - 1]

                    # V direction (vertical)
                    if 0 < dv < patch_height - 1:
                        depth_gradients_v[dv, du] = (depth_grid[dv + 1, du] - depth_grid[dv - 1, du]) / 2
                    elif dv == 0 and patch_height > 1:
                        depth_gradients_v[dv, du] = depth_grid[dv + 1, du] - depth_grid[dv, du]
                    elif dv == patch_height - 1 and patch_height > 1:
                        depth_gradients_v[dv, du] = depth_grid[dv, du] - depth_grid[dv - 1, du]

            # Check if gradients are within the threshold
            gradients_ok = (np.abs(depth_gradients_u) <= depth_gradient_threshold).all() and \
                           (np.abs(depth_gradients_v) <= depth_gradient_threshold).all()

            if gradients_ok:
                # Update last_patch_in_row and last_patch_in_col
                last_patch_in_row[iv] = patch
                last_patch_in_col[iu] = patch
                patches.append((iv, iu, patch))

    # Extract control points and colors from patches
    control_pts_patches = []
    color_patches = []
    for iv, iu, patch in patches:
        pp = []
        cc = []
        for row in patch:
            for pt in row:
                pp.append(pt['point'].tolist())
                cc.append(pt['color'].tolist())
        control_pts_patches.append(torch.tensor(pp, dtype=torch.float32).reshape(patch_height, patch_width, 3))
        color_patches.append(torch.tensor(cc, dtype=torch.float32).reshape(patch_height, patch_width, 3))

    return {'points': control_pts_patches, 'colors': color_patches}

class QuadNode:
    def __init__(self, x_min, y_min, x_max, y_max, points):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.points = points  # List of points (or references to valid_points)
        self.children = []
        self.is_leaf = True

def adaptive_partition(width, height, valid_pixels, valid_points, max_points_per_cell=50, max_depth=5):
    """
    Adaptive partitioning of the image plane using a quadtree approach.

    Args:
        width (int): Image width.
        height (int): Image height.
        valid_pixels (np.ndarray): Array of shape (N, 2) with (v, u) pixel coords for each point.
        valid_points (np.ndarray): Array of shape (N,) containing point dicts or references.
        max_points_per_cell (int): Threshold for subdividing a cell.
        max_depth (int): Maximum subdivision depth.

    Returns:
        root (QuadNode): Root node of the quadtree.
    """
    # Initial cell covers entire image
    root = QuadNode(0, 0, width, height, list(range(len(valid_points))))

    subdivide_cell(root, valid_pixels, valid_points, max_points_per_cell, max_depth)
    return root

def subdivide_cell(node, valid_pixels, valid_points, max_points_per_cell, max_depth, depth=0):
    if depth >= max_depth:
        return

    if len(node.points) <= max_points_per_cell:
        return

    # Subdivide into 4 quadrants
    x_mid = (node.x_min + node.x_max) // 2
    y_mid = (node.y_min + node.y_max) // 2

    # Child cells: top-left, top-right, bottom-left, bottom-right
    child_regions = [
        (node.x_min, node.y_min, x_mid, y_mid),       # top-left
        (x_mid, node.y_min, node.x_max, y_mid),       # top-right
        (node.x_min, y_mid, x_mid, node.y_max),       # bottom-left
        (x_mid, y_mid, node.x_max, node.y_max)        # bottom-right
    ]

    child_points_indices = [[] for _ in range(4)]

    # Assign points to children
    for idx in node.points:
        v, u = valid_pixels[idx]
        # Determine which child cell this point belongs to
        for ci, (xmin, ymin, xmax, ymax) in enumerate(child_regions):
            if (xmin <= u < xmax) and (ymin <= v < ymax):
                child_points_indices[ci].append(idx)
                break

    # If subdivision does not distribute points well, we might abort here
    # Check if any child is empty or all children fail to reduce complexity
    if all(len(cp) == len(node.points) for cp in child_points_indices):
        # No effective subdivision
        return

    # Create child nodes
    node.children = []
    for (xmin, ymin, xmax, ymax), cpts in zip(child_regions, child_points_indices):
        child_node = QuadNode(xmin, ymin, xmax, ymax, cpts)
        node.children.append(child_node)

    if any(len(c.child_points) > max_points_per_cell for c in node.children):
        # Recursively subdivide children
        for child in node.children:
            subdivide_cell(child, valid_pixels, valid_points, max_points_per_cell, max_depth, depth+1)

    node.is_leaf = False

def extract_patches_from_quadtree(root, valid_pixels, valid_points, patch_size=(4,4)):
    """
    Once the quadtree is built, each leaf node represents a cell with a manageable number of points.
    Use these cells to form patches.

    Args:
        root (QuadNode): Root of the quadtree.
        valid_pixels (np.ndarray): Pixel coordinates of points.
        valid_points (np.ndarray): Point data array.
        patch_size (tuple): Desired patch size for NURBS fitting.

    Returns:
        patches (list): A list of patches (control points).
    """
    leaves = get_leaves(root)
    patches = []
    for leaf in leaves:
        # If there are enough points to form a patch
        # Implement logic to select patch_height*patch_width points from leaf.points
        # Possibly interpolate or do local parameterization
        if len(leaf.points) >= patch_size[0]*patch_size[1]:
            # Extract a patch from these points
            patch_points, patch_colors = create_patch(leaf, valid_pixels, valid_points, patch_size)
            patches.append({'points': patch_points, 'colors': patch_colors})
    return patches

def get_leaves(node):
    if node.is_leaf:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(get_leaves(child))
    return leaves

def create_patch(leaf, valid_pixels, valid_points, patch_size):
    # Custom logic to organize leaf.points into patch_size grid
    # For simplicity, pick a subset of points and arrange them in a patch
    # In a real scenario, you'd do a local parameterization and selection
    patch_height, patch_width = patch_size
    selected_indices = leaf.points[:patch_height*patch_width]
    # Sort them by some criteria, e.g. pixel coordinates, to form a consistent patch
    # Sort by v, then u for instance
    pts_info = [(valid_pixels[i], valid_points[i]) for i in selected_indices]
    pts_info.sort(key=lambda x: (x[0][0], x[0][1]))  # sort by v then u
    # Reshape into patch format
    patch_points = []
    patch_colors = []
    for i in range(patch_height):
        row = pts_info[i*patch_width:(i+1)*patch_width]
        p_row = [r[1]['point'] for r in row]  # Extract 3D points
        c_row = [r[1]['color'] for r in row]  # Extract colors
        patch_points.append(p_row)
        patch_colors.append(c_row)

    return patch_points, patch_colors


def load_nurbs_patches_from_3dm(file_path="/ceph/hpc/home/eutzlilo/2dsplines/output/scan24/patches.3dm"):
    """
    Loads NURBS surfaces from a 3DM file and returns them as a tensor of shape (N, 4, 4, 3).
    Each surface must be degree 3 in both directions and have a 4x4 control point grid.

    Args:
        file_path (str): Path to the .3dm file.

    Returns:
        torch.Tensor: A tensor of shape (N, 4, 4, 3) containing the control points of the NURBS patches.
    """
    import rhino3dm
    model = rhino3dm.File3dm.Read(file_path)
    nurbs_patches = []

    # Iterate over objects in the .3dm file
    for obj in model.Objects:
        geo = obj.Geometry
        # Geo can be a Brep, Surface, Curve, etc.
        # If it's a Brep, we can extract its faces (each face has a surface)
        if isinstance(geo, rhino3dm.Brep):
            for f_i in range(len(geo.Faces)):
                face = geo.Faces[f_i]
                # face is a BrepFace, now you can call ToNurbsSurface()
                srf = face.ToNurbsSurface()
                if srf is not None:
                    patch = extract_4x4_patch(srf)
            # Extract surfaces from Brep faces
            # for face in geo.Faces:
            #     srf = face.ToNurbsSurface()
            #     if srf is not None:
            #         patch = extract_4x4_patch(srf)
            #         if patch is not None:
            #             nurbs_patches.append(patch)
        elif isinstance(geo, rhino3dm.NurbsSurface):
            # Directly a NURBS surface
            srf = geo
            patch = extract_4x4_patch(srf)
            if patch is not None:
                nurbs_patches.append(patch)

    if len(nurbs_patches) == 0:
        print("No suitable 4x4-degree-3 NURBS surfaces found in the file.")
        return None

    # Convert to torch tensor
    patches_tensor = torch.tensor(nurbs_patches, dtype=torch.float32)
    return patches_tensor


import rhino3dm
import numpy as np


# def extract_patches_from_nurbs_surface(srf):
#     """
#     Given a NURBS surface with multiple spans, extract each 4x4 patch.
#
#     Args:
#         srf (rhino3dm.NurbsSurface): The full multi-span NURBS surface.
#
#     Returns:
#         list of np.ndarray: A list of arrays, each of shape (4,4,3) representing one patch's control points.
#     """
#     # Extract degrees
#     deg_u = srf.Degree(0)
#     deg_v = srf.Degree(1)
#
#     # Confirm the surface is cubic in both directions
#     if deg_u != 3 or deg_v != 3:
#         print("Surface is not cubic in one or both directions. Cannot directly extract 4x4 patches.")
#         return []
#
#     # Get knot vectors
#     knots_u = srf.KnotsU
#     knots_v = srf.KnotsV
#
#     num_knots_u = len(knots_u)
#     num_knots_v = len(knots_v)
#
#     # For a cubic surface, each patch is defined between consecutive knots
#     # number_of_spans_u = num_knots_u - (deg_u+1)
#     # number_of_spans_v = num_knots_v - (deg_v+1)
#     # Each span in U corresponds to (u[i], u[i+1]) interval
#     # Each span in V corresponds to (v[j], v[j+1]) interval
#
#     patches = []
#
#     for i in range(deg_u, num_knots_u - deg_u):
#         for j in range(deg_v, num_knots_v - deg_v):
#             # Define intervals
#             u0, u1 = knots_u[i], knots_u[i + 1]
#             v0, v1 = knots_v[j], knots_v[j + 1]
#
#             # Extract the sub-surface representing this patch
#             sub_srf = srf.Trim(rhino3dm.Interval(u0, u1), rhino3dm.Interval(v0, v1))
#             if sub_srf is None:
#                 continue
#
#             # Convert to NURBS again to ensure a proper patch
#             sub_ns = sub_srf.ToNurbsSurface()
#             if sub_ns is None:
#                 continue
#
#             # Check if sub_ns is a single 4x4 patch
#             cpts_u = sub_ns.Points.CountU
#             cpts_v = sub_ns.Points.CountV
#
#             if sub_ns.Degree(0) == 3 and sub_ns.Degree(1) == 3 and cpts_u == 4 and cpts_v == 4:
#                 patch_points = extract_4x4_patch(sub_ns)
#                 if patch_points is not None:
#                     patches.append(patch_points)
#
#     return patches


# def extract_4x4_patch(srf):
#     """
#     Extract a 4x4x3 array of control points from a 4x4 NURBS patch.
#     Assuming srf is degree=3 in u and v, with 4x4 control points.
#     """
#     deg_u = srf.Degree(0)
#     deg_v = srf.Degree(1)
#     cpts_u = srf.Points.CountU
#     cpts_v = srf.Points.CountV
#
#     if deg_u == 3 and deg_v == 3 and cpts_u == 4 and cpts_v == 4:
#         patch_points = np.zeros((4, 4, 3), dtype=float)
#         for i in range(cpts_u):
#             for j in range(cpts_v):
#                 cp = srf.Points.GetControlPoint(i, j)
#                 patch_points[i, j, 0] = cp.X
#                 patch_points[i, j, 1] = cp.Y
#                 patch_points[i, j, 2] = cp.Z
#         return patch_points
#     return None

def extract_4x4_patch_points(srf, u_idx, v_idx):
    """
    Extract a 4x4 patch of control points from a multi-span NURBS surface for a given span index (u_idx, v_idx).
    Args:
        srf (rhino3dm.NurbsSurface): The full multi-span NURBS surface.
        u_idx (int): Index of the span in U direction.
        v_idx (int): Index of the span in V direction.

    Returns:
        np.ndarray or None: (4,4,3) array of control points if patch is valid, otherwise None.
    """
    deg_u = srf.Degree(0)
    deg_v = srf.Degree(1)

    if deg_u != 3 or deg_v != 3:
        return None

    cpts_u = srf.Points.CountU
    cpts_v = srf.Points.CountV

    # For degree=3 surfaces, each span corresponds to 4 consecutive control points
    # u_idx, v_idx correspond to the span indices in each direction
    # The control points for a particular patch are:
    # [u_idx : u_idx+4, v_idx : v_idx+4]
    # Validate indices
    if u_idx + 4 > cpts_u or v_idx + 4 > cpts_v:
        return None

    patch_points = np.zeros((4,4,3), dtype=float)
    for i in range(4):
        for j in range(4):
            cp = srf.Points.GetControlPoint(u_idx + i, v_idx + j)
            # cp has X, Y, Z, and W if rational. For now we ignore W, or assume W=1
            patch_points[i, j, 0] = cp.X
            patch_points[i, j, 1] = cp.Y
            patch_points[i, j, 2] = cp.Z

    return patch_points


def extract_patches_from_nurbs_surface(srf):
    """
    Given a NURBS surface with multiple spans, extract each 4x4 patch directly
    by indexing control points, without using Trim or ToNurbsSurface on sub-patches.

    Args:
        srf (rhino3dm.NurbsSurface): The full multi-span NURBS surface.

    Returns:
        list of np.ndarray: A list of arrays, each shape (4,4,3) representing one patch's control points.
    """
    deg_u = srf.Degree(0)
    deg_v = srf.Degree(1)

    # Confirm cubic in both directions
    if deg_u != 3 or deg_v != 3:
        print("Surface is not cubic in both directions.")
        return []

    knots_u = srf.KnotsU
    knots_v = srf.KnotsV

    num_knots_u = len(knots_u)
    num_knots_v = len(knots_v)

    # Number of spans in each direction
    # For a degree 3 surface, each span is defined by a pair of knots
    # The number of spans = (len(knot_vector) - (degree+1))
    num_spans_u = num_knots_u - (deg_u + 1)
    num_spans_v = num_knots_v - (deg_v + 1)

    # Check control points count
    cpts_u = srf.Points.CountU
    cpts_v = srf.Points.CountV
    # Ensure we have enough control points for these spans
    # Typically, control points = degree+num_spans in each direction
    # For degree=3, control_points_u = 3 + num_spans_u = num_spans_u + 3
    # Just a sanity check, not mandatory
    if cpts_u != num_spans_u + deg_u or cpts_v != num_spans_v + deg_v:
        print("Control points count does not match expected spans. Surface might be non-standard.")
        # Still proceed, but be cautious.

    patches = []
    # Each patch corresponds to a span combination
    # u_idx ranges from 0 to num_spans_u-1
    # v_idx ranges from 0 to num_spans_v-1
    for u_idx in range(num_spans_u):
        for v_idx in range(num_spans_v):
            patch_points = extract_4x4_patch_points(srf, u_idx, v_idx)
            if patch_points is not None:
                patches.append(patch_points)

    return patches

# @memory.cache
# @memory.cache
def load_3dm_nurbs_patches(file_path):
    """
    Loads a 3DM file, extracts NURBS surfaces, and returns all 4x4 patches.
    """
    model = rhino3dm.File3dm.Read(file_path)
    if model is None:
        print(f"Failed to read {file_path}")
        return None

    all_patches = []
    for obj in model.Objects:
        geo = obj.Geometry
        # Check if geometry is a Brep or NurbsSurface
        if isinstance(geo, rhino3dm.Brep):
            num_faces = len(geo.Faces)
            # Extract surfaces from Brep faces
            for f_i in range(num_faces):
                face = geo.Faces[f_i]
                srf = face.ToNurbsSurface()
                if srf is not None:
                    patches = extract_patches_from_nurbs_surface(srf)
                    all_patches.extend(patches)
        elif isinstance(geo, rhino3dm.NurbsSurface):
            # Directly handle a multi-span NURBS surface
            patches = extract_patches_from_nurbs_surface(geo)
            all_patches.extend(patches)
        else:
            # Not a Brep or NurbsSurface
            continue

    if len(all_patches) == 0:
        print("No suitable patches found.")
        return None

    # Convert to torch tensor if needed
    # import torch
    patches_tensor = torch.tensor(np.asarray(all_patches), dtype=torch.float32, device='cuda')
    # patches_tensor = (patches_tensor - patches_tensor.min()) / (patches_tensor.max() - patches_tensor.min())
    return patches_tensor


# -----------------------------------------------------------------------------
# NEW: load_3dm_nurbs_grids ----------------------------------------------------
import rhino3dm as r3d
import numpy as np
import torch
from scipy.interpolate import bisplrep, bisplev

#
# def make_clamped_uniform_knots(num_ctrl, degree):
#     """Generate clamped uniform knots, matching Spline class."""
#     knots = np.zeros(num_ctrl + degree + 1)
#     knots[degree:-degree] = np.linspace(0, 1, num_ctrl - degree + 1)
#     knots[-degree:] = 1.0
#     return knots


import rhino3dm as r3d
import numpy as np
import torch
from scipy.interpolate import bisplrep, bisplev


# def make_clamped_uniform_knots(num_ctrl, degree):
#     """Generate clamped uniform knots, matching Spline class."""
#     knots = np.zeros(num_ctrl + degree + 1)
#     knots[degree:-degree] = np.linspace(0, 1, num_ctrl - degree + 1)
#     knots[-degree:] = 1.0
#     return knots

def make_clamped_uniform_knots(num_ctrl, degree):
    """Generate clamped uniform knots for a B-Spline."""
    knots = np.zeros(num_ctrl + degree + 1)
    num_internal = num_ctrl - degree
    if num_internal > 0:
        knots[degree:degree + num_internal] = np.linspace(0, 1, num_internal)
    knots[degree + num_internal:] = 1.0
    return knots
import rhino3dm as r3d
import numpy as np
import torch
from scipy.interpolate import bisplrep, bisplev

def make_clamped_uniform_knots(num_ctrl, degree):
    """Generate clamped uniform knots for a B-Spline."""
    knots = np.zeros(num_ctrl + degree + 1)
    num_internal = num_ctrl - degree
    if num_internal > 0:
        knots[degree:degree + num_internal] = np.linspace(0, 1, num_internal)
    knots[degree + num_internal:] = 1.0
    return knots
import rhino3dm as r3d
import numpy as np
import torch
from scipy.interpolate import bisplrep, bisplev

def make_clamped_uniform_knots(num_ctrl, degree):
    """Generate clamped uniform knots for a B-Spline."""
    knots = np.zeros(num_ctrl + degree + 1)
    num_internal = num_ctrl - degree
    if num_internal > 0:
        knots[degree:degree + num_internal] = np.linspace(0, 1, num_internal)
    knots[degree + num_internal:] = 1.0
    return knots


import numpy as np
import torch
from typing import Dict, Callable
from geomdl import fitting
from sklearn.decomposition import PCA
from geomdl.BSpline import Surface as BSplineSurface
def voxel_sort_points(points_np: np.ndarray, size_u: int, size_v: int,voxel_size: float = 0.001) -> list:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # Extract voxel data
    voxels = np.asarray(voxel_grid.get_voxels())
    if len(voxels) == 0:
        raise ValueError("No voxels generated; adjust voxel_size or check PCD density")

    grid_indices = np.array([v.grid_index for v in voxels])  # [N, 3]

    # Compute centers vectorized
    origin = np.asarray(voxel_grid.origin)  # [3]
    voxel_size_arr = np.full(3, voxel_size) if isinstance(voxel_size, (int, float)) else np.asarray(voxel_size)
    centers = origin + (grid_indices + 0.5) * voxel_size_arr  # [N, 3]

    # Sort by indices (z-major, then v=y, u=x)
    sort_key = grid_indices[:, 2] * (size_u * size_v) + grid_indices[:, 1] * size_u + grid_indices[:, 0]
    sorted_indices = np.argsort(sort_key)
    sorted_centers = centers[sorted_indices]  # [N, 3]



    return sorted_centers[:size_u * size_v].tolist()  # Flat list


import torch


def estimate_depth(points: torch.Tensor, cam: Camera, use_intrinsics: bool = True) -> torch.Tensor:
    """
    Estimates depth for each point relative to the camera.

    Args:
        points: [N, 3] tensor of world-space 3D points.
        cam: Instance of your Camera class, with attributes R ([3,3] rotation), T ([3] or [3,1] translation),
             and optionally get_k() for [3,3] intrinsics.
        use_intrinsics: If True, performs full projection and returns normalized depths (for 2D mapping).

    Returns:
        [N] tensor of depths (z in camera space, masked if invalid).
    """
    device = cam.data_device
    if isinstance(points, np.ndarray):
        points = torch.tensor(points.tolist(), device=device).float()
    # R = torch.tensor(cam.R.tolist(), device=device).float()  # [3, 3]
    # T = torch.tensor(cam.T.tolist(), device=device).float().view(3, 1) if cam.T.ndim == 1 else torch.tensor(cam.T.tolist(), device=device).float()  # [3, 1]
    pts_in_cam = points @ cam.world_view_transform[:3,:3] + cam.world_view_transform[3,:3]
    depths = pts_in_cam[:, 2]
    # Homogenize points: [N, 3] -> [4, N]
    # Homogenize points: [N, 3] -> [4, N]
    points_homo = torch.cat([points.T, torch.ones(1, points.shape[0], device=device)], dim=0)  # [4, N]

    # World to camera: cam_points = R @ P_world + T
    # cam_points = R @ points_homo[:3] + T  # [3, N]

    # Depths as camera-space z

    # Mask invalid (behind or near-zero)
    depths = torch.where(depths > 1e-6, depths, torch.tensor(0.0, device=device))

    if use_intrinsics:
        K = cam.get_k().to(device)  # [3, 3]
        proj = pts_in_cam @ K   # [3, N]
        proj[:, 2] = torch.where(torch.abs(proj[:, 2]) < 1e-6, torch.tensor(1e-6, device=device), proj[:, 2])

        # proj /= proj[2:3]  # Normalize for (u,v)

        return torch.stack([proj[:, 0], proj[:, 1], depths], dim=1)  # [N, 3]: u, v, depth
    # points_homo = torch.cat([points.T, torch.ones(1, points.shape[0], device=device)], dim=0)  # [4, N]
    #
    # # World to camera: cam_points = R @ P_world + T (broadcast T)
    # cam_points = R @ points_homo[:3] + T  # [3, N]
    #
    # if use_intrinsics:
    #     K = cam.get_k().to(device)  # [3, 3]
    #     proj = K @ cam_points  # [3, N]
    #     proj[2] = torch.where(torch.abs(proj[2]) < 1e-6, torch.tensor(1e-6, device=device), proj[2])
    #     depths = proj[2] / (proj[2] + 1e-6)  # Normalized (for perspective)
    # else:
    #     depths = cam_points[2]  # Raw camera z

    # Mask invalid (behind or zero)

    # return depths

def project_and_check_consistency(control_points: np.ndarray, cameras: List[Camera], threshold: float = 0.1,
                                  device='cpu') -> np.ndarray:
    valid_mask = np.zeros(len(control_points), dtype=bool)

    # for _ in range(5):
    for cam in cameras:
        # cam = cameras.pop(random.randint(0, len(cameras) - 1))
        depth_map = getattr(cam, 'depth_map', None)
        if depth_map is None:
            proj_uvd = estimate_depth(torch.tensor(control_points, device=cam.data_device, dtype=torch.float32), cam, use_intrinsics=True)

        proj_uvd = proj_uvd.cpu().numpy()
        u, v, depth = proj_uvd[:, 0], proj_uvd[:, 1], proj_uvd[:, 2]

        uu, vv = np.meshgrid(np.arange(cam.image_width), np.arange(cam.image_height))
        target_points = np.column_stack((uu.ravel(), vv.ravel()))  # [H*W, 2]

        # Source: Projected (u,v,depth)
        source_uv = np.column_stack((u, v))  # [M, 2]
        depth_map = griddata(source_uv, depth, target_points, method='linear', fill_value='inf')  # [H*W]
        depth_map = depth_map.reshape(cam.image_height, cam.image_width)  # [H, W]

        # Visibility check
        # in_bounds = (0 <= u) & (u < cam.image_width) & (0 <= v) & (v < cam.image_height) &
        in_bounds = (depth > 0)
        # Sample from 2D depth_map
        u_clamp = np.clip(np.floor(u).astype(int), 0, cam.image_width - 1)
        v_clamp = np.clip(np.floor(v).astype(int), 0, cam.image_height - 1)
        sampled_depths = depth_map[v_clamp, u_clamp]
        discrepancy = np.abs(depth - sampled_depths) / (depth + 1e-6)
        consistent = discrepancy < threshold
        valid_mask |= (in_bounds & consistent)

    return control_points[valid_mask]


import numpy as np
from geomdl import helpers
from geomdl import BSpline  # Assume Surface is BSpline.Surface


def compute_basis_for_surface(surf, params_u=None, params_v=None, max_deriv=2):
    """
    Computes B-Spline basis functions and their derivatives for a given geomdl Surface
    at a grid of parametric points.

    Args:
        surf (geomdl.BSpline.Surface or geomdl.NURBS.Surface): The input surface.
        u (int): Number of evaluation points in u-direction (default: 100).
        v (int): Number of evaluation points in v-direction (default: 100).
        max_deriv (int): Maximum derivative order to compute (0 for basis only, up to 2; default: 2).

    Returns:
        dict: {
            'basis_u': np.array (num_points_u, ctrlpts_size_u) - Zeroth-order basis in u-dir.
            'ders_u': np.array (num_points_u, max_deriv+1, ctrlpts_size_u) - Derivatives in u-dir (ders_u[:, k, :] is k-th deriv).
            'basis_v': np.array (num_points_v, ctrlpts_size_v) - Zeroth-order basis in v-dir.
            'ders_v': np.array (num_points_v, max_deriv+1, ctrlpts_size_v) - Derivatives in v-dir.
            'params_u': np.array (num_points_u) - Parametric points in u.
            'params_v': np.array (num_points_v) - Parametric points in v.
        }
    # """
    # # Extract surface properties
    # degree_u = surf.degree_u
    # degree_v = surf.degree_v
    # knotvector_u = surf.knotvector_u
    # knotvector_v = surf.knotvector_v
    # ctrlpts_size_u = surf.ctrlpts_size_u
    # ctrlpts_size_v = surf.ctrlpts_size_v
    #
    # # Generate parametric evaluation points (uniform grid in [0,1])
    # if params_u is None:
    #     params_u = np.linspace(0, 1, 100)
    # if params_v is None:
    #     params_v = np.linspace(0, 1, 100)
    num_points_u = params_u.shape[0]
    num_points_v = params_v.shape[0]
    # # Preallocate arrays
    # basis_u = np.zeros((u, ctrlpts_size_u))
    # ders_u = np.zeros((u, max_deriv + 1, ctrlpts_size_u)) if max_deriv > 0 else None
    # basis_v = np.zeros((v, ctrlpts_size_v))
    # ders_v = np.zeros((v, max_deriv + 1, ctrlpts_size_v)) if max_deriv > 0 else None
    # # degree, knot_vector, num_ctrlpts, knots, func = find_span_linear
    # # Compute for u-direction
    # span_u = helpers.find_spans(degree_u, knotvector_u,len(knotvector_u)-degree_u-1, params_u)
    # span_v = helpers.find_spans(degree_v, knotvector_v,len(knotvector_v)-degree_v-1, params_v)
    # ders_vals = helpers.basis_function_ders(degree_u, knotvector_u, span_u, params_u, max_deriv)
    # ders_vals_t = torch.stack([torch.tensor(d) for d in ders_vals])
    # # degree, knot_vector, spans, knots, order
    # for i, u in enumerate(params_u):
    #     # Basis and derivatives all at once
    #     basis_u[i, span_u - degree_u:span_u + 1] = ders_vals[0]  # Zeroth-order
    # if max_deriv > 0:
    #     for k in range(1, max_deriv + 1):
    #         ders_u[i, k, span_u - degree_u:span_u + 1] = ders_vals[k]
    #
    # # Similarly for v-direction
    # for j, v in enumerate(params_v):
    #     # span_v = helpers.find_span(knotvector_v, degree_v, v)
    #     # ders_vals = helpers.basis_function_ders_all(degree_v, knotvector_v, span_v, v, max_deriv)
    #     basis_v[j, span_v - degree_v:span_v + 1] = ders_vals[0]
    #     if max_deriv > 0:
    #         for k in range(1, max_deriv + 1):
    #             ders_v[j, k, span_v - degree_v:span_v + 1] = ders_vals[k]
    #
    # result = {
    #     'bu': basis_u,
    #     'bv': basis_v,
    #     'samples_u': params_u,
    #     'samples_v': params_v
    # }
    # if max_deriv > 0:
    #     result['dbu'] = ders_u[:, 0, :]
    #     result['dbv'] = ders_v[:, 0, :]
    #     result['d2bu'] = ders_u[:,1,:]
    #     result['d2bv'] = ders_v[:,1,:]
    # Extract surface properties
    degree_u = surf.degree_u
    degree_v = surf.degree_v
    knotvector_u = np.array(surf.knotvector_u)  # Ensure NumPy for geomdl
    knotvector_v = np.array(surf.knotvector_v)
    ctrlpts_size_u = surf.ctrlpts_size_u
    ctrlpts_size_v = surf.ctrlpts_size_v

    # Generate parametric evaluation points (uniform grid in [0,1])
    params_u = np.linspace(0, 1, num_points_u)
    params_v = np.linspace(0, 1, num_points_v)

    # Preallocate full basis/derivative matrices (sparse, to be filled)
    basis_u = np.zeros((num_points_u, ctrlpts_size_u))
    ders_u = np.zeros((num_points_u, max_deriv + 1, ctrlpts_size_u)) if max_deriv > 0 else None
    basis_v = np.zeros((num_points_v, ctrlpts_size_v))
    ders_v = np.zeros((num_points_v, max_deriv + 1, ctrlpts_size_v)) if max_deriv > 0 else None

    # Compute for u-direction (fully vectorized)
    span_u = helpers.find_spans(degree_u, knotvector_u, ctrlpts_size_u, params_u)
    ders_vals_u = helpers.basis_functions_ders(degree_u, knotvector_u, span_u, params_u, max_deriv)
    ders_vals_u_np = np.array(ders_vals_u)  # (num_points_u, max_deriv + 1, degree_u + 1)
    span_u = np.asarray(span_u)
    # Vectorized assignment to full matrices (using advanced indexing)
    starts_u = np.asarray(span_u) - degree_u
    offsets = np.arange(degree_u + 1)
    col_indices_u = starts_u[:, np.newaxis] + offsets  # (num_points_u, degree_u + 1)
    row_indices_u = np.arange(num_points_u)[:, np.newaxis]  # (num_points_u, 1)

    # Assign zeroth-order basis
    basis_u[row_indices_u, col_indices_u] = ders_vals_u_np[:, 0, :]

    # Assign derivatives if requested
    if max_deriv > 0:
        # Broadcast for all deriv orders
        deriv_indices = np.arange(max_deriv + 1)[np.newaxis, :, np.newaxis]  # (1, max_deriv+1, 1)
        row_indices_3d = row_indices_u[:, np.newaxis,
                         :]  # (num_points_u, 1, 1) -> broadcast to (num_points_u, max_deriv+1, degree_u+1)
        col_indices_3d = col_indices_u[:, np.newaxis, :]  # (num_points_u, 1, degree_u+1)
        ders_u[row_indices_3d, deriv_indices, col_indices_3d] = ders_vals_u_np  # Vectorized scatter

    # Repeat for v-direction (symmetric)
    span_v = helpers.find_spans(degree_v, knotvector_v, ctrlpts_size_v, params_v)
    ders_vals_v = helpers.basis_functions_ders(degree_v, knotvector_v, span_v, params_v, max_deriv)
    ders_vals_v_np = np.array(ders_vals_v)  # (num_points_v, max_deriv + 1, degree_v + 1)
    span_v = np.asarray(span_v)

    starts_v = span_v - degree_v
    col_indices_v = starts_v[:, np.newaxis] + offsets  # Reuse offsets if degrees match, else recreate
    row_indices_v = np.arange(num_points_v)[:, np.newaxis]

    basis_v[row_indices_v, col_indices_v] = ders_vals_v_np[:, 0, :]

    if max_deriv > 0:
        row_indices_3d_v = row_indices_v[:, np.newaxis, :]
        col_indices_3d_v = col_indices_v[:, np.newaxis, :]
        ders_v[row_indices_3d_v, deriv_indices, col_indices_3d_v] = ders_vals_v_np
    basis_u = torch.tensor(basis_u.tolist(), device='cuda', dtype=torch.float32)
    basis_v = torch.tensor(basis_v.tolist(), device='cuda', dtype=torch.float32)
    ders_u = torch.tensor(ders_u.tolist(), device='cuda', dtype=torch.float32)
    ders_v = torch.tensor(ders_v.tolist(), device='cuda', dtype=torch.float32)
    return BasisFuncs(ders_u[:, 0, :], ders_u[:, 1, :], ders_u[:, 2, :], ders_v[:,0,:], ders_v[:, 1, :], ders_v[:, 2, :])


import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from typing import List, Tuple, Optional


def _find_best_grid_factors(K: int, aspect_ratio: float) -> Tuple[int, int]:
    """
    Finds the factor pair (size_u, size_v) of K such that
    size_u * size_v == K and (size_u / size_v) is closest to aspect_ratio.

    Returns:
        (size_u, size_v)
    """
    if K == 0:
        return 0, 0

    best_pair = (K, 1)
    min_ratio_diff = float('inf')

    # We only need to check up to sqrt(K)
    for v_test in range(1, int(np.sqrt(K)) + 1):
        if K % v_test == 0:
            u_test = K // v_test

            # Check pair (u_test, v_test)
            ratio = u_test / v_test
            ratio_diff = abs(ratio - aspect_ratio)
            if ratio_diff < min_ratio_diff:
                min_ratio_diff = ratio_diff
                best_pair = (u_test, v_test)

            # Check pair (v_test, u_test)
            ratio = v_test / u_test
            ratio_diff = abs(ratio - aspect_ratio)
            if ratio_diff < min_ratio_diff:
                min_ratio_diff = ratio_diff
                best_pair = (v_test, u_test)

    return best_pair


def get_balanced_grid_from_points(
        points_np: np.ndarray,
        associated_data: Optional[np.ndarray] = None,
        total_grid_points: int = None,
        merge_threshold: float = 1e-4,
        jitter_std: float = 1e-9
) -> Tuple[list, Optional[list], int, int]:
    """
    Parametrizes 3D points, calculates a balanced UV grid, and sorts both
    points and their associated data (e.g., RGB) to that grid.

    (Version 3: Fixes grid size mismatch bug)
    """

    has_data = associated_data is not None
    if has_data and len(points_np) != len(associated_data):
        raise ValueError("points_np and associated_data must have the same length.")

    # Step 1: Remove exact duplicates
    if points_np.shape[0] == 0:
        print("Warning: Received empty point cloud.")
        return [], None, 0, 0

    unique_points, unique_indices = np.unique(points_np, axis=0, return_index=True)
    if len(unique_points) < len(points_np):
        print(f"Removed {len(points_np) - len(unique_points)} exact duplicates")
        points_np = unique_points
        if has_data:
            associated_data = associated_data[unique_indices]

    # Step 2: Merge near-duplicates
    # ... (merge logic remains the same) ...
    tree = KDTree(points_np)
    to_merge_groups = []
    to_merge_data = []
    visited = set()
    for i in range(len(points_np)):
        if i in visited: continue
        close_idxs = tree.query_ball_point(points_np[i], r=merge_threshold)
        if len(close_idxs) > 1:
            avg_point = np.mean(points_np[close_idxs], axis=0)
            to_merge_groups.append(avg_point)
            if has_data:
                avg_data = np.mean(associated_data[close_idxs], axis=0)
                to_merge_data.append(avg_data)
            visited.update(close_idxs)

    if visited:
        unvisited_mask = np.ones(len(points_np), dtype=bool)
        unvisited_mask[list(visited)] = False
        final_points = [points_np[unvisited_mask]]
        final_data = [associated_data[unvisited_mask]] if has_data else []
        if to_merge_groups:
            final_points.append(np.array(to_merge_groups))
            if has_data:
                final_data.append(np.array(to_merge_data))
        points_np = np.concatenate(final_points, axis=0)
        if has_data:
            associated_data = np.concatenate(final_data, axis=0)
        print(f"Merged {len(visited)} near-duplicates. New total: {len(points_np)}")

    # Step 3: Add jitter (ONLY to geometry)
    if jitter_std > 0:
        jitter = np.random.normal(0, jitter_std, points_np.shape)
        points_np += jitter

    # Step 4: Determine total grid points (K)
    # If a target K is given, use it. Otherwise, use the size of the cleaned cloud.
    K = total_grid_points if total_grid_points is not None else len(points_np)

    # --- IMPORTANT ---
    # If the number of points (after cleaning) is not K, we must resample.
    # This is a warning that our input `total_ctrlpts` from filter_points
    # might not be respected if cleaning removes too many points.
    if len(points_np) != K and total_grid_points is not None:
        print(f"Warning: Point cloud length ({len(points_np)}) does not match "
              f"target grid size ({K}) after cleaning. Resampling...")
        # We must resample points_np to K points.
        # This is complex (e.g., iterative adding/removing).
        # For now, we will proceed, but this is a source of mismatch.
        # Easiest solution: just use the cleaned points as the target.
        K = len(points_np)
        print(f"Adjusted target grid size to {K}")

    if K == 0:
        return [], None, 0, 0

    # Step 5: PCA projection
    centroid = np.mean(points_np, axis=0)
    centered = points_np - centroid
    pca = PCA(n_components=2)
    uv_coords = pca.fit_transform(centered)

    # --- NEW: Calculate balanced size_u and size_v (FIXED) ---
    lambda1, lambda2 = pca.explained_variance_
    # sigma1 = np.sqrt(lambda1)
    # sigma2 = np.sqrt(lambda2)
    # aspect_ratio = sigma1 / (sigma2 + 1e-9)
    aspect_ratio = lambda1 / (lambda2 + 1e-9)
    if aspect_ratio < .125 or aspect_ratio > 8.0:
        print(f"Warning: Extreme PCA aspect ratio detected: {aspect_ratio:.4f}")
        aspect_ratio = max(aspect_ratio, 1.0 / aspect_ratio)

    # Use the new helper function to get an *exact* factorization
    # size_u, size_v = _find_best_grid_factors(K, aspect_ratio)
    size_u, size_v = K // int(np.sqrt(K * aspect_ratio)), K // int(np.sqrt(K / aspect_ratio))
    # K_actual now *always* equals K
    K_actual = size_u * size_v

    print(f"PCA aspect ratio: {aspect_ratio:.4f}. "
          f"Calculated grid: {size_u} (u) x {size_v} (v) = {K_actual} (Target: {K})")
    # --- End fixed logic ---

    # Step 6: Normalize UVs and sort points to the new grid
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_norm = (uv_coords - uv_min) / (uv_max + 1e-8 - uv_min)  # Corrected denominator

    u_grid = np.linspace(0, 1, size_u)
    v_grid = np.linspace(0, 1, size_v)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_points = np.column_stack((uu.ravel(), vv.ravel()))  # (K_actual, 2)

    uv_tree = KDTree(uv_norm)
    _, indices = uv_tree.query(grid_points)

    sorted_points = points_np[indices]
    # After sorted_points
    tree = KDTree(sorted_points)
    for i in range(len(sorted_points)):
        dups = tree.query_ball_point(sorted_points[i], merge_threshold)
        if len(dups) > 1:
            sorted_points[dups] += np.random.normal(0, merge_threshold*1e-1, (len(dups), 3))  # Micro-jitter
            # associated_data[dups] += np.random.normal(0, 1e-6, (len(dups), 3))  # Micro-jitter
    sorted_data = associated_data[indices] if has_data else None

    sorted_points_list = sorted_points.reshape(size_v, size_u, 3).reshape(-1, 3).tolist()
    sorted_data_list = sorted_data.reshape(size_v, size_u, -1).reshape(-1, sorted_data.shape[
        -1]).tolist() if has_data else None
    return sorted_points_list, sorted_data_list, size_u, size_v
def get_balanced_grid_from_pointse(
        points_np: np.ndarray,
        associated_data: Optional[np.ndarray] = None,
        total_grid_points: int = None,
        merge_threshold: float = 1e-6,
        jitter_std: float = 1e-8
) -> Tuple[list, Optional[list], int, int]:
    """
    Parametrizes 3D points, calculates a balanced UV grid, and sorts both
    points and their associated data (e.g., RGB) to that grid.

    Args:
        points_np: Input point cloud (N, 3).
        associated_data: Input data (e.g., RGB) corresponding to points (N, C).
        total_grid_points: The total number of points desired for the output grid (K).
                           If None, defaults to the number of points remaining
                           after cleaning.

    Returns:
        A tuple containing:
        - (list): The sorted 3D points (K, 3).
        - (list or None): The sorted associated data (K, C), or None.
        - (int): The calculated balanced grid width, size_u.
        - (int): The calculated balanced grid height, size_v.
    """

    has_data = associated_data is not None
    if has_data and len(points_np) != len(associated_data):
        raise ValueError("points_np and associated_data must have the same length.")

    # Step 1: Remove exact duplicates
    if points_np.shape[0] == 0:
        return [], None, 0, 0

    unique_points, unique_indices = np.unique(points_np, axis=0, return_index=True)
    if len(unique_points) < len(points_np):
        print(f"Removed {len(points_np) - len(unique_points)} exact duplicates")
        points_np = unique_points
        # NEW: Filter associated_data with the same indices
        if has_data:
            associated_data = associated_data[unique_indices]

    # Step 2: Merge near-duplicates (cluster and average)
    tree = KDTree(points_np)
    to_merge_groups = []
    to_merge_data = []  # NEW: To store merged data
    visited = set()

    for i in range(len(points_np)):
        if i in visited:
            continue
        close_idxs = tree.query_ball_point(points_np[i], r=merge_threshold)

        if len(close_idxs) > 1:
            avg_point = np.mean(points_np[close_idxs], axis=0)
            to_merge_groups.append(avg_point)

            # NEW: Average the corresponding data (e.g., colors)
            if has_data:
                # Use mean for colors. For other data, might need different logic.
                avg_data = np.mean(associated_data[close_idxs], axis=0)
                to_merge_data.append(avg_data)

            visited.update(close_idxs)

    if visited:
        unvisited_mask = np.ones(len(points_np), dtype=bool)
        unvisited_mask[list(visited)] = False

        final_points = [points_np[unvisited_mask]]
        final_data = [associated_data[unvisited_mask]] if has_data else []

        if to_merge_groups:
            final_points.append(np.array(to_merge_groups))
            if has_data:
                final_data.append(np.array(to_merge_data))

        points_np = np.concatenate(final_points, axis=0)
        if has_data:
            associated_data = np.concatenate(final_data, axis=0)

        print(f"Merged {len(visited)} near-duplicates into {len(to_merge_groups)} points. New total: {len(points_np)}")

    # Step 3: Add small jitter (ONLY to geometry)
    if jitter_std > 0:
        jitter = np.random.normal(0, jitter_std, points_np.shape)
        points_np += jitter

    # Step 4: Determine total grid points (K)
    K = total_grid_points if total_grid_points is not None else len(points_np)
    if K == 0:
        return [], None, 0, 0

    # Step 5: PCA projection (using ONLY geometry)
    centroid = np.mean(points_np, axis=0)
    centered = points_np - centroid
    pca = PCA(n_components=2)
    uv_coords = pca.fit_transform(centered)

    # Calculate balanced size_u and size_v
    lambda1, lambda2 = pca.explained_variance_
    sigma1 = np.sqrt(lambda1)
    sigma2 = np.sqrt(lambda2)
    aspect_ratio = sigma1 / (sigma2 + 1e-9)

    size_v = int(np.round(np.sqrt(K / aspect_ratio)))
    size_v = max(1, size_v)
    size_u = int(np.round(K / size_v))
    size_u = max(1, size_u)

    # K might change slightly due to rounding, so we use the *actual* grid size
    K_actual = size_u * size_v

    print(f"PCA aspect ratio: {aspect_ratio:.4f}. Calculated grid: {size_u}x{size_v} = {K_actual}")

    # Step 6: Normalize UVs and sort points to the new grid
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_norm = (uv_coords - uv_min) / (uv_max - uv_min + 1e-8)

    u_grid = np.linspace(0, 1, size_u)
    v_grid = np.linspace(0, 1, size_v)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_points = np.column_stack((uu.ravel(), vv.ravel()))  # (K_actual, 2)

    uv_tree = KDTree(uv_norm)
    _, indices = uv_tree.query(grid_points)

    # Apply final sorting indices to BOTH points and data
    sorted_points = points_np[indices]
    sorted_data = associated_data[indices] if has_data else None

    # Reshape to v-major, then flatten to (K, 3) list
    sorted_points_list = sorted_points.reshape(size_u, size_v, 3).reshape(-1, 3).tolist()
    sorted_data_list = sorted_data.reshape(size_u, size_v, -1).reshape(-1, sorted_data.shape[
        -1]).tolist() if has_data else None

    return sorted_points_list, sorted_data_list, size_u, size_v
def get_balanced_grid_from_points2(
        points_np: np.ndarray,
        total_grid_points: int = None,
        merge_threshold: float = 1e-6,
        jitter_std: float = 1e-8
) -> Tuple[list, int, int]:
    """
    Parametrizes 3D points, calculates a balanced UV grid, and sorts points to that grid.

    This function:
    1. Cleans the input points by removing exact and near-duplicates.
    2. Performs PCA to find the principal plane (U, V).
    3. Calculates the data's aspect ratio from the PCA eigenvalues.
    4. Determines a balanced (size_u, size_v) grid based on this ratio and
       the 'total_grid_points' (defaults to the number of cleaned points).
    5. Projects points to the UV plane and normalizes them.
    6. Creates the ideal grid and finds the nearest input point for each grid point.
    7. Returns the sorted points, size_u, and size_v.

    Args:
        points_np: Input point cloud (N, 3).
        total_grid_points: The total number of points desired for the output grid (K).
                           If None, defaults to the number of points remaining
                           after cleaning.
        merge_threshold: Radius for merging near-duplicate points.
        jitter_std: Standard deviation of Gaussian noise to add for stability.

    Returns:
        A tuple containing:
        - (list): The sorted points, reshaped to v-major order (K, 3).
        - (int): The calculated balanced grid width, size_u.
        - (int): The calculated balanced grid height, size_v.
    """

    # Step 1: Remove exact duplicates
    if points_np.shape[0] == 0:
        print("Warning: Received empty point cloud.")
        return [], 0, 0

    unique_points, unique_indices = np.unique(points_np, axis=0, return_index=True)
    if len(unique_points) < len(points_np):
        print(f"Removed {len(points_np) - len(unique_points)} exact duplicates")
        points_np = unique_points

    # Step 2: Merge near-duplicates (cluster and average)
    # --- This logic has been corrected ---
    tree = KDTree(points_np)
    to_merge_groups = []
    visited = set()
    for i in range(len(points_np)):
        if i in visited:
            continue
        # Find all points within the merge_threshold
        close_idxs = tree.query_ball_point(points_np[i], r=merge_threshold)

        if len(close_idxs) > 1:
            # Found a cluster to merge
            avg_point = np.mean(points_np[close_idxs], axis=0)
            to_merge_groups.append(avg_point)
            visited.update(close_idxs)

    if visited:
        # Rebuild points_np: keep all unvisited points
        unvisited_mask = np.ones(len(points_np), dtype=bool)
        unvisited_mask[list(visited)] = False
        final_points = [points_np[unvisited_mask]]

        # Add the new averaged points from merged groups
        if to_merge_groups:
            final_points.append(np.array(to_merge_groups))

        points_np = np.concatenate(final_points, axis=0)
        print(f"Merged {len(visited)} near-duplicates into {len(to_merge_groups)} points. New total: {len(points_np)}")
    # --- End corrected logic ---

    # Step 3: Add small jitter to avoid zero divisions (Gaussian noise)
    if jitter_std > 0:
        jitter = np.random.normal(0, jitter_std, points_np.shape)
        points_np += jitter
        # print(f"Added jitter (std={jitter_std}) for stability") # Often too noisy

    # Step 4: Determine total grid points (K)
    K = total_grid_points if total_grid_points is not None else len(points_np)
    if K == 0:
        print("Warning: No points left after cleaning.")
        return [], 0, 0

    # Step 5: PCA projection to find axes and aspect ratio
    centroid = np.mean(points_np, axis=0)
    centered = points_np - centroid
    pca = PCA(n_components=2)
    uv_coords = pca.fit_transform(centered)

    # --- NEW: Calculate balanced size_u and size_v ---
    lambda1, lambda2 = pca.explained_variance_
    sigma1 = np.sqrt(lambda1)
    sigma2 = np.sqrt(lambda2)
    aspect_ratio = sigma1 / (sigma2 + 1e-9)  # Ratio of principal axes (U/V)

    # Solve for Nu, Nv given:
    # 1) Nu * Nv = K
    # 2) Nu / Nv = aspect_ratio
    # -> Nv^2 * aspect_ratio = K => Nv = sqrt(K / aspect_ratio)

    size_v = int(np.round(np.sqrt(K / aspect_ratio)))
    size_v = max(1, size_v)  # Ensure at least 1

    # Derive size_u from K and size_v to get a product closer to K
    size_u = int(np.round(K / size_v))
    size_u = max(1, size_u)

    print(f"PCA aspect ratio (s1/s2): {aspect_ratio:.4f}")
    print(f"Calculated grid size: {size_u} (u) x {size_v} (v) = {size_u * size_v} (Target: {K})")
    # --- End new logic ---

    # Step 6: Normalize UVs and sort points to the new grid
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_norm = (uv_coords - uv_min) / (uv_max - uv_min + 1e-8)

    # Create the ideal grid based on calculated sizes
    u_grid = np.linspace(0, 1, size_u)
    v_grid = np.linspace(0, 1, size_v)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_points = np.column_stack((uu.ravel(), vv.ravel()))  # (K, 2)

    # Use a KD-tree on the *normalized point UVs*
    uv_tree = KDTree(uv_norm)

    # Find the *nearest original point* for each *ideal grid point*
    # This performs the sorting and resampling in one step
    _, indices = uv_tree.query(grid_points)

    sorted_points = points_np[indices]

    # Reshape to v-major, then flatten to (K, 3) list
    sorted_list = sorted_points.reshape(size_v, size_u, 3).reshape(-1, 3).tolist()

    return sorted_list, size_u, size_v
def parametrize_and_sort_points(points_np: np.ndarray, size_u: int, size_v: int, merge_threshold: float = 1e-6,
                                jitter_std: float = 1e-8) -> list:
    """
    Parametrizes unsorted 3D points to UV grid, handles duplicates/clusters, and sorts into v-major order.
    """
    # if len(points_np) != size_u * size_v:
    #     raise ValueError(f"Point count mismatch: {len(points_np)} != {size_u * size_v}")

    # Step 1: Remove exact duplicates
    unique_points, unique_indices = np.unique(points_np, axis=0, return_index=True)
    if len(unique_points) < len(points_np):
        print(f"Removed {len(points_np) - len(unique_points)} exact duplicates")
        points_np = unique_points  # Will pad later if needed

    # Step 2: Merge near-duplicates (cluster and average)
    tree = KDTree(points_np)
    to_merge = []
    visited = set()
    for i in range(len(points_np)):
        if i in visited: continue
        dists, idxs = tree.query(points_np[i], k=10)  # Check nearby
        close_idxs = idxs[dists < merge_threshold]
        if len(close_idxs) > 1:
            avg_point = np.mean(points_np[close_idxs], axis=0)
            to_merge.append((close_idxs, avg_point))
            visited.update(close_idxs)

    # if to_merge:
    #     merged_points = np.array([avg for _, avg in to_merge])
    #     points_np = merged_points
    #     print(f"Merged {sum(len(idxs) - 1 for idxs, _ in to_merge)} near-duplicates")

    # Step 3: Add small jitter to avoid zero divisions (Gaussian noise)
    if jitter_std > 0:
        jitter = np.random.normal(0, jitter_std, points_np.shape)
        points_np += jitter
        print(f"Added jitter (std={jitter_std}) for stability")

    # Step 4: Proceed with PCA projection and sorting (as before)
    # Center, PCA to UV, normalize, nearest to grid, etc.
    centroid = np.mean(points_np, axis=0)
    centered = points_np - centroid
    pca = PCA(n_components=2)
    uv_coords = pca.fit_transform(centered)
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_norm = (uv_coords - uv_min) / (uv_max - uv_min + 1e-8)

    u_grid = np.linspace(0, 1, size_u)
    v_grid = np.linspace(0, 1, size_v)
    uu, vv = np.meshgrid(u_grid, v_grid)
    tree = KDTree(uv_norm)
    grid_points = np.column_stack((uu.ravel(), vv.ravel()))
    _, indices = tree.query(grid_points)

    sorted_points = points_np[indices]
    return sorted_points.reshape(size_v, size_u, 3).reshape(-1, 3).tolist()  # v-major flat; transpose if needed





def elevate_surface_degree(surf, direction: str = 'u', num: int = 1):
    """
    Elevates degree of the BSpline surface in u or v direction, handling general cases by decomposing to Bezier patches.

    Args:
        surf: geomdl BSpline Surface object.
        direction: 'u' or 'v' for elevation axis.
        num: Number of elevations (default 1).

    Returns:
        Updated surface with elevated degree.
    """

    from geomdl import helpers
    import copy
    # Copy to avoid modifying original
    surf_copy = copy.deepcopy(surf)

    if direction == 'u':
        # Get unique interior knots (exclude ends)
        knots_u = np.unique(surf.knotvector_u[surf.degree_u + 1: -(surf.degree_u + 1)])

        # Insert to full multiplicity for Bezier decomposition
        for knot in knots_u:
            current_mult = surf.knotvector_u.count(knot)
            insert_num = surf.degree_u + 1 - current_mult
            if insert_num > 0:
                surf_copy.insert_knot(knot, direction='u', num=insert_num)

        # Now Bezier patches: Elevate each "curve" row
        elevated_ctrlpts2d = []
        for row in surf_copy.ctrlpts2d:  # Each v-row as curve
            flat_row = [pt for pt in row]  # Flat [x,y,z,...]
            elevated_row = helpers.degree_elevation(degree=surf.degree_u, ctrlpts=flat_row, num=num)
            elevated_ctrlpts2d.append([elevated_row[i:i + 3] for i in range(0, len(elevated_row), 3)])
        surf.ctrlpts2d = elevated_ctrlpts2d
        surf.degree_u += num

        # Note: Knot vector auto-updates in geomdl; if not, regenerate via utilities.generate_knot_vector
        surf.knotvector_u = geomdl.utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)

    elif direction == 'v':
        # Transpose, elevate as u, transpose back
        transposed_ctrlpts = list(map(list, zip(*surf.ctrlpts2d)))
        transposed_kv = surf.knotvector_v
        transposed_degree = surf.degree_v
        transposed_size = surf.ctrlpts_size_v

        # Insert for Bezier
        knots_v = np.unique(transposed_kv[transposed_degree + 1: -(transposed_degree + 1)])
        temp_surf = copy.deepcopy(surf)  # Temp for insertion
        temp_surf.ctrlpts2d = transposed_ctrlpts
        temp_surf.knotvector_u = transposed_kv  # Treat v as u
        for knot in knots_v:
            current_mult = temp_surf.knotvector_u.count(knot)
            insert_num = transposed_degree + 1 - current_mult
            if insert_num > 0:
                temp_surf.insert_knot(knot, direction='u', num=insert_num)

        # Elevate rows
        elevated_trans = []
        for row in temp_surf.ctrlpts2d:
            flat_row = [pt for pt in row]
            elevated_row = helpers.degree_elevation(degree=transposed_degree, ctrlpts=flat_row, num=num)
            elevated_trans.append([elevated_row[i:i + 3] for i in range(0, len(elevated_row), 3)])

        # Transpose back
        surf.ctrlpts2d = list(map(list, zip(*elevated_trans)))
        surf.degree_v += num
        from geomdl import utilities
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

    return surf


def insert_knot_in_high_error(surf, points_list, direction='u', error_thresh=0.01, num_insert=1):
    points_np = np.array(points_list)
    # Sample params in direction
    if direction == 'u':
        fixed_v = 0.5
        params = [[u, fixed_v] for u in np.linspace(0, 1, 100)]
    else:
        fixed_u = 0.5
        params = [[fixed_u, v] for v in np.linspace(0, 1, 100)]

    eval_points = np.array(surf.evaluate_list(params))
    errors = np.min(np.linalg.norm(eval_points[:, None] - points_np[None, :], axis=2), axis=1)

    high_error_idx = np.argmax(errors)
    if errors[high_error_idx] > error_thresh:
        param_to_insert = params[high_error_idx][0 if direction == 'u' else 1]
        surf.insert_knot(param_to_insert, direction=direction, num=num_insert)
    return surf


def compute_delta_error(surf1, surf2, num_samples: int = 100):
    """
    Computes the mean squared error between two BSpline surfaces by sampling at uniform parametric points.

    Args:
        surf1: Original geomdl BSpline Surface.
        surf2: Modified geomdl BSpline Surface (e.g., after knot removal).
        num_samples: Number of samples per u/v dimension (total samples = num_samples^2).

    Returns:
        Float scalar: Mean squared delta error.
    """
    # Generate uniform parametric grid
    u_samples = np.linspace(0, 1, num_samples)
    v_samples = np.linspace(0, 1, num_samples)
    params = [[u, v] for u in u_samples for v in v_samples]  # Flat list of [u,v] pairs

    # Evaluate both surfaces
    eval1 = np.array(surf1.evaluate_list(params))  # [num_samples^2, 3]
    eval2 = np.array(surf2.evaluate_list(params))  # [num_samples^2, 3]

    # Compute point-wise squared differences
    deltas = np.linalg.norm(eval1 - eval2, axis=1)  # [num_samples^2]
    mse = np.mean(deltas ** 2)

    return mse
def remove_knot_in_low_error(surf, direction='u', error_thresh=0.005):
    # Identify candidate knots (interior)
    knots = surf.knotvector_u if direction == 'u' else surf.knotvector_v
    interior_knots = np.unique(knots[surf.degree_u + 1: -(surf.degree_u + 1)]) if direction == 'u' else np.unique(
        knots[surf.degree_v + 1: -(surf.degree_v + 1)])

    for knot in interior_knots:
        # Temporarily remove and check error change
        temp_surf = copy.deepcopy(surf)
        try:
            temp_surf.remove_knot(knot, direction=direction, num=1)
            # Eval error on temp vs original (small if removable)
            # If delta_error < thresh, commit removal
            if compute_delta_error(surf, temp_surf) < error_thresh:
                surf.remove_knot(knot, direction=direction, num=1)
        except:
            pass  # Skip if not removable


import numpy as np
from geomdl import fitting


# (Requires _find_best_grid_factors and get_balanced_grid_from_points)

def filter_points(points_list, rgb_pcd_src, total_ctrlpts, cameras, use_approximate, centripetal, merge_threshold, degree_u=3,
                  degree_v=3, consistency_threshold=0.1,
                  max_iters=2):
    current_points = np.asarray(points_list)
    current_rgb = np.asarray(rgb_pcd_src)

    if len(current_points) != len(current_rgb):
        raise ValueError("Source points and RGB data must have the same length.")

    # --- Subsample to K points (total_ctrlpts) ---
    indices = np.arange(current_points.shape[0])
    shuffled_indices = np.random.permutation(indices)

    current_points = current_points[shuffled_indices][:total_ctrlpts]
    current_rgb = current_rgb[shuffled_indices][:total_ctrlpts]

    # --- NEW LOGIC: Fork based on approximation or interpolation ---
    # scale = normalize_point_cloud(current_points)
    # current_points /= scale


    if use_approximate:
        # --- PATH A: APPROXIMATION ---

        # 1. We DON'T need to sort or resample. We just need a balanced (u, v) size.
        #    We can run a simple PCA on the *subsampled* points just to get the ratio.
        centered = current_points - np.mean(current_points, axis=0)
        try:
            pca = PCA(n_components=2)
            pca.fit(centered)
            lambda1, lambda2 = pca.explained_variance_
            aspect_ratio = np.sqrt(lambda1) / (np.sqrt(lambda2) + 1e-9)
        except Exception:
            aspect_ratio = 1.0  # Fallback for safety

        size_u, size_v = _find_best_grid_factors(total_ctrlpts, aspect_ratio)
        # points_list, rgb_list, size_u, size_v = get_balanced_grid_from_points(
            # current_points,
            # current_rgb,
            # total_grid_points=total_ctrlpts
        # )
        # 2. Call approximate_surface with the RAW (unsorted) points.
        #    geomdl will handle the parameterization internally (using chord length).
        #    We must pass the points as a list of tuples/lists.
        surf = fitting.approximate_surface(
            current_points.tolist(),
            size_u, size_v,  # Note the v, u swap
            degree_u, degree_v,
            centripetal=centripetal
        )

        # 3. We can't return a sorted rgb_list because the points weren't sorted.
        #    The concept of a "color per control point" is less defined here,
        #    as the surface CPs don't lie *on* the input points.
        #    We return None for rgb_list to reflect this.
        rgb_list = None

    else:
        # --- PATH B: INTERPOLATION (Your previous logic) ---

        # 1. We MUST use the sorting function to create the grid.
        #    This is where the ZeroDivisionError happens if duplicates are made.
        points_list, rgb_list, size_u, size_v = get_balanced_grid_from_points(
            current_points,
            current_rgb,
            total_grid_points=total_ctrlpts,
            merge_threshold=merge_threshold,
        )

        # 2. Call interpolate_surface.
        #    RECOMMENDATION: Always use centripetal=True here. It's more stable
        #    and designed for non-uniform data, which this is.
        surf = fitting.interpolate_surface(
            points_list,
            size_v, size_u,  # Note the v, u swap
            # size_u, size_v,  # Note the v, u swap
            degree_u, degree_v,
            centripetal=centripetal  # Strongly recommended
        )
        # if size_v > size_
        # surf.transpose()
    return surf, rgb_list, size_u, size_v




def extract_connected_components(points_3d, max_edge_length=0.1):
    rips = RipsComplex(points=points_3d, max_edge_length=0.1)
    st = rips.create_simplex_tree(max_dimension=0)  # 0D for components
    persistence = st.persistence(min_persistence=0.01)  # Filter noise

    # Extract 0D features (components)
    components = []
    for interval in persistence:
        if interval[0] == 0:  # Dimension 0
            birth, death = interval[1]
            if death == float('inf'):  # Persistent components
                # Get points in this component (via filtration)
                comp_indices = st.filtration_indices(birth)  # Custom: Trace simplices
                components.append(points_3d[comp_indices])

    return components  # List[np.array (M_i, 3)]
def normalize_pts(pcd):
    min_bounds = pcd.min(axis=0)
    max_bounds = pcd.max(axis=0)
    scale_factor = max_bounds - min_bounds
    scale_factor[scale_factor < 1e-6] = 1.0  # Avoid div-zero on flat dims
    normalized_points = (pcd - min_bounds) / scale_factor
    return normalized_points, scale_factor

def resample_to_grid(filtered_ctrlpts: np.ndarray, size_u: int, size_v: int, method: str = 'linear') -> np.ndarray:
    """
    Resamples filtered control points to a uniform size_u x size_v grid.

    Args:
        filtered_ctrlpts: [M, 3] array of pruned points.
        size_u, size_v: Target grid dimensions.
        method: Interpolation method ('nearest', 'linear', 'cubic').

    Returns:
        [size_u * size_v, 3] resampled points.
    """
    if len(filtered_ctrlpts) < 3:
        raise ValueError("Too few points for resampling; adjust threshold or add jitter.")

    # Define target grid from bounding box (assume xy for u/v, z as value; adapt if PCA-rotated)
    min_bounds = filtered_ctrlpts.min(axis=0)
    max_bounds = filtered_ctrlpts.max(axis=0)
    u_grid = np.linspace(min_bounds[0], max_bounds[0], size_u)
    v_grid = np.linspace(min_bounds[1], max_bounds[1], size_v)
    uu, vv = np.meshgrid(u_grid, v_grid)
    target_grid = np.column_stack((uu.ravel(), vv.ravel()))  # [size_u * size_v, 2]

    # Source coords and values (x,y as params, z as interpolated; for full 3D, interp each dim separately)
    source_xy = filtered_ctrlpts[:, :2]
    resampled = np.zeros((size_u * size_v, 3))
    for dim in range(3):  # Interp x, y, z independently
        resampled[:, dim] = griddata(source_xy, filtered_ctrlpts[:, dim], target_grid, method=method,
                                     fill_value=np.median(filtered_ctrlpts[:, dim]))

    return resampled


def organize_points_by_camera_projection(points_3d, cameras, uv_grid_size=(100, 100), threshold=0.1):
    """
    Organizes 3D points by projecting onto camera planes, sorting by UV, and optionally gridding.

    Args:
        points_3d: np.array (N, 3) - 3D world points.
        cameras: List of dicts, each with 'K' (3x3 intrinsics), 'R' (3x3 rotation), 't' (3 translation), 'width', 'height'.
        uv_grid_size: Tuple (grid_u, grid_v) - Resolution for optional gridding.
        dist_threshold: Float - Depth consistency threshold across views (if multi-camera).

    Returns:
        organized_data: Dict {cam_id: {'sorted_points': np.array (M, 3), 'uv_coords': np.array (M, 2), 'gridded_points': np.array (grid_u, grid_v, 3)}}
    """
    organized_data = {}
    cam = cameras[0]
    uv_grid_size = cam.image_width, cam.image_height
    all_pts = []
    for cam_id, cam in enumerate(cameras):
        points_cam = points_3d @ cam.world_view_transform[:3, :3].cpu().numpy() + cam.world_view_transform[3, :3].cpu().numpy()

        depths = points_cam[:, 2]
        width, height = cam.image_width, cam.image_height

        # Project to camera space: points_cam = R @ (points_3d - t).T
        valid = depths > 0
        K = cam.get_k().cpu().numpy()
        # Project to image plane
        proj = (K @ points_cam[valid].T).T
        uv = proj[:, :2] / proj[:, 2:3]  # (M, 2)

        # In-bounds check
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        valid_indices = np.where(valid)[0][in_bounds]  # Global indices

        # Filtered points and UV
        filtered_points = points_3d[valid_indices]
        filtered_uv = uv[in_bounds]

        # Sort by UV (row-major: sort by v then u)
        sort_idx = np.lexsort((filtered_uv[:, 0], filtered_uv[:, 1]))  # v primary, u secondary
        sorted_points = filtered_points[sort_idx]
        sorted_uv = filtered_uv[sort_idx]

        # Optional gridding: Bin into grid, average points per bin
        gridded_points = np.full((uv_grid_size[1], uv_grid_size[0], 3), np.nan)  # (v, u, 3)
        u_bins = np.floor(sorted_uv[:, 0]).astype(int)#, np.linspace(0, width, uv_grid_size[0] + 1)) - 1
        v_bins = np.floor(sorted_uv[:, 1]).astype(int)#, np.linspace(0, height, uv_grid_size[1] + 1)) - 1
        for i in range(len(sorted_points)):
            u_bin, v_bin = u_bins[i], v_bins[i]
            if np.isnan(gridded_points[v_bin, u_bin, 0]):
                gridded_points[v_bin, u_bin] = sorted_points[i]
            else:
                # Average if multiple points in bin (for consistency)
                gridded_points[v_bin, u_bin] = (gridded_points[v_bin, u_bin] + sorted_points[i]) / 2

        organized_data[cam_id] = {
            'sorted_points': sorted_points,
            'uv_coords': sorted_uv,
            'gridded_points': gridded_points
        }
        all_pts.append(sorted_uv)
    #
    # # Multi-camera aggregation (optional: average across views if point seen in multiple)
    # if len(cameras) > 1:
    #     # Use KDTree for matching points across views
    #     all_points = np.vstack([data['sorted_points'] for data in organized_data.values()])
    #     tree = KDTree(all_points)
    #     unique_points = []
    #     for p in all_points:
    #         dists, idx = tree.query(p, k=len(cameras))
    #         close = all_points[idx[dists < threshold]]
    #         if len(close) > 1:
    #             avg_p = np.mean(close, axis=0)
    #             unique_points.append(avg_p)
    #         else:
    #             unique_points.append(p)
    #     # Dedup and return, but for simplicity, we'll keep per-cam for now

    return np.vstack(all_pts)


import os
import pickle
import hashlib
import numpy as np
from geomdl import fitting  # Make sure fitting is imported


def get_cached_nurbs_fit(
        scene_name: str,
        pcd: np.ndarray,
        pcd_rgb: np.ndarray,
        total_ctrlpts: int,
        use_approximate: bool,
        centripetal: bool,
        cameras,
        merge_threshold,
        cache_dir: str = "./nurbs_cache"
):
    """
    A wrapper for fit_nurbs_from_pointcloud that caches the result on disk.

    The cache key is generated from the scene_name, parameters,
    and a hash of the point cloud data.
    """

    # --- 1. Create a unique cache key ---

    # Hash the large numpy arrays. This is very fast.
    pcd_hash = hashlib.sha256(pcd.tobytes()).hexdigest()
    rgb_hash = hashlib.sha256(pcd_rgb.tobytes()).hexdigest()

    # Combine all parameters into a single key
    key_parts = [
        scene_name,
        str(total_ctrlpts),
        str(use_approximate),
        str(centripetal),
        pcd_hash,
        rgb_hash
    ]
    cache_key = "_".join(key_parts)
    cache_filename = f"{cache_key}.pkl"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # --- 2. Ensure cache directory exists ---
    os.makedirs(cache_dir, exist_ok=True)

    # --- 3. Try to load from cache ---
    try:
        if os.path.exists(cache_filepath):
            print(f"Loading cached NURBS for scene '{scene_name}'...")
            with open(cache_filepath, 'rb') as f:
                nurbs_attr = pickle.load(f)
            return nurbs_attr
    except Exception as e:
        print(f"Cache load failed ({e}). Re-computing...")
        if os.path.exists(cache_filepath):
            os.remove(cache_filepath)  # Remove corrupted file

    # --- 4. If cache miss, compute and save ---
    print(f"Cache miss. Computing NURBS for scene '{scene_name}'...")

    # Call your original, slow function
    nurbs_attr = fit_nurbs_from_pointcloud(
        pcd,
        pcd_rgb,
        total_ctrlpts,
        use_approximate=use_approximate,
        centripetal=centripetal,
        cameras=cameras,
        merge_threshold=merge_threshold
    )

    # Save the result to the cache file
    try:
        with open(cache_filepath, 'wb') as f:
            pickle.dump(nurbs_attr, f)
        print(f"Saved NURBS to cache: {cache_filepath}")
    except Exception as e:
        print(f"Error saving to cache: {e}")

    return nurbs_attr
def fit_nurbs_from_pointcloud(pts3d:np.ndarray, rgb_pts3d:np.ndarray, total_ctrlpts:int=10_000, degree_u: int = 3, degree_v: int = 3,
                              use_approximate: bool = False, centripetal: bool = False,
                              ctrlpts_size_u: Optional[int] = None,
                              ctrlpts_size_v: Optional[int] = None, device: str = 'cuda', merge_threshold=1e-6, eps: float = 1e-8, cameras=None) -> Dict[
    str, any]:
    """
    Fits a BSpline/NURBS surface from a point cloud (.ply file), handling parametrization via geomdl.

    Args:
        file_path: Path to .ply point cloud (vertices as points).
        size_u, size_v: Grid sizes for u/v (points must match size_u * size_v).
        degree_u, degree_v: Spline degrees.
        use_approximate: Use approximation (True) or interpolation (False).
        centripetal: Use centripetal parametrization.
        ctrlpts_size_u/v: Optional control grid sizes (for approximation compression).
        device, eps: For tensors and basis.

    Returns: Dict like load_nurbs_from_3dm: control_points [num_ctrl_u, num_ctrl_v, 4] (W=1 non-rational),
             knots_u/v, degrees, basis funcs. Assumes non-rational; extend for weights if needed.
    """
    # kwargs = {'centripetal': centripetal}
    try:
        surf, rgb_list, size_u, size_v = filter_points(pts3d, rgb_pts3d, total_ctrlpts, cameras=cameras, merge_threshold=merge_threshold,
                                                           use_approximate=False, centripetal=centripetal)
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError during NURBS fitting: {e}. Consider adding jitter or adjusting parameters.")
        print(f"Trying toggling cetnripetal to False")
        try:
            surf, rgb_list, size_u, size_v = filter_points(pts3d, rgb_pts3d, total_ctrlpts, cameras=cameras, merge_threshold=merge_threshold,
                                                       use_approximate=False, centripetal=centripetal)

        except Exception as e2:
            print(f"Second attempt also failed: {e2}. Aborting NURBS fitting.")
            surf, rgb_list, size_u, size_v = filter_points(pts3d, rgb_pts3d, total_ctrlpts // 2, cameras=cameras, merge_threshold=merge_threshold,
                                                           use_approximate=False, centripetal=centripetal)



    # if cameras is not None:
    #     surf, shuffled_indices, size_u, size_v = filter_points(points_list, cameras=cameras, use_approximate=use_approximate, centripetal=centripetal)
        # surf_rgb = fitting.interpolate_surface(pcd_rgb[shuffled_indices].tolist(), size_u, size_v, degree_u, degree_v, **kwargs)
    # else:
    #     # points_np = load_ply_points(file_path)
    #     if len(points_list) != size_u * size_v:
    #         raise ValueError(
    #             f"Point count {len(points_list)} != size_u * size_v = {size_u * size_v}; resample if unstructured.")
    #
    #     if use_approximate:
    #         if ctrlpts_size_u is not None: kwargs['ctrlpts_size_u'] = ctrlpts_size_u
    #         if ctrlpts_size_v is not None: kwargs['ctrlpts_size_v'] = ctrlpts_size_v
    #         surf = fitting.approximate_surface(points_list, size_u, size_v, degree_u, degree_v, **kwargs)
    #     else:
    #         surf = fitting.interpolate_surface(points_list, size_u, size_v, degree_u, degree_v, **kwargs)
    #
    # Step 4: Extract representation as tensors
    # Control points: From ctrlpts2d (list of u-rows, each v-list of [x,y,z])
    num_ctrl_u, num_ctrl_v = surf.ctrlpts_size_u, surf.ctrlpts_size_v
    control_points = torch.zeros((num_ctrl_u, num_ctrl_v, 4), dtype=torch.float32, device=device)
    for i in range(num_ctrl_u):
        for j in range(num_ctrl_v):
            pt = surf.ctrlpts2d[i][j]  # [x, y, z]
            control_points[i, j] = torch.tensor([pt[0], pt[1], pt[2], 1.0], device=device)  # Non-rational, W=1

    def basis_u_func(samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def basis_v_func(samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    # surf.transpose()
    return {
        'surf': [surf],
        'surf_rgb': [create_attribute_surface(surf, rgb_list)] if rgb_list is not None else None,
        'control_points': control_points,
        'rgb_pts': rgb_list,

        'degree_u': degree_u,
        'degree_v': degree_v,
        'is_rational': False,  # Default non-rational
        'size_u': size_u,
        'size_v':size_v
    }


def create_attribute_surface(geom_surf, attr_array:list):
    """
    Creates a corresponding BSpline surface for attributes (e.g., RGB) using the structure of the input geometric surface.

    Args:
        geom_surf (geomdl.BSpline.Surface): The fitted geometric surface providing the parametric structure.
        attr_array (list): Flat array of attributes, shape [size_u * size_v, 3] (e.g., RGB per data point).
        size_u (int): Number of data points along the u-direction.
        size_v (int): Number of data points along the v-direction.

    Returns:
        geomdl.BSpline.Surface: A new surface with control points set to interpolated attribute values.
    """
    size_u, size_v = geom_surf.ctrlpts_size_u, geom_surf.ctrlpts_size_v
    attr_array = np.array(attr_array)
    # Validate input
    # if attr_array.shape != (size_u * size_v, 3):
    #     raise ValueError(f"attr_array must be shape [{size_u * size_v}, 3]")

    # Create a new BSpline surface with the same parameters
    attr_surf = BSpline.Surface()
    attr_surf.degree_u = geom_surf.degree_u
    attr_surf.degree_v = geom_surf.degree_v
    attr_surf.ctrlpts_size_u = geom_surf.ctrlpts_size_u
    attr_surf.ctrlpts_size_v = geom_surf.ctrlpts_size_v
    attr_surf.knotvector_u = geom_surf.knotvector_u
    attr_surf.knotvector_v = geom_surf.knotvector_v
    attr_surf._pdim = geom_surf.pdimension


    attr_surf.ctrlpts2d = attr_array.reshape(-1, 3)[:size_u * size_v,...].reshape(size_u, size_v, 3).tolist()

    return attr_surf
def load_nurbs_from_3dm(file_path: str, device: str = 'cuda', eps: float = 1e-8):
    """
    Loads a NURBS surface from a .3dm file (single Patch entity).
    Validates knot lengths, standardizes U/V orientation, and returns Torch tensors.

    Args:
        file_path: Path to .3dm file.
        device: Torch device for tensors.
        eps: Epsilon for basis computation.

    Returns:
        Dict with:
            - 'control_points': torch.Tensor [num_ctrl_u, num_ctrl_v, 4] (weighted [X, Y, Z, W])
            - 'knots_u': torch.Tensor [len_knots_u]
            - 'knots_v': torch.Tensor [len_knots_v]
            - 'degree_u': int
            - 'degree_v': int
            - 'is_rational': bool
            - 'basis_u_func': Callable[[torch.Tensor], torch.Tensor] for U-basis (samples -> basis)
            - 'basis_v_func': Callable[[torch.Tensor], torch.Tensor] for V-basis
    """
    # Load .3dm file
    import rhino3dm
    model = rhino3dm.File3dm.Read(file_path)
    if model is None:
        raise ValueError(f"Failed to read {file_path}")

    # Find first NURBS surface (assume single Patch)
    surface = None
    for obj in model.Objects:
        geom = obj.Geometry
        if isinstance(geom, rhino3dm.NurbsSurface):
            surface = geom
            break
    if surface is None:
        raise ValueError("No NURBS surface found in .3dm file")

    # Extract degrees (order - 1)
    degree_u = surface.OrderU - 1
    degree_v = surface.OrderV - 1

    # Extract knot vectors (as lists, then to tensors)
    knots_u_list = [surface.KnotsU[i] for i in range(surface.KnotsU.Count)]
    knots_v_list = [surface.KnotsV[i] for i in range(surface.KnotsV.Count)]
    knots_u = torch.tensor(knots_u_list, dtype=torch.float32, device=device)
    knots_v = torch.tensor(knots_v_list, dtype=torch.float32, device=device)

    # Validate knot lengths
    num_ctrl_u = surface.Points.CountU
    num_ctrl_v = surface.Points.CountV
    expected_len_u = num_ctrl_u + degree_u + 1
    expected_len_v = num_ctrl_v + degree_v + 1
    if len(knots_u) != expected_len_u or len(knots_v) != expected_len_v:
        raise ValueError(
            f"Invalid knot lengths: U expected {expected_len_u}, got {len(knots_u)}; V expected {expected_len_v}, got {len(knots_v)}")

    # Extract control points (weighted if rational)
    is_rational = surface.IsRational
    control_points = torch.zeros((num_ctrl_u, num_ctrl_v, 4), dtype=torch.float32, device=device)
    for i in range(num_ctrl_u):
        for j in range(num_ctrl_v):
            pt = surface.Points.GetPoint(i, j)  # Returns Point4d (X, Y, Z, W)
            control_points[i, j] = torch.tensor([pt.X, pt.Y, pt.Z, pt.W], dtype=torch.float32, device=device)

    # Standardize U/V orientation: If num_ctrl_v > num_ctrl_u, transpose to make U primary
    if num_ctrl_v > num_ctrl_u:
        control_points = control_points.transpose(0, 1)  # Now [orig_v, orig_u, 4]
        knots_u, knots_v = knots_v, knots_u
        degree_u, degree_v = degree_v, degree_u
        num_ctrl_u, num_ctrl_v = num_ctrl_v, num_ctrl_u
        print(f"Transposed U/V for consistency: New U={num_ctrl_u}, V={num_ctrl_v}")

    # Basis functions as callables (using your differentiable basis)
    # def basis_u_func(samples: torch.Tensor) -> torch.Tensor:
    #     return _differentiable_bspline_basis(samples, knots_u, degree_u, device, eps)

    # def basis_v_func(samples: torch.Tensor) -> torch.Tensor:
    #     return _differentiable_bspline_basis(samples, knots_v, degree_v, device, eps)

    return {
        'control_points': control_points,  # [num_ctrl_u, num_ctrl_v, 4]; divide by W for positions if needed
        'knots_u': knots_u,
        'knots_v': knots_v,
        'degree_u': degree_u,
        'degree_v': degree_v,
        'is_rational': is_rational,
        # 'basis_u_func': basis_u_func,
        # 'basis_v_func': basis_v_func,
    }
def load_3dm_nurbs_grids(file_path, resample_if_non_uniform=False, new_H=80, new_W=80, dense_eval_res=200, up_vec=[0, 0, 1],
                         normalized_pcd=False):
    """
    Load a NURBS surface from a .3dm file, ensuring consistent U/V orientation and correct knot lengths.
    Resample to uniform knots if specified.

    Args:
        file_path (str): Path to the .3dm file.
        resample_if_non_uniform (bool): Whether to resample non-uniform knot surfaces.
        new_H (int): Number of control points in V direction (rows) for resampling.
        new_W (int): Number of control points in U direction (columns) for resampling.
        dense_eval_res (int): Resolution for dense evaluation grid during resampling.

    Returns:
        tuple: (ctrl_pts: torch.Tensor (H, W, 3), knot_u: torch.Tensor, knot_v: torch.Tensor).
    """
    # Read the 3DM file
    model = r3d.File3dm.Read(file_path)
    if not model.Objects:
        raise ValueError(f"No objects found in {file_path}")

    # Get the first object
    obj = model.Objects[0]
    geometry = obj.Geometry

    # Check if it's a Brep or NurbsSurface
    if isinstance(geometry, r3d.Brep):
        if not geometry.Faces:
            raise ValueError("Brep has no faces")
        surface = geometry.Faces[0].ToNurbsSurface()
        if surface is None:
            raise ValueError("Failed to convert Brep face to NurbsSurface")
    elif isinstance(geometry, r3d.NurbsSurface):
        surface = geometry
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    # Check if rational
    if surface.IsRational:
        print(f"Warning: Rational NURBS in {file_path}. Using homogeneous coordinates.")

    # Get domains and degrees
    domain_u = surface.Domain(0)
    domain_v = surface.Domain(1)
    min_u, max_u = domain_u.T0, domain_u.T1  # Use T0, T1
    min_v, max_v = domain_v.T0, domain_v.T1
    degree_u, degree_v = surface.Degree(0), surface.Degree(1)
    is_normalized_domain = (abs(min_u) < 1e-6 and abs(max_u - 1) < 1e-6 and
                            abs(min_v) < 1e-6 and abs(max_v - 1) < 1e-6)

    # Get control points
    H, W = surface.Points.CountV, surface.Points.CountU  # H=V, W=U
    ctrl_grid = np.zeros((H, W, 3))
    weights = np.ones((H, W, 1)) if surface.IsRational else None
    for i in range(H):
        for j in range(W):
            cp = surface.Points.GetControlPoint(j, i)  # Returns Point4d
            if surface.IsRational:
                w = cp.Weight
                ctrl_grid[i, j] = [cp.X, cp.Y, cp.Z]  # Homogeneous coords (w*X, w*Y, w*Z)
                weights[i, j] = w
            else:
                ctrl_grid[i, j] = [cp.X, cp.Y, cp.Z]

    # Get knots and normalize
    knots_u = np.array(list(surface.KnotsU))
    knots_v = np.array(list(surface.KnotsV))
    if not is_normalized_domain:
        knots_u = (knots_u - min_u) / (max_u - min_u)
        knots_v = (knots_v - min_v) / (max_v - min_v)

    # Check knot uniformity
    is_uniform = (np.allclose(np.diff(knots_u[degree_u:-(degree_u)]), 0, atol=1e-6) and
                  np.allclose(np.diff(knots_v[degree_v:-(degree_v)]), 0, atol=1e-6))

    # Fix knot vector by adding missing multiplicity
    def fix_knots(knots, num_ctrl, degree):
        expected_len = num_ctrl + degree + 1
        if len(knots) == num_ctrl + degree - 1:  # Rhino's compact form (e.g., 83 + 3 - 1 = 85)
            knots = np.concatenate([[0.0], knots, [1.0]])  # Add one 0 and one 1
        if len(knots) != expected_len:
            print(f"Knot length mismatch: got {len(knots)}, expected {expected_len}. Using uniform knots.")
            knots = make_clamped_uniform_knots(num_ctrl, degree)
        return knots

    knot_u = fix_knots(knots_u, W, degree_u)  # W=80, expect 84 knots
    knot_v = fix_knots(knots_v, H, degree_v)
    knot_u = torch.tensor(knot_u, dtype=torch.float32)
    knot_v = torch.tensor(knot_v, dtype=torch.float32)

    # Check surface orientation by evaluating normal at (u=0.5, v=0.5)
    u_mid, v_mid = (min_u + max_u) / 2, (min_v + max_v) / 2
    normal = surface.NormalAt(u_mid, v_mid)
    normal = np.array([normal.X, normal.Y, normal.Z])
    if (not is_uniform or not is_normalized_domain) and resample_if_non_uniform:
        print(f"Resampling surface in {file_path} (non-uniform: {not is_uniform}, domain: U[{min_u},{max_u}], V[{min_v},{max_v}])...")
        u_eval_dense = np.linspace(min_u, max_u, dense_eval_res)
        v_eval_dense = np.linspace(min_v, max_v, dense_eval_res)
        u_norm_dense = (u_eval_dense - min_u) / (max_u - min_u)
        v_norm_dense = (v_eval_dense - min_v) / (max_v - min_v)
        u_grid_norm, v_grid_norm = np.meshgrid(u_norm_dense, v_norm_dense, indexing='ij')
        u_flat_norm = u_grid_norm.ravel()
        v_flat_norm = v_grid_norm.ravel()

        x_dense = np.zeros((dense_eval_res, dense_eval_res))
        y_dense = np.zeros((dense_eval_res, dense_eval_res))
        z_dense = np.zeros((dense_eval_res, dense_eval_res))
        for i, u in enumerate(u_eval_dense):
            for j, v in enumerate(v_eval_dense):
                pt = surface.PointAt(u, v)
                x_dense[i, j], y_dense[i, j], z_dense[i, j] = pt.X, pt.Y, pt.Z

        if np.any(np.isnan(x_dense)) or np.any(np.isnan(y_dense)) or np.any(np.isnan(z_dense)):
            raise ValueError("NaN values in surface evaluation.")
        if np.std(x_dense) < 1e-6 or np.std(y_dense) < 1e-6 or np.std(z_dense) < 1e-6:
            print(f"Warning: Low variation (std: x={np.std(x_dense):.6f}, y={np.std(y_dense):.6f}, z={np.std(z_dense):.6f})")

        x_flat_dense = x_dense.ravel()
        y_flat_dense = y_dense.ravel()
        z_flat_dense = z_dense.ravel()

        knots_u_fit = make_clamped_uniform_knots(new_W, degree_u)
        knots_v_fit = make_clamped_uniform_knots(new_H, degree_v)
        try:
            tck_x = bisplrep(u_flat_norm, v_flat_norm, x_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                             task=-1, kx=degree_u, ky=degree_v, s=1.0)
            tck_y = bisplrep(u_flat_norm, v_flat_norm, y_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                             task=-1, kx=degree_u, ky=degree_v, s=1.0)
            tck_z = bisplrep(u_flat_norm, v_flat_norm, z_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                             task=-1, kx=degree_u, ky=degree_v, s=1.0)
        except ValueError as e:
            print(f"bisplrep failed: {e}. Knots: U={len(knots_u_fit)}, V={len(knots_v_fit)}")
            raise

        ctrl_x = tck_x[2].reshape(new_H, new_W)
        ctrl_y = tck_y[2].reshape(new_H, new_W)
        ctrl_z = tck_z[2].reshape(new_H, new_W)
        ctrl_grid = np.stack([ctrl_x, ctrl_y, ctrl_z], axis=-1)
        knot_u = torch.tensor(knots_u_fit, dtype=torch.float32)
        knot_v = torch.tensor(knots_v_fit, dtype=torch.float32)
        H, W = new_H, new_W

    # Normalize control points to [-1,1]
    scale = max(np.abs(ctrl_grid).max(), 1e-6)
    if normalized_pcd:
        ctrl_grid /= scale
    else:
        scale = 1.0 # to prevent accidental scaling
    ctrl_pts = torch.from_numpy(ctrl_grid).float().cuda()
    knot_u = knot_u.to('cuda')
    knot_v = knot_v.to('cuda')
    print(f"Loaded: ctrl_pts={ctrl_pts.shape}, knot_u={knot_u.shape}, knot_v={knot_v.shape}, flipped={normal[2] < 0}")
    return ctrl_pts, knot_u, knot_v, scale
def _load_3dm_nurbs_grids(file_path, resample_if_non_uniform=True, new_H=80, new_W=80, dense_eval_res=100):
    """
    Load a NURBS surface from a .3dm file and return its control points and knots.
    Resample to uniform knots if necessary using least-squares fit on dense evaluation.

    Args:
        file_path (str): Path to the .3dm file.
        resample_if_non_uniform (bool): Whether to resample non-uniform knot surfaces.
        new_H (int): Number of control points in V direction (rows) for resampling.
        new_W (int): Number of control points in U direction (columns) for resampling.
        dense_eval_res (int): Resolution for dense evaluation grid during resampling.

    Returns:
        tuple: (control_pts: torch.Tensor of shape (H, W, 3), knot_u: np.ndarray, knot_v: np.ndarray)
    """
    # Read the 3DM file
    model = r3d.File3dm.Read(file_path)
    if not model.Objects:
        raise ValueError(f"No objects found in {file_path}")

    # Get the first object
    obj = model.Objects[0]
    geometry = obj.Geometry

    # Check if it's a Brep or NurbsSurface
    if isinstance(geometry, r3d.Brep):
        if not geometry.Faces:
            raise ValueError("Brep has no faces")
        surface = geometry.Faces[0].ToNurbsSurface()
        if surface is None:
            raise ValueError("Failed to convert Brep face to NurbsSurface")
    elif isinstance(geometry, r3d.NurbsSurface):
        surface = geometry
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    # Check if rational and warn
    if surface.IsRational:
        print(f"Warning: Rational NURBS in {file_path}. Using homogeneous coordinates (w*X, w*Y, w*Z).")

    # Get original knots
    knots_u = np.array(list(surface.KnotsU))
    knots_v = np.array(list(surface.KnotsV))
    degree_u, degree_v = surface.Degree(0), surface.Degree(1)

    # Check knot uniformity
    is_uniform = (np.allclose(np.diff(knots_u[degree_u:-(degree_u)]), 0, atol=1e-6) and
                  np.allclose(np.diff(knots_v[degree_v:-(degree_v)]), 0, atol=1e-6))

    # Fix knot vector by adding missing multiplicity
    def fix_knots(knots, num_ctrl, degree):
        expected_len = num_ctrl + degree + 1
        if len(knots) == num_ctrl + degree - 1:  # Rhino's compact form
            # Add one 0 at start and one 1 at end to reach full multiplicity
            knots = np.concatenate([[0.0], knots, [1.0]])
        if len(knots) != expected_len:
            print(f"Knot length mismatch: got {len(knots)}, expected {expected_len}. Using uniform knots.")
            knots = make_clamped_uniform_knots(num_ctrl, degree)
        return knots

    if not is_uniform and resample_if_non_uniform:
        print(f"Non-uniform knots detected in {file_path}. Resampling with uniform knots...")
        # Dense evaluation grid
        u_eval_dense = np.linspace(0, 1, dense_eval_res)
        v_eval_dense = np.linspace(0, 1, dense_eval_res)
        u_grid_dense, v_grid_dense = np.meshgrid(u_eval_dense, v_eval_dense, indexing='ij')
        u_flat_dense = u_grid_dense.ravel()
        v_flat_dense = v_grid_dense.ravel()

        # Evaluate surface
        x_dense = np.zeros((dense_eval_res, dense_eval_res))
        y_dense = np.zeros((dense_eval_res, dense_eval_res))
        z_dense = np.zeros((dense_eval_res, dense_eval_res))
        for i, u in enumerate(u_eval_dense):
            for j, v in enumerate(v_eval_dense):
                pt = surface.PointAt(u, v)
                if pt is None or np.any(np.isnan([pt.X, pt.Y, pt.Z])) or np.any(np.isinf([pt.X, pt.Y, pt.Z])):
                    print(f"Warning: Invalid point at (u={u}, v={v}) in {file_path}. Using zero.")
                    x_dense[i, j], y_dense[i, j], z_dense[i, j] = 0, 0, 0
                else:
                    x_dense[i, j], y_dense[i, j], z_dense[i, j] = pt.X, pt.Y, pt.Z

        # Check for valid data
        if np.all(x_dense == 0) and np.all(y_dense == 0) and np.all(z_dense == 0):
            print(
                f"Error: All evaluated points are zero or invalid in {file_path}. Falling back to original control points.")
            is_uniform = True  # Skip resampling
        else:
            # Flatten data
            x_flat_dense = x_dense.ravel()
            y_flat_dense = y_dense.ravel()
            z_flat_dense = z_dense.ravel()

            # Generate clamped uniform knots
            knots_u_fit = make_clamped_uniform_knots(new_W, degree_u)  # W for U direction
            knots_v_fit = make_clamped_uniform_knots(new_H, degree_v)  # H for V direction

            # Verify knot lengths
            if len(knots_u_fit) != new_W + degree_u + 1 or len(knots_v_fit) != new_H + degree_v + 1:
                raise ValueError(
                    f"Invalid knot lengths: knots_u_fit={len(knots_u_fit)} (expected {new_W + degree_u + 1}), "
                    f"knots_v_fit={len(knots_v_fit)} (expected {new_H + degree_v + 1})")

            # Fit B-Splines with least-squares
            try:
                tck_x = bisplrep(u_flat_dense, v_flat_dense, x_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                                 task=-1, kx=degree_u, ky=degree_v, s=0.1)
                tck_y = bisplrep(u_flat_dense, v_flat_dense, y_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                                 task=-1, kx=degree_u, ky=degree_v, s=0.1)
                tck_z = bisplrep(u_flat_dense, v_flat_dense, z_flat_dense, tx=knots_u_fit, ty=knots_v_fit,
                                 task=-1, kx=degree_u, ky=degree_v, s=0.1)
            except ValueError as e:
                print(f"bisplrep failed: {e}. Falling back to original control points.")
                is_uniform = True  # Skip resampling
            else:
                # Extract control points from coefficients
                ctrl_x = tck_x[2].reshape(new_H, new_W)
                ctrl_y = tck_y[2].reshape(new_H, new_W)
                ctrl_z = tck_z[2].reshape(new_H, new_W)
                ctrl_grid = np.stack([ctrl_x, ctrl_y, ctrl_z], axis=-1)

    if is_uniform or not resample_if_non_uniform:
        # Use original control points
        H, W = surface.Points.CountV, surface.Points.CountU
        ctrl_grid = np.zeros((H, W, 3))
        for i in range(H):
            for j in range(W):
                pt = surface.Points.GetControlPoint(j, i)
                # pt = cp.Location
                # w = cp.Weight
                # Use homogeneous coords for rational surfaces
                # if surface.IsRational:
                #     ctrl_grid[i, j] = [pt.X * w, pt.Y * w, pt.Z * w]
                # else:
                ctrl_grid[i, j] = [pt.X, pt.Y, pt.Z]
        # knots_u_fit = knots_u
        # knots_v_fit = knots_v
        knot_u = fix_knots(knots_u, W, degree_u)
        knot_v = fix_knots(knots_v, H, degree_v)
        knot_u = torch.tensor(knot_u, dtype=torch.float32)
        knot_v = torch.tensor(knot_v, dtype=torch.float32)
    # Debug stats
    print(f"Control grid shape: {ctrl_grid.shape}, min: {ctrl_grid.min()}, max: {ctrl_grid.max()}")
    print(f"Knots U length: {len(knot_u)}, Knots V length: {len(knot_v)}")

    ctrl_pts_tensor = torch.from_numpy(ctrl_grid).float().cuda()
    return ctrl_pts_tensor, knot_u, knot_v
def load_3dm_nurbs_gridsa(file_path, resample_if_non_uniform=True, new_H=80, new_W=80):
    """
    Load a NURBS surface from a .3dm file and return its control points as a tensor.
    Resample to uniform knots if necessary.

    Args:
        file_path (str): Path to the .3dm file.
        resample_if_non_uniform (bool): Whether to resample non-uniform knot surfaces.
        new_H (int): Number of control points in U direction for resampling.
        new_W (int): Number of control points in V direction for resampling.

    Returns:
        torch.Tensor: Control points grid of shape (H, W, 3).
    """
    # Read the 3DM file
    model = r3d.File3dm.Read(file_path)
    if not model.Objects:
        raise ValueError(f"No objects found in {file_path}")

    # Get the first object
    obj = model.Objects[0]
    geometry = obj.Geometry

    # Check if it's a Brep or NurbsSurface
    if isinstance(geometry, r3d.Brep):
        if not geometry.Faces:
            raise ValueError("Brep has no faces")
        surface = geometry.Faces[0].ToNurbsSurface()
        if surface is None:
            raise ValueError("Failed to convert Brep face to NurbsSurface")
    elif isinstance(geometry, r3d.NurbsSurface):
        surface = geometry
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    # Check knot uniformity
    knots_u = np.array(list(surface.KnotsU))
    knots_v = np.array(list(surface.KnotsV))
    degree_u, degree_v = surface.Degree(0), surface.Degree(1)
    is_uniform = (np.allclose(np.diff(knots_u[degree_u:-(degree_u)]), 0, atol=1e-6) and
                  np.allclose(np.diff(knots_v[degree_v:-(degree_v)]), 0, atol=1e-6))

    if not is_uniform and resample_if_non_uniform:
        print(f"Non-uniform knots detected in {file_path}. Resampling surface...")
        # Create evaluation grid
        u_eval = np.linspace(0, 1, new_H)
        v_eval = np.linspace(0, 1, new_W)
        # Flatten the grid for bisplrep
        u_grid, v_grid = np.meshgrid(u_eval, v_eval, indexing='ij')
        u_flat = u_grid.ravel()  # Shape: (new_H * new_W,)
        v_flat = v_grid.ravel()  # Shape: (new_H * new_W,)
        x, y, z = np.zeros((new_H, new_W)), np.zeros((new_H, new_W)), np.zeros((new_H, new_W))
        for i, u in enumerate(u_eval):
            for j, v in enumerate(v_eval):
                pt = surface.PointAt(u, v)
                x[i, j], y[i, j], z[i, j] = pt.X, pt.Y, pt.Z

        # Flatten data for bisplrep
        x_flat = x.ravel()  # Shape: (new_H * new_W,)
        y_flat = y.ravel()
        z_flat = z.ravel()

        # Fit B-Splines for x, y, z
        tck_x = bisplrep(u_flat, v_flat, x_flat, kx=degree_u, ky=degree_v)
        tck_y = bisplrep(u_flat, v_flat, y_flat, kx=degree_u, ky=degree_v)
        tck_z = bisplrep(u_flat, v_flat, z_flat, kx=degree_u, ky=degree_v)

        # Evaluate to get control points on uniform grid
        x_ctrl = bisplev(u_eval, v_eval, tck_x)  # Shape: (new_H, new_W)
        y_ctrl = bisplev(u_eval, v_eval, tck_y)
        z_ctrl = bisplev(u_eval, v_eval, tck_z)

        # Stack into (new_H, new_W, 3)
        ctrl_grid = np.stack([x_ctrl, y_ctrl, z_ctrl], axis=-1)
    else:
        # Use original control points
        H, W = surface.Points.CountV, surface.Points.CountU
        ctrl_grid = np.zeros((H, W, 3))
        for i in range(H):
            for j in range(W):
                pt = surface.Points.GetPoint(j, i)
                ctrl_grid[i, j] = [pt.X, pt.Y, pt.Z]

    ctrl_pts_tensor = torch.from_numpy(ctrl_grid).float().cuda()
    return ctrl_pts_tensor, knots_u, knots_v

def load_3dm_nurbs_gridss(file_path):
    """
    Load a NURBS surface from a .3dm file and return its control points as a tensor.

    Args:
        file_path (str): Path to the .3dm file.

    Returns:
        torch.Tensor: Control points grid of shape (H, W, 3).
    """
    # Read the 3DM file
    model = r3d.File3dm.Read(file_path)
    if not model.Objects:
        raise ValueError(f"No objects found in {file_path}")

    # Get the first object
    obj = model.Objects[0]
    geometry = obj.Geometry

    # Check if it's a Brep or NurbsSurface
    if isinstance(geometry, r3d.Brep):
        if not geometry.Faces:
            raise ValueError("Brep has no faces")
        surface = geometry.Faces[0].ToNurbsSurface()
        if surface is None:
            raise ValueError("Failed to convert Brep face to NurbsSurface")
    elif isinstance(geometry, r3d.NurbsSurface):
        surface = geometry
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    # Get control points
    H, W = surface.Points.CountV, surface.Points.CountU  # Rhino: V is rows, U is columns
    ctrl_grid = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            pt = surface.Points.GetPoint(j, i)  # Note: (j,i) due to Rhino's U,V order
            ctrl_grid[i, j] = [pt.X, pt.Y, pt.Z]

    # Check knot uniformity (optional, for debugging)
    knots_u = np.array(list(surface.KnotsU))  # Convert KnotsU to list
    knots_v = np.array(list(surface.KnotsV))  # Convert KnotsV to list
    degree_u, degree_v = surface.Degree(0), surface.Degree(1)
    if not (np.allclose(np.diff(knots_u[degree_u:-(degree_u)]), 0, atol=1e-6) and
            np.allclose(np.diff(knots_v[degree_v:-(degree_v)]), 0, atol=1e-6)):
        print(f"Warning: Non-uniform knots detected in {file_path}. Consider resampling.")

    # Convert to torch tensor
    ctrl_pts_tensor = torch.from_numpy(ctrl_grid).float().cuda()
    return ctrl_pts_tensor

def load_3dm_nurbs_grid(file_path: str, device: str = "cuda"):
    """
    Load a .3dm file produced by Rhino3D, extract every NURBS surface *as‑is*
    (i.e. full control‑grid, NOT decomposed into 4×4 patches), and return a
    Python list of CUDA tensors, each shaped (U, V, 3).

    Parameters
    ----------
    file_path : str
        Path to the .3dm file.
    device : str
        Torch device on which to allocate the tensors.

    Returns
    -------
    List[torch.Tensor]
        One tensor per surface, shape (n_ctrl_u, n_ctrl_v, 3).  Returns an
        empty list if no valid surfaces are found.
    """
    model = rhino3dm.File3dm.Read(file_path)
    if model is None:
        print(f"[load_3dm_nurbs_grids] Failed to read {file_path}")
        return []

    surface_grids = []

    def _surface_to_tensor(srf: rhino3dm):
        """Convert any NurbsSurface to (U,V,3) torch tensor."""
        cnt_u, cnt_v = srf.Points.CountU, srf.Points.CountV
        if cnt_u == 0 or cnt_v == 0:
            return None
        grid = torch.zeros((cnt_u, cnt_v, 3), dtype=torch.float32, device=device)
        for i in range(cnt_u):
            for j in range(cnt_v):
                cp = srf.Points.GetControlPoint(i, j)
                grid[i, j, 0] = cp.X
                grid[i, j, 1] = cp.Y
                grid[i, j, 2] = cp.Z
        return grid

    # ------------------------------------------------------------------ scan --
    for obj in model.Objects:
        geo = obj.Geometry

        # Case 1: Brep – iterate faces
        if isinstance(geo, rhino3dm.Brep):
            for f_i in range(len(geo.Faces)):
                srf = geo.Faces[f_i].ToNurbsSurface()
                if srf is not None:
                    tensor = _surface_to_tensor(srf)
                    if tensor is not None:
                        surface_grids.append(tensor)

        # Case 2: stand‑alone NURBS surface
        elif isinstance(geo, rhino3dm.NurbsSurface):
            tensor = _surface_to_tensor(geo)
            if tensor is not None:
                surface_grids.append(tensor)

        # Else: ignore (curve, mesh, etc.)
    # ------------------------------------------------------------------------

    if len(surface_grids) == 0:
        print("[load_3dm_nurbs_grids] No NURBS surfaces found.")
    return torch.cat(surface_grids)



import numpy as np

def compute_bbox(points: np.ndarray,
                 trim_q: float = None):
    """
    Compute the axis-aligned bounding box of a point cloud.

    Parameters
    ----------
    points : (N, 3) ndarray
        Point positions in world units.
    trim_q : float in (0, 0.5) or None
        If given, discard points below the q-quantile
        and above the (1-q)-quantile **per axis** before
        computing the box.  E.g. trim_q=0.01 keeps the
        central 98 % of points along each axis.

    Returns
    -------
    bbox_min, bbox_max : 1-D arrays of length 3
        The lower-left-down and upper-right-up corners.
    """
    if trim_q is not None and 0.0 < trim_q < 0.5:
        lo = np.quantile(points, trim_q,  axis=0)
        hi = np.quantile(points, 1.0 - trim_q, axis=0)
        mask = np.all((points >= lo) & (points <= hi), axis=1)
        points = points[mask]          # keep in-liers only

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return bbox_min, bbox_max



def visual_comparison(gt_image, image, rend_normal, depthmap ,surf_normal, viewpoint_cam, title=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))  # Adjust the figure size as needed
    # Clamp images to the range [0, 1]
    # gt_image.clamp_(min=0, max=1)
    # image.clamp_(min=0, max=1)
    def prepare_image(img):
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            img = torch.clip(img, 0, 1)  # Clip values to [0, 1] range
        else:
            img = img.to(torch.float32) / 255.0  # Convert to float and normalize

        return img


    # Plot Ground Truth Image
    ax[0, 0].imshow(prepare_image(gt_image).cpu().detach().permute(1, 2, 0).numpy())
    ax[0, 0].set_title(f"Ground Truth (ID: {viewpoint_cam.uid})")
    ax[0, 0].axis('off')
    # Plot Splat Image
    ax[0, 1].imshow(prepare_image(image).cpu().detach().permute(1, 2, 0).numpy())
    ax[0, 1].set_title(f"Splat Image (ID: {viewpoint_cam.uid})")
    ax[0, 1].axis('off')
    # Plot Inferred Normals
    rend_normal = np.linalg.norm(prepare_image(rend_normal).cpu().detach().permute(1, 2, 0).numpy(), axis=2) * 0.5 + 0.5  # Normalize to [0, 1] range
    surf_normal = np.linalg.norm(prepare_image(surf_normal).cpu().detach().permute(1, 2, 0).numpy(), axis=2) * 0.5 + 0.5  # Normalize to [0, 1] range
    # rend_normal = rend_normal.cpu().detach().permute(1, 2, 0).numpy()
    depthmap = np.linalg.norm(depthmap.cpu().detach().permute(1, 2, 0).numpy(), axis=2) #* 0.5 + 0.5
    ax[1, 0].imshow(rend_normal, cmap='plasma')
    ax[1, 0].set_title(f"Inferred Normals (ID: {viewpoint_cam.uid})")
    ax[1, 0].axis('off')

    # Plot Depth Map
    ax[1, 1].imshow(depthmap, cmap='viridis')
    ax[1, 1].set_title(f"Surface normal (ID: {viewpoint_cam.uid})")
    ax[1, 1].axis('off')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def upsample_grid(grid: torch.Tensor, scale_factor: float = 2.0) -> torch.Tensor:
    """
    Upsample a grid by a given scale factor using bilinear interpolation.

    Args:
        grid (torch.Tensor): Input grid tensor of shape (H, W, C).
        scale_factor (float): Upsampling factor (default 2.0).

    Returns:
        torch.Tensor: Upsampled grid with shape ((H-1)*scale_factor+1, (W-1)*scale_factor+1, C).
    """
    H, W, C = grid.shape
    # Rearrange to (N, C, H, W) with N=1.
    grid_reshaped = grid.permute(2, 0, 1).unsqueeze(0)
    # Calculate the new spatial dimensions.
    H_new = int((H - 1) * scale_factor + 1)
    W_new = int((W - 1) * scale_factor + 1)
    # Use bilinear interpolation to upsample.
    upsampled = F.interpolate(grid_reshaped, size=(H_new, W_new), mode='bilinear', align_corners=True)
    # Rearrange back to (H_new, W_new, C).
    new_grid = upsampled.squeeze(0).permute(1, 2, 0)
    return new_grid




def generate_open_uniform_knot_vector(n_ctrl_pts: int, degree: int, device='cuda'):
    """
    n_ctrl_pts = number of control points in U (or V)
    degree     = spline degree (e.g. 3 for cubic)
    Returns    = tensor of shape (n_ctrl_pts + degree + 1,)
    """
    # interior spans count = n_ctrl_pts - degree + 1
    n_spans = n_ctrl_pts - degree + 1
    # start with degree zeros
    start = torch.zeros(degree, device=device)
    # then uniform from 0→1
    middle = torch.linspace(0, 1, steps=n_spans, device=device)
    # then degree ones
    end   = torch.ones(degree, device=device)
    return torch.cat([start, middle, end], dim=0)



def fit_bspline_surface(mesh_path, size_u=32, size_v=32):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = mesh.voxel_down_sample(voxel_size=0.005)
    mesh_points = np.asarray(mesh.vertices)
    # Project mesh vertices to structured grid (simplification for example)
    # points_2d = mesh_points[:, :2]
    xy = mesh_points[:, :2]  # XY plane
    # Bu = basis_functions(u, degree_u, knot_u)  # (N, size_u)
    # Bv = basis_functions(v, degree_v, knot_v)  # (N, size_v)

    # B = np.einsum('nu,nv->nuv', Bu, Bv).reshape(num_pts, size_u * size_v)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)

    uv = (xy - min_xy) / (max_xy - min_xy + 1e-8)  #
    z_vals = mesh_points[:, 2]

    # Generate grid points
    surf = fitting.approximate_surface(
        points=np.column_stack((uv, z_vals)),
        size_u=size_u,
        size_v=size_v,
        # centripetal=True,
        degree_u=3,
        degree_v=3
    )

    surf.vis = VisMPL.VisSurface()
    surf.render()

    return surf
def make_clamped_uniform_knots2(n_ctrl: int, degree: int, device='cuda'):
    """
    n_ctrl = number of control points in that direction (e.g. 4)
    degree = spline degree   (e.g. 3)
    returns U of shape (n_ctrl + degree + 1,)
    """
    # inner spans:
    n_spans = n_ctrl - degree
    # uniform between 0 and 1:
    inner = torch.linspace(0, 1, n_spans + 1, device=device)
    # repeat 0 and 1 degree times at ends:
    start = torch.zeros(degree, device=device)
    end   = torch.ones(degree, device=device)
    return torch.cat([start, inner, end], dim=0)  # shape (n_ctrl+degree+1,)

def cox_de_boor_batch(knots: torch.Tensor, degree: int, us: torch.Tensor):
    """
    knots: (N, K)  each row a knot vector U_i
    degree: scalar p
    us:    (N, D)  sample positions in [u0, uK]
    → returns B of shape (N, D, n_ctrl) where n_ctrl = K - p - 1
    """
    N, K = knots.shape
    _, D = us.shape
    device, dtype = knots.device, knots.dtype

    # pad so we can slice U[j+p+1] safely:
    pad = knots[:, -1:].repeat(1, degree)
    U   = torch.cat([knots, pad], dim=1)         # shape (N, K+degree)

    # flatten so we can do everything in one go:
    u_flat = us.reshape(-1)                      # (N*D,)
    PD = u_flat.shape[0]

    # build 0-th degree basis: indicator of [U[i],U[i+1])
    U_left  = U[:, :-1-degree].unsqueeze(1)      # (N,1, K-1-degree)
    U_right = U[:,  1:-degree ].unsqueeze(1)     # (N,1, K-1-degree)
    u_rep   = u_flat.view(N, D, 1)               # (N, D, 1)
    Nmat = ((u_rep >= U_left) & (u_rep < U_right)).float()  # (N, D, M0)

    # handle exact match at very end:
    # handle exact-match at the very end:
    end_mask = (u_rep.squeeze(-1) == knots[:, -1:])  # (N, D)
    if end_mask.any():
        # get the patch‐index (i) and sample‐index (j) for every True
        idx0, idx1 = end_mask.nonzero(as_tuple=True)  # each is length K
        # now explicitly set the last basis value for those (i,j) pairs
        Nmat[idx0, idx1, -1] = 1.0

    # recursion to degree p
    # Track current number of nonzero basis per row (initially M0)
    M = Nmat.shape[-1]
    for r in range(1, degree + 1):
        # previous basis count
        M_prev = M
        M = M_prev - 1
        oldN = Nmat  # shape (N, D, M_prev)

        # left and right denominator terms
        U_left_vals  = U[:, :M_prev]                          # (N, M_prev)
        Den1 = U[:, r : r + M_prev] - U_left_vals             # (N, M_prev)
        Den2 = U[:, r+1 : r+1 + M_prev] - U[:, 1 : 1 + M_prev]# (N, M_prev)

        # compute coefficients, shape (N, D, M_prev)
        a = (u_rep - U_left_vals.unsqueeze(1)) / Den1.unsqueeze(1).clamp(min=1e-8)
        b = (U[:, r+1 : r+1 + M_prev].unsqueeze(1) - u_rep) / Den2.unsqueeze(1).clamp(min=1e-8)

        # update basis: combine Nmat[i,j,k] and Nmat[i,j,k+1]
        # Nmat = a * Nmat + b * Nmat[..., 1:M_prev+1]
        Nmat = a[..., :M] * oldN[..., :M] + b[..., :M] * oldN[..., 1:M_prev]
        # now drop last column to reduce to (N, D, M)
        Nmat = Nmat[..., :M]

    # final #ctrl points = K - p - 1
    n_ctrl = K - degree - 1

    return Nmat[:, :, :n_ctrl] # (N, D, n_ctrl)

def cox_de_boor_derivative_batch(knots: torch.Tensor, degree: int, us: torch.Tensor) -> torch.Tensor:
    """
    Compute the first derivative of B-spline basis functions for each patch.
    Given knot vectors (N, K), degree p, and sample points us (N, D),
    returns dB of shape (N, D, n_ctrl) where n_ctrl = K - p - 1.
    Uses the identity:
        d/du N_{i,p}(u) = p / (u_{i+p} - u_i) * N_{i,p-1}(u)
                         - p / (u_{i+p+1} - u_{i+1}) * N_{i+1,p-1}(u)
    """
    # First compute the degree-(p-1) basis
    Bm1 = cox_de_boor_batch(knots, degree - 1, us)  # shape (N, D, n_ctrl_m1)
    N, D, _ = Bm1.shape
    # Number of control points
    n_ctrl = knots.shape[1] - degree - 1
    # Prepare output
    dB = torch.zeros((N, D, n_ctrl), device=knots.device, dtype=knots.dtype)
    # Compute denominators
    # For term1: u_{i+p} - u_i, sized (N, n_ctrl)
    denom1 = knots[:, degree:degree + n_ctrl] - knots[:, :n_ctrl]
    # For term2: u_{i+p+1} - u_{i+1}
    denom2 = knots[:, degree + 1:degree + 1 + n_ctrl] - knots[:, 1:1 + n_ctrl]
    # Compute coefficients, avoiding zero division
    coeff = float(degree)
    inv_denom1 = (coeff / denom1.clamp(min=1e-8)).unsqueeze(1)  # (N,1,n_ctrl)
    inv_denom2 = (coeff / denom2.clamp(min=1e-8)).unsqueeze(1)  # (N,1,n_ctrl)
    # term1: coeff/denom1 * B_{i,p-1}
    term1 = inv_denom1 * Bm1[..., :n_ctrl]
    # term2: coeff/denom2 * B_{i+1,p-1}
    term2 = inv_denom2 * Bm1[..., 1:n_ctrl + 1]
    # derivative
    dB = term1 - term2
    return dB

def generate_bspline_surface_grid(num_cells: int, cell_density: int = 4, device: str = 'cuda',
                                  requires_grad: bool = True, delta=0.25):
    """
    Generate a grid of points on the B-spline surface using batched operations.

    Args:
        num_cells (int): Number of patches/cells.
        cell_density (int): Number of points per cell (grid resolution).
        device (str): Device on which the tensors are allocated.
        requires_grad (bool): Whether the returned parameters require gradients.

    Returns:
        Tuple[nn.Parameter, nn.Parameter]: Parameters for the u and v coordinates.
    """
    u = torch.linspace(delta, 1-delta/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    v = torch.linspace(delta, 1-delta/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    # u = torch.linspace(0, 1-1/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    # v = torch.linspace(0, 1-1/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    u = u.repeat(num_cells, 1)
    v = v.repeat(num_cells, 1)
    return u, v

def generate_global_grid(patch_res: int = 4, device: str = 'cuda',
                        num_patches_U=80, num_patches_V=80,requires_grad: bool = False, delta=0.25,):
    """
    Generate a grid of points on the B-spline surface using batched operations.

    Args:
        num_cells (int): Number of patches/cells.
        cell_density (int): Number of points per cell (grid resolution).
        device (str): Device on which the tensors are allocated.
        requires_grad (bool): Whether the returned parameters require gradients.

    Returns:
        Tuple[nn.Parameter, nn.Parameter]: Parameters for the u and v coordinates.
    """
    U_res = patch_res * num_patches_U  # e.g., 4 patches in u, each res=4, U_res=16
    V_res = patch_res * num_patches_V  # similarly for v

    us_global = torch.linspace(0, 1, U_res, device=device, requires_grad=True)
    vs_global = torch.linspace(0, 1, V_res, device=device, requires_grad=True)

    # u = torch.linspace(delta, 1-delta/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    # v = torch.linspace(delta, 1-delta/cell_density, cell_density, device=device, dtype=torch.float32, requires_grad=requires_grad)
    # u = u.repeat(num_cells, 1)
    # v = v.repeat(num_cells, 1)
    return us_global, vs_global

def basis_eval(u, device='cuda'):
    M_bezier = torch.tensor([
        [-1., 3., -3., 1.],
        [3., -6., 3., 0.],
        [-3., 3., 0., 0.],
        [1., 0., 0., 0.]
    ], device=device) / 6  # **no division by 6**

    U = torch.stack([u ** 3, u ** 2, u, torch.ones_like(u)], dim=-1)  # (N,R,4)
    return U @ M_bezier.t()  # (N,R,4)


def basis_eval_derivative(u, device='cuda'):
    M_bezier = torch.tensor([
        [-1., 3., -3., 1.],
        [3., -6., 3., 0.],
        [-3., 3., 0., 0.],
        [1., 0., 0., 0.]
    ], device=device) / 6

    dU = torch.stack([3*u**2, 2*u, torch.ones_like(u), torch.full_like(u, fill_value=0.0)], dim=-1)
    return dU @ M_bezier.t()
    # U = torch.stack([u ** 3, u ** 2, u, torch.ones_like(u)], dim=-1)  # (N,R,4)
    # return torch.matmul(dU, M_bezier)


class BasisInAxis(NamedTuple):
    bu : torch.Tensor
    db : torch.Tensor = None
    d2b : torch.Tensor = None

class BasisUV(NamedTuple):
    b : torch.Tensor = None
    db : torch.Tensor = None
    d2b : torch.Tensor = None

class BasisFuncs2:
    bu : torch.Tensor
    dbu : torch.Tensor
    dbuu : torch.Tensor
    bv : torch.Tensor
    dbv : torch.Tensor
    dbvv : torch.Tensor
    def __init__(self, *args, **kwargs):
        self.bu   = kwargs.get('bu', None)
        self.dbu  = kwargs.get('dbu', None)
        self.dbuu = kwargs.get('dbuu', None)
        self.bv   = kwargs.get('bv', None)
        self.dbv  = kwargs.get('dbv', None)
        self.dbvv = kwargs.get('dbvv', None)


    def set_bu(self, u_basis):
        self.bu = u_basis[0]
        self.dbu = u_basis[1]
        self.dbuu = u_basis[2]

    def set_bv(self, v_basis):
        self.bv = v_basis[0]
        self.dbv = v_basis[1]
        self.dbvv = v_basis[2]

    def clear(self):
        self.bu = None
        self.dbu = None
        self.dbuu = None
        self.bv = None
        self.dbv = None
        self.dbvv = None



class BasisFuncs(nn.Module):
    def __init__(
            self,
            bu_data=None,  # Can be dict with 'basis', 'deriv1', 'deriv2'
            bv_data=None,  # Can be dict with 'basis', 'deriv1', 'deriv2'
    ):
        super(BasisFuncs, self).__init__()
        if bu_data is not None and bv_data is not None:
            self.bu, self.bv = bu_data['basis'], bv_data['basis']
            self.dbu, self.dbv = bu_data['deriv1'], bv_data['deriv1']
            self.dbuu, self.dbvv = bu_data['deriv2'], bv_data['deriv2']
        else:
            self.bu = None
            self.dbu = None
            self.dbuu = None
            self.bv = None
            self.dbv = None
            self.dbvv = None
    def update_basis(self, bu_data, bv_data):
        """Update basis with new data."""
        if isinstance(bu_data, dict):
            self.bu, self.bv = bu_data['basis']
            self.dbu, self.dbv = bu_data['deriv1']
            self.dbuu, self.dbvv = bu_data['deriv2']
        else:
            self.bu = bu_data
            self.bv = bv_data

    def clear(self):
        self.bu = None
        self.dbu = None
        self.dbuu = None
        self.bv = None
        self.dbv = None
        self.dbvv = None