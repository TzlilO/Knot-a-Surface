import gc
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from utils.sh_utils import SH2RGB

import torch
import math
import torch.nn.functional as F
def grid_densification(
    grid: torch.Tensor,
    scale_fac_u: float = 1.0,
    scale_fac_v: float = 1.0,
    reps: int = 1,
    patch_stride: int = 1
) -> torch.Tensor:
    """
    Upsamples a 2D grid tensor, ensuring the output dimensions are
    compliant with B-Spline patch specifications.

    Args:
        grid (torch.Tensor): Tensor of shape (H, W, D).
        reps (int): Number of times to apply the scaling. Total scale is scale_factor**reps.
        scale_factor (float): The upsampling factor for each repetition.
        patch_stride (int): The stride of the B-Spline patches. The output
                            dimension 'dim' will satisfy (dim - 1) % patch_stride == 0.

    Returns:
        torch.Tensor: The upsampled and resized grid.
    """
    if scale_fac_u == 1 and scale_fac_v == 1:
        return grid

    H_old, W_old, D = grid.shape

    # 1. Calculate the total effective scale factor
    total_scale_u = scale_fac_u ** reps
    total_scale_v = scale_fac_v ** reps

    # 2. Calculate the ideal, uncorrected target size
    # This can result in a float
    H_target = (H_old - 1) * total_scale_u + 1
    W_target = (W_old - 1) * total_scale_v + 1

    # 3. Adjust the target size to the next valid B-Spline dimension
    # We find the smallest multiple of patch_stride that is >= (target_size - 1)
    # This is done using the ceiling function.
    if patch_stride > 0:
        H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1
        W_final = math.ceil((W_target - 1) / patch_stride) * patch_stride + 1
    else:
        H_final = int(H_target)
        W_final = int(W_target)

    # 4. Perform a single, direct interpolation to the final valid size
    grid_reshaped = grid.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    upsampled = F.interpolate(
        grid_reshaped,
        size=(H_final, W_final),
        mode='bilinear',
        align_corners=True
    )

    new_grid = upsampled.squeeze(0).permute(1, 2, 0)
    return new_grid

import math
import torch
import torch.nn.functional as F
import math
import torch
import torch.nn.functional as F

def insert_knot(
    grid: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    t: float,
    direction: str = 'u'
):
    """
    Inserts a single knot into the B-spline control grid and knot vector in the specified direction.

    Args:
        grid (torch.Tensor): Control point grid of shape (H, W, D).
        knots (torch.Tensor): Knot vector for the insertion direction.
        degree (int): Degree of the B-spline.
        t (float): The knot value to insert.
        direction (str): 'u' for rows (H), 'v' for columns (W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated grid and updated knot vector.
    """
    if direction == 'v':
        grid = grid.transpose(0, 1)  # Treat V as the primary dimension

    H, W, C = grid.shape
    p = degree

    # Find the span k where knots[k] <= t < knots[k+1]
    for kk in range(len(knots) - 1):
        if knots[kk] <= t < knots[kk + 1]:
            k = kk
            break
    else:
        raise ValueError(f"Knot value {t} is not within any interval of the knot vector.")

    # Create new knot vector
    new_knots = torch.zeros(len(knots) + 1, device=knots.device, dtype=knots.dtype)
    new_knots[:k + 1] = knots[:k + 1]
    new_knots[k + 1] = t
    new_knots[k + 2:] = knots[k + 1:]

    # Create new control grid
    new_grid = torch.zeros(H + 1, W, C, device=grid.device, dtype=grid.dtype)

    # Left unchanged part: Q[i] = P[i] for i = 0 to k - p
    left_unchanged_end = k - p
    if left_unchanged_end >= 0:
        new_grid[:left_unchanged_end + 1] = grid[:left_unchanged_end + 1]

    # Right unchanged part: Q[i] = P[i - 1] for i = k + 1 to H
    right_start = k + 1
    new_grid[right_start:H + 1] = grid[right_start - 1:H]

    # Middle part: Q[i] = alpha * P[i] + (1 - alpha) * P[i - 1] for i = k - p + 1 to k
    for ii in range(max(k - p + 1, 0), k + 1):
        u_i = knots[ii]
        u_ip = knots[ii + p]
        denom = u_ip - u_i
        alpha = (t - u_i) / denom if denom != 0 else 0.5
        new_grid[ii] = alpha * grid[ii] + (1 - alpha) * grid[ii - 1]

    if direction == 'v':
        new_grid = new_grid.transpose(0, 1)

    return new_grid, new_knots


def insert_knot2(
    grid: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    t: float,
    direction: str = 'u'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inserts a single knot into the B-spline control grid and knot vector in the specified direction.

    Args:
        grid (torch.Tensor): Control point grid of shape (H, W, D).
        knots (torch.Tensor): Knot vector for the insertion direction.
        degree (int): Degree of the B-spline.
        t (float): The knot value to insert.
        direction (str): 'u' for rows (H), 'v' for columns (W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated grid and updated knot vector.
    """
    if direction == 'v':
        grid = grid.transpose(0, 1)  # Treat V as the primary dimension

    H, W, D = grid.shape
    p = degree

    # Find the span k where knots[k] <= t < knots[k+1]
    for kk in range(len(knots) - 1):
        if knots[kk] <= t < knots[kk + 1]:
            k = kk
            break
    else:
        raise ValueError(f"Knot value {t} is not within any interval of the knot vector.")

    # Create new knot vector
    new_knots = torch.zeros(len(knots) + 1, device=knots.device, dtype=knots.dtype)
    new_knots[:k + 1] = knots[:k + 1]
    new_knots[k + 1] = t
    new_knots[k + 2:] = knots[k + 1:]

    # Create new control grid
    new_grid = torch.zeros(H + 1, W, D, device=grid.device, dtype=grid.dtype)

    # Left unchanged part: Q[i] = P[i] for i = 0 to k - p
    left_unchanged_end = k - p
    if left_unchanged_end >= 0:
        new_grid[:left_unchanged_end + 1] = grid[:left_unchanged_end + 1]

    # Right unchanged part: Q[i] = P[i - 1] for i = k + 1 to H
    right_start = k + 1
    new_grid[right_start:H + 1] = grid[right_start - 1:H]

    # Middle part: Q[i] = alpha * P[i] + (1 - alpha) * P[i - 1] for i = k - p + 1 to k
    for ii in range(max(k - p + 1, 0), k + 1):
        u_i = knots[ii]
        u_ip = knots[ii + p]
        denom = u_ip - u_i
        alpha = (t - u_i) / denom if denom != 0 else 0.5
        new_grid[ii] = alpha * grid[ii] + (1 - alpha) * grid[ii - 1]

    if direction == 'v':
        new_grid = new_grid.transpose(0, 1)

    return new_grid, new_knots


import torch
import torch.nn.functional as F
import math


def grid_upscale_separate(
        grid: torch.Tensor,
        reps: int = 1,
        scale_factor: float = 2.0,
        patch_stride: int = 1,
        mode: str = 'upscale',
        knots_u: torch.Tensor = None,
        knots_v: torch.Tensor = None,
        degree_u: int = 3,
        degree_v: int = 3,
):
    """
    Resizes a 2D grid tensor. Supports upscaling (via interpolation or knot insertion)
    and downscaling (via adaptive knot removal/resampling).

    Args:
        grid (torch.Tensor): Tensor of shape (H, W, D).
        reps (int): Number of times to apply the scaling.
        scale_factor (float): Factor > 1.0 for upscaling, < 1.0 for downscaling.
        patch_stride (int): The stride of the B-Spline patches.
        mode (str): 'upscale' (standard), 'squarify' (equalize dims).
        knots_u (torch.Tensor): Knot vector for U dimension.
        knots_v (torch.Tensor): Knot vector for V dimension.
        degree_u (int): Degree for U dimension.
        degree_v (int): Degree for V dimension.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        Updated grid, knots_u, and knots_v.
    """

    # --- Helper: Adaptive Knot Resampling (Vectorized) ---
    def resample_knots_and_indices(
            knots: torch.Tensor,
            grid_slice: torch.Tensor,
            target_dim: int,
            degree: int
    ) -> torch.Tensor:
        """
        Generates a new knot vector of size (target_dim + degree + 1) based on
        geometric importance (curvature) of the grid slice.
        """
        # 1. Calculate Geometric Metric (Second derivative / curvature approximation)
        # grid_slice shape: (N, D) or (N, W, D) flattened. We want metric per row N.
        # We compute the L2 norm of the discrete Laplacian along the scaling dimension.

        if grid_slice.dim() == 3:  # (H, W, D) -> processing H
            # Average over W to get a single profile, then compute derivatives
            profile = grid_slice.mean(dim=1)  # (H, D)
        else:
            profile = grid_slice  # (W, D)

        # Central difference for 2nd derivative (Laplacian): P_{i+1} - 2P_i + P_{i-1}
        # Padding to keep shape consistent
        laplacian = torch.zeros_like(profile)
        laplacian[1:-1] = profile[:-2] - 2 * profile[1:-1] + profile[2:]
        laplacian[0] = laplacian[1]  # Border approximation
        laplacian[-1] = laplacian[-2]

        # Metric: Magnitude of curvature + small epsilon for flat regions
        # We also weight by current knot interval to avoid preserving bunched knots (divergence sources)
        # diffs: (N+p) -> we need mapping to N control points. Approximate with centered average.

        curvature = torch.norm(laplacian, dim=-1)  # (N,)

        # 2. Importance Distribution (PDF)
        # We want more knots where curvature is high.
        weights = curvature + 1e-6  # Add epsilon to avoid zero division in flat areas

        # Normalize to probability distribution
        pdf = weights / weights.sum()

        # 3. Inverse Transform Sampling (CDF) to pick new knot locations
        # This effectively removes knots from low-importance (flat) areas and keeps them in high-curvature areas.
        cdf = torch.cumsum(pdf, dim=0)
        cdf = cdf / cdf[-1]  # Ensure 0 to 1

        # Generate uniform sample points
        # We need (target_dim - degree + 1) internal spans, but simplified:
        # We simply need to find the indices in the old grid that map to the new grid.
        # However, for knots, we map the continuous value range.

        # Valid knot range (internal domain)
        min_k, max_k = knots[degree], knots[-degree - 1]
        valid_span = max_k - min_k

        # Uniform steps in CDF space map to Adaptive steps in Knot space
        uniform_steps = torch.linspace(0, 1, target_dim - degree + 1, device=knots.device)

        # Searchsorted finds indices where uniform steps fall into the CDF
        indices = torch.searchsorted(cdf, uniform_steps)
        indices = torch.clamp(indices, 0, len(knots) - 1)

        # Interpolate new internal knots from old knots based on these indices
        # (A simplified approach to getting new knot values)
        new_internal_knots = knots[indices + degree]  # Offset by degree to hit internal domain

        # Clamp to ensure strict range compliance
        new_internal_knots = torch.clamp(new_internal_knots, min_k, max_k)

        # Reconstruct full knot vector with open-uniform ends (clamped)
        new_knots = torch.cat([
            torch.full((degree,), min_k, device=knots.device),
            new_internal_knots,
            torch.full((degree,), max_k, device=knots.device)
        ])

        # Ensure strict monotonicity (fix potential numerical clusters)
        new_knots, _ = torch.sort(new_knots)

        return new_knots

    # --- Main Logic ---

    H_old, W_old, D = grid.shape
    total_scale = scale_factor ** reps

    # Calculate target sizes (raw)
    H_target_raw = (H_old - 1) * total_scale + 1
    W_target_raw = (W_old - 1) * total_scale + 1

    # Apply Patch Stride Constraints
    if patch_stride > 0:
        H_final = math.ceil((H_target_raw - 1) / patch_stride) * patch_stride + 1
        W_final = math.ceil((W_target_raw - 1) / patch_stride) * patch_stride + 1
    else:
        H_final = int(H_target_raw)
        W_final = int(W_target_raw)

    # Prevent dimension collapse (min size is usually degree + 1)
    min_u = degree_u + 1 if degree_u else 2
    min_v = degree_v + 1 if degree_v else 2
    H_final = max(H_final, min_u)
    W_final = max(W_final, min_v)

    # ---------------------------------------------------------
    # BRANCH 1: Downscaling (Knot Removal / Resampling)
    # ---------------------------------------------------------
    if scale_factor < 1.0:
        if knots_u is None or knots_v is None:
            raise ValueError("knots_u/v required for downscaling operation to adjust topology.")

        # 1. Update U-Dimension (Height)
        # Use 'area' interpolation for Grid (better for downscaling/anti-aliasing)
        # We perform it in two steps (U then V) to allow separate knot analysis.

        # Permute for Interpolate: (Batch, Channel, Height, Width)
        # Treat W as Batch or Channel?
        # F.interpolate takes (N, C, L) for 1D or (N, C, H, W) for 2D.

        # --- Process U (Height) ---
        # Generate new knots based on metric (High curvature rows kept)
        knots_u_new = resample_knots_and_indices(knots_u, grid, H_final, degree_u)

        # Resize Grid along H
        grid_perm = grid.permute(1, 2, 0)  # (W, D, H)
        grid_perm = F.interpolate(
            grid_perm,
            size=H_final,
            mode='linear',  # Linear often safer for control points than area to preserve bounds
            align_corners=True
        )
        grid_u = grid_perm.permute(2, 0, 1)  # (H_final, W, D)

        # --- Process V (Width) ---
        # Generate new knots based on metric (High curvature cols kept)
        # Pass permuted grid (W, H, D) so the function sees 'W' as the first dim
        knots_v_new = resample_knots_and_indices(knots_v, grid_u.permute(1, 0, 2), W_final, degree_v)

        # Resize Grid along W
        grid_perm = grid_u.permute(0, 2, 1)  # (H_final, D, W)
        grid_perm = F.interpolate(
            grid_perm,
            size=W_final,
            mode='linear',
            align_corners=True
        )
        grid_final = grid_perm.permute(0, 2, 1)  # (H_final, W_final, D)

        return grid_final, knots_u_new, knots_v_new

    # ---------------------------------------------------------
    # BRANCH 2: Upscaling (Interpolation or Knot Insertion)
    # ---------------------------------------------------------

    # Validation for Knot Insertion
    if (knots_u is None or knots_v is None or degree_u is None or degree_v is None):
        if mode == 'squarify':
            raise ValueError("knots and degrees required for squarify mode.")
        # Fallback to pure interpolation (Existing logic)
        # ... [Your existing interpolation code here] ...
        # (Included simplified version for completeness of the function block)
        grid_u = grid.permute(1, 2, 0).reshape(W_old * D, H_old).unsqueeze(0)
        upsampled_u = F.interpolate(grid_u, size=H_final, mode='linear', align_corners=True)
        upsampled_u = upsampled_u.squeeze(0).view(W_old, D, H_final).permute(2, 0, 1)

        grid_v = upsampled_u.permute(0, 2, 1).reshape(H_final * D, W_old).unsqueeze(0)
        upsampled_v = F.interpolate(grid_v, size=W_final, mode='linear', align_corners=True)
        new_grid = upsampled_v.squeeze(0).view(H_final, D, W_final).permute(0, 2, 1)
        return new_grid, knots_u, knots_v

    # Knot insertion approach (Existing logic adapted)
    H, W = H_old, W_old

    if mode == 'squarify':
        target_u = target_v = max(H_final, W_final)
    else:  # Upscale
        target_u = H_final
        target_v = W_final

    # Note: This loop is retained for 'squarify' / precise insertion as requested,
    # but strictly speaking, for pure upscaling, F.interpolate + uniform knot refinement
    # is often preferred in optimization to avoid topological noise.
    # Assuming 'insert_knot' is defined externally or in context.

    curr_grid = grid
    curr_knots_u = knots_u
    curr_knots_v = knots_v

    # Safety break to prevent infinite loops if target unreach
    max_iter = (target_u - H) + (target_v - W) + 10

    count = 0
    while (H < target_u or W < target_v) and count < max_iter:
        count += 1
        # Decision logic: Insert in U or V?
        if H < target_u and (W >= target_v or H <= W):
            direction = 'u'
            knots = curr_knots_u
            degree = degree_u
        elif W < target_v:
            direction = 'v'
            knots = curr_knots_v
            degree = degree_v
        else:
            break

        # Vectorized Span Selection
        diffs = knots[1:] - knots[:-1]
        seconddiffs = diffs[1:] - diffs[:-1]
        abs_sd = torch.abs(seconddiffs)

        if abs_sd.max() < 1e-8:
            insert_interval = torch.argmax(diffs).item()
        else:
            max_sd_idx = torch.argmax(abs_sd).item()
            sd = seconddiffs[max_sd_idx]
            insert_interval = max_sd_idx + 1 if sd > 0 else max_sd_idx

        # Calculate insertion t
        t = (knots[insert_interval] + knots[insert_interval + 1]) / 2.0

        # Call external insert_knot (assumed available from context)
        # If not available, one would use matrix multiplication here.
        curr_grid, new_knots = insert_knot(curr_grid, knots, degree, t.item(), direction)

        if direction == 'u':
            curr_knots_u = new_knots
            H += 1
        else:
            curr_knots_v = new_knots
            W += 1

    return curr_grid, curr_knots_u, curr_knots_v
def grid_upscale_separate2(
    grid: torch.Tensor,
    reps: int = 1,
    scale_factor: float = 2.0,
    patch_stride: int = 3,
    mode: str = 'upscale',
    knots_u: torch.Tensor = None,
    knots_v: torch.Tensor = None,
    degree_u: int = 3,
    degree_v: int = 3,
):
    """
    Upsamples a 2D grid tensor in two separate steps: first along the U dimension (height),
    then along the V dimension (width), using linear interpolation. Ensures the output
    dimensions are compliant with B-Spline patch specifications.

    In 'squarify' mode, gradually inserts knots in the smaller span in a greedy approach
    until the two dims are equally sized. Knots are inserted where knot spacings change abruptly.

    Args:
        grid (torch.Tensor): Tensor of shape (H, W, D).
        reps (int): Number of times to apply the scaling in 'upscale' mode. Total scale is scale_factor**reps.
        scale_factor (float): The upsampling factor for each repetition in 'upscale' mode.
        patch_stride (int): The stride of the B-Spline patches. The output
                            dimension 'dim' will satisfy (dim - 1) % patch_stride == 0.
        mode (str): 'upscale' for standard upscaling, 'squarify' for making dimensions equal via knot insertion.
        knots_u (torch.Tensor, optional): Knot vector for U dimension (required for 'squarify').
        knots_v (torch.Tensor, optional): Knot vector for V dimension (required for 'squarify').
        degree_u (int, optional): Degree for U dimension (required for 'squarify').
        degree_v (int, optional): Degree for V dimension (required for 'squarify').

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Updated grid,
        and updated knot vectors for U and V (None in 'upscale' mode).
    """
    if knots_u is None or knots_v is None or degree_u is None or degree_v is None:
        if mode == 'squarify' or scale_factor > 1.0:
            raise ValueError("knots_u, knots_v, degree_u, and degree_v are required when using knot insertion.")
        # Fallback to interpolation if no knots provided and no scaling or squarify
        if scale_factor == 1.0 and reps == 1:
            return grid, knots_u, knots_v

        H_old, W_old, D = grid.shape

        # Calculate the total effective scale factor
        total_scale = scale_factor ** reps

        # Calculate the ideal, uncorrected target size
        H_target = (H_old - 1) * total_scale + 1
        W_target = (W_old - 1) * total_scale + 1

        # Adjust the target size to the next valid B-Spline dimension
        if patch_stride > 0:
            H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1
            W_final = math.ceil((W_target - 1) / patch_stride) * patch_stride + 1
        else:
            H_final = int(H_target)
            W_final = int(W_target)

        # Step 1: Upscale along U (height H) using linear interpolation
        # Permute to (W_old, D, H_old) for correct ordering
        grid_u = grid.permute(1, 2, 0)  # (W_old, D, H_old)
        # Reshape to (W_old * D, H_old)
        grid_u = grid_u.reshape(W_old * D, H_old).unsqueeze(0)  # (1, W_old * D, H_old)
        upsampled_u = F.interpolate(
            grid_u,
            size=H_final,
            mode='linear',
            align_corners=True
        )  # (1, W_old * D, H_final)
        # Squeeze and reshape back to (W_old, D, H_final), then permute to (H_final, W_old, D)
        upsampled_u = upsampled_u.squeeze(0).view(W_old, D, H_final).permute(2, 0, 1)  # (H_final, W_old, D)

        # Step 2: Upscale along V (width W) using linear interpolation
        # Permute to (H_final, D, W_old) for correct ordering
        grid_v = upsampled_u.permute(0, 2, 1)  # (H_final, D, W_old)
        # Reshape to (H_final * D, W_old)
        grid_v = grid_v.reshape(H_final * D, W_old).unsqueeze(0)  # (1, H_final * D, W_old)
        upsampled_v = F.interpolate(
            grid_v,
            size=W_final,
            mode='linear',
            align_corners=True
        )  # (1, H_final * D, W_final)
        # Squeeze and reshape back to (H_final, D, W_final), then permute to (H_final, W_final, D)
        new_grid = upsampled_v.squeeze(0).view(H_final, D, W_final).permute(0, 2, 1)  # (H_final, W_final, D)

        return new_grid, knots_u, knots_v

    # Knot insertion approach
    H, W, D = grid.shape
    H_old, W_old = H, W
    total_scale = scale_factor ** reps
    H_target = (H_old - 1) * total_scale + 1
    W_target = (W_old - 1) * total_scale + 1
    if patch_stride > 0:
        H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1
        W_final = math.ceil((W_target - 1) / patch_stride) * patch_stride + 1
    else:
        H_final = int(H_target)
        W_final = int(W_target)

    if mode == 'squarify':
        target_u = target_v = max(H_final, W_final)
    elif mode == 'upscale':
        target_u = H_final
        target_v = W_final
    else:
        raise ValueError(f"Unknown mode: {mode}")

    while H < target_u or W < target_v:
        if H < target_u and (W >= target_v or H <= W):
            direction = 'u'
            knots = knots_u
            degree = degree_u
        elif W < target_v:
            direction = 'v'
            knots = knots_v
            degree = degree_v
        else:
            break  # Safety

        diffs = knots[1:] - knots[:-1]
        seconddiffs = diffs[1:] - diffs[:-1]
        abs_sd = torch.abs(seconddiffs)

        if abs_sd.max() < 1e-8:  # Essentially uniform, fallback to largest interval
            insert_interval = torch.argmax(diffs).item()
        else:
            max_sd_idx = torch.argmax(abs_sd).item()
            sd = seconddiffs[max_sd_idx]
            if sd > 0:
                insert_interval = max_sd_idx + 1
            else:
                insert_interval = max_sd_idx

        t = (knots[insert_interval] + knots[insert_interval + 1]) / 2

        grid, new_knots = insert_knot(grid, knots, degree, t.item(), direction)

        if direction == 'u':
            knots_u = new_knots
            H += 1
        else:
            knots_v = new_knots
            W += 1

    return grid, knots_u, knots_v



def grid_upscale(
    grid: torch.Tensor,
    reps: int = 1,
    scale_factor: float = 2.0,
    patch_stride: int = 3
) -> torch.Tensor:
    """
    Upsamples a 2D grid tensor, ensuring the output dimensions are
    compliant with B-Spline patch specifications.

    Args:
        grid (torch.Tensor): Tensor of shape (H, W, D).
        reps (int): Number of times to apply the scaling. Total scale is scale_factor**reps.
        scale_factor (float): The upsampling factor for each repetition.
        patch_stride (int): The stride of the B-Spline patches. The output
                            dimension 'dim' will satisfy (dim - 1) % patch_stride == 0.

    Returns:
        torch.Tensor: The upsampled and resized grid.
    """
    if scale_factor == 1.0 and reps == 1:
        return grid

    H_old, W_old, D = grid.shape

    # 1. Calculate the total effective scale factor
    total_scale = scale_factor ** reps

    # 2. Calculate the ideal, uncorrected target size
    # This can result in a float
    H_target = (H_old - 1) * total_scale + 1
    W_target = (W_old - 1) * total_scale + 1

    # 3. Adjust the target size to the next valid B-Spline dimension
    # We find the smallest multiple of patch_stride that is >= (target_size - 1)
    # This is done using the ceiling function.
    if patch_stride > 0:
        H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1
        W_final = math.ceil((W_target - 1) / patch_stride) * patch_stride + 1
    else:
        H_final = int(H_target)
        W_final = int(W_target)

    # 4. Perform a single, direct interpolation to the final valid size
    grid_reshaped = grid.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    upsampled = F.interpolate(
        grid_reshaped,
        size=(H_final, W_final),
        mode='bilinear',
        align_corners=True
    )

    new_grid = upsampled.squeeze(0).permute(1, 2, 0)
    return new_grid
import torch
import math
import torch.nn.functional as F


def update_knot_vectors(
    knots_u: torch.Tensor,
    # knots_v: torch.Tensor,
    H_old: int,
    # W_old: int,
    degree_u: int=3,
    # degree_v: int=3,
    reps: int = 1,
    scale_factor: float = 2.0,
    patch_stride: int = 3,
    device: str = 'cuda'
):
    """
    Updates non-uniform knot vectors to match an upscaled control point grid.

    Args:
        knots_u: Original knot vector for u-direction, length H_old + degree_u + 1.
        knots_v: Original knot vector for v-direction, length W_old + degree_v + 1.
        H_old: Original number of control points in u-direction (before upscaling).
        W_old: Original number of control points in v-direction (before upscaling).
        degree_u: Spline degree in u-direction (p).
        degree_v: Spline degree in v-direction (q).
        reps: Number of upscaling repetitions.
        scale_factor: Upsampling factor per repetition.
        patch_stride: Stride of B-spline patches.
        device: Torch device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Updated knot vectors (knots_u_new, knots_v_new).
    """

    def manual_interp(xp, fp, x, device='cuda'):
        """
        Manual 1D linear interpolation for compatibility with older PyTorch versions.
        """
        xp = xp.to(device)
        fp = fp.to(device)
        x = x.to(device)
        indices = torch.searchsorted(xp, x, right=True)
        indices = indices.clamp(1, len(xp) - 1)
        idx_left = indices - 1
        idx_right = indices
        x_left = xp[idx_left]
        x_right = xp[idx_right]
        f_left = fp[idx_left]
        f_right = fp[idx_right]
        weights = (x - x_left) / (x_right - x_left + 1e-6)
        weights = weights.clamp(0, 1)
        interpolated = f_left + weights * (f_right - f_left)
        return interpolated
    # Validate input knot vectors if provided
    if knots_u is not None:
        assert len(knots_u) == H_old + degree_u + 1, f"U-knot vector length mismatch: expected {H_old + degree_u + 1}, got {len(knots_u)}"

    # Enforce monotonicity on input knot vectors
    if knots_u is not None:
        knots_u = torch.sort(knots_u)[0]
        knots_u = torch.clamp(knots_u, 0, 1)

    # Calculate total scale factor
    total_scale = scale_factor ** reps

    # Calculate ideal target dimensions (same as grid_upscale)
    H_target = (H_old - 1) * total_scale + 1

    # Adjust to satisfy patch stride constraint
    H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1

    # New knot vector lengths
    knots_u_len = H_final + degree_u + 1

    # Extract internal knots
    internal_knots_u = knots_u[degree_u + 1: -(degree_u + 1)] if knots_u is not None else torch.linspace(0, 1, H_old - degree_u - 1, device=device)

    # Calculate number of internal knots for new vectors
    num_internal_knots_u = H_final - degree_u - 1

    # Create parameter points for interpolation
    t_old_u = torch.linspace(0, 1, len(internal_knots_u), device=device)
    t_new_u = torch.linspace(0, 1, num_internal_knots_u, device=device)

    # Interpolate internal knots
    internal_knots_u_new = manual_interp(t_old_u, internal_knots_u, t_new_u, device)

    # Ensure monotonicity
    min_knot_spacing = 1e-6
    diffs_u = internal_knots_u_new[1:] - internal_knots_u_new[:-1]
    diffs_u = torch.clamp(diffs_u, min=min_knot_spacing)
    internal_knots_u_new[1:] = internal_knots_u_new[0] + torch.cumsum(diffs_u, dim=0)

    # Normalize to [0, 1] range
    internal_knots_u_new = (internal_knots_u_new - internal_knots_u_new.min()) / (
        internal_knots_u_new.max() - internal_knots_u_new.min() + 1e-6
    )

    # Construct new knot vectors with clamped boundaries
    knots_u_new = torch.cat([
        torch.zeros(degree_u + 1, device=device),  # p+1 zeros
        internal_knots_u_new,                      # Interpolated internal knots
        torch.ones(degree_u + 1, device=device)    # p+1 ones
    ])

    # Verify lengths and monotonicity
    assert len(knots_u_new) == knots_u_len, f"New u-knot vector length mismatch: expected {knots_u_len}, got {len(knots_u_new)}"
    assert torch.all(knots_u_new[:-1] <= knots_u_new[1:]), "New u-knot vector is not monotonically non-decreasing"

    return knots_u_new
def update_knot_vectors2(
    knots_u: torch.Tensor,
    knots_v: torch.Tensor,
    H_old: int,
    W_old: int,
    degree_u: int=3,
    degree_v: int=3,
    reps: int = 1,
    scale_factor: float = 2.0,
    patch_stride: int = 3,
    device: str = 'cuda'
):
    """
    Updates non-uniform knot vectors to match an upscaled control point grid.

    Args:
        knots_u: Original knot vector for u-direction, length H_old + degree_u + 1.
        knots_v: Original knot vector for v-direction, length W_old + degree_v + 1.
        H_old: Original number of control points in u-direction (before upscaling).
        W_old: Original number of control points in v-direction (before upscaling).
        degree_u: Spline degree in u-direction (p).
        degree_v: Spline degree in v-direction (q).
        reps: Number of upscaling repetitions.
        scale_factor: Upsampling factor per repetition.
        patch_stride: Stride of B-spline patches.
        device: Torch device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Updated knot vectors (knots_u_new, knots_v_new).
    """
    # Validate input knot vectors
    assert len(knots_u) == H_old + degree_u + 1, f"U-knot vector length mismatch: expected {H_old + degree_u + 1}, got {len(knots_u)}"
    assert len(knots_v) == W_old + degree_v + 1, f"V-knot vector length mismatch: expected {W_old + degree_v + 1}, got {len(knots_v)}"

    # Enforce monotonicity on input knot vectors
    knots_u = torch.sort(knots_u)[0]
    knots_v = torch.sort(knots_v)[0]
    knots_u = torch.clamp(knots_u, 0, 1)
    knots_v = torch.clamp(knots_v, 0, 1)

    # Calculate total scale factor
    total_scale = scale_factor ** reps

    # Calculate ideal target dimensions (same as grid_upscale)
    H_target = (H_old - 1) * total_scale + 1
    W_target = (W_old - 1) * total_scale + 1

    # Adjust to satisfy patch stride constraint
    H_final = math.ceil((H_target - 1) / patch_stride) * patch_stride + 1
    W_final = math.ceil((W_target - 1) / patch_stride) * patch_stride + 1

    # New knot vector lengths
    knots_u_len = H_final + degree_u + 1
    knots_v_len = W_final + degree_v + 1

    # Extract internal knots
    internal_knots_u = knots_u[degree_u:-(degree_u)]  # Length: H_old - degree_u
    internal_knots_v = knots_v[degree_v:-(degree_v)]  # Length: W_old - degree_v

    # Calculate number of internal knots for new vectors
    num_internal_knots_u = H_final - degree_u  # Corrected: Remove +1 to match expected length
    num_internal_knots_v = W_final - degree_v

    # Create parameter points for interpolation
    t_old_u = torch.linspace(0, 1, len(internal_knots_u), device=device)
    t_new_u = torch.linspace(0, 1, num_internal_knots_u, device=device)
    t_old_v = torch.linspace(0, 1, len(internal_knots_v), device=device)
    t_new_v = torch.linspace(0, 1, num_internal_knots_v, device=device)

    # Interpolate internal knots
    internal_knots_u_new = manual_interp(t_old_u, internal_knots_u, t_new_u, device)
    internal_knots_v_new = manual_interp(t_old_v, internal_knots_v, t_new_v, device)

    # Ensure monotonicity
    min_knot_spacing = 1e-6
    internal_knots_u_new = torch.cumsum(
        torch.clamp(internal_knots_u_new[1:] - internal_knots_u_new[:-1], min=min_knot_spacing), dim=0
    ) + internal_knots_u_new[0]
    internal_knots_v_new = torch.cumsum(
        torch.clamp(internal_knots_v_new[1:] - internal_knots_v_new[:-1], min=min_knot_spacing), dim=0
    ) + internal_knots_v_new[0]

    # Normalize to [0, 1] range
    internal_knots_u_new = (internal_knots_u_new - internal_knots_u_new.min()) / (
        internal_knots_u_new.max() - internal_knots_u_new.min() + 1e-6
    )
    internal_knots_v_new = (internal_knots_v_new - internal_knots_v_new.min()) / (
        internal_knots_v_new.max() - internal_knots_v_new.min() + 1e-6
    )

    # Construct new knot vectors with clamped boundaries
    knots_u_new = torch.cat([
        torch.zeros(degree_u + 1, device=device),  # p+1 zeros
        internal_knots_u_new,                      # Interpolated internal knots
        torch.ones(degree_u + 1, device=device)    # p+1 ones
    ])
    knots_v_new = torch.cat([
        torch.zeros(degree_v + 1, device=device),  # q+1 zeros
        internal_knots_v_new,                      # Interpolated internal knots
        torch.ones(degree_v + 1, device=device)    # q+1 ones
    ])

    # Verify lengths and monotonicity
    assert len(knots_u_new) == knots_u_len, f"New u-knot vector length mismatch: expected {knots_u_len}, got {len(knots_u_new)}"
    assert len(knots_v_new) == knots_v_len, f"New v-knot vector length mismatch: expected {knots_v_len}, got {len(knots_v_new)}"
    assert torch.all(knots_u_new[:-1] <= knots_u_new[1:]), "New u-knot vector is not monotonically non-decreasing"
    assert torch.all(knots_v_new[:-1] <= knots_v_new[1:]), "New v-knot vector is not monotonically non-decreasing"

    return knots_u_new, knots_v_new
def normals_to_quaternions(normals, z, eps=1e-6):
    """
    Convert a batch of normal vectors (shape: [N, 3]) into quaternions
    representing the rotation that aligns the canonical z-axis [0,0,1]
    with each normal.

    Args:
        normals (torch.Tensor): Tensor of shape (N,3) containing normal vectors.
        eps (float): A small epsilon for numerical stability.

    Returns:
        torch.Tensor: Tensor of shape (N,4) containing normalized quaternions in (w, x, y, z) format.
    """
    # Normalize the input normals (avoid division by zero)
    n = F.normalize(normals, dim=-1)

    # Canonical z-axis
    # z = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)

    # Compute dot product between each normal and the z-axis
    # Shape: (N,)
    dot = (n * z).sum(dim=-1)

    # Compute cross product between z and n for each normal.
    # We need to expand z to (N,3) for broadcasting.
    cross = torch.cross(z.expand_as(n), n, dim=-1)

    # Form the quaternion candidate: q = [1+dot, cross]
    # This gives a tensor of shape (N,4) where the first component is the scalar part.
    q = torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1)

    # Check for nearly zero norm (which happens when the normal is nearly opposite to z)
    q_norm = q.norm(dim=-1, keepdim=True)
    # Create a fallback quaternion (e.g. 180 degree rotation about the x-axis)
    fallback = torch.tensor([0.0, 1.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q)

    # Use the fallback where needed
    q = torch.where(q_norm < eps, fallback, q)

    # Normalize the quaternion so that it represents a valid rotation
    q = F.normalize(q, dim=-1)
    return q

def reduce_channels_pca(features: torch.Tensor, target_dim: int = 3) -> torch.Tensor:
    """
    Reduce a tensor of shape (H, W, C) to (H, W, target_dim) using PCA.

    Args:
        features (torch.Tensor): Input tensor of shape (H, W, C).
        target_dim (int): Number of dimensions to reduce to (default: 3).

    Returns:
        torch.Tensor: A tensor of shape (H, W, target_dim) after PCA reduction, with values normalized to [0,1].
    """
    H, W, C = features.shape
    # Reshape to (H*W, C)
    reshaped = features.view(-1, C)

    # Center the data (subtract the mean per channel)
    mean = reshaped.mean(dim=0, keepdim=True)
    reshaped_centered = reshaped - mean

    # Apply PCA using torch.pca_lowrank.
    # V contains the principal components; we select the first target_dim components.
    U, S, V = torch.pca_lowrank(reshaped_centered, q=target_dim)
    # Project the centered features onto the top principal components.
    reduced = reshaped_centered @ V[:, :target_dim]

    # Reshape back to image shape (H, W, target_dim)
    reduced_img = reduced.view(H, W, target_dim)

    # Optionally, normalize the reduced image to [0, 1] for visualization.
    v_min, v_max = reduced_img.min(), reduced_img.max()
    if v_max - v_min > 1e-8:
        reduced_img = (reduced_img - v_min) / (v_max - v_min)
    else:
        reduced_img = torch.zeros_like(reduced_img)
    return reduced_img

def process_feature_grid(grid: torch.Tensor) -> torch.Tensor:
    """
    Process a feature grid (H, W, C) to ensure it has 3 channels for visualization.
    If the grid has more than 3 channels, PCA is applied to reduce it to 3 channels.
    If the grid has 1 or 2 channels, it is duplicated or padded appropriately.

    Args:
        grid (torch.Tensor): Input tensor of shape (H, W, C).

    Returns:
        torch.Tensor: Processed tensor of shape (H, W, 3) ready for display.
    """
    if grid.ndim == 2:
        # If grid is 2D, make it 3 channels.
        grid = grid.unsqueeze(-1).repeat(1, 1, 3)
    elif grid.ndim == 3:
        H, W, C = grid.shape
        if C == 1:
            grid = grid.repeat(1, 1, 3)
        elif C == 2:
            # Pad with a zero channel.
            pad = torch.zeros((H, W, 1), device=grid.device, dtype=grid.dtype)
            grid = torch.cat([grid, pad], dim=2)
        elif C > 3:
            try:
            # Use PCA to reduce to 3 channels.
                grid = reduce_channels_pca(grid, target_dim=3)
            except:
                return grid.mean(dim=-1).unsqueeze(-1)
        # If C == 3, no change is needed.
    else:
        raise ValueError("Unsupported grid shape: expected 2D or 3D tensor.")

    # Convert to numpy (if needed for visualization) after moving to CPU.
    return grid.cpu().numpy()

def visualize_all_grid_features(nurbs, global_grids, title=""):  ### TODO: Should be updated to exported
    """
    Visualize the global grid features for all parameters in a single figure with subplots.
    For each parameter, the corresponding grid is displayed and the parameter name is shown above the subplot.
    """
    # List of parameter names that we wish to visualize.
    # This list should match the keys in self.global_grids.
    vis_normal = lambda normal: np.uint8((normal[..., [1, 2, 0]] + 1) / 2 * 255)
    param_names = [p for p in global_grids.keys()]
    num_params = len(param_names)
    n_rows = int(np.ceil(np.sqrt(num_params)))
    n_cols = int(np.ceil(num_params / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    # Iterate over each parameter and plot its grid representation.
    for idx, pname in enumerate(param_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        if not nurbs.ctrl_grids[pname].requires_grad:
            continue
        # Retrieve the grid from the global grids; here we use the first element for visualization.
        grid_feature = nurbs.ctrl_grids[pname].detach().cpu()
        # If the parameter is a feature that needs conversion (for example SH coefficients)
        if pname == "features_dc":
            img = SH2RGB(nurbs.spherical_harmonics).view(nurbs.Us, nurbs.Vs, -1).detach().cpu().numpy()
        elif pname == "control_points":
            img = vis_normal(
                F.normalize(nurbs.pred_normals, dim=-1).view(nurbs.Us, nurbs.Vs, 3).detach().cpu().numpy())
        elif pname in ["control_weights", "opacity"]:
            img = grid_feature.sigmoid().view(*nurbs.feat_shape).numpy()
        elif pname == "scaling":
            img = nurbs.scaling_activation(grid_feature).view(*nurbs.feat_shape).detach().cpu().numpy()
        elif pname == "rotations":
            img = grid_feature.view(*nurbs.feat_shape).detach().cpu().numpy()
        elif pname == 'features_rest':
            continue
        else:
            # For other parameters, visualize the activated version (after sigmoid if desired).
            img = grid_feature.numpy()

        img = process_feature_grid(torch.from_numpy(img))
        if img.dtype == np.uint8:
            img = img.clip(0, 255)
        else:
            img = img.clip(0., 1.)

        # Optionally, rotate the image 90° clockwise.
        img = np.rot90(img, k=-3)
        ax.imshow(img)
        ax.set_title(pname, fontsize=10)
        ax.axis("off")

    # Hide any unused subplots if the grid is larger than the number of parameters.
    for idx in range(num_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    plt.close(fig)  # add this
    del fig, axes
    gc.collect()
    torch.cuda.empty_cache()  # optional but handy