import torch
import torch.nn.functional as F


def uv_tangent(tangents_u, tangents_v, dudv=None, lambda_curv=0.1):
    # Local import: modules.control_feature imports this module (circular).
    from modules.control_feature.quaternion_utils import matrix_to_quaternion
    """
    Build splat rotations from surface tangents.

    Gram-Schmidt frame [t̂u, t̂v⊥, n̂] — orthonormal with det(R) = +1.
    Orientation is made consistent by flipping the FULL frame (normal AND
    one tangent) where n_z <= 0, so the result stays a proper rotation.
    All ops are out-of-place: safe under autograd.
    """
    eps = 1e-10
    normals = torch.cross(tangents_u, tangents_v, dim=-1)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + eps)
    tu_norm = tangents_u / (tangents_u.norm(dim=-1, keepdim=True) + eps)
    tv_norm = tangents_v / (tangents_v.norm(dim=-1, keepdim=True) + eps)
    tv_ortho = tv_norm - (torch.einsum('hwi,hwi->hw', tv_norm, tu_norm).unsqueeze(-1) * tu_norm)
    tv_ortho = tv_ortho / (tv_ortho.norm(dim=-1, keepdim=True) + eps)

    # Consistent orientation: flip normal AND second tangent together so the
    # frame remains right-handed (det = +1) instead of becoming a reflection.
    flip = (normals[..., 2:3] <= 0)
    normals = torch.where(flip, -normals, normals)
    tv_ortho = torch.where(flip, -tv_ortho, tv_ortho)

    R = torch.stack([tu_norm, tv_ortho, normals], dim=-1)
    rotation = F.normalize(
        matrix_to_quaternion(R.reshape(-1, 3, 3)).reshape(*tangents_u.shape[:2], 4),
        dim=-1,
    )
    return rotation

def n2q(sample_normals):
    # Local import: modules.control_feature imports this module (circular).
    from modules.control_feature.utils.rotation_control import _normals_to_quaternions_robust
    H, W = sample_normals.shape[:2]
    flip_mask = (sample_normals[:, :, 2:3] < 0)
    sample_normals = torch.where(flip_mask, -sample_normals, sample_normals)
    # ── 4. Interpolate normals onto control grid ────────────────────
    #    Build a regular UV grid matching [H, W], then use griddata
    #    to scatter-interpolate the per-sample normals.
    # uu, vv = torch.meshgrid(u, v, indexing='ij')
    # grid_uv = torch.stack([uu.ravel(), vv.ravel()], dim=-1)  # [H*W, 2]

    grid_normals = torch.nn.functional.interpolate(
        sample_normals.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
        size=(H, W),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0) # [H, W, 3]
    # for dim in range(3):

        # grid_normals[:, dim] = torch.int(
        #     uv, sample_normals[:, dim], grid_uv,
        #     method='linear',
        #     fill_value=0.0,
        # )

    # Fill NaN holes with nearest-neighbor
    nan_mask = torch.isnan(grid_normals).any(axis=1)
    if nan_mask.any():
        raise ValueError("NaN values found in grid_normals after interpolation. Consider using a different interpolation method or filling strategy.")
    #     for dim in range(3):
    #         grid_normals[nan_mask, dim] = griddata(
    #             uv, sample_normals[:, dim], grid_uv[nan_mask],
    #             method='nearest',
    #         )

    # Re-normalize after interpolation (linear interp of unit vectors
    # doesn't yield unit vectors)
    grid_normals_norms = torch.linalg.norm(grid_normals, axis=-1, keepdims=True)
    grid_normals_norms = torch.clip(grid_normals_norms, 1e-8, None)
    grid_normals = grid_normals / grid_normals_norms

    # grid_normals = grid_normals.reshape(H, W, 3)

    # ── 5. Convert normals → quaternions ────────────────────────────
    #    We want the quaternion that rotates [0, 0, 1] to align with
    #    the normal. This is the same convention as normals_to_quaternions2
    #    and vectors_to_quaternions in modules/spline_utils.py.
    normals_t = grid_normals.reshape(-1, 3).to('cuda')  # [H*W, 3]
    quats = _normals_to_quaternions_robust(normals_t)  # [H*W, 4]
    quats = quats.reshape(H, W, 4)

    # ── 6. Spatial smoothing on the quaternion grid ─────────────────
    #    Iterative 3×3 geodesic mean to suppress high-frequency noise
    #    from mesh discretization artifacts.
    # for _ in range(3):
    #     quats = _smooth_quaternion_grid(quats)

    return quats
