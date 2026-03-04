import torch
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple

from scene.cameras import Camera


class RayInfo(NamedTuple):
    """Container for ray information with visibility."""
    directions: torch.Tensor  # [N, 3] normalized ray directions (world space)
    visible_mask: torch.Tensor  # [N] boolean mask - True if point is visible
    pixel_coords: torch.Tensor  # [N, 2] pixel coordinates (for adaptive sampling)
    depths: torch.Tensor  # [N] depth values (z in camera space)
    xyz: Optional[torch.Tensor] = None  # [N, 3] original 3D points (optional),
    stats: Optional[dict] = None  # Additional stats (optional)
    visibility_stats: Optional[torch.Tensor] = None  # [N, ...] additional visibility stats (optional)


def compute_ray_info(
        camera,
        surface_points: torch.Tensor,
        scale: float = 1.0,
        depth_min: float = 1e-3,
        depth_max=torch.inf,
        border_margin: float = 0.0,
        shape: Tuple[int, int] = None,
        visibility_stats: torch.Tensor = None
) -> RayInfo:
    """
    Compute complete ray information for surface points.

    Args:
        camera: Camera object
        surface_points:  [N, 3] world-space surface points
        scale: Resolution scale for ray grid
        depth_min: Minimum valid depth
        border_margin:  Margin from image border (in pixels) to consider visible

    Returns:
        RayInfo containing directions, visibility mask, pixel coords, and depths
    """
    device = surface_points.device
    if shape is None:
        Us, Vs = surface_points.shape[:2]
    else:
        Us, Vs = shape
    N = Us * Vs
    surface_points = surface_points.reshape(-1, 3)
    # Get image dimensions
    # W = Us #int(camera.image_width / scale)
    W = int(camera.image_width / scale)
    # H = Vs
    H = int(camera.image_height / scale)

    # Transform points to camera space
    cam_points = camera.world_to_camera(surface_points)  # [N, 3]
    depths = cam_points[:, 2]  # [N]

    # Project to pixel coordinates
    pixel_coords = camera.camera_to_image(cam_points)  # [N, 2] - (u, v)

    # Compute visibility mask
    depth_valid = depths > depth_min
    depth_valid = depth_valid & (depths < depth_max)
    in_bounds_x = (pixel_coords[:, 0] >= border_margin) & \
                  (pixel_coords[:, 0] < (W * scale - border_margin))
    in_bounds_y = (pixel_coords[:, 1] >= border_margin) & \
                  (pixel_coords[:, 1] < (H * scale - border_margin))
    in_bounds = in_bounds_x & in_bounds_y

    visible_mask = depth_valid & in_bounds

    # Get ray grid from camera
    # rays_cam = camera.get_rays(scale=scale)  # [H, W, 3]
    # rays_cam = F.normalize(rays_cam, dim=-1)
    dirs_cam = camera.get_rays(size=(Us, Vs)).reshape(-1, 3)  # [N, 3]
    # Normalize pixel coords to [-1, 1] for grid_sample
    px_norm = torch.zeros_like(pixel_coords)
    px_norm[:, 0] = 2 * (pixel_coords[:, 0] / scale) / (W - 1) - 1
    px_norm[:, 1] = 2 * (pixel_coords[:, 1] / scale) / (H - 1) - 1

    # Clamp for out-of-bounds points (grid_sample still needs valid coords)
    px_norm = px_norm.clamp(-1, 1)

    R = torch.tensor(camera.R, dtype=torch.float32, device=device)
    dirs_world = dirs_cam @ R

    # For non-visible points, use geometric fallback
    dirs_geom = surface_points - camera.camera_center.unsqueeze(0)
    dirs_geom = F.normalize(dirs_geom, dim=-1)

    # Blend:  use pixel rays where visible, geometric otherwise
    directions = torch.where(
        visible_mask.unsqueeze(-1),
        F.normalize(dirs_world, dim=-1),
        dirs_geom
    )

    return RayInfo(
        directions=directions,
        visible_mask=visible_mask,
        pixel_coords=pixel_coords,
        depths=depths,
        xyz=surface_points.reshape(-1, 3),
        stats={"avg_vis_rate": visibility_stats[..., :1],
                "radii": visibility_stats[..., 1:]}  if visibility_stats is not None else None

    )


def compute_front_facing_mask(
        ray_info: RayInfo,
        surface_points: torch.Tensor,  # [N, 3] or [Us, Vs, 3]
        surface_normals: torch.Tensor,  # [N, 3] or [Us, Vs, 3]
        camera: Camera,
        threshold: float = 0.0,  # Threshold for front-facing (0 = exactly perpendicular)
        return_cos_theta: bool = True  # If True, also return the cosine values
) -> torch.Tensor:
    """
    Compute which surface points are front-facing (visible) vs back-facing (hidden).

    Args:
        surface_points: 3D positions on the surface
        surface_normals: Surface normals (should be normalized)
        camera: Camera instance
        threshold: Dot product threshold (default 0, use small positive for margin)
        return_cos_theta: Whether to return cosine values along with mask

    Returns:
        front_facing_mask: Boolean tensor, True where surface faces camera
        (optional) cos_theta: Cosine of angle between normal and view direction
    """
    view_dir = torch.nn.functional.normalize(ray_info.directions, p=2, dim=-1).reshape(-1, 3)
    # view_dir = torch.nn.functional.normalize(camera.get_rays(), p=2, dim=-1)
    # depth_normal = out["depth_normal"].permute(1, 2, 0)
    original_shape = surface_points.shape[:2]
    surface_normals = surface_normals.reshape(-1, 3)
    depth_normal = torch.nn.functional.normalize(surface_normals, p=2, dim=-1)
    dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
    angle = torch.acos(dot)
    orthog_mask = angle > (80.0 / 180 * 3.14159)
    cos_theta = torch.cos(angle).reshape(original_shape)
    front_facing = ~orthog_mask.reshape(original_shape)

    if return_cos_theta:
        return front_facing, cos_theta
    return front_facing


def compute_oriented_normals(
        surface_normals: torch.Tensor,  # [N, 3] or [Us, Vs, 3]
        surface_points: torch.Tensor,  # [N, 3] or [Us, Vs, 3]
        camera_center: torch.Tensor,  # [3]
) -> torch.Tensor:
    """
    Orient normals to always face the camera (flip back-facing normals).

    This is useful for rendering where you want consistent normal orientation.

    Returns:
        oriented_normals:  Normals flipped to face camera where needed
    """
    original_shape = surface_normals.shape

    points_flat = surface_points.reshape(-1, 3)
    normals_flat = surface_normals.reshape(-1, 3)
    normals_flat = F.normalize(normals_flat, dim=-1)

    # View direction from surface to camera
    view_dirs = F.normalize(camera_center.unsqueeze(0) - points_flat, dim=-1)

    # Dot product
    cos_theta = (normals_flat * view_dirs).sum(dim=-1, keepdim=True)  # [N, 1]

    # Flip normals where back-facing (cos_theta < 0)
    flip_mask = (cos_theta < 0).float()
    oriented_normals = normals_flat * (1 - 2 * flip_mask)  # Flip sign where needed

    return oriented_normals.reshape(original_shape)


def compute_view_dependent_weight(
        surface_normals: torch.Tensor,
        surface_points: torch.Tensor,
        camera_center: torch.Tensor,
        falloff_type: str = 'cosine',  # 'cosine', 'binary', 'smooth'
        min_weight: float = 0.0,  # Minimum weight for back-facing
        smooth_width: float = 0.1  # Width for smooth falloff
) -> torch.Tensor:
    """
    Compute view-dependent weights based on surface orientation.

    Useful for:
    - Soft visibility in differentiable rendering
    - Importance sampling based on orientation
    - Gradual falloff at grazing angles

    Args:
        surface_normals: Surface normals
        surface_points:  3D positions
        camera_center: Camera position
        falloff_type: Type of weight falloff
            - 'cosine': weight = max(cos_theta, 0)
            - 'binary': weight = 1 if front-facing else 0
            - 'smooth': smooth sigmoid transition
        min_weight: Minimum weight (for back-facing regions)
        smooth_width: Width of smooth transition zone

    Returns:
        weights: View-dependent weights in [min_weight, 1]
    """
    original_shape = surface_points.shape[:-1]

    points_flat = surface_points.reshape(-1, 3)
    normals_flat = F.normalize(surface_normals.reshape(-1, 3), dim=-1)

    view_dirs = F.normalize(camera_center.unsqueeze(0) - points_flat, dim=-1)
    cos_theta = (normals_flat * view_dirs).sum(dim=-1)

    if falloff_type == 'binary':
        weights = (cos_theta > 0).float()

    elif falloff_type == 'cosine':
        weights = cos_theta.clamp(min=0)

    elif falloff_type == 'smooth':
        # Smooth sigmoid transition around cos_theta = 0
        weights = torch.sigmoid(cos_theta / smooth_width)

    else:
        raise ValueError(f"Unknown falloff_type: {falloff_type}")

    # Apply minimum weight
    weights = weights * (1 - min_weight) + min_weight

    return weights.reshape(original_shape)