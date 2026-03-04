"""
PLY export functionality for NURBS/Spline models.
Compatible with GaussianModel's PLY format for visualization and evaluation.
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Optional, Union
import os


def construct_list_of_attributes(num_sh_coeffs: int, scaling_dims: int = 3) -> list:
    """
    Construct the list of PLY attributes matching GaussianModel format.

    Args:
        num_sh_coeffs: Number of SH coefficients per color channel
        scaling_dims: Number of scaling dimensions (2 or 3)

    Returns:
        List of (name, dtype) tuples for PLY export
    """
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    # DC component of SH (first coefficient for each RGB channel)
    for i in range(3):
        attributes.append(f'f_dc_{i}')

    # Rest of SH coefficients
    for i in range((num_sh_coeffs - 1) * 3):
        attributes.append(f'f_rest_{i}')

    # Opacity
    attributes.append('opacity')

    # Scaling
    for i in range(scaling_dims):
        attributes.append(f'scale_{i}')

    # Rotation quaternion
    for i in range(4):
        attributes.append(f'rot_{i}')

    return attributes


def save_ply_single_surface(
        path: str,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        opacities: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        scaling_activation=torch.exp,
        opacity_activation=torch.sigmoid,
):
    """
    Save a single surface's Gaussians to PLY format.

    Args:
        path: Output PLY file path
        xyz:  [N, 3] positions
        normals: [N, 3] surface normals
        features_dc:  [N, 1, 3] or [N, 3] DC SH coefficients
        features_rest: [N, (SH-1), 3] rest SH coefficients
        opacities: [N, 1] raw opacity values (before activation)
        scaling: [N, 3] raw scaling values (before activation)
        rotation: [N, 4] quaternion rotations
        scaling_activation: Activation function for scaling
        opacity_activation:  Activation function for opacity
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Move to CPU and convert to numpy
    xyz_np = xyz.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()

    # Handle features shape
    if features_dc.dim() == 3:
        f_dc_np = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        f_dc_np = features_dc.detach().cpu().numpy()

    if features_rest.dim() == 3:
        f_rest_np = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        f_rest_np = features_rest.detach().cpu().numpy()

    # Raw values (not activated) for storage
    opacities_np = opacities.detach().cpu().numpy()
    scale_np = scaling.detach().cpu().numpy()
    rot_np = rotation.detach().cpu().numpy()

    # Construct attribute list
    num_sh = 1 + (f_rest_np.shape[1] // 3) if f_rest_np.size > 0 else 1
    scaling_dims = scale_np.shape[1] if scale_np.ndim > 1 else 3

    attributes = construct_list_of_attributes(num_sh, scaling_dims)
    dtype_full = [(attr, 'f4') for attr in attributes]

    # Create structured array
    n_points = xyz_np.shape[0]
    elements = np.empty(n_points, dtype=dtype_full)

    # Fill in data
    # Position
    elements['x'] = xyz_np[:, 0]
    elements['y'] = xyz_np[:, 1]
    elements['z'] = xyz_np[:, 2]

    # Normals
    elements['nx'] = normals_np[:, 0]
    elements['ny'] = normals_np[:, 1]
    elements['nz'] = normals_np[:, 2]

    # DC SH features
    for i in range(3):
        if f_dc_np.ndim > 1 and f_dc_np.shape[1] > i:
            elements[f'f_dc_{i}'] = f_dc_np[:, i]
        else:
            elements[f'f_dc_{i}'] = 0.0

    # Rest SH features
    for i in range(f_rest_np.shape[1] if f_rest_np.ndim > 1 else 0):
        elements[f'f_rest_{i}'] = f_rest_np[:, i]

    # Opacity
    elements['opacity'] = opacities_np.flatten()

    # Scaling
    for i in range(scaling_dims):
        if scale_np.ndim > 1:
            elements[f'scale_{i}'] = scale_np[:, i]
        else:
            elements[f'scale_{i}'] = scale_np[i] if i < len(scale_np) else 0.0

    # Rotation
    for i in range(4):
        elements[f'rot_{i}'] = rot_np[:, i]

    # Create PLY element and save
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"[PLY Export] Saved {n_points} Gaussians to {path}")


class PLYExportMixin:
    """
    Mixin class providing PLY export functionality for SplineModel and MultiSurfaceSplineModel.
    Add this to your model classes via multiple inheritance.
    """

    def save_ply(self, path: str, viewpoint_cam=None):
        """
        Save model to PLY format compatible with GaussianSplatting viewers.

        Args:
            path: Output PLY file path
            viewpoint_cam: Optional camera for computing oriented normals.
                          If None, uses geometric normals.
        """
        # Ensure we have fresh data
        if hasattr(self, 'invalidate_all_caches'):
            pass  # Don't invalidate - use current state

        # Get Gaussian properties
        xyz = self.get_xyz  # [N, 3]

        # Get normals - prefer geometric if no camera provided
        if viewpoint_cam is not None:
            normals = self.get_normal(viewpoint_cam)
        elif hasattr(self, 'surface_normals_raw'):
            normals = self.surface_normals_raw
        else:
            normals = self.get_smallest_axis()

        # Get features
        features = self.get_features  # [N, SH, 3]
        if features.dim() == 3:
            features_dc = features[:, : 1, :]  # [N, 1, 3]
            features_rest = features[:, 1:, :]  # [N, SH-1, 3]
        else:
            features_dc = features
            features_rest = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device)

        # Get other properties (raw, before activation)
        opacities = self.opacity.cache if self.opacity.cache is not None else \
            self.opacity.forward().view(-1, 1)

        scaling = self.scaling.cache if self.scaling.cache is not None else \
            self.scaling.forward().view(-1, self.state.scaling_dims)

        rotation = self.rotation.cache if self.rotation.cache is not None else \
            self.get_rotation

        # Normalize rotation quaternions
        rotation = torch.nn.functional.normalize(rotation, dim=-1)

        save_ply_single_surface(
            path=path,
            xyz=xyz,
            normals=normals,
            features_dc=features_dc,
            features_rest=features_rest,
            opacities=opacities,
            scaling=scaling,
            rotation=rotation,
            scaling_activation=self.scaling_activation,
            opacity_activation=self.opacity_activation,
        )

    def save_ply_with_metadata(self, path: str, viewpoint_cam=None, include_uv: bool = True):
        """
        Extended PLY export including UV coordinates and surface metadata.

        Args:
            path: Output PLY file path
            viewpoint_cam:  Optional camera for normals
            include_uv: Whether to include UV coordinates as additional attributes
        """
        # Get base data
        xyz = self.get_xyz

        if viewpoint_cam is not None:
            normals = self.get_normal(viewpoint_cam)
        else:
            normals = self.surface_normals_raw if hasattr(self, 'surface_normals_raw') else self.get_smallest_axis()

        features = self.get_features
        if features.dim() == 3:
            features_dc = features[:, :1, :].transpose(1, 2).flatten(start_dim=1)
            features_rest = features[:, 1:, :].transpose(1, 2).flatten(start_dim=1)
        else:
            features_dc = features
            features_rest = torch.zeros(xyz.shape[0], 0, device=xyz.device)

        opacities = self.opacity.cache if self.opacity.cache is not None else \
            self.opacity.forward().view(-1, 1)
        scaling = self.scaling.cache if self.scaling.cache is not None else \
            self.scaling.forward().view(-1, self.state.scaling_dims)
        rotation = torch.nn.functional.normalize(self.get_rotation, dim=-1)

        # Build attribute list
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # SH features
        for i in range(features_dc.shape[1]):
            attributes.append(f'f_dc_{i}')
        for i in range(features_rest.shape[1]):
            attributes.append(f'f_rest_{i}')

        attributes.append('opacity')

        for i in range(scaling.shape[1]):
            attributes.append(f'scale_{i}')

        for i in range(4):
            attributes.append(f'rot_{i}')

        # UV coordinates
        if include_uv:
            attributes.extend(['u', 'v'])

        # Surface index (for multi-surface)
        if hasattr(self, 'surfaces'):
            attributes.append('surface_idx')

        dtype_full = [(attr, 'f4') for attr in attributes]

        # Convert to numpy
        n_points = xyz.shape[0]
        elements = np.empty(n_points, dtype=dtype_full)

        xyz_np = xyz.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        elements['x'] = xyz_np[:, 0]
        elements['y'] = xyz_np[:, 1]
        elements['z'] = xyz_np[:, 2]
        elements['nx'] = normals_np[:, 0]
        elements['ny'] = normals_np[:, 1]
        elements['nz'] = normals_np[:, 2]

        f_dc_np = features_dc.detach().cpu().numpy()
        for i in range(f_dc_np.shape[1]):
            elements[f'f_dc_{i}'] = f_dc_np[:, i]

        f_rest_np = features_rest.detach().cpu().numpy()
        for i in range(f_rest_np.shape[1]):
            elements[f'f_rest_{i}'] = f_rest_np[:, i]

        elements['opacity'] = opacities.detach().cpu().numpy().flatten()

        scale_np = scaling.detach().cpu().numpy()
        for i in range(scale_np.shape[1]):
            elements[f'scale_{i}'] = scale_np[:, i]

        rot_np = rotation.detach().cpu().numpy()
        for i in range(4):
            elements[f'rot_{i}'] = rot_np[:, i]

        # UV coordinates
        if include_uv:
            Us, Vs = self.state.Us, self.state.Vs
            u_coords = torch.linspace(0, 1, Us, device=xyz.device).unsqueeze(1).expand(Us, Vs).flatten()
            v_coords = torch.linspace(0, 1, Vs, device=xyz.device).unsqueeze(0).expand(Us, Vs).flatten()
            elements['u'] = u_coords.cpu().numpy()
            elements['v'] = v_coords.cpu().numpy()

        # Surface index
        if hasattr(self, 'surfaces'):
            elements['surface_idx'] = self.surface_indices.cpu().numpy().astype(np.float32)

        # Save
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"[PLY Export] Saved {n_points} Gaussians with metadata to {path}")


# ============================================================================
# Integration with existing models
# ============================================================================

def add_save_ply_to_spline_model(SplineModel):
    """
    Dynamically add save_ply method to SplineModel class.
    Call this after importing SplineModel.
    """

    def save_ply(self, path: str, viewpoint_cam=None):
        """
        Save model to PLY format compatible with GaussianSplatting viewers.
        """
        xyz = self.get_xyz

        if viewpoint_cam is not None:
            normals = self.get_normal(viewpoint_cam)
        elif hasattr(self, 'surface_normals_raw'):
            normals = self.surface_normals_raw
        else:
            normals = self.get_smallest_axis()

        features = self.get_features
        if features.dim() == 3:
            features_dc = features[:, :1, :]
            features_rest = features[:, 1:, :]
        else:
            features_dc = features.unsqueeze(1)
            features_rest = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device)

        opacities = self.opacity.cache if self.opacity.cache is not None else \
            self.opacity.forward().view(-1, 1)

        scaling = self.scaling.cache if self.scaling.cache is not None else \
            self.scaling.forward().view(-1, self.state.scaling_dims)

        rotation = torch.nn.functional.normalize(self.get_rotation, dim=-1)

        save_ply_single_surface(
            path=path,
            xyz=xyz,
            normals=normals,
            features_dc=features_dc,
            features_rest=features_rest,
            opacities=opacities,
            scaling=scaling,
            rotation=rotation,
        )

    SplineModel.save_ply = save_ply
    return SplineModel


def add_save_ply_to_multi_surface_model(MultiSurfaceSplineModel):
    """
    Dynamically add save_ply method to MultiSurfaceSplineModel class.
    """

    def save_ply(self, path: str, viewpoint_cam=None):
        """
        Save all surfaces to a single PLY file.
        """
        # Collect from all active surfaces
        xyz_list = []
        normals_list = []
        features_dc_list = []
        features_rest_list = []
        opacities_list = []
        scaling_list = []
        rotation_list = []

        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if not active:
                continue

            xyz_list.append(surface.get_xyz)

            if viewpoint_cam is not None:
                normals_list.append(surface.get_normal(viewpoint_cam))
            elif hasattr(surface, 'surface_normals_raw'):
                normals_list.append(surface.surface_normals_raw)
            else:
                normals_list.append(surface.get_smallest_axis())

            features = surface.get_features
            if features.dim() == 3:
                features_dc_list.append(features[:, :1, :])
                features_rest_list.append(features[:, 1:, :])
            else:
                features_dc_list.append(features.unsqueeze(1))
                features_rest_list.append(
                    torch.zeros(surface.get_xyz.shape[0], 0, 3, device=features.device)
                )

            opacities_list.append(
                surface.opacity.cache if surface.opacity.cache is not None else
                surface.opacity.forward().view(-1, 1)
            )

            scaling_list.append(
                surface.scaling.cache if surface.scaling.cache is not None else
                surface.scaling.forward().view(-1, surface.state.scaling_dims)
            )

            rotation_list.append(
                torch.nn.functional.normalize(surface.get_rotation, dim=-1)
            )

        if not xyz_list:
            print("[PLY Export] No active surfaces to export")
            return

        # Concatenate all
        xyz = torch.cat(xyz_list, dim=0)
        normals = torch.cat(normals_list, dim=0)
        features_dc = torch.cat(features_dc_list, dim=0)
        features_rest = torch.cat(features_rest_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        scaling = torch.cat(scaling_list, dim=0)
        rotation = torch.cat(rotation_list, dim=0)

        save_ply_single_surface(
            path=path,
            xyz=xyz,
            normals=normals,
            features_dc=features_dc,
            features_rest=features_rest,
            opacities=opacities,
            scaling=scaling,
            rotation=rotation,
        )

    def save_ply_per_surface(self, directory: str, viewpoint_cam=None):
        """
        Save each surface to a separate PLY file.

        Args:
            directory: Output directory
            viewpoint_cam:  Optional camera for normals
        """
        os.makedirs(directory, exist_ok=True)

        for i, (surface, active, label) in enumerate(
                zip(self.surfaces, self._active_surfaces, self.labels)
        ):
            if not active:
                continue

            filename = f"surface_{i}_{label}.ply"
            path = os.path.join(directory, filename)
            surface.save_ply(path, viewpoint_cam)

    MultiSurfaceSplineModel.save_ply = save_ply
    MultiSurfaceSplineModel.save_ply_per_surface = save_ply_per_surface
    return MultiSurfaceSplineModel