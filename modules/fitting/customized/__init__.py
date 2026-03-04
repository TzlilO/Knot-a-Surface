"""
Multi-Surface SplineModel Extension

Handles multiple NURBS surfaces for:
1. Background/Object separation
2. K-component decomposition

Each surface is a separate SplineModel instance with shared training infrastructure.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from nurbs_from_pointcloud import (
    DecompositionMode,
    MultiSurfaceResult,
    NURBSSurfaceData,
    create_nurbs_from_pointcloud,
    surfaces_to_torch
)


@dataclass
class SurfaceRenderResult:
    """Rendering result from a single surface."""
    xyz: torch.Tensor  # [N, 3]
    features: torch.Tensor  # [N, SH_coeffs, 3]
    opacity: torch.Tensor  # [N, 1]
    scaling: torch.Tensor  # [N, 3]
    rotation: torch.Tensor  # [N, 4]
    surface_idx: int
    label: str


class MultiSurfaceSplineModel(nn.Module):
    """
    Container for multiple SplineModel instances.

    Enables:
    - Training multiple surfaces jointly
    - Rendering with proper compositing
    - Per-surface control over optimization

    Usage:
        >>> # From point cloud
        >>> model = MultiSurfaceSplineModel. from_pointcloud(
        ...      points, colors,
        ...     config=config,
        ...     args=args,
        ...     mode=DecompositionMode. BACKGROUND_OBJECT
        ... )

        >>> # Forward pass returns combined Gaussians
        >>> model. forward(camera)
        >>> xyz = model.get_xyz  # Combined from all surfaces
    """

    def __init__(
            self,
            surfaces: List['SplineModel'],  # Avoid circular import
            labels: List[str],
            decomposition_mode: DecompositionMode,
            point_labels: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.surfaces = nn.ModuleList(surfaces)
        self.labels = labels
        self.decomposition_mode = decomposition_mode
        self.num_surfaces = len(surfaces)

        # Store point-to-surface mapping
        self.register_buffer('point_labels', point_labels)

        # Surface-specific settings
        self._active_surfaces = [True] * self.num_surfaces
        self._surface_weights = [1.0] * self.num_surfaces

    @classmethod
    def from_pointcloud(
            cls,
            points: Union[np.ndarray, torch.Tensor],
            colors: Optional[Union[np.ndarray, torch.Tensor]],
            config,  # NurbsOptimizationParams
            args,  # Training args
            mode: DecompositionMode = DecompositionMode.SINGLE,
            resolution: Tuple[int, int] = (32, 32),
            spatial_lr_scale: float = 1.0,
            train_cam_uids: Optional[List] = None,
            **kwargs
    ) -> 'MultiSurfaceSplineModel':
        """
        Create MultiSurfaceSplineModel from point cloud.

        Args:
            points: [N, 3] XYZ coordinates
            colors: [N, 3] RGB values
            config: NurbsOptimizationParams
            args: Training arguments
            mode:  Decomposition mode
            resolution:  Control grid resolution per surface
            spatial_lr_scale: Learning rate scaling
            train_cam_uids: Camera UIDs for multi-view sampling
            **kwargs: Additional parameters
        """
        from . import SplineModel  # Import here to avoid circular

        # Create NURBS surfaces from point cloud
        result = create_nurbs_from_pointcloud(
            points, colors,
            resolution=resolution,
            mode=mode,
            **kwargs
        )

        # Create SplineModel for each surface
        surfaces = []
        labels = []

        for i, surf_data in enumerate(result.surfaces):
            # Convert to geomdl format
            from geomdl import BSpline

            geo_surf = BSpline.Surface()
            geo_surf.degree_u = surf_data.degree_u
            geo_surf.degree_v = surf_data.degree_v

            H, W, _ = surf_data.control_points.shape
            ctrlpts = []
            for ii in range(H):
                for jj in range(W):
                    ctrlpts.append(surf_data.control_points[ii, jj].tolist())
            geo_surf.set_ctrlpts(ctrlpts, H, W)
            geo_surf.knotvector_u = surf_data.knots_u.tolist()
            geo_surf.knotvector_v = surf_data.knots_v.tolist()

            # Color surface
            rgb_surf = BSpline.Surface()
            rgb_surf.degree_u = surf_data.degree_u
            rgb_surf.degree_v = surf_data.degree_v
            rgb_ctrlpts = []
            for ii in range(H):
                for jj in range(W):
                    rgb_ctrlpts.append(surf_data.control_colors[ii, jj].tolist())
            rgb_surf.set_ctrlpts(rgb_ctrlpts, H, W)
            rgb_surf.knotvector_u = surf_data.knots_u.tolist()
            rgb_surf.knotvector_v = surf_data.knots_v.tolist()

            # Create SplineModel
            spline_model = SplineModel(
                surf=[geo_surf],
                surf_rgb=[rgb_surf],
                config=config,
                args=args,
                spatial_lr_scale=spatial_lr_scale,
                train_cam_uids=train_cam_uids or [0],
                late_init=False
            )

            surfaces.append(spline_model)
            labels.append(surf_data.label)

        # Convert labels to tensor
        point_labels = torch.tensor(result.labels, dtype=torch.long)

        return cls(surfaces, labels, mode, point_labels)

    @classmethod
    def from_geomdl_surfaces(
            cls,
            surf_list: List,  # List of geomdl surfaces
            surf_rgb_list: List,  # List of geomdl color surfaces
            config,
            args,
            labels: Optional[List[str]] = None,
            mode: DecompositionMode = DecompositionMode.SINGLE,
            spatial_lr_scale: float = 1.0,
            train_cam_uids: Optional[List] = None
    ) -> 'MultiSurfaceSplineModel':
        """
        Create from pre-existing geomdl surfaces.
        """
        from . import SplineModel

        surfaces = []

        for i, (surf, surf_rgb) in enumerate(zip(surf_list, surf_rgb_list)):
            spline_model = SplineModel(
                surf=[surf],
                surf_rgb=[surf_rgb],
                config=config,
                args=args,
                spatial_lr_scale=spatial_lr_scale,
                train_cam_uids=train_cam_uids or [0],
                late_init=False
            )
            surfaces.append(spline_model)

        if labels is None:
            labels = [f"surface_{i}" for i in range(len(surfaces))]

        return cls(surfaces, labels, mode)

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, viewpoint_cam, **kwargs):
        """
        Forward pass for all active surfaces.
        """
        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if active:
                surface.forward(viewpoint_cam, **kwargs)

    # =========================================================================
    # Combined Gaussian Properties
    # =========================================================================

    @property
    def get_xyz(self) -> torch.Tensor:
        """Combined XYZ from all active surfaces."""
        xyz_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                xyz_list.append(surface.get_xyz)
        return torch.cat(xyz_list, dim=0) if xyz_list else torch.empty(0, 3)

    @property
    def get_features(self) -> torch.Tensor:
        """Combined SH features from all active surfaces."""
        feat_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                feat_list.append(surface.get_features)
        return torch.cat(feat_list, dim=0) if feat_list else torch.empty(0, 0, 3)

    @property
    def get_opacity(self) -> torch.Tensor:
        """Combined opacities from all active surfaces."""
        opac_list = []
        for surface, active, weight in zip(
                self.surfaces, self._active_surfaces, self._surface_weights
        ):
            if active:
                opac = surface.get_opacity
                # Apply surface weight
                opac_list.append(opac * weight)
        return torch.cat(opac_list, dim=0) if opac_list else torch.empty(0, 1)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Combined scaling from all active surfaces."""
        scale_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                scale_list.append(surface.get_scaling)
        return torch.cat(scale_list, dim=0) if scale_list else torch.empty(0, 3)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Combined rotations from all active surfaces."""
        rot_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                rot_list.append(surface.get_rotation)
        return torch.cat(rot_list, dim=0) if rot_list else torch.empty(0, 4)

    @property
    def active_sh_degree(self) -> int:
        """Active SH degree (assumes same for all surfaces)."""
        return self.surfaces[0].active_sh_degree if self.surfaces else 0

    # =========================================================================
    # Surface Management
    # =========================================================================

    def set_surface_active(self, idx: int, active: bool):
        """Enable/disable a surface."""
        if 0 <= idx < self.num_surfaces:
            self._active_surfaces[idx] = active

    def set_surface_weight(self, idx: int, weight: float):
        """Set opacity weight for a surface."""
        if 0 <= idx < self.num_surfaces:
            self._surface_weights[idx] = weight

    def get_surface_by_label(self, label: str) -> Optional['SplineModel']:
        """Get surface by label name."""
        for surf, lbl in zip(self.surfaces, self.labels):
            if lbl == label:
                return surf
        return None

    def get_surface_indices(self) -> torch.Tensor:
        """
        Get per-Gaussian surface indices.
        Useful for surface-specific loss computation.
        """
        indices = []
        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if active:
                n_gaussians = surface.state.Us * surface.state.Vs
                indices.append(torch.full((n_gaussians,), i, dtype=torch.long))
        return torch.cat(indices, dim=0) if indices else torch.empty(0, dtype=torch.long)

    # =========================================================================
    # Training
    # =========================================================================

    def training_setup(self, training_args):
        """Setup training for all surfaces."""
        for surface in self.surfaces:
            surface.training_setup(training_args)

    @property
    def optimizer(self):
        """Get combined optimizer (returns list for multi-surface)."""
        return [s.optimizer for s in self.surfaces]

    def update_learning_rate(self, iteration: int):
        """Update learning rates for all surfaces."""
        for surface in self.surfaces:
            surface.update_learning_rate(iteration)

    def update_parameters(self, iteration: int):
        """Update iteration-dependent parameters."""
        for surface in self.surfaces:
            surface.update_parameters(iteration)

    def oneupSHdegree(self):
        """Increase SH degree for all surfaces."""
        for surface in self.surfaces:
            surface.oneupSHdegree()

    def invalidate_all_caches(self):
        """Invalidate caches for all surfaces."""
        for surface in self.surfaces:
            surface.invalidate_all_caches()

    # =========================================================================
    # Optimization Step
    # =========================================================================

    def step(self):
        """Optimizer step for all surfaces."""
        for surface in self.surfaces:
            surface.optimizer.step()

    def zero_grad(self):
        """Zero gradients for all surfaces."""
        for surface in self.surfaces:
            surface.optimizer.zero_grad()

    # =========================================================================
    # Densification
    # =========================================================================

    def subdivide_grid(self, top_k: int = 3):
        """Subdivide grids for all surfaces."""
        for surface in self.surfaces:
            surface.subdivide_grid_interleaved(top_k=top_k)

    # =========================================================================
    # Serialization
    # =========================================================================

    def capture(self) -> Dict:
        """Capture state of all surfaces."""
        return {
            'surfaces': [s.capture() for s in self.surfaces],
            'labels': self.labels,
            'decomposition_mode': self.decomposition_mode.value,
            'active_surfaces': self._active_surfaces,
            'surface_weights': self._surface_weights,
            'point_labels': self.point_labels.cpu().numpy() if self.point_labels is not None else None
        }

    def restore(self, state_dict: Dict, train_model: bool = False):
        """Restore state for all surfaces."""
        for surface, surf_state in zip(self.surfaces, state_dict['surfaces']):
            surface.restore(surf_state, train_model=train_model)

        self.labels = state_dict['labels']
        self.decomposition_mode = DecompositionMode(state_dict['decomposition_mode'])
        self._active_surfaces = state_dict['active_surfaces']
        self._surface_weights = state_dict['surface_weights']

        if state_dict['point_labels'] is not None:
            self.point_labels = torch.tensor(state_dict['point_labels'], dtype=torch.long)

    def save(self, path: str):
        """Save model to disk."""
        import pickle
        state = self.capture()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, config, args, train_model: bool = False) -> 'MultiSurfaceSplineModel':
        """Load model from disk."""
        import pickle
        from . import SplineModel

        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Create empty surfaces
        surfaces = []
        for surf_state in state['surfaces']:
            surface = SplineModel(late_init=True, config=config, args=args)
            surface.restore(surf_state, train_model=train_model)
            surfaces.append(surface)

        model = cls(
            surfaces=surfaces,
            labels=state['labels'],
            decomposition_mode=DecompositionMode(state['decomposition_mode'])
        )

        model._active_surfaces = state['active_surfaces']
        model._surface_weights = state['surface_weights']

        if state['point_labels'] is not None:
            model.point_labels = torch.tensor(state['point_labels'], dtype=torch.long)

        return model

    # =========================================================================
    # Utility
    # =========================================================================

    @property
    def total_gaussians(self) -> int:
        """Total number of Gaussians across all active surfaces."""
        total = 0
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                total += surface.state.Us * surface.state.Vs
        return total

    @property
    def parameters_count(self) -> int:
        """Total parameter count across all surfaces."""
        return sum(s.parameters_count for s in self.surfaces)

    def get_normal(self, view_cam) -> torch.Tensor:
        """Combined normals from all active surfaces."""
        normal_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                normal_list.append(surface.get_normal(view_cam))
        return torch.cat(normal_list, dim=0) if normal_list else torch.empty(0, 3)

    def __repr__(self) -> str:
        active_count = sum(self._active_surfaces)
        return (
            f"MultiSurfaceSplineModel(\n"
            f"  num_surfaces={self.num_surfaces},\n"
            f"  active={active_count},\n"
            f"  mode={self.decomposition_mode.value},\n"
            f"  labels={self.labels},\n"
            f"  total_gaussians={self.total_gaussians}\n"
            f")"
        )