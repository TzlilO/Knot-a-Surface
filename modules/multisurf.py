"""
Multi-Surface SplineModel Extension

Handles multiple NURBS surfaces for:
1. Background/Object separation
2. K-component decomposition

Each surface is a separate SplineModel instance with shared training infrastructure.
"""
from collections import defaultdict

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from modules.KnotSurface import SplineModel, SamplingMode
from modules.fitting.nurbs_from_pointcloud import (
    DecompositionMode,
    create_nurbs_from_pointcloud,
)
from modules.knotvector import KnotVector
from utils.general_utils import get_expon_lr_func


@dataclass
class BatchedGaussians:
    """Holds batched Gaussian properties for efficient rendering."""
    xyz: torch.Tensor = None  # [total_N, 3]
    features: torch.Tensor = None  # [total_N, SH, 3]
    opacity: torch.Tensor = None  # [total_N, 1]
    scaling: torch.Tensor = None  # [total_N, 3]
    rotation: torch.Tensor = None  # [total_N, 4]
    surface_indices: torch.Tensor = None  # [total_N] - which surface each Gaussian belongs to

    @property
    def num_gaussians(self) -> int:
        return self.xyz.shape[0]


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


class UnifiedInterpolator:
    """
    Performs batched interpolation of all control features within each surface,
    then aggregates across surfaces efficiently.

    Key insight: Within a surface, basis functions are shared across all features.
    We exploit this by concatenating features before interpolation.
    """

    def __init__(self, multi_surface_model: 'MultiSurfaceSplineModel'):
        self.model = multi_surface_model
        self._feature_specs = self._build_feature_specs()

    def _build_feature_specs(self) -> List[Dict]:
        """
        Build specification of features for each surface.
        Returns list of dicts with channel counts per feature type.
        """
        specs = []
        for surf in self.model.surfaces:
            spec = {
                'H': surf.state.H,
                'W': surf.state.W,
                'Us': surf.state.Us,
                'Vs': surf.state.Vs,
                'position_ch': surf.position.feature_channels,
                'scaling_ch': surf.scaling.feature_channels,
                'rotation_ch': surf.rotation.feature_channels,
                'opacity_ch': surf.opacity.feature_channels,
                'sh_dc_ch': surf.spherical_harmonics.sh_dc.feature_channels,
                'sh_rest_ch': surf.spherical_harmonics.sh_rest.feature_channels,
            }
            spec['total_ch'] = sum([
                spec['position_ch'],
                spec['scaling_ch'],
                spec['rotation_ch'],
                spec['opacity_ch'],
                spec['sh_dc_ch'],
                spec['sh_rest_ch'],
            ])
            specs.append(spec)
        return specs

    def interpolate_all(self, cache: bool = True) -> 'BatchedGaussians':
        """
        Perform unified interpolation for all surfaces.

        Returns:
            BatchedGaussians with all properties concatenated
        """
        all_xyz = []
        all_scaling = []
        all_rotation = []
        all_opacity = []
        all_features = []
        all_indices = []

        for surf_idx, (surface, spec) in enumerate(zip(self.model.surfaces, self._feature_specs)):
            if not self.model._active_surfaces[surf_idx]:
                continue

            # Perform batched interpolation for this surface
            result = self._interpolate_surface_batched(surface, spec)

            n_gaussians = surface.state.Us * surface.state.Vs

            all_xyz.append(result['xyz'])
            all_scaling.append(result['scaling'])
            all_rotation.append(result['rotation'])
            all_opacity.append(result['opacity'])
            all_features.append(result['features'])
            all_indices.append(
                torch.full((n_gaussians,), surf_idx, dtype=torch.long, device='cuda')
            )

        if not all_xyz:
            device = self.model.device
            return BatchedGaussians(
                xyz=torch.empty(0, 3, device=device),
                scaling=torch.empty(0, 3, device=device),
                rotation=torch.empty(0, 4, device=device),
                opacity=torch.empty(0, 1, device=device),
                features=torch.empty(0, 0, 3, device=device),
                surface_indices=torch.empty(0, dtype=torch.long, device=device)
            )

        gaussians = BatchedGaussians(
            xyz=torch.cat(all_xyz, dim=0),
            scaling=torch.cat(all_scaling, dim=0),
            rotation=torch.cat(all_rotation, dim=0),
            opacity=torch.cat(all_opacity, dim=0),
            features=torch.cat(all_features, dim=0),
            surface_indices=torch.cat(all_indices, dim=0)
        )

        if cache:
            self.model._cached_gaussians = gaussians
            self.model._cache_valid = True

        return gaussians

    def _interpolate_surface_batched(
            self,
            surface: 'SplineModel',
            spec: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single batched interpolation for all features of one surface.

        Key optimization: Concatenate all control features, interpolate once,
        then split the result.
        """
        import opt_einsum as oe

        Us, Vs = surface.state.Us, surface.state.Vs
        H, W = surface.state.H, surface.state.W

        # === 1. Concatenate all control features ===
        # Each has shape (H*W, channels) - reshape to (H, W, channels)

        pos_ctrl = surface.position.features  # May include weights for rational
        scale_ctrl = surface.scaling.control_features.view(H, W, -1)
        rot_ctrl = surface.rotation.control_features.view(H, W, -1)
        opa_ctrl = surface.opacity.control_features.view(H, W, -1)
        sh_dc_ctrl = surface.spherical_harmonics.sh_dc.control_features.view(H, W, -1)
        sh_rest_ctrl = surface.spherical_harmonics.sh_rest.control_features.view(H, W, -1)

        # Handle position special case (may have weights prepended)
        if surface.position.is_rational:
            pos_ch = pos_ctrl.shape[-1]  # includes weight
        else:
            pos_ctrl = pos_ctrl.view(H, W, -1)
            pos_ch = 3

        # Concatenate along channel dimension
        all_controls = torch.cat([
            pos_ctrl.view(H, W, -1),
            scale_ctrl,
            rot_ctrl,
            opa_ctrl,
            sh_dc_ctrl,
            sh_rest_ctrl
        ], dim=-1)  # Shape: (H, W, total_channels)

        total_ch = all_controls.shape[-1]

        # === 2. Single batched interpolation ===
        # Use einsum with pre-computed basis
        bu = surface.basis.bu  # (Us, H) or (Us, Vs, H)
        bv = surface.basis.bv  # (Vs, W) or (Us, Vs, W)

        # Determine einsum path based on basis shape
        # if bu.dim() == 2:
            # Separable basis:  (Us, H), (Vs, W)
        if surface.state.full_basis:
            # Full grid basis: (Us, Vs, H), (Us, Vs, W)
            contract_path = 'uvh,uvw,hwc->uvc'
        else:
            contract_path = 'uh,vw,hwc->uvc'

        # Single interpolation for all channels
        interpolated = oe.contract(
            contract_path,
            bu, bv, all_controls,
            optimize='optimal'
        )  # Shape: (Us, Vs, total_channels)

        # === 3. Handle rational case ===
        # if surface.position.is_rational:
        #     # First channel is weight, normalize
        #     weights = interpolated[..., 0:1].clamp(min=1e-6)
        #     interpolated = interpolated[..., 1:] / weights
        #     pos_ch = 3  # After removing weight

        # === 4. Split result back into individual features ===
        idx = 0

        # Position
        xyz = interpolated[..., idx:idx + 3].reshape(-1, 3)
        idx += pos_ch if not surface.position.is_rational else 3

        # Scaling
        scale_ch = spec['scaling_ch']
        scaling_raw = interpolated[..., idx:idx + scale_ch].reshape(-1, scale_ch)

        scaling = surface.scaling_activation(scaling_raw)
        # Pad to 3D if needed
        if scale_ch == 2:
            scaling = torch.cat([
                scaling,
                torch.full((scaling.shape[0], 1), 1e-7, device=scaling.device)
            ], dim=-1)
        idx += scale_ch

        # Rotation
        rot_ch = spec['rotation_ch']
        rotation_raw = interpolated[..., idx:idx + rot_ch].reshape(-1, rot_ch)
        rotation = surface.rotation_activation(rotation_raw, dim=-1)
        idx += rot_ch

        # Opacity
        opa_ch = spec['opacity_ch']
        opacity_raw = interpolated[..., idx:idx + opa_ch].reshape(-1, opa_ch)
        opacity = surface.opacity_activation(opacity_raw)
        idx += opa_ch

        # SH DC
        sh_dc_ch = spec['sh_dc_ch']
        sh_dc = interpolated[..., idx:idx + sh_dc_ch].reshape(-1, 1, 3)
        idx += sh_dc_ch

        # SH Rest
        sh_rest_ch = spec['sh_rest_ch']
        sh_rest = interpolated[..., idx:idx + sh_rest_ch]
        shc = surface.state.shc
        sh_rest = sh_rest.reshape(-1, shc - 1, 3)

        # Combine SH
        features = torch.cat([sh_dc, sh_rest], dim=1)  # (N, shc, 3)

        return {
            'xyz': xyz,
            'scaling': scaling,
            'rotation': rotation,
            'opacity': opacity,
            'features': features,
        }
class MultiSurfaceSplineModel(nn.Module):
    """
    Container for multiple SplineModel instances.

    Enables:
    - Training multiple surfaces jointly
    - Rendering with proper compositing
    - Per-surface control over optimization

    Usage:
        >>> # From point cloud
        >>> model = MultiSurfaceSplineModel.from_pointcloud(
        ...      points, colors,
        ...     config=config,
        ...     args=args,
        ...     mode=DecompositionMode.BACKGROUND_OBJECT
        ... )

        >>> # Forward pass returns combined Gaussians
        >>> model. forward(camera)
        >>> xyz = model.get_xyz  # Combined from all surfaces
    """
    # Cached combined properties
    _cache_valid = False
    _cached_gaussians: Optional[BatchedGaussians] = None
    _surface_offsets = None
    def __init__(
            self,
            surfaces: List['SplineModel'] = None,  # Avoid circular import
            labels: List[str] = None,
            decomposition_mode: DecompositionMode = DecompositionMode.SINGLE,
            point_labels: Optional[torch.Tensor] = None,
            device: str = 'cuda',
            **kwargs
    ):
        super().__init__()
        self.point_labels = point_labels
        self.decomposition_mode = decomposition_mode
        if surfaces is not None:
            self._surfaces = nn.ModuleList(surfaces)  # type: List[SplineModel]
            self.labels = labels
            self.num_surfaces = len(surfaces)
            self.device = device

            # Surface configuration
            self._active_surfaces = [True] * self.num_surfaces
            self._surface_weights = [1.0] * self.num_surfaces

            # Unified optimizer (created in training_setup)
            self._optimizer: Optional[torch.optim.Optimizer] = None

            # if kwargs.get('setup_training', True):
            #     self.training_setup()
            self._update_surface_offsets()
            self._interpolator: Optional[UnifiedInterpolator] = None
            self._tessellator: Optional['ViewDependentTessellator'] = None
            torch.cuda.empty_cache()
            # Add at the end of your existing multisurf.py, after the class definition
            from ply_export import add_save_ply_to_multi_surface_model

            # Add save_ply methods to MultiSurfaceSplineModel
            add_save_ply_to_multi_surface_model(MultiSurfaceSplineModel)

            # Gradient accumulation settings
            self._accumulation_steps = 1
            self._current_step = 0
            self._should_sync_grads = True

    @property
    def surfaces(self) -> List['SplineModel']:
        return [s for s, active in zip(self._surfaces, self._active_surfaces) if active]

    @property
    def tessellator(self) -> 'ViewDependentTessellator':
        """Lazily initialize and return the view-dependent tessellator."""
        if self._tessellator is None:
            from modules.tessellation.ViewDependentTessellator import ViewDependentTessellator
            self._tessellator = ViewDependentTessellator()
        return self._tessellator

    def _update_surface_offsets(self):
        """Pre-compute cumulative offsets for surface indexing."""
        offsets = [0]

        for surface in self.surfaces:
            n = surface.state.Us * surface.state.Vs
            offsets.append(offsets[-1] + n)
            # surface.offset = offsets[-2]
        self._surface_offsets = torch.tensor(offsets, device=self.device)
        # self.register_buffer('_surface_offsets', )

    # =========================================================================
    # Unified Optimizer
    # =========================================================================
    # Proposed modification to multisurf.py training_setup
    def create_adaptive_scheduler(self, surface, training_args):
        """Scheduler that resets after knot insertion."""

        base_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * surface.spatial_lr_scale,
            lr_final=training_args.position_lr_final * surface.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps
        )

        def adaptive_lr(step):
            # Track steps since last subdivision
            steps_since_subdivision = step - surface._last_subdivision_step

            # Warm restart: temporarily boost LR after subdivision
            warmup_steps = getattr(training_args, 'subdivision_warmup_steps', 1000)
            if steps_since_subdivision < warmup_steps:
                warmup_factor = 0.5 + 0.5 * (steps_since_subdivision /
                                             warmup_steps)
                restart_lr = training_args.position_lr_init * surface.spatial_lr_scale * 0.3
                return restart_lr * warmup_factor

            # Otherwise use base exponential decay from subdivision point
            return base_scheduler(steps_since_subdivision)

        return adaptive_lr

    def local_planar_deviation_loss(self, weight=1.0):
        if weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        deviation_loss = 0.0
        for surface in self.surfaces:
            deviation_loss = surface.local_planar_deviation_loss() + deviation_loss
        return weight * deviation_loss
    def training_setup(self, **kwargs):
        """
        Setup single unified optimizer for all surfaces.
        This is the key optimization - one optimizer. step() instead of N.
        """
        all_param_groups = []
        train_args = {}
        # for surface_idx, surface in enumerate(self.surfaces):
        #     surface.training_setup()
        # 1. Initialize Scaler
        # scaler = SurfaceCharacteristicScaler(
        #     base_scale=self.surfaces[0].spatial_lr_scale,
        #     min_scale=0.1,
        #     max_scale=10.
        # )

        # 2. Compute scales for all surfaces
        # This analyzes the geometry initialized from the point cloud
        # position_scales = scaler.compute_scale_factors(self.surfaces)
        # 3. Build Optimizer Groups
        for suff, surface in enumerate(self.surfaces):
            training_args = surface.state.opt

            # --- APPLY SCALING HERE ---
            # print(f"  Adaptive LR Scale: {adaptive_lr_scale:.4f} (Spatial: {surface.spatial_lr_scale:.4f}, Characteristic: {position_scales[suff]:.4f})")
            train_args[suff] = training_args
            # Get base learnixng rates from surface
            # surface.spatial_lr_scale *= adaptive_lr_scale
            obj_factor = 1 if len(self.surfaces)== 1 else 1.
            spatial_lr_scale = surface.spatial_lr_scale
            obj_pos_factor = 1 if not surface.is_background else 1.0
            background_lr_scale_factor_pos = training_args.background_lr_scale_factor
            background_lr_scale_factor_texture = training_args.background_lr_scale_factor
            background_lr_scale_factor_dummy = 1
            feature_lr = training_args.feature_lr
            opacity_lr = training_args.opacity_lr
            scaling_lr = training_args.scaling_lr
            rotation_lr = training_args.rotation_lr
            knot_lr = training_args.knot_lr
            uv_lr = training_args.uv_lr_factor

            if surface.state.opt.refine_weights:
                all_param_groups.append({
                    'params': [surface.weights.control_features],
                    'lr': training_args.nurbs_weight_lr,
                    'name': surface.weights.name
                })
            all_param_groups.append({
                'params': [surface.position.control_features],
                'lr': training_args.position_lr_init * spatial_lr_scale, #,
                'name': surface.position.name
            })
            surface.scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final* spatial_lr_scale, #* adaptive_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

            # SH
            all_param_groups.append({
                'params': [surface.spherical_harmonics.sh_dc.control_features],
                'lr': feature_lr,
                'name': surface.spherical_harmonics.sh_dc.name,
            })
            all_param_groups.append({
                'params': [surface.spherical_harmonics.sh_rest.control_features],
                'lr': feature_lr / 20,
                'name': surface.spherical_harmonics.sh_rest.name,
            })

            # Opacity
            if surface.refine_opacity_active:
                all_param_groups.append({
                    'params': [surface.opacity.control_features],
                    'lr': opacity_lr,
                    'name': surface.opacity.name
                })

            # Scaling
            if surface.refine_scales_active:
                all_param_groups.append({
                    'params': [surface.scaling.control_features],
                    'lr': scaling_lr, # * adaptive_lr_scale,
                    'name': surface.scaling.name
                })


            # Rotation
            if surface.refine_rotations_active:
                all_param_groups.append({
                    'params': [surface.rotation.control_features],
                    'lr': rotation_lr,
                    'name': surface.rotation.name
                })

            # Knots (if optimizable)
            # u_factor = 1/surface.state.H
            # v_factor = 1/surface.state.W
            if surface.state.opt.optimize_knots:
                u_factor = torch.diff(surface.knot_u.internal_knots / 2)
                u_factor = u_factor[u_factor > 0].min()  # Avoid zero or negative factors
                v_factor = torch.diff(surface.knot_v.internal_knots / 2) #.nonzero().min(
                v_factor = v_factor[v_factor > 0].min()

                all_param_groups.append({
                    'params': [surface.knot_u._internal_knots],
                    'lr': knot_lr * u_factor,  # Scale by min interval to keep updates stable
                    'name': surface.knot_u.name
                })
                all_param_groups.append({
                    'params': [surface.knot_v._internal_knots],
                    'lr': knot_lr * v_factor,  # Scale by min interval to keep updates stable
                    'name': surface.knot_v.name

                })
            if surface.state.opt.optimize_intervals:
                u_name = surface.basis.uv_sampler.u_name
                v_name = surface.basis.uv_sampler.v_name
                delta_u = surface.basis.uv_sampler.delta_u
                delta_v = surface.basis.uv_sampler.delta_v
                delta_u = delta_u[delta_u > 0].min()  # Avoid zero or negative factors
                delta_v = delta_v[delta_v > 0].min()
                all_param_groups.append({
                    'params': [surface.basis.uv_sampler._interval_u],
                    'lr': uv_lr * delta_u,  # Scale by min interval to keep updates stable
                    'name': u_name,
                    # 'eps': 1e-8

                })
                all_param_groups.append({
                    'params': [surface.basis.uv_sampler._interval_v],
                    'lr': uv_lr * delta_v,  # Scale by min interval to keep updates stable
                    'name': v_name,
                    # 'eps':1e-8
                })
        self._optimizer = torch.optim.Adam(all_param_groups, lr=0.0, eps=1e-15)
        def invalidate_hook(optimizer, args, kwargs):
            self._invalidate_cache(True)  # Your invalidation logic here

        self._training_args = training_args

    def _create_schedulers(self, args):
        """Create LR schedulers for different parameter groups."""

        return {
            'xyz': get_expon_lr_func(
                lr_init=args.position_lr_init,
                lr_final=args.position_lr_final,
                max_steps=args.position_lr_max_steps
            )
        }

    def zero_grad(self):
        """Zero gradients for all surfaces."""
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    def update_learning_rate(self, iteration: int):
        """Update LR for all parameter groups."""
        if self._optimizer is None:
            return
        lr = {}
        for surface_idx, surface in enumerate(self.surfaces):
            lr[surface.label] = (surface.update_learning_rate(iteration, self.optimizer))
        # if iteration % 100 == 0:
            # print(f"[MultiSurfaceSplineModel] Iteration {iteration} - Learning Rates: {lr}")
        return lr


    @classmethod
    def from_pointcloud(
            cls,
            points: Union[np.ndarray, torch.Tensor],
            colors: Optional[Union[np.ndarray, torch.Tensor]],
            config,  # NurbsOptimizationParams
            args,  # Training args
            decomposition_mode: DecompositionMode = DecompositionMode.SINGLE,
            resolution: Tuple[int, int] = (32, 32),
            nerf_radius: float = 1.0,
            nerf_translate=0.0,
            train_cam_uids: Optional[List] = None,
            bg_resolution_scale: float = 1.0,
            object_resolution_scale: float = 2.0,
            cameras=None,
            faces=None,  # NEW
            use_least_squares=True,  # NEW

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



        # Create NURBS surfaces from point cloud
        result = create_nurbs_from_pointcloud(
            points, colors,
            resolution=resolution,
            mode=decomposition_mode,
            generate_adaptive_samples=False,  # We'll handle sampling adaptively within SplineModel
            nerf_radius=nerf_radius,
            nerf_translate=nerf_translate,
            bg_resolution_scale=bg_resolution_scale,
            object_resolution_scale=object_resolution_scale,
            cameras=cameras,
            faces = faces,
            use_least_squares = use_least_squares,
            ** kwargs)

        # Create SplineModel for each surface
        surfaces = []
        labels = []

        for i, surf_data in enumerate(result.surfaces):
            # Convert to geomdl format
            # from geomdl import BSpline
            # print(f"[MultiSurfaceSplineModel] Initializing surface {i} with label '{surf_data.label}' - Control Points: {surf_data.control_points.shape}, Knots U: {len(surf_data.knots_u)}, Knots V: {len(surf_data.knots_v)}")
            # print(f"  Sample control points: {surf_data.control_points[0, 0]}, {surf_data.control_points[-1, -1]}")
            # surf_data.
            spline_model = SplineModel(
                surf_data=surf_data.to_dict(),
                config=config,
                args=args,
                spatial_lr_scale=nerf_radius,
                train_cam_uids=train_cam_uids or [0],
                late_init=False,
                surf_uid=i,
                label=surf_data.label,
                is_background=(surf_data.label == 'background'),
                cameras=cameras,
                **kwargs
            )

            surfaces.append(spline_model)
            labels.append(surf_data.label)

        point_labels = torch.tensor(result.labels, dtype=torch.long)
        del result
        return cls(surfaces, labels, decomposition_mode, point_labels)

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
    def densify_sampling_density(self, quant=0.05):
        """Update sampling density for all surfaces based on camera."""

        for surface in self.surfaces:
            max_density = surface.state.max_sampling_density
            old_density = surface.state.sampling_density
            if old_density >= max_density:
                continue
            new_density = min(old_density + quant, max_density)
            surface.update_sampling_density(new_density)
            surface.invalidate_all_caches(force=True)
            surface.state.init_grad_accumulators()


        self._update_surface_offsets()
        self._invalidate_cache(force=True)

    def update_surface_blend(self):
        """Update sampling density for all surfaces based on camera."""

        for surface in self.surfaces:
            for module in surface.control_list:
                try:

                    prev_alpha_blend = module.blending_alpha
                    new_alpha_blend = max(prev_alpha_blend - 0.1, .0)

                    module.set_alpha(new_alpha=new_alpha_blend)
                except AttributeError:
                    print(f"Warning: Module {module} does not have set_alpha method.")
                    continue
                # new_alpha_blend = min(prev_alpha_blend + 0.1, 1.0)

        # self._invalidate_cache(force=True)
    # =========================================================================
    # Combined Gaussian Properties
    # =========================================================================
    @property
    def get_all_scaling_controls(self) -> torch.Tensor:
        """Combined scaling control points from all active surfaces."""
        cp_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                cp_list.append(surface.scaling.features)
        return torch.stack(cp_list, dim=0) if cp_list else torch.empty(0, 3)

    @property
    def get_all_rotation_controls(self) -> torch.Tensor:
        """Combined rotation control points from all active surfaces."""
        cp_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                cp_list.append(surface.rotation.features)
        return torch.stack(cp_list, dim=0) if cp_list else torch.empty(0, 4)

    @property
    def get_all_opacity_controls(self) -> torch.Tensor:
        """Combined opacity control points from all active surfaces."""
        cp_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                cp_list.append(surface.opacity.features)
        return torch.stack(cp_list, dim=0) if cp_list else torch.empty(0, 1)

    @property
    def get_all_feature_controls(self) -> torch.Tensor:
        """Combined SH feature control points from all active surfaces."""
        feat_dc = []
        feat_rest = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                feat_dc.append(surface.spherical_harmonics.sh_dc.features)
                feat_rest.append(surface.spherical_harmonics.sh_rest.features)
        dcs = torch.stack(feat_dc, dim=0) if feat_dc else torch.empty(0, surface.spherical_harmonics.sh_dc.num_features)
        rests = torch.stack(feat_rest, dim=0) if feat_rest else torch.empty(0, surface.spherical_harmonics.sh_rest.num_features)
        # feats_sh = torch.stack([dcs, rests], dim=-2)
        feats_sh = torch.cat([dcs, rests], dim=-1)
        return feats_sh #if feats_sh else torch.empty(0, surface.spherical_harmonics.num_features)
    @property
    def get_all_position_controls(self) -> torch.Tensor:
        """Combined control points from all active surfaces."""
        cp_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                cp_list.append(surface.position.features)
        return torch.stack(cp_list, dim=0) if cp_list else torch.empty(0, 3)


    @property
    def all_basis_u(self) -> torch.Tensor:
        basis_u_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                basis_u_list.append(surface.basis.bu)
        return torch.stack(basis_u_list, dim=0) if basis_u_list else torch.empty(0, surface.basis.bu.shape[1])
    @property
    def all_basis_v(self) -> torch.Tensor:
        basis_v_list = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                basis_v_list.append(surface.basis.bv)
        return torch.stack(basis_v_list, dim=0) if basis_v_list else torch.empty(0, surface.basis.bv.shape[1])

    @property
    def contract_path(self):
        return 'bfh,bhwc,bfw -> bfc' if self.surfaces[0].state.flatten_uv else 'buvh,buvw,bhwc->buvc'

    def _get_cached_gaussians(self) -> BatchedGaussians:
        """Get or compute cached Gaussian properties."""
        if self._cache_valid and self._cached_gaussians is not None:
            return self._cached_gaussians

        # Collect from active surfaces
        xyz_list = []
        feat_list = []
        opac_list = []
        scale_list = []
        rot_list = []
        idx_list = []

        for i, (surface, active, weight) in enumerate(
                zip(self.surfaces, self._active_surfaces, self._surface_weights)
        ):
            if not active:
                continue
            n = surface.state.Us * surface.state.Vs
            surface.recompute()
            xyz_list.append(surface.get_xyz)
            feat_list.append(surface.get_features)
            opac_list.append(surface.get_opacity)
            scale_list.append(surface.get_scaling)
            rot_list.append(surface.get_rotation)
            idx_list.append(torch.full((n,), i, dtype=torch.long, device=self.device))
            # rays_list.append()
            # with torch.no_grad():
            #     surface.ray_info()
        if not xyz_list:
            # No active surfaces
            self._cached_gaussians = BatchedGaussians(
                xyz=torch.empty(0, 3, device=self.device),
                features=torch.empty(0, 0, 3, device=self.device),
                opacity=torch.empty(0, 1, device=self.device),
                scaling=torch.empty(0, 3, device=self.device),
                rotation=torch.empty(0, 4, device=self.device),
                surface_indices=torch.empty(0, dtype=torch.long, device=self.device)
            )
        else:
            self._cached_gaussians = BatchedGaussians(
                xyz=torch.cat(xyz_list, dim=0).clone(),
                features=torch.cat(feat_list, dim=0).clone(),
                opacity=torch.cat(opac_list, dim=0).clone(),
                scaling=torch.cat(scale_list, dim=0).clone(),
                rotation=torch.cat(rot_list, dim=0).clone(),
                surface_indices=torch.cat(idx_list, dim=0)
            )
        self._cache_valid = True
        return self._cached_gaussians

    def _invalidate_cache(self, force=False):
        """Invalidate the cached Gaussians (all surfaces, then the aggregate)."""
        skip_aggregate = False
        for surface in self.surfaces:
            surface.invalidate_all_caches(force=force)
            if not force and surface.sampling_mode == SamplingMode.EVALUATION:
                skip_aggregate = True
        if skip_aggregate:
            return
        self._cache_valid = False
        self._cached_gaussians = None

    def update_uv_distribution_chhugani(self, camera):
        """Update UV sampling distribution for all active surfaces based on camera."""
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                surface.update_uv_distribution_chhugani(camera)

    @torch.no_grad()
    def refresh_chhugani_aggregation(
            self,
            cameras: List,
            max_views: int = 0,
            aggregation_mode: str = 'max',
    ):
        """
        Refresh Chhugani aggregation for all active surfaces.

        Call periodically in training loop:
            if iteration % aggregation_interval == 0:
                model.refresh_chhugani_aggregation(training_cameras)

        Args:
            cameras: All training cameras
            max_views: Max views to sample (0 = all)
            aggregation_mode: 'max', 'mean', or 'percentile_90'
        """
        from modules.tessellation.chhugani import refresh_aggregation

        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                refresh_aggregation(
                    surface, cameras,
                    max_views=max_views,
                    aggregation_mode=aggregation_mode,
                )

    @torch.no_grad()
    def mark_all_aggregations_stale(self):
        """Mark all surface aggregations as stale after geometry changes."""
        from modules.tessellation.chhugani import mark_aggregation_stale
        for surface in self.surfaces:
            mark_aggregation_stale(surface)
    @torch.no_grad()
    def apply_view_dependent_tessellation(self, camera, max_samples: int=None):
        """
        Applies adaptive tessellation to all active surfaces based on the camera view.

        Args:
            camera: Viewpoint camera.
            max_samples: Maximum grid dimension to prevent OOM.
        """


        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if active:
                # Compute new intervals
                Us, Vs = self.surfaces[0].state.Us, self.surfaces[0].state.Vs
                new_u, new_v = self.tessellator.compute_adaptive_intervals(
                    surface, camera, Us=Us, Vs=Vs
                )

                surface.uv_sampler.update_uv(new_u, new_v)

                surface.basis.forward(
                    surface.uv_sampler.forward(),
                    surface.knot_u(),
                    surface.knot_v()
                )


    @property
    def get_xyz(self) -> torch.Tensor:
        return self._get_cached_gaussians().xyz

    @property
    def get_features(self) -> torch.Tensor:
        return self._get_cached_gaussians().features

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._get_cached_gaussians().opacity

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._get_cached_gaussians().scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._get_cached_gaussians().rotation
    @property
    def xyz_grids(self) -> torch.Tensor:
        """Get per-surface XYZ grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                grids.append(surface.get_xyz.view(surface.state.Us, surface.state.Vs, 3))
        return grids


    @property
    def normal_grids(self) -> torch.Tensor:
        """Get per-surface normal grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                # normals = surface.get_real_normal(view_cam=viewpoint_cam)
                normals = nn.functional.normalize(surface.surface_normals(), dim=-1)
                grids.append(normals.view(surface.state.Us, surface.state.Vs, 3))
        return grids

    @property
    def global_normal_grids(self) -> torch.Tensor:
        """Get per-surface normal grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                # global_normals = nn.functional.normalize(surface.get_rotation_matrix().gather(2, torch.full((self.total_gaussians, 1), 2, device=self.device, dtype=torch.long)[..., None].expand(-1, 3, -1)).view(surface.grid_shape), dim=-1)

                global_normals = nn.functional.normalize(surface.get_smallest_axis(), dim=-1)
                grids.append(global_normals.view(surface.state.Us, surface.state.Vs, 3))
        return grids
    def weight_map_grids(self) -> torch.Tensor:
        """Get per-surface normal grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                # normals = surface.get_real_normal(view_cam=viewpoint_cam)
                weights = surface.weights_map()
                grids.append(weights.view(surface.state.Us, surface.state.Vs, 1))
        return grids

    @property
    def scaling_grids(self) -> torch.Tensor:
        """Get per-surface scaling grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                scaling = surface.get_scaling
                grids.append(scaling[..., :2].view(surface.state.Us, surface.state.Vs, 2))
        return grids

    @property
    def geo_scaling_grids(self) -> torch.Tensor:
        """Get per-surface scaling grids."""
        grids = []
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                scaling = surface.derive_scale()
                grids.append(scaling[..., :2].detach().view(surface.state.Us, surface.state.Vs, 2))
        return grids
    @property
    def rotation_grids(self) -> torch.Tensor:
        """Get per-surface rotation grids."""
        grids = []

        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                rotation = surface.get_rotation
                grids.append(rotation.view(surface.state.Us, surface.state.Vs, 4))
        return grids

    @property
    def surface_indices(self) -> torch.Tensor:
        """Which surface each Gaussian belongs to."""
        return self._get_cached_gaussians().surface_indices


    @property
    def interpolator(self) -> UnifiedInterpolator:
        """Lazy initialization of unified interpolator."""
        if self._interpolator is None:
            self._interpolator = UnifiedInterpolator(self)
        return self._interpolator

    def interpolate_all_unified(self, cache: bool = True) -> BatchedGaussians:
        """
        Perform unified batched interpolation across all surfaces.

        This is more efficient than accessing individual properties because:
        1. Single basis-feature contraction per surface (vs 6 separate ones)
        2. Reduces Python overhead from property access
        3. Better GPU utilization through larger operations

        Returns:
            BatchedGaussians with all Gaussian properties
        """
        return self.interpolator.interpolate_all(cache=cache)

    def forward_unified(self, viewpoint_cam, **kwargs) -> BatchedGaussians:
        """
        Forward pass using unified interpolation.

        More efficient alternative to forward() + property access.
        """
        self._invalidate_cache()

        # Forward each surface to update basis
        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if active:
                surface.forward(viewpoint_cam, **kwargs)

        # Single unified interpolation
        return self.interpolate_all_unified(cache=True)
    def interpolate_all(self, cache=True) -> torch.Tensor:
        import opt_einsum as oe
        cpts_w = self.get_all_position_controls
        scale = self.get_all_scaling_controls
        rot = self.get_all_rotation_controls
        opa = self.get_all_opacity_controls
        feat = self.get_all_feature_controls
        c = feat.shape[1] if feat.numel() > 0 else 0
        all_controls = torch.cat([cpts_w, scale, rot, opa, feat], dim=-1)
        prod = oe.contract(self.contract_path, self.all_basis_u, self.all_basis_v, all_controls).contiguous()  # Or basis_u @ controls.view(ctrl_u, -1).reshape(samples_u, ctrl_v * c).view(samples_u, ctrl_v, c)

        out_xyz = prod[..., :3]
        out_scale = prod[..., 3:6]
        out_rot = prod[..., 6:10]
        out_opa = prod[..., 10:11]
        out_feat = prod[..., 11:11 + c] if c > 0 else torch.empty(prod.shape[0], 0, device=prod.device)
        # self.gaussians = BatchedGaussians(
        #     xyz=out_xyz,
        #     features=out_feat,
        #     opacity=out_opa.sigmoid(),
        #     scaling=out_scale.exp(),
        #     rotation=torch.nn.functional.normalize(out_rot, dim=-1),
        #     surface_indices=self.get_surface_indices()
        # )
        # self.downsample(prod)
        self.cache = prod if cache else None
        return prod
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

    def train_set_unified(self, **kwargs):
        """
        Setup single unified optimizer for all surfaces.
        This is the key optimization - one optimizer. step() instead of N.
        """
        all_param_groups = []
        positions = []
        opacities = []
        scalings = []
        rotations = []
        features_dc = []
        features_rest = []
        knot_us = []
        knot_vs = []
        sampling_u = []
        sampling_v = []

        for surf_idx, surface in enumerate(self.surfaces):
            # Get base learning rates from surface
            training_args = surface.state.opt

            spatial_scale = surface.spatial_lr_scale
            positions.append(surface.position.control_features)
            scalings.append(surface.scaling.control_features)
            rotations.append(surface.rotation.control_features)
            opacities.append(surface.opacity.control_features)
            features_dc.append(surface.spherical_harmonics.sh_dc.control_features)
            features_rest.append(surface.spherical_harmonics.sh_rest.control_features)
            # Position
            # all_param_groups.append({
                # 'params': [surface.position.control_features],
                # 'lr': training_args.position_lr_init * spatial_scale,
                # 'name': f'xyz_s{surf_idx}'
            # })

            # SH
            # all_param_groups.append({
            #     'params': [surface.spherical_harmonics.sh_dc.control_features],
            #     'lr': training_args.feature_lr,
            #     'name': f'f_dc_s{surf_idx}'
            # })
            # all_param_groups.append({
            #     'params': [surface.spherical_harmonics.sh_rest.control_features],
            #     'lr': training_args.feature_lr / 20,
            #     'name': f'f_rest_s{surf_idx}'
            # })
            #
            # # Opacity
            # if surface.refine_opacity_active:
            #     all_param_groups.append({
            #         'params': [surface.opacity.control_features],
            #         'lr': training_args.opacity_lr,
            #         'name': f'opacity_s{surf_idx}'
            #     })
            #
            # # Scaling
            # if surface.refine_scales_active:
            #     all_param_groups.append({
            #         'params': [surface.scaling.control_features],
            #         'lr': training_args.scaling_lr * spatial_scale,
            #         'name': f'scaling_s{surf_idx}'
            #     })
            #
            # # Rotation
            # if surface.refine_rotations_active:
            #     all_param_groups.append({
            #         'params': [surface.rotation.control_features],
            #         'lr': training_args.rotation_lr,
            #         'name': f'rotation_s{surf_idx}'
            #     })

            # Knots (if optimizable)
            l = [{
                'params': positions, 'lr': training_args.position_lr_init * spatial_scale, 'name': 'xyz',
            }, {'params': features_dc, 'lr': training_args.feature_lr, 'name': 'f_dc'},
                {'params': features_rest, 'lr': training_args.feature_lr / 20, 'name': 'f_rest'},
                {'params': opacities, 'lr': training_args.opacity_lr, 'name': 'opacity'},
                {'params': scalings, 'lr': training_args.scaling_lr * spatial_scale, 'name': 'scaling'},
                {'params': rotations, 'lr': training_args.rotation_lr, 'name': 'rotation'}]
            if knot_us:
                l.append({'params': knot_us, 'lr': 0.0005, 'name': 'knot_u'})
            if knot_vs:
                l.append({'params': knot_vs, 'lr': 0.0005, 'name': 'knot_v'})
            if sampling_u:
                l.append({'params': sampling_u, 'lr': 0.0005, 'name': 'sampling_u'})
            if sampling_v:
                l.append({'params': sampling_v, 'lr': 0.0005, 'name': 'sampling_v'})

        self._optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    @property
    def use_app(self):
        return False

    def update_parameters(self, iteration: int):
        """Update iteration-dependent parameters."""
        BG_FREEZE_ITER = -1
        for surface in self.surfaces:
            surface.update_parameters(iteration)
            # if surface.is_background and iteration == BG_FREEZE_ITER:
            #     print(f"Freezing background parameters: {len(surface.control_list)}")
            #     for module in surface.control_list:
            #         print(f"Freezing background parameter {module.name} after iteration {iteration}")
            #         try:
            #             module.control_features.requires_grad_(False)
            #         except AttributeError:
            #             print(f"Warning: Module {module} does not have control_features attribute.")
            #             continue

    def oneupSHdegree(self):
        """Increase SH degree for all surfaces."""
        for surface in self.surfaces:
            surface.oneupSHdegree()


    def invalidate_all_caches(self):
        """Invalidate caches for all surfaces."""
        self._cache_valid = False
        self._cached_gaussians = None
        for surface in self.surfaces:
            surface.invalidate_all_caches()

    @property
    def optimizer(self):
        return self.surfaces[0].optimizer

    def add_subdivision_stats(self, mask, viewspace_points, viewspace_points_abs, visibility_filter, radii):
        """Log stats for all surfaces."""

        grad_norm = torch.norm(viewspace_points.grad[..., :2], dim=-1, keepdim=True)
        grad_norm_abs = torch.norm(viewspace_points_abs.grad[..., :2], dim=-1, keepdim=True)
        for i, surface in enumerate(self.surfaces):
            if surface.state.opt.subdiv_critertia in ['eikonal', 'spatial']:
                continue
            start_idx = self._surface_offsets[i]
            end_idx = self._surface_offsets[i + 1]
            surf_grad_norm = grad_norm[start_idx:end_idx]
            surf_grad_norm_abs = grad_norm_abs[start_idx:end_idx]
            surf_vis = visibility_filter[start_idx:end_idx]
            surf_radii = radii[start_idx:end_idx]
            surf_mask = mask[start_idx:end_idx] * surf_vis#[start_idx:end_idx]
            surface.state.add_subdivision_stats(surf_grad_norm, surf_grad_norm_abs, surf_mask, surf_radii, surf_vis)
    def state(self, uid=0):
        """Get state for a specific surface by UID."""
        return self.surfaces[uid].state

    def subdivide_surface(self,
                          grad_threshold: float,
                          grad_abs_threshold: float,
                          radii_threshold: float = 100.0,
                          top_k_rate: float = 0.,
                          foreach_surface_apply=True,
                          max_k: int = 2,
                          verbose: bool = False,
                          min_k: int = 0,
                          ):
        """
        Subdivide grids globally across all surfaces.
        Collects candidates from all surfaces, sorts them globally, and applies top_k.
        """
        from collections import defaultdict

        subdivision_candidates = []
        for surf in self.surfaces:
            subdivision_candidates.append(surf.get_subdivision_candidates(
                use_partitioning=surf.state.opt.use_spatial_partitioning,
                num_partitions=surf.state.opt.num_partitions))

        if verbose:
            total_candidates = sum(len(cands) for cands in subdivision_candidates)

        if foreach_surface_apply:
            for i, surface in enumerate(self.surfaces):

                min_k = max(min(min_k, int(0.5 * min(surface.state.H, surface.state.W))), 8)
                top_k = max(int(top_k_rate * min(surface.state.H, surface.state.W)), min_k)
                candidates = subdivision_candidates[i]

                candidates.sort(key=lambda x: x['score'], reverse=True)
                top_k_candidates = candidates[:top_k]

                # Apply batched subdivision directly
                surface.apply_subdivision(cands=top_k_candidates, optimizer=self._optimizer)

        else:
            # 1. Collect candidates from all surfaces
            all_candidates = []
            top_k = max(min(int(top_k_rate * min(self.state(0).H, self.state(0).W)), max_k), min_k)

            for i, surface in enumerate(self.surfaces):
                candidates = surface.get_subdivision_candidates(
                    use_partitioning=surface.state.opt.use_spatial_partitioning,
                    num_partitions=surface.state.opt.num_partitions,
                )
                if not candidates:
                    if verbose:
                        print(f"[Subdivision] No candidates exceed thresholds "
                              f"(grad={grad_threshold:.6f}, abs={grad_abs_threshold:.6f})")
                    return

                for cand in candidates:
                    cand['surf_idx'] = i
                all_candidates.extend(candidates)

            if not all_candidates:
                return

            # 2. Sort globally by score
            all_candidates.sort(key=lambda x: x['score'], reverse=True)

            # 3. Select top_k
            top_candidates = all_candidates[:top_k]

            # 4. Group candidates by surface to minimize offset recalculations
            candidates_by_surface = defaultdict(list)
            for cand in top_candidates:
                candidates_by_surface[cand['surf_idx']].append(cand)

            # 5. Apply batched subdivisions surface by surface
            for surf_idx in sorted(candidates_by_surface.keys()):
                surface = self.surfaces[surf_idx]
                surface_candidates = candidates_by_surface[surf_idx]

                # Apply batched subdivision directly
                surface.apply_subdivision(cands=surface_candidates, optimizer=self._optimizer)
    def subdivide_surfaces(self,
                           grad_threshold: float,
                           grad_abs_threshold: float,
                           radii_threshold: float = 100.0,
                           top_k_rate: float = 0.,
                           foreach_surface_apply=True,
                           max_k: int = 32,
                           verbose: bool = False,
                           min_k: int=12,
                           ):
        """
        Subdivide grids globally across all surfaces.
        Collects candidates from all surfaces, sorts them globally, and applies top_k.
        """
        subdivision_candidates = []
        for surf in self.surfaces:
            subdivision_candidates.append(surf.get_subdivision_candidates(
                use_partitioning=surf.state.opt.use_spatial_partitioning,
                num_partitions=surf.state.opt.num_partitions))
        if verbose:
            total_candidates = sum(len(cands) for cands in subdivision_candidates)
            print(f"[Subdivision] Total candidates across all surfaces: {total_candidates}")
            print("Using partitioning:", self.surfaces[0].state.opt.use_spatial_partitioning)
        if foreach_surface_apply:
            for i, surface in enumerate(self.surfaces):
                print(f"Before: Surface dimensions: H={surface.state.H}, W={surface.state.W}")

                min_k = max(min(min_k, int(0.5 * min(surface.state.H, surface.state.W))), 8)
                top_k = max(int(top_k_rate * min(surface.state.H, surface.state.W)), min_k)
                candidates = subdivision_candidates[i]
                print(f"Surface {i} - Candidates before filtering: {len(candidates)}")
                candidates.sort(key=lambda x: x['score'], reverse=True)
                top_k_candidates = candidates[:top_k]
                print(f"Surface {i} - Top {top_k} candidates selected for subdivision.")
                top_k_candidates.sort(key=lambda x: x['val'], reverse=True)
                for cand in top_k_candidates:
                    surface.subdivide_surface(cand=cand, optimizer=self._optimizer)
                print(f"After: Surface dimensions: H={surface.state.H}, W={surface.state.W}")

        else:
            # 1. Collect candidates from all surfaces
            all_candidates = []
            # min_dim = [(surface.state.H, surface.state.W) for surface in self.surfaces]

            top_k = max(min(int(top_k_rate * min(self.state(0).H, self.state(0).W)), max_k), min_k)

            for i, surface in enumerate(self.surfaces):
                candidates = surface.get_subdivision_candidates(
                    use_partitioning=surface.state.opt.use_spatial_partitioning,
                    num_partitions=surface.state.opt.num_partitions,
                )
                if not candidates:
                    if verbose:
                        print(f"[Subdivision] No candidates exceed thresholds "
                              f"(grad={grad_threshold:.6f}, abs={grad_abs_threshold:.6f})")
                    return

                for cand in candidates:
                    cand['surf_idx'] = i
                all_candidates.extend(candidates)

            if not all_candidates:
                return

            # 2. Sort globally by score
            all_candidates.sort(key=lambda x: x['score'], reverse=True)

            # 3. Select top_k
            top_candidates = all_candidates[:top_k]
            top_candidates.sort(key=lambda x: x['val'], reverse=True)
            # 4. Group candidates by surface to minimize offset recalculations
            candidates_by_surface = defaultdict(list)
            for cand in top_candidates:
                candidates_by_surface[cand['surf_idx']].append(cand)

            # 5. Apply subdivisions surface by surface
            for surf_idx in sorted(candidates_by_surface.keys()):
                surface = self.surfaces[surf_idx]
                surface_candidates = candidates_by_surface[surf_idx]
                surface_candidates.sort(key=lambda x: x['val'], reverse=True)
                for cand in surface_candidates:
                    surface.subdivide_surface(cand, optimizer=self._optimizer)

    def eikonal_loss_weighted(
            self,
            visibility_filter: Optional[torch.Tensor] = None,
            depth_weights: Optional[torch.Tensor] = None,
            weight: float = 1.0,
            eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Visibility and depth-weighted eikonal loss.

        Focuses regularization on:
        1. Visible surface regions (more important for rendering)
        2. Closer surfaces (higher impact on visual quality)

        Args:
            visibility_filter: [N] boolean or float mask of visible points
            depth_weights: [N] per-point depth-based weights (closer = higher)
            weight: Overall loss weight
            eps: Small value for numerical stability

        Returns:
            Weighted eikonal loss scalar
        """
        if weight <= 0:
            return torch.tensor(0.0, device=self.device)

        # Compute raw normals
        du = self.position.du
        dv = self.position.dv
        raw_normals = torch.cross(du, dv, dim=-1).reshape(-1, 3)

        # Compute norms
        norms = torch.linalg.norm(raw_normals, dim=-1)
        eikonal_error = (norms - 1.0).pow(2)

        # Apply visibility weighting
        if visibility_filter is not None:
            vis_weights = visibility_filter.float().reshape(-1)
        else:
            vis_weights = torch.ones_like(eikonal_error)

        # Apply depth weighting
        if depth_weights is not None:
            depth_w = depth_weights.reshape(-1)
        else:
            depth_w = torch.ones_like(eikonal_error)

        # Combined weighting
        combined_weights = vis_weights * depth_w

        # Weighted mean
        weighted_sum = (eikonal_error * combined_weights).sum()
        weight_total = combined_weights.sum().clamp(min=eps)

        loss = weighted_sum / weight_total

        return weight * loss

    def eikonal_loss_anisotropic(
            self,
            weight: float = 1.0,
            target_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Anisotropic eikonal loss that also penalizes non-uniform parameterization.

        In addition to unit normal length, this encourages:
        - ||∂S/∂u|| ≈ ||∂S/∂v|| (isotropic parameterization)
        - Prevents elongated/stretched surface patches

        Args:
            weight: Loss weight
            target_ratio: Target ratio of ||du|| / ||dv|| (1.0 = isotropic)

        Returns:
            Combined eikonal + isotropy loss
        """
        if weight <= 0:
            return torch.tensor(0.0, device=self.device)

        du = self.position.du.reshape(-1, 3)
        dv = self.position.dv.reshape(-1, 3)

        # Tangent magnitudes
        du_norm = torch.linalg.norm(du, dim=-1)
        dv_norm = torch.linalg.norm(dv, dim=-1)

        # Normal from cross product
        raw_normal = torch.cross(du, dv, dim=-1)
        normal_norm = torch.linalg.norm(raw_normal, dim=-1)

        # Standard eikonal:  ||n|| should be close to ||du|| * ||dv|| for orthogonal tangents
        # For general case, we want ||n|| = ||du|| * ||dv|| * sin(θ) ≈ constant

        # Simplified:  penalize deviation from expected area element
        expected_area = du_norm * dv_norm
        area_ratio = normal_norm / (expected_area + 1e-8)

        # Loss 1: Normal magnitude consistency (eikonal-like)
        eikonal_term = (area_ratio - 1.0).pow(2).mean()

        # Loss 2: Isotropy - tangent lengths should be similar
        ratio = du_norm / (dv_norm + 1e-8)
        isotropy_term = (ratio - target_ratio).pow(2).mean()

        # Combined loss
        loss = eikonal_term + 0.1 * isotropy_term

        return weight * loss

    def eikonal_loss_curvature_aware(
            self,
            weight: float = 1.0,
            curvature_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Curvature-aware eikonal loss.

        Allows higher normal variation in high-curvature regions
        while enforcing strict unit normals in flat regions.

        Args:
            weight: Base loss weight
            curvature_scale: How much to relax constraint in curved regions

        Returns:
            Curvature-weighted eikonal loss
        """
        if weight <= 0:
            return torch.tensor(0.0, device=self.device)

        # Compute raw normals
        du = self.position.du
        dv = self.position.dv
        raw_normals = torch.cross(du, dv, dim=-1).reshape(-1, 3)

        # Compute norms and eikonal error
        norms = torch.linalg.norm(raw_normals, dim=-1)
        eikonal_error = (norms - 1.0).pow(2)

        # Estimate local curvature from second derivatives
        try:
            # Use pre-computed second derivatives if available
            duu = self.d2Suu.reshape(-1, 3)
            dvv = self.d2Svv.reshape(-1, 3)

            # Approximate mean curvature magnitude
            curvature_estimate = (
                                         torch.linalg.norm(duu, dim=-1) +
                                         torch.linalg.norm(dvv, dim=-1)
                                 ) / 2.0

            # Normalize curvature to [0, 1] range
            curvature_normalized = curvature_estimate / (curvature_estimate.max() + 1e-8)

            # Weight:  lower weight in high curvature regions
            curvature_weight = 1.0 / (1.0 + curvature_scale * curvature_normalized)

        except Exception:
            # Fallback:  uniform weighting
            curvature_weight = torch.ones_like(eikonal_error)

        # Weighted loss
        loss = (eikonal_error * curvature_weight).mean()

        return weight * loss
    def eikonal_losses(self, weight: float) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        if weight <= 0.0:
            return loss
        losses = []
        for surface in self.surfaces:
            # if surface.is_background:
            #     continue
            losses.append(surface.eikonal_loss())
        return torch.stack(losses, dim=0).mean() * weight
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

    def get_states(self):
        l = [s.state for s in self.surfaces]
        return l

    @torch.no_grad()
    def prepare_grid_for_vis(self,view_cam):
        vis_log = {
                "sh_grid": {},
                "sh_cpt": {},
                "norm_grid": {},
                "visibility_cp": {},
            }
        for i, surf in enumerate(self.surfaces):
            normals = surf.get_normal(view_cam)
            normals.reshape(surf.state.Us, surf.state.Vs, -1)
            vis_log[f"weights_map_per_view_{i}"] = surf.weights_map().reshape(surf.state.Us, surf.state.Vs, 1)
            if hasattr(surf, 'ray') and surf.ray is not None and hasattr(surf.ray, 'depths'):
                u, v = surf.depth_discontinuities()    #.reshape(surf.state.Us, surf.state.Vs, -1)
                vis_log[f"depth_map_per_view_{i}"] = u.reshape(surf.state.Us, surf.state.Vs, -1)
            vis_log[f"sh_grid_{i}"] = surf.get_features.reshape(surf.state.Us, surf.state.Vs, -1)[..., :3]
            vis_log[f"sh_cpt_{i}"] = surf.spherical_harmonics.sh_dc.control_features.reshape(surf.state.H, surf.state.W, -1)[..., :3]
            vis_log[f"norm_grid_{i}"] = normals.reshape(surf.state.Us, surf.state.Vs, -1)
        return vis_log
    def get_states_by_idx(self, idx: int):
        return self.surfaces[idx].state

    @classmethod
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
    def set_knot_v(self, knot_v: torch.Tensor):
        """Set knot vector in v direction for a specific surface."""
        for surf in self.surfaces:
            surf.knot_v = KnotVector(surf.state, 'v', knot_v)
    def set_knot_u(self, knot_u: torch.Tensor):
        """Set knot vector in u direction for a specific surface."""
        for surf in self.surfaces:
            surf.knot_u = KnotVector(surf.state, 'u', knot_u)

    @classmethod
    def load(cls, path: str, train_model: bool = False) -> 'MultiSurfaceSplineModel':
        """Load model from disk."""
        state, i = torch.load(path)
        surfaces = []
        for surf_state in state['surfaces']:
            surface = SplineModel(late_init=True)
            surface.restore(surf_state, train_model=train_model)
            surfaces.append(surface)

        model = cls(
            labels=state['labels'],
            decomposition_mode=DecompositionMode(state['decomposition_mode'])
        )
        model._surfaces = surfaces

        model._active_surfaces = state['active_surfaces']
        model._surface_weights = state['surface_weights']

        if state['point_labels'] is not None:
            model.point_labels = torch.tensor(state['point_labels'], dtype=torch.long)
        model.training_setup()
        return model

    @property
    def total_gaussians(self) -> int:
        """Total number of Gaussians across all active surfaces."""
        total = 0
        for surface, active in zip(self.surfaces, self._active_surfaces):
            if active:
                total += surface.state.Us * surface.state.Vs #.numel()
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



    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale / 2) - 1, 0))
        depth_view = depth.squeeze()[st::scale, st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts - T) @ R.transpose(-1, -2)
        return pts

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale / 2) - 1, 0)
        depth_view = depth[None, :, st::scale, st::scale]
        W, H = int(fov_camera.image_width / scale), int(fov_camera.image_height / scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
            [points_in_camera_space[:, 0] * fov_camera.Fx / points_in_camera_space[:, 2] + fov_camera.Cx,
             points_in_camera_space[:, 1] * fov_camera.Fy / points_in_camera_space[:, 2] + fov_camera.Cy],
            -1).float() / scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) & \
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:, 2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask

    def reset_opacity(self):
        for surface in self.surfaces:
            surface.reset_opacity(optimizer=self.optimizer)

    def reset_scaling(self):
        for surface in self.surfaces:
            surface.reset_scaling(optimizer=self.optimizer)

    def prune_surface(
            self,
            pruning_candidates: List[List[dict]] = None,
            min_opacity: float = 0.005,
            max_screen_size: float = 20.0,
            extent: Optional[float] = None,
            top_k_rate: float = 0.0,
            min_k: int = 0,
            max_k: int = 16,
            verbose: bool = True,
    ) -> int:
        """
        Prune all surfaces based on 3DGS-like criteria.
        """
        total_removed = 0

        for i, surface in enumerate(self.surfaces):
            if verbose:
                print(f"\n[MultiSurface Pruning] Surface {i} ({self.labels[i]})")

            min_k = max(min(min_k, int(0.5 * min(surface.state.H, surface.state.W))), 8)
            max_k = max(min(max_k, int(0.5 * min(surface.state.H, surface.state.W)), max_k), min_k)
            top_k = max(min(int(top_k_rate * min(self.state(0).H, self.state(0).W)), max_k), min_k)

            cands = pruning_candidates[i] if pruning_candidates is not None else None

            if cands:
                top_k_candidates = cands[:top_k]
                removed = surface.prune_surface(
                    cands=top_k_candidates,
                    optimizer=self._optimizer,
                    error_tolerance=1e-4
                )

                if isinstance(removed, bool):
                    removed = len(top_k_candidates) if removed else 0

                total_removed += removed

                if verbose and removed > 0:
                    print(f"  [Pruning] Removed {removed} knots successfully")

        return total_removed
    def prune_all_surfaces2(
            self,
            pruning_candidates: List[List[dict]] = None,
            min_opacity: float = 0.005,
            max_screen_size: float = 20.0,
            extent: Optional[float] = None,
            top_k_rate: float = 0.0,
            min_k: int = 8,
            max_k: int = 16,
            verbose: bool = True,
    ) -> int:
        """
        Prune all surfaces based on 3DGS-like criteria.

        Args:
            min_opacity:  Opacity threshold for pruning
            max_screen_size: Maximum screen-space radius
            extent: Scene extent (auto-computed if None)
            max_removals_per_surface: Max removals per surface
            verbose:  Print progress

        Returns:
            Total number of successful removals
        """
        total_removed = 0

        for i, surface in enumerate(self.surfaces):
            if verbose:
                print(f"\n[MultiSurface Pruning] Surface {i} ({self.labels[i]})")

            min_k = max(min(min_k, int(0.5 * min(surface.state.H, surface.state.W))), 8)
            max_k = max(min(max_k, int(0.5 * min(surface.state.H, surface.state.W)), max_k), min_k)
            top_k = max(min(int(top_k_rate * min(self.state(0).H, self.state(0).W)), max_k), min_k)

            cands = pruning_candidates[i] if pruning_candidates is not None else None

            if cands:
                # Apply batched prune surface directly bypassing prune_grid's sequential loop
                top_k_candidates = cands[:top_k]
                removed = surface.prune_surface(
                    cands=top_k_candidates,
                    optimizer=self._optimizer,
                    error_tolerance=1e-4
                )

                # Compatibility if prune_surface returned bool instead of int
                if isinstance(removed, bool):
                    removed = len(top_k_candidates) if removed else 0

                total_removed += removed

                if verbose and removed > 0:
                    print(f"  [Pruning] Removed {removed} knots successfully")

        return total_removed

    def prune_all_surfaces2(
            self,
            pruning_candidates: List[List[dict]] = None,
            min_opacity: float = 0.005,
            max_screen_size: float = 20.0,
            extent: Optional[float] = None,
            top_k_rate: float = 0.0,
            min_k: int = 0,
            max_k: int = 16,
            verbose: bool = True,
    ) -> int:
        """
        Prune all surfaces based on 3DGS-like criteria.

        Args:
            min_opacity:  Opacity threshold for pruning
            max_screen_size: Maximum screen-space radius
            extent: Scene extent (auto-computed if None)
            max_removals_per_surface: Max removals per surface
            verbose:  Print progress

        Returns:
            Total number of successful removals
        """
        total_removed = 0

        for i, surface in enumerate(self.surfaces):
            if verbose:
                print(f"\n[MultiSurface Pruning] Surface {i} ({self.labels[i]})")
            min_k = max(min(min_k, int(0.5 * min(surface.state.H, surface.state.W))), min_k)
            max_k = max(min(max_k, int(0.5 * min(surface.state.H, surface.state.W)), max_k), min_k)
            top_k = max(min(int(top_k_rate * min(self.state(0).H, self.state(0).W)), max_k), min_k)

            # top_k = max(min(int(top_k_rate * surface.state.opt.num_partitions_prune), max_k), min_k)

            removed = surface.prune_grid(
                candidates = pruning_candidates[i] if pruning_candidates is not None else None,
                min_opacity=min_opacity,
                max_screen_size=max_screen_size,
                extent=extent,
                max_removals=top_k,
                optimizer=self._optimizer,
                verbose=verbose
            )
            total_removed += removed

        return total_removed

    def multi_view_trim_all(
            self,
            cameras: List,
            render_fn,
            pipe,
            background,
            app_model=None,
            min_observations: int = 1,
            row_threshold: float = 0.8,
            col_threshold: float = 0.8,
            top_k_rate: float = 0.0,
            max_k: int = 2,
            verbose: bool = True
    ) -> int:
        """
        Multi-view observation-based pruning for all surfaces.

        Args:
            cameras: List of training cameras
            render_fn:  Rendering function
            pipe: Pipeline parameters
            background: Background tensor
            app_model: Optional appearance model
            min_observations: Minimum view count to be considered "observed"
            row_threshold: Fraction of row that must be under-observed for removal
            col_threshold:  Fraction of column that must be under-observed for removal
            max_removals_per_surface: Maximum removals per surface
            verbose: Print progress

        Returns:
            Total number of successful removals across all surfaces
        """

        total_removed = 0
        if top_k_rate == 0.0:
            return total_removed
        if verbose:
            print(f"\n[Multi-View Trim] Processing {self.num_surfaces} surfaces...")

        # --- FIX START: Force consistent state before counting ---
        # 1. Invalidate caches to ensure we start fresh
        self._invalidate_cache(force=True)

        # 2. Ensure surface offsets are up to date with current H/W state
        self._update_surface_offsets()

        Us_Vs_total = sum(
            surface.state.Us * surface.state.Vs for surface, a in zip(self.surfaces, self._active_surfaces) if a)
        observe_cnt_combined = torch.zeros(Us_Vs_total, 1, device=self.device)

        for cam in cameras:
            # Forward pass
            self.forward(cam)

            # Render combined
            render_pkg = render_fn(cam, self, pipe, background, app_model=app_model,
                                   return_plane=False, return_depth_normal=False)

            # Get observation mask
            if "out_observe" in render_pkg:
                out_observe = render_pkg["out_observe"].view(-1)
                observe_cnt_combined[out_observe > 0] += 1
            elif "visibility_filter" in render_pkg:
                vis = render_pkg["visibility_filter"].view(-1)
                observe_cnt_combined[vis] += 1

        # Step 2: Process each surface
        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            top_k = min(int(top_k_rate * min(surface.state.H, surface.state.W)), max_k)

            if not active:
                continue

            if verbose:
                print(f"\n[Multi-View Trim] Surface {i} ('{self.labels[i]}')")

            # Get observation counts for this surface
            start = self._surface_offsets[i]
            end = self._surface_offsets[i + 1]
            observe_cnt_surface = observe_cnt_combined[start:end]

            # Get candidates
            candidates = surface.get_multi_view_trim_candidates(
                observe_cnt_surface,
                min_observations=min_observations,
                row_threshold=row_threshold,
                col_threshold=col_threshold
            )

            total_candidates = len(candidates['u']) + len(candidates['v'])

            if verbose:
                print(f"  Found {len(candidates['u'])} U-candidates, {len(candidates['v'])} V-candidates")

            if total_candidates == 0:
                continue

            # Sort by severity
            all_candidates = []
            for cand in candidates['u']:
                all_candidates.append({
                    'type': 'u',
                    'index': cand['index'],
                    'score': cand['under_observed_fraction'],
                    'reasons': [f"under_observed({cand['under_observed_fraction']:.2%})"]
                })
            for cand in candidates['v']:
                all_candidates.append({
                    'type': 'v',
                    'index': cand['index'],
                    'score': cand['under_observed_fraction'],
                    'reasons': [f"under_observed({cand['under_observed_fraction']:.2%})"]
                })

            all_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Batched index processing via the updated prune_surface negates the need for loop-based tracking
            top_k_candidates = all_candidates[:top_k]

            if top_k_candidates:
                removed_count = surface.prune_surface(
                    cands=top_k_candidates,
                    optimizer=self._optimizer,
                    error_tolerance=float('inf')
                )

                if removed_count:
                    total_removed += removed_count
                    if verbose:
                        print(f"    Removed {removed_count} under-observed candidates")

        if verbose:
            print(f"\n[Multi-View Trim] Total removed: {total_removed}")

        self._update_surface_offsets()
        self._invalidate_cache(force=True)

        for surface in self.surfaces:
            surface.state.init_grad_accumulators()

        return total_removed > 0
    def multi_view_trim_all2(
            self,
            cameras: List,
            render_fn,
            pipe,
            background,
            app_model=None,
            min_observations: int = 1,
            row_threshold: float = 0.8,
            col_threshold: float = 0.8,
            top_k_rate: float = 0.0,
            max_k: int = 2,
            verbose: bool = True
    ) -> int:
        """
        Multi-view observation-based pruning for all surfaces.

        Args:
            cameras: List of training cameras
            render_fn:  Rendering function
            pipe: Pipeline parameters
            background: Background tensor
            app_model: Optional appearance model
            min_observations: Minimum view count to be considered "observed"
            row_threshold: Fraction of row that must be under-observed for removal
            col_threshold:  Fraction of column that must be under-observed for removal
            max_removals_per_surface: Maximum removals per surface
            verbose: Print progress

        Returns:
            Total number of successful removals across all surfaces
        """

        total_removed = 0
        if top_k_rate == 0.0:
            return total_removed
        if verbose:
            print(f"\n[Multi-View Trim] Processing {self.num_surfaces} surfaces...")

        # --- FIX START: Force consistent state before counting ---
        # 1. Invalidate caches to ensure we start fresh
        self._invalidate_cache(force=True)

        # 2. Ensure surface offsets are up to date with current H/W state
        self._update_surface_offsets()


        Us_Vs_total = sum(surface.state.Us * surface.state.Vs for surface, a in zip(self.surfaces, self._active_surfaces) if a)
        observe_cnt_combined = torch.zeros(Us_Vs_total, 1, device=self.device)

        for cam in cameras:
            # Forward pass
            self.forward(cam)

            # Render combined
            render_pkg = render_fn(cam, self, pipe, background, app_model=app_model,
                                   return_plane=False, return_depth_normal=False)

            # Get observation mask
            if "out_observe" in render_pkg:
                out_observe = render_pkg["out_observe"].view(-1)
                observe_cnt_combined[out_observe > 0] += 1
            elif "visibility_filter" in render_pkg:
                vis = render_pkg["visibility_filter"].view(-1)
                observe_cnt_combined[vis] += 1

        # Step 2: Process each surface
        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            top_k = min(int(top_k_rate * min(surface.state.H, surface.state.W)), max_k)

            if not active:
                continue

            if verbose:
                print(f"\n[Multi-View Trim] Surface {i} ('{self.labels[i]}')")

            # Get observation counts for this surface
            start = self._surface_offsets[i]
            end = self._surface_offsets[i + 1]
            observe_cnt_surface = observe_cnt_combined[start:end]

            # Get candidates
            candidates = surface.get_multi_view_trim_candidates(
                observe_cnt_surface,
                min_observations=min_observations,
                row_threshold=row_threshold,
                col_threshold=col_threshold
            )

            total_candidates = len(candidates['u']) + len(candidates['v'])

            if verbose:
                print(f"  Found {len(candidates['u'])} U-candidates, {len(candidates['v'])} V-candidates")

            if total_candidates == 0:
                continue

            # Sort by severity
            all_candidates = []
            for cand in candidates['u']:
                all_candidates.append({
                    'type': 'u',
                    'index': cand['index'],
                    'score': cand['under_observed_fraction'],
                    'reasons': [f"under_observed({cand['under_observed_fraction']:.2%})"]
                })
            for cand in candidates['v']:
                all_candidates.append({
                    'type': 'v',
                    'index': cand['index'],
                    'score': cand['under_observed_fraction'],
                    'reasons': [f"under_observed({cand['under_observed_fraction']:.2%})"]
                })

            all_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Apply removals
            surface_removed = 0
            removed_u_indices = []
            removed_v_indices = []

            for cand in all_candidates[: top_k]:
                if surface_removed >= top_k:
                    break

                # Adjust index
                adjusted_idx = cand['index']
                if cand['type'] == 'u':
                    for removed in removed_u_indices:
                        if removed < cand['index']:
                            adjusted_idx -= 1
                else:
                    for removed in removed_v_indices:
                        if removed < cand['index']:
                            adjusted_idx -= 1

                prune_cand = {
                    'type': cand['type'],
                    'index': adjusted_idx,
                    'val': 0.0,
                    'reasons': cand['reasons'],
                    'score': cand['score']
                }

                success = surface.prune_surface(
                    prune_cand,
                    optimizer=self._optimizer,
                    error_tolerance=float('inf')
                )

                if success:
                    surface_removed += 1
                    total_removed += 1
                    if cand['type'] == 'u':
                        removed_u_indices.append(cand['index'])
                    else:
                        removed_v_indices.append(cand['index'])

                    if verbose:
                        print(f"    Removed {cand['type'].upper()} at idx={cand['index']} "
                              f"({cand['score']:.1%} under-observed)")

        if verbose:
            print(f"\n[Multi-View Trim] Total removed: {total_removed}")
        self._update_surface_offsets()
        self._invalidate_cache(force=True)
        for surface in self.surfaces:
            surface.state.init_grad_accumulators()
        return total_removed > 0

    def subdivide_and_cull(
            self,
            max_grad: float,
            grad_abs_threshold: float,
            min_opacity: float,
            extent: float,
            max_screen_size: Optional[float] = None,
            top_k_rate_subd: float = 0.1,
            max_prune_rate: float = 0.1,
            verbose: bool = True,
            # should_prune: bool=False,
            ) -> bool:
        """
        Combined densification and pruning for all surfaces.

        Performs global candidate selection across all surfaces,
        then applies top_k operations globally.
        """

        pruning_candidates = []
        for surf in self.surfaces:
            if max_prune_rate > 0:
                pruning_candidates.append(surf.get_pruning_candidates(
                    min_opacity=min_opacity,
                    max_screen_size=max_screen_size,
                    extent=extent,
                    use_partitioning=surf.state.opt.use_spatial_partitioning,
                    num_partitions=surf.state.opt.num_partitions_prune
                ))

        curr_num_gaussians = self.total_gaussians
        if top_k_rate_subd > 0:
            self.subdivide_surface(
                grad_threshold=max_grad,
                grad_abs_threshold=grad_abs_threshold,
                radii_threshold=max_screen_size,
                top_k_rate=top_k_rate_subd,
                verbose=verbose)

        final_num_gaussians = self.total_gaussians
        is_split = curr_num_gaussians != final_num_gaussians

        curr_num_gaussians = self.total_gaussians

        # Then prune each surface
        if max_prune_rate > 0:
            self.prune_surface(
                pruning_candidates=pruning_candidates,
                min_opacity=min_opacity,
                max_screen_size=max_screen_size,
                extent=extent,
                top_k_rate=max_prune_rate,
                verbose=verbose
            )


        final_num_gaussians = self.total_gaussians
        is_pruned = curr_num_gaussians != final_num_gaussians
        self._update_surface_offsets()
        for surf in self.surfaces:
            surf.state.init_grad_accumulators()
        self._invalidate_cache(force=True)
        return is_pruned or is_split

    def refine_sampling_positions(self, cameras=None):
        for i, surface in enumerate(self.surfaces):
            with torch.enable_grad():
                surface.optimize_intervals(
                    cameras=cameras,
                    num_steps=1000,
                    lr=0.005,
                    verbose=False
                )

    def enable_gradient_accumulation(self, num_views: int):
        """
        Enable gradient accumulation over multiple views.

        Args:
            num_views: Number of views to accumulate before optimizer step
        """
        self._accumulation_steps = num_views
        self._current_step = 0
        print(f"[Batch Optimization] Accumulating gradients over {num_views} views")

    def disable_gradient_accumulation(self):
        """Disable gradient accumulation (default behavior)."""
        self._accumulation_steps = 1
        self._current_step = 0

    def should_step(self) -> bool:
        """Check if we should perform optimizer step."""
        self._current_step += 1
        should_step = (self._current_step % self._accumulation_steps) == 0

        if should_step:
            self._current_step = 0  # Reset counter

        return should_step

    def get_grad_scale(self) -> float:
        """Get gradient scaling factor for averaging."""
        return 1.0 / self._accumulation_steps

    def step(self):
        """Optimizer step with gradient scaling."""
        if self._optimizer is not None:
            # Scale gradients by accumulation steps (for averaging)
            scale = self.get_grad_scale()
            if scale != 1.0:
                for param_group in self._optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param.grad.mul_(scale)

            self._optimizer.step()
        self._invalidate_cache(force=True)

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients (only when starting new batch)."""
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=set_to_none)

    def forward(self, viewpoint_cam, retain_graph: bool = False, **kwargs):
        self._invalidate_cache()  # Invalidate cache for fresh forward pass

        for i, (surface, active) in enumerate(zip(self.surfaces, self._active_surfaces)):
            if active:
                surface.uv_sampler.active_uid = str(viewpoint_cam.uid)
                surface.forward(viewpoint_cam, **kwargs)


    @property
    def should_retain_graph(self) -> bool:
        """Whether to retain computation graph for gradient accumulation."""
        return getattr(self, '_retain_graph', False)