from enum import Enum
from typing import Tuple, Optional

import torch

from arguments.nurbs_params import NurbsOptimizationParams
MAX_DENSITY = 8.0
import opt_einsum as oe


class SamplingMode(Enum):
    """Defines how UV sampling is handled during training."""
    STATIC = "static"
    OPTIMIZABLE = "optimizable"  # Intervals are nn.Parameters
    ADAPTIVE = "adaptive"  # Periodically recomputed, not optimized
    SNAPSHOT = "snapshot"  # Frozen from adaptive, then optimized
    EVALUATION = "evaluation"  # Frozen from adaptive, then optimized



class ModelState:
    """
    Centralized state manager for NURBS model components.
    Handles shared parameters, dimensions, and configuration across all modules.
    """

    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    def __init__(
            self,
            opt: NurbsOptimizationParams,

            H: int,
            W: int,
            device: str = 'cuda',
            degree: int = 3,
            stride: int = 1,
            active_sh_degree: int = 0,
            max_sh_degree: int = 3,
            surf_uid=0,
            sampling_density: float = 1.0,
            args=None,
            **kwargs
    ):
        # Core dimensions
        self.surf_uid = surf_uid
        self.is_background = kwargs.get('is_background', False)
        self.label = kwargs.get('label', f'surface_{surf_uid}')
        sampling_density = opt.sampling_density# if sampling_density is None else sampling_density
        self.sampling_density = int(sampling_density) if opt.random_sampling else sampling_density
        self.max_sampling_density = 1. if self.is_background else MAX_DENSITY
        self._H = H
        self._W = W

        self.optimizer_blend_strategy = 'zero'  # 'zero', 'neighbor_avg', 'learned'
        self.base_u = H
        self.base_v = W
        # Device and general config
        self.device = device
        self._uv_lr_factor = opt.uv_lr_factor
        self.deriv_order = 2

        # Patch configuration
        self.degree = degree
        self.p, self.q = degree, degree
        self.stride = stride

        self.opt=opt
        self.args = args
        # Feature configuration
        self.active_sh_degree = active_sh_degree #max_sh_degree
        self.max_sh_degree = max_sh_degree
        self.max_radii2D_cp = torch.zeros((H * W, 1), device=self.device)
        self.visibility_filter_cp = torch.zeros((H * W, 1), device=self.device)
        self.radii = torch.zeros(self.Us * self.Vs, 1, device=self.device)
        self.visibility_filter = torch.zeros(self.Us * self.Vs, 1, device=self.device)
        self.update_basis = False
        self.flat_view = False
        self.scaling_dims = kwargs.get('scaling_dims', 3)
        self._contract_path = None
        self._use_full_grid = False

        # Cache for computed values
        self._cache = {}
        # Dual gradient accumulators (like 3DGS)
        self._cpt_grad_accum = None  # Direction-aware (for clone-like ops)
        self._cpt_grad_accum_abs = None  # Magnitude-only (for split-like ops)
        self._cpt_denom = None
        self._cpt_denom_abs = None

        # For sampling grid
        self._xyz_grad_accum = None
        self._xyz_grad_accum_abs = None
        self._xyz_denom = None
        self._xyz_denom_abs = None
        if opt.sampling_strategy == 'adaptive':
            self._sampling_mode = SamplingMode.ADAPTIVE
        elif opt.sampling_strategy == 'optimize':
            self._sampling_mode = SamplingMode.OPTIMIZABLE
        else:
            self._sampling_mode = SamplingMode.STATIC
        # Warmup: start STATIC, switch to ADAPTIVE after warmup_iterations
        self._warmup_iterations = getattr(
            self.opt, 'adaptive_warmup_iterations', 5000
        )
        self._warmup_complete = False
        self._ada_sampling_interval = 500
        self.iteration = 0
        self.init_grad_accumulators()
    def update_sampling_density(self, new_density: float) -> None:
        """Update sampling density and reset related accumulators."""
        self.sampling_density = min(new_density, self.max_sampling_density)
        self.init_grad_accumulators()


    def init_grad_accumulators(self):
        """Initialize gradient accumulators (call after grid size is known)."""
        device = self.device
        self.clear_cached()

        # Control point accumulators
        cp_size = self.H * self.W
        self._cpt_grad_accum = torch.zeros(cp_size, 1, device=device)
        self._cpt_grad_accum_abs = torch.zeros(cp_size, 1, device=device)
        self._cpt_denom = torch.zeros(cp_size, 1, device=device)
        self._cpt_denom_abs = torch.zeros(cp_size, 1, device=device)

        # Sampling grid accumulators
        uv_size = self.Us * self.Vs
        self._xyz_grad_accum = torch.zeros(uv_size, 1, device=device)
        self._xyz_grad_accum_abs = torch.zeros(uv_size, 1, device=device)
        self.visibility_filter = torch.zeros(uv_size, 1, device=device)
        self._xyz_denom = torch.zeros(uv_size, 1, device=device)
        self._xyz_denom_abs = torch.zeros(uv_size, 1, device=device)

        # Max radii tracking (for screen-space size based decisions)
        self.max_radii2D = torch.zeros(uv_size, device=device)
        self.max_radii2D_cp = torch.zeros(cp_size, device=device)
    def set_Us(self, new_u) -> None:
        self.base_u = new_u
    def set_Vs(self, new_v) -> None:
        self.base_v = new_v

    @property
    def sampling_mode(self):
        return self._sampling_mode
    def update_samples(self, num_insertion, direction: str = 'both') -> None:
        """Update sampling dimensions based on current density factors."""
        if direction in ('u', 'both'):
            self.base_u += num_insertion

        if direction in ('v', 'both'):
            self.base_v += num_insertion
        return int(self.H * self.sampling_density) if self.bind_to_grid else int(self.base_u * self.sampling_density)
        # self._cache.clear()
    @torch.no_grad()
    def add_subdivision_stats(
            self,
            grad_norm: torch.Tensor,  # Standard gradient
            grad_norm_abs: torch.Tensor,  # Absolute gradient
            update_filter: torch.Tensor,  # Visibility mask
            radii: torch.Tensor,
            surf_vis: torch.Tensor):
        """
        Accumulate gradient statistics for densification decisions.

        This mirrors GaussianModel. add_densification_stats but for UV-grid points.

        Args:
            viewspace_point_tensor: Gradient tensor (signed, for direction-aware metric)
            viewspace_point_tensor_abs: Gradient tensor (for magnitude-aware metric)
            update_filter:  Boolean mask of visible points
            radii: Screen-space radii for size-based decisions
        """
        # if viewspace_point_tensor.grad is None:
        #     return
        # Initialize mapper if needed
        # if not hasattr(self, '_mapper') or self._mapper is None:
        #     self.init_mapper(knot_u, knot_v)


        # Ensure accumulators exist
        if self._xyz_grad_accum is None:
            print("Initializing gradient accumulators...")
            self.init_grad_accumulators()

        self._xyz_grad_accum[update_filter] += grad_norm[update_filter]
        self._xyz_denom[update_filter] += 1

        self._xyz_grad_accum_abs[update_filter] += grad_norm_abs[update_filter]
        self._xyz_denom_abs[update_filter] += 1
        self.visibility_filter[update_filter] += 1.0
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter],
            radii[update_filter]
        )



    def get_densification_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get averaged gradient metrics for densification.

        Returns:
            grads: Direction-aware gradient metric (for knot insertion)
            grads_abs: Magnitude-aware metric (for additional splits)
        """
        grads = self._xyz_grad_accum / (self._xyz_denom + 1e-8)
        grads_abs = self._xyz_grad_accum_abs / (self._xyz_denom_abs + 1e-8)

        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0

        return grads, grads_abs

    def reset_densification_stats(self):
        """Reset accumulators after densification."""
        if self._xyz_grad_accum is not None:
            self._xyz_grad_accum.zero_()
            self._xyz_grad_accum_abs.zero_()
            self._xyz_denom.zero_()
            self._xyz_denom_abs.zero_()
            self.max_radii2D.zero_()

    # @property
    # def contract_path(self) -> str:
    #
    #     if self.ndim_interp == 2:
    #         oe_path = 'hu,uvc,vw->hwc'
    #     else:
    #         oe_path = 'hwu, hwv, uvc -> hwc'
    #
    #     return oe_path
    @property
    def optimal_contract_path(self) -> str:
        return oe.contract_path(self.contract_path,
                                                 *[self.basis_shape_u, self.basis_shape_v,
                                                   self.control_layout])[0]
    @property
    def grid_layout(self):
        return self.H * self.W, -1  # hwu operands

    @property
    def uv_lr_factor(self):
        return self._uv_lr_factor

    @property
    def num_control_points_u(self) -> int:
        return self._H

    @property
    def num_control_points_v(self) -> int:
        return self._W
    @property
    def hw2uv(self) -> str:
        """Einstein summation path for height-width to UV coordinate transformation."""
        return 'uhh, wwv -> uv'# self._hw2uv

    @property
    def uv2xyz(self) -> str:
        """Einstein summation path for UV to XYZ coordinate transformation."""
        return 'uv, hwc -> uvc'# self._uv2xyz
    @property
    def shc(self) -> int:
        return (self.max_sh_degree + 1) ** 2

    @property
    def subdiv_k_limit(self):
        return 64
    @property
    def prune_k_limit(self):
        return 64
    @property
    def H(self) -> int:
        """Effective height after factoring."""
        return self._H #// self.curr_factor
    @property
    def H_eff(self) -> int:
        """Effective height after factoring."""
        return self._H

    @property
    def W(self) -> int:
        """Effective width after factoring."""
        return self._W #

    @property
    def W_eff(self) -> int:
        """Effective width after factoring."""
        return self._W


    @property
    def bind_to_grid(self):
        return getattr(self.opt, 'bind_uv_to_grid', False)
    @property
    def Us(self) -> int:
        """Effective height after factoring."""
        return int(self.H * self.sampling_density) if self.bind_to_grid else int(self.base_u * self.sampling_density)


    @property
    def Vs(self) -> int:
        """Effective height after factoring."""
        return int(self.W * self.sampling_density) if self.bind_to_grid else int(self.base_v * self.sampling_density)

    @property
    def use_bmm(self):
        return False
    @property
    def full_basis(self):
        return self._use_full_grid

    @property
    def flatten_uv(self):
        return False
    @property
    def Bu_layout(self):
        return (-1, self.Us) if self.flat_view else (self.Us, self.H_eff, self.H_eff)

    @property
    def Bv_layout(self):
        return (-1, self.Vs) if self.flat_view else (self.W_eff, self.W_eff, self.Vs)
    @property
    def curr_factor(self) -> int:
        """Current downsampling factor."""
        return self._curr_factor

    @property
    def num_patches_u(self) -> int:
        """Number of patches in U direction."""
        return ((self.H - self.degree - 1) // self.stride + 1)

    @property
    def num_patches_v(self) -> int:
        """Number of patches in V direction."""
        return ((self.W - self.degree - 1) // self.stride + 1)

    def update_factor(self, new_factor: int) -> None:
        """Update downsampling factor and clear caches."""
        self._curr_factor = max(new_factor, 1)
        self._cache.clear()

    @property
    def control_layout(self) -> tuple:
        return self.H, self.W, -1

    @property
    def interval_sampling_layout(self) -> tuple:
        return (self.Us, self.Vs) if self.full_basis else (-1, 1)
    @property
    def sampling_layout(self) -> tuple:
        return (self.Us, self.Vs, -1)# if self.uv_grid else (1, -1)
    def get_visibility_weight(self) -> torch.Tensor:
        return self.visibility_filter_cp / self.visibility_filter_cp.max()

    def update_cpts_grad(self, new_grid_grads, new_vis, radii_grid):
        self.cpt_grad = new_grid_grads.reshape(self.H * self.W, 1)
        self.visibility_filter_cp = new_vis.reshape(self.H * self.W, 1)
        self.max_radii2D_cp = radii_grid.reshape(self.H * self.W, 1)
    def update_subdivision_stats(self, new_grids, reset=False):
        '''
        replicate the accumulated grids from the saved ones:
        torch.cat([self._xyz_grad_accum, self._xyz_grad_accum_abs, self._xyz_denom, self._xyz_denom_abs, self.visibility_filter, self.max_radii2D], dim=-1)
        :param new_grids:
        :return:
        '''
        new_grid_grads = new_grids[:,0]
        new_grid_grads_abs = new_grids[:,1]
        denom_grid = new_grids[:,2]
        denom_abs_grid = new_grids[:,3]
        new_vis = new_grids[:,4]
        max_radii_grid = new_grids[:,5]
        self._xyz_grad_accum = new_grid_grads.reshape(self.Us * self.Vs, 1)
        self._xyz_grad_accum_abs = new_grid_grads_abs.reshape(self.Us * self.Vs, 1)
        self._xyz_denom = denom_grid.reshape(self.Us * self.Vs, 1)
        self._xyz_denom_abs = denom_abs_grid.reshape(self.Us * self.Vs, 1)
        self.visibility_filter = new_vis.reshape(self.Us * self.Vs, 1)
        self.max_radii2D = torch.zeros_like(max_radii_grid.reshape(self.Us * self.Vs))
        if reset:
            self.reset_densification_stats()


    def update_xyz_grad(self, new_grid_grads, new_grid_vis):# decay_factor=1.0):
        self.xyz_grad = new_grid_grads.reshape(self.Us * self.Vs, 1)
        self.visibility_filter = new_grid_vis.reshape(self.Us * self.Vs, 1)

    def log_radii(self, radii) -> None:
        if not hasattr(self, 'max_radii'):
            self.max_radii = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.max_radii_cp = torch.zeros(self.H * self.W, 1, device=self.device)
        self.max_radii = torch.max(self.max_radii, radii.view(-1, 1))
        self.max_radii_cp = torch.max(self.max_radii, radii.view(-1, 1))

    def get_max_radii2D(self) -> torch.Tensor:
        return (self.max_radii / (self.visibility_filter + 1e-8)).reshape(self.control_layout)

    def get_max_radii2D_cp(self) -> torch.Tensor:
        return (self.max_radii / (self.visibility_filter + 1e-8)).reshape(self.control_layout)

    def get_all_accum(self) -> torch.Tensor:
        return torch.cat([self._xyz_grad_accum, self._xyz_grad_accum_abs, self._xyz_denom, self._xyz_denom_abs, self.visibility_filter, self.max_radii2D.unsqueeze(1)], dim=-1)
        # return torch.cat([self.cpt_grad, self.cpt_grad_abs, self.visibility_filter_cp, self.radii_cp, self.denom_cp, self.denom_abs], dim=1).reshape(self.H, self.W, -1)
        # return (self.cpt_grad * self.get_visibility_weight()).reshape(self.H, self.W)
    def get_all_accum_grids(self) -> torch.Tensor:
        return torch.cat([self._xyz_grad_accum, self._xyz_grad_accum_abs, self._xyz_denom, self._cpt_denom,])
        # return torch.cat([self.xyz_grad, self.xyz_grad_abs, self.visibility_filter, self.radii, self.denom, self.denom_abs], dim=1).reshape(self.Us, self.Vs, -1)
        # return (self.cpt_grad * self.get_visibility_weight()).reshape(self.H, self.W)

    def get_grid_grads(self):
        return (self._xyz_grad_accum / self._xyz_denom.clamp(min=1e-6)).reshape(self.Us, self.Vs)

    def get_grid_grads_abs(self):
        return (self._xyz_grad_accum_abs / self._xyz_denom_abs.clamp(min=1e-6)).reshape(self.Us, self.Vs)

    def get_vis_ctrlpts(self) -> torch.Tensor:
        return self.visibility_filter_cp.reshape(self.H, self.W)
    def get_radii_ctrlpts(self) -> torch.Tensor:
        return self.max_radii2D_cp.reshape(self.H, self.W)
    def get_radii(self) -> torch.Tensor:
        return (self.radii / self.denom.clamp(1e-6)).reshape(self.Us, self.Vs)
    def get_vis_sampling(self) -> torch.Tensor:
        return self.visibility_filter.reshape(self.Us, self.Vs, 1) # / (self.visibility_map + 1e-8)).reshape(self.Us, self.Vs, -1)
    def get_grads(self) -> torch.Tensor:
        return self.xyz_grad.reshape(self.Us, self.Vs) # / (self.visibility_map + 1e-8)).reshape(self.Us, self.Vs, -1)

    def cpt_accum_grad(self, cpt_grad:torch.Tensor, visibility_filter_cp:torch.Tensor, radii_cp=None) -> None:
        if not hasattr(self, 'cpt_grad') or self.cpt_grad.numel() == 0:
            self.cpt_grad = torch.zeros(self.H * self.W, 1, device=self.device)
            self.visibility_filter_cp = torch.zeros(self.H * self.W, 1, device=self.device)
            self.max_radii2D_cp = torch.zeros(self.H * self.W, 1, device=self.device)
            self.denom_cp = torch.zeros(self.H * self.W, 1, device=self.device)
            self.denom_abs = torch.zeros(self.H * self.W, 1, device=self.device)
        self.cpt_grad += cpt_grad.reshape(self.H * self.W, 1)
        self.visibility_filter_cp += visibility_filter_cp.to(torch.float32).reshape(self.H * self.W, 1)
        if radii_cp is not None:
            self.max_radii2D_cp += radii_cp.to(torch.float32).reshape(self.H * self.W, 1)

    def xyz_accum_grad(self, xyz_grad: torch.Tensor, xyz_grad_abs:torch.Tensor, visibility_filter: torch.Tensor, radii=None) -> None:
        """Accumulate gradient for XYZ positions."""
        if not hasattr(self, 'xyz_grad') or self.xyz_grad.numel() == 0:
            self.xyz_grad = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.visibility_filter = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.radii = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.denom = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.denom_abs = torch.zeros(self.Us * self.Vs, 1, device=self.device)
            self.xyz_grad_abs = torch.zeros(self.Us * self.Vs, 1, device=self.device)

        # Reshape as UV grid both grads and visibility filter
        self.xyz_grad += xyz_grad.reshape(self.Us * self.Vs, 1)
        self.xyz_grad_abs += xyz_grad_abs.reshape(self.Us * self.Vs, 1)
        self.visibility_filter += visibility_filter.to(torch.float32).reshape(self.Us * self.Vs, 1)
        self.denom[visibility_filter > 0] += 1.0
        self.denom_abs[visibility_filter > 0] += 1.0

        if radii is not None:
            self.radii += radii.to(torch.float32).reshape(self.Us * self.Vs, 1)

    def log_stats(self, **kwargs) -> None:
        xyz_grad = kwargs.get('xyz_grad', None)
        xyz_grad_abs = kwargs.get('xyz_grad_abs', None)
        radii = kwargs.get('radii', None)
        visibility_filter = kwargs.get('visibility_filter', None)
        if xyz_grad is not None and visibility_filter is not None:

            self.xyz_accum_grad(xyz_grad, xyz_grad_abs, visibility_filter, radii)
            return

        cpt_grad = kwargs.get('cpt_grad', None)
        radii_cp = kwargs.get('radii_cp', None)
        visibility_filter_cp = kwargs.get('visibility_filter_cp', None)
        if cpt_grad is not None and visibility_filter_cp is not None:
            self.cpt_accum_grad(cpt_grad, visibility_filter_cp, radii_cp)



    def update_dimensions(self, new_H: int, new_W: int) -> None:
        """Update base dimensions."""
        self._H = new_H
        self._W = new_W

    def invalidate_caches(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def init_mapper(self, knot_u: torch.Tensor, knot_v: torch.Tensor):
        """Initialize the sampling-to-control mapper."""
        from .mapping import SamplingToControlMapper

        self._mapper = SamplingToControlMapper(
            state=self,
            knot_u=knot_u,
            knot_v=knot_v,
            degree=self.degree,
            assignment_mode='bilinear'  # Good balance of accuracy and speed
        )

    @property
    def mapper(self) -> 'SamplingToControlMapper':
        """Get the sampling-to-control mapper."""
        if not hasattr(self, '_mapper') or self._mapper is None:
            raise RuntimeError("Mapper not initialized.  Call init_mapper() first.")
        return self._mapper

    def clear_cached(self) -> 'SamplingToControlMapper':
        """Get the sampling-to-control mapper."""
        if hasattr(self, '_mapper'):
            del self._mapper
    def update_mapper_knots(self, knot_u: torch.Tensor, knot_v: torch.Tensor):
        """Update mapper after knot insertion."""
        if hasattr(self, '_mapper') and self._mapper is not None:
            self._mapper.update_knots(knot_u, knot_v)

    def aggregate_sampling_stats(
            self,
            sample_grads: torch.Tensor,  # [Us*Vs, C] or [Us, Vs, C]
            sample_u: torch.Tensor,
            sample_v: torch.Tensor,
            visibility: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate gradient statistics from sampling space to control space.

        Returns:
            ctrl_grads: [H, W, C] aggregated gradients
            ctrl_visibility: [H, W] visibility fraction per control
        """
        result = self.mapper.aggregate_to_control(
            sample_grads,
            sample_u,
            sample_v,
            visibility=visibility,
            reduction='mean'
        )
        return result.values, result.visibility
    def __str__(self) -> str:
        return (f"ModelState("
                f"\nControl-Grid Size: H={self.H}, W={self.W}, "
                f"\nSampling-Size: H'={self.Us}, W={self.Vs}) ")


# def update_uvs_multiview(model, current_cam, all_cameras):
#     # 1. Identify neighbors
#     # Assuming current_cam.nearest_id is a list of indices or UIDs
#     neighbor_indices = current_cam.nearest_id[:2]  # Take top 2 closest
#     neighbors = [all_cameras[i] for i in neighbor_indices]
#
#     # 2. Update with consistency
#     model.update_uv_distribution_chhugani(
#         camera=current_cam,
#         neighbor_cameras=neighbors,
#         curvature_weight=0.8,
#         silhouette_weight=1.0,
#         neighbor_weight=0.7  # High weight to enforce consistency
#     )
