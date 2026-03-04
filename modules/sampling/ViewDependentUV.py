from typing import Tuple, Optional, NamedTuple
import torch
from torch import nn
import torch.nn.functional as F

from model.modules import SamplerUV, ModelState
from model.spline_utils import inverse_sigmoid
from utils.sh_utils import eval_sh

C0 = 0.28209479177387814


class UVSamplingResult(NamedTuple):
    """Container for UV sampling results."""
    uv: torch.Tensor  # [Us, Vs, 2] warped UV coordinates
    visible_mask: torch.Tensor  # [Us, Vs] visibility mask
    ray_info: 'RayInfo'  # Full ray information


class ViewDependentUV(SamplerUV):
    """
    View-dependent UV sampler with visibility-aware processing.
    """

    def __init__(
            self,
            state: ModelState,
            sh_degree: int = 3,
            warp_scale: float = 1.,
            visibility_fallback: str = 'geometric',  # 'geometric', 'nearest', 'zero'
            **kwargs
    ):
        kwargs['num_channels'] = (sh_degree + 1) ** 2
        super().__init__(state, mode='sh', **kwargs)

        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.sh_degree = sh_degree
        self._active_sh_degree = 3
        self.warp_scale = warp_scale
        self.visibility_fallback = visibility_fallback
        self.C0 = C0

        eps = 1e-3
        Us = int(state.opt.sampling_density * state.H)
        Vs = int(state.opt.sampling_density * state.W)
        self._Us = Us
        self._Vs = Vs
        self._num_samples = Us * Vs

        self._init_parameters(state.device, Us, Vs, eps, kwargs)

        # Caches
        self.cache = None
        self.cache_visible_mask = None
        self.cache_ray_info = None
        self._training_step = 0

    def _init_parameters(self, device, Us, Vs, eps, kwargs):
        """Initialize or load parameters."""

        if kwargs.get('loaded_us_dc', None) is not None:
            # Load from checkpoint
            self._load_parameters(kwargs)
        else:
            # Fresh initialization
            self._init_fresh_parameters(device, Us, Vs, eps)

        # Base grid for regularization
        base_u = torch.linspace(eps, 1 - eps, Us, device=device)
        base_v = torch.linspace(eps, 1 - eps, Vs, device=device)
        self.register_buffer('_base_u', base_u)
        self.register_buffer('_base_v', base_v)

    def _init_fresh_parameters(self, device, Us, Vs, eps):
        """Initialize parameters for fresh training."""

        # Create uniform grid
        base_u = torch.linspace(eps, 1 - eps, Us, device=device)
        base_v = torch.linspace(eps, 1 - eps, Vs, device=device)

        # Expand to full grid:  [Us*Vs, 1, 1]
        base_u_grid = base_u.view(Us, 1).expand(Us, Vs).reshape(Us * Vs, 1, 1)
        base_v_grid = base_v.view(1, Vs).expand(Us, Vs).reshape(Us * Vs, 1, 1)

        # DC coefficients:  store as logit(uv)
        dc_u = torch.logit(base_u_grid.clamp(eps, 1 - eps))
        dc_v = torch.logit(base_v_grid.clamp(eps, 1 - eps))

        # Higher-order SH:  initialized to zero
        num_higher_sh = (self.sh_degree + 1) ** 2 - 1
        rest_u = torch.zeros(Us * Vs, 1, num_higher_sh, device=device)
        rest_v = torch.zeros(Us * Vs, 1, num_higher_sh, device=device)

        self._interval_u_dc = nn.Parameter(dc_u.contiguous(), requires_grad=True)
        self._interval_v_dc = nn.Parameter(dc_v.contiguous(), requires_grad=True)
        self._interval_u_rest = nn.Parameter(rest_u.contiguous(), requires_grad=True)
        self._interval_v_rest = nn.Parameter(rest_v.contiguous(), requires_grad=True)

    def _load_parameters(self, kwargs):
        """Load parameters from checkpoint."""
        loaded_u_dc = kwargs['loaded_us_dc'].to(self.device)
        loaded_v_dc = kwargs['loaded_vs_dc'].to(self.device)
        loaded_u_rest = kwargs.get('loaded_us_rest',
                                   torch.zeros(self._num_samples, 1, (self.sh_degree + 1) ** 2 - 1)).to(self.device)
        loaded_v_rest = kwargs.get('loaded_vs_rest',
                                   torch.zeros(self._num_samples, 1, (self.sh_degree + 1) ** 2 - 1)).to(self.device)

        if self.evaluate_mode:
            self.register_buffer('_interval_u_dc', loaded_u_dc)
            self.register_buffer('_interval_v_dc', loaded_v_dc)
            self.register_buffer('_interval_u_rest', loaded_u_rest)
            self.register_buffer('_interval_v_rest', loaded_v_rest)
        else:
            self._interval_u_dc = nn.Parameter(loaded_u_dc, requires_grad=True)
            self._interval_v_dc = nn.Parameter(loaded_v_dc, requires_grad=True)
            self._interval_u_rest = nn.Parameter(loaded_u_rest, requires_grad=True)
            self._interval_v_rest = nn.Parameter(loaded_v_rest, requires_grad=True)

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree

    def oneUpSHdegree(self) -> None:
        if self._active_sh_degree < self.sh_degree:
            self._active_sh_degree += 1
            print(f"[ViewDependentUV] SH degree increased to {self._active_sh_degree}")

    @property
    def uv_sh_features(self) -> torch.Tensor:
        """Get SH features for eval_sh."""
        num_active = (self._active_sh_degree + 1) ** 2

        # DC:  sigmoid -> divide by C0 so eval_sh recovers UV
        dc_u = torch.sigmoid(self._interval_u_dc) / self.C0
        dc_v = torch.sigmoid(self._interval_v_dc) / self.C0

        if num_active == 1:
            sh_u, sh_v = dc_u, dc_v
        else:
            rest_active = num_active - 1
            rest_u = self._interval_u_rest[..., :rest_active] * self.warp_scale
            rest_v = self._interval_v_rest[..., : rest_active] * self.warp_scale
            sh_u = torch.cat([dc_u, rest_u], dim=-1)
            sh_v = torch.cat([dc_v, rest_v], dim=-1)

        return torch.cat([sh_u, sh_v], dim=1)  # [Us*Vs, 2, num_sh]

    def forward(
            self,
            ray_info: 'RayInfo',
            return_full: bool = False
    ) -> torch.Tensor:
        """
        Compute view-dependent UV sampling with visibility awareness.

        Args:
            ray_info: RayInfo from compute_ray_info()
            return_full: If True, return UVSamplingResult with all info

        Returns:
            uv: [Us, Vs, 2] or UVSamplingResult if return_full=True
        """
        dirs = ray_info.directions  # [N, 3]
        visible_mask = ray_info.visible_mask  # [N]

        # Validate input
        if dirs.shape[0] != self._num_samples:
            raise ValueError(
                f"Expected {self._num_samples} directions, got {dirs.shape[0]}"
            )

        # Normalize directions
        dirs = F.normalize(dirs, dim=-1)

        # Get SH features and evaluate
        sh_features = self.uv_sh_features  # [Us*Vs, 2, num_sh]
        uv_flat = eval_sh(self._active_sh_degree, sh_features, dirs)  # [Us*Vs, 2]

        # Reshape to grid
        uv = uv_flat.reshape(self._Us, self._Vs, 2)
        visible_grid = visible_mask.reshape(self._Us, self._Vs)

        # Handle non-visible samples based on fallback strategy
        if not visible_grid.all():
            uv = self._handle_non_visible(uv, visible_grid)

        # Clamp to valid range
        # uv = uv.clamp(0.0, 1.0)
        # Cache for loss computation
        # u, v = F.normalize(uv[..., 0], dim=0), F.normalize(uv[..., 1], dim=1)
        u, v = uv[..., 0], uv[..., 1]
        u = u.cumsum(dim=0)
        v = v.cumsum(dim=1)
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        uv = torch.stack([u, v], dim=-1)


        self.cache = uv
        self.cache_visible_mask = visible_grid
        self.cache_ray_info = ray_info

        if return_full:
            return UVSamplingResult(
                uv=uv,
                visible_mask=visible_grid,
                ray_info=ray_info
            )

        # u, v = uv[..., 0].sort(dim=0)[0], uv[..., 1].sort(dim=1)[0]
        return uv

    def _handle_non_visible(
            self,
            uv: torch.Tensor,
            visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Handle UV values for non-visible samples.

        Args:
            uv: [Us, Vs, 2] UV coordinates
            visible_mask: [Us, Vs] boolean mask

        Returns:
            uv: [Us, Vs, 2] with non-visible samples handled
        """
        if self.visibility_fallback == 'zero':
            # Set non-visible to center of UV space
            uv = torch.where(
                visible_mask.unsqueeze(-1),
                uv,
                torch.full_like(uv, 0.5)
            )

        elif self.visibility_fallback == 'nearest':
            # Interpolate from nearest visible samples
            uv = self._interpolate_from_visible(uv, visible_mask)

        elif self.visibility_fallback == 'geometric':
            # Keep the geometric fallback (already applied in ray computation)
            pass

        return uv

    def _interpolate_from_visible(
            self,
            uv: torch.Tensor,
            visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate non-visible UV values from nearby visible samples.
        Uses distance-weighted average of visible neighbors.
        """
        if visible_mask.all():
            return uv

        Us, Vs = self._Us, self._Vs
        device = uv.device

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(Us, device=device),
            torch.arange(Vs, device=device),
            indexing='ij'
        )

        # Get visible sample coordinates and values
        vis_y = y_coords[visible_mask]
        vis_x = x_coords[visible_mask]
        vis_uv = uv[visible_mask]  # [num_visible, 2]

        if len(vis_y) == 0:
            # No visible samples - return uniform grid
            return self._get_base_uv_grid()

        # For each non-visible sample, find nearest visible and interpolate
        non_vis_mask = ~visible_mask
        non_vis_y = y_coords[non_vis_mask]
        non_vis_x = x_coords[non_vis_mask]

        # Compute distances to all visible samples
        # [num_non_visible, num_visible]
        dist_y = (non_vis_y.unsqueeze(1) - vis_y.unsqueeze(0)).float()
        dist_x = (non_vis_x.unsqueeze(1) - vis_x.unsqueeze(0)).float()
        distances = torch.sqrt(dist_y ** 2 + dist_x ** 2 + 1e-6)

        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Weighted average of visible UV values
        interpolated = torch.einsum('nv,vd->nd', weights, vis_uv)

        # Fill in non-visible samples
        uv_filled = uv.clone()
        uv_filled[non_vis_mask] = interpolated

        return uv_filled

    def _get_base_uv_grid(self) -> torch.Tensor:
        """Get the base uniform UV grid."""
        base_u = self._base_u.view(-1, 1).expand(self._Us, self._Vs)
        base_v = self._base_v.view(1, -1).expand(self._Us, self._Vs)
        return torch.stack([base_u, base_v], dim=-1)

    # ==================== LOSS FUNCTIONS ====================

    def visibility_aware_coverage_loss(self, weight: float = 1e-2) -> torch.Tensor:
        """
        Coverage loss that only considers visible samples.
        """
        if self.cache is None or self.cache_visible_mask is None:
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        uv = self.cache
        mask = self.cache_visible_mask

        if not mask.any():
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        # Get visible UV values
        uv_visible = uv[mask]  # [num_visible, 2]

        # Variance of visible samples
        var_u = torch.var(uv_visible[:, 0])
        var_v = torch.var(uv_visible[:, 1])

        # Penalize low variance
        variance_loss = -torch.log(var_u + 1e-6) - torch.log(var_v + 1e-6)

        return weight * variance_loss

    def visibility_ratio_loss(self, target_ratio: float = 0.8, weight: float = 1e-2) -> torch.Tensor:
        """
        Encourage a minimum visibility ratio.
        Penalizes if too many samples are non-visible.
        """
        if self.cache_visible_mask is None:
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        current_ratio = self.cache_visible_mask.float().mean()

        # Penalize if below target
        loss = F.relu(target_ratio - current_ratio)

        return weight * loss

    def adaptive_sampling_loss(
            self,
            rendered: torch.Tensor,
            target: torch.Tensor,
            weight: float = 0.01
    ) -> torch.Tensor:
        """
        Encourage more UV samples in high-error image regions.
        Only considers visible samples.

        Args:
            rendered: [H, W, 3] or [3, H, W] rendered image
            target: [H, W, 3] or [3, H, W] target image
        """
        if self.cache is None or self.cache_ray_info is None:
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        # Ensure [H, W, 3] format
        if rendered.shape[0] == 3:
            rendered = rendered.permute(1, 2, 0)
            target = target.permute(1, 2, 0)

        H, W = rendered.shape[: 2]

        # Per-pixel error
        error_map = (rendered - target).pow(2).mean(dim=-1)  # [H, W]

        # Get pixel coordinates from ray_info
        pixel_coords = self.cache_ray_info.pixel_coords  # [N, 2]
        visible_mask = self.cache_ray_info.visible_mask  # [N]

        if not visible_mask.any():
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        # Normalize pixel coords for grid_sample
        px_norm = torch.zeros_like(pixel_coords)
        px_norm[:, 0] = 2 * pixel_coords[:, 0] / (W - 1) - 1
        px_norm[:, 1] = 2 * pixel_coords[:, 1] / (H - 1) - 1
        px_norm = px_norm.clamp(-1, 1)

        # Sample error at UV locations
        grid = px_norm.view(1, -1, 1, 2)  # [1, N, 1, 2]
        error_batch = error_map.view(1, 1, H, W)  # [1, 1, H, W]

        sampled_error = F.grid_sample(
            error_batch, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).view(-1)  # [N]

        # Only consider visible samples
        sampled_error_visible = sampled_error[visible_mask]

        # Compute UV spacing (density proxy)
        uv = self.cache
        du = torch.diff(uv[..., 0], dim=0).abs()  # [Us-1, Vs]
        dv = torch.diff(uv[..., 1], dim=1).abs()  # [Us, Vs-1]

        # Density is inverse of spacing
        density_u = 1.0 / (du + 1e-6)
        density_v = 1.0 / (dv + 1e-6)

        # Reshape error to grid and match density dimensions
        error_grid = sampled_error.reshape(self._Us, self._Vs)
        visible_grid = visible_mask.reshape(self._Us, self._Vs)

        # Match dimensions with density
        error_u = error_grid[:-1, :]
        error_v = error_grid[:, :-1]
        mask_u = visible_grid[:-1, :] & visible_grid[1:, :]
        mask_v = visible_grid[:, :-1] & visible_grid[:, 1:]

        # We want:  high error -> high density (positive correlation)
        # Loss: negative of correlation (minimize to maximize correlation)
        if mask_u.any():
            corr_u = (error_u[mask_u] * density_u[mask_u]).mean()
        else:
            corr_u = 0.0

        if mask_v.any():
            corr_v = (error_v[mask_v] * density_v[mask_v]).mean()
        else:
            corr_v = 0.0

        # Negative because we want to maximize correlation
        correlation_loss = -(corr_u + corr_v)

        return weight * correlation_loss

    def depth_consistency_loss(self, weight: float = 1e-3) -> torch.Tensor:
        """
        Encourage smooth UV variation with respect to depth.
        Nearby depths should have similar UV warping.
        """
        if self.cache is None or self.cache_ray_info is None:
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        depths = self.cache_ray_info.depths.reshape(self._Us, self._Vs)
        uv = self.cache
        visible = self.cache_visible_mask

        # Compute depth gradients
        depth_du = torch.diff(depths, dim=0)  # [Us-1, Vs]
        depth_dv = torch.diff(depths, dim=1)  # [Us, Vs-1]

        # Compute UV gradients
        uv_du = torch.diff(uv[..., 0], dim=0)
        uv_dv = torch.diff(uv[..., 1], dim=1)

        # Visibility mask for gradients
        mask_u = visible[:-1, :] & visible[1:, :]
        mask_v = visible[:, :-1] & visible[:, 1:]

        if not mask_u.any() or not mask_v.any():
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        # Penalize UV changes that don't correlate with depth changes
        # (smooth surfaces should have smooth UV mapping)
        depth_weight_u = 1.0 / (depth_du.abs() + 0.1)
        depth_weight_v = 1.0 / (depth_dv.abs() + 0.1)

        smoothness_u = (uv_du.pow(2) * depth_weight_u)[mask_u].mean()
        smoothness_v = (uv_dv.pow(2) * depth_weight_v)[mask_v].mean()

        return weight * (smoothness_u + smoothness_v)

    def center_focus_loss(self, sigma: float = 0.3, weight: float = 1e-2) -> torch.Tensor:
        """
        Encourage higher UV density near image center.
        Uses a Gaussian weighting centered on the image.
        """
        if self.cache is None or self.cache_ray_info is None:
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        pixel_coords = self.cache_ray_info.pixel_coords  # [N, 2]
        visible = self.cache_ray_info.visible_mask

        if not visible.any():
            return torch.tensor(0.0, device=self._interval_u_dc.device)

        # Normalize pixel coords to [0, 1]
        H, W = self.state.H, self.state.W  # Assuming these exist
        px_norm = pixel_coords.clone()
        px_norm[:, 0] = pixel_coords[:, 0] / W
        px_norm[:, 1] = pixel_coords[:, 1] / H

        # Distance from center [0.5, 0.5]
        center = torch.tensor([0.5, 0.5], device=pixel_coords.device)
        dist_from_center = ((px_norm - center) ** 2).sum(dim=-1).sqrt()

        # Gaussian weight (higher near center)
        importance = torch.exp(-dist_from_center ** 2 / (2 * sigma ** 2))

        # UV spacing
        uv = self.cache
        du = torch.diff(uv[..., 0], dim=0).abs().reshape(-1)
        dv = torch.diff(uv[..., 1], dim=1).abs().reshape(-1)

        # We want smaller spacing (higher density) where importance is high
        # Correlate spacing with importance (want negative correlation)
        visible_flat = visible.reshape(-1)

        # Approximate:  penalize if high-importance regions have large spacing
        importance_grid = importance.reshape(self._Us, self._Vs)
        importance_u = (importance_grid[:-1, :] + importance_grid[1:, :]) / 2
        importance_v = (importance_grid[:, :-1] + importance_grid[:, 1:]) / 2

        # Loss: importance-weighted spacing (minimize)
        loss_u = (importance_u.reshape(-1) * du).mean()
        loss_v = (importance_v.reshape(-1) * dv).mean()

        return weight * (loss_u + loss_v)

    # ==================== UTILITIES ====================

    def step(self):
        """Call each training step to update internal counters."""
        self._training_step += 1

    def get_visibility_stats(self) -> dict:
        """Get visibility statistics for logging."""
        if self.cache_visible_mask is None:
            return {'visibility_ratio': 0.0}

        mask = self.cache_visible_mask
        return {
            'visibility_ratio': mask.float().mean().item(),
            'num_visible': mask.sum().item(),
            'num_total': mask.numel()
        }

    def export_intervals_u(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._interval_u_dc, self._interval_u_rest

    def export_intervals_v(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._interval_v_dc, self._interval_v_rest


# from typing import Tuple
#
# import torch
# from torch import nn
#
# from model.modules import SamplerUV, ModelState
# from model.spline_utils import inverse_sigmoid
# from utils.sh_utils import eval_sh
#
# # Custom UV activation functions (not RGB!)
# def UV2SH(uv: torch.Tensor) -> torch.Tensor:
#     """
#     Convert UV coordinates [0, 1] to SH DC coefficient space.
#     We want:  eval_sh(deg=0, sh_dc, _) = C0 * sh_dc = uv
#     Therefore: sh_dc = uv / C0
#     """
#     C0 = 0.28209479177387814
#     return uv / C0
#
#
# def SH2UV(sh_output: torch.Tensor) -> torch.Tensor:
#     """
#     Convert SH evaluation output back to UV space [0, 1].
#     Since eval_sh already multiplies by C0 for DC, and higher-order
#     terms add view-dependent offsets, we just need to clamp to [0, 1].
#
#     For DC-only (degree 0): eval_sh output = C0 * (uv / C0) = uv ✓
#     """
#     return sh_output  # No transformation needed if UV2SH is correct
#
#
# class ViewDependentUV(SamplerUV):
#     """
#     Module to handle sampling intervals (u/v samples).
#     """
#
#     def __init__(
#             self,
#             state: ModelState,
#             sh_degree: int = 3,
#             view_dependence_scale=0.1,
#             **kwargs
#     ):
#         kwargs['num_channels'] = (sh_degree + 1) ** 2
#
#
#         super().__init__(state, mode='sh', **kwargs)
#         self.evaluate_mode = kwargs.get('evaluate_mode', False)
#         self.sh_degree = sh_degree
#         self._active_sh_degree = 0
#         self.view_dependence_scale = view_dependence_scale
#
#         # Store for proper inverse/forward transforms
#         self.C0 = 0.28209479177387814
#
#         eps = 1e-3
#         Us = int(state.opt.sampling_density * state.H)
#         Vs = int(state.opt.sampling_density * state.W)
#         self._Us = Us
#         self._Vs = Vs
#
#         if kwargs.get('loaded_us_dc', None) is not None:
#             initial_samples_u_dc = kwargs.get('loaded_us', None)[0].to(self.device)
#             initial_samples_v_dc = kwargs.get('loaded_vs', None)[0].to(self.device)
#             initial_samples_u_rest = kwargs.get('loaded_us', None)[1].to(self.device)
#             initial_samples_v_rest = kwargs.get('loaded_vs', None)[1].to(self.device)
#             self._interval_u_dc = initial_samples_u_dc if self.evaluate_mode else nn.Parameter(initial_samples_u_dc, requires_grad=True)
#             self._interval_v_dc = initial_samples_v_dc if self.evaluate_mode else nn.Parameter(initial_samples_v_dc, requires_grad=True)
#             self._interval_u_rest = initial_samples_u_rest if self.evaluate_mode else nn.Parameter(initial_samples_u_rest, requires_grad=True)
#             self._interval_v_rest = initial_samples_v_rest if self.evaluate_mode else nn.Parameter(initial_samples_v_rest, requires_grad=True)
#
#             # Cache for loss computation
#             self.cache = None
#             self._base_u = torch.linspace(eps, 1-eps, Us, device=state.device)
#             self._base_v = torch.linspace(eps, 1-eps, Vs, device=state.device)
#
#         else:
#
#             # Create uniform grid in UV space [eps, 1-eps]
#             base_samples_u = torch.linspace(eps, 1 - eps, Us, device=state.device)
#             base_samples_v = torch.linspace(eps, 1 - eps, Vs, device=state.device)
#
#             # Reshape for grid:  [Us*Vs, 1, 1]
#             base_samples_u = base_samples_u.unsqueeze(1).repeat(1, Vs).reshape(Us * Vs, 1, 1)
#             base_samples_v = base_samples_v.unsqueeze(0).repeat(Us, 1).reshape(Us * Vs, 1, 1)
#
#             # Initialize DC coefficients so that:  C0 * dc = base_sample
#             # Therefore: dc = base_sample / C0
#             # base_dc_u = inverse_sigmoid(base_samples_u / self.C0)
#             # base_dc_v = inverse_sigmoid(base_samples_v / self.C0)
#
#             base_dc_u = torch.logit(base_samples_u)
#             base_dc_v = torch.logit(base_samples_v)
#
#             # Higher-order SH coefficients start near zero (minimal view-dependence initially)
#             # These will be scaled by view_dependence_scale during forward pass
#             num_higher_sh = (self.sh_degree + 1) ** 2 - 1
#             u_sh_rest = inverse_sigmoid(torch.zeros(Us * Vs, 1, num_higher_sh, device=state.device) + eps)
#             v_sh_rest = inverse_sigmoid(torch.zeros(Us * Vs, 1, num_higher_sh, device=state.device) + eps)
#
#             # Parameters
#             self._interval_u_dc = nn.Parameter(base_dc_u.contiguous(), requires_grad=True)
#             self._interval_v_dc = nn.Parameter(base_dc_v.contiguous(), requires_grad=True)
#             self._interval_u_rest = nn.Parameter(u_sh_rest.contiguous(), requires_grad=True)
#             self._interval_v_rest = nn.Parameter(v_sh_rest.contiguous(), requires_grad=True)
#
#             # Cache for loss computation
#             self.cache = None
#             self._base_u = base_samples_u.squeeze()  # Store original for regularization
#             self._base_v = base_samples_v.squeeze()
#
#     @property
#     def active_sh_degree(self) -> int:
#         return self._active_sh_degree
#
#     @property
#     def interval_u(self) -> torch.Tensor:
#         """Full SH coefficients for U, truncated to active degree."""
#         num_active = (self._active_sh_degree + 1) ** 2
#         rest_active = num_active - 1
#         if rest_active > 0:
#             return torch.cat([
#                 (torch.sigmoid(self._interval_u_dc) / self.C0),
#                 torch.sigmoid(self._interval_u_rest[..., : rest_active])
#             ], dim=-1)
#         return torch.sigmoid(self._interval_u_dc) / self.C0
#
#     @property
#     def interval_v(self) -> torch.Tensor:
#         """Full SH coefficients for V, truncated to active degree."""
#         num_active = (self._active_sh_degree + 1) ** 2
#         rest_active = num_active - 1
#         if rest_active > 0:
#             return torch.cat([
#                 (torch.sigmoid(self._interval_v_dc) / self.C0),
#                 torch.sigmoid(self._interval_v_rest[..., : rest_active])
#             ], dim=-1)
#         return torch.sigmoid(self._interval_v_dc) / self.C0
#
#     @property
#     def uv_sh_features(self) -> torch.Tensor:
#         """Combined UV SH features [N, 2, num_sh_coeffs]."""
#         return torch.cat([self.interval_u, self.interval_v], dim=1)
#
#     @property
#     def uv_dc_features(self) -> torch.Tensor:
#         """DC-only features for U and V."""
#         return torch.cat([self._interval_u_dc, self._interval_v_dc], dim=1)
#
#     @property
#     def uv_sh_features_rest(self) -> torch.Tensor:
#         """Higher-order SH features (excluding DC)."""
#         num_active = (self._active_sh_degree + 1) ** 2
#         rest_active = num_active - 1
#         if rest_active > 0:
#             return torch.cat([
#                 self._interval_u_rest[..., :rest_active],
#                 self._interval_v_rest[..., :rest_active]
#             ], dim=1)
#         return torch.zeros(self._interval_u_dc.shape[0], 2, 0,
#                            device=self._interval_u_dc.device)
#
#     def forward(self, dirs: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass to compute UV intervals based on view direction.
#
#         Args:
#             dirs: View directions [N, 3] or broadcastable shape
#
#         Returns:
#             UV coordinates [Us, Vs, 2] in range [0, 1]
#         """
#         # Evaluate SH:  output is already in UV space due to our initialization
#         # Shape: [Us*Vs, 2] where dim 1 is (u, v)
#         uv = eval_sh(self._active_sh_degree, self.uv_sh_features, dirs)
#
#         # Reshape to grid
#         uv = uv.reshape(self._Us, self._Vs, 2)
#
#         # Sort to maintain monotonicity (u along dim 0, v along dim 1)
#         # u, v = uv[..., 0], uv[..., 1]
#         # u_sorted = u.sort(dim=0)[0]
#         # v_sorted = v.sort(dim=1)[0]
#         # uv = torch.stack([u_sorted, v_sorted], dim=-1)
#         # uv = torch.stack([u, v], dim=-1)
#
#
#         # Clamp to valid UV range
#         uv = uv.clamp(0.0, 1.0)
#
#
#         # Cache for loss computation
#         # self.cache = uv
#
#         return uv
#
#     def oneUpSHdegree(self) -> None:
#         """Increase active SH degree by 1, up to max."""
#         if self._active_sh_degree < self.sh_degree:
#             self._active_sh_degree += 1
#             print(f"SH degree increased to {self._active_sh_degree}")
#
#     def coverage_loss(self, weight: float = 1e-2) -> torch.Tensor:
#         """
#         Regularization loss to encourage good coverage of UV space.
#
#         Returns a loss that:
#         1. Penalizes deviation from uniform coverage (low variance = bad)
#         2. Penalizes boundary violations
#         """
#         if self.cache is None:
#             return torch.tensor(0.0, device=self._interval_u_dc.device)
#
#         uv = self.cache
#
#         # Encourage spread:  penalize LOW variance (we want high variance = good coverage)
#         # Using negative variance or 1/variance can cause instability, so we use:
#         # loss = -log(var + eps) which encourages higher variance
#         var_u = torch.var(uv[..., 0])
#         var_v = torch.var(uv[..., 1])
#         variance_loss = -torch.log(var_u + 1e-6) - torch.log(var_v + 1e-6)
#
#         # Penalize values too close to boundaries or outside [0, 1]
#         boundary_margin = 0.02
#         boundary_loss = (
#                 torch.relu(boundary_margin - uv).sum() +  # Too close to 0
#                 torch.relu(uv - (1 - boundary_margin)).sum()  # Too close to 1
#         )
#
#         return weight * (variance_loss + boundary_loss)
#
#     def uniformity_loss(self, weight: float = 1e-6) -> torch.Tensor:
#         """
#         Loss to encourage uniform spacing between adjacent samples.
#         """
#         if self.cache is None:
#             return torch.tensor(0.0, device=self._interval_u_dc.device)
#
#         uv = self.cache
#
#         # Compute spacing between adjacent samples
#         du = torch.diff(uv[..., 0], dim=0)  # [Us-1, Vs]
#         dv = torch.diff(uv[..., 1], dim=1)  # [Us, Vs-1]
#
#         # Penalize variance in spacing (want uniform gaps)
#         spacing_var_u = torch.var(du)
#         spacing_var_v = torch.var(dv)
#
#         return weight * (spacing_var_u + spacing_var_v)
#
#     def dc_regularization_loss(self, weight: float = 1e-4) -> torch.Tensor:
#         """
#         Regularize DC components to stay close to initial uniform distribution.
#         This prevents the model from collapsing to a small region early in training.
#         """
#         # Current DC values (in UV space)
#         current_u_dc = self._interval_u_dc * self.C0  # Convert back to UV
#         current_v_dc = self._interval_v_dc * self.C0
#
#         # L2 loss to original positions
#         loss_u = torch.mean((current_u_dc.squeeze() - self._base_u.detach()) ** 2)
#         loss_v = torch.mean((current_v_dc.squeeze() - self._base_v.detach()) ** 2)
#
#         return weight * (loss_u + loss_v)
#
#     def export_intervals_u(self, activate: bool = True) -> torch.Tensor:
#         """Export U intervals, optionally clamped to [0, 1]."""
#         # u_dc = self._interval_u_dc * self.C0  # Convert to UV space
#         # if activate:
#         #     return u_dc.clamp(0.0, 1.0)
#         return self._interval_u_dc, self._interval_u_rest
#
#     def export_intervals_v(self, activate: bool = True) -> torch.Tensor:
#         """Export V intervals, optionally clamped to [0, 1]."""
#         # v_dc = self._interval_v_dc * self.C0  # Convert to UV space
#         # if activate:
#         #     return v_dc.clamp(0.0, 1.0)
#         return self._interval_v_dc, self._interval_v_rest
#
#     def subdivide_feature(
#             self,
#             direction: str,
#             knots: torch.Tensor,
#             degree: int,
#             val: float,
#             insertion_fn: callable
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes new UV samples after subdivision for the 1D interval case.
#
#         Args:
#             direction: 'u' or 'v'
#             knots: Current knot vector
#             degree:  Spline degree
#             val: Value to insert
#             insertion_fn: Function to compute new control points
#
#         Returns:
#             Tuple of (new_knots, new_features)
#         """
#         if direction == 'u':
#             features = self._interval_u_dc
#         elif direction == 'v':
#             features = self._interval_v_dc
#         else:
#             raise ValueError(f"direction must be 'u' or 'v', got {direction}")
#
#         # Apply insertion function to compute new features
#         new_knots, new_features = insertion_fn(knots, features, degree, val)
#
#         return new_knots, new_features
#
#
# def get_dirs_from_surface_points(
#         camera,
#         surface_points: torch.Tensor,
#         normalize: bool = False
# ) -> torch.Tensor:
#     """
#     Compute view directions from camera to surface points.
#
#     This is the most geometrically correct approach if you have
#     the 3D positions of your surface samples.
#
#     Args:
#         camera: Camera object with camera_center attribute
#         surface_points: [N, 3] or [Us, Vs, 3] world-space surface points
#         normalize: Whether to normalize directions
#
#     Returns:
#         dirs: Same shape as input but last dim is 3 (directions)
#     """
#     original_shape = surface_points.shape[:-1]
#     points_flat = surface_points.reshape(-1, 3)  # [N, 3]
#     Us, Vs = original_shape
#     # Direction from camera to surface point
#     camera_center = camera.get_rays(size=(Us, Vs)).reshape(-1, 3)
#     # [3]
#     dirs = points_flat - camera_center
#
#     if normalize:
#         dir_norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-8
#         dirs = dirs / dir_norms
#
#     return dirs.reshape(-1, 3)
#
#
#
#
# import torch.nn.functional as F
# def get_view_directions_for_surface_points(
#         camera,
#         surface_points: torch.Tensor,
#         scale: float = 1.0,
#         return_in_world_space: bool = True
# ) -> torch.Tensor:
#     """
#     Compute view directions by projecting surface points to pixels,
#     then sampling from the camera's ray grid.
#
#     This provides geometrically accurate ray directions that are
#     consistent with the camera's intrinsic parameters.
#
#     Args:
#         camera: Camera object with get_rays, world_to_camera, R, T
#         surface_points: [N, 3] world-space surface points
#         scale: Resolution scale for ray grid
#         return_in_world_space:  If True, transform rays to world space
#
#     Returns:
#         dirs: [N, 3] normalized ray directions
#     """
#     device = surface_points.device
#     N = surface_points.shape[0]
#
#     # Step 1: Get the full ray grid from camera [W, H, 3] in camera space
#     rays_cam = camera.get_rays(scale=scale)  # [W, H, 3]
#     H, W = rays_cam.shape[0], rays_cam.shape[1]
#
#     # Normalize rays (they're not unit vectors by default)
#     rays_cam = F.normalize(rays_cam, dim=-1)
#
#     # Step 2: Project surface points to pixel coordinates
#     # First transform to camera space
#     cam_points = camera.world_to_camera(surface_points)  # [N, 3]
#
#     # Then project to pixel coords
#     pixel_coords = camera.camera_to_image(cam_points)  # [N, 2] - (u, v)
#
#     # Step 3: Normalize pixel coords to [-1, 1] for grid_sample
#     pixel_coords_norm = torch.zeros_like(pixel_coords)
#     pixel_coords_norm[:, 0] = 2 * (pixel_coords[:, 0] / scale) / (W - 1) - 1  # x
#     pixel_coords_norm[:, 1] = 2 * (pixel_coords[:, 1] / scale) / (H - 1) - 1  # y
#
#     # Clamp to valid range (handles points outside image)
#     pixel_coords_norm = pixel_coords_norm.clamp(-1, 1)
#
#     # Step 4: Sample rays at projected pixel locations
#     # grid_sample expects [B, C, H, W] input and [B, H_out, W_out, 2] grid
#     rays_bhwc = rays_cam.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
#     grid = pixel_coords_norm.view(1, N, 1, 2)  # [1, N, 1, 2]
#
#     sampled_rays = F.grid_sample(
#         rays_bhwc,
#         grid,
#         mode='bilinear',
#         align_corners=True,
#         padding_mode='border'
#     )  # [1, 3, N, 1]
#
#     # Reshape to [N, 3]
#     dirs_cam = sampled_rays.squeeze().T  # [N, 3]
#
#     # Step 5: Transform to world space if requested
#     if return_in_world_space:
#         dirs_world = transform_directions_to_world(camera, dirs_cam)
#         return F.normalize(dirs_world, dim=-1)
#
#     return F.normalize(dirs_cam, dim=-1)
#
#
# def transform_directions_to_world(camera, dirs_cam: torch.Tensor) -> torch.Tensor:
#     """
#     Transform direction vectors from camera space to world space.
#
#     For directions (not points), we only apply the rotation, not translation.
#
#     Args:
#         camera: Camera object with R attribute
#         dirs_cam: [N, 3] directions in camera space
#
#     Returns:
#         dirs_world:  [N, 3] directions in world space
#     """
#     # R is the rotation matrix from world to camera (R @ world_vec = cam_vec)
#     # So R. T transforms from camera to world
#     R = torch.tensor(camera.R, dtype=torch.float32, device=dirs_cam.device)
#
#     # dirs_world = dirs_cam @ R (since R.T. T = R, and we're doing row vectors)
#     # Actually:  cam = world @ R. T, so world = cam @ R
#     dirs_world = dirs_cam @ R
#
#     return dirs_world
#
#
# def get_view_directions_with_visibility_mask(
#         camera,
#         surface_points: torch.Tensor,
#         scale: float = 1.0
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Get view directions and a mask indicating which points are visible
#     (project within the image bounds and have positive depth).
#
#     Returns:
#         dirs: [N, 3] normalized ray directions in world space
#         visible_mask: [N] boolean mask
#     """
#     device = surface_points.device
#     N = surface_points.shape[0]
#
#     # Get camera-space points for depth check
#     cam_points = camera.world_to_camera(surface_points)  # [N, 3]
#
#     # Check positive depth (in front of camera)
#     depth_valid = cam_points[:, 2] > 0
#
#     # Project to pixels
#     pixel_coords = camera.camera_to_image(cam_points)  # [N, 2]
#
#     # Check within image bounds
#     W, H = camera.image_width / scale, camera.image_height / scale
#     in_bounds = (
#             (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W * scale) &
#             (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H * scale)
#     )
#
#     visible_mask = depth_valid & in_bounds
#
#     # Get directions (will be clamped for out-of-bounds points)
#     dirs = get_view_directions_for_surface_points(
#         camera, surface_points, scale=scale, return_in_world_space=True
#     )
#
#     return dirs, visible_mask

