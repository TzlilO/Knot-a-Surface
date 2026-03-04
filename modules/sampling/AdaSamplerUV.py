from typing import Tuple, Optional, List, Dict, NamedTuple
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SamplingContext:
    """Container for all information needed for UV sampling."""
    camera: 'Camera'
    visibility_filter: torch.Tensor  # [Us*Vs] or [Us, Vs] from renderer
    radii: Optional[torch.Tensor]  # [Us*Vs] splat radii (optional)
    normals_grid: Optional[torch.Tensor]  # [Us, Vs, 3] surface normals (optional)
    xyz_grid: Optional[torch.Tensor]  # [Us, Vs, 3] 3D points (optional)
    depth_map: Optional[torch.Tensor]  # [H, W] rendered depth (optional)
    error_map: Optional[torch.Tensor]  # [H, W] reconstruction error (optional)


class AdvancedUVSampler(nn.Module):
    """
    Advanced UV sampler with statistical analysis and ray-based visibility. 

    This class replaces the original SamplerUV and provides: 
    - Proper ray directions using camera's get_rays()
    - Error-based adaptive sampling
    - Depth and frequency-aware weighting
    - Multi-view consistency
    - View-dependent sampling via SH (optional)
    """

    def __init__(
            self,
            state: 'ModelState',
            n_bins_u: int = 32,
            n_bins_v: int = 32,
            use_error_weighting: bool = True,
            use_depth_weighting: bool = True,
            use_frequency_weighting: bool = True,
            error_weight: float = 0.4,
            depth_weight: float = 0.2,
            frequency_weight: float = 0.2,
            visibility_weight: float = 0.2,
            **kwargs
    ):
        super().__init__()

        self.state = state
        self.device = state.device
        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.should_optimize = self.state.opt.optimize_intervals and not kwargs.get('evaluate_mode', False)
        self.num_channels = kwargs.get('num_channels', 1)
        self.sh_degree = kwargs.get('sh_degree', None)
        self.mode = kwargs.get('mode', 'adaptive')
        # Sampling configuration
        self.n_bins_u = n_bins_u
        self.n_bins_v = n_bins_v
        self.use_error_weighting = use_error_weighting
        self.use_depth_weighting = use_depth_weighting
        self.use_frequency_weighting = use_frequency_weighting

        # Weight factors (should sum to ~1)
        self.error_weight = error_weight
        self.depth_weight = depth_weight
        self.frequency_weight = frequency_weight
        self.visibility_weight = visibility_weight

        # Grid dimensions
        eps = 1e-3
        Us = int(state.opt.sampling_density * state.H)
        Vs = int(state.opt.sampling_density * state.W)
        self._Us = Us
        self._Vs = Vs
        self._num_samples = Us * Vs
        self._eps = eps

        # Base uniform grid (registered as buffer - not a parameter)
        base_u = torch.linspace(eps, 1 - eps, Us, device=state.device)
        base_v = torch.linspace(eps, 1 - eps, Vs, device=state.device)

        grid_u, grid_v = torch.meshgrid(base_u, base_v, indexing='ij')
        base_grid = torch.stack([grid_u, grid_v], dim=-1)  # [Us, Vs, 2]

        self.register_buffer('_base_grid', base_grid)
        self.register_buffer('_base_u', base_u)
        self.register_buffer('_base_v', base_v)

        # Per-view diff grids (learnable or computed)
        # Using nn. ParameterDict for per-camera storage
        self._diff_u_cache: Dict[int, torch.Tensor] = {}
        self._diff_v_cache: Dict[int, torch.Tensor] = {}

        # History for multi-view consistency
        self._visibility_history: Dict[int, torch.Tensor] = {}
        self._error_history: Dict[int, torch.Tensor] = {}
        self._weight_history: Dict[int, torch.Tensor] = {}

        # Current active view
        self.active_uid: Optional[int] = None

        # Cached results
        self.cache: Optional[torch.Tensor] = None
        self.cache_weights: Optional[torch.Tensor] = None
        self.cache_ray_info: Optional[dict] = None

    @property
    def Us(self) -> int:
        return self._Us

    @property
    def Vs(self) -> int:
        return self._Vs

    # ==================== PUBLIC API ====================

    def get_uniform_grid(self) -> torch.Tensor:
        """Get the base uniform UV grid [Us, Vs, 2]."""
        return self._base_grid.clone()

    def forward(
            self,
            ctx: SamplingContext,
            xyz_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Main sampling method - computes adaptive UV sampling.

        Args:
            ctx: SamplingContext with camera and optional rendering info
            xyz_grid: [Us, Vs, 3] surface points (required if not in ctx)

        Returns:
            sampled_uv:  [Us, Vs, 2] sampled UV coordinates
        """
        self.active_uid = ctx.camera.uid

        # Use provided xyz or from context
        if xyz_grid is not None:
            surface_points = xyz_grid
        elif ctx.xyz_grid is not None:
            surface_points = ctx.xyz_grid
        else:
            raise ValueError("Must provide xyz_grid either directly or in context")

        # Ensure correct shape
        if surface_points.dim() == 2:
            surface_points = surface_points.reshape(self._Us, self._Vs, 3)

        # Compute all weights
        combined_weights = self._compute_combined_weights(ctx, surface_points)

        # Get diff grids for this view
        diff_u = self.diff_u(ctx.camera.uid)
        diff_v = self.diff_v(ctx.camera.uid)

        # Sample using importance-weighted binning
        sampled_uv = self._importance_weighted_sampling(
            weights=combined_weights,
            diff_u_grid=diff_u,
            diff_v_grid=diff_v
        )

        # Cache results
        self.cache = sampled_uv
        self.cache_weights = combined_weights

        # Update history
        self._update_history(ctx.camera.uid, combined_weights)

        return sampled_uv

    def sample_uv(
            self,
            camera: 'Camera',
            visibility_filter: torch.Tensor,
            xyz_grid: torch.Tensor,
            radii: Optional[torch.Tensor] = None,
            normals_grid: Optional[torch.Tensor] = None,
            rendered_image: Optional[torch.Tensor] = None,
            target_image: Optional[torch.Tensor] = None,
            depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method matching the expected API.

        Args:
            camera: Camera object
            visibility_filter:  [Us*Vs] or [Us, Vs] visibility from renderer
            xyz_grid: [Us, Vs, 3] or [Us*Vs, 3] surface points
            radii: [Us*Vs] splat radii (optional)
            normals_grid: [Us, Vs, 3] surface normals (optional, computed if None)
            rendered_image:  Rendered image for error computation (optional)
            target_image: Target image for error computation (optional)
            depth_map: [H, W] rendered depth (optional)

        Returns: 
            sampled_uv: [Us, Vs, 2] sampled UV coordinates
        """
        # Reshape xyz if needed
        if xyz_grid.dim() == 2:
            xyz_grid = xyz_grid.reshape(self._Us, self._Vs, 3)

        # Compute normals if not provided
        if normals_grid is None:
            normals_grid = self._compute_surface_normals(xyz_grid)

        # Compute error map if images provided
        error_map = None
        if rendered_image is not None and target_image is not None:
            error_map = self._compute_error_map(rendered_image, target_image)

        # Create context
        ctx = SamplingContext(
            camera=camera,
            visibility_filter=visibility_filter,
            radii=radii,
            normals_grid=normals_grid,
            xyz_grid=xyz_grid,
            depth_map=depth_map,
            error_map=error_map
        )

        return self.forward(ctx, xyz_grid)

    def diff_u(self, cam_uid: int) -> torch.Tensor:
        """Get or compute diff grid in U direction for a camera."""
        if cam_uid not in self._diff_u_cache:
            # Initialize with uniform spacing
            self._diff_u_cache[cam_uid] = torch.ones(
                self._Us, self._Vs, device=self.device
            ) / self._Us
        return self._diff_u_cache[cam_uid]

    def diff_v(self, cam_uid: int) -> torch.Tensor:
        """Get or compute diff grid in V direction for a camera."""
        if cam_uid not in self._diff_v_cache:
            # Initialize with uniform spacing
            self._diff_v_cache[cam_uid] = torch.ones(
                self._Us, self._Vs, device=self.device
            ) / self._Vs
        return self._diff_v_cache[cam_uid]

    def update_diff_grids(
            self,
            cam_uid: int,
            sampled_uv: torch.Tensor
    ):
        """
        Update diff grids based on sampled UV for future iterations.

        Args:
            cam_uid: Camera UID
            sampled_uv: [Us, Vs, 2] sampled UV coordinates
        """
        # Compute actual spacing from sampled UV
        diff_u = torch.diff(sampled_uv[..., 0], dim=0)  # [Us-1, Vs]
        diff_v = torch.diff(sampled_uv[..., 1], dim=1)  # [Us, Vs-1]

        # Pad to full size
        diff_u = F.pad(diff_u, (0, 0, 0, 1), mode='replicate')  # [Us, Vs]
        diff_v = F.pad(diff_v, (0, 1, 0, 0), mode='replicate')  # [Us, Vs]

        # Exponential moving average with existing
        alpha = 0.3
        if cam_uid in self._diff_u_cache:
            self._diff_u_cache[cam_uid] = (
                    alpha * diff_u.abs() +
                    (1 - alpha) * self._diff_u_cache[cam_uid]
            )
            self._diff_v_cache[cam_uid] = (
                    alpha * diff_v.abs() +
                    (1 - alpha) * self._diff_v_cache[cam_uid]
            )
        else:
            self._diff_u_cache[cam_uid] = diff_u.abs()
            self._diff_v_cache[cam_uid] = diff_v.abs()

    # ==================== WEIGHT COMPUTATION ====================

    def _compute_combined_weights(
            self,
            ctx: SamplingContext,
            xyz_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined importance weights from all sources.
        """
        device = xyz_grid.device
        Us, Vs = self._Us, self._Vs

        # 1. Visibility weight (always used)
        visibility = ctx.visibility_filter.reshape(Us, Vs).float()

        # 2. Ray-based orientation weight
        orientation_weight = self._compute_orientation_weight(ctx, xyz_grid)

        # 3. Error weight (if available)
        if self.use_error_weighting and ctx.error_map is not None:
            error_weight = self._compute_error_weight(ctx, xyz_grid)
        else:
            error_weight = torch.ones(Us, Vs, device=device)

        # 4. Depth weight (if available)
        if self.use_depth_weighting and ctx.depth_map is not None:
            depth_weight = self._compute_depth_weight(ctx, xyz_grid)
        else:
            depth_weight = torch.ones(Us, Vs, device=device)

        # 5. Frequency weight (surface curvature)
        if self.use_frequency_weighting and ctx.normals_grid is not None:
            frequency_weight = self._compute_frequency_weight(ctx.normals_grid)
        else:
            frequency_weight = torch.ones(Us, Vs, device=device)

        # 6. Radii weight (if available)
        if ctx.radii is not None:
            radii_weight = self._compute_radii_weight(ctx.radii)
        else:
            radii_weight = torch.ones(Us, Vs, device=device)

        # Combine weights
        combined = (
                visibility * orientation_weight *
                (
                        self.visibility_weight * visibility +
                        self.error_weight * error_weight +
                        self.depth_weight * depth_weight +
                        self.frequency_weight * frequency_weight
                ) *
                radii_weight
        )

        # Normalize
        combined = combined / (combined.sum() + 1e-8)

        # Optional: blend with multi-view history
        combined = self._blend_with_history(ctx.camera.uid, combined)

        return combined

    def _compute_orientation_weight(
            self,
            ctx: SamplingContext,
            xyz_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute orientation weight using proper ray directions from get_rays().
        """
        Us, Vs = self._Us, self._Vs
        device = xyz_grid.device
        camera = ctx.camera

        # Get pixel-aligned rays
        H, W = camera.image_height, camera.image_width
        pixel_rays = camera.get_rays(scale=1.0)  # [H, W, 3] camera space
        pixel_rays = F.normalize(pixel_rays, dim=-1)

        # Project surface points to pixels
        xyz_flat = xyz_grid.reshape(-1, 3)
        cam_points = camera.world_to_camera(xyz_flat)
        pixel_coords = camera.camera_to_image(cam_points)
        depths = cam_points[:, 2]

        # Visibility check
        visible = (
                (depths > 1e-3) &
                (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
                (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
        )

        # Sample rays at pixel locations
        px_norm = torch.zeros_like(pixel_coords)
        px_norm[:, 0] = (2 * pixel_coords[:, 0] / (W - 1) - 1).clamp(-1, 1)
        px_norm[:, 1] = (2 * pixel_coords[:, 1] / (H - 1) - 1).clamp(-1, 1)

        rays_nchw = pixel_rays.permute(2, 0, 1).unsqueeze(0)
        grid = px_norm.view(1, Us * Vs, 1, 2)

        sampled_rays_cam = F.grid_sample(
            rays_nchw, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).squeeze().T  # [N, 3]

        # Transform to world space
        R = torch.tensor(camera.R, dtype=torch.float32, device=device)
        ray_dirs_world = sampled_rays_cam @ R
        ray_dirs_world = F.normalize(ray_dirs_world, dim=-1)

        # Fallback for non-visible points
        ray_dirs_geom = xyz_flat - camera.camera_center.unsqueeze(0)
        ray_dirs_geom = F.normalize(ray_dirs_geom, dim=-1)

        ray_dirs = torch.where(
            visible.unsqueeze(-1),
            ray_dirs_world,
            ray_dirs_geom
        ).reshape(Us, Vs, 3)

        # Store ray info for potential later use
        self.cache_ray_info = {
            'ray_dirs': ray_dirs,
            'visible': visible.reshape(Us, Vs),
            'pixel_coords': pixel_coords.reshape(Us, Vs, 2),
            'depths': depths.reshape(Us, Vs)
        }

        # Compute orientation:  normal · (-ray_dir)
        if ctx.normals_grid is not None:
            normals = ctx.normals_grid
        else:
            normals = self._compute_surface_normals(xyz_grid)

        cos_theta = torch.einsum('ijk,ijk->ij', normals, -ray_dirs)
        orientation_weight = cos_theta.clamp(0, 1) ** 0.5  # Soft falloff

        # Zero weight for non-visible points
        orientation_weight = orientation_weight * visible.float().reshape(Us, Vs)

        return orientation_weight

    def _compute_error_weight(
            self,
            ctx: SamplingContext,
            xyz_grid: torch.Tensor
    ) -> torch.Tensor:
        """Sample reconstruction error at UV locations."""
        Us, Vs = self._Us, self._Vs
        device = xyz_grid.device
        camera = ctx.camera

        H, W = ctx.error_map.shape

        # Use cached pixel coords if available
        if self.cache_ray_info is not None:
            pixel_coords = self.cache_ray_info['pixel_coords'].reshape(-1, 2)
        else:
            xyz_flat = xyz_grid.reshape(-1, 3)
            cam_points = camera.world_to_camera(xyz_flat)
            pixel_coords = camera.camera_to_image(cam_points)

        # Sample error map
        px_norm = torch.zeros_like(pixel_coords)
        px_norm[:, 0] = (2 * pixel_coords[:, 0] / (W - 1) - 1).clamp(-1, 1)
        px_norm[:, 1] = (2 * pixel_coords[:, 1] / (H - 1) - 1).clamp(-1, 1)

        error_batch = ctx.error_map.view(1, 1, H, W)
        grid = px_norm.view(1, Us * Vs, 1, 2)

        sampled_error = F.grid_sample(
            error_batch, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).view(Us, Vs)

        # Normalize and apply non-linear boost
        error_weight = sampled_error / (sampled_error.max() + 1e-6)
        error_weight = error_weight ** 0.5  # Square root for gentler boost
        error_weight = 0.3 + 0.7 * error_weight  # Base weight

        return error_weight

    def _compute_depth_weight(
            self,
            ctx: SamplingContext,
            xyz_grid: torch.Tensor
    ) -> torch.Tensor:
        """Weight based on depth consistency and distance."""
        Us, Vs = self._Us, self._Vs
        device = xyz_grid.device

        # Get depths from cache or compute
        if self.cache_ray_info is not None:
            depths = self.cache_ray_info['depths']
            pixel_coords = self.cache_ray_info['pixel_coords'].reshape(-1, 2)
        else:
            xyz_flat = xyz_grid.reshape(-1, 3)
            cam_points = ctx.camera.world_to_camera(xyz_flat)
            depths = cam_points[:, 2].reshape(Us, Vs)
            pixel_coords = ctx.camera.camera_to_image(cam_points)

        H, W = ctx.depth_map.shape

        # Sample rendered depth
        px_norm = torch.zeros(Us * Vs, 2, device=device)
        px_norm[:, 0] = (2 * pixel_coords[:, 0] / (W - 1) - 1).clamp(-1, 1)
        px_norm[:, 1] = (2 * pixel_coords[:, 1] / (H - 1) - 1).clamp(-1, 1)

        depth_batch = ctx.depth_map.view(1, 1, H, W)
        grid = px_norm.view(1, Us * Vs, 1, 2)

        rendered_depth = F.grid_sample(
            depth_batch, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).view(Us, Vs)

        # Depth consistency weight
        depth_diff = (depths - rendered_depth).abs()
        consistency_weight = torch.exp(-depth_diff / (depths.mean() * 0.1 + 1e-6))

        # Inverse depth weight (closer = more important)
        inv_depth = 1.0 / (depths + 0.1)
        inv_depth = inv_depth / (inv_depth.max() + 1e-6)

        return consistency_weight * inv_depth

    def _compute_frequency_weight(
            self,
            normals: torch.Tensor
    ) -> torch.Tensor:
        """Weight based on surface curvature (normal variation)."""
        # Normal gradients as curvature proxy
        dn_du = torch.diff(normals, dim=0)
        dn_dv = torch.diff(normals, dim=1)

        curvature_u = dn_du.norm(dim=-1)
        curvature_v = dn_dv.norm(dim=-1)

        # Pad to full size
        curvature_u = F.pad(curvature_u, (0, 0, 0, 1), mode='replicate')
        curvature_v = F.pad(curvature_v, (0, 1, 0, 0), mode='replicate')

        curvature = (curvature_u + curvature_v) / 2
        frequency_weight = curvature / (curvature.max() + 1e-6)
        frequency_weight = 0.5 + 0.5 * frequency_weight

        return frequency_weight

    def _compute_radii_weight(
            self,
            radii: torch.Tensor
    ) -> torch.Tensor:
        """Weight based on splat radii (smaller = finer detail)."""
        radii_grid = radii.reshape(self._Us, self._Vs)

        # Smaller radius = higher weight
        inv_radii = 1.0 / (radii_grid + 1e-6)
        radii_weight = inv_radii / (inv_radii.max() + 1e-6)
        radii_weight = 0.5 + 0.5 * radii_weight

        return radii_weight

    # ==================== SAMPLING ====================

    def _importance_weighted_sampling(
            self,
            weights: torch.Tensor,
            diff_u_grid: torch.Tensor,
            diff_v_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample UV points using importance-weighted binning.
        """
        device = weights.device
        Us, Vs = self._Us, self._Vs
        num_samples = Us * Vs

        # Flatten
        weights_flat = weights.view(-1)
        uv_grid = self._base_grid
        u_flat = uv_grid[..., 0].view(-1)
        v_flat = uv_grid[..., 1].view(-1)

        # Compute non-uniform bin edges from diff grids
        avg_diff_u = diff_u_grid.mean(dim=1)
        avg_diff_v = diff_v_grid.mean(dim=0)

        edges_u = torch.cat([
            torch.tensor([0.], device=device),
            torch.cumsum(avg_diff_u, dim=0)
        ])
        if edges_u[-1] > 0:
            edges_u = edges_u / edges_u[-1]
        else:
            edges_u = torch.linspace(0, 1, Us + 1, device=device)

        edges_v = torch.cat([
            torch.tensor([0.], device=device),
            torch.cumsum(avg_diff_v, dim=0)
        ])
        if edges_v[-1] > 0:
            edges_v = edges_v / edges_v[-1]
        else:
            edges_v = torch.linspace(0, 1, Vs + 1, device=device)

        # Coarse bins
        n_bins_u, n_bins_v = self.n_bins_u, self.n_bins_v
        fine_idx_u = torch.linspace(0, Us, n_bins_u + 1, device=device).long()
        fine_idx_v = torch.linspace(0, Vs, n_bins_v + 1, device=device).long()
        edges_u_coarse = edges_u[fine_idx_u.clamp(0, Us)]
        edges_v_coarse = edges_v[fine_idx_v.clamp(0, Vs)]

        # Assign points to bins
        bin_idx_u = (torch.searchsorted(edges_u_coarse, u_flat) - 1).clamp(0, n_bins_u - 1)
        bin_idx_v = (torch.searchsorted(edges_v_coarse, v_flat) - 1).clamp(0, n_bins_v - 1)

        # Compute per-bin weights
        bin_weights = torch.zeros(n_bins_u * n_bins_v, device=device)
        flat_bin_idx = bin_idx_u * n_bins_v + bin_idx_v
        bin_weights.scatter_add_(0, flat_bin_idx, weights_flat)

        # Normalize to probability
        prob_sum = bin_weights.sum()
        if prob_sum > 0:
            bin_probs = bin_weights / prob_sum
        else:
            bin_probs = torch.ones_like(bin_weights) / bin_weights.numel()

        # Sample bins
        sampled_bin_idxs = torch.multinomial(bin_probs, num_samples, replacement=True)

        bin_u_idx = sampled_bin_idxs // n_bins_v
        bin_v_idx = sampled_bin_idxs % n_bins_v

        # Sample uniformly within bins
        bin_width_u = (edges_u_coarse[1:] - edges_u_coarse[:-1])[bin_u_idx]
        bin_width_v = (edges_v_coarse[1:] - edges_v_coarse[:-1])[bin_v_idx]

        lower_u = edges_u_coarse[bin_u_idx]
        lower_v = edges_v_coarse[bin_v_idx]

        rand_u = torch.rand(num_samples, device=device) * bin_width_u + lower_u
        rand_v = torch.rand(num_samples, device=device) * bin_width_v + lower_v

        # Sort for monotonic grid
        samples = torch.stack([rand_u, rand_v], dim=-1)

        v_sort_idx = torch.argsort(samples[:, 1])
        v_sorted = samples[v_sort_idx]
        rows = v_sorted.view(Us, Vs, 2)

        u_per_row = rows[..., 0]
        _, u_sort_idxs = torch.sort(u_per_row, dim=1)

        sampled_grid = torch.gather(
            rows, 1,
            u_sort_idxs.unsqueeze(-1).expand(-1, -1, 2)
        )

        return sampled_grid

    # ==================== UTILITIES ====================

    def _compute_surface_normals(self, xyz_grid: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from 3D point grid."""
        du = torch.zeros_like(xyz_grid)
        dv = torch.zeros_like(xyz_grid)

        # Central differences
        du[1:-1] = (xyz_grid[2:] - xyz_grid[:-2]) / 2
        du[0] = xyz_grid[1] - xyz_grid[0]
        du[-1] = xyz_grid[-1] - xyz_grid[-2]

        dv[:, 1:-1] = (xyz_grid[:, 2:] - xyz_grid[:, :-2]) / 2
        dv[:, 0] = xyz_grid[:, 1] - xyz_grid[:, 0]
        dv[:, -1] = xyz_grid[:, -1] - xyz_grid[:, -2]

        normals = torch.cross(du, dv, dim=-1)
        return F.normalize(normals, dim=-1)

    def _compute_error_map(
            self,
            rendered: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-pixel reconstruction error."""
        # Handle both [H, W, 3] and [3, H, W] formats
        if rendered.shape[0] == 3:
            rendered = rendered.permute(1, 2, 0)
            target = target.permute(1, 2, 0)

        return (rendered - target).pow(2).mean(dim=-1)

    def _blend_with_history(
            self,
            cam_uid: int,
            current_weights: torch.Tensor,
            blend_factor: float = 0.2
    ) -> torch.Tensor:
        """Blend current weights with historical weights for stability."""
        if cam_uid in self._weight_history:
            blended = (
                    (1 - blend_factor) * current_weights +
                    blend_factor * self._weight_history[cam_uid]
            )
            return blended / (blended.sum() + 1e-8)
        return current_weights

    def _update_history(
            self,
            cam_uid: int,
            weights: torch.Tensor
    ):
        """Update history for multi-view consistency."""
        self._weight_history[cam_uid] = weights.detach().clone()

    def get_multi_view_prior(
            self,
            current_cam_uid: int,
            nearby_cam_uids: List[int],
            blend_factor: float = 0.3
    ) -> Optional[torch.Tensor]:
        """Get aggregated weights from nearby views."""
        weights = []
        for uid in nearby_cam_uids:
            if uid in self._weight_history and uid != current_cam_uid:
                weights.append(self._weight_history[uid])

        if not weights:
            return None

        return torch.stack(weights).mean(dim=0)

    # ==================== EXPORT / SERIALIZATION ====================

    def export_state(self) -> dict:
        """Export sampler state for checkpointing."""
        return {
            'diff_u_cache': {k: v.cpu() for k, v in self._diff_u_cache.items()},
            'diff_v_cache': {k: v.cpu() for k, v in self._diff_v_cache.items()},
            'weight_history': {k: v.cpu() for k, v in self._weight_history.items()},
        }

    def load_state(self, state: dict):
        """Load sampler state from checkpoint."""
        if 'diff_u_cache' in state:
            self._diff_u_cache = {
                k: v.to(self.device) for k, v in state['diff_u_cache'].items()
            }
        if 'diff_v_cache' in state:
            self._diff_v_cache = {
                k: v.to(self.device) for k, v in state['diff_v_cache'].items()
            }
        if 'weight_history' in state:
            self._weight_history = {
                k: v.to(self.device) for k, v in state['weight_history'].items()
            }

    # ==================== DIAGNOSTICS ====================
    #
    # def get_stats(self) -> dict:
    #     """Get sampling statistics for logging."""
    #     stats = {
    #         'num_cached_views': len(self._weight_history),
    #         'grid_size': (self._Us, self._Vs),
    #     }
    #
    #     if self.cache_weights is not None:
    #         w = self.cache_weights
    #         stats.update({
    #             'weight_min': w.min().item(),
    #             'weight_max': w.max().item(),
    #             'weight_mean': w.mean().item(),
    #             'weight_std': w.std().item(),
    #             'effective_samples': (1. 0 / (w ** 2).sum()).item(),  # ESS
    #         })
    #
    #         if self.cache_ray_info is not None:
    #             visible = self.cache_ray_info['visible']
    #             stats['visibility_ratio'] = visible.float().mean().item()
    #
    #         return stats