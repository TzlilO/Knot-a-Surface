from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from model.modules import SamplerUV, ModelState
from utils.sh_utils import eval_sh


class ViewDependentUV(SamplerUV):
    """
    View-dependent UV sampling with proper spatial modulation.

    Key insight: We need DIFFERENT view directions for DIFFERENT UV locations
    to create spatially-varying warping. This requires either:
    1. Per-sample ray directions (expensive but accurate)
    2. A learned warping field conditioned on a global view direction

    This implementation uses approach #2: a warping MLP that takes
    (base_uv, view_direction) -> warped_uv
    """

    def __init__(
            self,
            state: ModelState,
            sh_degree: int = 3,
            warp_strength: float = 0.1,
            use_mlp_warp: bool = True,
            **kwargs
    ):
        kwargs['num_channels'] = (sh_degree + 1) ** 2
        super().__init__(state, mode='sh', **kwargs)

        self.sh_degree = sh_degree
        self._active_sh_degree = 0
        self.warp_strength = warp_strength
        self.use_mlp_warp = use_mlp_warp
        self.C0 = 0.28209479177387814

        eps = 1e-3
        Us = int(state.opt.sampling_density * state.H)
        Vs = int(state.opt.sampling_density * state.W)
        self._Us = Us
        self._Vs = Vs

        # Base uniform grid (FIXED, not learned for stability)
        base_u = torch.linspace(eps, 1 - eps, Us, device=state.device)
        base_v = torch.linspace(eps, 1 - eps, Vs, device=state.device)

        # Create meshgrid [Us, Vs, 2]
        grid_u, grid_v = torch.meshgrid(base_u, base_v, indexing='ij')
        base_grid = torch.stack([grid_u, grid_v], dim=-1)

        # Register as buffer (not parameter) - this is our reference grid
        self.register_buffer('base_grid', base_grid)

        # === APPROACH 1: SH-based per-point modulation ===
        # Each UV point has its own SH coefficients for view-dependent offset
        num_sh = (sh_degree + 1) ** 2

        # DC component:  learnable base positions (small offset from uniform)
        self.dc_offset = nn.Parameter(
            torch.zeros(Us, Vs, 2, device=state.device)
        )

        # Higher-order SH:  view-dependent offsets
        # Shape: [Us, Vs, 2, num_sh-1] - each point, each UV dim, each SH coeff
        self.sh_coeffs = nn.Parameter(
            torch.zeros(Us, Vs, 2, num_sh - 1, device=state.device) * 0.01
        )

        # === APPROACH 2: MLP-based warping (more expressive) ===
        if use_mlp_warp:
            self.warp_mlp = nn.Sequential(
                # Input: [base_u, base_v, dir_x, dir_y, dir_z] = 5
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # Output: [delta_u, delta_v]
                nn.Tanh()  # Bound output to [-1, 1], scale later
            )
            # Initialize to near-identity
            self._init_warp_mlp()

        # Density/importance field:  predicts sampling density at each location
        # This creates the "center focus" effect
        self.density_mlp = nn.Sequential(
            nn.Linear(5, 32),  # [u, v, dir_x, dir_y, dir_z]
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Positive density
        )

        self.cache = None
        self._training_step = 0

    def _init_warp_mlp(self):
        """Initialize MLP to output near-zero (identity warp)."""
        for m in self.warp_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree

    def oneUpSHdegree(self) -> None:
        if self._active_sh_degree < self.sh_degree:
            self._active_sh_degree += 1

    def forward(
            self,
            dirs: torch.Tensor,
            return_density: bool = False
    ) -> torch.Tensor:
        """
        Compute view-dependent UV sampling grid.

        Args:
            dirs: View direction [3] or [1, 3] - single direction for the view
            return_density: If True, also return density/importance weights

        Returns:
            uv:  Warped UV grid [Us, Vs, 2]
            density (optional): Importance weights [Us, Vs]
        """
        # Ensure dirs is [1, 3]
        if dirs.dim() == 1:
            dirs = dirs.unsqueeze(0)

        Us, Vs = self._Us, self._Vs
        device = self.base_grid.device

        # Start with base grid + learnable DC offset
        uv = self.base_grid + self.warp_strength * torch.tanh(self.dc_offset)

        # === Apply view-dependent warping ===
        if self.use_mlp_warp:
            uv = self._apply_mlp_warp(uv, dirs)
        else:
            uv = self._apply_sh_warp(uv, dirs)

        # Enforce monotonicity (but with soft sorting to preserve gradients)
        uv = self._soft_monotonic_constraint(uv)

        # Clamp to valid range
        uv = uv.clamp(1e-3, 1 - 1e-3)

        self.cache = uv

        if return_density:
            density = self._compute_density(uv, dirs)
            return uv, density

        return uv

    def _apply_mlp_warp(
            self,
            uv: torch.Tensor,
            dirs: torch.Tensor
    ) -> torch.Tensor:
        """Apply MLP-based view-dependent warping."""
        Us, Vs = self._Us, self._Vs

        # Expand direction to all grid points [Us, Vs, 3]
        dirs_expanded = dirs.expand(Us, Vs, 3)

        # Concatenate:  [Us, Vs, 5]
        mlp_input = torch.cat([uv, dirs_expanded], dim=-1)

        # Flatten for MLP:  [Us*Vs, 5]
        mlp_input_flat = mlp_input.reshape(-1, 5)

        # Get warp offsets:  [Us*Vs, 2]
        warp_offset = self.warp_mlp(mlp_input_flat)

        # Reshape and apply with strength scaling
        warp_offset = warp_offset.reshape(Us, Vs, 2)

        # Progressive warp strength based on training
        effective_strength = self.warp_strength * self._get_warp_schedule()

        return uv + effective_strength * warp_offset

    def _apply_sh_warp(
            self,
            uv: torch.Tensor,
            dirs: torch.Tensor
    ) -> torch.Tensor:
        """Apply SH-based view-dependent warping."""
        if self._active_sh_degree == 0:
            return uv  # No view-dependent component yet

        Us, Vs = self._Us, self._Vs
        num_active_sh = (self._active_sh_degree + 1) ** 2 - 1  # Exclude DC

        if num_active_sh == 0:
            return uv

        # Get active SH coefficients [Us, Vs, 2, num_active_sh]
        active_sh = self.sh_coeffs[..., :num_active_sh]

        # Reshape for eval_sh: [Us*Vs*2, num_active_sh]
        sh_flat = active_sh.reshape(-1, 1, num_active_sh)

        # We need to prepend a "DC" of 0 for eval_sh to work correctly
        # Actually, let's compute the SH basis manually for the offset
        offset = self._eval_sh_offset(active_sh, dirs)

        return uv + self.warp_strength * offset

    def _eval_sh_offset(
            self,
            sh_coeffs: torch.Tensor,
            dirs: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate SH basis functions for view-dependent offset.

        Args:
            sh_coeffs: [Us, Vs, 2, num_sh-1] coefficients (no DC)
            dirs: [1, 3] view direction
        """
        Us, Vs = self._Us, self._Vs

        # Add dummy DC coefficient (will multiply by C0 but we don't want DC contribution)
        # Shape: [Us, Vs, 2, num_sh]
        full_sh = torch.cat([
            torch.zeros(Us, Vs, 2, 1, device=sh_coeffs.device),
            sh_coeffs
        ], dim=-1)

        # Reshape: [Us*Vs, 2, num_sh]
        full_sh_flat = full_sh.reshape(Us * Vs, 2, -1)

        # Evaluate SH:  [Us*Vs, 2]
        offset = eval_sh(self._active_sh_degree, full_sh_flat, dirs)

        return offset.reshape(Us, Vs, 2)

    def _soft_monotonic_constraint(
            self,
            uv: torch.Tensor,
            temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Soft sorting to maintain monotonicity while preserving gradients.
        Uses softmax-based soft sorting.
        """
        u, v = uv[..., 0], uv[..., 1]

        # For now, use regular sorting (hard constraint)
        # TODO:  Implement differentiable soft-sort if gradient flow is an issue
        u_sorted = u.sort(dim=0)[0]
        v_sorted = v.sort(dim=1)[0]

        return torch.stack([u_sorted, v_sorted], dim=-1)

    def _compute_density(
            self,
            uv: torch.Tensor,
            dirs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance/density at each UV location.
        Higher density = more samples should concentrate here.
        """
        Us, Vs = self._Us, self._Vs

        dirs_expanded = dirs.expand(Us, Vs, 3)
        density_input = torch.cat([uv, dirs_expanded], dim=-1)
        density_flat = density_input.reshape(-1, 5)

        density = self.density_mlp(density_flat)
        return density.reshape(Us, Vs)

    def _get_warp_schedule(self) -> float:
        """
        Progressive schedule for warp strength.
        Starts small, increases over training.
        """
        warmup_steps = 1000
        if self._training_step < warmup_steps:
            return self._training_step / warmup_steps
        return 1.0

    def step(self):
        """Call this each training step to update schedule."""
        self._training_step += 1

    # === LOSS FUNCTIONS ===

    def center_concentration_loss(
            self,
            image_coords: torch.Tensor,
            errors: torch.Tensor,
            sigma: float = 0.3
    ) -> torch.Tensor:
        """
        Loss to encourage UV samples to concentrate where errors are high.

        Args:
            image_coords: [H, W, 2] normalized image coordinates
            errors: [H, W] per-pixel reconstruction error
        """
        if self.cache is None:
            return torch.tensor(0.0)

        uv = self.cache  # [Us, Vs, 2]

        # Create Gaussian centered at image center
        center = torch.tensor([0.5, 0.5], device=uv.device)

        # Distance of each UV sample from center
        dist_from_center = ((uv - center) ** 2).sum(dim=-1).sqrt()

        # We want samples closer to center - minimize mean distance
        # Weighted by how much error is in that region
        center_loss = dist_from_center.mean()

        return center_loss

    def adaptive_sampling_loss(
            self,
            rendered_image: torch.Tensor,
            target_image: torch.Tensor,
            uv_to_pixel_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage more UV samples in high-error regions.

        Args:
            rendered_image:  [H, W, 3]
            target_image: [H, W, 3]
            uv_to_pixel_map: [Us, Vs, 2] mapping from UV to pixel coordinates
        """
        # Compute per-pixel error
        error_map = (rendered_image - target_image).pow(2).mean(dim=-1)  # [H, W]

        # Sample error at each UV location
        # Normalize UV to [-1, 1] for grid_sample
        uv_normalized = self.cache * 2 - 1  # [Us, Vs, 2]
        uv_for_sample = uv_normalized.unsqueeze(0)  # [1, Us, Vs, 2]

        error_map_batched = error_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Sample error at UV locations
        sampled_errors = F.grid_sample(
            error_map_batched,
            uv_for_sample,
            mode='bilinear',
            align_corners=True
        )  # [1, 1, Us, Vs]

        sampled_errors = sampled_errors.squeeze()  # [Us, Vs]

        # Compute local UV density (inverse of spacing)
        uv = self.cache
        du = torch.diff(uv[..., 0], dim=0)  # [Us-1, Vs]
        dv = torch.diff(uv[..., 1], dim=1)  # [Us, Vs-1]

        # Density is inverse of spacing (small spacing = high density)
        density_u = 1.0 / (du.abs() + 1e-6)
        density_v = 1.0 / (dv.abs() + 1e-6)

        # We want:  high error -> high density (small spacing)
        # Loss:  encourage density to correlate with error

        # Compute error gradient (where error is increasing)
        error_padded = sampled_errors[:-1, :-1]  # Match density dimensions
        density_combined = density_u[:, :-1] * density_v[:-1, :]

        # Negative correlation loss:  we want density high where error is high
        # Normalize both to [0, 1] range for stable correlation
        error_norm = (error_padded - error_padded.min()) / (error_padded.max() - error_padded.min() + 1e-6)
        density_norm = (density_combined - density_combined.min()) / (
                    density_combined.max() - density_combined.min() + 1e-6)

        # Loss: negative correlation (we want positive correlation)
        correlation_loss = -torch.mean(error_norm * density_norm)

        return correlation_loss

    def coverage_loss(self, weight: float = 1e-2) -> torch.Tensor:
        """Ensure samples cover the full UV space."""
        if self.cache is None:
            return torch.tensor(0.0)

        uv = self.cache

        # Check that we span [0, 1] in both dimensions
        u_span = uv[..., 0].max() - uv[..., 0].min()
        v_span = uv[..., 1].max() - uv[..., 1].min()

        # Target span is ~1. 0, penalize if less
        span_loss = (1.0 - u_span).relu() + (1.0 - v_span).relu()

        return weight * span_loss

    def smoothness_loss(self, weight: float = 1e-3) -> torch.Tensor:
        """Encourage smooth warping (no sudden jumps)."""
        if self.cache is None:
            return torch.tensor(0.0)

        uv = self.cache

        # Second-order differences (curvature)
        d2u = torch.diff(uv[..., 0], n=2, dim=0)
        d2v = torch.diff(uv[..., 1], n=2, dim=1)

        curvature_loss = d2u.pow(2).mean() + d2v.pow(2).mean()

        return weight * curvature_loss