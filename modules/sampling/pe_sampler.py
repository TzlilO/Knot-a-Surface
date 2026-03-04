"""
Positional Encoding based UV Sampler.

Optimizes sampling intervals in positional encoding space for smoother
optimization and multi-scale representation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, Any, List
import math

from model.modules import ModelState, SamplerUV
from model.modules.sampling import insert_knot_1d_to_optimizer
from utils.general_utils import inverse_sigmoid


class PositionalEncodingSampler(nn.Module):
    """
    Sampler that optimizes UV intervals in positional encoding space.

    The key insight:  instead of optimizing raw [0,1] values, we optimize
    coefficients in a Fourier basis, which provides:
    1.Multi-scale control (low freq = global shifts, high freq = local adjustments)
    2.Smoother optimization landscape
    3.Natural frequency-based learning rate scaling

    Storage format (multi-view, non-grid):
        _interval_u: ParameterDict with keys '0', '1', ...
                     Each value is [Us, encoding_dim]
        _interval_v: ParameterDict with keys '0', '1', ...
                     Each value is [Vs, encoding_dim]
    """

    def __init__(
            self,
            state: ModelState,
            initial_samples_u: Optional[torch.Tensor] = None,
            initial_samples_v: Optional[torch.Tensor] = None,
            num_frequencies: int = 6,  # Number of PE frequencies
            include_input: bool = True,  # Include raw value in encoding
            log_sampling: bool = True,  # Log-spaced frequencies vs linear
            late_init: bool = False,  # Log-spaced frequencies vs linear
            **kwargs
    ):
        super().__init__()

        self.state = state
        self.device = state.device
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling

        # Configuration
        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.should_optimize = state.opt.optimize_intervals and not self.evaluate_mode
        self.num_channels = kwargs.get('num_channels', 1)
        self.mode = kwargs.get('mode', 'single' if self.num_channels == 1 else 'multi')
        self.activation = self._decode
        self.inverse_activation = self._encode
        # Compute encoding dimension
        # For each frequency:  sin and cos components
        self.encoding_dim = 2 * num_frequencies
        base_u = None
        base_v = None
        if include_input:
            self.encoding_dim += 1
        if initial_samples_u is not None:
            base_u = initial_samples_u
        else:
            base_u = torch.linspace(0, num_frequencies - 1, self.state.Us, device=self.device)
        if initial_samples_v is not None:
            base_v = initial_samples_v
        else:
            base_v = torch.linspace(0, num_frequencies - 1, self.state.Vs, device=self.device)


        # Create frequency bands
        if log_sampling:
            # Frequencies: 2^0, 2^1, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies, device=self.device)
        else:
            # Linear spacing
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies, device=self.device)

        self.register_buffer('freq_bands', freq_bands * math.pi)

        # Initialize intervals
        Us, Vs = state.Us, state.Vs
        self._init_intervals(Us, Vs, base_u, base_v, kwargs)

        # Visibility tracking
        self.vis_probs = {}
        for i in range(self.num_channels):
            self.vis_probs[i] = torch.zeros((Us, Vs, 2), device=self.device)

        self.uv_viewpoint = {}
        self.active_uid = None

    def _init_intervals(self, Us:  int, Vs: int, base_u, base_v, kwargs:  dict):
        """Initialize interval parameters in PE space."""
        eps = 1e-6

        # Base uniform intervals in raw [0, 1] space
        # base_u = torch.linspace(eps, 1 - eps, Us, device=self.device)
        # base_v = torch.linspace(eps, 1 - eps, Vs, device=self.device)

        # Handle grid vs non-grid mode
        if self.state.uv_grid:
            base_u = base_u.unsqueeze(1).expand(Us, Vs)
            base_v = base_v.unsqueeze(0).expand(Us, Vs)

        if self.should_optimize:
            # Encode to PE space for optimization
            if self.mode == 'single':
                encoded_u = self._encode(base_u)  # [Us, encoding_dim] or [Us, Vs, encoding_dim]
                encoded_v = self._encode(base_v)
                self._interval_u = nn.Parameter(encoded_u.contiguous(), requires_grad=True)
                self._interval_v = nn.Parameter(encoded_v.contiguous(), requires_grad=True)
            else:
                # Multi-view mode:  ParameterDict with string keys
                self._interval_u = nn.ParameterDict()
                self._interval_v = nn.ParameterDict()
                for i in range(self.num_channels):
                    encoded_u = self._encode(base_u.clone())
                    encoded_v = self._encode(base_v.clone())
                    self._interval_u[str(i)] = nn.Parameter(
                        encoded_u.clone().contiguous(), requires_grad=True
                    )
                    self._interval_v[str(i)] = nn.Parameter(
                        encoded_v.clone().contiguous(), requires_grad=True
                    )
        else:
            # Non-optimizable:  store raw values
            if self.mode == 'single':
                self._interval_u = base_u.contiguous()
                self._interval_v = base_v.contiguous()
            else:
                self._interval_u = {str(i): base_u.clone() for i in range(self.num_channels)}
                self._interval_v = {str(i): base_v.clone() for i in range(self.num_channels)}

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode values to positional encoding space.

        Args:
            x: Input tensor with values in [0, 1], shape [...,]

        Returns:
            Encoded tensor, shape [..., encoding_dim]
        """
        # Expand x for broadcasting with frequencies

        x_expanded = x.unsqueeze(-1) if x.ndim == 1 else x.unsqueeze(-1)



        # Compute sin and cos for each frequency
        angles = x_expanded * self.freq_bands.to(x.device)

        # Interleave sin and cos:  [sin(f1*x), cos(f1*x), sin(f2*x), cos(f2*x), ...]
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.include_input:
            # Prepend raw value (useful for preserving DC component)
            encoded = torch.cat([x_expanded, encoded], dim=-1)

        return encoded

    def _decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode from positional encoding space back to [0, 1].

        Args:
            encoded:  Encoded tensor, shape [..., encoding_dim]

        Returns:
            Decoded values in [0, 1], shape [...]
        """
        if self.include_input:
            # The first component is the raw value (DC)
            decoded = encoded[..., 0]
        else:
            # Use the first sin/cos pair to reconstruct
            num_freq = self.num_frequencies
            sin_components = encoded[..., : num_freq]
            cos_components = encoded[..., num_freq:]
            decoded = (torch.atan2(sin_components[..., 0], cos_components[..., 0]) / math.pi + 1) / 2

        return decoded.clamp(1e-6, 1 - 1e-6)

    @property
    def interval_u(self) -> torch.Tensor:
        """Get U intervals (decoded from PE space if optimizing)."""
        if self.should_optimize:
            if self.mode == 'single':
                return self._decode(self._interval_u)
            else:
                uid = str(self.active_uid) if self.active_uid is not None else '0'
                if uid not in self._interval_u:
                    return self.get_uniform_grid().unbind(-1)[0].to(self.device)
                return self._decode(self._interval_u[uid])
        else:
            if self.mode == 'single':
                return self._interval_u
            else:
                uid = str(self.active_uid) if self.active_uid is not None else '0'
                if uid not in self._interval_u:
                    return self.get_uniform_grid().unbind(-1)[0].to(self.device)
                return self._interval_u[uid]

    @property
    def interval_v(self) -> torch.Tensor:
        """Get V intervals (decoded from PE space if optimizing)."""
        if self.should_optimize:
            if self.mode == 'single':
                return self._decode(self._interval_v)
            else:
                uid = str(self.active_uid) if self.active_uid is not None else '0'
                if uid not in self._interval_v:
                    return self.get_uniform_grid().unbind(-1)[1].to(self.device)
                return self._decode(self._interval_v[uid])
        else:
            if self.mode == 'single':
                return self._interval_v
            else:
                uid = str(self.active_uid) if self.active_uid is not None else '0'
                if uid not in self._interval_v:
                    return self.get_uniform_grid().unbind(-1)[1].to(self.device)
                return self._interval_v[uid]

    def update_interval_v(self, new_intervals_data:[List[Tuple[torch.Tensor, int]]], optimizer=None):
        for uid in self._interval_v.keys():
            i = int(uid)
            new_interval, inset_idx = new_intervals_data[i]
            opt_tensors = insert_knot_1d_to_optimizer((new_interval), f'view_{uid}_interval_v_surf_{self.state.surf_uid}', insert_idx=inset_idx, optimizer=optimizer)
            self._interval_v[uid] = opt_tensors

    def update_interval_u(self, new_intervals_data:[List[Tuple[torch.Tensor, int]]], optimizer=None):
        for uid in self._interval_u.keys():
            i = int(uid)
            new_interval, inset_idx = new_intervals_data[i]
            opt_tensors = insert_knot_1d_to_optimizer((new_interval), f'view_{uid}_interval_u_surf_{self.state.surf_uid}', insert_idx=inset_idx,
                                                   optimizer=optimizer)
            self._interval_u[uid] = opt_tensors

    def forward(self) -> torch.Tensor:
        """
        Get UV grid for current view.

        Returns:
            UV coordinates [Us, Vs, 2] or (u, v) tuple depending on state.uv_grid
        """
        u = self.interval_u
        v = self.interval_v

        if self.should_optimize:
            # Sort to maintain monotonicity
            if self.state.uv_grid:
                u = u.sort(dim=0)[0]
                v = v.sort(dim=1)[0]
            else:
                u = u.sort(dim=0)[0]
                v = v.sort(dim=0)[0]

        if self.state.uv_grid:
            return torch.stack([u, v], dim=-1)
        else:
            return (u, v)

    def get_uniform_samplings(self, UVshape=None):
        if self.state.uv_grid:
            return self.get_uniform_grid(UVshape=UVshape)
        else:
            return self.get_uniform_intervals(UVshape=UVshape)

    def get_uniform_intervals(self, UVshape=None):
        Us, Vs = UVshape if UVshape is not None else (self.state.Us, self.state.Vs)
        sampled_u = torch.linspace(0.0, 1.0, Us, device=self.device).detach().squeeze()
        sampled_v = torch.linspace(0.0, 1.0, Vs, device=self.device).detach().squeeze()
        return sampled_u, sampled_v

    def get_uniform_grid(self, UVshape:  Tuple[int, int] = None) -> torch.Tensor:
        """Get uniform UV grid."""
        Us, Vs = UVshape if UVshape is not None else (self.state.Us, self.state.Vs)
        eps = 1e-6

        u = torch.linspace(eps, 1 - eps, Us, device=self.device)
        v = torch.linspace(eps, 1 - eps, Vs, device=self.device)

        if self.state.uv_grid:
            u = u.unsqueeze(1).expand(Us, Vs)
            v = v.unsqueeze(0).expand(Us, Vs)
            return torch.stack([u, v], dim=-1)
        else:
            return torch.stack([u, v], dim=-1)

    def get_randomized_grid(self, UVshape=None):
        Us, Vs = UVshape if UVshape is not None else (self.state.Us, self.state.Vs)
        sampled_u = torch.rand(Us, Vs, device=self.device)
        sampled_v = torch.rand(Us, Vs, device=self.device)
        return torch.stack([sampled_u, sampled_v], dim=-1)

    def evalpts_u(self):
        return self.interval_u.view(self.state.Us, self.state.Vs) if self.state.uv_grid else self.interval_u.view(-1)

    def evalpts_v(self):
        return self.interval_v.view(self.state.Us, self.state.Vs) if self.state.uv_grid else self.interval_v.view(-1)

    # =========================================================================
    # Diff methods for regularization
    # =========================================================================

    def diff_us(self, uid=None) -> torch.Tensor:
        """Compute U differences for regularization."""
        u = self.export_intervals_u(activate=True)

        if self.state.uv_grid:
            if self.mode == 'multi':
                prepend = torch.zeros_like(u[: , : 1, :])
                return torch.diff(u, dim=1, prepend=prepend)
            else:
                prepend = torch.zeros_like(u[:1, :])
                return torch.diff(u, dim=0, prepend=prepend)
        else:
            if self.mode == 'multi':
                # u shape: [num_channels, Us]
                prepend = torch.zeros_like(u[:, :1])
                return torch.diff(u, dim=1, prepend=prepend).clamp(1e-8)
            else:
                prepend = torch.tensor([0.], device=self.device)
                return torch.diff(u.squeeze(), dim=0, prepend=prepend).clamp(1e-8)

    def diff_vs(self, uid=None) -> torch.Tensor:
        """Compute V differences for regularization."""
        v = self.export_intervals_v(activate=True)

        if self.state.uv_grid:
            if self.mode == 'multi':
                prepend = torch.zeros_like(v[:, :, :1])
                return torch.diff(v, dim=2, prepend=prepend)
            else:
                prepend = torch.zeros_like(v[: , :1])
                return torch.diff(v, dim=1, prepend=prepend)
        else:
            if self.mode == 'multi':
                # v shape: [num_channels, Vs]
                prepend = torch.zeros_like(v[:, :1])
                return torch.diff(v, dim=1, prepend=prepend).clamp(1e-8)
            else:
                prepend = torch.tensor([0.], device=self.device)
                return torch.diff(v.squeeze(), dim=0, prepend=prepend).clamp(1e-8)

    def diff_u(self, uid=None):
        """Alias for diff_us for compatibility."""
        return self.diff_us(uid)

    def diff_v(self, uid=None):
        """Alias for diff_vs for compatibility."""
        return self.diff_vs(uid)

    # =========================================================================
    # Export methods
    # =========================================================================

    def export_intervals_u(self, activate: bool = True) -> torch.Tensor:
        """Export U intervals, optionally decoded."""
        if self.mode == 'single':
            if self.should_optimize:
                if activate:
                    return self._decode(self._interval_u)
                return self._interval_u
            return self._interval_u
        else:
            if self.should_optimize:
                tensors = [self._interval_u[str(i)] for i in range(self.num_channels)]
                stacked = torch.stack(tensors, dim=0)  # [num_channels, Us, encoding_dim]
                if activate:
                    # Decode each channel
                    decoded = torch.stack([self._decode(t) for t in tensors], dim=0)
                    return decoded  # [num_channels, Us]
                return stacked
            return torch.stack([self._interval_u[str(i)] for i in range(self.num_channels)], dim=0)

    def export_intervals_v(self, activate: bool = True) -> torch.Tensor:
        """Export V intervals, optionally decoded."""
        if self.mode == 'single':
            if self.should_optimize:
                if activate:
                    return self._decode(self._interval_v)
                return self._interval_v
            return self._interval_v
        else:
            if self.should_optimize:
                tensors = [self._interval_v[str(i)] for i in range(self.num_channels)]
                stacked = torch.stack(tensors, dim=0)  # [num_channels, Vs, encoding_dim]
                if activate:
                    decoded = torch.stack([self._decode(t) for t in tensors], dim=0)
                    return decoded  # [num_channels, Vs]
                return stacked
            return torch.stack([self._interval_v[str(i)] for i in range(self.num_channels)], dim=0)

    # =========================================================================
    # Subdivision methods
    # =========================================================================

    def subdivide_feature_with_density(
            self,
            direction: str,
            knots: torch.Tensor,
            degree: int,
            val: float,
            insert_idx: float,
            num_insertions: int,
            insertion_fn,
            optimizer=None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Subdivide sampling intervals after knot insertion.

        For PE sampler, we need to:
        1.Decode intervals to raw [0, 1] space
        2.Insert new samples
        3.Re-encode back to PE space

        Args:
            direction: 'u' or 'v'
            knots: Current knot vector
            degree: Spline degree
            val: Parameter value where subdivision occurs
            insertion_fn: Function to compute interpolated values
            insert_idx: Index where to insert

        Returns:
            new_intervals_u, new_intervals_v:  Lists of updated interval tensors per view
        """
        new_intervals_u = []
        new_intervals_v = []

        # Compute density (number of samples to insert)
        density = int(self.state.sampling_density if direction == 'u' else self.state.sampling_density)
        insert_idx_knot = torch.searchsorted(knots, val, side='right').item() - 1

        for uid in range(self.num_channels):
            uid_str = str(uid)

            if direction == 'u':
                # Get current encoded intervals
                if self.should_optimize:
                    encoded_u = self._interval_u[uid_str]
                    # Decode to raw space
                    samples_u = self._decode(encoded_u)
                else:
                    samples_u = self._interval_u[uid_str]
                # Search in ACTIVATED space, not logit space!
                insert_idx = torch.searchsorted(
                    samples_u.view(-1),
                    knots[insert_idx_knot],# val is already in [0,1]
                    side='right'
                ).item()
                # Insert in decoded space

                new_u_raw, sample_insert_idx_ = self._insert_knot_1d_activated(
                    samples_u,
                    knots,
                    degree,
                    val,
                    insert_idx
                )
                # Sort for monotonicity
                new_u_raw = torch.sort(new_u_raw.squeeze(), dim=0)[0]

                # Re-encode to PE space
                if self.should_optimize:
                    new_u_encoded = self._encode(new_u_raw)
                    new_intervals_u.append((new_u_encoded, insert_idx))
                else:
                    new_intervals_u.append((new_u_raw, insert_idx))

            else:  # direction == 'v'
                if self.should_optimize:
                    encoded_v = self._interval_v[uid_str]
                    samples_v = self._decode(encoded_v)
                else:
                    samples_v = self._interval_v[uid_str]
                insert_idx = torch.searchsorted(
                    samples_v.view(-1),
                    knots[insert_idx_knot],  # val is already in [0,1]
                    side='right'
                ).item()

                new_v_raw, sample_insert_idx = self._insert_knot_1d_activated(
                    samples_v,
                    knots,
                    degree,
                    val,
                    insert_idx
                )

                new_v_raw = torch.sort(new_v_raw.squeeze(), dim=0)[0]

                if self.should_optimize:
                    new_v_encoded = self._encode(new_v_raw)
                    new_intervals_v.append((new_v_encoded, insert_idx))
                else:
                    new_intervals_v.append((new_v_raw, insert_idx))
        # Update optimizer state
        if direction == 'u':
            self.update_interval_u(new_intervals_u, optimizer=optimizer)
        else:
            self.update_interval_v(new_intervals_v, optimizer=optimizer)

    def subdivide_feature_with_density2(
            self,
            direction: str,
            knots: torch.Tensor,
            degree: int,
            val: float,
            insert_idx: float,
            num_insertions: int,
            insertion_fn,
            optimizer=None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Subdivide sampling intervals maintaining density invariant.

        CRITICAL: All operations must respect the activation space:
        - Storage is in LOGIT space (unbounded)
        - Geometric operations happen in ACTIVATED space [0,1]
        - New values must be transformed to logit space before storage
        """
        new_intervals_subd_data = []

        for uid in range(self.num_channels):
            if direction == 'u':
                interval_raw = self._interval_u[str(uid)]  # LOGIT space
            else:
                interval_raw = self._interval_v[str(uid)]  # LOGIT space

            # === CRITICAL FIX 1: Work in ACTIVATED space for geometric operations ===
            interval_activated = self._decode(interval_raw)
            # interval_activated = (interval_raw)  # [0, 1] space

            # Find insertion index using ACTIVATED values
            insert_idx_knot = torch.searchsorted(knots, val, side='right').item() - 1

            # Search in ACTIVATED space, not logit space!
            insert_idx = torch.searchsorted(
                interval_activated.view(-1),
                torch.tensor(val, device=interval_activated.device),  # val is already in [0,1]
                side='right'
            ).item()
            insert_idx = max(0, min(insert_idx, interval_activated.numel()))
            density = int(self.state.sampling_density if direction == 'u' else self.state.sampling_density)

            # === CRITICAL FIX 2: Insert in ACTIVATED space, then convert back ===
            new_interval, _ = self._insert_knot_1d_activated(
                interval_raw,
                knots,
                degree,
                val,
                insert_idx
            )
            # new_interval, sample_insert_idx = self._insert_samples_1d(
            #     interval_raw, knots, degree, val, density, insert_idx=insert_idx
            # )

            # === CRITICAL FIX 3: Sort in ACTIVATED space to ensure monotonicity ===
            # new_interval_ord = torch.sort(new_interval_activated.view(-1))[1]

            # === CRITICAL FIX 4: Clamp BEFORE inverse to avoid extreme logit values ===
            eps = 0 #1e-12  # Safer margin than 1e-6
            # new_interval_clamped = new_interval_activated_sorted.clamp(eps, 1.0 - eps)

            # === CRITICAL FIX 4: Clamp BEFORE inverse to avoid extreme logit values ===
            eps = 0 #1e-12  # Safer margin than 1e-6
            new_interval_clamped = new_interval#[new_interval_ord].clamp(eps, 1.0 - eps)

            # === CRITICAL FIX 5: Convert back to LOGIT space for storage ===
            new_interval_logit = (new_interval_clamped)
            # new_interval_logit = self._encode(new_interval_clamped.squeeze())
            #
            # # Check for NaNs/Infs before proceeding
            # if torch.isnan(new_interval_logit).any() or torch.isinf(new_interval_logit).any():
            #     print(f"[WARNING] NaN/Inf detected in interval after subdivision!")
            #     print(
            #         f"  Activated range: [{new_interval_activated_sorted.min():.6f}, {new_interval_activated_sorted.max():.6f}]")
            #     print(f"  Clamped range: [{new_interval_clamped.min():.6f}, {new_interval_clamped.max():.6f}]")
            #     # Fallback:  use uniform spacing
            #     new_interval_logit = self.inverse_activation(
            #         torch.linspace(eps, 1 - eps, new_interval_clamped.numel(), device=interval_raw.device)
            #     )
            #
            new_intervals_subd_data.append((new_interval_logit, insert_idx))

        # Update optimizer state
        if direction == 'u':
            self.update_interval_u(new_intervals_subd_data, optimizer=optimizer)
        else:
            self.update_interval_v(new_intervals_subd_data, optimizer=optimizer)

    def _insert_knot_1d_activated(
            self,
            intervals_activated: torch.Tensor,  # Already in [0,1] space
            knots: torch.Tensor,
            degree: int,
            u_bar: float,
            insert_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Insert a value into an ACTIVATED (sigmoid-space) interval tensor.
        All interpolation happens in [0,1] space for numerical stability.
        """
        device = intervals_activated.device
        N = intervals_activated.shape[0]
        if  intervals_activated.ndim == 2:
            C = intervals_activated.shape[1]
        else:
            C = 1
        N = intervals_activated.shape[0]
        intervals = intervals_activated.view(-1, C).squeeze()

        # Clamp insert_idx to valid range
        insert_idx = max(0, min(insert_idx, N))

        # Create output tensor
        new_intervals = torch.zeros(N + 1, device=device, dtype=intervals.dtype)

        # Copy prefix
        if insert_idx > 0:
            new_intervals[: insert_idx] = intervals[:insert_idx]

        # Copy suffix
        if insert_idx < N:
            new_intervals[insert_idx + 1:] = intervals[insert_idx:]

        # Insert new value (u_bar is already in [0,1] space)
        # Optionally interpolate for smoothness
        if insert_idx == 0:
            new_intervals[0] = u_bar
        elif insert_idx == N:
            new_intervals[N] = u_bar
        else:
            # Use u_bar directly (it's the knot value in parameter space)
            # Or interpolate between neighbors for smoother result
            left_val = intervals[insert_idx - 1]
            right_val = intervals[insert_idx]

            # Option A: Direct insertion
            new_intervals[insert_idx] = u_bar

            # Option B:  Interpolated insertion (smoother)
            # alpha = (u_bar - left_val) / (right_val - left_val + 1e-8)
            # alpha = alpha.clamp(0, 1)
            # new_intervals[insert_idx] = (1 - alpha) * left_val + alpha * right_val

        # Update knot vector (for compatibility)
        new_knots = torch.cat([knots[: insert_idx + degree + 1],
                               torch.tensor([u_bar], device=device),
                               knots[insert_idx + degree + 1:]])

        return new_intervals, new_knots

    def capture_state(self) -> dict:
        """Capture sampler state."""
        state = {
            'mode': self.mode,
            'num_channels': self.num_channels,
            'should_optimize': self.should_optimize,
            'random': getattr(self, 'random', False),
        }

        # Capture intervals based on mode
        if self.mode == 'single':
            if self.should_optimize:
                state['interval_u'] = self._interval_u.data.clone().cpu()
                state['interval_v'] = self._interval_v.data.clone().cpu()
            else:
                state['interval_u'] = self._interval_u.clone().cpu()
                state['interval_v'] = self._interval_v.clone().cpu()

        elif self.mode == 'multi':
            # Multi-view:  store as list
            state['interval_u'] = {}
            state['interval_u_raw'] = {}
            state['interval_v'] = {}
            state['interval_v_raw'] = {}

            for uid in self._interval_u.keys():
                if self.should_optimize:
                    state['interval_u_raw'][uid] = self._interval_u[uid].data.clone().cpu()#.sigmoid()
                    state['interval_v_raw'][uid] = self._interval_v[uid].data.clone().cpu()#
                    state['interval_u'][uid] = self._interval_u[uid].data.clone().cpu().sigmoid()
                    state['interval_v'][uid] = self._interval_v[uid].data.clone().cpu().sigmoid()
                else:
                    state['interval_u'][uid] = self._interval_u[uid].clone().cpu()
                    state['interval_u_raw'][uid] = self._interval_u[uid].clone().cpu()
                    state['interval_v'][uid] = self._interval_v[uid].clone().cpu()
                    state['interval_v_raw'][uid] = self._interval_v[uid].clone().cpu()

        state['activation'] = 'sigmoid' if self.should_optimize else 'identity'
        state['inverse_activation'] = 'logit' if self.should_optimize else 'identity'
        # Visibility probabilities
        if hasattr(self, 'vis_probs') and self.vis_probs is not None:
            if isinstance(self.vis_probs, dict):
                state['vis_probs'] = {
                    k: v.clone().cpu() for k, v in self.vis_probs.items()
                }
            else:
                state['vis_probs'] = self.vis_probs.clone().cpu()


        return state

    @classmethod
    def from_state(
            cls,
            state: dict,
            model_state: 'ModelState',
            device: str = 'cuda',
            evaluate_mode: bool = False
    ) -> 'SamplerUV':
        """Restore SamplerUV from captured state."""

        mode = state['mode']
        num_channels = state['num_channels']
        should_optimize = state['should_optimize'] #and not evaluate_mode

        # Create instance with late_init to avoid default initialization
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        ACTIVATIONS = {'sigmoid': torch.sigmoid, 'logit': inverse_sigmoid, 'inverse_sigmoid': inverse_sigmoid,
                       'identity': lambda x: x}
        # Basic attributes
        instance.state = model_state
        instance.device = device
        instance.mode = mode
        instance.num_channels = num_channels
        instance.should_optimize = should_optimize and not evaluate_mode
        instance.random = state.get('random', False)
        instance.evaluate_mode = evaluate_mode
        instance.activation = ACTIVATIONS[state['activation']] if should_optimize else ACTIVATIONS['identity']
        instance.inverse_activation = ACTIVATIONS[state['inverse_activation']] if should_optimize else ACTIVATIONS['identity']
        if mode == 'single':
            u_data = state['interval_u'].to(device)
            v_data = state['interval_v'].to(device)

            if should_optimize:
                instance._interval_u = nn.Parameter(u_data, requires_grad=True)
                instance._interval_v = nn.Parameter(v_data, requires_grad=True)
            else:
                instance._interval_u = u_data
                instance._interval_v = v_data

        elif mode == 'multi':
            instance._interval_u = nn.ParameterDict()
            instance._interval_v = nn.ParameterDict()
            state_u = state['interval_u_raw'] if should_optimize else state['interval_u']
            state_v = state['interval_v_raw'] if should_optimize else state['interval_v']
            for k, t in state_u.items():
                if should_optimize:
                    instance._interval_u[k] = nn.Parameter(t.to(device), requires_grad=True)
                else:
                    instance._interval_u[k] = t.to(device)
            for k, t in state_v.items():
                if should_optimize:
                    instance._interval_v[k] = nn.Parameter(t.to(device), requires_grad=True)
                else:
                    instance._interval_v[k] = t.to(device)
            #
            # else:
            #     instance._interval_u = {}
            #     instance._interval_v = {}
            #     for i in range(num_channels):
            #         instance._interval_u[str(i)] = state['interval_u'][''].to(device)
            #         instance._interval_v[str(i)] = state['interval_v'][''].to(device)

        # Restore visibility probs
        if 'vis_probs' in state:
            if isinstance(state['vis_probs'], dict):
                instance.vis_probs = {
                    k: v.to(device) for k, v in state['vis_probs'].items()
                }
            else:
                instance.vis_probs = state['vis_probs'].to(device)
        else:
            # Initialize empty
            Us, Vs = model_state.Us, model_state.Vs
            instance.vis_probs = torch.zeros(
                (num_channels, Us, Vs, 2), device=device
            )

        # Restore UV viewpoint cache
        if 'uv_viewpoint' in state:
            instance.uv_viewpoint = {
                k: v.to(device) for k, v in state['uv_viewpoint'].items()
            }
        else:
            instance.uv_viewpoint = {}

        instance.active_uid = None

        return instance


    def _insert_knot_1d_activated(
            self,
            intervals: torch.Tensor,  # In ACTIVATED space [0, 1]
            knots: torch.Tensor,
            degree: int,
            u_bar: float,
            insert_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1D knot insertion for interval parameters.

        IMPORTANT: This operates in ACTIVATED (probability) space.
        """
        device = intervals.device
        # intervals = intervals.view(-1)
        # N = intervals.shape[0]
        device = intervals.device
        N = intervals.shape[0]
        if  intervals.ndim == 2:
            C = intervals.shape[1]
        else:
            C = 1
        N = intervals.shape[0]
        intervals = intervals.view(-1, C).squeeze()
        if not isinstance(u_bar, torch.Tensor):
            u_bar = torch.tensor([u_bar], device=device, dtype=intervals.dtype)
        else:
            u_bar = u_bar.view(1)

        # Clamp insert_idx
        insert_idx = max(0, min(insert_idx, N))

        # Create new interval tensor
        new_intervals = torch.zeros(N + 1, C, device=device, dtype=intervals.dtype).squeeze()

        # Prefix copy
        if insert_idx > 0:
            new_intervals[: insert_idx] = intervals[:insert_idx]

        # Suffix copy
        if insert_idx < N:
            new_intervals[insert_idx + 1:] = intervals[insert_idx:]

        # Insert the new value (u_bar is already in [0,1] activated space)
        new_intervals[insert_idx] = u_bar.squeeze()

        # Update knot vector (if needed)
        k = torch.searchsorted(knots, u_bar, side='right').item() - 1
        new_knots = torch.cat([knots[:k + 1], u_bar, knots[k + 1:]])

        return new_intervals, new_knots


    def subdivide_feature_with_density2(
            self,
            direction: str,
            knots:  torch.Tensor,
            degree: int,
            val: float,
            insert_idx: Optional[int],
            num_insertions: int,
            insertion_fn
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Subdivide sampling intervals maintaining density invariant.

        Args:
            direction: 'u' or 'v'
            knots: Current knot vector
            degree: Spline degree
            val: Parameter value where subdivision occurs
            num_insertions: Number of new samples to insert
            insertion_fn: Function to compute interpolated values

        Returns:
            new_intervals_u, new_intervals_v, sample_insert_idx
        """
        new_intervals_u = []
        new_intervals_v = []
        sample_insert_idx = 0

        for uid in range(self.num_channels):
            uid_str = str(uid)

            if direction == 'u':
                if self.should_optimize:
                    encoded_u = self._interval_u[uid_str]
                    samples_u = self._decode(encoded_u)
                else:
                    samples_u = self._interval_u[uid_str]

                new_u_raw, sample_insert_idx = self._insert_samples_1d(
                    samples_u, knots, degree, val, num_insertions
                )
                new_u_raw = torch.sort(new_u_raw.squeeze(), dim=0)[0]

                if self.should_optimize:
                    new_intervals_u.append(self._encode(new_u_raw))
                else:
                    new_intervals_u.append(new_u_raw)

            else:
                if self.should_optimize:
                    encoded_v = self._interval_v[uid_str]
                    samples_v = self._decode(encoded_v)
                else:
                    samples_v = self._interval_v[uid_str]

                new_v_raw, sample_insert_idx = self._insert_samples_1d(
                    samples_v, knots, degree, val, num_insertions
                )
                new_v_raw = torch.sort(new_v_raw.squeeze(), dim=0)[0]

                if self.should_optimize:
                    new_intervals_v.append(self._encode(new_v_raw))
                else:
                    new_intervals_v.append(new_v_raw)

        if direction == 'u':
            return (new_intervals_u, None, sample_insert_idx)
        else:
            return (new_intervals_v, None, sample_insert_idx)

    def _insert_samples_1d(
            self,
            samples: torch.Tensor,
            knots: torch.Tensor,
            degree: int,
            val:  float,
            num_insertions: int
    , insert_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Insert samples into a 1D interval array.

        Args:
            samples: Current samples [N] in decoded (raw [0,1]) space
            knots: Knot vector for mapping
            degree: Spline degree
            val: Parameter value where to insert
            num_insertions:  Number of samples to insert

        Returns:
            new_samples: [N + num_insertions] tensor
            insert_idx: Index where insertion started
        """
        if num_insertions <= 0:
            return samples, 0

        N = samples.shape[0]
        device = samples.device

        # Map val to sample space
        # insert_idx = self._map_param_to_sample_idx(val, knots, N)

        device = 'cuda'
        N = samples.shape[0]

        # Boundary handling
        if insert_idx == 0:
            new_vals = samples[0:1]#.unsqueeze(1).repeat(num_insertions)
        elif insert_idx >= N:
            new_vals = samples[-1:]#.unsqueeze(1).repeat(num_insertions)
        else:
            # Interpolate between neighbors
            left = samples[insert_idx - 1]
            right = samples[insert_idx]
            alphas = torch.linspace(0, 1, num_insertions + 2, device=device)[1:-1]#.unsqueeze(1).repeat(1, samples.shape[-1])

            new_vals = (1 - alphas) * left + alphas * right
        # Concatenate
        new_samples = torch.cat([
            samples[:insert_idx],
            new_vals,
            samples[insert_idx:]
        ], dim=0)

        return new_samples, insert_idx

    def _map_param_to_sample_idx(self, val: float, knots: torch.Tensor, sample_size: int) -> int:
        """Map parameter value to sample index."""
        knot_min, knot_max = knots[0].item(), knots[-1].item()
        normalized = (val - knot_min) / (knot_max - knot_min + 1e-8)
        sample_idx = int(normalized * sample_size)
        return max(0, min(sample_idx, sample_size))

    # =========================================================================
    # Update methods for optimizer integration
    # =========================================================================
    #
    # def update_interval_u(self, new_u):
    #     """Update U interval (for non-optimizable mode)."""
    #     if self.mode == 'single':
    #         self._interval_u = new_u
    #     else:
    #         # Assumes new_u is for current active_uid
    #         uid = str(self.active_uid) if self.active_uid is not None else '0'
    #         self._interval_u[uid] = new_u
    #
    # def update_interval_v(self, new_v):
    #     """Update V interval (for non-optimizable mode)."""
    #     if self.mode == 'single':
    #         self._interval_v = new_v
    #     else:
    #         uid = str(self.active_uid) if self.active_uid is not None else '0'
    #         self._interval_v[uid] = new_v

    def set_samples(self, **kwargs):
        """Set samples from kwargs (for compatibility)."""
        samplings_u = kwargs.get('samplings_u', None)
        samplings_v = kwargs.get('samplings_v', None)

        if samplings_u is not None:
            for i in range(min(len(samplings_u), self.num_channels)):
                raw_u = samplings_u[i].clamp(1e-6, 1 - 1e-6)
                if self.should_optimize:
                    self._interval_u[str(i)] = nn.Parameter(
                        self._encode(raw_u).contiguous(), requires_grad=True
                    )
                else:
                    self._interval_u[str(i)] = raw_u.contiguous()

        if samplings_v is not None:
            for i in range(min(len(samplings_v), self.num_channels)):
                raw_v = samplings_v[i].clamp(1e-6, 1 - 1e-6)
                if self.should_optimize:
                    self._interval_v[str(i)] = nn.Parameter(
                        self._encode(raw_v).contiguous(), requires_grad=True
                    )
                else:
                    self._interval_v[str(i)] = raw_v.contiguous()

    # =========================================================================
    # Regularization
    # =========================================================================

    def get_pe_regularization_loss(self, weight:  float = 0.01) -> torch.Tensor:
        """
        Regularization loss on PE coefficients.

        Encourages:
        1.Smooth variations (penalize high-frequency components)
        2.Monotonicity preservation
        """
        if not self.should_optimize:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        if self.mode == 'single':
            pe_u = self._interval_u
            pe_v = self._interval_v
        else:
            pe_u = torch.stack([self._interval_u[str(i)] for i in range(self.num_channels)])
            pe_v = torch.stack([self._interval_v[str(i)] for i in range(self.num_channels)])

        # Penalize high-frequency components more heavily
        start_idx = 1 if self.include_input else 0

        for i in range(self.num_frequencies):
            freq_weight = (i + 1) ** 2

            sin_idx = start_idx + i
            cos_idx = start_idx + self.num_frequencies + i

            if sin_idx < pe_u.shape[-1] and cos_idx < pe_u.shape[-1]:
                loss += freq_weight * (pe_u[..., sin_idx].pow(2).mean() + pe_u[..., cos_idx].pow(2).mean())
                loss += freq_weight * (pe_v[..., sin_idx].pow(2).mean() + pe_v[..., cos_idx].pow(2).mean())

        return weight * loss

    def compute_losses(self, prob_map: torch.Tensor, w_a=0.000001, w_b=0.0, eps=1e-8) -> torch.Tensor:
        """Compute monotonicity and density-matching losses."""
        if w_a == 0.0 and w_b == 0.0:
            return torch.tensor(0.0, device=prob_map.device)

        diff_u = self.diff_us() + eps
        diff_v = self.diff_vs() + eps

        # Monotonicity loss
        mono_loss_u = F.relu(-diff_u).mean()
        mono_loss_v = F.relu(-diff_v).mean()
        mono_loss = (mono_loss_u + mono_loss_v) * 1e3

        return mono_loss * w_a

    # =========================================================================
    # Serialization
    # =========================================================================

    def capture(self) -> Dict[str, Any]:
        """Capture state for serialization."""
        data = {
            'mode': self.mode,
            'num_channels': self.num_channels,
            'num_frequencies': self.num_frequencies,
            'include_input':  self.include_input,
            'log_sampling':  self.log_sampling,
            'should_optimize': self.should_optimize,
            'encoding_dim': self.encoding_dim,
        }

        # Save vis_probs
        vis_probs_cpu = {}
        for k, v in self.vis_probs.items():
            vis_probs_cpu[k] = v.detach().cpu()
        data['vis_probs'] = vis_probs_cpu

        if self.should_optimize:
            if self.mode == 'single':
                data['pe_u'] = self._interval_u.detach().cpu()
                data['pe_v'] = self._interval_v.detach().cpu()
            else:
                data['pe_u'] = {k: v.detach().cpu() for k, v in self._interval_u.items()}
                data['pe_v'] = {k: v.detach().cpu() for k, v in self._interval_v.items()}
        else:
            if self.mode == 'single':
                data['u'] = self._interval_u.detach().cpu() if isinstance(self._interval_u, torch.Tensor) else self._interval_u
                data['v'] = self._interval_v.detach().cpu() if isinstance(self._interval_v, torch.Tensor) else self._interval_v
            else:
                data['u'] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in self._interval_u.items()}
                data['v'] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in self._interval_v.items()}

        return data

    def restore(self, data: Dict[str, Any]):
        """Restore state from serialization."""
        self.mode = data['mode']
        self.num_channels = data['num_channels']
        self.num_frequencies = data['num_frequencies']
        self.include_input = data['include_input']
        self.log_sampling = data['log_sampling']
        self.encoding_dim = data.get('encoding_dim', 2 * self.num_frequencies + (1 if self.include_input else 0))

        device = self.state.device

        if data['should_optimize']:
            self.should_optimize = True
            if self.mode == 'single':
                self._interval_u = nn.Parameter(data['pe_u'].to(device), requires_grad=True)
                self._interval_v = nn.Parameter(data['pe_v'].to(device), requires_grad=True)
            else:
                self._interval_u = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in data['pe_u'].items()
                })
                self._interval_v = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in data['pe_v'].items()
                })
        else:
            self.should_optimize = False
            if self.mode == 'single':
                self._interval_u = data['u'].to(device) if isinstance(data['u'], torch.Tensor) else data['u']
                self._interval_v = data['v'].to(device) if isinstance(data['v'], torch.Tensor) else data['v']
            else:
                self._interval_u = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data['u'].items()}
                self._interval_v = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data['v'].items()}

        if 'vis_probs' in data:
            for k, v in data['vis_probs'].items():
                self.vis_probs[k] = v.to(device)