"""
UV Sampler with optimizable intervals.
Supports single-view and multi-view modes.
"""

"""
UV Sampler with optimizable intervals.
Supports single-view and multi-view modes.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from modules import ModelState
from modules.optim_utils import (
    splice_1d_optimizer_state,
    replace_param_in_optimizer,
    MomentumStrategy,
)
from utils.general_utils import inverse_sigmoid


class SamplerUV(nn.Module):
    """
    Module to handle sampling intervals (u/v samples).

    Modes:
        - 'single': One set of intervals for all views
        - 'multi': Per-view intervals (ParameterDict)
    """

    def __init__(
        self,
        state: ModelState,
        late_init: bool = False,
        mode: str = 'single',
        **kwargs,
    ):
        super().__init__()
        self.state = state
        self.device = state.device
        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.should_optimize = state.opt.optimize_intervals
        self.num_channels = kwargs.get('num_channels', 1)
        self.mode = mode
        self._uv_grid = None

        if not late_init:

            self._init_intervals(base_u=kwargs.get('base_u', None), base_v=kwargs.get('base_v', None))


    # =====================================================================
    # Initialization
    # =====================================================================

    # def _init_intervals(self):
    #     """Initialize interval parameters."""
    #     Us, Vs = self.state.H, self.state.W
    #     base_u = torch.linspace(0, 1, Us + 2, device=self.device)[1:-1]
    #     base_v = torch.linspace(0, 1, Vs + 2, device=self.device)[1:-1]
    def _init_intervals(self, **kwargs):
        """Initialize interval parameters."""
        Us, Vs = self.state.H, self.state.W
        if kwargs.get('base_u', None) is not None and kwargs.get('base_v', None) is not None:
            base_u = torch.tensor(kwargs['base_u'], device=self.device)
            base_v = torch.tensor(kwargs['base_v'], device=self.device)
        else:
            base_u = torch.linspace(0, 1, Us+2, device=self.device)[1:-1]
            base_v = torch.linspace(0, 1, Vs+2, device=self.device)[1:-1]

        if self.should_optimize:
            base_u = nn.Parameter(
                self.inverse_activation(base_u).contiguous().detach().clone(),
                requires_grad=True,
            )
            base_v = nn.Parameter(
                self.inverse_activation(base_v).contiguous().detach().clone(),
                requires_grad=True,
            )

        self._interval_u = base_u
        self._interval_v = base_v
        self.u_name = f'interval_u_surf_{self.state.surf_uid}'
        self.v_name = f'interval_v_surf_{self.state.surf_uid}'

    # =====================================================================
    # Activations
    # =====================================================================

    # Margin keeps samples strictly interior to [0,1]: boundary basis rows
    # are degenerate and a plain sigmoid saturates (vanishing gradients)
    # when samples drift to the ends — once stuck at 0/1 they cannot recover.
    _MARGIN = 1e-3

    @property
    def activation(self):
        if not self.should_optimize:
            return lambda x: x
        m = self._MARGIN
        return lambda x: m + (1.0 - 2.0 * m) * torch.sigmoid(x)

    @property
    def inverse_activation(self):
        if not self.should_optimize:
            return lambda x: x
        m = self._MARGIN
        return lambda y: inverse_sigmoid(
            ((y - m) / (1.0 - 2.0 * m)).clamp(1e-6, 1.0 - 1e-6)
        )

    # =====================================================================
    # Interval Access (with upsampling / stochastic sampling)
    # =====================================================================

    @property
    def interval_u(self) -> torch.Tensor:
        """Get current U intervals (activated, sorted)."""
        activated = self.activation(self._interval_u)

        if self.state.sampling_density != 1:
            activated = torch.nn.functional.interpolate(
                activated.unsqueeze(0).unsqueeze(0),
                size=(self.state.Us,),
                mode='linear',
            ).squeeze()

        return torch.sort(activated)[0]

    @property
    def interval_v(self) -> torch.Tensor:
        """Get current V intervals (activated, sorted)."""
        activated = self.activation(self._interval_v)

        if self.state.sampling_density != 1:
            activated = torch.nn.functional.interpolate(
                activated.unsqueeze(0).unsqueeze(0),
                size=(self.state.Vs,),
                mode='linear',
            ).squeeze()

        return torch.sort(activated)[0]

    @property
    def uv_grid(self):
        return self._uv_grid

    # =====================================================================
    # Interval Replacement (no size change)
    # =====================================================================

    def replace_interval_u(self, new_u: torch.Tensor):
        self._interval_u = torch.sort(new_u)[0]

    def replace_interval_v(self, new_v: torch.Tensor):
        self._interval_v = torch.sort(new_v)[0]

    def update_intervals(self, new_u: torch.Tensor, new_v: torch.Tensor):
        self.replace_interval_u(new_u)
        self.replace_interval_v(new_v)

    def update_intervals_global(self, new_u: torch.Tensor, new_v: torch.Tensor):
        self._interval_u_global = torch.sort(new_u)[0]
        self._interval_v_global = torch.sort(new_v)[0]
        self.update_intervals(new_u, new_v)

    @torch.no_grad()
    def redistribute_by_visibility(self, vis_u, vis_v, floor: float = 0.2):
        """
        Adaptive tessellation: re-place the SAME number of samples so their
        density follows observed visibility (frustum + occlusion stats),
        with a density floor so currently-unseen regions are never starved
        (they may become visible from other views).

        vis_u: [Us] mean visibility per u-row; vis_v: [Vs] per v-column.
        """
        def _redistribute(vis, positions, n_out, margin):
            w = vis.float().clamp(min=0)
            w = w / w.max().clamp(min=1e-12)
            w = floor + (1.0 - floor) * w                     # density floor
            device = positions.device
            # Piecewise-constant density over segments around current samples
            edges = torch.cat([
                torch.zeros(1, device=device),
                (positions[1:] + positions[:-1]) / 2,
                torch.ones(1, device=device),
            ])
            seg = (edges[1:] - edges[:-1]).clamp(min=1e-12) * w
            cdf = torch.cumsum(seg, 0)
            cdf = cdf / cdf[-1]
            q = torch.linspace(0, 1, n_out + 2, device=device)[1:-1]
            idx = torch.searchsorted(cdf, q).clamp(0, len(positions) - 1)
            cdf_lo = torch.cat([torch.zeros(1, device=device), cdf])[idx]
            t = (q - cdf_lo) / (cdf[idx] - cdf_lo).clamp(min=1e-12)
            new = edges[idx] + t * (edges[idx + 1] - edges[idx])
            return new.clamp(margin, 1.0 - margin).sort()[0]

        cur_u = self.interval_u.detach()
        cur_v = self.interval_v.detach()
        new_u = _redistribute(vis_u, cur_u, self._interval_u.shape[0], self._MARGIN)
        new_v = _redistribute(vis_v, cur_v, self._interval_v.shape[0], self._MARGIN)

        # Write back IN PLACE (same element count -> no optimizer surgery).
        raw_u = self.inverse_activation(new_u)
        raw_v = self.inverse_activation(new_v)
        if isinstance(self._interval_u, nn.Parameter):
            self._interval_u.data.copy_(raw_u)
            self._interval_v.data.copy_(raw_v)
        else:
            self._interval_u = raw_u.to(self._interval_u.device)
            self._interval_v = raw_v.to(self._interval_v.device)

    def invalidate(self):
        if self.state.sampling_mode == 'adaptive':
            if self.state.full_basis:
                self._uv_grid = None
            else:
                self._interval_u = None
                self._interval_v = None

    # =====================================================================
    # Grid helpers
    # =====================================================================

    def update_uv_grid(self, new_uv_grid: torch.Tensor):
        sorted_u = torch.sort(new_uv_grid[..., 0], dim=0)[0]
        sorted_v = torch.sort(new_uv_grid[..., 1], dim=1)[0]
        self._uv_grid = torch.stack([sorted_u, sorted_v], dim=-1)

    def create_uv_grid(self, Us: int, Vs: int):
        if self.state.full_basis:
            self._uv_grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, Us, device='cuda'),
                    torch.linspace(0, 1, Vs, device='cuda'),
                    indexing='ij',
                ),
                dim=-1,
            )
        else:
            self._uv_grid = None

    # =====================================================================
    # Spacing / Deltas
    # =====================================================================

    def diff_u(self) -> float:
        return 1 / self.state.Us

    def diff_v(self) -> float:
        return 1 / self.state.Vs

    @property
    def delta_u(self) -> torch.Tensor:
        if not self.state.full_basis:
            u = self.interval_u.view(self.state.Us, 1)
            prepend = torch.zeros(1, self.state.Vs, device=self.device)
            return torch.diff(u.expand(-1, self.state.Vs), dim=0, prepend=prepend)
        else:
            u = self.uv_grid[..., 0]
            prepend = -u[0:1, :]
            return torch.diff(u, dim=0, prepend=prepend).abs()

    @property
    def delta_v(self) -> torch.Tensor:
        if not self.state.full_basis:
            v = self.interval_v.view(1, self.state.Vs)
            prepend = torch.zeros(self.state.Us, 1, device=self.device)
            return torch.diff(v.expand(self.state.Us, -1), dim=1, prepend=prepend)
        else:
            v = self.uv_grid[..., 1]
            prepend = -v[:, 0:1]
            return torch.diff(v, dim=1, prepend=prepend).abs()

    def forward(self):
        u = self.interval_u
        v = self.interval_v
        if self.state.full_basis:
            u_grid = u.unsqueeze(1).expand(-1, self.state.Vs)
            v_grid = v.unsqueeze(0).expand(self.state.Us, -1)
            return torch.stack([u_grid, v_grid], dim=-1)
        else:
            return (u, v)

    # =====================================================================
    # Internal: raw interval access
    # =====================================================================

    def _get_raw_interval(self, direction: str) -> torch.Tensor:
        """Raw (logit-space if optimizing) interval for direction."""
        return self._interval_u if direction == 'u' else self._interval_v

    def _group_name(self, direction: str) -> str:
        """Optimizer group name for direction."""
        return self.u_name if direction == 'u' else self.v_name

    # =====================================================================
    # Subdivision (Knot Insertion)
    # =====================================================================

    def subdivide(
        self,
        direction: str,
        val: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Insert a new UV sample after Boehm knot insertion at parameter value `val`.

        Steps:
            1. Compute insertion point in activated [0,1] space
            2. Create new sample value (midpoint of neighbors)
            3. Convert to logit space for storage
            4. Splice optimizer state (preserving existing momentum)
        """
        raw = self._get_raw_interval(direction)
        if raw is not None:
            new_interval = self._subdivide_single(direction, optimizer, raw, val)
            setattr(self, f'_interval_{direction}', new_interval)

        raw_global = getattr(self, f'_interval_{direction}_global', None)
        if raw_global is not None:
            new_global = self._subdivide_single(direction, optimizer, raw_global, val)
            setattr(self, f'_interval_{direction}_global', new_global)

    def _subdivide_single(
        self,
        direction: str,
        optimizer: Optional[torch.optim.Optimizer],
        raw: torch.Tensor,
        val: float,
    ) -> torch.Tensor:
        """Insert one sample into a single interval tensor."""
        # 1. Activate to [0,1]
        activated = self.activation(raw) if self.should_optimize else raw
        activated_flat = activated.view(-1)

        # 2. Find insertion index
        insert_idx = torch.searchsorted(
            activated_flat,
            torch.tensor(val, device=self.device),
            side='right',
        ).item()
        insert_idx = max(0, min(insert_idx, activated_flat.numel()))

        # 3. Compute new sample (midpoint of neighbors)
        left_idx = max(0, insert_idx - 1)
        right_idx = min(activated_flat.numel() - 1, insert_idx)
        new_val = (activated_flat[left_idx] + activated_flat[right_idx]) / 2.0

        # 4. Splice into activated space, then sort
        new_activated = torch.cat([
            activated_flat[:insert_idx],
            new_val.unsqueeze(0),
            activated_flat[insert_idx:],
        ])
        new_activated = torch.sort(new_activated)[0]

        # 5. Convert back to logit space
        new_raw = self.inverse_activation(new_activated.clamp(1e-6, 1 - 1e-6))

        # 6. Update optimizer state
        if self.should_optimize:
            if optimizer is None:
                raise ValueError(
                    "Optimizer must be provided when subdividing optimizable intervals."
                )
            group_name = self._group_name(direction)
            result = splice_1d_optimizer_state(
                optimizer=optimizer,
                group_name=group_name,
                new_tensor=new_raw,
                insert_idx=insert_idx,
                num_entries=1,
                remove=False,
                strategy=MomentumStrategy.NEIGHBOR_AVG,
            )
            if result is not None:
                return result

            # Fallback: group not found (shouldn't happen in normal flow)
            return nn.Parameter(
                new_raw.contiguous().requires_grad_(True)
            )

        return new_raw

    # =====================================================================
    # Pruning (Knot Removal)
    # =====================================================================

    def prune_uv(
        self,
        direction: str,
        removed_idx: int,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Remove the UV sample(s) corresponding to a removed control point.

        Args:
            direction:   'u' or 'v'
            removed_idx: Control point index being removed
            optimizer:   Optimizer to update
        """
        density = 1  # samples per control point
        dim_size = self.state.H if direction == 'u' else self.state.W

        sample_start = removed_idx * density
        sample_end = min(sample_start + density, dim_size)
        num_remove = sample_end - sample_start

        if num_remove <= 0:
            return

        # Prune local interval
        logits = self._get_raw_interval(direction)
        if logits is not None:
            new_logits = self._remove_from_interval(
                direction, logits, optimizer, sample_start, num_remove
            )
            setattr(self, f'_interval_{direction}', new_logits)

        # Prune global interval (if exists)
        logits_global = getattr(self, f'_interval_{direction}_global', None)
        if logits_global is not None:
            new_global = self._remove_from_interval(
                direction, logits_global, optimizer, sample_start, num_remove
            )
            setattr(self, f'_interval_{direction}_global', new_global)

    def _remove_from_interval(
        self,
        direction: str,
        logits: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        remove_start: int,
        num_remove: int,
    ) -> torch.Tensor:
        """Remove entries from interval and update optimizer."""
        new_logits = torch.cat([
            logits[:remove_start],
            logits[remove_start + num_remove:],
        ], dim=0)

        if self.should_optimize:
            group_name = self._group_name(direction)
            result = splice_1d_optimizer_state(
                optimizer=optimizer,
                group_name=group_name,
                new_tensor=new_logits,
                insert_idx=remove_start,
                num_entries=num_remove,
                remove=True,
            )
            if result is not None:
                return result

        return new_logits

    # =====================================================================
    # Batch update (used by multi-surface paths)
    # =====================================================================

    def update_interval_u(
        self, new_intervals_data: Dict[str, Tuple], optimizer=None
    ):
        """Update U intervals from external insertion data."""
        for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
            if self.should_optimize and optimizer is not None:
                result = splice_1d_optimizer_state(
                    optimizer=optimizer,
                    group_name=self.u_name,
                    new_tensor=new_interval,
                    insert_idx=insert_idx,
                    num_entries=1,
                    remove=False,
                    strategy=MomentumStrategy.ZERO,
                )
                if result is not None:
                    new_interval = result

            if self.mode == 'multi':
                self._interval_u[uid_str] = new_interval
            else:
                self._interval_u = new_interval

    def update_interval_v(
        self, new_intervals_data: Dict[str, Tuple], optimizer=None
    ):
        """Update V intervals from external insertion data."""
        for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
            if self.should_optimize and optimizer is not None:
                result = splice_1d_optimizer_state(
                    optimizer=optimizer,
                    group_name=self.v_name,
                    new_tensor=new_interval,
                    insert_idx=insert_idx,
                    num_entries=1,
                    remove=False,
                    strategy=MomentumStrategy.ZERO,
                )
                if result is not None:
                    new_interval = result

            self._interval_v = new_interval

    # =====================================================================
    # Optimizer replacement (no size change — pure reset)
    # =====================================================================

    def replace_tensor_to_optimizer(
        self,
        tensor: torch.Tensor,
        name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[nn.Parameter]:
        """Replace parameter in optimizer with zero-initialized momentum."""
        if optimizer is None:
            return None
        return replace_param_in_optimizer(optimizer, name, tensor)

    # =====================================================================
    # Serialization
    # =====================================================================

    def capture_state(self) -> dict:
        state = {
            'mode': self.mode,
            'num_channels': self.num_channels,
            'should_optimize': self.should_optimize,
        }

        if self.should_optimize:
            state['interval_u'] = self._interval_u.data.clone().cpu()
            state['interval_v'] = self._interval_v.data.clone().cpu()
        else:
            state['interval_u'] = self._interval_u.clone().cpu()
            state['interval_v'] = self._interval_v.clone().cpu()

        if hasattr(self, '_interval_u_global'):
            if self.should_optimize:
                state['interval_u_global'] = self._interval_u_global.data.clone().cpu()
                state['interval_v_global'] = self._interval_v_global.data.clone().cpu()
            else:
                state['interval_u_global'] = self._interval_u_global.clone().cpu()
                state['interval_v_global'] = self._interval_v_global.clone().cpu()

        return state

    @classmethod
    def from_state(
        cls,
        state: dict,
        model_state: ModelState,
        device: str = 'cuda',
        evaluate_mode: bool = False,
    ) -> 'SamplerUV':
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.state = model_state
        instance.device = device
        instance.mode = state['mode']
        instance.num_channels = state['num_channels']
        instance.should_optimize = state['should_optimize']
        instance.evaluate_mode = evaluate_mode
        instance._uv_grid = None

        u_data = state['interval_u'].to(device)
        v_data = state['interval_v'].to(device)

        if instance.should_optimize:
            instance._interval_u = nn.Parameter(u_data, requires_grad=True)
            instance._interval_v = nn.Parameter(v_data, requires_grad=True)
        else:
            instance._interval_u = u_data
            instance._interval_v = v_data

        u_global = state.get('interval_u_global')
        v_global = state.get('interval_v_global')
        if u_global is not None and v_global is not None:
            if instance.should_optimize:
                instance._interval_u_global = nn.Parameter(
                    u_global.to(device), requires_grad=True
                )
                instance._interval_v_global = nn.Parameter(
                    v_global.to(device), requires_grad=True
                )
            else:
                instance._interval_u_global = u_global.to(device)
                instance._interval_v_global = v_global.to(device)

        instance.u_name = f'interval_u_surf_{model_state.surf_uid}'
        instance.v_name = f'interval_v_surf_{model_state.surf_uid}'

        return instance
#
# import numpy as np
# import torch
# import torch.nn as nn
# from typing import Tuple, Optional, Dict, Any
# from modules import ModelState
# from utils.general_utils import inverse_sigmoid
# def insert_knot_1d_to_optimizer(
#         new_interval: torch.Tensor,  # [N+1] or [N+1, 1]
#         group_name: str,
#         insert_idx: int,
#         optimizer=None,
#         mode='zero',
#         new_lr=None,
#         should_remove=False) -> Dict[str, torch.Tensor]:
#     """
#     Robustly inserts new elements into optimizer state for 1D interval parameters.
#     Handles knot insertion for sampler intervals with proper momentum interpolation.
#
#     Args:
#         new_interval: New interval tensor after knot insertion [N+1] or [N+1, 1]
#         group_name: Parameter group name (e.g., 'view_0_interval_u_surf_0')
#         insert_val: The knot value (u_bar) that was inserted
#         optimizer: Optimizer instance (uses self.optimizer if None)
#         optimizer_idx: Index for multi-surface scenarios (optional)
#
#     Returns:
#         Dict mapping group_name to updated parameter
#
#     Key Features:
#         - Handles 1D interval vectors (no grid reshaping needed)
#         - Interpolates Adam momentum from neighbors
#         - Preserves monotonicity during insertion
#         - Supports both raw and activated parameter spaces
#     """
#     optimizer = optimizer if optimizer is not None else optimizer
#     optimizable_tensors = {}
#
#     for group in optimizer.param_groups:
#         if group["name"] != group_name:
#             continue
#
#         # Get old parameter
#         old_param = group['params'][0]
#         # if old_param.ndim > 1:
#         old_size = old_param.shape[0]
#         new_size = new_interval.shape[0]
#         # else:
#         #     old_size = old_param.shape[0]
#         #     new_size = new_interval.shape[]
#
#         # Validate size increment
#         expected_increment = new_size - old_size
#         if expected_increment <= 0:
#             print(f"[WARNING] No size increase for {group_name}:  {old_size} -> {new_size}")
#             continue
#
#         # Get optimizer state
#         stored_state = optimizer.state.get(old_param, None)
#
#         # === 1. Determine Insertion Strategy ===
#         # For intervals, we need to find WHERE the new value(s) were inserted
#         # Strategy: Compare activated values to find insertion index
#
#         # === 2. Interpolate Optimizer State ===
#         if stored_state is not None:
#             old_exp_avg = stored_state["exp_avg"]
#             old_exp_avg_sq = stored_state["exp_avg_sq"]
#             orig_avg_shape = old_exp_avg.shape
#             orig_avg_sq_shape = old_exp_avg_sq.shape
#             # Flatten momentum tensors
#             old_exp_avg_flat = old_exp_avg.view(orig_avg_shape)
#             old_exp_avg_sq_flat = old_exp_avg_sq.view(orig_avg_sq_shape)
#
#             # Determine number of elements to insert
#             num_insert = expected_increment
#
#             # === Neighbor-based Interpolation ===
#             # Find valid neighbor indices for momentum interpolation
#             idx_left = max(0, insert_idx - 1)
#             idx_right = min(old_size - 1, insert_idx)
#             # Interpolate momentum (average of neighbors)
#             if idx_left == idx_right:
#                 # Edge case: single neighbor
#                 mom_avg = old_exp_avg_flat[idx_left: idx_left + 1]
#                 mom_sq_avg = old_exp_avg_sq_flat[idx_left:idx_left + 1]
#             else:
#                 # Normal case: average left and right
#                 mom_avg = (old_exp_avg_flat[idx_left: idx_left + 1] +
#                            old_exp_avg_flat[idx_right:idx_right + 1]) / 2.0
#                 mom_sq_avg = (old_exp_avg_sq_flat[idx_left: idx_left + 1] +
#                               old_exp_avg_sq_flat[idx_right:idx_right + 1]) / 2.0
#
#             if not should_remove:
#
#                 # Repeat for all inserted elements
#                 if len(orig_avg_sq_shape) == 1:
#                     mom_insert = mom_avg.repeat(num_insert)
#                     mom_sq_insert = mom_sq_avg.repeat(num_insert)
#                 else:
#                     mom_insert = mom_avg.repeat(num_insert, *[1 for _ in range(1, len(orig_avg_shape))])
#                     mom_sq_insert = mom_sq_avg.repeat(num_insert, *[1 for _ in range(1, len(orig_avg_sq_shape))])
#                 if mode == 'zero':
#                     mom_insert = torch.zeros_like(mom_insert)
#                     mom_sq_insert = torch.zeros_like(mom_sq_insert)
#                 # === Concatenate:  [old[: idx], inserted, old[idx:]] ===
#                 new_exp_avg = torch.cat([
#                     old_exp_avg_flat[:insert_idx],
#                     mom_insert,
#                     old_exp_avg_flat[insert_idx:]
#                 ], dim=0)
#
#                 new_exp_avg_sq = torch.cat([
#                     old_exp_avg_sq_flat[:insert_idx],
#                     mom_sq_insert,
#                     old_exp_avg_sq_flat[insert_idx:]
#                 ], dim=0)
#             else:
#                 # Removal case
#                 new_exp_avg = torch.cat([
#                     old_exp_avg_flat[:insert_idx],
#                     old_exp_avg_flat[insert_idx + 1:]
#                 ], dim=0)
#
#                 new_exp_avg_sq = torch.cat([
#                     old_exp_avg_sq_flat[:insert_idx],
#                     old_exp_avg_sq_flat[insert_idx + 1:]
#                 ], dim=0)
#
#
#
#
#             # === 3. Update Parameter and State ===
#             # Remove old state
#             del optimizer.state[old_param]
#             if len(orig_avg_shape) > 1:
#                 stored_state["exp_avg"] = new_exp_avg.reshape(-1, orig_avg_shape[-1])
#                 stored_state["exp_avg_sq"] = new_exp_avg_sq.reshape(-1, orig_avg_sq_shape[-1])
#             else:
#                 stored_state["exp_avg"] = new_exp_avg.reshape(-1)
#                 stored_state["exp_avg_sq"] = new_exp_avg_sq.reshape(-1)
#             # Create new parameter (preserve shape)
#             new_param = nn.Parameter(
#                 new_interval.view_as(new_interval).contiguous().requires_grad_(True)
#             )
#
#             # Replace in group
#             group["params"][0] = new_param
#             # group["lr"] = new_lr if new_lr is not None else group["lr"]
#
#             # Create new state
#             optimizer.state[group['params'][0]] = stored_state
#
#         else:
#             # No stored state - just update parameter
#             new_param = nn.Parameter(
#                 new_interval.view_as(new_interval).contiguous().requires_grad_(True)
#             )
#             group["params"][0] = new_param
#
#         optimizable_tensors[group_name] = group["params"][0]
#         return optimizable_tensors[group_name]
#
#
#
#
# class SamplerUV(nn.Module):
#     """
#     Module to handle sampling intervals (u/v samples).
#
#     Modes:
#         - 'single': One set of intervals for all views
#         - 'multi': Per-view intervals (ParameterDict)
#     """
#
#     def __init__(
#             self,
#             state: ModelState,
#             late_init=False,
#             mode='single',
#             **kwargs
#     ):
#         super(SamplerUV, self).__init__()
#
#         self.state = state
#         self.device = state.device
#         self.evaluate_mode = kwargs.get('evaluate_mode', False)
#         self.should_optimize = state.opt.optimize_intervals
#         self.num_channels = kwargs.get('num_channels', 1)
#         self.mode = mode
#         self._uv_grid = None
#
#         if not late_init:
#             self._init_intervals(base_u=kwargs.get('base_u', None), base_v=kwargs.get('base_v', None))
#
#     def replace_interval_u(self, new_u):
#         self._interval_u = torch.sort(new_u)[0]
#
#     def replace_interval_v(self, new_v):
#         self._interval_v = torch.sort(new_v)[0]
#
#     def update_intervals_global(self, new_u, new_v):
#         self._interval_u_global = torch.sort(new_u)[0]
#         self._interval_v_global = torch.sort(new_v)[0]
#         self.update_intervals(new_u, new_v)
#     def update_intervals(self, new_u, new_v):
#         self.replace_interval_u(new_u)
#         self.replace_interval_v(new_v)
#     def update_uv_grid(self, new_uv_grid):
#         sorted_in_u = torch.sort(new_uv_grid[..., 0], dim=0)[0]
#         sorted_in_v = torch.sort(new_uv_grid[..., 1], dim=1)[0]
#         self._uv_grid = torch.stack([sorted_in_u, sorted_in_v], dim=-1)
#     def create_uv_grid(self, Us, Vs):
#         self._uv_grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, Us, device='cuda'),
#                                    torch.linspace(0, 1, Vs, device='cuda'), indexing='ij'),
#                     dim=-1) if self.state.full_basis else None
#     def invalidate(self):
#         if self.state.sampling_mode == 'adaptive':
#             if self.state.full_basis:
#                 self._uv_grid = None
#             else:
#                 self._interval_u = None
#                 self._interval_v = None
#     @property
#     def activation(self):
#         return torch.sigmoid if self.should_optimize else lambda x: x
#
#     @property
#     def inverse_activation(self):
#         return inverse_sigmoid if self.should_optimize else lambda x: x
#     @property
#     def random_sampling(self):
#         return False
#             self._init_intervals(base_u=kwargs.get('base_u', None), base_v=kwargs.get('base_v', None))

#     def _init_intervals(self, **kwargs):
#         """Initialize interval parameters."""
#         Us, Vs = self.state.H, self.state.W
#         if kwargs.get('base_u', None) is not None and kwargs.get('base_v', None) is not None:
#             base_samples_u = torch.tensor(kwargs['base_u'], device=self.device)
#             base_samples_v = torch.tensor(kwargs['base_v'], device=self.device)
#         else:
#             base_samples_u = torch.linspace(0, 1, Us+2, device=self.device)[1:-1]
#             base_samples_v = torch.linspace(0, 1, Vs+2, device=self.device)[1:-1]
#
#         # Convert to logit space if optimizing
#         if self.should_optimize:
#             base_samples_u = nn.Parameter(self.inverse_activation(base_samples_u).contiguous().detach().clone(), requires_grad=True)
#             base_samples_v = nn.Parameter(self.inverse_activation(base_samples_v).contiguous().detach().clone(), requires_grad=True)
#
#         # Initialize based on mode
#         self._interval_u = base_samples_u
#         self._interval_v = base_samples_v
#         # self._uv_grid = torch.stack(torch.meshgrid(base_samples_u, base_samples_v, indexing='ij'), dim=-1) if self.state.uv_grid else None
#
#         self.u_name = f'interval_u_surf_{self.state.surf_uid}'
#         self.v_name = f'interval_v_surf_{self.state.surf_uid}'
#
#     @property
#     def interval_u(self):
#         """Get current U intervals (activated)."""
#         activated = self.activation(self._interval_u)
#
#         if not self.should_optimize:
#             return torch.linspace(0.0, 1.0, self.state.Us+2, device=self.state.device)[1:-1].detach().clone()
#
#         if self.random_sampling: # Use interval values to define Gaussian-distributions centered around them, and sample from those distributions according to the sampling-density
#             activated = self.sample_from_interval(activated, 1/(self.state.Us * 10), 10/(self.state.Us))
#
#         elif self.state.sampling_density != 1:
#             activated = torch.nn.functional.interpolate(
#                 activated.unsqueeze(0).unsqueeze(0),
#                 size=(self.state.Us,),
#                 mode='linear',
#             ).squeeze()
#
#
#         return torch.sort(activated)[0]
#
#     def sample_from_interval(self, activated, min_delta, max_delta):
#         activated = torch.sort(activated)[0]
#         std = (torch.diff(activated, prepend=-activated[0].unsqueeze(0)).abs().clamp(min_delta, max_delta) / 5) ** 2
#         gaussian_d = torch.distributions.Normal(loc=activated, scale=std)
#         activated = gaussian_d.sample(sample_shape=(self.state.sampling_density,)).flatten()
#         activated = activated.clamp(0, 1)
#         return activated
#
#     @property
#     def uv_grid(self):
#         return self._uv_grid
#
#     @property
#     def interval_v(self):
#         """Get current V intervals (activated)."""
#         activated = self.activation(self._interval_v)
#         if not self.should_optimize:
#             return torch.linspace(0.0, 1.0, self.state.Vs+2, device=self.state.device)[1:-1].detach().clone()
#         if self.random_sampling: # Use interval values to define Gaussian-distributions centered around them, and sample from those distributions according to the sampling-density
#             activated = self.sample_from_interval(activated, 1 / (self.state.Vs * 10), 3 / (self.state.Vs))
#
#
#         elif self.state.sampling_density != 1:
#             activated = torch.nn.functional.interpolate(
#                 activated.unsqueeze(0).unsqueeze(0),
#                 size=(self.state.Vs,),
#                 mode='linear',
#             ).squeeze()
#
#         # return self.activation(self._interval_v)
#         return torch.sort(activated)[0]
#
#     def diff_u(self):
#         """Spacing for U intervals."""
#         u = self.interval_u
#         return 1/self.state.Us #torch.diff(u, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)
#
#     def diff_v(self):
#         """Spacing for V intervals."""
#         v = self.interval_v
#         return 1/self.state.Vs # torch.diff(v, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)
#
#     @property
#     def delta_u(self):
#         """Grid deltas for U."""
#         if not self.state.full_basis:
#             u = self.interval_u.view(self.state.Us, 1)
#             prepend = torch.zeros(1, self.state.Vs, device=self.device)
#             return torch.diff(u.expand(-1, self.state.Vs), dim=0, prepend=prepend)
#         else: # Case where self.uv_grid is [Us, Vs, 2]
#             u = self.uv_grid[..., 0]
#             prepend = -u[0:1, :]  # Prepend the first row to maintain shape
#             return torch.diff(u, dim=0, prepend=prepend).abs()
#
#
#     @property
#     def delta_v(self):
#         """Grid deltas for V."""
#         if not self.state.full_basis:
#             v = self.interval_v.view(1, self.state.Vs)
#             prepend = torch.zeros(self.state.Us, 1, device=self.device)
#             return torch.diff(v.expand(self.state.Us, -1), dim=1, prepend=prepend)
#         else:
#             v = self.uv_grid[..., 1]
#             prepend = -v[:, 0:1] #torch.zeros(self.state.Us, 1, device=self.device)
#             return torch.diff(v, dim=1, prepend=prepend).abs()
#
#
#     def forward(self):
#         """Return UV grid or tuple."""
#         u = self.interval_u
#         v = self.interval_v
#
#         if self.state.full_basis:
#             u_grid = u.unsqueeze(1).expand(-1, self.state.Vs)
#             v_grid = v.unsqueeze(0).expand(self.state.Us, -1)
#             return torch.stack([u_grid, v_grid], dim=-1)
#         else:
#             return (u, v)
#
#     # =========================================================================
#     # Subdivision (Knot Insertion)
#     # =========================================================================
#
#
#     def _get_raw_interval(self, direction: str) -> torch.Tensor:
#         """
#         Retrieve the raw (logit-space if optimizing) interval parameter
#         for the given direction and current mode/view.
#
#         Returns:
#             Raw interval tensor (logit space if should_optimize, else [0,1])
#         """
#
#         return self._interval_u if direction == 'u' else self._interval_v
#     def _insert_knot_1d_activated(
#             self,
#             intervals_activated: torch.Tensor,
#             knots: torch.Tensor,
#             degree: int,
#             u_bar: float,
#             insert_idx: int
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Insert samples in activated [0,1] space.
#
#         Args:
#             intervals_activated: Current intervals in [0,1]
#             knots: Knot vector
#             degree: Spline degree
#             u_bar: Value to insert
#             insert_idx: Where to insert
#
#         Returns:
#             new_intervals: Updated intervals
#             new_knots: Updated knot vector
#         """
#         device = intervals_activated.device
#         intervals = intervals_activated.view(-1)
#         N = intervals.shape[0]
#
#         # Clamp insert_idx
#         insert_idx = max(0, min(insert_idx, N))
#
#         # Create output tensor
#         new_intervals = torch.zeros(N + 1, device=device, dtype=intervals.dtype)
#
#         # Copy prefix
#         if insert_idx > 0:
#             new_intervals[:insert_idx] = intervals[:insert_idx]
#
#         # Copy suffix
#         if insert_idx < N:
#             new_intervals[insert_idx + 1:] = intervals[insert_idx:]
#
#         # Insert new value (interpolated)
#         if insert_idx == 0:
#             new_intervals[0] = u_bar
#         elif insert_idx == N:
#             new_intervals[N] = u_bar
#         else:
#             # Linear interpolation for smoothness
#             left_val = intervals[insert_idx - 1]
#             right_val = intervals[insert_idx]
#             alpha = (u_bar - left_val) / (right_val - left_val + 1e-8)
#             alpha = alpha.clamp(0, 1)
#             new_intervals[insert_idx] = (1 - alpha) * left_val + alpha * right_val
#
#         # Update knot vector
#         new_knots = torch.cat([
#             knots[:insert_idx + degree + 1],
#             torch.tensor([u_bar], device=device),
#             knots[insert_idx + degree + 1:]
#         ])
#
#         return new_intervals, new_knots
#
#
#     def subdivide(
#             self,
#             direction: str,
#             val: float,
#             optimizer=None
#     ):
#         """
#         Subdivide intervals after knot insertion.
#
#         CRITICAL LOGIC:
#         1. Work in ACTIVATED [0,1] space for geometric operations
#         2. Insert new samples
#         3. Convert back to LOGIT space for storage
#         4. Update optimizer with CORRECT group name and SAFE state init
#         """
#         # Get raw intervals (logit space if optimizing)
#         raw = self._get_raw_interval(direction)
#         raw_global = getattr(self, f'_interval_{direction}_global', None)
#         if raw is not None:
#             new_interval = self.subdivide_single_interval(direction, optimizer, raw, val)
#             setattr(self, f'_interval_{direction}', new_interval)
#         if raw_global is not None:
#             new_interval_global = self.subdivide_single_interval(direction, optimizer, raw_global, val)
#             setattr(self, f'_interval_{direction}_global', new_interval_global)
#
#
#     def subdivide_single_interval(self, direction, optimizer, raw, val):
#         # Activate to [0,1] for geometry
#         interval_activated = self.activation(raw) if self.should_optimize else raw
#         insert_idx_uv = torch.searchsorted(
#             interval_activated.view(-1),
#             torch.tensor(val, device=self.device),
#             side='right',
#         ).item()
#         insert_idx_uv = max(0, min(insert_idx_uv, interval_activated.numel() - 1))
#         # Insert new sample in activated space (midpoint of neighbors)
#         left = interval_activated[insert_idx_uv: insert_idx_uv + 1]
#         right = interval_activated[insert_idx_uv + 1: insert_idx_uv + 2]
#         if right.numel() > 0:
#             new_val = (left + right) / 2
#         else:
#             # Edge case: inserting at the end
#             new_val = left
#         new_interval_activated = torch.cat(
#             (interval_activated[:insert_idx_uv], new_val, interval_activated[insert_idx_uv:])
#         )
#         new_interval_activated = torch.sort(new_interval_activated.view(-1))[0]
#         # Convert back to logit space
#         new_interval = self.inverse_activation(new_interval_activated.clamp(1e-6, 1 - 1e-6))
#         if self.should_optimize:
#             assert optimizer is not None, "Optimizer must be provided when optimizing intervals"
#             # Use the CORRECT group name that matches training_setup
#             group_name = self.u_name if direction == 'u' else self.v_name
#             result = self.replace_tensor_to_optimizer(
#                 new_interval, group_name, optimizer=optimizer
#             )
#             if result is not None:
#                 new_interval = result
#             else:
#                 # Fallback: wrap as parameter manually
#                 new_interval = nn.Parameter(
#                     new_interval.contiguous().requires_grad_(True)
#                 )
#         return new_interval
#
#     def update_interval_u(self, new_intervals_data: Dict[str, Tuple], optimizer=None):
#         """Update U intervals in optimizer."""
#         for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
#             if self.should_optimize:
#                 new_interval = insert_knot_1d_to_optimizer(
#                     new_interval,
#                     self.u_name,
#                     insert_idx=insert_idx,
#                     optimizer=optimizer,
#                     should_remove=False
#                 )
#                 print(f'new interval for {self.u_name}: {new_interval}')
#
#             if self.mode == 'multi':
#                 self._interval_u[uid_str] = new_interval
#             else:
#                 self._interval_u = new_interval
#
#     def prune_uv(self, direction: str, removed_idx: int, optimizer):
#         """
#         Prune UV intervals by removing a knot/control point.
#
#         For softmax-based sampler, we remove logit entries and update optimizer.
#
#         Args:
#             direction: 'u' or 'v' - which dimension to prune
#             removed_idx: Index of the control point being removed
#             optimizer: Optimizer to update
#         """
#         Us_old = self.state.H
#         Vs_old = self.state.W
#
#         # Calculate sampling density
#         density = 1 #int(self.state.sampling_density)
#
#         # Calculate which samples to remove
#         sample_start = removed_idx * density
#         sample_end = min(sample_start + density, Us_old if direction == 'u' else Vs_old)
#         logits = self._get_raw_interval(direction)
#         logits_global = getattr(self, f'_interval_{direction}_global', None)
#
#         new_logits = self.remove_from_interval(direction, logits, optimizer, sample_end, sample_start)
#         setattr(self, f'_interval_{direction}', new_logits)
#         if logits_global is not None:
#             new_logits_global = self.remove_from_interval(direction, logits, optimizer, sample_end, sample_start)
#             setattr(self, f'_interval_{direction}_global', new_logits_global)
#
#
#     def remove_from_interval(self, direction, logits, optimizer, sample_end, sample_start):
#         param_name = self.u_name if direction == 'u' else self.v_name
#         # Remove the logit entries corresponding to the removed samples
#         new_logits = torch.cat([
#             logits[:sample_start],
#             logits[sample_end:]
#         ], dim=0)
#         opt_tensor = self.replace_tensor_to_optimizer(
#             new_logits, param_name, optimizer=optimizer
#         )
#         if opt_tensor is None:
#             opt_tensor = new_logits
#
#         return opt_tensor
#
#     def replace_tensor_to_optimizer(self, tensor, name, optimizer=None):
#         optimizable_tensors = {}
#
#         for group in optimizer.param_groups:
#             if group["name"] == name:
#                 stored_state = optimizer.state.get(group['params'][0], None)
#                 if stored_state is not None:
#                     stored_state["exp_avg"] = torch.zeros_like(tensor)
#                     stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
#                 del optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(tensor.contiguous().requires_grad_(True))
#                 optimizer.state[group["params"][0]] = stored_state
#
#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors.get(name)
#
#
#     def update_interval_v(self, new_intervals_data: Dict[str, Tuple], optimizer=None):
#         """Update U intervals in optimizer."""
#         for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
#             if self.should_optimize:
#                 new_interval = insert_knot_1d_to_optimizer(
#                     new_interval,
#                     self.v_name,
#                     insert_idx=insert_idx,
#                     optimizer=optimizer,
#                     should_remove=False
#                 )
#                 # print(f'new interval for {self.u_name}: {new_interval}')
#
#             self._interval_v = new_interval
#     # =========================================================================
#     # Serialization
#     # =========================================================================
#
#     def capture_state(self) -> dict:
#         """Capture state for saving."""
#         state = {
#             'mode': self.mode,
#             'num_channels': self.num_channels,
#             'should_optimize': self.should_optimize,
#         }
#
#         if self.should_optimize:
#             state['interval_u'] = self._interval_u.data.clone().cpu()
#             state['interval_v'] = self._interval_v.data.clone().cpu()
#         else:
#             state['interval_u'] = self._interval_u.clone().cpu()
#             state['interval_v'] = self._interval_v.clone().cpu()
#         if hasattr(self, '_interval_u_global'):
#             if self.should_optimize:
#                 state['interval_u_global'] = self._interval_u_global.data.clone().cpu()
#                 state['interval_v_global'] = self._interval_v_global.data.clone().cpu()
#             else:
#                 state['interval_u_global'] = self._interval_u_global.clone().cpu()
#                 state['interval_v_global'] = self._interval_v_global.clone().cpu()
#
#
#         return state
#
#     @classmethod
#     def from_state(
#             cls,
#             state: dict,
#             model_state: ModelState,
#             device: str = 'cuda',
#             evaluate_mode: bool = False
#     ) -> 'SamplerUV':
#         """Restore from saved state."""
#         mode = state['mode']
#         num_channels = state['num_channels']
#         should_optimize = state['should_optimize'] #and not evaluate_mode
#
#         instance = cls.__new__(cls)
#         nn.Module.__init__(instance)
#
#         instance.state = model_state
#         instance.device = device
#         instance.mode = mode
#         instance.num_channels = num_channels
#         instance.should_optimize = should_optimize
#         instance.evaluate_mode = evaluate_mode
#
#         # Restore intervals
#         u_data = state['interval_u'].to(device)
#         v_data = state['interval_v'].to(device)
#         u_data_global = state.get('interval_u_global', None)
#         v_data_global = state.get('interval_v_global', None)
#
#
#         if should_optimize:
#             instance._interval_u = nn.Parameter(u_data, requires_grad=True)
#             instance._interval_v = nn.Parameter(v_data, requires_grad=True)
#         else:
#             instance._interval_u = u_data
#             instance._interval_v = v_data
#             if u_data_global is not None and v_data_global is not None:
#                 instance._interval_u_global = u_data_global.to(device)
#                 instance._interval_v_global = v_data_global.to(device)
#
#         return instance
