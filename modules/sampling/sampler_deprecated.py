"""
UV Sampler with optimizable intervals.
Supports single-view and multi-view modes.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from model.modules import ModelState
from utils.general_utils import inverse_sigmoid
def insert_knot_1d_to_optimizer(
        new_interval: torch.Tensor,  # [N+1] or [N+1, 1]
        group_name: str,
        insert_idx: int,
        optimizer=None,
        mode='zero',
        new_lr=None,
        should_remove=False) -> Dict[str, torch.Tensor]:
    """
    Robustly inserts new elements into optimizer state for 1D interval parameters.
    Handles knot insertion for sampler intervals with proper momentum interpolation.

    Args:
        new_interval: New interval tensor after knot insertion [N+1] or [N+1, 1]
        group_name: Parameter group name (e.g., 'view_0_interval_u_surf_0')
        insert_val: The knot value (u_bar) that was inserted
        optimizer: Optimizer instance (uses self.optimizer if None)
        optimizer_idx: Index for multi-surface scenarios (optional)

    Returns:
        Dict mapping group_name to updated parameter

    Key Features:
        - Handles 1D interval vectors (no grid reshaping needed)
        - Interpolates Adam momentum from neighbors
        - Preserves monotonicity during insertion
        - Supports both raw and activated parameter spaces
    """
    optimizer = optimizer if optimizer is not None else optimizer
    optimizable_tensors = {}

    for group in optimizer.param_groups:
        if group["name"] != group_name:
            continue

        # Get old parameter
        old_param = group['params'][0]
        # if old_param.ndim > 1:
        old_size = old_param.shape[0]
        new_size = new_interval.shape[0]
        # else:
        #     old_size = old_param.shape[0]
        #     new_size = new_interval.shape[]

        # Validate size increment
        expected_increment = new_size - old_size
        if expected_increment <= 0:
            print(f"[WARNING] No size increase for {group_name}:  {old_size} -> {new_size}")
            continue

        # Get optimizer state
        stored_state = optimizer.state.get(old_param, None)

        # === 1. Determine Insertion Strategy ===
        # For intervals, we need to find WHERE the new value(s) were inserted
        # Strategy: Compare activated values to find insertion index

        # === 2. Interpolate Optimizer State ===
        if stored_state is not None:
            old_exp_avg = stored_state["exp_avg"]
            old_exp_avg_sq = stored_state["exp_avg_sq"]
            orig_avg_shape = old_exp_avg.shape
            orig_avg_sq_shape = old_exp_avg_sq.shape
            # Flatten momentum tensors
            old_exp_avg_flat = old_exp_avg.view(orig_avg_shape)
            old_exp_avg_sq_flat = old_exp_avg_sq.view(orig_avg_sq_shape)

            # Determine number of elements to insert
            num_insert = expected_increment

            # === Neighbor-based Interpolation ===
            # Find valid neighbor indices for momentum interpolation
            idx_left = max(0, insert_idx - 1)
            idx_right = min(old_size - 1, insert_idx)
            # Interpolate momentum (average of neighbors)
            if idx_left == idx_right:
                # Edge case: single neighbor
                mom_avg = old_exp_avg_flat[idx_left: idx_left + 1]
                mom_sq_avg = old_exp_avg_sq_flat[idx_left:idx_left + 1]
            else:
                # Normal case: average left and right
                mom_avg = (old_exp_avg_flat[idx_left: idx_left + 1] +
                           old_exp_avg_flat[idx_right:idx_right + 1]) / 2.0
                mom_sq_avg = (old_exp_avg_sq_flat[idx_left: idx_left + 1] +
                              old_exp_avg_sq_flat[idx_right:idx_right + 1]) / 2.0

            if not should_remove:

                # Repeat for all inserted elements
                if len(orig_avg_sq_shape) == 1:
                    mom_insert = mom_avg.repeat(num_insert)
                    mom_sq_insert = mom_sq_avg.repeat(num_insert)
                else:
                    mom_insert = mom_avg.repeat(num_insert, *[1 for _ in range(1, len(orig_avg_shape))])
                    mom_sq_insert = mom_sq_avg.repeat(num_insert, *[1 for _ in range(1, len(orig_avg_sq_shape))])
                if mode == 'zero':
                    mom_insert = torch.zeros_like(mom_insert)
                    mom_sq_insert = torch.zeros_like(mom_sq_insert)
                # === Concatenate:  [old[: idx], inserted, old[idx:]] ===
                new_exp_avg = torch.cat([
                    old_exp_avg_flat[:insert_idx],
                    mom_insert,
                    old_exp_avg_flat[insert_idx:]
                ], dim=0)

                new_exp_avg_sq = torch.cat([
                    old_exp_avg_sq_flat[:insert_idx],
                    mom_sq_insert,
                    old_exp_avg_sq_flat[insert_idx:]
                ], dim=0)
            else:
                # Removal case
                new_exp_avg = torch.cat([
                    old_exp_avg_flat[:insert_idx],
                    old_exp_avg_flat[insert_idx + 1:]
                ], dim=0)

                new_exp_avg_sq = torch.cat([
                    old_exp_avg_sq_flat[:insert_idx],
                    old_exp_avg_sq_flat[insert_idx + 1:]
                ], dim=0)




            # === 3. Update Parameter and State ===
            # Remove old state
            del optimizer.state[old_param]
            if len(orig_avg_shape) > 1:
                stored_state["exp_avg"] = new_exp_avg.reshape(-1, orig_avg_shape[-1])
                stored_state["exp_avg_sq"] = new_exp_avg_sq.reshape(-1, orig_avg_sq_shape[-1])
            else:
                stored_state["exp_avg"] = new_exp_avg.reshape(-1)
                stored_state["exp_avg_sq"] = new_exp_avg_sq.reshape(-1)
            # Create new parameter (preserve shape)
            new_param = nn.Parameter(
                new_interval.view_as(new_interval).contiguous().requires_grad_(True)
            )

            # Replace in group
            group["params"][0] = new_param
            # group["lr"] = new_lr if new_lr is not None else group["lr"]

            # Create new state
            optimizer.state[group['params'][0]] = stored_state

        else:
            # No stored state - just update parameter
            new_param = nn.Parameter(
                new_interval.view_as(new_interval).contiguous().requires_grad_(True)
            )
            group["params"][0] = new_param

        optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors[group_name]




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
            late_init=False,
            mode='single',
            **kwargs
    ):
        super(SamplerUV, self).__init__()

        self.state = state
        self.device = state.device
        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.should_optimize = state.opt.optimize_intervals and not self.evaluate_mode
        self.num_channels = kwargs.get('num_channels', 1)
        self.mode = mode

        # Activation functions

        if not late_init:
            self._init_intervals(kwargs)

    @property
    def activation(self):
        return torch.sigmoid if self.should_optimize else lambda x: x

    @property
    def inverse_activation(self):
        return inverse_sigmoid if self.should_optimize else lambda x: x
    @property
    def random_sampling(self):
        return False # self.state.opt.random_sampling and not self.should_optimize

    def _init_intervals(self, kwargs):
        """Initialize interval parameters."""
        eps = 1e-6
        # initial_samples_u = kwargs.get('initial_samples_u', None)#
        # initial_samples_v = kwargs.get('initial_samples_v', None)#
        Us, Vs = self.state.H, self.state.W

        # if initial_samples_u is None:
        # else:
            # if isinstance(initial_samples_u, np.ndarray):
            #     initial_samples_u=torch.from_numpy(initial_samples_u)
        # base_samples_u = initial_samples_u.to(self.device).clamp(eps, 1-eps)
        base_samples_u = torch.linspace(0, 1, Us, device=self.device)


        # if initial_samples_v is None:
        base_samples_v = torch.linspace(0, 1, Vs, device=self.device)
        # # else:
        #     if isinstance(initial_samples_v, np.ndarray):
        #         initial_samples_v = torch.from_numpy(initial_samples_v)
        #     base_samples_v = initial_samples_v.to(self.device).clamp(eps, 1-eps)

        # Convert to logit space if optimizing
        if self.should_optimize:
            base_samples_u = nn.Parameter((base_samples_u).contiguous(), requires_grad=True)
            base_samples_v = nn.Parameter((base_samples_v).contiguous(), requires_grad=True)

        # Initialize based on mode
        self._interval_u = base_samples_u
        self._interval_v = base_samples_v

        self.u_name = f'interval_u_surf_{self.state.surf_uid}'
        self.v_name = f'interval_v_surf_{self.state.surf_uid}'

    @property
    def interval_u(self):
        """Get current U intervals (activated)."""
        activated = (self._interval_u)


        if self.random_sampling: # Use interval values to define Gaussian-distributions centered around them, and sample from those distributions according to the sampling-density
            activated = self.sample_from_interval(activated, 1/(self.state.Us * 10), 10/(self.state.Us))

        elif self.state.sampling_density != 1:
            activated = F.interpolate(
                activated.unsqueeze(0).unsqueeze(0),
                size=(self.state.Us,),
                mode='linear',
            ).squeeze()


        return torch.sort(activated)[0]

    def sample_from_interval(self, activated, min_delta, max_delta):
        activated = torch.sort(activated)[0]
        std = (torch.diff(activated, prepend=-activated[0].unsqueeze(0)).abs().clamp(min_delta, max_delta) / 5) ** 2
        gaussian_d = torch.distributions.Normal(loc=activated, scale=std)
        activated = gaussian_d.sample(sample_shape=(self.state.sampling_density,)).flatten()
        activated = activated.clamp(0, 1)
        return activated

    @property
    def interval_v(self):
        """Get current V intervals (activated)."""
        activated = (self._interval_v)

        if self.random_sampling: # Use interval values to define Gaussian-distributions centered around them, and sample from those distributions according to the sampling-density
            activated = self.sample_from_interval(activated, 1 / (self.state.Vs * 10), 3 / (self.state.Vs))


        elif self.state.sampling_density != 1:
            activated = F.interpolate(
                activated.unsqueeze(0).unsqueeze(0),
                size=(self.state.Vs,),
                mode='linear',
            ).squeeze()


        # return self.activation(self._interval_v)
        return torch.sort(activated)[0]

    def diff_u(self):
        """Spacing for U intervals."""
        u = self.interval_u
        return torch.diff(u, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)

    def diff_v(self):
        """Spacing for V intervals."""
        v = self.interval_v
        return torch.diff(v, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)

    @property
    def delta_u(self):
        """Grid deltas for U."""
        u = self.interval_u.view(self.state.Us, 1)
        prepend = torch.zeros(1, self.state.Vs, device=self.device)
        return torch.diff(u.expand(-1, self.state.Vs), dim=0, prepend=prepend)
        # return torch.diff(u, dim=0, prepend=-u[1].unsqueeze(0)).squeeze()


    @property
    def delta_v(self):
        """Grid deltas for V."""
        v = self.interval_v.view(1, self.state.Vs)
        prepend = torch.zeros(self.state.Us, 1, device=self.device)
        return torch.diff(v.expand(self.state.Us, -1), dim=1, prepend=prepend)

    def forward(self):
        """Return UV grid or tuple."""
        u = self.interval_u
        v = self.interval_v

        if self.state.uv_grid:
            u_grid = u.unsqueeze(1).expand(-1, self.state.Vs)
            v_grid = v.unsqueeze(0).expand(self.state.Us, -1)
            return torch.stack([u_grid, v_grid], dim=-1)
        else:
            return (u, v)

    # =========================================================================
    # Subdivision (Knot Insertion)
    # =========================================================================


    def _get_raw_interval(self, direction: str) -> torch.Tensor:
        """
        Retrieve the raw (logit-space if optimizing) interval parameter
        for the given direction and current mode/view.

        Returns:
            Raw interval tensor (logit space if should_optimize, else [0,1])
        """

        return self._interval_u if direction == 'u' else self._interval_v
    def _insert_knot_1d_activated(
            self,
            intervals_activated: torch.Tensor,
            knots: torch.Tensor,
            degree: int,
            u_bar: float,
            insert_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Insert samples in activated [0,1] space.

        Args:
            intervals_activated: Current intervals in [0,1]
            knots: Knot vector
            degree: Spline degree
            u_bar: Value to insert
            insert_idx: Where to insert

        Returns:
            new_intervals: Updated intervals
            new_knots: Updated knot vector
        """
        device = intervals_activated.device
        intervals = intervals_activated.view(-1)
        N = intervals.shape[0]

        # Clamp insert_idx
        insert_idx = max(0, min(insert_idx, N))

        # Create output tensor
        new_intervals = torch.zeros(N + 1, device=device, dtype=intervals.dtype)

        # Copy prefix
        if insert_idx > 0:
            new_intervals[:insert_idx] = intervals[:insert_idx]

        # Copy suffix
        if insert_idx < N:
            new_intervals[insert_idx + 1:] = intervals[insert_idx:]

        # Insert new value (interpolated)
        if insert_idx == 0:
            new_intervals[0] = u_bar
        elif insert_idx == N:
            new_intervals[N] = u_bar
        else:
            # Linear interpolation for smoothness
            left_val = intervals[insert_idx - 1]
            right_val = intervals[insert_idx]
            alpha = (u_bar - left_val) / (right_val - left_val + 1e-8)
            alpha = alpha.clamp(0, 1)
            new_intervals[insert_idx] = (1 - alpha) * left_val + alpha * right_val

        # Update knot vector
        new_knots = torch.cat([
            knots[:insert_idx + degree + 1],
            torch.tensor([u_bar], device=device),
            knots[insert_idx + degree + 1:]
        ])

        return new_intervals, new_knots


    def subdivide(
            self,
            direction: str,
            knots: torch.Tensor,
            degree: int,
            val: float,
            insert_idx: int,
            num_insertions: int,
            optimizer=None
    ):
        """
        Subdivide intervals after knot insertion.

        CRITICAL LOGIC:
        1. Work in ACTIVATED [0,1] space for geometric operations
        2. Insert new samples
        3. Convert back to LOGIT space for storage
        4. Update optimizer with CORRECT group name and SAFE state init
        """
        # Get raw intervals (logit space if optimizing)
        raw = self._get_raw_interval(direction)

        # Activate to [0,1] for geometry
        interval_activated = self.activation(raw) if self.should_optimize else raw

        insert_idx_uv = torch.searchsorted(
            interval_activated.view(-1),
            torch.tensor(val, device=self.device),
            side='right',
        ).item()
        insert_idx_uv = max(0, min(insert_idx_uv, interval_activated.numel() - 1))

        # Insert new sample in activated space (midpoint of neighbors)
        left = interval_activated[insert_idx_uv : insert_idx_uv + 1]
        right = interval_activated[insert_idx_uv + 1 : insert_idx_uv + 2]

        if right.numel() > 0:
            new_val = (left + right) / 2
        else:
            # Edge case: inserting at the end
            new_val = left

        new_interval_activated = torch.cat(
            (interval_activated[:insert_idx_uv], new_val, interval_activated[insert_idx_uv:])
        )
        new_interval_activated = torch.sort(new_interval_activated.view(-1))[0]

        # Convert back to logit space
        new_interval = self.inverse_activation(new_interval_activated.clamp(1e-6, 1 - 1e-6))

        if self.should_optimize and optimizer is not None:
            # Use the CORRECT group name that matches training_setup
            group_name = self.u_name if direction=='u' else self.v_name
            result = self.replace_tensor_to_optimizer(
                new_interval, group_name, optimizer=optimizer
            )
            if result is not None:
                new_interval = result
            else:
                # Fallback: wrap as parameter manually
                new_interval = nn.Parameter(
                    new_interval.contiguous().requires_grad_(True)
                )

        # Assign back to the correct storage location
        if self.mode == 'single':
            if direction == 'u':
                self._interval_u = new_interval
            else:
                self._interval_v = new_interval
        else:
            uid = str(self.active_uid) if self.active_uid is not None else '0'
            if direction == 'u':
                self._interval_u[uid] = new_interval
            else:
                self._interval_v[uid] = new_interval



    def update_interval_u(self, new_intervals_data: Dict[str, Tuple], optimizer=None):
        """Update U intervals in optimizer."""
        for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
            if self.should_optimize:
                new_interval = insert_knot_1d_to_optimizer(
                    new_interval,
                    self.u_name,
                    insert_idx=insert_idx,
                    optimizer=optimizer,
                    should_remove=False
                )
                print(f'new interval for {self.u_name}: {new_interval}')

            if self.mode == 'multi':
                self._interval_u[uid_str] = new_interval
            else:
                self._interval_u = new_interval

    def prune_uv(self, direction: str, removed_idx: int, optimizer):
        """
        Prune UV intervals by removing a knot/control point.

        For softmax-based sampler, we remove logit entries and update optimizer.

        Args:
            direction: 'u' or 'v' - which dimension to prune
            removed_idx: Index of the control point being removed
            optimizer: Optimizer to update
        """
        Us_old = self.state.H
        Vs_old = self.state.W

        # Calculate sampling density
        density = 1#int(self.state.sampling_density)

        # Calculate which samples to remove
        sample_start = removed_idx * density
        sample_end = min(sample_start + density, Us_old if direction == 'u' else Vs_old)
        num_samples_to_remove = 1# sample_end - sample_start

        if num_samples_to_remove <= 0:
            return

        for uid_str in (self._interval_u if self.mode == 'multi' else ['0']):
            if self.should_optimize:
                if direction == 'u':
                    logits = self._interval_u if self.mode == 'multi' else self._interval_u
                    param_name = f'view_{uid_str}_interval_u_surf_{self.state.surf_uid}'
                else:
                    logits = self._interval_v if self.mode == 'multi' else self._interval_v
                    param_name = f'view_{uid_str}_interval_v_surf_{self.state.surf_uid}'

                # Remove the logit entries corresponding to the removed samples
                new_logits = torch.cat([
                    logits[:sample_start],
                    logits[sample_end:]
                ], dim=0)

                if optimizer is not None:
                    opt_tensor = self.replace_tensor_to_optimizer(
                        new_logits, param_name, uid=uid_str, optimizer=optimizer
                    )
                    if opt_tensor is not None:
                        new_logits = opt_tensor

                # Update the parameter
                if self.mode == 'multi':
                    if direction == 'u':
                        self._interval_u[uid_str] = new_logits
                    else:
                        self._interval_v[uid_str] = new_logits
                else:
                    if direction == 'u':
                        self._interval_u = new_logits
                    else:
                        self._interval_v = new_logits
            else:
                # Non-optimizable mode: work with raw intervals
                if direction == 'u':
                    intervals = self._interval_u if self.mode == 'multi' else self._interval_u
                else:
                    intervals = self._interval_v if self.mode == 'multi' else self._interval_v

                new_intervals = torch.cat([
                    intervals[:sample_start],
                    intervals[sample_end:]
                ], dim=0)

                if self.mode == 'multi':
                    if direction == 'u':
                        self._interval_u = new_intervals
                    else:
                        self._interval_v = new_intervals
                else:
                    if direction == 'u':
                        self._interval_u = new_intervals
                    else:
                        self._interval_v = new_intervals

    def replace_tensor_to_optimizer(self, tensor, name, uid=None, optimizer=None):
        optimizable_tensors = {}
        # if optimizer is None:
        #     optimizer = self.optimizer

        # uid = 0 if uid is None else uid  # For ParameterList
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.contiguous().requires_grad_(True))
                # optimizer.state[group['params'][uid]] = stored_state
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors.get(name)

    def update_interval_v(self, new_intervals_data: Dict[str, Tuple], optimizer=None):
        """Update V intervals in optimizer."""
        for uid_str, (new_interval, insert_idx) in new_intervals_data.items():
            if self.should_optimize:
                suffix = f'_surf_{self.state.surf_uid}'
                if self.mode != 'single':
                    group_name = f'view_{uid_str}_interval_v{suffix}'

                else:
                    group_name = f'interval_v{suffix}'

                new_interval = insert_knot_1d_to_optimizer(
                    new_interval,
                    self.v_name,
                    insert_idx=insert_idx,
                    optimizer=optimizer,
                    should_remove=False
                )

            if self.mode == 'multi':
                self._interval_v[uid_str] = new_interval
            else:
                self._interval_v = new_interval

    # =========================================================================
    # Serialization
    # =========================================================================

    def capture_state(self) -> dict:
        """Capture state for saving."""
        state = {
            'mode': self.mode,
            'num_channels': self.num_channels,
            'should_optimize': self.should_optimize,
        }

        if self.mode == 'single':
            if self.should_optimize:
                state['interval_u'] = self._interval_u.data.clone().cpu()
                state['interval_v'] = self._interval_v.data.clone().cpu()
            else:
                state['interval_u'] = self._interval_u.clone().cpu()
                state['interval_v'] = self._interval_v.clone().cpu()
        else:
            state['interval_u'] = {}
            state['interval_v'] = {}
            for uid in self._interval_u.keys():
                if self.should_optimize:
                    state['interval_u'][uid] = self._interval_u[uid].data.clone().cpu()
                    state['interval_v'][uid] = self._interval_v[uid].data.clone().cpu()
                else:
                    state['interval_u'][uid] = self._interval_u[uid].clone().cpu()
                    state['interval_v'][uid] = self._interval_v[uid].clone().cpu()

        if hasattr(self, 'vis_probs'):
            state['vis_probs'] = self.vis_probs.clone().cpu()

        return state

    @classmethod
    def from_state(
            cls,
            state: dict,
            model_state: ModelState,
            device: str = 'cuda',
            evaluate_mode: bool = False
    ) -> 'SamplerUV':
        """Restore from saved state."""
        mode = state['mode']
        num_channels = state['num_channels']
        should_optimize = state['should_optimize'] and not evaluate_mode

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.state = model_state
        instance.device = device
        instance.mode = mode
        instance.num_channels = num_channels
        instance.should_optimize = should_optimize
        instance.evaluate_mode = evaluate_mode

        # Restore intervals
        if mode == 'single':
            u_data = state['interval_u'].to(device)
            v_data = state['interval_v'].to(device)

            if should_optimize:
                instance._interval_u = nn.Parameter(u_data, requires_grad=True)
                instance._interval_v = nn.Parameter(v_data, requires_grad=True)
            else:
                instance._interval_u = u_data
                instance._interval_v = v_data
        else:
            if should_optimize:
                instance._interval_u = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in state['interval_u'].items()
                })
                instance._interval_v = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in state['interval_v'].items()
                })
            else:
                instance._interval_u = {k: v.to(device) for k, v in state['interval_u'].items()}
                instance._interval_v = {k: v.to(device) for k, v in state['interval_v'].items()}

        # Restore visibility
        if 'vis_probs' in state:
            instance.vis_probs = state['vis_probs'].to(device)
        else:
            instance.vis_probs = torch.zeros((num_channels, model_state.Us, model_state.Vs, 2), device=device)

        instance.active_uid = '0'
        return instance


"""
Softmax-based UV Sampler with guaranteed monotonicity.

Key insight: Optimize DIFFERENCES (Δu) via softmax instead of absolute positions.
This ensures monotonicity via cumulative sum.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
import math


class SoftmaxIntervalSampler(nn.Module):
    """
    Learnable UV intervals with guaranteed monotonicity via softmax.

    Storage:
        - _logits_u, _logits_v: Learnable weights for spacing distribution

    Derivation:
        logits → softmax → weights (sum=1) → cumsum → monotonic intervals [0,1]
    """

    def __init__(
            self,
            state: ModelState,
            initial_u: Optional[torch.Tensor] = None,
            initial_v: Optional[torch.Tensor] = None,
            **kwargs
    ):
        super().__init__()

        self.state = state
        self.device = state.device
        self.should_optimize = state.opt.optimize_intervals #and not kwargs.get('evaluate_mode', False)
        self.num_channels = kwargs.get('num_channels', 1)
        self.mode = kwargs.get('mode', 'single' if self.num_channels == 1 else 'multi')

        Us, Vs = state.Us, state.Vs
        eps = 1e-6

        # Initialize from uniform or provided intervals
        if initial_u is None:
            initial_u = torch.linspace(eps, 1 - eps, Us, device=self.device)
        else:
            initial_u = initial_u.to(self.device)

        if initial_v is None:
            initial_v = torch.linspace(eps, 1 - eps, Vs, device=self.device)
        else:
            initial_v = initial_v.to(self.device)

        delta_u_init = torch.diff(initial_u, prepend=initial_u[:1])
        delta_v_init = torch.diff(initial_v, prepend=initial_v[:1])

        if self.should_optimize:
            if self.mode == 'single':
                self._interval_u = nn.Parameter(
                    torch.log(delta_u_init * (Us - 1) + eps)
                )
                self._interval_v = nn.Parameter(
                    torch.log(delta_v_init * (Vs - 1) + eps)
                )
            else:
                # Multi-view mode
                self._interval_u = nn.ParameterDict({
                    str(i): nn.Parameter(torch.log(delta_u_init.clone() * (Us - 1) + eps))
                    for i in range(self.num_channels)
                })
                self._interval_v = nn.ParameterDict({
                    str(i): nn.Parameter(torch.log(delta_v_init.clone() * (Vs - 1) + eps))
                    for i in range(self.num_channels)
                })
        else:
            # Non-optimizable: store intervals directly
            if self.mode == 'single':
                self._interval_u = initial_u
                self._interval_v = initial_v
            else:
                self._interval_u = {str(i): initial_u.clone() for i in range(self.num_channels)}
                self._interval_v = {str(i): initial_v.clone() for i in range(self.num_channels)}

        self.active_uid = '0'
        # self.vis_probs = torch.zeros((self.num_channels, Us, Vs, 2), device=self.device)

    @property
    def interval_u(self) -> torch.Tensor:
        """Monotonic U intervals in [0, 1]."""
        if not self.should_optimize:
            if self.mode == 'single':
                return self._interval_u
            else:
                uid = str(self.active_uid)
                return self._interval_u.get(uid, self._interval_u['0'])

        # Get logits for current view
        if self.mode == 'single':
            logits = self._interval_u
        else:
            uid = str(self.active_uid)
            logits = self._interval_u.get(uid, self._interval_u['0'])

        # Softmax → positive weights summing to 1
        weights = F.softmax(logits, dim=0)

        # Cumulative sum → monotonic intervals
        intervals = torch.cumsum(weights, dim=0)

        # Normalize to [0, 1] (safety, softmax already sums to 1)
        intervals = intervals / (intervals[-1] + 1e-8)

        return intervals#.clamp(1e-6, 1 - 1e-6)

    @property
    def interval_v(self) -> torch.Tensor:
        """Monotonic V intervals in [0, 1]."""
        if not self.should_optimize:
            if self.mode == 'single':
                return self._interval_v
            else:
                uid = str(self.active_uid)
                return self._interval_v.get(uid, self._interval_v['0'])

        if self.mode == 'single':
            logits = self._interval_v
        else:
            uid = str(self.active_uid)
            logits = self._interval_v.get(uid, self._interval_v['0'])

        weights = F.softmax(logits, dim=0)
        intervals = torch.cumsum(weights, dim=0)
        return (intervals / (intervals[-1] + 1e-8))#.clamp(1e-6, 1 - 1e-6)

    def get_density_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-sample density (1/Δu) for regularization.

        Returns:
            density_u, density_v: Higher values = finer sampling
        """
        if not self.should_optimize:
            u = self.interval_u
            v = self.interval_v
            delta_u = torch.diff(u, prepend=torch.tensor([0.], device=self.device))
            delta_v = torch.diff(v, prepend=torch.tensor([0.], device=self.device))
            return 1.0 / (delta_u + 1e-8), 1.0 / (delta_v + 1e-8)

        # Density = inverse of spacing
        if self.mode == 'single':
            weights_u = F.softmax(self._interval_u, dim=0)
            weights_v = F.softmax(self._interval_v, dim=0)
        else:
            uid = str(self.active_uid)
            weights_u = F.softmax(self._interval_u.get(uid, self._interval_u['0']), dim=0)
            weights_v = F.softmax(self._interval_v.get(uid, self._interval_v['0']), dim=0)

        density_u = 1.0 / (weights_u + 1e-8)
        density_v = 1.0 / (weights_v + 1e-8)

        return density_u, density_v

    def regularization_loss(self, target_entropy: Optional[float] = None) -> torch.Tensor:
        """
        Entropy regularization to prevent collapse or over-concentration.

        Args:
            target_entropy: Target entropy (default: log(Us) for uniform)

        Returns:
            Entropy loss
        """
        if not self.should_optimize:
            return torch.tensor(0.0, device=self.device)

        if self.mode == 'single':
            weights_u = F.softmax(self._interval_u, dim=0)
            weights_v = F.softmax(self._interval_v, dim=0)
        else:
            uid = str(self.active_uid)
            weights_u = F.softmax(self._interval_u.get(uid, self._interval_u['0']), dim=0)
            weights_v = F.softmax(self._interval_v.get(uid, self._interval_v['0']), dim=0)

        # Entropy: -Σ p log p
        entropy_u = -(weights_u * torch.log(weights_u + 1e-8)).sum()
        entropy_v = -(weights_v * torch.log(weights_v + 1e-8)).sum()

        if target_entropy is None:
            # Target: entropy of uniform distribution
            target_entropy = math.log(self.state.Us)

        # Penalize deviation from target
        loss = (entropy_u - target_entropy).pow(2) + (entropy_v - target_entropy).pow(2)

        return loss * 0.01  # Small weight

    def diff_u(self):
        """Spacing for U intervals."""
        u = self.interval_u
        return torch.diff(u, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)

    def diff_v(self):
        """Spacing for V intervals."""
        v = self.interval_v
        return torch.diff(v, prepend=torch.tensor([0.], device=self.device)).clamp(1e-6)

    @property
    def delta_u(self):
        """Grid deltas for U."""
        # if self.state.uv_grid:
        u = self.interval_u.view(self.state.Us, 1)
        prepend = torch.zeros(1, self.state.Vs, device=self.device)
        # return torch.diff(u.expand(-1, self.state.Vs), dim=0, prepend=prepend)
        return torch.diff(u, dim=0, prepend=prepend).squeeze()


    @property
    def delta_v(self, expand=True):
        """Grid deltas for V."""
        # if self.state.uv_grid:
        v = self.interval_v.view(1, self.state.Vs)
        prepend = torch.zeros(self.state.Us, 1, device=self.device)
        # if expand:
        #     return torch.diff(v.expand(self.state.Us, -1), dim=1, prepend=prepend)
        # else:
        return torch.diff(v, dim=1, prepend=prepend).squeeze()
        # return self.diff_v()

    def forward(self):
        """Return UV grid or tuple."""
        u = self.interval_u
        v = self.interval_v

        if self.state.uv_grid:
            u_grid = u.unsqueeze(1).expand(-1, self.state.Vs)
            v_grid = v.unsqueeze(0).expand(self.state.Us, -1)
            return torch.stack([u_grid, v_grid], dim=-1)
        else:
            return (u, v)

    # =========================================================================
    # Subdivision
    # =========================================================================

    def subdivide(
            self,
            direction: str,
            knots: torch.Tensor,
            degree: int,
            val: float,
            insert_idx: int,
            num_insertions: int,
            insertion_fn,
            optimizer=None
    ):
        """
        Subdivide by inserting new logits.

        Strategy:
        1. Find insertion point in interval space
        2. Insert new logit entries
        3. Initialize new logits to match local spacing
        """
        new_data = {}

        for uid_str in (self._interval_u.keys() if self.mode == 'multi' else ['0']):
            # Get current intervals
            intervals = self.interval_u if direction == 'u' else self.interval_v

            # Find insertion index
            insert_idx = torch.searchsorted(
                intervals.view(-1),
                torch.tensor(val, device=self.device),
                side='right'
            ).item()
            insert_idx = max(0, min(insert_idx, intervals.numel()))

            # Get current logits
            if self.should_optimize:
                if direction == 'u':
                    logits = self._interval_u[uid_str] if self.mode == 'multi' else self._interval_u
                else:
                    logits = self._interval_v[uid_str] if self.mode == 'multi' else self._interval_v

                # Insert new logit
                # Initialize to average of neighbors for smooth insertion
                if insert_idx == 0:
                    new_logit = logits[0].clone()
                elif insert_idx >= logits.numel():
                    new_logit = logits[-1].clone()
                else:
                    new_logit = (logits[insert_idx - 1] + logits[insert_idx]) / 2

                new_logits = torch.cat([
                    logits[:insert_idx],
                    new_logit.unsqueeze(0),
                    logits[insert_idx:]
                ])
            else:
                # Non-optimizable: insert in interval space directly
                intervals_flat = intervals.view(-1)
                if insert_idx == 0:
                    new_val = val
                elif insert_idx >= intervals_flat.numel():
                    new_val = val
                else:
                    alpha = (val - intervals_flat[insert_idx - 1]) / \
                            (intervals_flat[insert_idx] - intervals_flat[insert_idx - 1] + 1e-8)
                    new_val = (1 - alpha.clamp(0, 1)) * intervals_flat[insert_idx - 1] + \
                              alpha.clamp(0, 1) * intervals_flat[insert_idx]

                new_intervals = torch.cat([
                    intervals_flat[:insert_idx],
                    torch.tensor([new_val], device=self.device),
                    intervals_flat[insert_idx:]
                ])
                new_logits = new_intervals

            new_data[uid_str] = (new_logits, insert_idx)

        # Update
        if direction == 'u':
            self.update_interval_u(new_data, optimizer)
        else:
            self.update_interval_v(new_data, optimizer)


    def prune_uv(self, direction: str, removed_idx: int, optimizer):
        """
        Prune UV intervals by removing a knot/control point.

        For softmax-based sampler, we remove logit entries and update optimizer.

        Args:
            direction: 'u' or 'v' - which dimension to prune
            removed_idx: Index of the control point being removed
            optimizer: Optimizer to update
        """
        Us_old = self.state.Us
        Vs_old = self.state.Vs

        # Calculate sampling density
        density = 1#int(self.state.sampling_density)

        # Calculate which samples to remove
        sample_start = removed_idx * density
        sample_end = min(sample_start + density, Us_old if direction == 'u' else Vs_old)
        # num_samples_to_remove = sample_end - sample_start
        num_samples_to_remove = 1
        if num_samples_to_remove <= 0:
            return

        for uid_str in (self._logits_u.keys() if self.mode == 'multi' else ['0']):
            if self.should_optimize:
                if direction == 'u':
                    logits = self._interval_u[uid_str] if self.mode == 'multi' else self._interval_u
                    param_name = f'view_{uid_str}_interval_u_surf_{self.state.surf_uid}'
                else:
                    logits = self._interval_v[uid_str] if self.mode == 'multi' else self._interval_v
                    param_name = f'view_{uid_str}_interval_v_surf_{self.state.surf_uid}'

                # Remove the logit entries corresponding to the removed samples
                new_logits = torch.cat([
                    logits[:sample_start],
                    logits[sample_end:]
                ], dim=0)

                if optimizer is not None:
                    opt_tensor = self.replace_tensor_to_optimizer(
                        new_logits, param_name, uid=uid_str, optimizer=optimizer
                    )
                    if opt_tensor is not None:
                        new_logits = opt_tensor

                # Update the parameter
                if self.mode == 'multi':
                    if direction == 'u':
                        self._interval_u[uid_str] = new_logits
                    else:
                        self._interval_v[uid_str] = new_logits
                else:
                    if direction == 'u':
                        self._interval_u = new_logits
                    else:
                        self._interval_v = new_logits
            else:
                # Non-optimizable mode: work with raw intervals
                if direction == 'u':
                    intervals = self._interval_u[uid_str] if self.mode == 'multi' else self._interval_u
                else:
                    intervals = self._interval_v[uid_str] if self.mode == 'multi' else self._interval_v

                new_intervals = torch.cat([
                    intervals[:sample_start],
                    intervals[sample_end:]
                ], dim=0)

                if self.mode == 'multi':
                    if direction == 'u':
                        self._interval_u[uid_str] = new_intervals
                    else:
                        self._interval_v[uid_str] = new_intervals
                else:
                    if direction == 'u':
                        self._interval_u = new_intervals
                    else:
                        self._interval_v = new_intervals

    def replace_tensor_to_optimizer(self, tensor, name, uid=None, optimizer=None):
        """
        Replace a tensor in the optimizer state.

        Args:
            tensor: New tensor to use
            name: Parameter group name
            uid: Optional UID for multi-view mode
            optimizer: Optimizer instance

        Returns:
            Optimizable tensor with updated state
        """
        optimizable_tensors = {}

        if optimizer is None:
            return None

        uid_idx = 0 if uid is None else int(uid) if isinstance(uid, str) else uid

        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][uid_idx], None)

                if stored_state is not None:
                    # Reset momentum buffers to match new size
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del optimizer.state[group['params'][uid_idx]]

                # Create new parameter
                group["params"][uid_idx] = nn.Parameter(tensor.contiguous().requires_grad_(True))

                # Restore state
                if stored_state is not None:
                    optimizer.state[group["params"][uid_idx]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][uid_idx]

        return optimizable_tensors.get(name)

    def update_interval_u(self, new_data: Dict[str, Tuple], optimizer=None):
        """Update U logits/intervals."""
        for uid_str, (new_vals, insert_idx) in new_data.items():
            if self.should_optimize:
                suffix = f'_surf_{self.state.surf_uid}'
                group_name = f'view_{uid_str}_interval_u{suffix}'

                new_vals = insert_knot_1d_to_optimizer(
                    new_vals,
                    group_name,
                    insert_idx=insert_idx,
                    optimizer=optimizer,
                    should_remove=False
                )

            if self.mode == 'multi':
                if self.should_optimize:
                    self._interval_u[uid_str] = new_vals
                else:
                    self._interval_u[uid_str] = new_vals
            else:
                if self.should_optimize:
                    self._interval_u = new_vals
                else:
                    self._interval_u = new_vals

    def update_interval_v(self, new_data: Dict[str, Tuple], optimizer=None):
        """Update V logits/intervals."""
        for uid_str, (new_vals, insert_idx) in new_data.items():
            if self.should_optimize:
                suffix = f'_surf_{self.state.surf_uid}'
                group_name = f'view_{uid_str}_interval_v{suffix}'

                new_vals = insert_knot_1d_to_optimizer(
                    new_vals,
                    group_name,
                    insert_idx=insert_idx,
                    optimizer=optimizer,
                    should_remove=False
                )

            if self.mode == 'multi':
                if self.should_optimize:
                    self._interval_v[uid_str] = new_vals
                else:
                    self._interval_v[uid_str] = new_vals
            else:
                if self.should_optimize:
                    self._interval_v = new_vals
                else:
                    self._interval_v = new_vals

    # =========================================================================
    # Serialization
    # =========================================================================

    def capture_state(self) -> dict:
        """Capture state."""
        state = {
            'mode': self.mode,
            'num_channels': self.num_channels,
            'should_optimize': self.should_optimize,
        }

        if self.should_optimize:
            if self.mode == 'single':
                state['logits_u'] = self._interval_u.data.clone().cpu()
                state['logits_v'] = self._interval_v.data.clone().cpu()
            else:
                state['logits_u'] = {k: v.data.clone().cpu() for k, v in self._interval_u.items()}
                state['logits_v'] = {k: v.data.clone().cpu() for k, v in self._interval_v.items()}
        else:
            if self.mode == 'single':
                state['interval_u'] = self._interval_u.clone().cpu()
                state['interval_v'] = self._interval_v.clone().cpu()
            else:
                state['interval_u'] = {k: v.clone().cpu() for k, v in self._interval_u.items()}
                state['interval_v'] = {k: v.clone().cpu() for k, v in self._interval_v.items()}

        if hasattr(self, 'vis_probs'):
            state['vis_probs'] = self.vis_probs.clone().cpu()

        return state

    @classmethod
    def from_state(
            cls,
            state: dict,
            model_state: ModelState,
            device: str = 'cuda',
            evaluate_mode: bool = False
    ) -> 'SoftmaxIntervalSampler':
        """Restore from state."""
        mode = state['mode']
        num_channels = state['num_channels']
        should_optimize = state['should_optimize'] and not evaluate_mode

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.state = model_state
        instance.device = device
        instance.mode = mode
        instance.num_channels = 1
        instance.should_optimize = should_optimize

        # Restore logits or intervals
        if should_optimize:
            if mode == 'single':
                instance._interval_u = nn.Parameter(state['logits_u'].to(device), requires_grad=True)
                instance._interval_v = nn.Parameter(state['logits_v'].to(device), requires_grad=True)
            else:
                instance._interval_u = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in state['logits_u'].items()
                })
                instance._interval_v = nn.ParameterDict({
                    k: nn.Parameter(v.to(device), requires_grad=True)
                    for k, v in state['logits_v'].items()
                })
        else:
            if mode == 'single':
                instance._interval_u = state['interval_u'].to(device)
                instance._interval_v = state['interval_v'].to(device)
            else:
                instance._interval_u = {k: v.to(device) for k, v in state['interval_u'].items()}
                instance._interval_v = {k: v.to(device) for k, v in state['interval_v'].items()}

        # Restore visibility
        if 'vis_probs' in state:
            instance.vis_probs = state['vis_probs'].to(device)
        else:
            instance.vis_probs = torch.zeros((num_channels, model_state.Us, model_state.Vs, 2), device=device)

        instance.active_uid = '0'
        return instance
