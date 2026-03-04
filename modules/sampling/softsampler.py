
"""
Softmax-based UV Sampler with guaranteed monotonicity.

Key insight: Optimize DIFFERENCES (Δu) via softmax instead of absolute positions.
This ensures monotonicity via cumulative sum.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict, Optional
import math

from model.modules.sampling.SamplerUV import insert_knot_1d_to_optimizer


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
            state: 'ModelState',
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
        eps = 1e-3

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

        if self.state.full_basis:
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