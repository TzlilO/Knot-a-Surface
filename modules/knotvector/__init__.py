"""
Robust KnotVector module with proper initialization and serialization.
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from utils.general_utils import inverse_sigmoid

"""
Robust KnotVector module with proper initialization and serialization.
"""
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from utils.general_utils import inverse_sigmoid


class KnotVector(nn.Module):
    """
    Module to handle knot vectors for B-spline/NURBS.

    Knot vector structure for degree p with n control points:
    - Total knots: n + p + 1
    - Clamped:  first (p+1) knots = 0, last (p+1) knots = 1
    - Internal knots: n - p - 1 values in (0, 1)

    IMPORTANT: Internal knots must satisfy 0 < kv < 1 (strictly interior).
    """

    def __init__(
            self,
            state,  # ModelState
            direction: str = 'u',
            initial_knots: Optional[torch.Tensor] = None,
            num_control: Optional[int] = None,
            **kwargs
    ):
        super(KnotVector, self).__init__()
        self.state = state
        self.direction = direction
        self.degree = state.degree
        self.knot_uid = kwargs.get('name', f'knot_{direction}')
        self.name = kwargs.get('name', f'knot_{direction}_{self.knot_uid}')

        # Determine number of control points
        if num_control is not None:
            self._num_control = num_control
        elif direction == 'u':
            self._num_control = state.H
        else:  # 'v'
            self._num_control = state.W

        device = state.device

        # Calculate expected internal knot count
        n_internal_expected = self._num_control - self.degree - 1

        # Initialize or process knot vector
        if initial_knots is not None:
            internal_knots = self._extract_internal_knots_robust(
                initial_knots.to(device),
                n_internal_expected
            )
        else:
            # Create uniform internal knots
            internal_knots = self._create_uniform_internal_knots(n_internal_expected, device)

        # Validate internal knots
        internal_knots = self._validate_internal_knots(internal_knots, device)

        # Setup optimization mode
        self.should_optimize = state.opt.optimize_knots# and not evaluate_mode
        # Transform to unconstrained space
        internal_clamped = internal_knots.clamp(1e-6, 1.0 - 1e-6)
        raw_internal = self.inverse_activation(internal_clamped)
        if self.should_optimize:

            self._internal_knots = nn.Parameter(
                raw_internal.contiguous(),
                requires_grad=True
            )
        else:

            self.register_buffer('_internal_knots_buffer', raw_internal.contiguous())
            self._internal_knots = self._internal_knots_buffer

        # Debug output
        self._debug_print_init_summary()
    @property
    def activation(self):
        """Activation function to ensure internal knots are in (0, 1)."""
        return torch.sigmoid if self.should_optimize else lambda x: x

    @property
    def inverse_activation(self):
        """Inverse of the activation function for optimization."""
        return inverse_sigmoid if self.should_optimize else lambda x: x
    def _create_uniform_internal_knots(self, n_internal: int, device: torch.device) -> torch.Tensor:
        """Create uniformly spaced internal knots in (0, 1)."""
        if n_internal <= 0:
            return torch.tensor([], device=device, dtype=torch.float32)

        # Create n_internal knots uniformly spaced in (0, 1)
        # Using linspace from eps to 1-eps ensures strict interior
        eps = 1e-4
        return torch.linspace(eps, 1.0 - eps, n_internal, device=device)

    def _extract_internal_knots_robust(
            self,
            knots: torch.Tensor,
            n_internal_expected: int
    ) -> torch.Tensor:
        """
        Robustly extract internal knots from any input format.

        Handles:
        1. Full clamped knot vector [0,0,0,0, .. internal. ., 1,1,1,1]
        2. Just internal knots [0. 25, 0.5, 0.75]
        3. Internal + partial padding
        4. Edge cases (empty, wrong size, etc.)

        Returns:
            Tensor of internal knots with values strictly in (0, 1)
        """
        knots = knots.squeeze().flatten()
        device = knots.device
        n = len(knots)
        p = self.degree

        # Handle empty input
        if n == 0:
            print(f"[KnotVector-{self.direction}] Empty input, creating uniform internal knots")
            return self._create_uniform_internal_knots(n_internal_expected, device)

        # =====================================================================
        # CASE 1: Already just internal knots (correct count, all in (0,1))
        # =====================================================================
        if n == n_internal_expected:
            # Check if all values are strictly interior
            if (knots > 1e-6).all() and (knots < 1.0 - 1e-6).all():
                print(f"[KnotVector-{self.direction}] Input appears to be internal knots (n={n})")
                return knots.clone()

        # =====================================================================
        # CASE 2: Full clamped knot vector
        # =====================================================================
        expected_full = self._num_control + p + 1

        if n == expected_full:
            # Extract by skipping clamped ends
            if n_internal_expected > 0:
                internal = knots[p + 1: -(p + 1)].clone()
                print(f"[KnotVector-{self.direction}] Extracted {len(internal)} internal from full vector (n={n})")
                return self._filter_to_interior(internal, device)
            else:
                print(
                    f"[KnotVector-{self.direction}] No internal knots expected (degree={p}, n_ctrl={self._num_control})")
                return torch.tensor([], device=device, dtype=torch.float32)

        # =====================================================================
        # CASE 3: Heuristic extraction - find values strictly in (0, 1)
        # =====================================================================
        eps = 1e-6
        interior_mask = (knots > eps) & (knots < 1.0 - eps)
        interior_vals = knots[interior_mask]

        if len(interior_vals) == n_internal_expected:
            print(f"[KnotVector-{self.direction}] Heuristic extraction found {len(interior_vals)} interior knots")
            return interior_vals.clone()

        if len(interior_vals) > 0 and len(interior_vals) != n_internal_expected:
            print(
                f"[KnotVector-{self.direction}] WARNING: Found {len(interior_vals)} interior knots, expected {n_internal_expected}")

            if len(interior_vals) > n_internal_expected:
                # Too many - take uniformly spaced subset
                indices = torch.linspace(0, len(interior_vals) - 1, n_internal_expected, device=device).long()
                return interior_vals[indices]
            else:
                # Too few - interpolate to fill gaps
                return self._interpolate_to_count(interior_vals, n_internal_expected, device)

        # =====================================================================
        # CASE 4: No valid interior knots found - create uniform
        # =====================================================================
        print(f"[KnotVector-{self.direction}] No valid interior knots in input, creating uniform")
        return self._create_uniform_internal_knots(n_internal_expected, device)

    def _filter_to_interior(self, knots: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Filter knots to ensure all values are strictly in (0, 1).
        Removes any 0s or 1s that might have leaked through.
        """
        eps = 1e-6

        # Filter out boundary values
        mask = (knots > eps) & (knots < 1.0 - eps)
        filtered = knots[mask]

        # If filtering removed too many, clamp instead
        if len(filtered) == 0 and len(knots) > 0:
            # All values were at boundaries - clamp them inward
            return knots.clamp(eps, 1.0 - eps)

        return filtered

    def _interpolate_to_count(
            self,
            knots: torch.Tensor,
            target_count: int,
            device: torch.device
    ) -> torch.Tensor:
        """Interpolate sparse knots to reach target count."""
        if len(knots) == 0:
            return self._create_uniform_internal_knots(target_count, device)

        if len(knots) == 1:
            # Single knot - create uniform around it
            center = knots[0].item()
            return torch.linspace(
                max(1e-4, center - 0.4),
                min(1.0 - 1e-4, center + 0.4),
                target_count,
                device=device
            )

        # Linear interpolation to target count
        old_params = torch.linspace(0, 1, len(knots), device=device)
        new_params = torch.linspace(0, 1, target_count, device=device)

        # Use searchsorted for interpolation
        indices = torch.searchsorted(old_params, new_params).clamp(1, len(knots) - 1)

        # Linear interpolation weights
        t = (new_params - old_params[indices - 1]) / (old_params[indices] - old_params[indices - 1] + 1e-8)

        new_knots = knots[indices - 1] * (1 - t) + knots[indices] * t
        return new_knots.clamp(1e-4, 1.0 - 1e-4)

    def _validate_internal_knots(self, knots: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Final validation to ensure knots are:
        1. All in (0, 1) strictly
        2. Sorted in ascending order
        3. Have reasonable spacing
        """
        if len(knots) == 0:
            return knots

        # Ensure sorted
        knots = knots.sort()[0]

        # Clamp to strict interior
        eps = 1e-6
        knots = knots.clamp(eps, 1.0 - eps)

        # Check for degenerate cases (all same value)
        if len(knots) > 1:
            knot_range = knots.max() - knots.min()
            if knot_range < 1e-4:
                # Spread them out uniformly
                print(f"[KnotVector-{self.direction}] WARNING: Near-degenerate knots, spreading uniformly")
                return self._create_uniform_internal_knots(len(knots), device)

        return knots

    def _debug_print_init_summary(self):
        """Print initialization summary for debugging."""
        internal = self.internal_knots
        full = self.knots

        print(f"[KnotVector-{self.direction}] Initialization Summary:")
        print(f"  Degree: {self.degree}")
        print(f"  Num Control Points: {self._num_control}")
        print(f"  Internal Knots: {len(internal)}")
        print(f"  Full Knot Vector Length: {len(full)}")

        if len(internal) > 0:
            print(f"  Internal Range: [{internal.min().item():.6f}, {internal.max().item():.6f}]")

            # Check for issues
            if (internal <= 0).any() or (internal >= 1).any():
                print(f"  ⚠️ WARNING: Internal knots outside (0,1)!")
                print(f"     Values <= 0: {(internal <= 0).sum().item()}")
                print(f"     Values >= 1: {(internal >= 1).sum().item()}")

        # Verify full vector structure
        zeros_count = (full < 1e-6).sum().item()
        ones_count = (full > 1 - 1e-6).sum().item()
        print(
            f"  Full Vector:  {zeros_count} zeros, {ones_count} ones, {len(full) - zeros_count - ones_count} interior")

    @property
    def num_control(self) -> int:
        """Number of control points for this direction."""
        return self._num_control

    @num_control.setter
    def num_control(self, value: int):
        self._num_control = value

    @property
    def internal_knots(self) -> torch.Tensor:
        """Get activated internal knots in (0, 1) range."""
        return self.activation(self._internal_knots)

    @property
    def knots(self) -> torch.Tensor:
        """Get full clamped knot vector."""
        internal = self.internal_knots
        device = internal.device

        zeros = torch.zeros(self.degree + 1, device=device)
        ones = torch.ones(self.degree + 1, device=device)

        if len(internal) == 0:
            full = torch.cat([zeros, ones])
        else:
            full = torch.cat([zeros, internal, ones])

        # Sort to ensure monotonicity
        return full.sort()[0]

    def forward(self) -> torch.Tensor:
        """Return full knot vector."""
        return self.knots

    def __call__(self) -> torch.Tensor:
        """Shorthand for forward."""
        return self.forward()

    # =========================================================================
    # Knot Modification
    # =========================================================================

    def update_knot_vector(
            self,
            parent,  # SplineModel
            new_internal_knots: torch.Tensor,
            u_bar: Optional[float] = None,
            optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Update internal knots, handling optimizer state if needed.
        """
        # Validate new knots
        new_internal_knots = self._validate_internal_knots(
            new_internal_knots,
            new_internal_knots.device
        )

        if self.should_optimize:
            # Transform to unconstrained space
            new_clamped = new_internal_knots.clamp(1e-6, 1.0 - 1e-6)
            new_raw = self.inverse_activation(new_clamped)

            if u_bar is not None:
                u_bar_clamped = torch.tensor(
                    u_bar, device=self.state.device
                ).clamp(1e-6, 1.0 - 1e-6)
                # u_bar_raw = self.inverse_activation(u_bar_clamped)

                opt_dict = parent.replace_tensor_to_optimizer(
                    new_raw,
                    self.name,
                    optimizer=optimizer
                )
                self._internal_knots = opt_dict#[self.name]
            else:
                self._internal_knots = parent.replace_tensor_to_optimizer(
                    new_raw,
                    self.name,
                    optimizer=optimizer
                )
        else:
            if hasattr(self, '_internal_knots_buffer'):
                self._internal_knots_buffer = new_internal_knots.contiguous()
                self._internal_knots = self._internal_knots_buffer
            else:
                self._internal_knots = new_internal_knots.contiguous()

        # Update control point count
        self._num_control = len(new_internal_knots) + self.degree + 1

    # =========================================================================
    # Serialization
    # =========================================================================
    def capture_state(self) -> Dict[str, Any]:
        """Capture state for checkpointing."""
        data = {
            'direction': self.direction,
            'degree': self.degree,
            'num_control': self._num_control,
            'should_optimize': self.should_optimize,
        }

        data['raw_parameter'] = self._internal_knots.detach().cpu().clone()
        data['internal_knots'] = self.internal_knots.detach().cpu().clone()

        return data

    @classmethod
    def from_state(
            cls,
            state_dict: Dict[str, Any],
            state: 'ModelState',
            device='cuda',
            evaluate_mode: bool = False

    ) -> 'KnotVector':
        """Restore KnotVector from captured state."""
        direction = state_dict['direction']
        num_control = state_dict['num_control']
        if not evaluate_mode:
            internal_knots = nn.Parameter(state_dict['raw_parameter'], device=device, requires_grad=True)
        else:
            internal_knots = state_dict['internal_knots'].to(state.device)

        instance = cls(
            state=state,
            direction=direction,
            evaluate_mode=evaluate_mode,
            num_control=num_control,
        )
        instance._internal_knots = internal_knots

        return instance

    def __repr__(self) -> str:
        internal = self.internal_knots
        internal_range = f"[{internal.min():.4f}, {internal.max():.4f}]" if len(internal) > 0 else "[]"
        return (
            f"KnotVector(direction='{self.direction}', degree={self.degree}, "
            f"num_control={self._num_control}, optimize={self.should_optimize}, "
            f"n_internal={len(internal)}, range={internal_range})"
        )

    def capture_state2(self) -> dict:
        """Capture knot vector state."""
        state = {
            'direction': self.direction,
            'degree': self.degree,
            'should_optimize': self.should_optimize,
        }

        if self.should_optimize:
            # Internal knots are parameters
            state['internal_knots'] = self._internal_knots.data.clone().cpu()
        else:
            # Internal knots are buffers
            state['internal_knots'] = self._internal_knots.clone().cpu()

        # Full knot vector for reconstruction validation
        state['full_knots'] = self.knots.clone().cpu()

        # Clamped boundary values
        # state['clamp_min'] = getattr(self, 'clamp_min', 0.0)
        # state['clamp_max'] = getattr(self, 'clamp_max', 1.0)

        return state

    @classmethod
    def from_state2(
            cls,
            state: dict,
            model_state: 'ModelState',
            device: str = 'cuda',
            evaluate_mode: bool = False
    ) -> 'KnotVector':
        """Restore KnotVector from captured state."""

        # Reconstruct full knots from internal
        internal_knots = state['internal_knots'].to(device)

        # Create instance
        instance = cls(
            model_state,
            direction=state['direction'],
            initial_knots=internal_knots,
            evaluate_mode=evaluate_mode or not state['should_optimize']
        )

        # Validate reconstruction
        reconstructed = instance.knots
        expected = state['full_knots'].to(device)

        if not torch.allclose(reconstructed, expected, atol=1e-6):
            print(f"[KnotVector] Warning: Reconstructed knots differ from saved.  "
                  f"Max diff: {(reconstructed - expected).abs().max().item():.6f}")

        return instance


