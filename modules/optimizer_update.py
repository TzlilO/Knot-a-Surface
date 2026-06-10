from typing import Optional, Tuple, Dict

import torch
from torch import nn


def insert_tensors_to_optimizer(
    self,
    tensors_dict: Dict[str, Tuple[torch.Tensor, int]],
    direction: str = 'u',
    degree: int = None,
    u_bar: float = None,
    insert_idx: int = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    momentum_strategy: str = 'zero',  # 'zero', 'neighbor_avg', 'neighbor_max', 'copy_nearest', 'boehm'
    old_knots: Optional[torch.Tensor] = None,
    old_H: Optional[int] = None,
    old_W: Optional[int] = None,
) -> Dict[str, nn.Parameter]:
    """
    Update optimizer parameters and Adam state after B-spline knot insertion.

    After a knot is inserted in the control grid (Boehm's algorithm), the
    optimizer's parameter tensors must be replaced and the Adam momentum
    buffers (exp_avg, exp_avg_sq) must be resized to match the new grid.

    Args:
        tensors_dict: Mapping from param group name to (new_grid, insert_idx).
                      new_grid has shape (H_new, W_new, C) or (H_new, W_new, *ch).
        direction:    'u' (insert row) or 'v' (insert column).
        degree:       B-spline degree (needed for 'boehm' strategy).
        u_bar:        Inserted knot value (needed for 'boehm' strategy).
        insert_idx:   Span index k where the knot was inserted.
        optimizer:    Optimizer instance (defaults to self.optimizer).
        momentum_strategy:
            'zero'          - Zero-initialize momentum for the inserted row/column.
            'neighbor_avg'  - Average of the two adjacent rows/columns.
            'neighbor_max'  - Element-wise max of the two adjacent rows/columns.
            'copy_nearest'  - Copy from the nearest existing row/column.
            'boehm'         - Apply Boehm's knot insertion blending (same as control points).
        old_knots:    Pre-insertion knot vector. Required for 'boehm' strategy.
                      If None and strategy is 'boehm', falls back to 'zero'.
        old_H:        Grid height BEFORE insertion. If None, inferred from self.state._H.
        old_W:        Grid width BEFORE insertion. If None, inferred from self.state._W.

    Returns:
        Dict mapping param group names to their new nn.Parameter objects.
    """
    if optimizer is None:
        optimizer = self.optimizer

    is_v = (direction == 'v')
    optimizable_tensors = {}

    for group in optimizer.param_groups:
        name = group["name"]
        value = tensors_dict.get(name)
        if value is None:
            continue

        new_grid, _ = value

        # --- 1. Determine feature channel shape ---
        ch = self._get_channel_shape(name, new_grid)

        old_param = group['params'][0]
        stored_state = optimizer.state.get(old_param, None)

        # --- 2. Update Adam momentum buffers ---
        if stored_state is not None:
            stored_state = self._update_momentum_for_insertion(
                stored_state=stored_state,
                name=name,
                ch=ch,
                is_v=is_v,
                degree=degree,
                u_bar=u_bar,
                insert_idx=insert_idx,
                momentum_strategy=momentum_strategy,
                old_knots=old_knots,
                old_H=old_H,
                old_W=old_W,
            )

        # --- 3. Replace parameter in optimizer ---
        new_param_flat = new_grid.reshape(-1, *ch)
        new_param = nn.Parameter(new_param_flat.contiguous(), requires_grad=True)

        # Atomic swap: delete old state -> replace param -> reassign state
        if stored_state is not None:
            del optimizer.state[old_param]
            group["params"][0] = new_param
            optimizer.state[new_param] = stored_state
        else:
            group["params"][0] = new_param

        optimizable_tensors[name] = new_param

    return optimizable_tensors


def _get_channel_shape(
    self,
    name: str,
    new_grid: torch.Tensor,
) -> Tuple[int, ...]:
    """
    Resolve the per-element channel shape for a named parameter group.

    SH features have multi-dimensional channel shapes (e.g., (1, 3) for DC,
    (15, 3) for rest); all other parameters use the last dim of new_grid.
    """
    if name.startswith('f_dc'):
        return tuple(self.spherical_harmonics.sh_dc.control_features.shape[1:])
    elif name.startswith('f_rest'):
        return tuple(self.spherical_harmonics.sh_rest.control_features.shape[1:])
    else:
        return (new_grid.shape[-1],)


def _update_momentum_for_insertion(
    self,
    stored_state: dict,
    name: str,
    ch: Tuple[int, ...],
    is_v: bool,
    degree: Optional[int],
    u_bar: Optional[float],
    insert_idx: Optional[int],
    momentum_strategy: str,
    old_knots: Optional[torch.Tensor],
    old_H: Optional[int],
    old_W: Optional[int],
) -> dict:
    """
    Resize Adam momentum buffers after a single knot insertion.

    Reshapes the flat momentum to (H, W, C), inserts a new row (or column
    via transpose), applies the chosen strategy to fill the new entries,
    then flattens back to (H_new * W_new, *ch).

    Returns the mutated stored_state dict.
    """
    H = old_H if old_H is not None else self.state._H
    W = old_W if old_W is not None else self.state._W

    for key in ("exp_avg", "exp_avg_sq"):
        mom = stored_state[key]

        # --- Reshape to grid ---
        try:
            mom_grid = mom.view(H, W, -1)
        except RuntimeError:
            print(
                f"[insert_tensors_to_optimizer] Shape mismatch for '{name}' "
                f"{key}: expected {H}*{W}={H * W} elements, got {mom.numel()}. "
                f"Reinitializing to zeros."
            )
            # Compute expected new shape
            new_H = (H + 1) if not is_v else H
            new_W = (W + 1) if is_v else W
            C = mom.numel() // (H * W) if (H * W) > 0 else 1
            stored_state[key] = torch.zeros(
                new_H * new_W, *ch,
                dtype=mom.dtype, device=mom.device
            )
            continue

        # --- Transpose so insertion is always along dim 0 ---
        if is_v:
            mom_grid = mom_grid.permute(1, 0, 2)  # (W, H, C)

        # --- Insert new row ---
        new_mom_grid = self._insert_momentum_row(
            mom_grid=mom_grid,
            key=key,
            insert_idx=insert_idx,
            degree=degree,
            u_bar=u_bar,
            strategy=momentum_strategy,
            old_knots=old_knots,
        )

        # --- Transpose back ---
        if is_v:
            new_mom_grid = new_mom_grid.permute(1, 0, 2)  # (H, W_new, C)

        # --- Flatten and store ---
        stored_state[key] = new_mom_grid.reshape(-1, *ch).contiguous()

    return stored_state


def _insert_momentum_row(
    self,
    mom_grid: torch.Tensor,
    key: str,
    insert_idx: Optional[int],
    degree: Optional[int],
    u_bar: Optional[float],
    strategy: str,
    old_knots: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Insert a single row into a (D, N, C) momentum grid using the given strategy.

    D is the dimension being extended (H for 'u', W for 'v' after transpose).

    Strategies:
        'zero'          - New row is all zeros.
        'neighbor_avg'  - Average of rows [idx-1] and [idx].
        'neighbor_max'  - Element-wise max of rows [idx-1] and [idx].
                          (Preferred for exp_avg_sq to avoid underestimating variance.)
        'copy_nearest'  - Copy the nearest existing row.
        'boehm'         - Boehm's knot insertion blending (convex combination).
                          Requires degree, u_bar, old_knots.

    Returns:
        Tensor of shape (D+1, N, C).
    """
    D, N, C = mom_grid.shape
    device = mom_grid.device
    dtype = mom_grid.dtype

    # Determine the row position where we insert.
    # After Boehm's insertion, the affected rows are [k-p+1 .. k],
    # and the new grid has D+1 rows. The 'insert_idx' from the caller
    # is the span index k. The first truly "new" row is at position
    # (k - degree + 1) through k, but for simple strategies we treat
    # the insertion as adding one row at position (k - degree + 1).
    # For strategies other than 'boehm', we use a simpler model:
    # just splice one new row at the position where control point
    # count increased.
    if insert_idx is not None and degree is not None:
        # The new row is effectively at position (k - degree + 1) in the new grid
        # but for neighbor-based strategies, the simplest correct insertion point
        # is k (the span), since rows before (k-p+1) are copied verbatim and
        # rows after k are shifted. We insert at k-degree+1 (first affected row).
        row_pos = max(0, min(insert_idx - degree + 1, D))
    else:
        # Fallback: insert at the end
        row_pos = D

    if strategy == 'boehm':
        if old_knots is not None and degree is not None and u_bar is not None:
            return self._boehm_momentum_insertion(
                mom_grid, old_knots, degree, u_bar, insert_idx
            )
        else:
            # Cannot do Boehm without knots — fall back to zero
            print(
                f"[insert_tensors_to_optimizer] 'boehm' strategy requested but "
                f"old_knots/degree/u_bar not provided. Falling back to 'zero'."
            )
            strategy = 'zero'

    if strategy == 'zero':
        new_row = torch.zeros(1, N, C, device=device, dtype=dtype)

    elif strategy == 'neighbor_avg':
        left = max(0, row_pos - 1)
        right = min(D - 1, row_pos)
        new_row = ((mom_grid[left] + mom_grid[right]) / 2.0).unsqueeze(0)

    elif strategy == 'neighbor_max':
        left = max(0, row_pos - 1)
        right = min(D - 1, row_pos)
        new_row = torch.max(mom_grid[left], mom_grid[right]).unsqueeze(0)

    elif strategy == 'copy_nearest':
        nearest = min(row_pos, D - 1)
        new_row = mom_grid[nearest].unsqueeze(0).clone()

    else:
        raise ValueError(
            f"Unknown momentum_strategy '{strategy}'. "
            f"Expected one of: 'zero', 'neighbor_avg', 'neighbor_max', "
            f"'copy_nearest', 'boehm'."
        )

    # Splice: [0..row_pos) + new_row + [row_pos..D)
    new_mom = torch.cat([
        mom_grid[:row_pos],
        new_row,
        mom_grid[row_pos:],
    ], dim=0)

    assert new_mom.shape[0] == D + 1, (
        f"Momentum insertion produced shape {new_mom.shape[0]}, expected {D + 1}"
    )
    return new_mom


def _boehm_momentum_insertion(
    self,
    mom_grid: torch.Tensor,
    old_knots: torch.Tensor,
    degree: int,
    u_bar: float,
    k: int,
) -> torch.Tensor:
    """
    Apply Boehm's knot insertion algorithm to a momentum grid.

    This mirrors insert_knot_u() but operates on optimizer state rather
    than control points. The result is a (D+1, N, C) tensor.

    NOTE: This is semantically questionable for Adam statistics — use
    'zero' or 'neighbor_avg' for more principled behavior. Provided
    for backward compatibility.
    """
    D, N, C = mom_grid.shape
    device = mom_grid.device

    new_mom = torch.zeros(D + 1, N, C, device=device, dtype=mom_grid.dtype)

    # Prefix: rows [0, k-degree] are unchanged
    prefix_len = k - degree + 1
    if prefix_len > 0:
        new_mom[:prefix_len] = mom_grid[:prefix_len]

    # Suffix: rows [k, D-1] -> [k+1, D]
    if k < D:
        new_mom[k + 1:] = mom_grid[k:]

    # Affected rows: [k-degree+1, k] — Boehm's convex combination
    if not isinstance(u_bar, torch.Tensor):
        u_bar_t = torch.tensor(u_bar, device=device, dtype=old_knots.dtype)
    else:
        u_bar_t = u_bar.to(device=device, dtype=old_knots.dtype)

    indices = torch.arange(k - degree + 1, k + 1, device=device)
    denom = old_knots[indices + degree] - old_knots[indices] + 1e-10
    alpha = ((u_bar_t - old_knots[indices]) / denom).clamp(0.0, 1.0)

    # Q_i = (1-a)*P_{i-1} + a*P_i
    new_mom[indices] = (
        (1.0 - alpha[:, None, None]) * mom_grid[indices - 1]
        + alpha[:, None, None] * mom_grid[indices]
    )

    return new_mom