"""
Optimizer state management utilities for parameter resize operations.

Used by KnotVector, SamplerUV, and SplineModel when knot insertion or
removal changes the size of optimizable parameters mid-training.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn


class MomentumStrategy(Enum):
    """How to initialize Adam momentum for newly inserted parameter entries."""
    ZERO = 'zero'
    NEIGHBOR_AVG = 'neighbor_avg'
    INTERPOLATE = 'interpolate'


# =========================================================================
# Core: Atomic optimizer state mutation
# =========================================================================

def replace_param_in_optimizer(
    optimizer: torch.optim.Optimizer,
    group_name: str,
    new_tensor: torch.Tensor,
    new_state: Optional[Dict[str, torch.Tensor]] = None,
    param_idx: int = 0,
) -> Optional[nn.Parameter]:
    """
    Atomically replace a parameter in an optimizer group.

    This is the single source of truth for the 5-step dance:
        1. Locate group by name
        2. Read old state
        3. Delete old state entry (keyed by old param identity)
        4. Create new nn.Parameter
        5. Re-key state under new param identity

    Args:
        optimizer:   The optimizer instance.
        group_name:  Name of the param group (group["name"]).
        new_tensor:  Raw tensor data for the new parameter.
        new_state:   If provided, use this as the new optimizer state dict.
                     If None and old state existed, state is zeroed to match new_tensor.
                     If None and no old state existed, no state is created.
        param_idx:   Index within group["params"] (usually 0).

    Returns:
        The newly created nn.Parameter, or None if group_name was not found.
    """
    for group in optimizer.param_groups:
        if group["name"] != group_name:
            continue

        old_param = group["params"][param_idx]
        old_state = optimizer.state.pop(old_param, None)

        new_param = nn.Parameter(
            new_tensor.contiguous().requires_grad_(True)
        )
        group["params"][param_idx] = new_param

        # Determine what state to attach
        if new_state is not None:
            optimizer.state[new_param] = new_state
        elif old_state is not None:
            # Default: zero-initialize to match new shape
            old_state["exp_avg"] = torch.zeros_like(new_tensor)
            old_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
            # Preserve step count if present
            optimizer.state[new_param] = old_state
        # else: no old state, no new state → parameter has never been stepped

        return new_param

    return None


# =========================================================================
# 1D splice: insert / remove entries in a flat parameter
# =========================================================================

def splice_1d_optimizer_state(
    optimizer: torch.optim.Optimizer,
    group_name: str,
    new_tensor: torch.Tensor,
    insert_idx: int,
    num_entries: int = 1,
    remove: bool = False,
    strategy: MomentumStrategy = MomentumStrategy.ZERO,
) -> Optional[nn.Parameter]:
    """
    Insert or remove entries in a 1D parameter's optimizer state.

    For insertion:  old[N] → new[N + num_entries]
    For removal:    old[N] → new[N - num_entries]

    Momentum for new entries is initialized according to `strategy`:
        ZERO         → zeros
        NEIGHBOR_AVG → average of left/right neighbors

    Args:
        optimizer:    Optimizer instance.
        group_name:   Parameter group name.
        new_tensor:   The already-resized parameter tensor.
        insert_idx:   Splice point (entries go before this index for insert,
                      entries [idx, idx+num_entries) are removed for remove).
        num_entries:  How many entries to insert/remove.
        remove:       True for removal, False for insertion.
        strategy:     Momentum initialization strategy.

    Returns:
        New nn.Parameter, or None if group not found.
    """
    for group in optimizer.param_groups:
        if group["name"] != group_name:
            continue

        old_param = group["params"][0]
        old_state = optimizer.state.get(old_param, None)

        if old_state is None:
            # Never stepped — just replace the parameter
            return replace_param_in_optimizer(
                optimizer, group_name, new_tensor
            )

        old_avg = old_state["exp_avg"]
        old_avg_sq = old_state["exp_avg_sq"]
        old_len = old_avg.shape[0]

        # Validate
        if remove:
            if insert_idx + num_entries > old_len:
                raise RuntimeError(
                    f"Cannot remove {num_entries} entries at idx {insert_idx} "
                    f"from tensor of length {old_len}."
                )
            new_avg = torch.cat([
                old_avg[:insert_idx],
                old_avg[insert_idx + num_entries:]
            ], dim=0)
            new_avg_sq = torch.cat([
                old_avg_sq[:insert_idx],
                old_avg_sq[insert_idx + num_entries:]
            ], dim=0)
        else:
            # Insertion
            if insert_idx < 0 or insert_idx > old_len:
                raise RuntimeError(
                    f"insert_idx={insert_idx} out of range [0, {old_len}]."
                )

            fill_avg, fill_avg_sq = _compute_fill_momentum(
                old_avg, old_avg_sq, insert_idx, num_entries, strategy
            )

            new_avg = torch.cat([
                old_avg[:insert_idx],
                fill_avg,
                old_avg[insert_idx:]
            ], dim=0)
            new_avg_sq = torch.cat([
                old_avg_sq[:insert_idx],
                fill_avg_sq,
                old_avg_sq[insert_idx:]
            ], dim=0)

        # Shape validation against new_tensor
        if new_avg.shape[0] != new_tensor.shape[0]:
            raise RuntimeError(
                f"Momentum length {new_avg.shape[0]} != "
                f"new_tensor length {new_tensor.shape[0]} for '{group_name}'."
            )

        new_state = {
            **old_state,
            "exp_avg": new_avg.contiguous(),
            "exp_avg_sq": new_avg_sq.contiguous(),
        }

        return replace_param_in_optimizer(
            optimizer, group_name, new_tensor, new_state=new_state
        )

    return None


def _compute_fill_momentum(
    old_avg: torch.Tensor,
    old_avg_sq: torch.Tensor,
    insert_idx: int,
    num_entries: int,
    strategy: MomentumStrategy,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute momentum values for newly inserted entries."""
    entry_shape = old_avg.shape[1:]  # works for both 1D ([N]) and multi-dim ([N, ...])

    if strategy == MomentumStrategy.ZERO:
        fill_avg = old_avg.new_zeros(num_entries, *entry_shape)
        fill_avg_sq = old_avg_sq.new_zeros(num_entries, *entry_shape)

    elif strategy == MomentumStrategy.NEIGHBOR_AVG:
        old_len = old_avg.shape[0]
        idx_left = max(0, insert_idx - 1)
        idx_right = min(old_len - 1, insert_idx)

        if idx_left == idx_right:
            avg_val = old_avg[idx_left]
            avg_sq_val = old_avg_sq[idx_left]
        else:
            avg_val = (old_avg[idx_left] + old_avg[idx_right]) / 2.0
            avg_sq_val = (old_avg_sq[idx_left] + old_avg_sq[idx_right]) / 2.0

        fill_avg = avg_val.unsqueeze(0).expand(num_entries, *entry_shape).contiguous()
        fill_avg_sq = avg_sq_val.unsqueeze(0).expand(num_entries, *entry_shape).contiguous()

    else:
        raise ValueError(f"Unknown strategy for 1D fill: {strategy}")

    return fill_avg, fill_avg_sq