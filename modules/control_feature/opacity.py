"""Opacity control features (sigmoid-activated)."""

from typing import Tuple

import torch

from .base import ControlFeature
from utils.general_utils import inverse_sigmoid


class OpacityControl(ControlFeature):
    """
    Learnable per-control-point opacity.

    Stored as logits; activation = sigmoid, inverse = inverse_sigmoid.
    """

    def __init__(self, state, control_grid, basis, **kwargs):
        super().__init__(state, control_grid, basis, **kwargs)

    @property
    def activation(self):
        return torch.sigmoid

    @property
    def inverse_activation(self):
        return inverse_sigmoid

    def forward(self) -> torch.Tensor:
        return super().forward() if self.control_features is not None else None

    def compute_inserted_grid(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=2, blend_strength=0.5, use_blend=False,old_H=None, old_W=None
    ) -> Tuple[torch.Tensor, int]:
        return super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        return super(OpacityControl, cls).from_state(
            state, model_state, basis, device,
        )