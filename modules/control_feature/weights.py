"""NURBS weight control features (sigmoid-activated)."""

from typing import Tuple

import torch

from .base import ControlFeature


class WeightControl(ControlFeature):
    """
    Learnable NURBS weights for rational B-spline surfaces.

    Stored as logits; activation = sigmoid, inverse = logit.
    Used by PositionControl for the rational denominator.
    """

    def __init__(self, state, control_grid, basis, **kwargs):
        super().__init__(state, control_grid, basis, **kwargs)

    @property
    def activation(self):
        return torch.sigmoid

    @property
    def inverse_activation(self):
        return torch.logit

    @property
    def features(self):
        return self.activation(self.control_features).view(
            self.state.H, self.state.W, self.control_features.shape[-1],
        )

    def interpolate_samples(self) -> torch.Tensor:
        return super().interpolate_samples()

    forward = interpolate_samples

    def get_weights(self):
        return self.features

    def compute_inserted_grid(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=None, blend_strength=0.3, use_blend=False,
    ) -> Tuple[torch.Tensor, int]:
        return super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )

    def capture_state(self) -> dict:
        state = super().capture_state()
        state['weights_channels'] = self.control_features.shape[-1]
        return state

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        return super(WeightControl, cls).from_state(
            state, model_state, basis, device,
        )