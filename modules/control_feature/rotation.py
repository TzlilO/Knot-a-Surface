"""Rotation control features (unit quaternions)."""

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

from .base import ControlFeature
from .quaternion_utils import quaternion_mean, slerp
from modules.spline_formulas import uv_tangent, n2q

if TYPE_CHECKING:
    from .position import PositionControl
    from modules.ModelState import ModelState
    from modules.basis import BasisFunction


class RotationControl(ControlFeature):
    """
    Learnable rotation control grid (unit quaternions w, x, y, z).

    Supports both absolute and residual rotation modes.
    Uses SLERP (not linear interpolation) for blending during
    knot insertion/removal to stay on the quaternion manifold.
    """

    position: 'PositionControl'

    def __init__(self, state, control_grid, basis, use_residual=False, **kwargs):
        super().__init__(state, control_grid, basis, **kwargs)
        if position:= kwargs.get('position'):
            self.set_position(position)
    @property
    def activation(self):
        return lambda x: F.normalize(x, dim=-1)

    def forward(self) -> torch.Tensor:
        # A convex B-spline combination of unit quaternions is NOT unit --
        # renormalize after interpolation (fused path applies no activation).
        return F.normalize(super().forward(), dim=-1)

    def set_position(self, position: 'PositionControl'):
        self.position = position


    # ------------------------------------------------------------------
    # Insertion: rotation-specific post-processing
    # ------------------------------------------------------------------
    def compute_inserted_grid(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=3, blend_strength=0.3, use_blend=False, old_H=None, old_W=None
    ) -> Tuple[torch.Tensor, int]:
        new_grid, insert_idx = super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        new_slice = new_grid[insert_idx: insert_idx + degree + 1]

        if self.state.opt.residual_rots:
            new_slice = torch.zeros_like(new_slice)
            new_slice[..., 0] = 1.0  # Identity quaternion
        else:
            if blend_strength > 0 and blend_radius is not None:
                blend_radius = blend_radius if blend_radius is not None else degree
                # FIX: Use control-grid-level finite differences (H, W, 3)
                # instead of sample-grid derivatives (Us, Vs, 3).
                # dsdu() and dsdv() are computed on the control grid and have
                # the correct shape for indexing by insert_idx.
            tangent = (
                self.position.dsdu()[insert_idx: insert_idx + degree + 1]
                if direction == 'u'
                else self.position.dsdv()[insert_idx: insert_idx + degree + 1]
            )

            self.position.invalidate()  # Invalidate position cache to ensure dsdu/dsdv are up-to-date
            tan_u, tan_v = self.position.dsdu(), self.position.dsdv()
            H, W = (self.state.H, self.state.W) if not direction == 'v' else (self.state.W, self.state.H)
            if direction == 'v':
                tan_u, tan_v = tan_u.permute(1, 0, 2), tan_v.permute(1, 0, 2)

            normals = torch.cross(tan_u, tan_v, dim=-1)
            # Degenerate FD tangents (coincident/boundary ctrl points) give
            # zero or non-finite normals — substitute a safe default.
            bad = (~torch.isfinite(normals)).any(dim=-1) | (normals.norm(dim=-1) < 1e-12)
            if bad.any():
                normals = normals.clone()
                normals[bad] = torch.tensor([0.0, 0.0, 1.0], device=normals.device)
            target_rots = n2q(normals).reshape(H, W, 4)

            new_grid[insert_idx: insert_idx + degree + 1] = target_rots[insert_idx: insert_idx + degree + 1]
            # for i in range(new_slice.shape[0]):
            #     w = blend_strength * torch.exp(
            #         torch.tensor(-0.5 * (i / max(blend_radius / 2, 1)) ** 2)
            #     )
            #     new_slice[i] = slerp(new_slice[i], target_rots[i], t=w.item())
            #
            # # if blend_strength > 0 and blend_radius is not None:
            #     blend_radius = blend_radius if blend_radius is not None else degree
            #     # tangent = (
            #     #     self.position.dsdu()[insert_idx: insert_idx + degree + 1]
            #     #     if direction == 'u'
            #     #     else self.position.dsdv()[insert_idx: insert_idx + degree + 1]
            #     # )
            #     # target_rots = uv_tangent(tangent.reshape(-1, 3))
            # target_rots = (n2q(self.position.dsdu().cross(self.position.dsdv(), dim=-1)))
            # if direction == 'v':
            #     target_rots = target_rots.permute(1, 0, 2)
            # new_slice = target_rots[insert_idx: insert_idx + degree + 1]
            #
            # for i in range(new_slice.shape[0]):
            #     w = blend_strength * torch.exp(
            #         torch.tensor(-0.5 * (i / max(blend_radius / 2, 1)) ** 2)
            #     )
            #     new_slice[i] = slerp(new_slice[i], target_rots[i], t=w.item())

        new_grid[insert_idx: insert_idx + degree + 1] = new_slice

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid.contiguous(), insert_idx
    def compute_inserted_grid2(
        self, direction, knots, degree, val, insert_idx,
        insertion_fn, blend_radius=None, blend_strength=0.3, use_blend=False,
    ) -> Tuple[torch.Tensor, int]:
        new_grid, insert_idx = super().compute_inserted_grid(
            direction, knots, degree, val, insert_idx,
            insertion_fn, blend_radius, blend_strength, use_blend=use_blend,
        )

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        new_slice = new_grid[insert_idx: insert_idx + degree + 1]

        if self.state.opt.residual_rots:
            new_slice = torch.zeros_like(new_slice)
            new_slice[..., 0] = 1.0  # Identity quaternion
        else:
            if blend_strength > 0 and blend_radius is not None:
                blend_radius = blend_radius if blend_radius is not None else degree
                tangent = (
                    self.position.dSu[insert_idx: insert_idx + degree + 1]
                    if direction == 'u'
                    else self.position.dSv[insert_idx: insert_idx + degree + 1]
                )
                target_rots = uv_tangent(tangent.reshape(-1, 3))
                for i in range(new_slice.shape[0]):
                    w = blend_strength * torch.exp(
                        torch.tensor(-0.5 * (i / max(blend_radius / 2, 1)) ** 2)
                    )
                    new_slice[i] = slerp(new_slice[i], target_rots[i], t=w.item())

        new_grid[insert_idx: insert_idx + degree + 1] = new_slice

        if direction == 'v':
            new_grid = new_grid.permute(1, 0, 2)

        return new_grid, insert_idx

    # ------------------------------------------------------------------
    # Insertion blending: SLERP-based (overrides linear base)
    # ------------------------------------------------------------------

    def _apply_insertion_blending(
        self, grid, insert_idx, degree, blend_radius, blend_strength,
        direction='ortho',
    ) -> torch.Tensor:
        if blend_strength <= 0 or blend_radius <= 0:
            return grid

        new_dim, other_dim, ch = grid.shape
        device = grid.device

        insert_start = insert_idx
        insert_end = min(insert_idx + degree + 1, new_dim)

        inserted_region = grid[insert_start:insert_end]
        inserted_mean = quaternion_mean(inserted_region.view(-1, 4))
        inserted_mean = inserted_mean.unsqueeze(0).expand(other_dim, 4)

        blended_grid = grid.clone()

        for offset in range(1, blend_radius + 1):
            weight = blend_strength * torch.exp(
                torch.tensor(-0.5 * (offset / (blend_radius / 2)) ** 2, device=device)
            )
            # Before insertion
            idx_before = insert_start - offset
            if idx_before >= 0:
                for j in range(other_dim):
                    blended_grid[idx_before, j] = slerp(
                        blended_grid[idx_before, j], inserted_mean[j], t=weight.item(),
                    )
            # After insertion
            idx_after = insert_end - 1 + offset
            if idx_after < new_dim:
                for j in range(other_dim):
                    blended_grid[idx_after, j] = slerp(
                        blended_grid[idx_after, j], inserted_mean[j], t=weight.item(),
                    )

        return blended_grid

    # ------------------------------------------------------------------
    # Removal: delegates to base, no rotation-specific compensation
    # ------------------------------------------------------------------

    def compute_removed_grid(
        self, direction, remove_idx, blend_radius=3,
        blend_strength=0.5, use_blend=False,
    ) -> torch.Tensor:
        blend_radius = blend_radius if blend_radius is not None else self.state.degree
        return super().compute_removed_grid(
            direction, remove_idx,
            blend_radius=blend_radius,
            blend_strength=blend_strength,
            use_blend=use_blend,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def capture_state(self) -> dict:
        state = super().capture_state()
        state['rotation_dim'] = 4
        return state

    @classmethod
    def from_state(cls, state, model_state, basis, device='cuda'):
        return super(RotationControl, cls).from_state(
            state, model_state, basis, device,
        )