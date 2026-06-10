
"""
Batched Global Interval Optimization

Aggregates rendering losses and Chhugani importance information across
multiple views and surfaces in batches, rather than processing one
view/surface at a time.

Key Design Decisions:
  1. Accumulate gradients across a BATCH of views before stepping.
     This gives a more stable gradient direction than single-view updates.
  2. Aggregate Chhugani importance maps across the batch to produce a
     "consensus" importance that respects multiple viewpoints.
  3. Process all active surfaces in parallel where possible.

Usage in training loop:
    optimizer_module = BatchedIntervalOptimizer(model, training_cameras)

    # Periodically (e.g., every 2000 iterations):
    optimizer_module.optimize(
        render_fn=render_fn,
        pipe=pipe,
        background=background,
        num_steps=30,
        batch_size=4,       # Views per gradient step
        chhugani_weight=0.1,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
import random

from modules.basis import compute_bases_uv_diff


@dataclass
class BatchConfig:
    """Configuration for batched interval optimization."""
    num_steps: int = 50               # Total optimization steps
    batch_size: int = 2               # Views per gradient step
    lr: float = 0.05                # Learning rate for interval params
    chhugani_weight: float = 0.5      # Weight for Chhugani importance prior
    reconstruction_weight: float = .1 # Weight for rendering loss
    smoothness_weight: float = 0.05   # Penalize jagged intervals
    grad_clip: float = 1.0            # Gradient clipping norm
    warmup_steps: int = 5             # LR warmup steps


class _SurfaceIntervalState:
    """
    Holds the optimizable interval parameters and per-surface accumulators
    for one surface during batched optimization.
    """

    def __init__(self, surface: 'SplineModel', device: str = 'cuda'):
        self.surface = surface
        self.device = device


        # Snapshot the current intervals as starting point
        self.original_u = surface.uv_sampler._interval_u.detach().clone()
        self.original_v = surface.uv_sampler._interval_v.detach().clone()
        Us = self.original_u.shape[0]
        Vs = self.original_v.shape[0]


        # Create optimizable parameters in logit space
        from utils.general_utils import inverse_sigmoid
        self.param_u = nn.Parameter(
            inverse_sigmoid(self.original_u.clamp(1e-6, 1 - 1e-6)),
            requires_grad=True
        )
        self.param_v = nn.Parameter(
            inverse_sigmoid(self.original_v.clamp(1e-6, 1 - 1e-6)),
            requires_grad=True
        )

        # Accumulators for importance aggregation across batch
        self.importance_u_accum = torch.zeros(
            Us, device=device
        )
        self.importance_v_accum = torch.zeros(
            Vs, device=device
        )
        self.accum_count = 0

    @property
    def surf_id(self):
        return self.surface.state.surf_uid
    @property
    def activated_u(self) -> torch.Tensor:
        """Activated and sorted U intervals."""
        return torch.sort(torch.sigmoid(self.param_u))[0]

    @property
    def activated_v(self) -> torch.Tensor:
        """Activated and sorted V intervals."""
        return torch.sort(torch.sigmoid(self.param_v))[0]

    def reset_accumulators(self):
        self.importance_u_accum.zero_()
        self.importance_v_accum.zero_()
        self.accum_count = 0

    def accumulate_importance(
        self, importance_u: torch.Tensor, importance_v: torch.Tensor
    ):
        """
        Add importance from one view to the running accumulator.

        Uses max-aggregation (not mean): if ANY view needs high density
        somewhere, that region should be dense.
        """
        self.importance_u_accum = torch.max(
            self.importance_u_accum, importance_u
        )
        self.importance_v_accum = torch.max(
            self.importance_v_accum, importance_v
        )
        self.accum_count += 1

    @property
    def aggregated_importance_u(self) -> torch.Tensor:
        """Normalized aggregated importance for U."""
        imp = self.importance_u_accum
        total = imp.sum()
        if total < 1e-10:
            return torch.ones_like(imp) / imp.shape[0]
        return imp / total

    @property
    def aggregated_importance_v(self) -> torch.Tensor:
        """Normalized aggregated importance for V."""
        imp = self.importance_v_accum
        total = imp.sum()
        if total < 1e-10:
            return torch.ones_like(imp) / imp.shape[0]
        return imp / total


class BatchedIntervalOptimizer:
    """
    Optimizes UV sampling intervals for all surfaces simultaneously,
    accumulating information across batches of views.

    Unlike the single-view `optimize_intervals`, this:
      1. Samples a BATCH of cameras per step
      2. Renders all of them (forward only, no backward yet)
      3. Computes Chhugani importance for all views in the batch
      4. Aggregates importance via max-pooling across the batch
      5. Computes a combined loss (reconstruction + Chhugani prior)
      6. Steps the interval optimizer ONCE with the aggregated gradient

    This is more stable and produces intervals that work well across views.
    """

    def __init__(
        self,
        model: 'MultiSurfaceSplineModel',
        training_cameras: List,
        config: Optional[BatchConfig] = None,
    ):
        self.model = model
        self.cameras = training_cameras
        self.config = config or BatchConfig()
        self.base_lr = self.config.lr

        self._states: List[_SurfaceIntervalState] = []
        self._optimizer: Optional[torch.optim.Adam] = None

    def effective_lr(self, uid) -> float:
        u_factor = self.model.surfaces[uid].uv_sampler.delta_u
        v_factor = self.model.surfaces[uid].uv_sampler.delta_v
        u_min = u_factor[u_factor > 0].mean() / 2
        v_min = v_factor[v_factor > 0].mean() / 2
        return self.base_lr * (u_min + v_min) / 2
    def _init_states(self):
        """Initialize per-surface interval states and the shared optimizer."""
        self._states = []
        all_params = []

        for surface in self.model.surfaces:
            state = _SurfaceIntervalState(surface)
            self._states.append(state)
            all_params.extend([state.param_u, state.param_v])

        self._optimizer = torch.optim.Adam(
            all_params, lr=self.config.lr, eps=1e-15
        )

    def _sample_batch(self) -> List:
        """Sample a batch of cameras."""
        batch_size = min(self.config.batch_size, len(self.cameras))
        return random.sample(self.cameras, batch_size)

    @torch.no_grad()
    def _compute_batch_importance(self, batch_cameras: List):
        """
        Compute and aggregate Chhugani importance across the batch
        for all surfaces.

        This is the key batching insight: rather than computing importance
        per-view and immediately converting to intervals, we AGGREGATE
        importance across the entire batch first, then convert once.
        """
        from modules.tessellation.chhugani import (
            _compute_flatness_map,
            _classify_patches,
            _compute_subdivision_depth,
            _spans_to_importance,
            ChhuganiParams,
        )

        for surf_state in self._states:
            surf_state.reset_accumulators()

        for camera in batch_cameras:
            for surf_state in self._states:
                surface = surf_state.surface

                params = getattr(surface, '_chhugani_params', None)
                if params is None:
                    params = ChhuganiParams(
                        pixel_threshold=2.0,
                        silhouette_threshold=0.2,
                        backface_factor=0.05,
                        min_density=0.01,
                        coarse_u=max(surface.state.H * 4, 64),
                        coarse_v=max(surface.state.W * 4, 64),
                    )

                # Use the full Chhugani pipeline
                try:
                    flatness_u, flatness_v = _compute_flatness_map(
                        surface, camera, params
                    )
                    cos_u, cos_v, bf_u, bf_v = _classify_patches(
                        surface, camera, params
                    )
                    subdiv_u = _compute_subdivision_depth(
                        flatness_u, params.pixel_threshold
                    )
                    subdiv_v = _compute_subdivision_depth(
                        flatness_v, params.pixel_threshold
                    )
                    imp_u = _spans_to_importance(
                        subdiv_u, surface.knot_u().detach(),
                        surface.state.degree, params.coarse_u,
                        cos_u, bf_u, params,
                        camera.camera_center.device,
                    )
                    imp_v = _spans_to_importance(
                        subdiv_v, surface.knot_v().detach(),
                        surface.state.degree, params.coarse_v,
                        cos_v, bf_v, params,
                        camera.camera_center.device,
                    )

                    # Resample to match Us/Vs if needed
                    Us = surface.state.H
                    Vs = surface.state.W
                    if imp_u.shape[0] != Us:
                        imp_u = F.interpolate(
                            imp_u.unsqueeze(0).unsqueeze(0),
                            size=Us, mode='linear', align_corners=True
                        ).squeeze()
                    if imp_v.shape[0] != Vs:
                        imp_v = F.interpolate(
                            imp_v.unsqueeze(0).unsqueeze(0),
                            size=Vs, mode='linear', align_corners=True
                        ).squeeze()

                    surf_state.accumulate_importance(imp_u, imp_v)

                except Exception as e:
                    # Fallback: uniform importance
                    surf_state.accumulate_importance(
                        torch.ones(surface.state.H, device='cuda'),
                        torch.ones(surface.state.W, device='cuda'),
                    )

    def _chhugani_prior_loss(
        self,
        surf_state: _SurfaceIntervalState,
    ) -> torch.Tensor:
        """
        Loss that pulls intervals toward the Chhugani-derived distribution.

        The idea: the aggregated importance defines a target PDF.
        The current intervals define an empirical CDF.
        We minimize the Wasserstein-1 distance between them.

        This acts as a PRIOR — the reconstruction loss is the likelihood.
        Together they produce intervals that render well AND respect geometry.
        """
        device = surf_state.param_u.device
        Us = surf_state.surface.state.H
        Vs = surf_state.surface.state.W

        # Current intervals (activated, sorted)
        u_intervals = surf_state.activated_u
        v_intervals = surf_state.activated_v

        # Target CDF from aggregated importance
        target_pdf_u = surf_state.aggregated_importance_u
        target_pdf_v = surf_state.aggregated_importance_v

        target_cdf_u = torch.cumsum(target_pdf_u, dim=0)
        target_cdf_v = torch.cumsum(target_pdf_v, dim=0)

        # Empirical CDF from current intervals
        # Intervals in [0,1] → their CDF is just their sorted positions
        empirical_cdf_u = u_intervals  # Already sorted, in [0,1]
        empirical_cdf_v = v_intervals

        # Normalize target CDF endpoints to match empirical
        if target_cdf_u[-1] > 0:
            target_cdf_u = target_cdf_u / target_cdf_u[-1]
        if target_cdf_v[-1] > 0:
            target_cdf_v = target_cdf_v / target_cdf_v[-1]

        # Wasserstein-1 = integral of |CDF_target - CDF_empirical|
        # Approximate with L1 distance at matching quantile points
        loss_u = (empirical_cdf_u - target_cdf_u).abs().mean()
        loss_v = (empirical_cdf_v - target_cdf_v).abs().mean()

        return loss_u + loss_v

    def _smoothness_loss(
        self,
        surf_state: _SurfaceIntervalState,
    ) -> torch.Tensor:
        """
        Penalize jagged interval distributions.

        Adjacent interval spacings should not vary wildly. This prevents
        the optimizer from creating distributions with extreme density
        spikes that cause basis function numerical issues.
        """
        u = surf_state.activated_u
        v = surf_state.activated_v

        # Second differences of intervals (penalize acceleration)
        if u.shape[0] > 2:
            d2u = u[2:] - 2 * u[1:-1] + u[:-2]
            loss_u = d2u.pow(2).mean()
        else:
            loss_u = torch.tensor(0.0, device=u.device)

        if v.shape[0] > 2:
            d2v = v[2:] - 2 * v[1:-1] + v[:-2]
            loss_v = d2v.pow(2).mean()
        else:
            loss_v = torch.tensor(0.0, device=v.device)

        return loss_u + loss_v

    def _reconstruction_loss_batch(
        self,
        batch_cameras: List,
        render_fn: Callable,
        pipe,
        background: torch.Tensor,
        app_model=None,
    ) -> torch.Tensor:
        """
        Compute rendering reconstruction loss across a batch of views.

        For each view in the batch:
          1. Set intervals on all surfaces
          2. Recompute basis
          3. Render
          4. Compare to GT
          5. Accumulate loss

        The key: we do NOT call optimizer.step() between views.
        We accumulate the loss, then backprop once through the sum.
        """
        total_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        valid_views = 0

        for camera in batch_cameras:
            # 1. Update intervals and basis for all surfaces
            for surf_state in self._states:
                surface = surf_state.surface
                u_sorted = surf_state.activated_u
                v_sorted = surf_state.activated_v

                # Compute differentiable basis
                basis_data = compute_bases_uv_diff(
                    u_sorted, v_sorted,
                    surface.knot_u(),
                    surface.knot_v(),
                    surface.state.H,
                    surface.state.W,
                    degree=3,
                )
                surface.basis.replace_funcs(basis_data)
                surface.invalidate_control_features()

            # 2. Invalidate multi-surface cache
            self.model._invalidate_cache(force=True)

            # 3. Render
            try:
                render_pkg = render_fn(
                    camera, self.model, pipe, background,
                    app_model=app_model,
                    return_plane=False,
                    return_depth_normal=False,
                )
                rendered_image = render_pkg['render']

                # 4. Compare to GT
                gt_image = camera.original_image.cuda()
                view_loss = F.l1_loss(rendered_image, gt_image)

                total_loss = total_loss + view_loss
                valid_views += 1

            except Exception as e:
                # Skip views that fail (e.g., OOM)
                continue

        if valid_views > 0:
            total_loss = total_loss / valid_views

        return total_loss

    def optimize(
        self,
        render_fn: Callable,
        pipe,
        background: torch.Tensor,
        app_model=None,
        config: Optional[BatchConfig] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Main entry point: batched interval optimization.

        Returns:
            List of (global_u, global_v) per surface.
        """
        cfg = config or self.config

        # Initialize states and optimizer
        self._init_states()

        # ===================================================================
        # Phase 1: Aggregate Chhugani importance across ALL training cameras
        # ===================================================================
        # This is done ONCE, not per step, since geometry doesn't change.
        # We process cameras in chunks to avoid excessive memory.

        chunk_size = max(cfg.batch_size * 2, 8)
        for chunk_start in range(0, len(self.cameras), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(self.cameras))
            chunk_cameras = self.cameras[chunk_start:chunk_end]
            self._compute_batch_importance(chunk_cameras)

        # ===================================================================
        # Phase 2: Iterative optimization with batched rendering
        # ===================================================================
        for step in range(cfg.num_steps):
            self._optimizer.zero_grad()

            # Learning rate warmup
            if step < cfg.warmup_steps:
                lr_scale = 1#((step + 1) / cfg.warmup_steps)
                lr = 0.0
                for surf_state in self._states:
                    lr += (self.effective_lr(surf_state.surf_id) * lr_scale)
                lr = lr / len(self._states)
                for pg in self._optimizer.param_groups:
                    pg['lr'] = lr

            # Sample a batch of cameras for this step
            batch_cameras = self._sample_batch()

            # --- Combined loss ---
            loss = torch.tensor(0.0, device='cuda', requires_grad=True)

            # A. Reconstruction loss (differentiable through intervals → basis → render)
            if cfg.reconstruction_weight > 0:
                recon_loss = self._reconstruction_loss_batch(
                    batch_cameras, render_fn, pipe, background, app_model
                )

                loss = loss + cfg.reconstruction_weight * recon_loss

            # B. Chhugani prior loss (per surface)
            if cfg.chhugani_weight > 0:
                for surf_state in self._states:
                    prior_loss = self._chhugani_prior_loss(surf_state)
                    loss = loss + cfg.chhugani_weight * prior_loss

            # C. Smoothness regularization
            if cfg.smoothness_weight > 0:
                for surf_state in self._states:
                    smooth_loss = self._smoothness_loss(surf_state)
                    loss = loss + cfg.smoothness_weight * smooth_loss

            # --- Backward + step ---
            if loss.requires_grad and not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()



                # Gradient clipping
                all_params = []
                for s in self._states:
                    all_params.extend([s.param_u, s.param_v])
                torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)

                self._optimizer.step()
            else:
                print(f"[BatchedIntervalOpt] Step {step}/{cfg.num_steps} | Loss is NaN or Inf, skipping step.")

            if step % 10 == 0:
                print(
                    f"[BatchedIntervalOpt] Step {step}/{cfg.num_steps} | "
                    f"Loss: {loss.item():.6f}"
                )

        # ===================================================================
        # Phase 3: Extract final intervals and apply
        # ===================================================================
        results = []
        # for surfaces in self.surfaces:
        #     surfaces.uv_sampler.update_intervals_global()
        for surf_state in self._states:
            final_u = surf_state.activated_u.detach().clone()
            final_v = surf_state.activated_v.detach().clone()

            results.append((final_u, final_v))

            # Log change magnitude
            change_u = (final_u - surf_state.original_u).abs().mean()
            change_v = (final_v - surf_state.original_v).abs().mean()
            print(
                f"[BatchedIntervalOpt] Surface "
                f"({surf_state.surface.state.H}x{surf_state.surface.state.W}): "
                f"ΔU={change_u:.6f}, ΔV={change_v:.6f}"
            )

        # Cleanup
        self._states = []
        self._optimizer = None
        torch.cuda.empty_cache()

        return results


# =========================================================================
# Integration with Chhugani per-view tessellation
# =========================================================================

def apply_global_intervals_as_prior(
    surface: 'SplineModel',
    camera,
    global_weight: float = 0.3,
):
    """
    When _global_intervals exist, blend them with per-view Chhugani
    intervals. This lets the global optimization set a "base" distribution
    that per-view Chhugani refines.

    Call this INSIDE update_uv_distribution_chhugani, after computing
    the per-view intervals but before storing/applying them.

    Strategy: weighted average in CDF space.
      - Global intervals define a "consensus" distribution across views
      - Per-view intervals adapt to the specific camera
      - Blending in CDF space preserves monotonicity
    """
    if not hasattr(surface, '_global_intervals') or surface._global_intervals is None:
        return None, None

    global_u, global_v = surface._global_intervals
    device = global_u.device

    # Get per-view intervals from forward context
    ctx = surface._forward_context
    uid = camera.uid

    if uid not in ctx.warped_us:
        return global_u, global_v

    view_u = ctx.warped_us[uid]
    view_v = ctx.warped_vs[uid]

    # Blend: weighted average (both are sorted, so result stays sorted)
    blended_u = (1 - global_weight) * view_u + global_weight * global_u
    blended_v = (1 - global_weight) * view_v + global_weight * global_v

    # Re-sort for safety
    blended_u = torch.sort(blended_u)[0]
    blended_v = torch.sort(blended_v)[0]

    return blended_u, blended_v