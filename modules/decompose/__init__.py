
# (continuing from _merge_clusters)
"""
Warmup-to-Decomposition controller.

Sits outside MultiSurfaceSplineModel entirely.
Reads from the model, writes a NEW model back to the caller.

Design principle: zero changes to existing KnotSurface.py / multisurf.py.
All integration is in the training loop.

Segmentation modes:
  1. DEPTH_UV       — Original depth+color k-means in UV space
  2. SEMANTIC       — Multi-feature segmentation on the control grid
  3. DEPTH_SEMANTIC — Depth coarse segmentation → semantic refinement
  4. CONTOUR        — Level-set contour: FG = inside, BG = outside (NEW)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum

from utils.sh_utils import SH2RGB


# ======================================================================
# Enums
# ======================================================================

class DecompPhase(Enum):
    WARMUP = "warmup"  # Single surface, train normally
    DEPTH_COLLECTION = "depth_col"  # Still training, now recording depths
    READY = "ready"  # Trigger decomposition this step
    DECOMPOSED = "decomposed"  # Done, training K surfaces normally


class SegmentationMode(Enum):
    DEPTH_UV = "depth_uv"  # Original: k-means on depth+color+uv
    SEMANTIC = "semantic"  # Multi-feature segmentation on control grid
    DEPTH_SEMANTIC = "depth_semantic"  # Depth coarse → semantic refinement
    CONTOUR = "contour"  # Level-set contour decomposition (NEW)
    HYBRID = "hybrid"  # Level-set contour decomposition (NEW)


# ======================================================================
# Config
# ======================================================================

@dataclass
@dataclass
class ControllerConfig:
    """
    Configuration for the Warmup-to-Decomposition controller.

    Controls the full pipeline: warmup phase → depth collection → segmentation →
    mask-guided grid resampling into separate BG/FG surfaces.

    The pipeline has four stages:
      1. WARMUP: Train a single surface until quality threshold is met.
      2. DEPTH_COLLECTION: Continue training, recording per-view depth maps
         from the surface to build an aggregated depth field.
      3. READY: Trigger decomposition — build a level-set field from depth,
         color, and normal cues, extract a binary FG/BG mask, then resample
         the parent grid into two child grids via adaptive subdivision.
      4. DECOMPOSED: Train the resulting K surfaces jointly.

    Parameters are grouped by stage and function:
      - Warmup & triggering
      - Segmentation mode selection
      - Level-set field construction (what signals, how much each matters)
      - Level-set smoothing (contour regularity)
      - Mask extraction & morphological cleanup (binary mask quality)
      - Mask coverage bounds (prevent degenerate masks)
      - Legacy segmentation modes (DEPTH_UV, SEMANTIC, DEPTH_SEMANTIC)
      - Complementary opacity (only for full-grid clone path, not subdivision)
    """

    # =====================================================================
    # Warmup & Decomposition Triggering
    # =====================================================================

    warmup_iters: int = 4000
    """Minimum training iterations before decomposition can trigger.
    The single surface must train at least this many steps to establish
    meaningful geometry. Too low → noisy depth/color → bad segmentation.
    Too high → wasted single-surface iterations. Typical range: 2000–6000."""

    n_depth_views: int = 32
    """Number of depth maps to collect before triggering decomposition.
    Each training view during DEPTH_COLLECTION phase records one depth map.
    These are median-aggregated into a single robust depth field.
    More views → more robust depth (less view-dependent noise), but delays
    decomposition. Typical range: 15–50."""

    min_psnr_to_decompose: float = 20.0
    """PSNR threshold (dB) required to exit warmup. Ensures the surface has
    converged enough that depth/color cues are meaningful. If the scene is
    inherently low-quality (sparse views, textureless), lower this.
    Typical range: 18.0–25.0."""

    n_components: int = 2
    """Number of segments for legacy modes (DEPTH_UV, SEMANTIC, DEPTH_SEMANTIC).
    2 = background + foreground. K > 2 = multi-object decomposition.
    CONTOUR mode always produces exactly 2 components regardless of this value."""

    smooth_sigma: float = 1.5
    """Gaussian kernel sigma for smoothing the SH color grid before computing
    color variance in the level-set field. Higher → more spatial averaging,
    suppresses high-frequency SH noise, but may blur real color boundaries.
    Applied at 0.5× this value for the color variance cue. Typical: 1.0–3.0."""

    depth_fg_quantile: float = 0.35
    """Fixed depth quantile for FG/BG split when use_otsu=False.
    Points closer than this quantile of the depth distribution are labeled FG.
    Lower → smaller FG region. Only used as fallback when Otsu is disabled.
    Typical range: 0.2–0.5."""

    min_component_frac: float = 0.05
    """Minimum fraction of total control points a segment must occupy to survive.
    Components smaller than this are absorbed into their nearest neighbor.
    Prevents spurious micro-segments from noise. Typical range: 0.02–0.10."""

    bg_lr_scale: float = 1.0
    """Learning rate multiplier for background surface position parameters
    post-decomposition. Higher → BG converges faster (useful since BG is
    typically simpler geometry). Applied via background_lr_scale_factor in
    the optimizer setup. Typical range: 0.3–1.0 for dampening, 1.0–5.0 for
    boosting."""

    obj_lr_scale: float = 1.0
    """Learning rate multiplier for foreground/object surface parameters
    post-decomposition. Usually 1.0 (no change from base LR)."""

    # =====================================================================
    # Segmentation Mode Selection
    # =====================================================================

    segmentation_mode: SegmentationMode = SegmentationMode.CONTOUR
    """Which segmentation algorithm to use for decomposition.

    CONTOUR (recommended): Builds a continuous scalar level-set field φ(u,v)
      from depth, color variance, and normal variance. The zero-crossing of φ
      defines the FG/BG boundary. Produces smooth, topology-aware masks.
      Uses mask-guided grid resampling (subdivision approach).

    DEPTH_UV: K-means clustering on depth + color + UV position features.
      Simple but sensitive to initialization and depth noise.

    SEMANTIC: K-means on a richer feature set (color, normals, opacity,
      depth, spatial position). Better than DEPTH_UV for textured scenes.

    DEPTH_SEMANTIC: Two-stage — coarse depth segmentation via Otsu/multi-
      threshold, then semantic refinement via cluster signature matching.
      Most robust of the legacy modes but slower.

    Legacy modes (DEPTH_UV, SEMANTIC, DEPTH_SEMANTIC) use bounding-box
    cropping, which creates the 'grey hole' artifacts. CONTOUR with
    subdivision resampling eliminates this."""

    # =====================================================================
    # Otsu Thresholding
    # =====================================================================

    use_otsu: bool = True
    """Use Otsu's method for automatic depth threshold selection.
    When True: finds the threshold that maximizes inter-class variance
    (adaptive to the scene's actual depth distribution).
    When False: uses depth_fg_quantile as a fixed percentile threshold.
    Otsu is almost always better; disable only for debugging or scenes
    where the depth histogram is pathologically multi-modal."""

    otsu_num_bins: int = 256
    """Histogram resolution for Otsu's method. More bins → finer threshold
    search but slower. 256 is standard. Lower (64–128) for very small grids
    where the histogram is sparse. Higher (512) rarely helps."""

    # =====================================================================
    # Semantic Segmentation Feature Weights
    # (Used by SEMANTIC and DEPTH_SEMANTIC modes only)
    # =====================================================================

    semantic_color_weight: float = 0.0
    """Weight of SH DC color features in the semantic feature vector.
    Higher → segments follow color boundaries. Set to 0 to ignore color
    (useful for textureless scenes). Typical range: 0.0–0.6."""

    semantic_normal_weight: float = 0.4
    """Weight of surface normal features in the semantic feature vector.
    Higher → segments follow curvature boundaries. Very effective for
    separating flat backgrounds from curved objects. Typical: 0.2–0.5."""

    semantic_opacity_weight: float = 0.1
    """Weight of opacity features in the semantic feature vector.
    Higher → segments follow opacity boundaries (transparent vs opaque).
    Usually low since opacity is noisy early in training. Typical: 0.0–0.2."""

    semantic_spatial_weight: float = 0.5
    """Weight of UV spatial coordinates in the semantic feature vector.
    Higher → segments are spatially contiguous (prevents fragmented masks).
    Acts as a regularizer. Too high → ignores actual feature boundaries.
    Typical: 0.2–0.8."""

    semantic_depth_weight: float = 0.05
    """Weight of aggregated depth in the semantic feature vector.
    Low because depth is already the primary cue in other stages.
    Prevents redundancy with depth-first modes. Typical: 0.0–0.2."""

    # =====================================================================
    # Semantic Feature Toggles
    # =====================================================================

    semantic_use_normals: bool = True
    """Include surface normals (computed from finite differences on the
    position control grid) in the semantic feature vector. Disable for
    scenes where normal estimation is unreliable (very sparse grids)."""

    semantic_use_opacity: bool = True
    """Include sigmoid(opacity control features) in the semantic feature
    vector. Disable if opacity is not being refined (refine_opacities=False)
    or if opacity hasn't converged by warmup end."""

    # =====================================================================
    # Depth-Semantic Refinement
    # (Used by DEPTH_SEMANTIC mode only)
    # =====================================================================

    depth_coarse_components: int = 3
    """Number of coarse depth clusters in the first stage of DEPTH_SEMANTIC.
    More clusters → finer initial segmentation, but more merging needed.
    2 = simple FG/BG split, 3+ = captures depth layers. Typical: 2–5."""

    semantic_merge_threshold: float = 0.2
    """Distance threshold for agglomerative merging of over-segmented clusters
    in DEPTH_SEMANTIC's second stage. Lower → less merging (keeps more
    segments). Higher → more aggressive merging toward n_components.
    The effective cutoff is 3× this value. Typical: 0.1–0.4."""

    # =====================================================================
    # Morphological Cleanup
    # (Applied to binary masks in all modes)
    # =====================================================================

    cc_min_area_frac: float = 0.01
    """Connected components smaller than this fraction of total grid area
    are removed. Eliminates isolated noise pixels in the mask.
    Higher → more aggressive cleanup (may remove thin features).
    Typical: 0.01–0.05."""

    morph_erode_iters: int = 1
    """Binary erosion iterations applied to segment masks in legacy
    postprocessing (_postprocess_semantic). Shrinks mask boundaries.
    NOTE: Only affects SEMANTIC and DEPTH_SEMANTIC modes, NOT CONTOUR.
    Typical: 0–2."""

    morph_close_iters: int = 3
    """Binary closing iterations (dilation followed by erosion). Fills narrow
    gaps and smooths mask boundaries. Applied in both legacy modes and
    CONTOUR's _cleanup_contour_mask. Higher → smoother boundaries but may
    merge nearby segments. Typical: 1–3."""

    boundary_refine_width: int = 1
    """Pixel width of the boundary zone for feature-based refinement in
    DEPTH_SEMANTIC mode. Pixels within this distance of a segment boundary
    are reassigned based on nearest cluster centroid in feature space.
    Higher → wider refinement zone. Typical: 1–3."""

    # =====================================================================
    # Contour / Level-Set Mode
    # (Controls the full CONTOUR pipeline)
    # =====================================================================

    # --- Level-set field construction ---

    contour_depth_weight: float = 0.4
    """Contribution of depth to the level-set field φ(u,v).

    Depth is the primary geometric cue: closer points → FG, farther → BG.
    The depth signal is further scaled by a confidence factor based on the
    depth range (flat scenes with small depth range get lower confidence).

    Higher → depth-dominant segmentation (good for scenes with clear
    depth separation). Lower → rely more on color/normal cues.
    If your scene has a foreground object clearly in front of a background,
    increase this. If depth is ambiguous (e.g., wall+poster), decrease it.
    Typical range: 0.2–0.8."""

    contour_color_weight: float = .05
    """Contribution of local color variance to the level-set field.

    Computed as the local variance of the SH DC color grid over a 5×5
    neighborhood. Regions with high color variance → more likely FG
    (objects tend to have more texture detail than flat backgrounds).

    Higher → textured regions become FG. Problematic for textured
    backgrounds (e.g., patterned wallpaper) or uniform objects.
    Typical range: 0.1–0.5."""

    contour_normal_weight: float = 0.5
    """Contribution of surface normal variance to the level-set field.

    Computed as the local variance of finite-difference normals over a
    5×5 neighborhood. High curvature regions → more likely FG.

    Set to 0 to disable (saves computation). Useful for scenes where
    the foreground has complex geometry (curved objects) vs flat background.
    Typical range: 0.0–0.3."""

    # --- Level-set smoothing ---

    contour_smooth_iters: int = 2
    """Number of Laplacian smoothing passes on the level-set field φ.

    Each iteration averages φ with its 4-connected neighbors, weighted
    by contour_smooth_lambda. This regularizes the contour, preventing
    jagged or noisy boundaries.

    Higher → smoother contour (may lose thin features like poles/antennas).
    Lower → more faithful to raw cues (may have noisy boundaries).
    Effective smoothing ≈ smooth_iters × smooth_lambda.
    Typical range: 3–10."""

    contour_smooth_lambda: float = 0.25
    """Step size for each Laplacian smoothing iteration.

    At each iteration: φ ← φ + λ(mean_neighbors − φ).
    λ=0: no smoothing. λ=1: full replacement with neighbor mean.

    Keep this moderate to avoid over-smoothing. The product
    (smooth_iters × smooth_lambda) controls total diffusion.
    Typical range: 0.1–0.5."""

    # --- Mask extraction & morphological cleanup ---

    contour_boundary_width: int = 2
    """Width (in control points) of the soft transition zone for
    complementary opacity masks.

    NOTE: Only used in the full-grid clone path (_build_complementary_
    opacity_masks). Has NO effect when using subdivision-based resampling
    (_resample_grid_to_mask), since that path uses hard binary masks.

    Controls the sigmoid scale: scale = boundary_width / 3.
    Higher → wider soft transition. Typical: 2–5."""

    contour_min_fg_frac: float = 0.01
    """Minimum allowed foreground fraction of the total control grid.

    If the raw mask has less FG than this threshold, the level-set
    threshold is automatically lowered to capture more (multiplied by 1.5×).
    Prevents degenerate decompositions where FG is nearly empty.
    Typical range: 0.03–0.15."""

    contour_max_fg_frac: float = 0.85
    """Maximum allowed foreground fraction of the total control grid.

    If FG exceeds this, the threshold is raised to shrink FG (divided
    by 0.8×). Prevents degenerate decompositions where BG is too small
    to represent the actual background.
    Typical range: 0.70–0.95."""

    contour_use_morphological: bool = True
    """Enable scipy-based morphological cleanup of the binary contour mask.

    When True, the raw φ > 0 mask is processed through:
      1. Hole filling (binary_fill_holes)
      2. Small connected component removal (< cc_min_area_frac)
      3. Small interior BG hole filling (< 10% of grid, not touching border)
      4. Dilation (contour_dilate_iters)
      5. Closing (morph_close_iters)

    When False, the raw thresholded mask is used directly. Disable for
    debugging or if scipy is unavailable."""

    contour_dilate_iters: int = 1
    """Binary dilation iterations applied to the FG mask during cleanup.

    Expands the FG region outward by this many pixels. Useful for
    capturing boundary control points that are 'on the edge' of the
    foreground. Higher → FG grows, captures more border region.
    Set to 0 for no dilation. Typical range: 0–2."""

    # --- Complementary opacity (full-grid clone path only) ---

    contour_bg_opacity_inside: float = 0.01
    """Minimum BG opacity inside the FG region (floor value).

    Only used in the full-grid clone path (_build_complementary_opacity_
    masks), NOT in subdivision-based resampling.

    Prevents zero gradients in the BG surface's FG region, allowing
    the boundary to shift during training. Typical: 0.005–0.05."""

    contour_fg_opacity_outside: float = 0.01
    """Minimum FG opacity outside the FG region (floor value).

    Only used in the full-grid clone path. Same rationale as above but
    for the FG surface's BG region. Typical: 0.005–0.05."""
    # =====================================================================
    # Hybrid Depth-Contour Mode (HYBRID)
    # Variational level-set initialized from continuous depth evidence,
    # with edge-aware anisotropic smoothing from color/normal gradients.
    # =====================================================================

    # --- Depth Evidence ---
    hybrid_depth_sharpness: float = 3.0
    hybrid_depth_method: str = "otsu"       # "otsu", "histogram_gap", "quantile"
    hybrid_local_sigma_kernel: int = 5

    # --- Edge-Aware Smoothness Modulation ---
    hybrid_edge_from_color: bool = True
    hybrid_edge_from_normals: bool = True
    hybrid_edge_sensitivity: float = 0.5    # β in g = 1/(1 + β|∇I|²)

    # --- Variational Level-Set Solver ---
    hybrid_lambda_data: float = 1.0
    hybrid_lambda_smooth: float = 0.3
    hybrid_lambda_area: float = 0.05
    hybrid_target_fg_fraction: float = 0.3
    hybrid_solver_iterations: int = 50
    hybrid_solver_dt: float = 0.1
    hybrid_solver_epsilon: float = 0.5      # Heaviside width

    # --- Boundary Handling ---
    hybrid_confidence_threshold: float = 0.8
    hybrid_boundary_dilate: int = 1         # Expand FG mask at boundary
# ======================================================================
# Controller
# ======================================================================

class WarmupDecompositionController:
    """
    Stateful controller. One instance per training run.

    Usage in train.py:
        controller = WarmupDecompositionController(cfg)
        for iter, camera, psnr in training_loop:
            if controller.phase == DecompPhase.READY:
                model = controller.decompose(model, config, args, train_cam_uids)
                continue

            controller.update(iter, model, camera, psnr)
    """

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.phase = DecompPhase.WARMUP
        self._depth_buffer: List[torch.Tensor] = []
        self._psnr_history: List[float] = []

    # ------------------------------------------------------------------
    # Called every training iteration
    # ------------------------------------------------------------------
    def update(
            self,
            iteration: int,
            model: "MultiSurfaceSplineModel",
            camera,
            psnr: float = 0.0,
    ) -> bool:
        self._psnr_history.append(psnr)

        if self.phase == DecompPhase.WARMUP:
            if (iteration >= self.cfg.warmup_iters and
                    psnr >= self.cfg.min_psnr_to_decompose):
                print(f"[Decomp] Warmup complete at iter {iteration}, "
                      f"PSNR={psnr:.2f}. Starting depth collection.")
                self.phase = DecompPhase.READY if self.cfg.n_depth_views == 0 else DecompPhase.DEPTH_COLLECTION
            if self.phase == DecompPhase.READY:
                return True

        elif self.phase == DecompPhase.DEPTH_COLLECTION:
            self._record_depth(model, camera)
            if len(self._depth_buffer) >= self.cfg.n_depth_views:
                print(f"[Decomp] Collected {len(self._depth_buffer)} depth views. "
                      f"Ready to decompose.")
                self.phase = DecompPhase.READY
                return True

        return False

    def _record_depth(self, model: "MultiSurfaceSplineModel", camera):
        surface = model.surfaces[0]
        H, W = surface.state.H, surface.state.W

        with torch.no_grad():
            depth = surface.uv_depth()
            Us, Vs = surface.state.Us, surface.state.Vs

            depth_2d = depth.reshape(Us, Vs, 1).permute(2, 0, 1).unsqueeze(0)
            depth_ctrl = F.interpolate(
                depth_2d.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze().cpu()

            self._depth_buffer.append(depth_ctrl)

    # ------------------------------------------------------------------
    # decompose — dispatches to contour mode or legacy modes
    # ------------------------------------------------------------------
    def reset_depth_buffer(self):
        self._depth_buffer = []
    def decompose(
            self,
            model: "MultiSurfaceSplineModel",
            config,
            args,
    ) -> "MultiSurfaceSplineModel":
        assert self.phase == DecompPhase.READY, \
            f"decompose() called in wrong phase: {self.phase}"

        surface = model.surfaces[0]
        H, W = surface.state.H, surface.state.W
        device = surface.device
        train_cam_uids = surface.train_cam_uids

        agg_depth = self._aggregate_depth(H, W, device)

        # CONTOUR mode: full-grid clone with complementary opacity
        if self.cfg.segmentation_mode == SegmentationMode.CONTOUR:
            # return self._decompose_contour(
            return self._decompose_contour(
                model, surface, agg_depth, config, args, train_cam_uids, device
            )
        if self.cfg.segmentation_mode == SegmentationMode.HYBRID:
            return self._decompose_hybrid(
                model, surface, agg_depth, config, args, train_cam_uids, device
            )

        # Legacy modes: bounding-box based
        seg_result = self._segment(surface, agg_depth, device)

        surfaces, labels = [], []
        for mask, label, is_bg in zip(
                seg_result["masks"],
                seg_result["labels"],
                seg_result["is_background"]
        ):
            sub = self._build_subsurface(
                surface, mask, label, is_bg, config, args, train_cam_uids
            )
            surfaces.append(sub)
            labels.append(label)

        from ..fitting.nurbs_from_pointcloud import DecompositionMode
        new_model = model.__class__(
            surfaces=surfaces,
            labels=labels,
            decomposition_mode=DecompositionMode.BACKGROUND_OBJECT,
            device=device,
            setup_training=False,
        )

        self.phase = DecompPhase.DECOMPOSED
        print(f"[Decomp] Done. New model: {new_model}")
        return new_model

    # ==================================================================
    #  NEW: CONTOUR-BASED DECOMPOSITION
    # ==================================================================

        # ======================================================================
        # 4. Add the method (delegates to the pipeline)
        # ======================================================================

    def _decompose_hybrid(
            self,
            model,
            surface,
            agg_depth,
            config,
            args,
            train_cam_uids,
            device,
    ):
        """Hybrid depth-contour decomposition. See HybridLevelSetPipeline."""
        from model.modules.decompose.level_set import _decompose_hybrid
        print(f"[Hybrid Decomp] Starting hybrid depth-contour decomposition.")
        return _decompose_hybrid(
            self, model, surface, agg_depth, config, args,
            train_cam_uids, device
        )
    def _decompose_contour(
            self,
            model: "MultiSurfaceSplineModel",
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            config,
            args,
            train_cam_uids: list,
            device: str,
    ) -> "MultiSurfaceSplineModel":
        """
        Contour-based decomposition v4: Mask-guided grid resampling.

        Each surface gets a NEW rectangular grid where every control point
        is sourced from the mask-positive region of the parent. No dead zones,
        no opacity suppression, no bbox artifacts.

        How it works:
        1. Compute FG/BG masks via level-set (unchanged)
        2. For each mask, resample the parent's control features:
           - Row pass: extract mask-positive columns, interpolate to tgt_W
           - Column pass: extract valid rows, interpolate to tgt_H
        3. Result: compact grids with no wasted control points
        """
        H, W = surface.state.H, surface.state.W
        print(f"[Contour Decomp v4] Starting mask-guided resampling on {H}×{W} grid")

        # ============ STEP 1-3: Level-set → binary mask (unchanged) ============
        phi = self._build_level_set_field(surface, agg_depth, H, W, device)
        phi = self._smooth_level_set(phi, H, W)
        fg_mask = self._extract_contour_mask(phi, H, W, device)

        # Optional: visualization
        from matplotlib import pyplot as plt
        sh_dc_raw = surface.spherical_harmonics.sh_dc.control_features.detach()
        from utils.sh_utils import SH2RGB
        color_grid = SH2RGB(sh_dc_raw.view(H, W, 3)).clamp(0, 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].set_title("FG Mask")
        axes[0, 0].imshow(fg_mask.cpu(), cmap='gray')
        axes[0, 1].set_title("FG Masked Color")
        axes[0, 1].imshow((color_grid * fg_mask.unsqueeze(-1)).cpu())
        axes[1, 0].set_title("BG Mask")
        axes[1, 0].imshow((~fg_mask).cpu(), cmap='gray')
        axes[1, 1].set_title("BG Masked Color")
        axes[1, 1].imshow((color_grid * (~fg_mask).unsqueeze(-1)).cpu())
        plt.tight_layout()
        plt.show()

        # ============ Validate + adjust mask coverage ============
        fg_frac = fg_mask.float().mean().item()
        print(f"[Contour Decomp v4] FG fraction: {fg_frac:.1%}")

        if fg_frac < self.cfg.contour_min_fg_frac:
            print(f"[Contour Decomp v4] WARNING: FG too small ({fg_frac:.1%}). Adjusting.")
            phi_sorted = phi.flatten().sort().values
            target_idx = int((1 - self.cfg.contour_min_fg_frac * 1.5) * phi_sorted.numel())
            new_threshold = phi_sorted[target_idx].item()
            fg_mask = phi > new_threshold
            fg_mask = self._cleanup_contour_mask(fg_mask, H, W, device)
            fg_frac = fg_mask.float().mean().item()

        if fg_frac > self.cfg.contour_max_fg_frac:
            print(f"[Contour Decomp v4] WARNING: FG too large ({fg_frac:.1%}). Adjusting.")
            phi_sorted = phi.flatten().sort().values
            target_idx = int((1 - self.cfg.contour_max_fg_frac * 0.8) * phi_sorted.numel())
            new_threshold = phi_sorted[target_idx].item()
            fg_mask = phi > new_threshold
            fg_mask = self._cleanup_contour_mask(fg_mask, H, W, device)
            fg_frac = fg_mask.float().mean().item()

        bg_mask = ~fg_mask

        # ============ STEP 4: Resample grids guided by masks ============
        bg_surface = self._resample_grid_to_mask(
            surface, bg_mask, "background", is_background=True,
            config=config, args=args, train_cam_uids=train_cam_uids,
        )
        fg_surface = self._resample_grid_to_mask(
            surface, fg_mask, "object_0", is_background=False,
            config=config, args=args, train_cam_uids=train_cam_uids,
        )

        # ============ STEP 5: Diagnostics ============
        bg_H, bg_W = bg_surface.state.H, bg_surface.state.W
        fg_H, fg_W = fg_surface.state.H, fg_surface.state.W
        total_new = bg_H * bg_W + fg_H * fg_W
        total_orig = H * W
        print(f"\n[Contour Decomp v4 Summary]")
        print(f"  Original: {H}×{W} = {total_orig} control points")
        print(f"  BG: {bg_H}×{bg_W} = {bg_H * bg_W} "
              f"({bg_H * bg_W / total_orig:.1%} of original)")
        print(f"  FG: {fg_H}×{fg_W} = {fg_H * fg_W} "
              f"({fg_H * fg_W / total_orig:.1%} of original)")
        print(f"  Total: {total_new} ({total_new / total_orig:.1%} of original)")

        # ============ STEP 6: Assemble new model ============
        from ..fitting.nurbs_from_pointcloud import DecompositionMode
        new_model = model.__class__(
            surfaces=[bg_surface, fg_surface],
            labels=["background", "object_0"],
            decomposition_mode=DecompositionMode.BACKGROUND_OBJECT,
            device=device,
            setup_training=False,
        )

        self.phase = DecompPhase.DECOMPOSED
        print(f"[Contour Decomp v4] Done. New model: {new_model}")
        return new_model

    def _clone_surface_with_eval_mask(
            self,
            parent: "SplineModel",
            eval_mask: torch.Tensor,  # [H, W] float in [0, 1]
            label: str,
            is_background: bool,
            config,
            args,
            train_cam_uids: list,
    ) -> "SplineModel":
        """
        Clone the parent surface with a POST-INTERPOLATION opacity mask.

        Critical difference from previous approaches:
        - We do NOT modify control-point opacity (B-splines defeat this)
        - We do NOT crop the grid (concave masks defeat this)
        - Instead, we store an eval_mask on the surface that is applied
          AFTER B-spline interpolation, at the Gaussian sampling level

        The eval_mask is stored as a [H, W] tensor on the surface.
        During forward pass, it's interpolated to [Us, Vs] using the
        same B-spline basis, then multiplied into the opacity output.

        This works because:
        1. The mask multiplication happens AFTER interpolation
        2. B-spline smoothing of the mask actually helps (smooth transitions)
        3. No control-point modification means no B-spline leaking
        4. Full grid means both surfaces benefit from full warmup quality
        """
        from model.modules.KnotSurface import SplineModel

        H, W = parent.state.H, parent.state.W
        degree = parent.state.degree
        device = parent.device

        # --- Build geomdl surfaces (unchanged from warmup) ---
        pos_ctrl = parent.position.control_features.detach().clone().view(H, W, 3)
        geo_surf = self._make_geomdl_surface(pos_ctrl, H, W, degree)

        sh_dc = parent.spherical_harmonics.sh_dc.control_features.detach().clone().view(H, W, 3)
        rgb_ctrl = (sh_dc * 0.28209479177387814 + 0.5).clamp(0, 1)
        rgb_surf = self._make_geomdl_surface(rgb_ctrl, H, W, degree)

        spline_model = SplineModel(
            surf=geo_surf,
            surf_rgb=rgb_surf,
            config=config,
            args=args,
            spatial_lr_scale=parent.spatial_lr_scale,
            train_cam_uids=train_cam_uids,
            late_init=False,
            surf_uid=hash(label) % 1000,
            skip_opt=True,
            label=label,
            is_background=is_background,
        )

        # --- Copy ALL features from parent (exact, no modification) ---
        self._transfer_all_features(parent, spline_model, H, W)

        # --- Store the evaluation mask ---
        # This is the key: the mask is NOT a control-point feature.
        # It's stored separately and applied post-interpolation.
        # We store it as raw values (not as a ControlFeature) to avoid
        # it being modified by B-spline interpolation in unexpected ways.
        spline_model.register_buffer(
            '_decomp_eval_mask',
            eval_mask.clone().to(device)  # [H, W]
        )
        spline_model._has_decomp_mask = True

        print(f"  [{label}] Full grid clone {H}×{W} with eval mask "
              f"(active: {(eval_mask > 0.5).float().mean():.1%})")

        return spline_model

    def _resample_grid_to_mask(
            self,
            parent: "SplineModel",
            mask: torch.Tensor,  # [H, W] bool
            label: str,
            is_background: bool,
            config,
            args,
            train_cam_uids: list,
    ) -> "SplineModel":
        """
        Mask-guided grid resampling via adaptive subdivision.
        """
        from modules.KnotSurface import SplineModel

        H, W = parent.state.H, parent.state.W
        degree = parent.state.degree
        device = parent.device

        # ========== Target grid size ==========
        mask_frac = mask.float().mean().item()

        if is_background:
            tgt_H = max(degree + 2, int(H * max(mask_frac, 0.5)))
            tgt_W = max(degree + 2, int(W * max(mask_frac, 0.5)))
        else:
            tgt_H = max(degree + 2, int(H * max(mask_frac ** 0.5, 0.4)))
            tgt_W = max(degree + 2, int(W * max(mask_frac ** 0.5, 0.4)))

        print(f"  [{label}] Mask fraction: {mask_frac:.1%}, "
              f"Target grid: {tgt_H}×{tgt_W} (from {H}×{W})")

        # ========== Gather all feature grids from parent ==========
        feature_grids = self._gather_all_feature_grids(parent, H, W)

        # Separate metadata from actual tensor grids
        metadata = {k: v for k, v in feature_grids.items() if isinstance(v, str) or isinstance(v, int)}
        tensor_grids = {k: v.detach().clone() for k, v in feature_grids.items() if isinstance(v, torch.Tensor)}

        # ========== Phase 1: Row-wise resampling ==========
        row_valid = mask.any(dim=1)  # [H]
        valid_row_indices = torch.where(row_valid)[0]

        if valid_row_indices.numel() == 0:
            raise ValueError(f"Empty mask for '{label}'")

        intermediate_grids = {}
        for feat_name, grid in tensor_grids.items():
            C = grid.shape[-1]
            intermediate = torch.zeros(valid_row_indices.numel(), tgt_W, C, device=device)
            if feat_name == 'scaling':
                grid[~mask] = grid[~mask].clamp(max=-6)  # Suppress scaling in BG region
                grid = (grid.exp() / 2.0).log()


                # grid = grid.clamp(min=-12)  # Ensure no extreme scaling anywhere

                # if is_background:
                    # Background: negative values → smaller scaling, positive → keep
                    # grid[~mask] = grid[~mask].clamp(max=-6)  # Suppress scaling in BG region
                # else:
                #     grid[~mask] = grid[~mask].clamp(max=-6)  # Suppress scaling in BG region

            for out_row, src_row in enumerate(valid_row_indices):
                row_mask = mask[src_row]  # [W] bool
                active_cols = torch.where(row_mask)[0]

                if active_cols.numel() == 0:
                    continue

                active_values = grid[src_row, active_cols]  # [n_active, C]
                src_positions = active_cols.float() / max(W - 1, 1)

                tgt_positions = torch.linspace(
                    src_positions[0].item(),
                    src_positions[-1].item(),
                    tgt_W, device=device
                )

                intermediate[out_row] = self._interp_1d(
                    src_positions, active_values, tgt_positions
                )

            intermediate_grids[feat_name] = intermediate

        n_valid_rows = valid_row_indices.numel()

        # ========== Phase 2: Column-wise resampling ==========
        src_row_positions = valid_row_indices.float() / max(H - 1, 1)

        tgt_row_positions = torch.linspace(
            src_row_positions[0].item(),
            src_row_positions[-1].item(),
            tgt_H, device=device
        )

        final_grids = {}
        for feat_name, inter_grid in intermediate_grids.items():
            C = inter_grid.shape[-1]
            final = torch.zeros(tgt_H, tgt_W, C, device=device)

            for col in range(tgt_W):
                col_values = inter_grid[:, col, :]  # [n_valid_rows, C]
                final[:, col, :] = self._interp_1d(
                    src_row_positions, col_values, tgt_row_positions
                )

            final_grids[feat_name] = final

        # Carry metadata forward
        final_grids.update(metadata)

        # ========== Phase 3: Build SplineModel ==========
        # CRITICAL: Use the resampled position and color grids directly
        # so that SplineModel.__init__ computes correct tangents/scaling/rotation
        # from the ACTUAL data that will be used.
        pos_grid = final_grids['position'].detach().clone()  # [tgt_H, tgt_W, 3]
        sh_dc_grid = final_grids['sh_dc'].detach().clone()
        from utils.sh_utils import SH2RGB
        rgb_grid = SH2RGB(sh_dc_grid[..., :3]).clamp(0, 1).detach().clone()
        geo_surf = self._make_geomdl_surface(pos_grid, tgt_H, tgt_W, degree)
        rgb_surf = self._make_geomdl_surface(rgb_grid, tgt_H, tgt_W, degree)
        assert len(geo_surf.knotvector_u) == tgt_H + degree + 1, \
            f"Knot U: {len(geo_surf.knotvector_u)} != {tgt_H + degree + 1}"
        assert len(geo_surf.knotvector_v) == tgt_W + degree + 1, \
            f"Knot V: {len(geo_surf.knotvector_v)} != {tgt_W + degree + 1}"

        spline_model = SplineModel(
            surf=geo_surf,
            surf_rgb=rgb_surf,
            config=config,
            args=args,
            spatial_lr_scale=parent.spatial_lr_scale,
            train_cam_uids=train_cam_uids,
            late_init=False,
            surf_uid=0 if is_background else 1,
            skip_opt=True,
            label=label,
            is_background=is_background,
            feature_grids=final_grids,  # Pass all resampled grids for later assignment
        )

        # === Defensive: verify all module names are strings ===
        for module in spline_model.control_list:
            # if module.control_features is None:
            #     continue
            if not isinstance(module.name, str):
                # print(f"  [WARNING] Module name is {type(module.name)}: {module.name}")
                # Attempt to recover
                if module is spline_model.position:
                    module.name = f"xyz_{spline_model.surf_uid}"
                elif module is spline_model.spherical_harmonics.sh_dc:
                    module.name = f"f_dc_{spline_model.surf_uid}"
                elif module is spline_model.spherical_harmonics.sh_rest:
                    module.name = f"f_rest_{spline_model.surf_uid}"
                elif module is spline_model.scaling:
                    module.name = f"scaling_{spline_model.surf_uid}"
                elif module is spline_model.rotation:
                    module.name = f"rotation_{spline_model.surf_uid}"
                elif module is spline_model.opacity:
                    module.name = f"opacity_{spline_model.surf_uid}"
                elif hasattr(module, 'name') and module is spline_model.weights:
                    module.name = f"weights_{spline_model.surf_uid}"
                print(f"  [FIXED] Module name set to: {module.name}")
        # ========== Phase 4: Overwrite features with resampled values ==========
        self._apply_resampled_features(spline_model, final_grids, tgt_H, tgt_W)

        # ========== Phase 5: CRITICAL — Force full recompute ==========
        # After overwriting features, ALL cached values are stale.
        # The basis functions are fine (they depend on knots/UV, not features),
        # but any cached interpolation results, tangents, etc. must be cleared.
        torch.cuda.synchronize()
        spline_model.invalidate_all_caches(force=True)

        # Validate: ensure basis dimensions match control grid
        bu_shape = spline_model.basis.bu.shape
        bv_shape = spline_model.basis.bv.shape
        expected_H = spline_model.state.H
        expected_W = spline_model.state.W

        if spline_model.state.full_basis:
            # Full grid basis: bu is [Us, Vs, H], bv is [Us, Vs, W]
            assert bu_shape[-1] == expected_H, \
                f"Basis U dim mismatch: bu[...,-1]={bu_shape[-1]} != H={expected_H}"
            assert bv_shape[-1] == expected_W, \
                f"Basis V dim mismatch: bv[...,-1]={bv_shape[-1]} != W={expected_W}"
        else:
            # Separable basis: bu is [Us, H], bv is [Vs, W]
            assert bu_shape[-1] == expected_H, \
                f"Basis U dim mismatch: bu[-1]={bu_shape[-1]} != H={expected_H}"
            assert bv_shape[-1] == expected_W, \
                f"Basis V dim mismatch: bv[-1]={bv_shape[-1]} != W={expected_W}"

        # Validate feature shapes match
        for module in spline_model.control_list:
            if module.control_features is None:
                continue
            total_elements = module.control_features.shape[0]
            expected_elements = expected_H * expected_W
            # SH rest may have a different layout
            if hasattr(module, 'num_sh_coeffs') and module.num_sh_coeffs > 1:
                # Could be [H*W*num_coeffs, 3] or [H*W, num_coeffs*3]
                if total_elements != expected_elements and total_elements != expected_elements * module.num_sh_coeffs:
                    print(f"  [WARNING] {module.name}: shape[0]={total_elements}, "
                          f"expected {expected_elements} or {expected_elements * module.num_sh_coeffs}")
            else:
                if total_elements != expected_elements:
                    print(f"  [WARNING] {module.name}: shape[0]={total_elements}, "
                          f"expected {expected_elements}")

        # Quick smoke test: can we evaluate the surface?
        # try:
        #     with torch.no_grad():
        #         test_xyz = spline_model.get_xyz
        #         assert test_xyz.shape == (spline_model.state.Us * spline_model.state.Vs, 3), \
        #             f"XYZ shape: {test_xyz.shape}"
        #         assert not test_xyz.isnan().any(), "XYZ contains NaN"
        #         assert not test_xyz.isinf().any(), "XYZ contains Inf"
        #         print(f"  [{label}] Smoke test passed: {test_xyz.shape[0]} Gaussians, "
        #               f"bbox=[{test_xyz.min(0).values.tolist()}, {test_xyz.max(0).values.tolist()}]")
        # except Exception as e:
        #     print(f"  [{label}] SMOKE TEST FAILED: {e}")
        #     raise RuntimeError(
        #         f"Resampled surface '{label}' failed validation. "
        #         f"Grid: {tgt_H}×{tgt_W}, H={expected_H}, W={expected_W}, "
        #         f"Bu={bu_shape}, Bv={bv_shape}"
        #     ) from e
        #
        # print(f"  [{label}] Resampled grid: {tgt_H}×{tgt_W} "
        #       f"({tgt_H * tgt_W} ctrl pts, {tgt_H * tgt_W / (H * W):.1%} of original)")

        return spline_model

    def _gather_all_feature_grids(
            self,
            parent: "SplineModel",
            H: int, W: int,
    ) -> dict:
        """
        Extract all control-point feature grids from parent surface.
        Returns dict of {name: [H, W, C] tensor} plus metadata keys prefixed with '_'.
        """
        grids = {}

        grids['position'] = parent.position.control_features.data.view(H, W, 3).clone()

        sh_dc_cf = parent.spherical_harmonics.sh_dc.control_features.data.clone()
        sh_dc_ch = sh_dc_cf.shape[-1]
        grids['sh_dc'] = sh_dc_cf.view(H, W, sh_dc_ch)

        # SH rest: figure out the layout
        sh_rest_cf = parent.spherical_harmonics.sh_rest.control_features.data.clone()
        sh_rest_total = sh_rest_cf.shape[0]
        sh_rest_last = sh_rest_cf.shape[-1] if sh_rest_cf.dim() > 1 else 1

        if sh_rest_total == H * W:
            # Layout: [H*W, (shc-1)*3]
            grids['sh_rest'] = sh_rest_cf.view(H, W, sh_rest_last)
            grids['_sh_rest_layout'] = 'flat'
            grids['_sh_rest_last_dim'] = sh_rest_last
        else:
            # Layout: [H*W*num_coeffs, 3] — flatten to [H, W, num_coeffs*3]
            num_coeffs = parent.spherical_harmonics.sh_rest.num_sh_coeffs
            grids['sh_rest'] = sh_rest_cf.view(H, W, num_coeffs * 3)
            grids['_sh_rest_layout'] = 'folded'
            grids['_sh_rest_num_coeffs'] = num_coeffs

        if parent.refine_opacity_active:
            grids['opacity'] = parent.opacity.control_features.view(H, W, 1).clone()#.sigmoid().view(H, W, 1).clone()

        if parent.refine_scales_active:
            scale_cf = parent.scaling.control_features # [H*W, scale_ch]
            scale_ch = scale_cf.shape[-1]
            grids['scaling'] = scale_cf.view(H, W, scale_ch).clone()

        if parent.refine_rotations_active:
            grids['rotation'] = parent.rotation.features.view(H, W, 4).clone()

        if (parent.refine_weights_active
                and parent.weights.control_features is not None):
            w_cf = parent.weights.control_features.data
            w_ch = w_cf.shape[-1]
            grids['weights'] = w_cf.view(H, W, w_ch).clone()

        return grids

    def _interp_1d(
            self,
            src_positions: torch.Tensor,  # [N] monotonically increasing
            src_values: torch.Tensor,  # [N, C]
            tgt_positions: torch.Tensor,  # [M]
    ) -> torch.Tensor:
        """
        1D linear interpolation with extrapolation clamping.

        Given source samples at non-uniform positions, produce values
        at target positions via piecewise linear interpolation.

        Handles edge cases:
        - tgt outside src range → clamp to nearest source value
        - Single source point → replicate
        - Duplicate source positions → take first
        """
        N = src_positions.shape[0]
        M = tgt_positions.shape[0]
        C = src_values.shape[-1]
        device = src_values.device

        if N == 0:
            return torch.zeros(M, C, device=device)

        if N == 1:
            return src_values[0:1].expand(M, -1).clone()

        # Clamp target positions to source range
        tgt_clamped = tgt_positions.clamp(src_positions[0], src_positions[-1])

        # Find insertion indices: for each target, which source interval it falls in
        # searchsorted gives index where tgt would be inserted to maintain sorted order
        idx_right = torch.searchsorted(src_positions, tgt_clamped, right=True)
        idx_right = idx_right.clamp(1, N - 1)  # ensure valid pair
        idx_left = idx_right - 1

        # Compute interpolation weights
        pos_left = src_positions[idx_left]
        pos_right = src_positions[idx_right]
        span = (pos_right - pos_left).clamp(min=1e-8)
        t = (tgt_clamped - pos_left) / span
        t = t.clamp(0, 1).unsqueeze(-1)  # [M, 1]

        # Interpolate
        val_left = src_values[idx_left]  # [M, C]
        val_right = src_values[idx_right]  # [M, C]
        result = (1 - t) * val_left + t * val_right

        return result
    def _interp_1d_rotation(
            self,
            src_positions: torch.Tensor,  # [N] monotonically increasing
            src_quats: torch.Tensor,  # [N, 4] unit quaternions
            tgt_positions: torch.Tensor,  # [M]
    ) -> torch.Tensor:
        """
        1D SLERP interpolation for quaternions.
        Falls back to linear + normalize for efficiency when quaternions are close.
        """
        N = src_positions.shape[0]
        M = tgt_positions.shape[0]
        device = src_quats.device

        if N == 0:
            # Return identity quaternion
            out = torch.zeros(M, 4, device=device)
            out[:, 0] = 1.0
            return out

        if N == 1:
            return F.normalize(src_quats[0:1].expand(M, -1).clone(), dim=-1)

        tgt_clamped = tgt_positions.clamp(src_positions[0], src_positions[-1])
        idx_right = torch.searchsorted(src_positions, tgt_clamped, right=True)
        idx_right = idx_right.clamp(1, N - 1)
        idx_left = idx_right - 1

        pos_left = src_positions[idx_left]
        pos_right = src_positions[idx_right]
        span = (pos_right - pos_left).clamp(min=1e-8)
        t = ((tgt_clamped - pos_left) / span).clamp(0, 1)

        q0 = F.normalize(src_quats[idx_left], dim=-1)
        q1 = F.normalize(src_quats[idx_right], dim=-1)

        # Ensure shortest path
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)

        # Linear interpolation + renormalize (NLERP)
        # Good enough for neighboring control points, much faster than SLERP
        result = (1 - t.unsqueeze(-1)) * q0 + t.unsqueeze(-1) * q1
        result = F.normalize(result, dim=-1, eps=1e-6)

        return result

    def _apply_resampled_features(
            self,
            child: "SplineModel",
            final_grids: dict,
            tgt_H: int, tgt_W: int,
    ):
        """
        Overwrite the SplineModel's control features with the resampled grids.

        SplineModel.__init__ already created default features; we replace them
        with our mask-guided resampled values.
        """
        with torch.no_grad():
            # Position
            if 'position' in final_grids:
                pos = final_grids['position'].detach().clone().reshape(-1, 3)
                assert child.position.control_features.data.shape == pos.shape, \
                    f"Position shape mismatch: {child.position.control_features.data.shape} vs {pos.shape}"
                child.position.control_features.data.copy_(pos)

            # SH DC
            if 'sh_dc' in final_grids:
                sh_dc = final_grids['sh_dc'].detach().clone()
                c_dc = child.spherical_harmonics.sh_dc.control_features.data
                c_ch = c_dc.shape[-1]
                s_ch = sh_dc.shape[-1]
                min_ch = min(c_ch, s_ch)
                c_dc_grid = c_dc.view(tgt_H, tgt_W, c_ch)
                c_dc_grid[..., :min_ch] = sh_dc[..., :min_ch]
                child.spherical_harmonics.sh_dc.control_features.data.copy_(
                    c_dc_grid.reshape(-1, c_ch)
                )

            # SH Rest
            if 'sh_rest' in final_grids:
                sh_rest = final_grids['sh_rest'].detach().clone()  # [tgt_H, tgt_W, X]
                c_rest = child.spherical_harmonics.sh_rest.control_features.data
                c_total = c_rest.shape[0]
                c_last = c_rest.shape[-1] if c_rest.dim() > 1 else 1

                s_feat = sh_rest.shape[-1]

                if c_total == tgt_H * tgt_W:
                    # Layout: [tgt_H*tgt_W, (shc-1)*3]
                    min_f = min(c_last, s_feat)
                    c_grid = c_rest.view(tgt_H, tgt_W, c_last)
                    c_grid[..., :min_f] = sh_rest[..., :min_f]
                    child.spherical_harmonics.sh_rest.control_features.data.copy_(
                        c_grid.reshape(tgt_H * tgt_W, c_last)
                    )
                else:
                    # Layout: [tgt_H*tgt_W*num_coeffs, 3]
                    c_num = child.spherical_harmonics.sh_rest.num_sh_coeffs
                    # sh_rest is [tgt_H, tgt_W, num_coeffs*3], reshape to match
                    c_grid = c_rest.view(tgt_H, tgt_W, c_num, 3)
                    s_num = s_feat // 3
                    min_n = min(c_num, s_num)
                    s_grid = sh_rest.view(tgt_H, tgt_W, s_num, 3)
                    c_grid[..., :min_n, :] = s_grid[..., :min_n, :]
                    child.spherical_harmonics.sh_rest.control_features.data.copy_(
                        c_grid.reshape(tgt_H * tgt_W * c_num, 3)
                    )

            # Opacity
            # if 'opacity' in final_grids and child.refine_opacity_active:
            #     opa = final_grids['opacity'].detach().clone().reshape(-1, 1)
            #     assert child.opacity.control_features.data.shape == opa.shape, \
            #         f"Opacity shape: {child.opacity.control_features.data.shape} vs {opa.shape}"
            #     child.opacity.control_features.data.copy_(opa)
            #
            # # Scaling
            if 'scaling' in final_grids and child.refine_scales_active:
                scale = final_grids['scaling'].detach().clone()
                s_ch = scale.shape[-1]
                c_ch = child.scaling.control_features.data.shape[-1]
                if s_ch == c_ch:
                    child.scaling.control_features.data.copy_(scale.reshape(-1, c_ch))
                else:
                    min_ch = min(s_ch, c_ch)
                    c_grid = child.scaling.control_features.data.view(tgt_H, tgt_W, c_ch).detach().clone()
                    c_grid[..., :min_ch] = scale[..., :min_ch]
                    child.scaling.control_features.data.copy_(c_grid.reshape(-1, c_ch))

            # Rotation
            if 'rotation' in final_grids and child.refine_rotations_active:
                rot = final_grids['rotation'].reshape(-1, 4).detach().clone()
                F.normalize(child.rotation.control_features.data.copy_(rot), dim=-1)

            # Weights
            if ('weights' in final_grids
                    and child.refine_weights_active
                    and child.weights.control_features is not None):
                w = final_grids['weights'].detach().clone()
                w_ch = w.shape[-1]
                child.weights.control_features.data.copy_(w.reshape(-1, w_ch))

        print(f"  [Resample Transfer] Applied {len(final_grids)} feature grids "
              f"to [{tgt_H}×{tgt_W}] for '{child.label}'")

    def _crop_surface_to_mask(
            self,
            parent: "SplineModel",
            mask: torch.Tensor,  # [H, W] bool — True = this surface owns this region
            label: str,
            is_background: bool,
            config,
            args,
            train_cam_uids: list,
    ) -> "SplineModel":
        """
        Crop the parent surface's control grid to the bounding box of `mask`,
        with degree-sized padding for B-spline support overlap.

        Each child surface gets a tight rectangular sub-grid, with uniform
        knot vectors. No resampling — control point values are copied exactly.
        """
        from model.modules.KnotSurface import SplineModel

        H, W = parent.state.H, parent.state.W
        degree = parent.state.degree
        device = parent.device

        # ========== STEP 1: Compute padded bounding box ==========
        rows_any = mask.any(dim=1)  # [H]
        cols_any = mask.any(dim=0)  # [W]

        if not rows_any.any() or not cols_any.any():
            raise ValueError(f"Empty mask for '{label}' — no positive entries")

        row_indices = torch.where(rows_any)[0]
        col_indices = torch.where(cols_any)[0]

        r0_raw = row_indices[0].item()
        r1_raw = row_indices[-1].item() + 1
        c0_raw = col_indices[0].item()
        c1_raw = col_indices[-1].item() + 1

        # Pad by degree for B-spline support coverage
        pad = degree
        r0 = max(0, r0_raw - pad)
        r1 = min(H, r1_raw + pad)
        c0 = max(0, c0_raw - pad)
        c1 = min(W, c1_raw + pad)

        # Ensure minimum size for valid B-spline
        min_ctrl = degree + 2
        if (r1 - r0) < min_ctrl:
            center_r = (r0_raw + r1_raw) // 2
            r0 = max(0, center_r - min_ctrl // 2)
            r1 = min(H, r0 + min_ctrl)
            r0 = max(0, r1 - min_ctrl)
        if (c1 - c0) < min_ctrl:
            center_c = (c0_raw + c1_raw) // 2
            c0 = max(0, center_c - min_ctrl // 2)
            c1 = min(W, c0 + min_ctrl)
            c0 = max(0, c1 - min_ctrl)

        sub_H = r1 - r0
        sub_W = c1 - c0

        print(f"  [{label}] Mask bbox: rows=[{r0_raw}:{r1_raw}], cols=[{c0_raw}:{c1_raw}]")
        print(f"  [{label}] Padded crop: [{r0}:{r1}, {c0}:{c1}] → {sub_H}×{sub_W} "
              f"(from {H}×{W}, {sub_H * sub_W / (H * W):.1%} of original)")

        # ========== STEP 2: Crop control features from parent ==========
        pos_ctrl = parent.position.control_features.detach().clone().view(H, W, 3)
        sub_pos = pos_ctrl[r0:r1, c0:c1].contiguous()  # [sub_H, sub_W, 3]

        sh_dc = parent.spherical_harmonics.sh_dc.control_features.detach().clone()
        sh_dc_grid = sh_dc.view(H, W, -1)
        sub_sh_dc = sh_dc_grid[r0:r1, c0:c1]
        from utils.sh_utils import SH2RGB
        sub_rgb = SH2RGB(sub_sh_dc[..., :3]).clamp(0, 1).contiguous()

        # ========== STEP 3: Use UNIFORM knot vectors for the sub-grid ==========
        # This is the critical fix: the knot vector MUST have exactly
        # sub_H + degree + 1 knots (and similarly for V).
        # A uniform clamped knot vector always satisfies this.

        # ========== STEP 3: PRESERVE KNOT VECTORS for the sub-grid ==========
        orig_knots_u = parent.knot_u()
        orig_knots_v = parent.knot_v()

        # Extract the exact parametric span for the cropped control points
        knots_u = self._crop_knot_vector(orig_knots_u, degree, H, r0, r1)
        knots_v = self._crop_knot_vector(orig_knots_v, degree, W, c0, c1)

        # ========== STEP 4: Build geomdl surfaces with explicit knots ==========
        geo_surf = self._make_geomdl_surface_with_knots(sub_pos, sub_H, sub_W, degree, knots_u, knots_v)
        rgb_surf = self._make_geomdl_surface_with_knots(sub_rgb, sub_H, sub_W, degree, knots_u, knots_v)

        # Verify knot vector consistency
        assert len(geo_surf.knotvector_u) == sub_H + degree + 1, \
            f"Knot U mismatch: {len(geo_surf.knotvector_u)} != {sub_H + degree + 1}"
        assert len(geo_surf.knotvector_v) == sub_W + degree + 1, \
            f"Knot V mismatch: {len(geo_surf.knotvector_v)} != {sub_W + degree + 1}"
        # knots_u = self._uniform_knots(sub_H, degree)
        # knots_v = self._uniform_knots(sub_W, degree)

        # ========== STEP 4: Build geomdl surfaces ==========
        # geo_surf = self._make_geomdl_surface(sub_pos, sub_H, sub_W, degree)
        # rgb_surf = self._make_geomdl_surface(sub_rgb, sub_H, sub_W, degree)
        #
        # # Verify knot vector consistency
        # assert len(geo_surf.knotvector_u) == sub_H + degree + 1, \
        #     f"Knot U mismatch: {len(geo_surf.knotvector_u)} != {sub_H + degree + 1}"
        # assert len(geo_surf.knotvector_v) == sub_W + degree + 1, \
        #     f"Knot V mismatch: {len(geo_surf.knotvector_v)} != {sub_W + degree + 1}"

        # ========== STEP 5: Create SplineModel ==========
        spline_model = SplineModel(
            surf=geo_surf,
            surf_rgb=rgb_surf,
            config=config,
            args=args,
            spatial_lr_scale=parent.spatial_lr_scale,
            train_cam_uids=train_cam_uids,
            late_init=False,
            surf_uid=hash(label) % 1000,
            skip_opt=True,
            label=label,
            is_background=is_background,
        )

        # ========== STEP 6: Transfer ALL cropped features exactly ==========
        # After SplineModel.__init__, it has default-initialized features.
        # We overwrite them with the exact parent values from the crop region.
        self._transfer_cropped_features(
            parent, spline_model, r0, r1, c0, c1, sub_H, sub_W
        )

        # ========== STEP 7: Attenuate opacity in padding zone ==========
        sub_mask = mask[r0:r1, c0:c1]  # [sub_H, sub_W]
        self._attenuate_padding_opacity(
            spline_model, sub_mask, sub_H, sub_W, degree, device
        )

        return spline_model

    def _crop_knot_vector(
            self,
            full_knots: torch.Tensor,
            degree: int,
            full_size: int,  # Kept for signature compatibility
            start: int,  # crop start index
            end: int,  # crop end index (exclusive)
    ) -> list:
        """
        Mathematically exact knot span extraction.
        Extracts the corresponding knot span for control points [start:end],
        enforces clamped boundaries to preserve parametric speed, and normalizes to [0, 1].
        """
        sub_size = end - start

        # The exact knot span corresponding to control points [start:end]
        # For N control points, we need exactly N + degree + 1 knots.
        raw_knots = full_knots[start: end + degree + 1].clone()

        # Enforce clamped boundaries:
        # The first (degree + 1) knots must anchor to the domain start
        domain_start = raw_knots[degree].item()
        raw_knots[:degree + 1] = domain_start

        # The last (degree + 1) knots must anchor to the domain end
        domain_end = raw_knots[-(degree + 1)].item()
        raw_knots[-(degree + 1):] = domain_end

        # Normalize the extracted span to [0, 1] to match standard SamplerUV coordinates
        span = domain_end - domain_start
        if span > 1e-8:
            normalized_knots = (raw_knots - domain_start) / span
        else:
            # Fallback only if the extracted domain is completely degenerate
            return self._uniform_knots(sub_size, degree)

        return normalized_knots.cpu().tolist()
    def _crop_knot_vector2(
            self,
            full_knots: torch.Tensor,
            degree: int,
            full_size: int,  # H or W
            start: int,  # crop start index
            end: int,  # crop end index (exclusive)
    ) -> list:
        """
        Compute a valid knot vector for the cropped sub-grid.

        The key insight: when we take control points [start:end] from a grid
        of size `full_size`, we need a knot vector with (end - start + degree + 1)
        knots that preserves the relative parameterization.

        Strategy: Extract the relevant portion of the original knot vector
        and re-normalize to [0, 1]. This preserves the relative spacing
        that the warmup optimization learned.
        """
        sub_size = end - start
        n_knots_needed = sub_size + degree + 1

        # The internal knots of the full vector correspond to control point boundaries
        # For a uniform-ish knot vector, internal knot i maps to the region
        # around control point i
        full_internal = full_knots[degree + 1: -(degree + 1)]  # Internal knots only
        n_full_internal = len(full_internal)

        if n_full_internal == 0 or sub_size <= degree + 1:
            # Fallback: uniform knot vector
            return self._uniform_knots(sub_size, degree)

        # Map control point indices to parametric values
        # Internal knot i roughly corresponds to the boundary between
        # control points i and i+1 (for clamped B-splines)
        n_internal_needed = sub_size - degree - 1

        if n_internal_needed <= 0:
            return self._uniform_knots(sub_size, degree)

        # Extract internal knots corresponding to the crop range
        # Internal knot index i corresponds to the boundary after
        # control point (degree + i) in the original grid
        internal_start = max(0, start - degree)
        internal_end = min(n_full_internal, end - degree)

        if internal_end <= internal_start:
            # Not enough internal knots in range — generate uniform
            return self._uniform_knots(sub_size, degree)

        relevant_internal = full_internal[internal_start:internal_end]

        # Resample to get exactly n_internal_needed knots
        if len(relevant_internal) == n_internal_needed:
            sub_internal = relevant_internal
        elif len(relevant_internal) > n_internal_needed:
            # Subsample uniformly
            indices = torch.linspace(0, len(relevant_internal) - 1, n_internal_needed).long()
            sub_internal = relevant_internal[indices]
        else:
            # Interpolate to get more knots
            t = torch.linspace(0, 1, n_internal_needed, device=relevant_internal.device)
            t_src = torch.linspace(0, 1, len(relevant_internal), device=relevant_internal.device)
            sub_internal = torch.interp(t, t_src, relevant_internal)

        # Normalize to [0, 1]
        lo = sub_internal.min()
        hi = sub_internal.max()
        if hi - lo > 1e-8:
            sub_internal = (sub_internal - lo) / (hi - lo)
        else:
            sub_internal = torch.linspace(0, 1, n_internal_needed + 2,
                                          device=sub_internal.device)[1:-1]

        # Clamp to (0, 1) strictly
        sub_internal = sub_internal.clamp(1e-6, 1 - 1e-6)

        # Build full clamped knot vector
        knots = (
                [0.0] * (degree + 1) +
                sub_internal.cpu().tolist() +
                [1.0] * (degree + 1)
        )

        return knots

    def _make_geomdl_surface_with_knots(
            self,
            ctrl_pts: torch.Tensor,
            H: int, W: int,
            degree: int,
            knots_u: list,
            knots_v: list,
    ):
        """Build geomdl surface with explicit knot vectors."""
        from geomdl import BSpline

        surf = BSpline.Surface()
        surf.degree_u = degree
        surf.degree_v = degree

        ctrlpts = ctrl_pts.reshape(-1, 3).detach().clone().cpu().tolist()
        surf.set_ctrlpts(ctrlpts, H, W)

        surf.knotvector_u = knots_u if isinstance(knots_u, list) else knots_u.tolist()
        surf.knotvector_v = knots_v if isinstance(knots_v, list) else knots_v.tolist()

        return surf

    def _transfer_cropped_features(
            self,
            parent: "SplineModel",
            child: "SplineModel",
            r0: int, r1: int,
            c0: int, c1: int,
            sub_H: int, sub_W: int,
    ):
        """
        Transfer features by exact crop — no resampling, no interpolation.

        IMPORTANT: SplineModel.__init__ already initialized child's control
        features with default values (from the geomdl surface for position,
        zeros/defaults for everything else). We OVERWRITE them here with
        the exact parent values from the crop region.

        This must handle the specific storage layouts used by each ControlFeature
        subclass in __init__.py.
        """
        H, W = parent.state.H, parent.state.W

        with torch.no_grad():
            # === Position ===
            # Parent layout: [H*W, 3] stored as control_features
            parent_pos = parent.position.control_features.data.view(H, W, 3)
            child_pos_data = parent_pos[r0:r1, c0:c1].reshape(-1, 3).contiguous()
            assert child.position.control_features.data.shape == child_pos_data.shape, \
                f"Position shape mismatch: child={child.position.control_features.data.shape}, " \
                f"cropped={child_pos_data.shape}"
            child.position.control_features.data.copy_(child_pos_data)

            # === SH DC ===
            # Layout: [H*W, 3] (dc component: 1 coeff * 3 channels, stored flat)
            parent_sh_dc = parent.spherical_harmonics.sh_dc.control_features.data
            p_dc_ch = parent_sh_dc.shape[-1]  # Should be 3
            parent_dc_grid = parent_sh_dc.view(H, W, p_dc_ch)
            child_dc_data = parent_dc_grid[r0:r1, c0:c1].reshape(-1, p_dc_ch).contiguous()

            c_dc = child.spherical_harmonics.sh_dc.control_features.data
            if c_dc.shape == child_dc_data.shape:
                c_dc.copy_(child_dc_data)
            else:
                # Channel mismatch — copy what we can
                c_dc_ch = c_dc.shape[-1]
                min_ch = min(p_dc_ch, c_dc_ch)
                c_dc_grid = c_dc.view(sub_H, sub_W, c_dc_ch)
                c_dc_grid[..., :min_ch] = parent_dc_grid[r0:r1, c0:c1, :min_ch]
                child.spherical_harmonics.sh_dc.control_features.data.copy_(
                    c_dc_grid.reshape(-1, c_dc_ch)
                )

            # === SH Rest ===
            # In SHControl.__init__, sh_rest stores features as:
            #   features[:, :, 1:].transpose(1, 2) → shape [H*W, (shc-1)*3]
            # But control_features is stored flat: [H*W, (shc-1)*3]
            p_rest = parent.spherical_harmonics.sh_rest
            c_rest = child.spherical_harmonics.sh_rest

            p_rest_cf = p_rest.control_features.data  # [H*W, (shc-1)*3]
            c_rest_cf = c_rest.control_features.data  # [sub_H*sub_W, (shc-1)*3]

            p_rest_ch = p_rest_cf.shape[-1]
            c_rest_ch = c_rest_cf.shape[-1]

            # Both should be [N, num_coeffs*3] where N = H*W or sub_H*sub_W
            if p_rest_cf.shape[0] == H * W and c_rest_cf.shape[0] == sub_H * sub_W:
                p_rest_grid = p_rest_cf.view(H, W, p_rest_ch)
                cropped_rest = p_rest_grid[r0:r1, c0:c1]  # [sub_H, sub_W, p_rest_ch]

                if p_rest_ch == c_rest_ch:
                    c_rest_cf_new = cropped_rest.reshape(-1, c_rest_ch).contiguous()
                    c_rest.control_features.data.copy_(c_rest_cf_new)
                else:
                    min_rest_ch = min(p_rest_ch, c_rest_ch)
                    c_rest_grid = c_rest_cf.view(sub_H, sub_W, c_rest_ch)
                    c_rest_grid[..., :min_rest_ch] = cropped_rest[..., :min_rest_ch]
                    c_rest.control_features.data.copy_(c_rest_grid.reshape(-1, c_rest_ch))
            else:
                # Fallback: the SH rest might be stored with coeffs folded into dim 0
                # i.e., [H*W*num_coeffs, 3]
                p_num = p_rest.num_sh_coeffs
                c_num = c_rest.num_sh_coeffs

                if p_rest_cf.shape[0] == H * W * p_num and p_rest_cf.shape[-1] == 3:
                    p_rest_grid = p_rest_cf.view(H, W, p_num, 3)
                    min_coeffs = min(p_num, c_num)

                    if c_rest_cf.shape[0] == sub_H * sub_W * c_num:
                        c_rest_grid = c_rest_cf.view(sub_H, sub_W, c_num, 3)
                        c_rest_grid[..., :min_coeffs, :] = p_rest_grid[r0:r1, c0:c1, :min_coeffs, :]
                        c_rest.control_features.data.copy_(
                            c_rest_grid.reshape(sub_H * sub_W * c_num, 3)
                        )
                    else:
                        print(f"  [Transfer] WARNING: SH rest layout unknown. "
                              f"Parent: {p_rest_cf.shape}, Child: {c_rest_cf.shape}. Skipping.")

            # === Opacity ===
            if parent.refine_opacity_active and child.refine_opacity_active:
                p_opa = parent.opacity.control_features.data.view(H, W, 1)
                child_opa = p_opa[r0:r1, c0:c1].reshape(-1, 1).contiguous()
                assert child.opacity.control_features.data.shape == child_opa.shape, \
                    f"Opacity shape mismatch: {child.opacity.control_features.data.shape} vs {child_opa.shape}"
                child.opacity.control_features.data.copy_(child_opa)

            # === Scaling ===
            if parent.refine_scales_active and child.refine_scales_active:
                p_scale = parent.scaling.control_features.data
                scale_ch = p_scale.shape[-1]
                p_scale_grid = p_scale.view(H, W, scale_ch)
                child_scale = p_scale_grid[r0:r1, c0:c1].reshape(-1, scale_ch).contiguous()
                assert child.scaling.control_features.data.shape == child_scale.shape, \
                    f"Scaling shape mismatch: {child.scaling.control_features.data.shape} vs {child_scale.shape}"
                child.scaling.control_features.data.copy_(child_scale)

            # === Rotation ===
            if parent.refine_rotations_active and child.refine_rotations_active:
                p_rot = parent.rotation.control_features.data.view(H, W, 4)
                child_rot = p_rot[r0:r1, c0:c1].reshape(-1, 4).contiguous()
                assert child.rotation.control_features.data.shape == child_rot.shape, \
                    f"Rotation shape mismatch: {child.rotation.control_features.data.shape} vs {child_rot.shape}"
                child.rotation.control_features.data.copy_(child_rot)

            # === Weights (NURBS) ===
            if (parent.refine_weights_active and child.refine_weights_active
                    and parent.weights.control_features is not None
                    and child.weights.control_features is not None):
                p_w = parent.weights.control_features.data
                w_ch = p_w.shape[-1]
                p_w_grid = p_w.view(H, W, w_ch)
                child_w = p_w_grid[r0:r1, c0:c1].reshape(-1, w_ch).contiguous()
                child.weights.control_features.data.copy_(child_w)

        print(f"  [Transfer] Exact crop [{r0}:{r1}, {c0}:{c1}] → "
              f"[{sub_H}×{sub_W}] for '{child.label}'")

    def _attenuate_padding_opacity(
            self,
            surface: "SplineModel",
            sub_mask: torch.Tensor,  # [sub_H, sub_W] bool
            sub_H: int, sub_W: int,
            degree: int,
            device: str,
    ):
        """
        Gently attenuate opacity for control points in the padding zone
        (outside the mask but inside the cropped bounding box).
        """
        if not surface.refine_opacity_active:
            return

        padding_mask = ~sub_mask
        n_padding = padding_mask.sum().item()
        n_total = sub_H * sub_W
        print(f"  [Padding] '{surface.label}': "
              f"{n_padding}/{n_total} padding points ({n_padding / n_total:.1%})")

        if n_padding == 0:
            return

        try:
            from scipy.ndimage import distance_transform_edt
            mask_np = sub_mask.cpu().numpy().astype(np.float64)
            dist_from_mask = distance_transform_edt(1 - mask_np)
            dist_tensor = torch.tensor(dist_from_mask, device=device, dtype=torch.float32)
        except ImportError:
            dist_tensor = padding_mask.float()

        # Sigmoid falloff: full opacity at mask boundary, decaying outward
        scale = max(degree / 2.0, 1.0)
        attenuation = torch.sigmoid(-(dist_tensor - 0.5) / scale)
        attenuation[sub_mask] = 1.0  # Preserve mask-interior opacity

        with torch.no_grad():
            raw_opacity = surface.opacity.control_features.data.view(sub_H, sub_W, 1)
            opacity_prob = torch.sigmoid(raw_opacity)
            modulated = opacity_prob * attenuation.unsqueeze(-1)
            modulated = modulated.clamp(1e-4, 1 - 1e-4)

            from utils.general_utils import inverse_sigmoid
            new_raw = inverse_sigmoid(modulated)
            surface.opacity.control_features.data.copy_(new_raw.reshape(-1, 1))

        final_opacity = torch.sigmoid(surface.opacity.control_features.data.view(sub_H, sub_W))
        print(f"  [Padding] '{surface.label}' opacity: "
              f"mask_mean={final_opacity[sub_mask].mean():.4f}, "
              f"padding_mean={final_opacity[padding_mask].mean():.4f}")
    # ------------------------------------------------------------------
    # Level-set field construction
    # ------------------------------------------------------------------

    def _build_level_set_field(
            self,
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        """
        Build scalar field φ(u,v) where:
          φ > 0  →  foreground (object)
          φ < 0  →  background
          φ = 0  →  contour boundary

        Combines:
        1. Depth: closer = more likely FG
        2. Color homogeneity: BG = uniform, FG = varied
        3. Normal variance: FG = higher curvature
        4. Boundary prior: UV grid edges → BG
        """
        field = torch.zeros(H, W, device=device)

        # --- (A) Depth cue ---
        d_min, d_max = agg_depth.min(), agg_depth.max()
        depth_norm = (agg_depth - d_min) / (d_max - d_min + 1e-8)

        if self.cfg.use_otsu:
            depth_binary = self._otsu_threshold(depth_norm)
            mean_depth_0 = depth_norm[depth_binary == 0].mean()
            mean_depth_1 = depth_norm[depth_binary == 1].mean()
            fg_label = 0 if mean_depth_0 < mean_depth_1 else 1
            depth_field = torch.where(
                depth_binary == fg_label,
                torch.ones_like(depth_norm),
                -torch.ones_like(depth_norm)
            ).float()
        else:
            threshold = depth_norm.quantile(self.cfg.depth_fg_quantile)
            depth_field = -(depth_norm - threshold)

        depth_range = d_max - d_min
        depth_confidence = (depth_range / (d_max + 1e-8)).clamp(0, 1)
        field += self.cfg.contour_depth_weight * depth_field * depth_confidence

        # --- (B) Color homogeneity cue ---
        sh_dc_raw = surface.spherical_harmonics.sh_dc.control_features.detach()
        color_grid = SH2RGB(sh_dc_raw.view(H, W, 3)).clamp(0, 1)
        color_grid = self._smooth_grid(color_grid, self.cfg.smooth_sigma * 0.5)

        color_var = self._compute_local_variance(color_grid, kernel_size=5)
        var_median = color_var.median()
        color_field = (color_var - var_median) / (color_var.std() + 1e-8)
        color_field = color_field.clamp(-2, 2) / 2

        field += self.cfg.contour_color_weight * color_field

        # --- (C) Normal variance cue ---
        if self.cfg.contour_normal_weight > 0:
            normal_grid = self._extract_control_normals(surface, H, W, device)
            normal_var = self._compute_local_variance(normal_grid, kernel_size=5)
            normal_median = normal_var.median()
            normal_field = (normal_var - normal_median) / (normal_var.std() + 1e-8)
            normal_field = normal_field.clamp(-2, 2) / 2
            field += self.cfg.contour_normal_weight * normal_field

        # --- (D) Boundary prior ---
        boundary_field = self._compute_boundary_distance_field(H, W, device)
        boundary_bias = (boundary_field - 0.5) * 0.5
        field += boundary_bias

        print(f"[Level-Set] Field stats: min={field.min():.3f}, max={field.max():.3f}, "
              f"mean={field.mean():.3f}, std={field.std():.3f}")

        return field

    def _compute_local_variance(
            self,
            grid: torch.Tensor,
            kernel_size: int = 5,
    ) -> torch.Tensor:
        """Per-pixel local variance of a multi-channel grid. Returns [H, W]."""
        H, W, C = grid.shape
        device = grid.device
        pad = kernel_size // 2

        g = grid.permute(2, 0, 1).unsqueeze(0)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        kernel = kernel / (kernel_size * kernel_size)

        padded = F.pad(g, (pad, pad, pad, pad), mode='reflect')

        local_mean = torch.zeros_like(g)
        local_mean_sq = torch.zeros_like(g)
        for c in range(C):
            ch = padded[:, c:c + 1]
            local_mean[:, c:c + 1] = F.conv2d(ch, kernel)
            local_mean_sq[:, c:c + 1] = F.conv2d(ch ** 2, kernel)

        local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)
        total_var = local_var.sum(dim=1).squeeze(0)
        return total_var

    def _compute_boundary_distance_field(
            self,
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        """[0,1] field: 0 = on UV boundary, 1 = at center (Chebyshev distance)."""
        u_dist = torch.arange(H, device=device, dtype=torch.float32)
        u_dist = torch.min(u_dist, (H - 1) - u_dist)
        v_dist = torch.arange(W, device=device, dtype=torch.float32)
        v_dist = torch.min(v_dist, (W - 1) - v_dist)

        uu, vv = torch.meshgrid(u_dist, v_dist, indexing='ij')
        boundary_dist = torch.min(uu, vv)

        max_dist = boundary_dist.max()
        if max_dist > 0:
            boundary_dist = boundary_dist / max_dist
        return boundary_dist

    # ------------------------------------------------------------------
    # Level-set smoothing
    # ------------------------------------------------------------------

    def _smooth_level_set(
            self,
            phi: torch.Tensor,
            H: int, W: int,
    ) -> torch.Tensor:
        """Iterative Laplacian smoothing to regularize the contour."""
        for _ in range(self.cfg.contour_smooth_iters):
            laplacian = torch.zeros_like(phi)
            count = torch.zeros_like(phi)

            laplacian[1:] += phi[:-1];
            count[1:] += 1
            laplacian[:-1] += phi[1:];
            count[:-1] += 1
            laplacian[:, 1:] += phi[:, :-1];
            count[:, 1:] += 1
            laplacian[:, :-1] += phi[:, 1:];
            count[:, :-1] += 1

            mean_neighbor = laplacian / count.clamp(min=1)
            phi = phi + self.cfg.contour_smooth_lambda * (mean_neighbor - phi)

        return phi

    # ------------------------------------------------------------------
    # Contour mask extraction
    # ------------------------------------------------------------------

    def _extract_contour_mask(
            self,
            phi: torch.Tensor,
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        fg_mask = phi > 0
        if self.cfg.contour_use_morphological:
            fg_mask = self._cleanup_contour_mask(fg_mask, H, W, device)
        return fg_mask

    def _cleanup_contour_mask(
            self,
            mask: torch.Tensor,
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        """Morphological cleanup: fill holes, remove small CCs, close boundary."""
        try:
            from scipy import ndimage

            arr = mask.cpu().numpy().astype(np.int32)
            total = H * W

            arr = ndimage.binary_fill_holes(arr).astype(np.int32)

            labeled, num_features = ndimage.label(arr)
            for comp_id in range(1, num_features + 1):
                comp_mask = labeled == comp_id
                if comp_mask.sum() / total < self.cfg.cc_min_area_frac:
                    arr[comp_mask] = 0

            bg_arr = 1 - arr
            bg_labeled, bg_features = ndimage.label(bg_arr)
            for comp_id in range(1, bg_features + 1):
                comp_mask = bg_labeled == comp_id
                touches_border = (
                        comp_mask[0, :].any() or comp_mask[-1, :].any() or
                        comp_mask[:, 0].any() or comp_mask[:, -1].any()
                )
                if not touches_border and comp_mask.sum() / total < 0.1:
                    arr[comp_mask] = 1

            if self.cfg.contour_dilate_iters > 0:
                arr = ndimage.binary_dilation(
                    arr, iterations=self.cfg.contour_dilate_iters
                ).astype(np.int32)

            if self.cfg.morph_close_iters > 0:
                arr = ndimage.binary_closing(
                    arr, iterations=self.cfg.morph_close_iters
                ).astype(np.int32)

            return torch.tensor(arr, device=device, dtype=torch.bool)

        except ImportError:
            print("[Warning] scipy not available, returning raw contour mask")
            return mask

    # ------------------------------------------------------------------
    # Signed distance from contour
    # ------------------------------------------------------------------

    def _compute_signed_distance(
            self,
            fg_mask: torch.Tensor,
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        """
        Signed distance transform: positive inside FG, negative inside BG.
        Used for soft opacity transitions at the boundary.
        """
        try:
            from scipy.ndimage import distance_transform_edt

            fg_np = fg_mask.cpu().numpy().astype(np.float64)
            dist_inside = distance_transform_edt(fg_np)
            dist_outside = distance_transform_edt(1 - fg_np)
            signed_dist = dist_inside - dist_outside

            return torch.tensor(signed_dist, device=device, dtype=torch.float32)

        except ImportError:
            print("[Warning] scipy not available, using hard boundary")
            return fg_mask.float() * 2 - 1

    # ------------------------------------------------------------------
    # Complementary opacity masks
    # ------------------------------------------------------------------

    def _build_complementary_opacity_masks(
        self,
        signed_dist: torch.Tensor,
        fg_mask: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build soft, complementary opacity masks.

        Critical invariant: at every UV point,
            fg_opacity_mask + bg_opacity_mask ≈ 1

        This ensures:
        1. No volumetric overlap — each point "belongs" to one surface
        2. Smooth transitions at contour boundary
        3. Gradient flow through boundary during training
        """
        bw = max(self.cfg.contour_boundary_width, 1)
        scale = float(bw) / 3.0

        fg_opacity_mask = torch.sigmoid(signed_dist / scale)
        bg_opacity_mask = 1.0 - fg_opacity_mask

        # Floor: prevents zero gradients in suppressed regions
        min_fg = self.cfg.contour_fg_opacity_outside
        min_bg = self.cfg.contour_bg_opacity_inside

        fg_opacity_mask = fg_opacity_mask.clamp(min=min_fg)
        bg_opacity_mask = bg_opacity_mask.clamp(min=min_bg)

        # Renormalize for complementarity
        total = fg_opacity_mask + bg_opacity_mask
        fg_opacity_mask = fg_opacity_mask / total
        bg_opacity_mask = bg_opacity_mask / total

        print(f"[Opacity Masks] FG: min={fg_opacity_mask.min():.4f}, "
              f"max={fg_opacity_mask.max():.4f}, mean={fg_opacity_mask.mean():.4f}")
        print(f"[Opacity Masks] BG: min={bg_opacity_mask.min():.4f}, "
              f"max={bg_opacity_mask.max():.4f}, mean={bg_opacity_mask.mean():.4f}")

        return fg_opacity_mask, bg_opacity_mask

    # ------------------------------------------------------------------
    # Clone surface with opacity modulation
    # ------------------------------------------------------------------
    def _clone_surface_with_opacity_mask(
            self,
            parent: "SplineModel",
            opacity_mask: torch.Tensor,  # [H, W] soft mask in [0, 1]
            label: str,
            is_background: bool,
            config,
            args,
            train_cam_uids: List,
    ) -> "SplineModel":
        """
        Clone the parent surface, enforcing disjointness through BOTH
        position collapse AND hard opacity suppression.

        Key insight: opacity-only modulation fails because B-spline basis
        functions smear control-point opacity across (degree+1)² neighbors.
        A control point with opacity=0.01 next to one with opacity=0.99
        produces ~0.5 opacity at the boundary after interpolation.

        Solution: for suppressed regions, we:
        1. Set opacity to extreme negative raw values (sigmoid → ~0)
        2. Collapse positions toward the active region's centroid
        3. Zero out SH coefficients (no color contribution)
        4. Minimize scaling (tiny Gaussians even if opacity leaks)

        This ensures suppressed control points contribute NOTHING to rendering
        even after B-spline basis function blending.
        """
        from geomdl import BSpline
        from model.modules.KnotSurface import SplineModel

        H, W = parent.state.H, parent.state.W
        degree = parent.state.degree
        device = parent.device

        # --- Build the geomdl surfaces (unchanged) ---
        pos_ctrl = parent.position.control_features.detach().clone().view(H, W, 3)
        geo_surf = self._make_geomdl_surface(pos_ctrl, H, W, degree)

        sh_dc = parent.spherical_harmonics.sh_dc.control_features.detach().clone().view(H, W, 3)
        rgb_ctrl = (SH2RGB(sh_dc)).clamp(0, 1)
        rgb_surf = self._make_geomdl_surface(rgb_ctrl, H, W, degree)

        spline_model = SplineModel(
            surf=geo_surf,
            surf_rgb=rgb_surf,
            config=config,
            args=args,
            spatial_lr_scale=parent.spatial_lr_scale,
            train_cam_uids=train_cam_uids,
            late_init=False,
            surf_uid=hash(label) % 1000,
            skip_opt=True,
            label=label,
            is_background=is_background,
        )

        # --- Copy ALL features from parent ---
        self._transfer_all_features(parent, spline_model, H, W)

        # --- NOW enforce disjointness with hard suppression ---
        self._apply_hard_suppression(
            spline_model, opacity_mask, parent, H, W, device
        )

        return spline_model

    def _apply_hard_suppression(
            self,
            surface: "SplineModel",
            active_mask: torch.Tensor,  # [H, W] float, 1.0 = active, ~0 = suppressed
            parent: "SplineModel",
            H: int, W: int,
            device: str,
    ):
        """
        Hard suppression of control points in inactive regions.

        This replaces the soft opacity modulation that was failing due to
        B-spline basis function support overlap.

        For suppressed control points (active_mask < threshold):
        1. Opacity → -20 in raw space (sigmoid(-20) ≈ 2e-9)
        2. Positions → collapse toward active region centroid
        3. SH DC → zero (no color emission)
        4. SH rest → zero
        5. Scaling → very small (minimize any leaking Gaussian's footprint)
        """
        SUPPRESSION_THRESHOLD = 0.3  # Below this → fully suppressed
        OPACITY_FLOOR_RAW = -10.0  # sigmoid(-20) ≈ 2e-9
        SCALE_FLOOR_RAW = -6.0  # exp(-15) ≈ 3e-7

        # Binary suppression mask: True = this control point should be killed
        suppress = (active_mask < SUPPRESSION_THRESHOLD)  # [H, W] bool
        active = ~suppress

        n_suppressed = suppress.sum().item()
        n_active = active.sum().item()
        print(f"  [Hard Suppress] '{surface.label}': "
              f"{n_active} active, {n_suppressed} suppressed "
              f"({n_suppressed / (H * W) * 100:.1f}% killed)")

        if n_suppressed == 0:
            return

        with torch.no_grad():
            # === 1. OPACITY: slam to floor ===
            if surface.refine_opacity_active:
                raw_opacity = surface.opacity.control_features.data.view(H, W, 1)
                raw_opacity[suppress] = OPACITY_FLOOR_RAW
                surface.opacity.control_features.data.copy_(raw_opacity.reshape(-1, 1))

            # === 2. POSITIONS: collapse toward active centroid ===
            # This prevents suppressed control points from "pulling" the
            # B-spline surface into wrong regions via basis function overlap
            pos = surface.position.control_features.data.view(H, W, 3)

            if n_active > 0:
                # Compute centroid of active region
                active_centroid = pos[active].mean(dim=0)  # [3]

                # Collapse suppressed points toward centroid with a gradient
                # Points near the boundary get partial collapse (smoother transition)
                # Points deep in suppressed region get full collapse

                # Use signed distance for graduated collapse
                try:
                    from scipy.ndimage import distance_transform_edt
                    suppress_np = suppress.cpu().numpy().astype(np.float64)
                    dist_from_active = distance_transform_edt(suppress_np)
                    dist_tensor = torch.tensor(
                        dist_from_active, device=device, dtype=torch.float32
                    )
                    # Normalize: 0 at boundary, 1 deep inside
                    max_dist = dist_tensor.max()
                    if max_dist > 0:
                        collapse_strength = (dist_tensor / max_dist).clamp(0, 1)
                    else:
                        collapse_strength = torch.ones(H, W, device=device)
                except ImportError:
                    collapse_strength = torch.ones(H, W, device=device)

                # Graduated collapse: lerp between original position and centroid
                # Near boundary: keep mostly original (for smooth B-spline transition)
                # Deep inside: full collapse to centroid
                collapse_strength_3d = collapse_strength.unsqueeze(-1)  # [H,W,1]
                pos[suppress] = (
                        (1 - collapse_strength_3d[suppress]) * pos[suppress] +
                        collapse_strength_3d[suppress] * active_centroid.unsqueeze(0)
                )
                surface.position.control_features.data.copy_(pos.reshape(-1, 3))

            # === 3. SH DC: zero out color in suppressed region ===
            sh_dc_cf = surface.spherical_harmonics.sh_dc.control_features.data
            sh_dc_shape = sh_dc_cf.shape
            sh_dc_grid = sh_dc_cf.clone().view(H, W, -1)
            sh_dc_grid[suppress] = 0.0
            surface.spherical_harmonics.sh_dc.control_features.data.copy_(
                sh_dc_grid.reshape(sh_dc_shape)
            )

            # === 4. SH REST: zero out ===
            sh_rest_cf = surface.spherical_harmonics.sh_rest.control_features.data
            num_coeffs = surface.spherical_harmonics.sh_rest.num_sh_coeffs
            actual_shape = sh_rest_cf.shape  # Could be [H*W*num_coeffs, 3] or [H*W, num_coeffs*3]

            # Determine storage layout by checking dimensions
            total_elements = actual_shape[0]
            last_dim = actual_shape[-1] if sh_rest_cf.dim() > 1 else 1

            if total_elements == H * W * num_coeffs and last_dim == 3:
                # Layout: [H*W*num_coeffs, 3] — coeffs folded into first dim
                sh_rest_grid = sh_rest_cf.clone().view(H, W, num_coeffs, 3)
                sh_rest_grid[suppress] = 0.0
                surface.spherical_harmonics.sh_rest.control_features.data.copy_(
                    sh_rest_grid.reshape(H * W * num_coeffs, 3)
                )
            elif total_elements == H * W and last_dim == num_coeffs * 3:
                # Layout: [H*W, num_coeffs*3] — coeffs folded into last dim
                sh_rest_grid = sh_rest_cf.clone().view(H, W, num_coeffs * 3)
                sh_rest_grid[suppress] = 0.0
                surface.spherical_harmonics.sh_rest.control_features.data.copy_(
                    sh_rest_grid.reshape(H * W, num_coeffs * 3)
                )
            else:
                # Fallback: just clone, reshape to [H, W, ...], zero suppress, reshape back
                sh_rest_grid = sh_rest_cf.clone().view(H, W, -1)
                sh_rest_grid[suppress] = 0.0
                surface.spherical_harmonics.sh_rest.control_features.data.copy_(
                    sh_rest_grid.reshape(actual_shape)
                )

            # === 5. SCALING: minimize suppressed Gaussians ===
            if surface.refine_scales_active:
                scale_data = surface.scaling.control_features.data
                scale_ch = scale_data.shape[-1]
                scale_grid = scale_data.view(H, W, scale_ch)
                scale_grid[suppress] = SCALE_FLOOR_RAW
                surface.scaling.control_features.data.copy_(
                    scale_grid.reshape(-1, scale_ch)
                )

        # --- Validation ---
        with torch.no_grad():
            if surface.refine_opacity_active:
                final_opacity = torch.sigmoid(
                    surface.opacity.control_features.data.view(H, W)
                )
                print(f"  [Hard Suppress] '{surface.label}' opacity stats: "
                      f"active_mean={final_opacity[active].mean():.4f}, "
                      f"suppressed_max={final_opacity[suppress].max():.2e}, "
                      f"active_frac={(final_opacity > 0.01).float().mean():.1%}")

    def _transfer_all_features(
            self,
            parent: "SplineModel",
            child: "SplineModel",
            H: int, W: int,
    ):
        """Full grid feature copy — no cropping, no resampling."""
        with torch.no_grad():
            child.position.control_features.data.copy_(
                parent.position.control_features.data
            )

            child.spherical_harmonics.sh_dc.control_features.data.copy_(
                parent.spherical_harmonics.sh_dc.control_features.data
            )

            p_rest = parent.spherical_harmonics.sh_rest
            c_rest = child.spherical_harmonics.sh_rest
            if p_rest.num_sh_coeffs == c_rest.num_sh_coeffs:
                c_rest.control_features.data.copy_(p_rest.control_features.data)
            else:
                min_coeffs = min(p_rest.num_sh_coeffs, c_rest.num_sh_coeffs)
                src = p_rest.control_features.data.view(H, W, p_rest.num_sh_coeffs, 3)
                dst = c_rest.control_features.data.view(H, W, c_rest.num_sh_coeffs, 3)
                dst[:, :, :min_coeffs, :] = src[:, :, :min_coeffs, :]

            if parent.refine_opacity_active and child.refine_opacity_active:
                child.opacity.control_features.data.copy_(
                    parent.opacity.control_features.data
                )

            if parent.refine_scales_active and child.refine_scales_active:
                child.scaling.control_features.data.copy_(
                    parent.scaling.control_features.data
                )

            if parent.refine_rotations_active and child.refine_rotations_active:
                child.rotation.control_features.data.copy_(
                    parent.rotation.control_features.data
                )

            if (parent.refine_weights_active and child.refine_weights_active
                    and parent.weights.control_features is not None
                    and child.weights.control_features is not None):
                child.weights.control_features.data.copy_(
                    parent.weights.control_features.data
                )

        print(f"  [Transfer] Full grid copy: [{H}x{W}] → [{H}x{W}] for '{child.label}'")

    def _apply_opacity_mask(
            self,
            surface: "SplineModel",
            opacity_mask: torch.Tensor,
            parent: "SplineModel",
            H: int, W: int,
    ):
        """
        Modulate opacity control points by mask.

        In inverse-sigmoid space:
            o_new_raw = inverse_sigmoid(sigmoid(o_raw) * mask)
        """
        if not surface.refine_opacity_active:
            return

        with torch.no_grad():
            raw_opacity = surface.opacity.control_features.data.view(H, W, 1)
            opacity_prob = torch.sigmoid(raw_opacity)
            mask_3d = opacity_mask.unsqueeze(-1)
            modulated = opacity_prob * mask_3d
            modulated = modulated.clamp(1e-4, 1 - 1e-4)

            from utils.general_utils import inverse_sigmoid
            new_raw = inverse_sigmoid(modulated)

            surface.opacity.control_features.data.copy_(new_raw.reshape(-1, 1))

        final_opacity = torch.sigmoid(surface.opacity.control_features.data.view(H, W))
        print(f"  [Opacity] '{surface.label}': mean={final_opacity.mean():.4f}, "
              f"min={final_opacity.min():.4f}, max={final_opacity.max():.4f}, "
              f"active_frac={(final_opacity > 0.1).float().mean():.1%}")

    # ------------------------------------------------------------------
    # Contour diagnostics
    # ------------------------------------------------------------------

    def _log_contour_stats(
            self,
            fg_mask, bg_mask, fg_opacity, bg_opacity,
            signed_dist, H, W,
    ):
        fg_area = fg_mask.float().mean().item()
        bg_area = bg_mask.float().mean().item()
        boundary_pixels = ((signed_dist.abs() < self.cfg.contour_boundary_width).float().mean().item())

        total_opacity = fg_opacity + bg_opacity
        complementarity_error = (total_opacity - 1.0).abs().mean().item()

        overlap_threshold = 0.3
        both_active = ((fg_opacity > overlap_threshold) &
                       (bg_opacity > overlap_threshold)).float().mean().item()

        print(f"\n[Contour Decomposition Summary]")
        print(f"  Grid: {H}x{W} = {H * W} control points")
        print(f"  FG area: {fg_area:.1%} ({int(fg_area * H * W)} points)")
        print(f"  BG area: {bg_area:.1%} ({int(bg_area * H * W)} points)")
        print(f"  Boundary zone: {boundary_pixels:.1%}")
        print(f"  Complementarity error: {complementarity_error:.6f}")
        print(f"  Overlap (both > {overlap_threshold}): {both_active:.1%}")

    # ==================================================================
    #  LEGACY SEGMENTATION DISPATCHER
    # ==================================================================

    def _segment(
            self,
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            device: str,
    ) -> Dict:
        mode = self.cfg.segmentation_mode
        print(f"[Decomp] Segmentation mode: {mode.value}")

        if mode == SegmentationMode.DEPTH_UV:
            return self._segment_depth_uv(surface, agg_depth, device)
        elif mode == SegmentationMode.SEMANTIC:
            return self._segment_semantic(surface, agg_depth, device)
        elif mode == SegmentationMode.DEPTH_SEMANTIC:
            return self._segment_depth_then_semantic(surface, agg_depth, device)
        elif mode == SegmentationMode.CONTOUR:
            raise RuntimeError("CONTOUR mode should use _decompose_contour, not _segment")
        else:
            raise ValueError(f"Unknown segmentation mode: {mode}")

    # ==================================================================
    #  MODE 1: DEPTH_UV
    # ==================================================================

    def _segment_depth_uv(
            self,
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            device: str,
    ) -> Dict:
        H, W = surface.state.H, surface.state.W

        sh_dc_raw = surface.spherical_harmonics.sh_dc.control_features.detach()
        color_grid = SH2RGB(sh_dc_raw.view(H, W, 3)).clamp(0, 1)
        color_grid = self._smooth_grid(color_grid, self.cfg.smooth_sigma)

        d_min, d_max = agg_depth.min(), agg_depth.max()
        depth_norm = (agg_depth - d_min) / (d_max - d_min + 1e-8)

        color_weighted = color_grid * 0.6
        depth_weighted = depth_norm.unsqueeze(-1) * 0.4

        u = torch.linspace(0, 1, H, device=device)
        v = torch.linspace(0, 1, W, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        spatial = torch.stack([uu, vv], dim=-1) * 0.15

        features = torch.cat([color_weighted, depth_weighted, spatial], dim=-1)

        init_labels = None
        if self.cfg.use_otsu and self.cfg.n_components == 2:
            init_labels = self._otsu_threshold(depth_norm)

        label_map = self._kmeans(
            features, k=self.cfg.n_components, max_iter=150,
            init_labels=init_labels,
        )

        label_map = self._postprocess(label_map, H, W)
        return self._assign_labels(label_map, color_grid, depth_norm, surface, H, W, device)

    # ==================================================================
    #  MODE 2: SEMANTIC
    # ==================================================================

    def _segment_semantic(
            self,
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            device: str,
    ) -> Dict:
        H, W = surface.state.H, surface.state.W

        feature_channels = []
        channel_names = []

        sh_dc_raw = surface.spherical_harmonics.sh_dc.control_features.detach()
        color_grid = SH2RGB(sh_dc_raw.view(H, W, 3)).clamp(0, 1)
        color_grid = self._smooth_grid(color_grid, self.cfg.smooth_sigma)
        feature_channels.append(color_grid * self.cfg.semantic_color_weight)
        channel_names.append(f"color[3] x {self.cfg.semantic_color_weight}")

        if self.cfg.semantic_use_normals:
            normal_grid = self._extract_control_normals(surface, H, W, device)
            normal_grid = self._smooth_grid(normal_grid, self.cfg.smooth_sigma * 0.5)
            feature_channels.append(normal_grid * self.cfg.semantic_normal_weight)
            channel_names.append(f"normals[3] x {self.cfg.semantic_normal_weight}")

        if self.cfg.semantic_use_opacity and surface.refine_opacity_active:
            opacity_raw = surface.opacity.control_features.detach().view(H, W, 1)
            opacity_grid = torch.sigmoid(opacity_raw)
            opacity_grid = self._smooth_grid(opacity_grid, self.cfg.smooth_sigma)
            feature_channels.append(opacity_grid * self.cfg.semantic_opacity_weight)
            channel_names.append(f"opacity[1] x {self.cfg.semantic_opacity_weight}")

        d_min, d_max = agg_depth.min(), agg_depth.max()
        depth_norm = (agg_depth - d_min) / (d_max - d_min + 1e-8)
        feature_channels.append(
            depth_norm.unsqueeze(-1) * self.cfg.semantic_depth_weight
        )
        channel_names.append(f"depth[1] x {self.cfg.semantic_depth_weight}")

        u = torch.linspace(0, 1, H, device=device)
        v = torch.linspace(0, 1, W, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        spatial = torch.stack([uu, vv], dim=-1)
        feature_channels.append(spatial * self.cfg.semantic_spatial_weight)
        channel_names.append(f"spatial[2] x {self.cfg.semantic_spatial_weight}")

        features = torch.cat(feature_channels, dim=-1)
        total_dim = features.shape[-1]
        print(f"[Semantic Seg] Feature channels: {channel_names}, total dim={total_dim}")

        label_map = self._kmeans(features, k=self.cfg.n_components, max_iter=200)
        label_map = self._postprocess_semantic(label_map, H, W)

        return self._assign_labels(label_map, color_grid, depth_norm, surface, H, W, device)

    # ==================================================================
    #  MODE 3: DEPTH → SEMANTIC REFINEMENT
    # ==================================================================

    def _segment_depth_then_semantic(
            self,
            surface: "SplineModel",
            agg_depth: torch.Tensor,
            device: str,
    ) -> Dict:
        H, W = surface.state.H, surface.state.W

        d_min, d_max = agg_depth.min(), agg_depth.max()
        depth_norm = (agg_depth - d_min) / (d_max - d_min + 1e-8)

        n_coarse = self.cfg.depth_coarse_components

        if n_coarse == 2 and self.cfg.use_otsu:
            coarse_labels = self._otsu_threshold(depth_norm)
        elif n_coarse > 2:
            coarse_labels = self._multi_threshold_depth(depth_norm, n_coarse, device)
        else:
            coarse_labels = self._otsu_threshold(depth_norm)

        print(f"[Depth→Semantic] Stage 1: {n_coarse} coarse depth clusters")
        coarse_labels = self._postprocess(coarse_labels, H, W)

        sh_dc_raw = surface.spherical_harmonics.sh_dc.control_features.detach()
        color_grid = SH2RGB(sh_dc_raw.view(H, W, 3)).clamp(0, 1)
        color_grid = self._smooth_grid(color_grid, self.cfg.smooth_sigma)

        normal_grid = self._extract_control_normals(surface, H, W, device)

        opacity_grid = None
        if surface.refine_opacity_active:
            opacity_raw = surface.opacity.control_features.detach().view(H, W, 1)
            opacity_grid = torch.sigmoid(opacity_raw)

        unique_labels = coarse_labels.unique()
        cluster_sigs: Dict[int, Dict] = {}

        for lbl in unique_labels:
            mask = coarse_labels == lbl
            if mask.sum() < 2:
                continue
            sig: Dict = {}
            sig['mean_color'] = color_grid[mask].mean(dim=0)
            sig['std_color'] = color_grid[mask].std(dim=0)
            sig['mean_normal'] = normal_grid[mask].mean(dim=0)
            sig['normal_variance'] = normal_grid[mask].var(dim=0).sum()
            sig['mean_depth'] = depth_norm[mask].mean()
            sig['depth_variance'] = depth_norm[mask].var()
            if opacity_grid is not None:
                sig['mean_opacity'] = opacity_grid[mask].mean()
                sig['opacity_variance'] = opacity_grid[mask].var()
            sig['size_frac'] = mask.float().mean()
            sig['touches_boundary'] = (
                    mask[0, :].any() or mask[-1, :].any() or
                    mask[:, 0].any() or mask[:, -1].any()
            )
            cluster_sigs[lbl.item()] = sig

        print(f"[Depth→Semantic] Stage 2: signatures for "
              f"{len(cluster_sigs)} clusters")

        if len(cluster_sigs) > self.cfg.n_components:
            merged_labels = self._merge_clusters(
                coarse_labels, cluster_sigs,
                self.cfg.n_components,
                self.cfg.semantic_merge_threshold,
            )
        else:
            merged_labels = coarse_labels

        merged_labels = self._refine_boundaries(
            merged_labels, color_grid, normal_grid, depth_norm,
            opacity_grid, H, W, device,
            boundary_width=self.cfg.boundary_refine_width,
        )

        merged_labels = self._postprocess_semantic(merged_labels, H, W)

        return self._assign_labels(
            merged_labels, color_grid, depth_norm,
            surface, H, W, device,
        )

    # ==================================================================
    #  SHARED HELPERS
    # ==================================================================

    def _extract_control_normals(
            self,
            surface: "SplineModel",
            H: int, W: int,
            device: str,
    ) -> torch.Tensor:
        pos = surface.position.control_features.detach().view(H, W, 3)

        du = torch.zeros_like(pos)
        dv = torch.zeros_like(pos)

        du[:-1] = pos[1:] - pos[:-1]
        du[-1] = du[-2]

        dv[:, :-1] = pos[:, 1:] - pos[:, :-1]
        dv[:, -1] = dv[:, -2]

        normals = torch.cross(du, dv, dim=-1)
        normals = F.normalize(normals, dim=-1, eps=1e-8)
        return normals

    def _otsu_threshold(
            self,
            values: torch.Tensor,
            num_bins: int = None,
    ) -> torch.Tensor:
        if num_bins is None:
            num_bins = self.cfg.otsu_num_bins

        device = values.device
        flat = values.reshape(-1)

        hist = torch.histc(flat, bins=num_bins, min=0.0, max=1.0)
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist_prob = hist / hist.sum()

        omega = torch.cumsum(hist_prob, dim=0)
        mu = torch.cumsum(hist_prob * bin_centers, dim=0)
        mu_total = mu[-1]

        omega1 = 1.0 - omega
        valid = (omega > 1e-10) & (omega1 > 1e-10)

        mu0 = torch.where(valid, mu / omega.clamp(min=1e-10), torch.zeros_like(mu))
        mu1 = torch.where(valid, (mu_total - mu) / omega1.clamp(min=1e-10), torch.zeros_like(mu))

        sigma_between = omega * omega1 * (mu0 - mu1) ** 2
        sigma_between[~valid] = 0.0

        best_idx = sigma_between.argmax()
        threshold = bin_centers[best_idx]

        print(f"[Otsu] Threshold: {threshold.item():.4f}, "
              f"σ²_B: {sigma_between[best_idx].item():.6f}")

        return (values > threshold).long()

    def _multi_threshold_depth(
            self,
            depth_norm: torch.Tensor,
            n_classes: int,
            device: str,
    ) -> torch.Tensor:
        if n_classes <= 1:
            return torch.zeros_like(depth_norm, dtype=torch.long)
        if n_classes == 2:
            return self._otsu_threshold(depth_norm)

        labels = self._otsu_threshold(depth_norm)
        current_label = 2
        remaining = n_classes - 2

        for parent_lbl in range(2):
            if remaining <= 0:
                break
            mask = labels == parent_lbl
            if mask.sum() < 10:
                continue
            sub_vals = depth_norm[mask]
            d_lo, d_hi = sub_vals.min(), sub_vals.max()
            if d_hi - d_lo < 0.05:
                continue

            sub_norm = (sub_vals - d_lo) / (d_hi - d_lo + 1e-8)
            sub_binary = self._otsu_threshold(
                sub_norm.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            thr_norm = sub_norm[sub_binary == 1].min() if sub_binary.any() else 0.5
            threshold = d_lo + thr_norm * (d_hi - d_lo)

            split_mask = mask & (depth_norm > threshold)
            labels[split_mask] = current_label
            current_label += 1
            remaining -= 1

        return labels

    def _merge_clusters(
            self,
            labels: torch.Tensor,
            signatures: Dict[int, Dict],
            target_k: int,
            merge_threshold: float,
    ) -> torch.Tensor:
        """Agglomerative merging of over-segmented clusters."""
        active_labels = list(signatures.keys())
        if len(active_labels) <= target_k:
            return labels

        n = len(active_labels)
        dist_matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(i + 1, n):
                si = signatures[active_labels[i]]
                sj = signatures[active_labels[j]]

                color_dist = (si['mean_color'] - sj['mean_color']).norm().item()

                normal_cos = F.cosine_similarity(
                    si['mean_normal'].unsqueeze(0),
                    sj['mean_normal'].unsqueeze(0),
                ).item()
                normal_dist = 1.0 - abs(normal_cos)

                opacity_dist = 0.0
                if 'mean_opacity' in si and 'mean_opacity' in sj:
                    opacity_dist = abs(
                        si['mean_opacity'].item() - sj['mean_opacity'].item()
                    )

                boundary_bonus = (
                    -0.1 if (si['touches_boundary'] and sj['touches_boundary'])
                    else 0.0
                )

                depth_dist = abs(si['mean_depth'].item() - sj['mean_depth'].item())

                d = (0.4 * color_dist
                     + 0.25 * normal_dist
                     + 0.15 * opacity_dist
                     + 0.2 * depth_dist
                     + boundary_bonus)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        merge_map = {l: l for l in active_labels}

        while len(set(merge_map.values())) > target_k:
            remaining = sorted(set(merge_map.values()))
            min_dist = float('inf')
            merge_pair = None

            for ii, li in enumerate(remaining):
                for jj in range(ii + 1, len(remaining)):
                    lj = remaining[jj]
                    oi = active_labels.index(li) if li in active_labels else None
                    oj = active_labels.index(lj) if lj in active_labels else None
                    if oi is None or oj is None:
                        continue
                    d = dist_matrix[oi, oj].item()
                    if d < min_dist:
                        min_dist = d
                        merge_pair = (li, lj)

            if merge_pair is None or min_dist > merge_threshold * 3:
                break

            li, lj = merge_pair
            for k in merge_map:
                if merge_map[k] == lj:
                    merge_map[k] = li
            print(f"[Merge] {lj} → {li} (dist={min_dist:.4f})")

        new_labels = labels.clone()
        for old_lbl, new_lbl in merge_map.items():
            if old_lbl != new_lbl:
                new_labels[labels == old_lbl] = new_lbl

        unique_new = new_labels.unique()
        for new_idx, old_val in enumerate(unique_new):
            new_labels[new_labels == old_val] = new_idx

        return new_labels

    def _refine_boundaries(
            self,
            labels: torch.Tensor,
            color_grid: torch.Tensor,
            normal_grid: torch.Tensor,
            depth_norm: torch.Tensor,
            opacity_grid: Optional[torch.Tensor],
            H: int, W: int,
            device: str,
            boundary_width: int = 2,
    ) -> torch.Tensor:
        refined = labels.clone()

        boundary_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        for di in range(-boundary_width, boundary_width + 1):
            for dj in range(-boundary_width, boundary_width + 1):
                if di == 0 and dj == 0:
                    continue
                src_r0 = max(0, di);
                src_r1 = min(H, H + di)
                src_c0 = max(0, dj);
                src_c1 = min(W, W + dj)
                tgt_r0 = max(0, -di);
                tgt_r1 = min(H, H - di)
                tgt_c0 = max(0, -dj);
                tgt_c1 = min(W, W - dj)

                if src_r0 >= src_r1 or src_c0 >= src_c1:
                    continue
                if tgt_r0 >= tgt_r1 or tgt_c0 >= tgt_c1:
                    continue

                shifted = labels[src_r0:src_r1, src_c0:src_c1]
                target = labels[tgt_r0:tgt_r1, tgt_c0:tgt_c1]
                boundary_mask[tgt_r0:tgt_r1, tgt_c0:tgt_c1] |= (shifted != target)

        if not boundary_mask.any():
            return refined

        n_boundary = boundary_mask.sum().item()
        print(f"[Boundary Refine] {n_boundary} boundary pixels "
              f"({n_boundary / (H * W) * 100:.1f}% of grid)")

        feat_list = [color_grid * 0.5, normal_grid * 0.3, depth_norm.unsqueeze(-1) * 0.2]
        if opacity_grid is not None:
            feat_list.append(opacity_grid * 0.1)
        features = torch.cat(feat_list, dim=-1)

        unique_labels = labels.unique()
        centres: Dict[int, torch.Tensor] = {}
        for lbl in unique_labels:
            interior = (labels == lbl) & (~boundary_mask)
            if interior.sum() > 0:
                centres[lbl.item()] = features[interior].mean(dim=0)

        if len(centres) < 2:
            return refined

        center_stack = torch.stack(list(centres.values()))
        center_labels = list(centres.keys())

        boundary_features = features[boundary_mask]
        dists = torch.cdist(boundary_features, center_stack)
        nearest = dists.argmin(dim=1)

        boundary_indices = boundary_mask.nonzero()
        for idx in range(boundary_indices.shape[0]):
            bi, bj = boundary_indices[idx]
            refined[bi, bj] = center_labels[nearest[idx].item()]

        return refined

    # ==================================================================
    #  LABEL ASSIGNMENT
    # ==================================================================

    def _assign_labels(
            self,
            label_map: torch.Tensor,
            color_grid: torch.Tensor,
            depth_norm: torch.Tensor,
            surface: "SplineModel",
            H: int, W: int,
            device: str,
    ) -> Dict:
        normal_grid = self._extract_control_normals(surface, H, W, device)

        opacity_grid = None
        if surface.refine_opacity_active:
            opacity_raw = surface.opacity.control_features.detach().view(H, W, 1)
            opacity_grid = torch.sigmoid(opacity_raw)

        unique = label_map.unique()
        stats = []

        for lbl in unique:
            mask = label_map == lbl
            score = 0.0

            if (mask[0, :].any() or mask[-1, :].any() or
                    mask[:, 0].any() or mask[:, -1].any()):
                score += 3.0

            mean_bright = color_grid[mask].mean().item()
            if mean_bright > 0.65:
                score += 2.0

            mean_depth = depth_norm[mask].mean().item()
            median_depth = depth_norm.median().item()
            if mean_depth > median_depth:
                score += 1.5

            normal_var = normal_grid[mask].var(dim=0).sum().item()
            if normal_var < 0.1:
                score += 1.0

            if opacity_grid is not None:
                mean_opa = opacity_grid[mask].mean().item()
                opa_var = opacity_grid[mask].var().item()
                if mean_opa > 0.7 and opa_var < 0.05:
                    score += 0.5

            size_frac = mask.float().mean().item()
            if size_frac > 0.5:
                score += 1.0

            stats.append({
                "label_val": lbl.item(),
                "mask": mask,
                "bg_score": score,
                "size": size_frac,
                "brightness": mean_bright,
                "normal_var": normal_var,
            })

        stats.sort(key=lambda x: x["bg_score"], reverse=True)

        masks, labels, is_background = [], [], []
        fg_count = 0
        for i, s in enumerate(stats):
            is_bg = (i == 0)
            masks.append(s["mask"])
            is_background.append(is_bg)
            if is_bg:
                labels.append("background")
                print(f"  [BG]  score={s['bg_score']:.1f}  size={s['size']:.1%}  "
                      f"bright={s['brightness']:.2f}  normal_var={s['normal_var']:.4f}")
            else:
                labels.append(f"object_{fg_count}")
                print(f"  [OBJ_{fg_count}]  score={s['bg_score']:.1f}  size={s['size']:.1%}  "
                      f"bright={s['brightness']:.2f}  normal_var={s['normal_var']:.4f}")
                fg_count += 1

        return {"masks": masks, "labels": labels, "is_background": is_background}

    # ==================================================================
    #  POSTPROCESS
    # ==================================================================

    def _postprocess_semantic(
            self,
            label_map: torch.Tensor,
            H: int, W: int,
    ) -> torch.Tensor:
        try:
            from scipy import ndimage
            from scipy.ndimage import distance_transform_edt

            arr = label_map.cpu().numpy().astype(np.int32)
            unique = np.unique(arr)
            total = H * W

            if self.cfg.morph_close_iters > 0:
                for lbl in unique:
                    mask = arr == lbl
                    closed = ndimage.binary_closing(mask, iterations=self.cfg.morph_close_iters)
                    fill_region = closed & ~np.isin(arr, [l for l in unique if l != lbl])
                    arr[fill_region] = lbl

            cleaned = np.full_like(arr, -1)
            valid_label = 0
            for lbl in unique:
                mask = arr == lbl
                labeled_array, num_features = ndimage.label(mask)
                for comp_id in range(1, num_features + 1):
                    comp_mask = labeled_array == comp_id
                    if comp_mask.sum() / total >= self.cfg.cc_min_area_frac:
                        cleaned[comp_mask] = valid_label
                if (cleaned == valid_label).any():
                    valid_label += 1

            if self.cfg.morph_erode_iters > 0:
                for lbl in range(valid_label):
                    mask = cleaned == lbl
                    eroded = ndimage.binary_erosion(mask, iterations=self.cfg.morph_erode_iters)
                    cleaned[mask & ~eroded] = -1

            unassigned = cleaned == -1
            if unassigned.any() and valid_label > 0:
                all_dists = np.full((valid_label, H, W), np.inf)
                for lbl in range(valid_label):
                    all_dists[lbl] = distance_transform_edt(cleaned != lbl)
                nearest_label = np.argmin(all_dists, axis=0)
                cleaned[unassigned] = nearest_label[unassigned]

            return torch.tensor(cleaned, device=label_map.device, dtype=torch.long)

        except ImportError:
            print("[Warning] scipy not available, falling back to basic postprocess")
            return self._postprocess(label_map, H, W)

    # ==================================================================
    #  UNCHANGED INTERNALS
    # ==================================================================

    def _aggregate_depth(
            self,
            H: int,
            W: int,
            device: str,
    ) -> torch.Tensor:
        stack = torch.stack(self._depth_buffer, dim=0).to(device)

        valid = (stack > 0.001) & torch.isfinite(stack)
        stack = stack.clone()
        stack[~valid] = float('nan')

        try:
            agg = stack.nanmedian(dim=0).values
        except AttributeError:
            masked = torch.where(valid, stack, torch.zeros_like(stack))
            agg = masked.sum(0) / valid.float().sum(0).clamp(min=1)

        nan_mask = torch.isnan(agg)
        if nan_mask.any():
            valid_max = agg[~nan_mask].max() if (~nan_mask).any() \
                else torch.tensor(10.0, device=device)
            agg = torch.where(nan_mask, valid_max * 2.0, agg)

        return agg

    def _smooth_grid(
            self,
            grid: torch.Tensor,
            sigma: float,
    ) -> torch.Tensor:
        if sigma <= 0:
            return grid

        H, W, C = grid.shape
        k = max(3, int(6 * sigma + 1) | 1)
        k = min(k, min(H, W) // 2 * 2 - 1)
        if k < 3:
            return grid

        device = grid.device
        pad = k // 2
        x = torch.arange(k, device=device, dtype=torch.float32) - pad
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)

        g = grid.permute(2, 0, 1).unsqueeze(0)
        padded = F.pad(g, (pad, pad, pad, pad), mode='reflect')
        out = []
        for c in range(C):
            ch = padded[:, c:c + 1]
            out.append(F.conv2d(ch, kernel_2d))
        return torch.cat(out, dim=1).squeeze(0).permute(1, 2, 0)

    def _kmeans(
            self,
            features: torch.Tensor,
            k: int,
            max_iter: int = 150,
            init_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H, W, D = features.shape
        device = features.device
        flat = features.reshape(-1, D)
        N = flat.shape[0]

        if init_labels is not None:
            centroids = []
            for j in range(k):
                mask = init_labels.reshape(-1) == j
                if mask.any():
                    centroids.append(flat[mask].mean(0))
                else:
                    centroids.append(flat[torch.randint(0, N, (1,)).item()])
            centroids = torch.stack(centroids)
        else:
            centroids = [flat[torch.randint(0, N, (1,)).item()]]
            for _ in range(1, k):
                dists = torch.stack([
                    torch.cdist(flat, c.unsqueeze(0)).squeeze() for c in centroids
                ], dim=1).min(dim=1).values
                probs = dists ** 2
                probs /= probs.sum()
                centroids.append(flat[torch.multinomial(probs, 1).item()])
            centroids = torch.stack(centroids)

        labels = torch.zeros(N, dtype=torch.long, device=device)
        for _ in range(max_iter):
            dists = torch.cdist(flat, centroids)
            new_labels = dists.argmin(dim=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            for j in range(k):
                m = labels == j
                if m.any():
                    centroids[j] = flat[m].mean(0)

        return labels.reshape(H, W)

    def _postprocess(
            self,
            label_map: torch.Tensor,
            H: int,
            W: int,
    ) -> torch.Tensor:
        try:
            from scipy import ndimage
            arr = label_map.cpu().numpy().astype(np.int32)
            unique = np.unique(arr)
            total = H * W

            out = np.zeros_like(arr)
            valid_label = 0
            for lbl in unique:
                mask = arr == lbl
                filled = ndimage.binary_fill_holes(mask)
                if filled.sum() / total >= self.cfg.min_component_frac:
                    eroded = ndimage.binary_erosion(filled, iterations=2)
                    out[eroded] = valid_label
                    valid_label += 1

            unassigned = out == 0
            if unassigned.any() and valid_label > 1:
                from scipy.ndimage import distance_transform_edt
                for lbl in range(1, valid_label):
                    dist = distance_transform_edt(out != lbl)
                    out[unassigned & (dist < 4)] = lbl

            return torch.tensor(out, device=label_map.device, dtype=torch.long)
        except ImportError:
            return label_map

    # ==================================================================
    #  LEGACY: subsurface building (kept for DEPTH_UV, SEMANTIC, DEPTH_SEMANTIC)
    # ==================================================================

    def _build_subsurface(
            self,
            parent: "SplineModel",
            mask: torch.Tensor,
            label: str,
            is_background: bool,
            config,
            args,
            train_cam_uids: List,
    ) -> "SplineModel":
        from geomdl import BSpline
        from model.modules.KnotSurface import SplineModel

        H, W = parent.state.H, parent.state.W
        degree = parent.state.degree
        device = parent.device

        rows = torch.where(mask.any(dim=1))[0]
        cols = torch.where(mask.any(dim=0))[0]
        if rows.numel() == 0 or cols.numel() == 0:
            raise ValueError(f"Empty mask for '{label}'")

        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1

        min_ctrl = degree + 2
        r1 = max(r1, r0 + min_ctrl)
        c1 = max(c1, c0 + min_ctrl)
        r1, c1 = min(r1, H), min(c1, W)
        sub_H, sub_W = r1 - r0, c1 - c0

        if is_background:
            tgt_H = max(min_ctrl, int(sub_H * 0.6))
            tgt_W = max(min_ctrl, int(sub_W * 0.6))
        else:
            tgt_H = max(min_ctrl, sub_H)
            tgt_W = max(min_ctrl, sub_W)

        print(f"  [{label}] BB=[{r0}:{r1}, {c0}:{c1}]  "
              f"sub={sub_H}x{sub_W} → target={tgt_H}x{tgt_W}")

        pos_ctrl = parent.position.control_features.detach().clone().view(H, W, 3)
        sub_pos = self._resample(pos_ctrl[r0:r1, c0:c1], tgt_H, tgt_W)

        sh_dc = parent.spherical_harmonics.sh_dc.control_features.detach().clone().view(H, W, 3)
        sub_rgb_sh = sh_dc[r0:r1, c0:c1]
        sub_rgb = (self._resample(sub_rgb_sh, tgt_H, tgt_W) * 0.28209479177387814 + 0.5).clamp(0, 1)

        geo_surf = self._make_geomdl_surface(sub_pos, tgt_H, tgt_W, degree)
        rgb_surf = self._make_geomdl_surface(sub_rgb, tgt_H, tgt_W, degree)

        spline_model = SplineModel(
            surf=geo_surf,
            surf_rgb=rgb_surf,
            config=config,
            args=args,
            spatial_lr_scale=parent.spatial_lr_scale,
            train_cam_uids=train_cam_uids,
            late_init=False,
            surf_uid=hash(label) % 1000,
            skip_opt=True,
            label=label,
            is_background=is_background,
        )

        self._transfer_features(parent, spline_model, r0, r1, c0, c1, tgt_H, tgt_W)
        return spline_model

    def _transfer_features(
            self,
            parent: "SplineModel",
            child: "SplineModel",
            r0: int, r1: int,
            c0: int, c1: int,
            tgt_H: int,
            tgt_W: int,
    ):
        H, W = parent.state.H, parent.state.W
        device = parent.device

        with torch.no_grad():

            def xfer(src_cf, dst_cf, C: int):
                src = src_cf.detach().view(H, W, C)
                sub = src[r0:r1, c0:c1]
                resampled = self._resample(sub, tgt_H, tgt_W)
                target_flat = resampled.reshape(-1, C)
                dst_cf = dst_cf.reshape_as(target_flat)
                if dst_cf.data.shape != target_flat.shape:
                    raise RuntimeError(
                        f"Shape mismatch in xfer: dst={dst_cf.data.shape}, "
                        f"src_after_resample={target_flat.shape}, C={C}"
                    )
                dst_cf.data.copy_(target_flat)

            def xfer_sh_rest(src_cf, dst_cf, num_coeffs: int):
                src = src_cf.detach().view(H, W, num_coeffs, 3)
                sub = src[r0:r1, c0:c1]
                sub_H, sub_W = sub.shape[:2]
                sub_spatial = sub.permute(2, 3, 0, 1).reshape(1, num_coeffs * 3, sub_H, sub_W)
                resampled = F.interpolate(
                    sub_spatial.float(), size=(tgt_H, tgt_W),
                    mode='bilinear', align_corners=True
                )
                resampled = resampled.squeeze(0).reshape(num_coeffs, 3, tgt_H, tgt_W)
                resampled = resampled.permute(2, 3, 0, 1).reshape(-1, 3)
                dst_cf = dst_cf.reshape_as(resampled)
                if dst_cf.data.shape != resampled.shape:
                    raise RuntimeError(
                        f"Shape mismatch in xfer_sh_rest: "
                        f"dst={dst_cf.data.shape}, "
                        f"src_after_resample={resampled.shape}, "
                        f"num_coeffs={num_coeffs}, tgt=({tgt_H},{tgt_W})"
                    )
                dst_cf.data.copy_(resampled)

            xfer(parent.position.features, child.position.features, 3)

            xfer(
                parent.spherical_harmonics.sh_dc.features.detach().clone(),
                child.spherical_harmonics.sh_dc.features.detach().clone(),
                3
            )

            parent_num_coeffs = parent.spherical_harmonics.sh_rest.num_sh_coeffs
            child_num_coeffs = child.spherical_harmonics.sh_rest.num_sh_coeffs

            if parent_num_coeffs == child_num_coeffs:
                xfer_sh_rest(
                    parent.spherical_harmonics.sh_rest.features,
                    child.spherical_harmonics.sh_rest.features,
                    parent_num_coeffs
                )
            else:
                print(
                    f"  [Transfer] WARNING: SH rest coeff mismatch "
                    f"parent={parent_num_coeffs} child={child_num_coeffs}. "
                    f"Transferring min({parent_num_coeffs},{child_num_coeffs}) coeffs, "
                    f"zeroing remainder."
                )
                min_coeffs = min(parent_num_coeffs, child_num_coeffs)
                src = parent.spherical_harmonics.sh_rest.features.detach()
                src = src.view(H, W, parent_num_coeffs, 3)
                src_min = src[:, :, :min_coeffs, :].contiguous()

                sub = src_min[r0:r1, c0:c1]
                sub_H, sub_W = sub.shape[:2]
                sub_s = sub.permute(2, 3, 0, 1).reshape(1, min_coeffs * 3, sub_H, sub_W)
                resampled = F.interpolate(sub_s.float(), size=(tgt_H, tgt_W),
                                          mode='bilinear', align_corners=True)
                resampled = resampled.squeeze(0).reshape(min_coeffs, 3, tgt_H, tgt_W)
                resampled = resampled.permute(2, 3, 0, 1).reshape(-1, 3)

                dst = child.spherical_harmonics.sh_rest.features
                dst = dst.view(tgt_H, tgt_W, child_num_coeffs, 3)
                dst[:, :, :min_coeffs, :] = resampled.view(tgt_H, tgt_W, min_coeffs, 3)

            if (parent.refine_opacity_active and child.refine_opacity_active
                    and parent.opacity.features is not None):
                xfer(parent.opacity.features, child.opacity.features, 1)

            if (parent.refine_scales_active and child.refine_scales_active
                    and parent.scaling.features is not None):
                scaling_C = parent.scaling.features.shape[-1]
                xfer(parent.scaling.features, child.scaling.features, scaling_C)

            if (parent.refine_rotations_active and child.refine_rotations_active
                    and parent.rotation.features is not None):
                xfer(parent.rotation.features, child.rotation.features, 4)

        print(
            f"  [Transfer] Complete: parent [{H}x{W}] crop [{r0}:{r1},{c0}:{c1}]"
            f" → child [{tgt_H}x{tgt_W}]"
        )

    def _reset_child_caches(self, new_model):
        """
        Reset ALL caches on all child surfaces after decomposition.
        Called AFTER del old_model + empty_cache(), so all new allocations
        are from clean CUDA memory.
        """
        for surface in new_model.surfaces:
            Us, Vs = surface.state.Us, surface.state.Vs

            for module in surface.control_list:
                if module.control_features is None:
                    continue

                C = module.control_features.shape[-1]
                module._cache = None
                surface.invalidate_all_caches(force=True)
                surface.state.init_grad_accumulators()
            surface.basis.recompute()

            assert surface.basis.bu is not None, \
                f"basis.bu is None after recompute for surface '{surface.label}'"
            assert not surface.basis.bu.isnan().any(), \
                f"basis.bu contains NaN for surface '{surface.label}'"
            Us, Vs = surface.state.Us, surface.state.Vs

            for module in surface.control_list:
                if module.control_features is None:
                    continue

                C = module.control_features.shape[-1]
                module._cache = None
                module._previous_cache = torch.zeros(Us, Vs, C, device=surface.device)
                module.set_alpha(0.0)

            surface.invalidate_all_caches(force=True)
            surface.basis.recompute()

    # ==================================================================
    #  UTILITIES
    # ==================================================================

    def _resample(
            self,
            grid: torch.Tensor,
            tgt_H: int,
            tgt_W: int,
    ) -> torch.Tensor:
        sub_H, sub_W, C = grid.shape
        if sub_H == tgt_H and sub_W == tgt_W:
            return grid.clone()
        g = grid.permute(2, 0, 1).unsqueeze(0).float()
        r = F.interpolate(g, size=(tgt_H, tgt_W), mode='bilinear', align_corners=True)
        return r.squeeze(0).permute(1, 2, 0)

    def _make_geomdl_surface(
            self,
            ctrl_pts: torch.Tensor,
            H: int,
            W: int,
            degree: int,
    ):
        from geomdl import BSpline

        surf = BSpline.Surface()
        surf.degree_u = degree
        surf.degree_v = degree

        ctrlpts = ctrl_pts.reshape(-1, 3).detach().clone().cpu().tolist()
        surf.set_ctrlpts(ctrlpts, H, W)

        surf.knotvector_u = self._uniform_knots(H, degree)
        surf.knotvector_v = self._uniform_knots(W, degree)
        return surf

    @staticmethod
    def _uniform_knots(n: int, degree: int) -> List[float]:
        n_internal = n - degree - 1
        if n_internal <= 0:
            internal = []
        else:
            internal = np.linspace(0, 1, n_internal + 2)[1:-1].tolist()
        return [0.0] * (degree + 1) + internal + [1.0] * (degree + 1)


def safe_decompose(controller, model, config, args):
    """
    Wrapper that ensures clean CUDA state before and after decomposition.
    """
    import torch

    # === Step 1: Synchronize and verify clean state ===
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # === Step 2: Decompose ===
    new_model = controller.decompose(model, config, args)

    # === Step 3: Delete old model BEFORE any new allocations ===
    model._invalidate_cache(True)

    del model._surfaces[0]
    del model
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # === Step 4: Reset caches on new model ===
    controller._reset_child_caches(new_model)

    # === Step 5: Setup optimizer ===
    new_model.training_setup()
    new_model._update_surface_offsets()
    # for surf in new_model.surfaces:
    #     surf.state.opt.sampling_density = 0.25
    #     surf.update_sampling_density(surf.state.opt.sampling_density)
    #     from model.modules.sampling.SamplerUV import SamplerUV
    #
    #     surf.uv_sampler = SamplerUV(
    #         state=surf.state,
    #         mode='single',
    #     )
    #     from model.modules.basis import BasisFunction
    #     surf.basis = BasisFunction(surf.state, surf.uv_sampler, knot_u=surf.knot_u, knot_v=surf.knot_v)

    # === Step 6: Final sync and validation ===
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    #
    # # Validate all surfaces can evaluate without crashing
    # for i, surf in enumerate(new_model.surfaces):
    #     try:
    #         with torch.no_grad():
    #             xyz = surf.get_xyz
    #             feat = surf.get_features
    #             opa = surf.get_opacity
    #             scale = surf.get_scaling
    #             rot = surf.get_rotation
    #             print(f"  [Validate] Surface {i} '{surf.label}': "
    #                   f"{xyz.shape[0]} Gaussians OK")
    #     except Exception as e:
    #         print(f"  [CRITICAL] Surface {i} '{surf.label}' FAILED: {e}")
    #         raise

    # torch.cuda.synchronize()
    return new_model


def _validate_contour_decomposition(model: "MultiSurfaceSplineModel"):
    """
    Post-decomposition sanity checks for contour mode.
    Ensures the two surfaces are actually disjoint.
    """
    if len(model.surfaces) < 2:
        return

    bg_surf = model.surfaces[0]
    fg_surf = model.surfaces[1]

    with torch.no_grad():
        bg_opacity = torch.sigmoid(
            bg_surf.opacity.control_features.data.view(
                bg_surf.state.H, bg_surf.state.W
            )
        )
        fg_opacity = torch.sigmoid(
            fg_surf.opacity.control_features.data.view(
                fg_surf.state.H, fg_surf.state.W
            )
        )

        # Check complementarity
        total = bg_opacity + fg_opacity
        complementarity_err = (total - 1.0).abs().mean().item()

        # Check overlap (both > 0.3)
        overlap = ((bg_opacity > 0.3) & (fg_opacity > 0.3)).float().mean().item()

        # Check that BG and FG have distinct active regions
        bg_active = (bg_opacity > 0.5).float().mean().item()
        fg_active = (fg_opacity > 0.5).float().mean().item()

        print(f"\n[Validation] Contour decomposition check:")
        print(f"  Complementarity error: {complementarity_err:.6f} "
              f"({'OK' if complementarity_err < 0.05 else 'WARNING'})")
        print(f"  Overlap (both > 0.3): {overlap:.1%} "
              f"({'OK' if overlap < 0.15 else 'WARNING: high overlap'})")
        print(f"  BG active (>0.5): {bg_active:.1%}")
        print(f"  FG active (>0.5): {fg_active:.1%}")

        if overlap > 0.5:
            print(f"  [WARNING] Overlap is {overlap:.1%} — surfaces are NOT disjoint!")
            print(f"  This suggests the level-set field did not produce a clean contour.")
            print(f"  Consider: increasing contour_smooth_iters, adjusting depth/color weights,")
            print(f"  or checking that the warmup surface has meaningful depth variation.")