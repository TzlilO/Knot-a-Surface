"""
Concrete integration of hybrid depth-contour decomposition
into WarmupDecompositionController.

Design constraints (from examining the codebase):
1. Must return MultiSurfaceSplineModel from decompose()
2. Must reuse _resample_grid_to_mask() for surface construction
3. Must fit the SegmentationMode enum dispatch pattern
4. Must not modify KnotSurface.py or multisurf.py
5. Must produce disjoint FG/BG masks (not overlapping)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from enum import Enum

from modules.decompose import DecompPhase
from modules.fitting.nurbs_from_pointcloud import DecompositionMode

# ======================================================================
# Step 1: Extend the SegmentationMode enum
# ======================================================================

# In the actual file, add to the existing enum:
#
# class SegmentationMode(Enum):
#     DEPTH_UV = "depth_uv"
#     SEMANTIC = "semantic"
#     DEPTH_SEMANTIC = "depth_semantic"
#     CONTOUR = "contour"
#     HYBRID = "hybrid"          # <-- NEW


# ======================================================================
# Step 2: Add hybrid config fields to ControllerConfig
# ======================================================================

# These go into the existing ControllerConfig dataclass.
# Grouped under a new section header.

HYBRID_CONFIG_FIELDS = """
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
"""


# ======================================================================
# Step 3: The Hybrid Pipeline — Self-Contained Module
# ======================================================================

class HybridLevelSetPipeline:
    """
    Encapsulates the full hybrid pipeline:
        aggregated_depth + surface → binary FG mask

    This is a STATELESS helper. The controller owns all state.
    All methods are pure functions of their inputs.

    The pipeline replaces _build_level_set_field + _smooth_level_set +
    _extract_contour_mask from the CONTOUR mode with a principled
    variational formulation.
    """

    def __init__(self, cfg: "ControllerConfig"):
        self.cfg = cfg

    def run(
        self,
        surface: "SplineModel",
        agg_depth: torch.Tensor,  # [H, W]
        H: int, W: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline: depth + surface features → optimized mask.

        Returns:
            fg_mask:    [H, W] bool — binary foreground mask
            phi:        [H, W] float — optimized level-set field
            confidence: [H, W] float — per-point confidence in [0, 1]
        """
        # Phase 1: Continuous depth evidence
        evidence, threshold, depth_confidence = self._build_depth_evidence(
            agg_depth, H, W, device
        )

        # Phase 2: Edge indicator for anisotropic smoothness
        edge_indicator = self._compute_edge_indicator(
            surface, H, W, device
        )

        # Phase 3: Variational level-set optimization
        phi = self._solve_level_set(
            evidence, edge_indicator, H, W, device
        )

        # Phase 4: Extract mask with confidence
        fg_mask, confidence = self._extract_mask(phi, H, W, device)

        return fg_mask, phi, confidence

    # ------------------------------------------------------------------
    # Phase 1: Depth Evidence
    # ------------------------------------------------------------------

    def _build_depth_evidence(
        self,
        agg_depth: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Continuous depth evidence E_d(u,v) ∈ [-1, +1].

        Key improvement over CONTOUR mode's binary Otsu:
        - Preserves continuous information near the threshold
        - Locally adaptive sharpness (uncertain where depth varies locally)
        - Confidence output enables downstream boundary handling

        +1 = strong FG evidence (close), -1 = strong BG evidence (far)
        """
        d_min, d_max = agg_depth.min(), agg_depth.max()
        d_range = d_max - d_min

        if d_range < 1e-6:
            print("[HybridPipeline] WARNING: Near-zero depth range.")
            return (
                torch.zeros(H, W, device=device),
                0.5,
                torch.zeros(H, W, device=device),
            )

        depth_norm = (agg_depth - d_min) / (d_range + 1e-8)

        # Find threshold (reuse existing Otsu or use histogram gap)
        threshold = self._find_depth_threshold(depth_norm, device)

        # Local depth variance → adaptive sharpness
        local_var = self._local_variance_2d(
            depth_norm, self.cfg.hybrid_local_sigma_kernel, device
        )
        var_norm = local_var / (local_var.max() + 1e-8)

        # High local variance → lower sharpness (more uncertainty)
        alpha = self.cfg.hybrid_depth_sharpness
        adaptive_alpha = alpha / (1.0 + 2.0 * var_norm)

        # Continuous evidence via tanh
        # Negative sign: lower depth (closer) → positive (FG)
        evidence = -torch.tanh(adaptive_alpha * (depth_norm - threshold))

        # Confidence from evidence magnitude + depth range quality
        range_confidence = (d_range / (d_max + 1e-8)).clamp(0, 1)
        confidence = evidence.abs() * range_confidence

        fg_frac = (evidence > 0).float().mean().item()
        print(f"[HybridPipeline] Depth evidence: threshold={threshold:.4f}, "
              f"range_conf={range_confidence:.3f}, FG(evidence>0)={fg_frac:.1%}")

        return evidence, threshold, confidence

    def _find_depth_threshold(
        self, depth_norm: torch.Tensor, device: str
    ) -> float:
        """Dispatch to threshold method."""
        method = self.cfg.hybrid_depth_method

        if method == "otsu":
            return self._otsu_threshold_scalar(depth_norm, device)
        elif method == "histogram_gap":
            return self._histogram_gap_threshold(depth_norm, device)
        elif method == "quantile":
            return self.cfg.depth_fg_quantile
        else:
            raise ValueError(f"Unknown threshold method: {method}")

    def _otsu_threshold_scalar(
        self, values: torch.Tensor, device: str, num_bins: int = 256
    ) -> float:
        """Otsu's method returning a scalar threshold value."""
        flat = values.reshape(-1)
        hist = torch.histc(flat, bins=num_bins, min=0.0, max=1.0)
        bin_centers = torch.linspace(0, 1, num_bins, device=device)
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

        return bin_centers[sigma_between.argmax()].item()

    def _histogram_gap_threshold(
        self, values: torch.Tensor, device: str, num_bins: int = 128
    ) -> float:
        """Find largest valley in smoothed depth histogram."""
        flat = values.reshape(-1)
        hist = torch.histc(flat, bins=num_bins, min=0.0, max=1.0)
        bin_centers = torch.linspace(0, 1, num_bins, device=device)

        # Smooth histogram
        kernel = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32, device=device)
        kernel = kernel / kernel.sum()
        hist_smooth = F.conv1d(
            hist.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=2
        ).squeeze()

        # Find deepest valley in interior
        margin = num_bins // 10
        interior = hist_smooth[margin:-margin]
        valley_idx = interior.argmin().item() + margin
        return bin_centers[valley_idx].item()

    def _local_variance_2d(
        self, grid: torch.Tensor, kernel_size: int, device: str
    ) -> torch.Tensor:
        """Local variance of a 2D scalar field."""
        H, W = grid.shape
        pad = kernel_size // 2
        g = grid.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        kernel = kernel / (kernel_size ** 2)
        padded = F.pad(g, (pad, pad, pad, pad), mode='reflect')
        local_mean = F.conv2d(padded, kernel)
        local_mean_sq = F.conv2d(padded ** 2, kernel)
        return (local_mean_sq - local_mean ** 2).clamp(min=0).squeeze()

    # ------------------------------------------------------------------
    # Phase 2: Edge Indicator
    # ------------------------------------------------------------------

    def _compute_edge_indicator(
        self,
        surface: "SplineModel",
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """
        Edge indicator g(u,v) ∈ (0, 1].

        g = 1/(1 + β·|∇I|²)

        This is the KEY FUSION POINT. Instead of adding color/normal cues
        to the data term (which causes textured-background false positives
        in CONTOUR mode), we use them to modulate WHERE the contour can
        be sharp vs. where it must be smooth.

        g ≈ 1: homogeneous region → full smoothing → contour follows depth
        g ≈ 0: feature edge → low smoothing → contour can follow the edge
        """
        from utils.sh_utils import SH2RGB

        grad_mag_sq = torch.zeros(H, W, device=device)

        if self.cfg.hybrid_edge_from_color:
            sh_dc = surface.spherical_harmonics.sh_dc.control_features.detach()
            color = SH2RGB(sh_dc.view(H, W, 3)).clamp(0, 1)

            # Forward-difference gradient (Sobel would be better but this
            # matches the grid's finite-difference scheme for consistency)
            du_color = torch.zeros_like(color)
            dv_color = torch.zeros_like(color)
            du_color[:-1] = color[1:] - color[:-1]
            du_color[-1] = du_color[-2]
            dv_color[:, :-1] = color[:, 1:] - color[:, :-1]
            dv_color[:, -1] = dv_color[:, -2]

            grad_mag_sq += (du_color.pow(2) + dv_color.pow(2)).sum(dim=-1)

        if self.cfg.hybrid_edge_from_normals:
            pos = surface.position.control_features.detach().view(H, W, 3)
            du = torch.zeros_like(pos); du[:-1] = pos[1:] - pos[:-1]; du[-1] = du[-2]
            dv = torch.zeros_like(pos); dv[:, :-1] = pos[:, 1:] - pos[:, :-1]; dv[:, -1] = dv[:, -2]
            normals = F.normalize(torch.cross(du, dv, dim=-1), dim=-1, eps=1e-8)

            du_n = torch.zeros_like(normals); du_n[:-1] = normals[1:] - normals[:-1]; du_n[-1] = du_n[-2]
            dv_n = torch.zeros_like(normals); dv_n[:, :-1] = normals[:, 1:] - normals[:, :-1]; dv_n[:, -1] = dv_n[:, -2]

            grad_mag_sq += (du_n.pow(2) + dv_n.pow(2)).sum(dim=-1)

        # Normalize to [0, 1]
        gmax = grad_mag_sq.max()
        if gmax > 1e-8:
            grad_mag_sq = grad_mag_sq / gmax

        # Edge indicator: g = 1/(1 + β|∇|²)
        beta = self.cfg.hybrid_edge_sensitivity / (grad_mag_sq.mean() + 1e-8)
        g = 1.0 / (1.0 + beta * grad_mag_sq)

        print(f"[HybridPipeline] Edge indicator: min={g.min():.3f}, "
              f"max={g.max():.3f}, mean={g.mean():.3f}")

        return g

    # ------------------------------------------------------------------
    # Phase 3: Variational Level-Set Solver
    # ------------------------------------------------------------------

    def _solve_level_set(
        self,
        depth_evidence: torch.Tensor,  # [H, W] in [-1, +1]
        edge_indicator: torch.Tensor,  # [H, W] in (0, 1]
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """
        Minimize E(φ) = λ_d·E_data + λ_s·E_smooth + λ_a·E_area
        via gradient descent on the discrete grid.

        E_data   = ∫|H(φ) - E_d_01|²    (match depth evidence)
        E_smooth = ∫ g·|∇φ|²             (edge-aware regularization)
        E_area   = (∫H(φ) - A_target)²   (prevent degenerate masks)

        This replaces CONTOUR mode's:
          _build_level_set_field (ad-hoc weighted sum)
          + _smooth_level_set (uniform Laplacian)
        with a single principled energy minimization.
        """
        eps = self.cfg.hybrid_solver_epsilon
        dt = self.cfg.hybrid_solver_dt
        lam_d = self.cfg.hybrid_lambda_data
        lam_s = self.cfg.hybrid_lambda_smooth
        lam_a = self.cfg.hybrid_lambda_area
        A_target = self.cfg.hybrid_target_fg_fraction
        n_iters = self.cfg.hybrid_solver_iterations

        # CFL stability check
        h = 1.0 / max(H, W)
        dt_max = h * h / (4.0 * lam_s + 1e-8)
        if dt > dt_max:
            print(f"[HybridPipeline] WARNING: dt={dt} > CFL limit {dt_max:.4f}. "
                  f"Clamping to {dt_max:.4f}.")
            dt = dt_max * 0.9

        # Initialize φ from depth evidence (warm start)
        phi = depth_evidence.clone()

        # Map evidence to [0,1] for Heaviside comparison
        E_d_01 = (depth_evidence + 1.0) / 2.0

        for it in range(n_iters):
            # Smooth Heaviside and derivative
            H_phi = torch.sigmoid(phi / eps)
            delta_phi = H_phi * (1.0 - H_phi) / eps

            # --- Data gradient: ∂E_data/∂φ = 2(H(φ) - E_d) · δ(φ) ---
            grad_data = 2.0 * (H_phi - E_d_01) * delta_phi

            # --- Smoothness gradient: -div(g·∇φ) ---
            # Weighted Laplacian with edge indicator g
            grad_smooth = self._weighted_laplacian(phi, edge_indicator, H, W)

            # --- Area gradient: 2(area - A_target) · δ(φ) / (H·W) ---
            area = H_phi.mean()
            grad_area = 2.0 * (area - A_target) * delta_phi / (H * W)

            # --- Combined update ---
            grad_total = lam_d * grad_data - lam_s * grad_smooth + lam_a * grad_area
            phi = phi - dt * grad_total

            if it % max(n_iters // 5, 1) == 0 or it == n_iters - 1:
                fg_frac = (phi > 0).float().mean().item()
                e_data = ((H_phi - E_d_01) ** 2).mean().item()
                print(f"  [LevelSet {it:3d}/{n_iters}] "
                      f"FG={fg_frac:.1%} E_data={e_data:.4f} area={area:.3f}")

        return phi

    def _weighted_laplacian(
        self,
        phi: torch.Tensor,
        g: torch.Tensor,
        H: int, W: int,
    ) -> torch.Tensor:
        """
        div(g·∇φ) via finite differences.

        For each point, sum over 4-connected neighbors:
            Σ g_avg · (φ_neighbor - φ_center)
        where g_avg = (g_center + g_neighbor) / 2
        """
        result = torch.zeros_like(phi)

        # Up/down
        g_avg_u = (g[1:, :] + g[:-1, :]) / 2.0
        diff_u = phi[1:, :] - phi[:-1, :]
        flux_u = g_avg_u * diff_u
        result[:-1, :] += flux_u
        result[1:, :] -= flux_u

        # Left/right
        g_avg_v = (g[:, 1:] + g[:, :-1]) / 2.0
        diff_v = phi[:, 1:] - phi[:, :-1]
        flux_v = g_avg_v * diff_v
        result[:, :-1] += flux_v
        result[:, 1:] -= flux_v

        return result

    # ------------------------------------------------------------------
    # Phase 4: Mask Extraction
    # ------------------------------------------------------------------

    def _extract_mask(
        self,
        phi: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract binary mask + confidence from optimized φ.

        The binary mask goes to _resample_grid_to_mask (existing code).
        The confidence is logged for diagnostics.
        """
        fg_mask = phi > 0

        # Confidence from φ magnitude
        tau = self.cfg.hybrid_solver_epsilon * 2.0
        confidence = torch.sigmoid(phi.abs() / tau)

        # Morphological cleanup — reuse existing infrastructure
        fg_mask = self._cleanup(fg_mask, H, W, device)

        return fg_mask, confidence

    def _cleanup(
        self,
        mask: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """Minimal cleanup. The variational solver already produces clean masks,
        so we only need hole filling and tiny-component removal."""
        try:
            from scipy import ndimage

            arr = mask.cpu().numpy().astype(np.int32)
            total = H * W

            # Fill holes
            arr = ndimage.binary_fill_holes(arr).astype(np.int32)

            # Remove tiny components
            labeled, n_feat = ndimage.label(arr)
            for cid in range(1, n_feat + 1):
                comp = labeled == cid
                if comp.sum() / total < self.cfg.cc_min_area_frac:
                    arr[comp] = 0

            # Optional dilation
            if self.cfg.hybrid_boundary_dilate > 0:
                arr = ndimage.binary_dilation(
                    arr, iterations=self.cfg.hybrid_boundary_dilate
                ).astype(np.int32)

            return torch.tensor(arr, device=device, dtype=torch.bool)

        except ImportError:
            return mask


# ======================================================================
# Step 4: Integration into WarmupDecompositionController
# ======================================================================

def _decompose_hybrid(
    controller: "WarmupDecompositionController",
    model: "MultiSurfaceSplineModel",
    surface: "SplineModel",
    agg_depth: torch.Tensor,
    config,
    args,
    train_cam_uids: list,
    device: str,
) -> "MultiSurfaceSplineModel":
    """
    Drop-in replacement for _decompose_contour.

    This function has the EXACT same signature and return type.
    It can be monkey-patched or called via dispatch.

    Integration point: add to WarmupDecompositionController.decompose():

        if self.cfg.segmentation_mode == SegmentationMode.HYBRID:
            return _decompose_hybrid(
                self, model, surface, agg_depth, config, args,
                train_cam_uids, device
            )
    """
    H, W = surface.state.H, surface.state.W
    cfg = controller.cfg

    print(f"\n[Hybrid Decomp] Grid: {H}×{W}")

    # ============ Run hybrid pipeline ============
    pipeline = HybridLevelSetPipeline(cfg)
    fg_mask, phi, confidence = pipeline.run(surface, agg_depth, H, W, device)

    # ============ Validate + adjust coverage (same logic as CONTOUR) ============
    fg_frac = fg_mask.float().mean().item()
    print(f"[Hybrid Decomp] Raw FG fraction: {fg_frac:.1%}")

    if fg_frac < cfg.contour_min_fg_frac:
        print(f"[Hybrid Decomp] FG too small ({fg_frac:.1%}). "
              f"Lowering threshold.")
        phi_sorted = phi.flatten().sort().values
        target_idx = int((1 - cfg.contour_min_fg_frac * 1.5) * phi_sorted.numel())
        new_threshold = phi_sorted[target_idx].item()
        fg_mask = phi > new_threshold
        fg_mask = pipeline._cleanup(fg_mask, H, W, device)
        fg_frac = fg_mask.float().mean().item()

    if fg_frac > cfg.contour_max_fg_frac:
        print(f"[Hybrid Decomp] FG too large ({fg_frac:.1%}). "
              f"Raising threshold.")
        phi_sorted = phi.flatten().sort().values
        target_idx = int((1 - cfg.contour_max_fg_frac * 0.8) * phi_sorted.numel())
        new_threshold = phi_sorted[target_idx].item()
        fg_mask = phi > new_threshold
        fg_mask = pipeline._cleanup(fg_mask, H, W, device)
        fg_frac = fg_mask.float().mean().item()

    bg_mask = ~fg_mask

    # ============ Log confidence at boundary ============
    boundary = confidence < cfg.hybrid_confidence_threshold
    boundary_frac = boundary.float().mean().item()
    print(f"[Hybrid Decomp] FG: {fg_frac:.1%}, "
          f"Boundary (low confidence): {boundary_frac:.1%}")

    # ============ Resample grids (reuse existing infrastructure) ============
    # This is UNCHANGED from _decompose_contour — the mask is just
    # produced by a better algorithm.

    bg_surface = controller._resample_grid_to_mask(
        surface, bg_mask, "background", is_background=True,
        config=config, args=args, train_cam_uids=train_cam_uids,
    )
    fg_surface = controller._resample_grid_to_mask(
        surface, fg_mask, "object_0", is_background=False,
        config=config, args=args, train_cam_uids=train_cam_uids,
    )

    # ============ Diagnostics ============
    bg_H, bg_W = bg_surface.state.H, bg_surface.state.W
    fg_H, fg_W = fg_surface.state.H, fg_surface.state.W
    total_new = bg_H * bg_W + fg_H * fg_W
    total_orig = H * W

    print(f"\n[Hybrid Decomp Summary]")
    print(f"  Original: {H}×{W} = {total_orig}")
    print(f"  BG: {bg_H}×{bg_W} = {bg_H * bg_W} ({bg_H * bg_W / total_orig:.1%})")
    print(f"  FG: {fg_H}×{fg_W} = {fg_H * fg_W} ({fg_H * fg_W / total_orig:.1%})")
    print(f"  Total: {total_new} ({total_new / total_orig:.1%})")

    # ============ Assemble model ============

    new_model = model.__class__(
        surfaces=[bg_surface, fg_surface],
        labels=["background", "object_0"],
        decomposition_mode=DecompositionMode.BACKGROUND_OBJECT,
        device=device,
        setup_training=False,
    )

    controller.phase = controller.__class__.__dict__['DecompPhase'].DECOMPOSED \
        if hasattr(controller, 'DecompPhase') else None
    # Safer:
    controller.phase = DecompPhase.DECOMPOSED

    print(f"[Hybrid Decomp] Done. {new_model}")
    return new_model