"""
Hybrid Depth-Contour Decomposition.

Fuses UV depth-map evidence with variational level-set optimization
to produce topology-aware, smooth segmentation masks.

Key differences from existing modes:
1. Depth evidence is continuous (not binary Otsu)
2. Level-set is optimized variationally (not heuristic construction)
3. Color/normal cues modulate the smoothness term (not the data term)
4. Boundary region gets explicit handling during resampling
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from enum import Enum


@dataclass
class HybridDecompConfig:
    """Configuration for the hybrid depth-contour decomposition."""

    # --- Depth Evidence ---
    depth_evidence_sharpness: float = 3.0
    """Controls tanh transition sharpness around depth threshold.
    Higher → sharper binary-like evidence. Lower → softer, more uncertainty.
    The key insight: this should be SCENE-ADAPTIVE based on depth histogram 
    separation. If foreground and background depths are well-separated, 
    increase this. If they overlap, decrease it."""

    depth_threshold_method: str = "otsu"
    """How to find the depth cut: 'otsu', 'histogram_gap', 'quantile'.
    'histogram_gap' finds the largest gap in the depth histogram — more 
    robust than Otsu for multi-modal distributions."""

    depth_local_sigma_kernel: int = 5
    """Kernel size for computing local depth variance σ_d(u,v).
    This makes the evidence field adaptive: regions with high local depth 
    variance get softer evidence (more uncertainty), while regions with 
    uniform depth get hard evidence."""

    # --- Color/Normal Modulation ---
    color_variance_as_smoothness_weight: bool = True
    """Instead of adding color variance to the data term (current CONTOUR 
    approach), use it to MODULATE the smoothness term. High color variance 
    → lower smoothness → allow sharper boundaries. This prevents textured 
    backgrounds from being classified as foreground."""

    normal_variance_as_smoothness_weight: bool = True
    """Same principle for normal variance. High curvature regions get 
    reduced smoothness, allowing the contour to follow geometric features."""

    feature_edge_weight: float = 0.5
    """How strongly color/normal edges reduce smoothness. 0 = uniform 
    smoothness everywhere. 1 = zero smoothness at feature edges."""

    # --- Variational Level-Set ---
    lambda_data: float = 1.0
    """Weight for data fidelity term. The depth evidence drives the 
    segmentation; this should almost always be 1.0."""

    lambda_smooth: float = 0.3
    """Weight for Tikhonov smoothness. Higher → smoother contour, 
    potentially losing thin features. Lower → noisier contour following 
    every depth fluctuation."""

    lambda_area: float = 0.05
    """Weight for area constraint. Prevents degenerate solutions where 
    FG shrinks to nothing or expands to everything."""

    target_fg_fraction: float = 0.3
    """Target foreground area fraction for the area constraint.
    This is a soft target — the data term can override it. But it 
    provides a prior that prevents collapse."""

    levelset_iterations: int = 50
    """Number of gradient descent steps for level-set optimization.
    More iterations → better convergence but slower. The solution 
    typically converges in 20-50 steps for typical grid sizes."""

    levelset_dt: float = 0.1
    """Step size for level-set gradient descent. Too large → oscillation. 
    Too small → slow convergence. CFL-like condition: dt < h² / (4 * λ_smooth) 
    where h = 1/max(H,W)."""

    levelset_epsilon: float = 0.5
    """Width of the smooth Heaviside approximation. Controls how 
    'sharp' the zero-crossing is. Smaller → sharper but harder to 
    optimize. Larger → smoother transition zone."""

    # --- Confidence & Boundary Handling ---
    confidence_threshold: float = 0.8
    """Confidence below which a control point is considered 'boundary'.
    Boundary points get special handling during resampling: they're 
    duplicated to both surfaces with complementary opacity."""

    boundary_duplication: bool = True
    """Whether to duplicate boundary-region control points to both 
    surfaces. If False, boundary points are assigned to the surface 
    with higher φ magnitude (hard assignment)."""

    # --- Mask Cleanup ---
    min_component_area: float = 0.02
    """Minimum connected component area (as fraction of grid) to keep."""

    fill_holes: bool = True
    """Fill interior holes in the foreground mask."""

    # --- Resampling ---
    fg_grid_scale: float = 1.0
    """Scale factor for FG grid resolution relative to mask area.
    1.0 = proportional to mask area. >1.0 = oversample FG."""

    bg_grid_scale: float = 0.7
    """Scale factor for BG grid resolution. <1.0 = subsample BG 
    (background typically needs fewer control points)."""


class DepthEvidenceBuilder:
    """
    Constructs a continuous depth evidence field E_d(u,v) ∈ [-1, +1].

    +1 = strong foreground evidence (close to camera)
    -1 = strong background evidence (far from camera)
     0 = ambiguous / at threshold

    Key improvement over current approach:
    - Continuous (not binary Otsu)
    - Locally adaptive (uses local depth variance)
    - Preserves uncertainty near the decision boundary
    """

    def __init__(self, config: HybridDecompConfig):
        self.cfg = config

    def build(
        self,
        agg_depth: torch.Tensor,  # [H, W] aggregated depth
        device: str,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Build depth evidence field.

        Returns:
            evidence: [H, W] in [-1, +1]
            threshold: scalar depth threshold used
            confidence: [H, W] in [0, 1] — how certain we are about this point
        """
        H, W = agg_depth.shape

        # 1. Normalize depth to [0, 1]
        d_min, d_max = agg_depth.min(), agg_depth.max()
        d_range = d_max - d_min
        if d_range < 1e-6:
            # Flat scene — no depth separation possible
            print("[DepthEvidence] WARNING: Near-zero depth range. "
                  "Depth cue is unreliable.")
            return (
                torch.zeros(H, W, device=device),
                0.5,
                torch.zeros(H, W, device=device),
            )

        depth_norm = (agg_depth - d_min) / (d_range + 1e-8)

        # 2. Find threshold
        threshold = self._find_threshold(depth_norm)

        # 3. Compute local depth variance (adaptive sharpness)
        local_sigma = self._local_variance(depth_norm, self.cfg.depth_local_sigma_kernel)
        # Normalize: high variance → low sharpness
        sigma_norm = local_sigma / (local_sigma.max() + 1e-8)
        adaptive_sharpness = self.cfg.depth_evidence_sharpness / (1.0 + 2.0 * sigma_norm)

        # 4. Continuous evidence via tanh
        evidence = -torch.tanh(
            adaptive_sharpness * (depth_norm - threshold)
        )
        # Negative sign: closer (lower depth) → positive (FG)

        # 5. Confidence: |evidence| → high confidence far from threshold
        # But also modulated by depth range quality
        range_confidence = (d_range / (d_max + 1e-8)).clamp(0, 1)
        confidence = evidence.abs() * range_confidence

        print(f"[DepthEvidence] Threshold: {threshold:.4f}, "
              f"Range confidence: {range_confidence:.3f}, "
              f"FG fraction (evidence>0): {(evidence > 0).float().mean():.1%}")

        return evidence, threshold, confidence

    def _find_threshold(self, depth_norm: torch.Tensor) -> float:
        """Find optimal depth threshold."""
        if self.cfg.depth_threshold_method == "otsu":
            return self._otsu(depth_norm)
        elif self.cfg.depth_threshold_method == "histogram_gap":
            return self._histogram_gap(depth_norm)
        elif self.cfg.depth_threshold_method == "quantile":
            return 0.35  # Fixed quantile fallback
        else:
            raise ValueError(f"Unknown method: {self.cfg.depth_threshold_method}")

    def _otsu(self, values: torch.Tensor, num_bins: int = 256) -> float:
        """Otsu's method — same as existing but returns float threshold."""
        device = values.device
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

    def _histogram_gap(self, values: torch.Tensor, num_bins: int = 128) -> float:
        """
        Find the largest gap in the depth histogram.

        More robust than Otsu for multi-modal distributions where the
        foreground/background modes aren't well-separated but there's
        a clear valley between them.
        """
        device = values.device
        flat = values.reshape(-1)

        hist = torch.histc(flat, bins=num_bins, min=0.0, max=1.0)
        bin_centers = torch.linspace(0, 1, num_bins, device=device)

        # Smooth histogram to find valleys
        kernel = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32, device=device)
        kernel = kernel / kernel.sum()
        hist_smooth = F.conv1d(
            hist.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=2
        ).squeeze()

        # Find the deepest valley (minimum between two peaks)
        # Ignore edges (first/last 10%)
        margin = num_bins // 10
        interior = hist_smooth[margin:-margin]
        valley_idx = interior.argmin().item() + margin

        return bin_centers[valley_idx].item()

    def _local_variance(
        self,
        grid: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        """Compute local variance of a 2D field."""
        H, W = grid.shape
        device = grid.device
        pad = kernel_size // 2

        g = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        kernel = kernel / (kernel_size * kernel_size)

        padded = F.pad(g, (pad, pad, pad, pad), mode='reflect')
        local_mean = F.conv2d(padded, kernel)
        local_mean_sq = F.conv2d(padded ** 2, kernel)

        variance = (local_mean_sq - local_mean ** 2).clamp(min=0).squeeze()
        return variance


class SmoothnessModulator:
    """
    Computes spatially-varying smoothness weights from color and normal cues.

    Key insight: Color and normal variance should NOT be added to the
    data term (which drives FG/BG assignment). Instead, they should
    MODULATE the smoothness term: where there are strong feature edges,
    reduce smoothness to allow the contour to follow them.

    This is the standard approach in edge-aware image segmentation
    (cf. geodesic active contours, Caselles et al. 1997).
    """

    def __init__(self, config: HybridDecompConfig):
        self.cfg = config

    def compute_edge_indicator(
        self,
        surface: "SplineModel",
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """
        Compute edge indicator g(u,v) ∈ (0, 1].

        g = 1 / (1 + β * |∇I|²)

        where |∇I|² combines color and normal gradient magnitudes.
        g ≈ 1 in homogeneous regions (full smoothness)
        g ≈ 0 at edges (reduced smoothness → contour can be sharp)

        Returns:
            g: [H, W] edge indicator
        """
        from utils.sh_utils import SH2RGB

        gradient_magnitude_sq = torch.zeros(H, W, device=device)

        # Color gradient
        if self.cfg.color_variance_as_smoothness_weight:
            sh_dc = surface.spherical_harmonics.sh_dc.control_features.detach()
            color_grid = SH2RGB(sh_dc.view(H, W, 3)).clamp(0, 1)

            # Sobel-like gradient (more directional than local variance)
            color_grad_u = torch.zeros_like(color_grid)
            color_grad_v = torch.zeros_like(color_grid)
            color_grad_u[:-1] = color_grid[1:] - color_grid[:-1]
            color_grad_u[-1] = color_grad_u[-2]
            color_grad_v[:, :-1] = color_grid[:, 1:] - color_grid[:, :-1]
            color_grad_v[:, -1] = color_grad_v[:, -2]

            color_grad_mag = (
                color_grad_u.pow(2).sum(dim=-1) +
                color_grad_v.pow(2).sum(dim=-1)
            )
            gradient_magnitude_sq += color_grad_mag

        # Normal gradient
        if self.cfg.normal_variance_as_smoothness_weight:
            pos = surface.position.control_features.detach().view(H, W, 3)
            du = torch.zeros_like(pos)
            dv = torch.zeros_like(pos)
            du[:-1] = pos[1:] - pos[:-1]
            du[-1] = du[-2]
            dv[:, :-1] = pos[:, 1:] - pos[:, :-1]
            dv[:, -1] = dv[:, -2]
            normals = F.normalize(torch.cross(du, dv, dim=-1), dim=-1, eps=1e-8)

            normal_grad_u = torch.zeros_like(normals)
            normal_grad_v = torch.zeros_like(normals)
            normal_grad_u[:-1] = normals[1:] - normals[:-1]
            normal_grad_u[-1] = normal_grad_u[-2]
            normal_grad_v[:, :-1] = normals[:, 1:] - normals[:, :-1]
            normal_grad_v[:, -1] = normal_grad_v[:, -2]

            normal_grad_mag = (
                normal_grad_u.pow(2).sum(dim=-1) +
                normal_grad_v.pow(2).sum(dim=-1)
            )
            gradient_magnitude_sq += normal_grad_mag

        # Normalize
        if gradient_magnitude_sq.max() > 1e-8:
            gradient_magnitude_sq = gradient_magnitude_sq / gradient_magnitude_sq.max()

        # Edge indicator: low at edges, high in flat regions
        beta = self.cfg.feature_edge_weight / (gradient_magnitude_sq.mean() + 1e-8)
        g = 1.0 / (1.0 + beta * gradient_magnitude_sq)

        return g


class VariationalLevelSet:
    """
    Solves the variational level-set problem on the UV control grid.

    Minimizes:
        E(φ) = λ_data · ∫|H(φ) - E_d|²
             + λ_smooth · ∫ g(u,v) |∇φ|²
             + λ_area · (∫H(φ) - A_target)²

    where:
        H(φ) = sigmoid(φ/ε)  — smooth Heaviside
        E_d   — depth evidence field
        g     — edge-aware smoothness modulator

    Solved by gradient descent on φ (no PDE discretization needed
    since the grid is already discrete and small).
    """

    def __init__(self, config: HybridDecompConfig):
        self.cfg = config

    def solve(
        self,
        depth_evidence: torch.Tensor,  # [H, W] in [-1, +1]
        edge_indicator: torch.Tensor,  # [H, W] in (0, 1]
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """
        Optimize level-set field φ.

        Returns:
            phi: [H, W] optimized level-set field
        """
        eps = self.cfg.levelset_epsilon
        dt = self.cfg.levelset_dt
        lam_d = self.cfg.lambda_data
        lam_s = self.cfg.lambda_smooth
        lam_a = self.cfg.lambda_area
        A_target = self.cfg.target_fg_fraction

        # Initialize φ with depth evidence (warm start)
        phi = depth_evidence.clone()

        for iteration in range(self.cfg.levelset_iterations):
            # Smooth Heaviside and its derivative
            H_phi = torch.sigmoid(phi / eps)
            delta_phi = H_phi * (1 - H_phi) / eps  # Derivative of sigmoid

            # === Data term gradient ===
            # ∂E_data/∂φ = 2(H(φ) - E_d) · δ(φ)
            # Map E_d from [-1,+1] to [0,1] for comparison with H(φ)
            E_d_01 = (depth_evidence + 1) / 2
            grad_data = 2 * (H_phi - E_d_01) * delta_phi

            # === Smoothness term gradient ===
            # ∂E_smooth/∂φ = -div(g · ∇φ)
            # Discretized as weighted Laplacian
            grad_smooth = self._weighted_laplacian(phi, edge_indicator)

            # === Area term gradient ===
            # ∂E_area/∂φ = 2(area - A_target) · δ(φ) / (H·W)
            area = H_phi.mean()
            grad_area = 2 * (area - A_target) * delta_phi / (H * W)

            # === Combined gradient descent step ===
            grad_total = (
                lam_d * grad_data
                - lam_s * grad_smooth  # Negative because Laplacian is already negative
                + lam_a * grad_area
            )

            phi = phi - dt * grad_total

            # Logging
            if iteration % 10 == 0 or iteration == self.cfg.levelset_iterations - 1:
                fg_frac = (phi > 0).float().mean().item()
                energy_data = ((H_phi - E_d_01) ** 2).mean().item()
                energy_smooth = (edge_indicator * self._gradient_magnitude(phi)).mean().item()
                print(f"  [LevelSet iter {iteration:3d}] "
                      f"FG: {fg_frac:.1%}, "
                      f"E_data: {energy_data:.4f}, "
                      f"E_smooth: {energy_smooth:.4f}, "
                      f"area: {area:.3f}")

        return phi

    def _weighted_laplacian(
        self,
        phi: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute div(g · ∇φ) using finite differences.

        Discretization:
            div(g∇φ) ≈ Σ_{neighbors} g_{avg} * (φ_neighbor - φ_center) / h²

        where g_avg = (g_center + g_neighbor) / 2 for consistency.
        """
        H, W = phi.shape
        result = torch.zeros_like(phi)

        # Up neighbor
        if H > 1:
            g_avg = (g[1:] + g[:-1]) / 2
            diff = phi[1:] - phi[:-1]
            result[:-1] += g_avg * diff
            result[1:] -= g_avg * diff

        # Left neighbor
        if W > 1:
            g_avg = (g[:, 1:] + g[:, :-1]) / 2
            diff = phi[:, 1:] - phi[:, :-1]
            result[:, :-1] += g_avg * diff
            result[:, 1:] -= g_avg * diff

        return result

    def _gradient_magnitude(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute |∇φ|² for energy monitoring."""
        grad_u = torch.zeros_like(phi)
        grad_v = torch.zeros_like(phi)
        grad_u[:-1] = phi[1:] - phi[:-1]
        grad_v[:, :-1] = phi[:, 1:] - phi[:, :-1]
        return grad_u ** 2 + grad_v ** 2


class ConfidenceAwareMaskExtractor:
    """
    Extracts FG/BG masks with confidence-aware boundary handling.

    Unlike hard thresholding, this produces:
    1. Binary mask (for grid resampling)
    2. Confidence map (for boundary point handling)
    3. Boundary mask (for optional duplication)
    """

    def __init__(self, config: HybridDecompConfig):
        self.cfg = config

    def extract(
        self,
        phi: torch.Tensor,  # [H, W] optimized level-set
        H: int, W: int,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract masks with confidence.

        Returns dict with:
            'fg_mask': [H, W] bool
            'bg_mask': [H, W] bool
            'confidence': [H, W] float in [0, 1]
            'boundary_mask': [H, W] bool — uncertain region
            'signed_distance': [H, W] float — from contour
        """
        # Binary mask
        fg_mask = phi > 0

        # Confidence from φ magnitude
        tau = self.cfg.levelset_epsilon * 2  # Scale with Heaviside width
        confidence = torch.sigmoid(phi.abs() / tau)

        # Boundary = low confidence
        boundary_mask = confidence < self.cfg.confidence_threshold

        # Morphological cleanup
        fg_mask = self._cleanup(fg_mask, H, W, device)

        # Signed distance (for complementary opacity if needed)
        signed_dist = self._compute_signed_distance(fg_mask, H, W, device)

        # Validate coverage
        fg_frac = fg_mask.float().mean().item()
        boundary_frac = boundary_mask.float().mean().item()
        print(f"[MaskExtract] FG: {fg_frac:.1%}, "
              f"Boundary: {boundary_frac:.1%}, "
              f"Confidence mean: {confidence.mean():.3f}")

        return {
            'fg_mask': fg_mask,
            'bg_mask': ~fg_mask,
            'confidence': confidence,
            'boundary_mask': boundary_mask,
            'signed_distance': signed_dist,
        }

    def _cleanup(
        self,
        mask: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """Morphological cleanup: fill holes, remove small CCs."""
        try:
            from scipy import ndimage

            arr = mask.cpu().numpy().astype(np.int32)
            total = H * W

            # Fill holes
            if self.cfg.fill_holes:
                arr = ndimage.binary_fill_holes(arr).astype(np.int32)

            # Remove small components
            labeled, num_features = ndimage.label(arr)
            for comp_id in range(1, num_features + 1):
                comp = labeled == comp_id
                if comp.sum() / total < self.cfg.min_component_area:
                    arr[comp] = 0

            # Also clean BG: remove small BG holes
            bg_labeled, bg_features = ndimage.label(1 - arr)
            for comp_id in range(1, bg_features + 1):
                comp = bg_labeled == comp_id
                touches_border = (
                    comp[0, :].any() or comp[-1, :].any() or
                    comp[:, 0].any() or comp[:, -1].any()
                )
                if not touches_border and comp.sum() / total < 0.1:
                    arr[comp] = 1

            return torch.tensor(arr, device=device, dtype=torch.bool)

        except ImportError:
            return mask

    def _compute_signed_distance(
        self,
        fg_mask: torch.Tensor,
        H: int, W: int,
        device: str,
    ) -> torch.Tensor:
        """Signed distance from contour boundary."""
        try:
            from scipy.ndimage import distance_transform_edt
            fg_np = fg_mask.cpu().numpy().astype(np.float64)
            dist_in = distance_transform_edt(fg_np)
            dist_out = distance_transform_edt(1 - fg_np)
            return torch.tensor(dist_in - dist_out, device=device, dtype=torch.float32)
        except ImportError:
            return fg_mask.float() * 2 - 1