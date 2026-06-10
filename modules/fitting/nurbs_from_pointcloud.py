"""
NURBS Surface Fitting from Point Clouds

Preprocesses an input point cloud, performs optional scene decomposition
(single / background-object / K-components), and fits initial B-spline
surfaces with adaptive resolution and sampling.

Produces NURBSSurfaceData instances consumed by SplineModel / MultiSurfaceSplineModel.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Optional, Dict, Union, Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from geomdl import BSpline

from modules.fitting.mesh_patch import least_squares_bspline_surface
from modules.fitting.postprocessing import CameraObject, compute_camera_consistent_normals, \
    compute_observation_weights


# =============================================================================
# Enums and Configuration
# =============================================================================


class DecompositionMode(Enum):
    SINGLE = "single"
    BACKGROUND_OBJECT = "background"
    K_COMPONENTS = "k_components"


@dataclass
class SurfaceConfig:
    """Configuration for NURBS surface fitting."""

    # --- Grid Resolution ---
    adaptive_resolution: bool = True
    base_resolution: int = 128
    min_resolution: int = 128
    max_resolution: int = 512
    resolution_complexity_sensitivity: float = 1.0

    # Fallback fixed resolution
    resolution_u: int = 32
    resolution_v: int = 32

    # --- NURBS Parameters ---
    degree_u: int = 3
    degree_v: int = 3
    smoothing: float = 0.05

    # --- Decomposition Parameters ---
    decomposition_mode: DecompositionMode = DecompositionMode.K_COMPONENTS
    n_components: int = 4

    # --- Feature Weights for Segmentation ---
    weight_spatial: float = 1
    weight_normal: float = 0.0
    weight_color: float = 1.0
    weight_complexity: float = 0.0

    # --- Connectivity Graph ---
    connectivity_k: int = 4
    min_component_size: int = 1
    use_soft_edges: bool = False
    edge_color_adaptive: bool = False

    # --- Background/Object ---
    bg_detection_method: str = "hybrid"
    bg_distance_threshold: float = 0.025
    bg_min_size_ratio: float = 0.5
    bg_max_size_ratio: float = 0.8

    # --- UV Parameterization ---
    use_geodesic_uv: bool = True
    parameterization: str = "spherical"
    generate_adaptive_samples: bool = True
    sampling_resolution_factor: float = 1.0

    # --- Connectivity Constraints ---
    max_normal_angle_diff: float = 60.0

    # --- Quality Parameters ---
    outlier_removal: bool = True
    outlier_std_ratio: float = 2.0
    normal_estimation_k: int = 30

    # --- Per-component resolution ---
    bg_resolution: Optional[Tuple[int, int]] = (16, 16)
    object_resolution: Optional[Tuple[int, int]] = (64, 64)
    bg_resolution_scale: float = 0.5
    object_resolution_scale: float = 2.0


    # --- Camera-Aware Fitting (NEW) ---
    use_camera_weights: bool = False
    use_camera_normals: bool = False

    # --- Post-Fit Optimization (NEW) ---
    post_fit_enabled: bool = False
    post_fit_iterations: int = 2000
    post_fit_lr: float = 1e-3
    post_fit_chamfer_weight: float = .5
    post_fit_smoothness_weight: float = 0.25
    post_fit_normal_weight: float = 0.25
    post_fit_num_samples: int = 2048
    post_fit_convergence_threshold: float = 1e-8
    post_fit_patience: int = 200

@dataclass
class AdaptiveSamplingResult:
    """Result from adaptive sampling generation."""

    intervals_u: np.ndarray  # [Us] sorted U coordinates
    intervals_v: np.ndarray  # [Vs] sorted V coordinates
    grid_u: Optional[np.ndarray]  # [Us, Vs] U coordinate at each sample (or None)
    grid_v: Optional[np.ndarray]  # [Us, Vs] V coordinate at each sample (or None)
    complexity_map: np.ndarray  # [H_ctrl, W_ctrl] complexity at control points
    density_map: np.ndarray  # [Us, Vs] target density used

    def get_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.intervals_u, self.intervals_v

    def get_2d(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.grid_u, self.grid_v

    def to_uv_grid(self) -> Optional[np.ndarray]:
        if self.grid_u is not None and self.grid_v is not None:
            return np.stack([self.grid_u, self.grid_v], axis=-1)
        return None


@dataclass
class DecompositionResult:
    """Result of scene decomposition."""

    components: List[np.ndarray]
    labels: np.ndarray
    mode: DecompositionMode
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_components(self) -> int:
        return len(self.components)

    def get_component_sizes(self) -> List[int]:
        return [len(c) for c in self.components]


# =============================================================================
# Data Containers
# =============================================================================

@dataclass
class PostFitConfig:
    """Configuration for post-fit Chamfer optimization."""

    num_iterations: int = 5000
    learning_rate: float = 1e-3
    lr_decay: float = 0.999  # Exponential decay per iteration

    # Loss weights
    chamfer_weight: float = 1.0
    smoothness_weight: float = 0.1
    normal_weight: float = 0.0  # Weight for normal consistency term

    # Sampling
    num_surface_samples: int = 4096  # Points sampled on surface per iteration
    batch_points: int = 8192  # Target points per batch (0 = all)

    # Camera weighting
    use_camera_weights: bool = True

    # Convergence
    convergence_threshold: float = 1e-7
    patience: int = 200

    verbose: bool = True


class BSplinePostFitter:
    """
    Refines a fitted BSpline surface to minimize Chamfer distance
    to the input point cloud.

    This is the critical missing step: the initial grid resampling
    gives a reasonable starting point, but doesn't actually optimize
    for geometric fidelity. This class does gradient-descent refinement
    of the control points against the input data.

    The optimization minimizes:
        L = w_c * Chamfer(S, P) + w_s * Smoothness(S) + w_n * NormalConsistency(S, P)

    where:
        - Chamfer(S, P): bidirectional Chamfer distance between
          surface samples S and point cloud P
        - Smoothness: second-derivative magnitude of control grid
          (prevents overfitting / oscillation)
        - NormalConsistency: alignment between surface normals and
          point cloud normals (if available)
    """

    def __init__(self, config: Optional[PostFitConfig] = None):
        self.config = config or PostFitConfig()

    def refine(
            self,
            surface_data: 'NURBSSurfaceData',
            target_points: np.ndarray,
            target_normals: Optional[np.ndarray] = None,
            point_weights: Optional[np.ndarray] = None,
            cameras: Optional[List[CameraObject]] = None,
    ) -> 'NURBSSurfaceData':
        """
        Refine control points to minimize Chamfer distance.

        Args:
            surface_data: Initial BSpline surface from grid fitting
            target_points: [N, 3] input point cloud to fit against
            target_normals: [N, 3] optional point normals
            point_weights: [N] per-point importance weights
            cameras: Training cameras for view-weighted refinement

        Returns:
            Refined NURBSSurfaceData with optimized control points
        """
        cfg = self.config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Prepare target data ---
        target_pts = torch.tensor(
            target_points, dtype=torch.float32, device=device
        )
        target_tree = cKDTree(target_points)

        if target_normals is not None:
            target_nrm = torch.tensor(
                target_normals, dtype=torch.float32, device=device
            )
        else:
            target_nrm = None

        # Compute camera-based weights if available
        if point_weights is not None:
            pt_weights = torch.tensor(
                point_weights, dtype=torch.float32, device=device
            )
        elif cfg.use_camera_weights and cameras:
            normals_np = target_normals
            if normals_np is None:
                proc = PointCloudProcessor(target_points)
                normals_np = proc.estimate_normals()
            w = compute_observation_weights(target_points, cameras, normals_np)
            pt_weights = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            pt_weights = torch.ones(len(target_points), device=device)

        # --- Prepare control points as optimizable parameters ---
        H, W, _ = surface_data.control_points.shape
        ctrl_pts = torch.tensor(
            surface_data.control_points.copy(),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        ctrl_colors = torch.tensor(
            surface_data.control_colors.copy(),
            dtype=torch.float32,
            device=device,
        )  # Colors are NOT optimized here

        knots_u = torch.tensor(
            surface_data.knots_u, dtype=torch.float32, device=device
        )
        knots_v = torch.tensor(
            surface_data.knots_v, dtype=torch.float32, device=device
        )
        degree_u = surface_data.degree_u
        degree_v = surface_data.degree_v

        # --- Build basis matrices for evaluation ---
        # We sample the surface at fixed UV locations and optimize control points
        n_samples_u = min(max(H * 8, 64), 1024)
        n_samples_v = min(max(W * 8, 64), 1024)

        u_eval = torch.linspace(
            knots_u[degree_u].item(),
            knots_u[-(degree_u + 1)].item(),
            n_samples_u,
            device=device,
        )
        v_eval = torch.linspace(
            knots_v[degree_v].item(),
            knots_v[-(degree_v + 1)].item(),
            n_samples_v,
            device=device,
        )

        # Pre-compute basis matrices (don't change during optimization)
        Bu = self._compute_basis_matrix(
            u_eval, knots_u, degree_u, H
        )  # [n_samples_u, H]
        Bv = self._compute_basis_matrix(
            v_eval, knots_v, degree_v, W
        )  # [n_samples_v, W]

        # --- Optimizer ---
        optimizer = torch.optim.Adam([ctrl_pts], lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.lr_decay
        )

        best_loss = float("inf")
        best_ctrl = ctrl_pts.data.clone()
        patience_counter = 0
        chamfer_ema = None

        if cfg.verbose:
            print(
                f"[PostFit] Refining {H}x{W} control grid against "
                f"{len(target_points)} points ({cfg.num_iterations} iters)"
            )

        for iteration in range(cfg.num_iterations):
            optimizer.zero_grad()

            step1 = torch.einsum("uh,hwc->uwc", Bu, ctrl_pts)
            surface_pts = torch.einsum("uwc,vw->uvc", step1, Bv)
            surface_pts_flat = surface_pts.reshape(-1, 3)  # [M, 3]

            # --- Subsample if needed ---
            M = surface_pts_flat.shape[0]
            if cfg.num_surface_samples > 0 and M > cfg.num_surface_samples:
                idx = torch.randperm(M, device=device)[: cfg.num_surface_samples]
                surface_sample = surface_pts_flat[idx]
            else:
                surface_sample = surface_pts_flat

            N = target_pts.shape[0]
            if cfg.batch_points > 0 and N > cfg.batch_points:
                tidx = torch.randperm(N, device=device)[: cfg.batch_points]
                target_batch = target_pts[tidx]
                weight_batch = pt_weights[tidx]
                normal_batch = target_nrm[tidx] if target_nrm is not None else None
            else:
                target_batch = target_pts
                weight_batch = pt_weights
                normal_batch = target_nrm

            # --- Chamfer distance (bidirectional, weighted) ---
            chamfer_loss = self._weighted_chamfer(
                surface_sample, target_batch, weight_batch
            )

            # --- Smoothness regularization ---
            smooth_loss = self._smoothness_loss(ctrl_pts)

            # --- Normal consistency (optional) ---
            if cfg.normal_weight > 0 and normal_batch is not None:
                normal_loss = self._normal_consistency_loss(
                    surface_pts, Bu, Bv, ctrl_pts, target_batch, normal_batch
                )
            else:
                normal_loss = torch.tensor(0.0, device=device)

            # --- Total loss ---
            # Anneal smoothness: strong early (keeps the grid stable while the
            # Chamfer term does large moves), decaying to 10% so fine detail
            # is not regularized away in the sharpening phase.
            smooth_anneal = max(0.1, 1.0 - iteration / max(1.0, 0.8 * cfg.num_iterations))
            total_loss = (
                    cfg.chamfer_weight * chamfer_loss
                    + cfg.smoothness_weight * smooth_anneal * smooth_loss
                    + cfg.normal_weight * normal_loss
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Convergence check ---
            # Snapshot on an EMA of the pure Chamfer term: per-iteration
            # losses are stochastic (random surface/target subsets), and
            # geometric fidelity — not the regularizers — is what makes a
            # good training init.
            loss_val = total_loss.item()
            chamfer_ema = (
                chamfer_loss.item() if chamfer_ema is None
                else 0.9 * chamfer_ema + 0.1 * chamfer_loss.item()
            )
            if chamfer_ema < best_loss - cfg.convergence_threshold:
                best_loss = chamfer_ema
                best_ctrl = ctrl_pts.data.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                if cfg.verbose:
                    print(
                        f"[PostFit] Converged at iteration {iteration} "
                        f"(loss={best_loss:.6f})"
                    )
                break

            if cfg.verbose and iteration % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[PostFit] Iter {iteration:4d} | "
                    f"Chamfer={chamfer_loss.item():.6f} "
                    f"Smooth={smooth_loss.item():.6f} "
                    f"Normal={normal_loss.item():.6f} "
                    f"Total={loss_val:.6f} | LR={lr:.2e}"
                )

        # --- Accept refinement only if it beats the initial fit ---
        # Deterministic dense Chamfer of initial vs refined control grids;
        # on clean/dense clouds the LS fit can already be optimal and the
        # stochastic refinement must not degrade it.
        with torch.no_grad():
            def _dense_chamfer(cp: torch.Tensor) -> float:
                sp = torch.einsum("uh,hwc->uwc", Bu, cp)
                sp = torch.einsum("uwc,vw->uvc", sp, Bv).reshape(-1, 3)
                s_step = max(1, sp.shape[0] // 4096)
                t_step = max(1, target_pts.shape[0] // 8192)
                d = torch.cdist(sp[::s_step], target_pts[::t_step])
                return (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item()

            init_ctrl = torch.tensor(
                surface_data.control_points, dtype=torch.float32, device=device
            )
            init_cd = _dense_chamfer(init_ctrl)
            refined_cd = _dense_chamfer(best_ctrl)
            if init_cd <= refined_cd:
                if cfg.verbose:
                    print(
                        f"[PostFit] Keeping initial fit "
                        f"(CD {init_cd:.6f} <= refined {refined_cd:.6f})"
                    )
                best_ctrl = init_ctrl
            elif cfg.verbose:
                print(
                    f"[PostFit] Refinement improved dense CD: "
                    f"{init_cd:.6f} -> {refined_cd:.6f}"
                )

        # --- Build refined surface data ---
        refined = NURBSSurfaceData(
            control_points=best_ctrl.detach().cpu().numpy().astype(np.float32),
            control_colors=surface_data.control_colors,  # Unchanged
            knots_u=surface_data.knots_u,
            knots_v=surface_data.knots_v,
            degree_u=surface_data.degree_u,
            degree_v=surface_data.degree_v,
            label=surface_data.label,
            point_indices=surface_data.point_indices,
            bounds=surface_data.bounds,
            sampling_u_1D=surface_data.sampling_u_1D,
            sampling_v_1D=surface_data.sampling_v_1D,
            grid_samplings_u=surface_data.grid_samplings_u,
            grid_samplings_v=surface_data.grid_samplings_v,
            complexity_map=surface_data.complexity_map,
            sampling_density_map=surface_data.sampling_density_map,
        )
        return refined

    def _compute_basis_matrix(
            self,
            params: torch.Tensor,
            knots: torch.Tensor,
            degree: int,
            n_ctrl: int,
    ) -> torch.Tensor:
        """
        Compute B-spline basis function matrix.

        Uses the Cox-de Boor recursion, evaluated for all parameter
        values simultaneously.

        Args:
            params: [M] parameter values (u or v)
            knots:  [n_ctrl + degree + 1] knot vector
            degree: B-spline degree
            n_ctrl: Number of control points

        Returns:
            B: [M, n_ctrl] basis function values
        """
        # Exact triangular-table algorithm (tested vs geomdl) — same basis
        # the training path uses, so the refined init matches what the
        # SplineModel will evaluate.
        from modules.basis import bspline_basis_and_derivs_1d

        (N,) = bspline_basis_and_derivs_1d(params, knots, degree, max_deriv=0)
        assert N.shape == (params.shape[0], n_ctrl), (
            f"Basis shape mismatch: got {N.shape}, expected "
            f"({params.shape[0]}, {n_ctrl}). Knots: {len(knots)}, degree: {degree}"
        )
        return N

    def _weighted_chamfer(
            self,
            source: torch.Tensor,
            target: torch.Tensor,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted bidirectional Chamfer distance with outlier truncation.
        """
        # Source -> Target: for each surface point, find nearest target
        diff_s2t = source.unsqueeze(1) - target.unsqueeze(0)
        dist_s2t = (diff_s2t ** 2).sum(dim=-1)  # [M, N]
        min_s2t, _ = dist_s2t.min(dim=1)  # [M]

        s2t_threshold = torch.quantile(min_s2t, 0.95)
        loss_s2t = min_s2t[min_s2t <= s2t_threshold].mean()

        # Target -> Source: for each target point, find nearest surface point
        min_t2s, _ = dist_s2t.min(dim=0)  # [N]

        # Apply truncation here as well, weighted by observation quality
        t2s_threshold = torch.quantile(min_t2s, 0.95)
        valid_t2s_mask = min_t2s <= t2s_threshold

        valid_min_t2s = min_t2s[valid_t2s_mask]
        valid_weights = weights[valid_t2s_mask]

        loss_t2s = (valid_min_t2s * valid_weights).sum() / (valid_weights.sum() + 1e-8)

        return loss_s2t + loss_t2s


    def _smoothness_loss(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Second-derivative smoothness regularization on control grid.

        Penalizes large second differences (discrete Laplacian),
        preventing the surface from overfitting / oscillating.

        Args:
            ctrl_pts: [H, W, 3] control point grid

        Returns:
            Scalar smoothness loss
        """
        H, W, _ = ctrl_pts.shape
        loss = torch.tensor(0.0, device=ctrl_pts.device)

        # Second differences in U direction
        if H > 2:
            d2u = ctrl_pts[2:] - 2 * ctrl_pts[1:-1] + ctrl_pts[:-2]
            loss = loss + (d2u ** 2).mean()

        # Second differences in V direction
        if W > 2:
            d2v = ctrl_pts[:, 2:] - 2 * ctrl_pts[:, 1:-1] + ctrl_pts[:, :-2]
            loss = loss + (d2v ** 2).mean()

        return loss

    def _normal_consistency_loss(
            self,
            surface_pts: torch.Tensor,
            Bu: torch.Tensor,
            Bv: torch.Tensor,
            ctrl_pts: torch.Tensor,
            target_pts: torch.Tensor,
            target_normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normal consistency between surface tangent normals and target normals.

        For each target point, find the nearest surface point and check
        that the surface normal at that location aligns with the target normal.

        Args:
            surface_pts: [n_u, n_v, 3] evaluated surface
            Bu, Bv: Basis matrices
            ctrl_pts: [H, W, 3] control points (for derivative computation)
            target_pts: [N, 3] target points
            target_normals: [N, 3] target normals

        Returns:
            Scalar normal consistency loss
        """
        n_u, n_v, _ = surface_pts.shape

        # Compute surface tangents via finite differences on the evaluated grid
        du = torch.zeros_like(surface_pts)
        du[:-1] = surface_pts[1:] - surface_pts[:-1]
        du[-1] = du[-2]

        dv = torch.zeros_like(surface_pts)
        dv[:, :-1] = surface_pts[:, 1:] - surface_pts[:, :-1]
        dv[:, -1] = dv[:, -2]

        # Surface normals
        surf_normals = torch.cross(du, dv, dim=-1)
        surf_normals = torch.nn.functional.normalize(surf_normals, dim=-1, eps=1e-8)
        surf_normals_flat = surf_normals.reshape(-1, 3)  # [M, 3]
        surface_flat = surface_pts.reshape(-1, 3)

        # Find nearest surface point for each target
        # Use a subsample for efficiency
        n_check = min(1024, target_pts.shape[0])
        tidx = torch.randperm(target_pts.shape[0], device=target_pts.device)[:n_check]
        t_pts = target_pts[tidx]
        t_nrm = target_normals[tidx]

        dists = torch.cdist(t_pts, surface_flat)  # [n_check, M]
        nearest_idx = dists.argmin(dim=1)  # [n_check]

        nearest_surf_normals = surf_normals_flat[nearest_idx]  # [n_check, 3]

        # Normal alignment: 1 - |dot(n_surface, n_target)|
        # Using absolute value since normals may be flipped
        dots = torch.abs((nearest_surf_normals * t_nrm).sum(dim=-1))
        loss = (1.0 - dots).mean()

        return loss

@dataclass
class NURBSSurfaceData:
    """Container for a single NURBS surface."""

    control_points: np.ndarray  # [H, W, 3] XYZ
    control_colors: np.ndarray  # [H, W, 3] RGB
    knots_u: np.ndarray
    knots_v: np.ndarray
    degree_u: int
    degree_v: int
    label: str = "surface"
    point_indices: Optional[np.ndarray] = None
    bounds: Optional[Dict[str, np.ndarray]] = None
    sampling_u_1D: Optional[np.ndarray] = None
    sampling_v_1D: Optional[np.ndarray] = None
    grid_samplings_u: Optional[np.ndarray] = None
    grid_samplings_v: Optional[np.ndarray] = None
    complexity_map: Optional[np.ndarray] = None
    sampling_density_map: Optional[np.ndarray] = None

    def to_torch(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Convert to torch tensors."""
        result = {
            "control_points": torch.tensor(
                self.control_points, dtype=torch.float32, device=device
            ),
            "control_colors": torch.tensor(
                self.control_colors, dtype=torch.float32, device=device
            ),
        }
        if self.knots_u is not None:
            result["knots_u"] = torch.tensor(
                self.knots_u, dtype=torch.float32, device=device
            )
        if self.knots_v is not None:
            result["knots_v"] = torch.tensor(
                self.knots_v, dtype=torch.float32, device=device
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "control_points": self.control_points.tolist(),
            "control_colors": self.control_colors.tolist(),
            "knots_u": self.knots_u.tolist(),
            "knots_v": self.knots_v.tolist(),
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "label": self.label,
            "point_indices": self.point_indices.tolist() if self.point_indices is not None else None,
            "bounds": {k: v.tolist() for k, v in self.bounds.items()} if self.bounds is not None else None,
            "sampling_u_1D": self.sampling_u_1D.tolist() if self.sampling_u_1D is not None else None,
            "sampling_v_1D": self.sampling_v_1D.tolist() if self.sampling_v_1D is not None else None,
            "grid_samplings_u": self.grid_samplings_u.tolist() if self.grid_samplings_u is not None else None,
            "grid_samplings_v": self.grid_samplings_v.tolist() if self.grid_samplings_v is not None else None,
            "complexity_map": self.complexity_map.tolist() if self.complexity_map is not None else None,
            "sampling_density_map": self.sampling_density_map.tolist() if self.sampling_density_map is not None else None,
        }


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for adaptive UV sampling generation."""

    sampling_resolution_factor: float = 1.0

    # Complexity weights
    weight_curvature: float = 1.0
    weight_color_variance: float = 0.5
    weight_normal_variance: float = 0.8
    weight_edge_proximity: float = 0.3

    # Density control
    min_density_ratio: float = 0.3
    max_density_ratio: float = 3.0
    smoothing_sigma: float = 2.0

    # Monotonicity enforcement
    enforce_monotonic: bool = True
    min_spacing_ratio: float = 0.01


@dataclass
class MultiSurfaceResult:
    """Result container for multi-surface decomposition."""

    surfaces: List[NURBSSurfaceData]
    decomposition_mode: DecompositionMode
    labels: np.ndarray
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Point Cloud Processor
# =============================================================================


class PointCloudProcessor:
    """Handles point cloud preprocessing, normal estimation, and feature extraction."""

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        self.points = np.asarray(points, dtype=np.float64)
        self.colors = (
            np.asarray(colors, dtype=np.float64) if colors is not None else None
        )
        self.n_points = len(points)

        self._normals: Optional[np.ndarray] = None
        self._kdtree: Optional[cKDTree] = None
        self._bounds: Optional[Dict[str, np.ndarray]] = None
        self._extent: Optional[float] = None
        self._complexity: Optional[np.ndarray] = None
        self._curvature: Optional[np.ndarray] = None

    @property
    def kdtree(self) -> cKDTree:
        if self._kdtree is None:
            self._kdtree = cKDTree(self.points)
        return self._kdtree

    @property
    def bounds(self) -> Dict[str, np.ndarray]:
        if self._bounds is None:
            self._bounds = {
                "min": self.points.min(axis=0),
                "max": self.points.max(axis=0),
                "center": self.points.mean(axis=0),
            }
        return self._bounds

    @property
    def extent(self) -> float:
        if self._extent is None:
            self._extent = float(
                np.linalg.norm(self.bounds["max"] - self.bounds["min"])
            )
        return self._extent

    def estimate_normals(self, k: int = 30) -> np.ndarray:
        """Estimate consistent surface normals using PCA."""
        if self._normals is not None:
            return self._normals

        k = min(k, self.n_points - 1)
        normals = np.zeros_like(self.points)
        _, indices = self.kdtree.query(self.points, k=k)

        for i in range(self.n_points):
            neighbors = self.points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                normals[i] = Vt[-1]
            except np.linalg.LinAlgError:
                normals[i] = np.array([0, 0, 1])

        # Orient normals away from scene center
        center_vecs = self.points - self.bounds["center"]
        flip_mask = np.sum(normals * center_vecs, axis=1) < 0
        normals[flip_mask] *= -1

        self._normals = normals
        return normals

    def estimate_curvature(self, k: int = 16) -> np.ndarray:
        """Estimate surface curvature using normal variation."""
        if self._curvature is not None:
            return self._curvature

        k = min(k, self.n_points - 1)
        normals = self.estimate_normals(k=k)
        _, indices = self.kdtree.query(self.points, k=k)

        curvatures = np.zeros(self.n_points)
        for i in range(self.n_points):
            neighbor_normals = normals[indices[i]]
            cov = np.cov(neighbor_normals.T)
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
            except np.linalg.LinAlgError:
                curvatures[i] = 0.0

        self._curvature = curvatures
        return curvatures

    def compute_local_complexity(self, k: int = 16) -> np.ndarray:
        """
        Computes a scalar complexity score [0, 1] per point.
        Combines geometric roughness (normal variance) and color variation.
        """
        if self._complexity is not None:
            return self._complexity

        k = min(k, self.n_points - 1)
        _, indices = self.kdtree.query(self.points, k=k)
        normals = self.estimate_normals(k=k)

        # Geometric complexity via normal variance
        neighbor_normals = normals[indices]
        geo_complexity = np.var(neighbor_normals, axis=1).sum(axis=1)
        p95_geo = np.percentile(geo_complexity, 95)
        geo_complexity = np.clip(geo_complexity / (p95_geo + 1e-8), 0, 1)

        # Color complexity
        if self.colors is not None:
            neighbor_colors = self.colors[indices]
            color_complexity = np.var(neighbor_colors, axis=1).sum(axis=1)
            p95_col = np.percentile(color_complexity, 95)
            color_complexity = np.clip(color_complexity / (p95_col + 1e-8), 0, 1)
        else:
            color_complexity = np.zeros_like(geo_complexity)

        complexity = 0.7 * geo_complexity + 0.3 * color_complexity
        ptp = np.ptp(complexity)
        if ptp > 1e-8:
            complexity = (complexity - complexity.min()) / ptp
        else:
            complexity = np.zeros_like(complexity)

        self._complexity = complexity
        return self._complexity

    def compute_normal_variance(self, k: int = 16) -> np.ndarray:
        """Compute local normal variance (high at edges/corners)."""
        k = min(k, self.n_points - 1)
        normals = self.estimate_normals()
        _, indices = self.kdtree.query(self.points, k=k)
        neighbor_normals = normals[indices]
        return np.var(neighbor_normals, axis=1).sum(axis=1)

    def compute_color_variance(self, k: int = 16) -> np.ndarray:
        """Compute local color variance (high at texture boundaries)."""
        if self.colors is None:
            return np.zeros(self.n_points)
        k = min(k, self.n_points - 1)
        _, indices = self.kdtree.query(self.points, k=k)
        neighbor_colors = self.colors[indices]
        return np.var(neighbor_colors, axis=1).sum(axis=1)

    def remove_outliers(
        self, std_ratio: float = 2.0, k: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers based on mean k-NN distance."""
        k = min(k, self.n_points - 1)
        distances, _ = self.kdtree.query(self.points, k=k)
        mean_distances = distances[:, 1:].mean(axis=1)

        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + std_ratio * global_std

        outlier_mask = mean_distances > threshold
        clean_indices = np.where(~outlier_mask)[0]

        return clean_indices, outlier_mask


# =============================================================================
# Scene Decomposer
# =============================================================================


class SceneDecomposer:
    """
    Scene decomposition with feature-aware clustering
    and connectivity-constrained partitioning.
    """

    def __init__(self, processor: PointCloudProcessor, config: SurfaceConfig):
        self.processor = processor
        self.config = config
        self._surface_graph: Optional[csr_matrix] = None

    def decompose_background_object(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Semantic background/foreground separation.

        Returns:
            bg_indices, fg_indices: Integer index arrays into the point cloud.
        """
        if self._surface_graph is None:
            self._build_surface_graph()

        method = self.config.bg_detection_method
        if method == "distance":
            bg_scores = self._compute_distance_bg_scores()
        elif method == "normal":
            bg_scores = self._compute_normal_bg_scores()
        elif method == "density":
            bg_scores = self._compute_density_bg_scores()
        elif method == "hybrid":
            bg_scores = self._compute_hybrid_bg_scores()
        else:
            raise ValueError(f"Unknown bg_detection_method: {method}")

        bg_mask, fg_mask = self._partition_with_graph_cut(bg_scores)

        bg_indices = np.where(bg_mask)[0]
        fg_indices = np.where(fg_mask)[0]

        if len(bg_indices) < self.config.min_component_size:
            warnings.warn("Background too small, returning single surface")
            all_idx = np.arange(self.processor.n_points)
            return all_idx, np.array([], dtype=np.intp)

        if len(fg_indices) < self.config.min_component_size:
            warnings.warn("Foreground too small, returning single surface")
            all_idx = np.arange(self.processor.n_points)
            return all_idx, np.array([], dtype=np.intp)

        return bg_indices, fg_indices

    def _build_surface_graph(self) -> csr_matrix:
        """Build connectivity graph respecting surface topology."""
        points = self.processor.points
        normals = self.processor.estimate_normals()
        n_points = len(points)

        tree = cKDTree(points)
        k_density = min(16, n_points - 1)
        distances, _ = tree.query(points, k=k_density)

        median_spacing = np.median(distances[:, 1])
        local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)

        adaptive_radius = (
            median_spacing * 2.5 / (local_density / local_density.mean() + 0.5)
        )
        adaptive_radius = np.clip(adaptive_radius, median_spacing, median_spacing * 5)

        rows, cols, weights = [], [], []
        normal_threshold = np.cos(np.radians(self.config.max_normal_angle_diff))

        for i in range(n_points):
            neighbors = tree.query_ball_point(points[i], adaptive_radius[i])
            for j in neighbors:
                if j <= i:
                    continue

                normal_dot = np.dot(normals[i], normals[j])
                if normal_dot < normal_threshold:
                    continue

                edge_dir = points[j] - points[i]
                edge_len = np.linalg.norm(edge_dir)
                if edge_len < 1e-8:
                    continue
                edge_dir /= edge_len

                tangent_dev_i = abs(np.dot(edge_dir, normals[i]))
                tangent_dev_j = abs(np.dot(edge_dir, normals[j]))
                if tangent_dev_i > 0.7 or tangent_dev_j > 0.7:
                    continue

                dist_weight = np.exp(-edge_len / (self.processor.extent * 0.1))
                normal_weight = (normal_dot + 1) / 2
                tangent_weight = 1.0 - 0.5 * (tangent_dev_i + tangent_dev_j)
                weight = dist_weight * normal_weight * tangent_weight

                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([weight, weight])

        self._surface_graph = csr_matrix(
            (weights, (rows, cols)), shape=(n_points, n_points)
        )
        return self._surface_graph

    def _partition_with_graph_cut(
        self, bg_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Partition points using spectral clustering on the surface graph."""
        n_points = len(bg_scores)

        graph = self._surface_graph.copy()
        rows, cols = graph.nonzero()

        topo_weights = np.array(graph[rows, cols]).flatten()
        score_diff = np.abs(bg_scores[rows] - bg_scores[cols])
        score_sim = np.exp(-(score_diff**2) / 0.1)

        combined_weights = topo_weights * score_sim

        weighted_graph = csr_matrix(
            (combined_weights, (rows, cols)), shape=(n_points, n_points)
        )
        weighted_graph = 0.5 * (weighted_graph + weighted_graph.T)

        try:
            clustering = SpectralClustering(
                n_clusters=2,
                affinity="precompute",
                assign_labels="kmeans",
                random_state=42,
            )
            affinity = weighted_graph.toarray()
            np.fill_diagonal(affinity, 0)
            labels = clustering.fit_predict(affinity)
        except Exception as e:
            warnings.warn(f"Spectral clustering failed: {e}. Using threshold fallback.")
            threshold = np.percentile(
                bg_scores, 100 * (1 - self.config.bg_distance_threshold)
            )
            labels = (bg_scores > threshold).astype(int)

        # Determine which cluster is background (higher mean bg_score)
        cluster_0_score = bg_scores[labels == 0].mean()
        cluster_1_score = bg_scores[labels == 1].mean()
        bg_mask = labels == (0 if cluster_0_score > cluster_1_score else 1)
        fg_mask = ~bg_mask

        return bg_mask, fg_mask

    def _compute_distance_bg_scores(self) -> np.ndarray:
        """Distance-based background score using centroid, principal plane, and UV boundary."""
        points = self.processor.points
        centroid = points.mean(axis=0)

        dist_from_center = np.linalg.norm(points - centroid, axis=1)
        dist_norm = (dist_from_center - dist_from_center.min()) / (
            np.ptp(dist_from_center) + 1e-8
        )

        pca = PCA(n_components=3)
        pca.fit(points - centroid)

        normal_dir = pca.components_[2]
        dist_from_plane = np.abs(np.dot(points - centroid, normal_dir))
        plane_norm = (dist_from_plane - dist_from_plane.min()) / (
            np.ptp(dist_from_plane) + 1e-8
        )

        uv_proj = pca.transform(points - centroid)[:, :2]
        uv_min, uv_max = uv_proj.min(axis=0), uv_proj.max(axis=0)
        uv_center = (uv_min + uv_max) / 2
        uv_extent = (uv_max - uv_min) / 2 + 1e-8
        uv_dist = np.linalg.norm((uv_proj - uv_center) / uv_extent, axis=1)
        boundary_score = np.clip(uv_dist, 0, 1)

        bg_score = 0.3 * dist_norm + 0.3 * plane_norm + 0.4 * boundary_score
        return bg_score

    def _compute_normal_bg_scores(self) -> np.ndarray:
        """Normal-based background score: consistent normals aligned with dominant direction."""
        normals = self.processor.estimate_normals()
        points = self.processor.points
        tree = cKDTree(points)

        k = min(20, len(points) - 1)
        _, indices = tree.query(points, k=k)

        normal_consistency = np.zeros(len(points))
        for i in range(len(points)):
            neighbor_normals = normals[indices[i]]
            dots = np.abs(neighbor_normals @ normals[i])
            normal_consistency[i] = dots.mean()

        normal_pca = PCA(n_components=1)
        normal_pca.fit(normals)
        dominant_normal = normal_pca.components_[0]
        alignment = np.abs(normals @ dominant_normal)

        bg_score = 0.5 * normal_consistency + 0.5 * alignment
        return bg_score

    def _compute_density_bg_scores(self) -> np.ndarray:
        """Density-based background score: low complexity = likely background."""
        complexity = self.processor.compute_local_complexity()
        return 1.0 - complexity

    def _compute_offset_from_center(self) -> np.ndarray:
        """Normalized distance from scene center."""
        points = self.processor.points
        center = self.processor.bounds["center"]
        distances = np.linalg.norm(points - center, axis=1)
        ptp = np.ptp(distances)
        if ptp > 1e-8:
            distances = (distances - distances.min()) / ptp
        else:
            distances = np.zeros_like(distances)
        return distances

    def _compute_hybrid_bg_scores(self) -> np.ndarray:
        """Combine density and offset indicators with adaptive weighting."""
        density_score = self._compute_density_bg_scores()
        offset_score = self._compute_offset_from_center()

        density_conf = 1.0 / (density_score.std() + 0.1)
        offset_conf = 1.0 / (offset_score.std() + 0.1)
        total_conf = density_conf + offset_conf + 1e-10

        bg_score = (
            density_score * (density_conf / total_conf)
            + offset_score * (offset_conf / total_conf)
        )
        return bg_score

    def decompose_k_components(self, k: Optional[int] = None) -> List[np.ndarray]:
        """
        Decompose scene into K components using spatially-constrained
        agglomerative clustering with enhanced features.
        """
        if k is None:
            k = self.config.n_components

        points = self.processor.points
        n_points = len(points)

        if n_points < self.config.min_component_size * k:
            warnings.warn(
                "Too few points for K-components. Returning single component."
            )
            return [np.arange(n_points)]

        features = self._compute_enhanced_features()

        if self.config.use_soft_edges:
            connectivity = self._build_weighted_connectivity()
        else:
            connectivity = kneighbors_graph(
                points,
                n_neighbors=self.config.connectivity_k,
                include_self=False,
                n_jobs=-1,
            )
            connectivity = 0.5 * (connectivity + connectivity.T)

        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="euclidean",
                linkage="ward",
                connectivity=connectivity,
            )
            labels = model.fit_predict(features)
        except Exception as e:
            warnings.warn(
                f"Agglomerative clustering failed ({e}), falling back to K-Means"
            )
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(features)

        component_indices = []
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) >= self.config.min_component_size:
                component_indices.append(idx)

        if len(component_indices) == 0:
            return [np.arange(n_points)]

        return component_indices

    def _compute_enhanced_features(self) -> np.ndarray:
        """Compute rich feature vectors for segmentation."""
        points = self.processor.points
        normals = self.processor.estimate_normals(k=self.config.normal_estimation_k)
        colors = (
            self.processor.colors
            if self.processor.colors is not None
            else np.zeros_like(points)
        )

        scaler = StandardScaler()
        feat_xyz = scaler.fit_transform(points) * self.config.weight_spatial
        feat_normal = normals * self.config.weight_normal

        try:
            from skimage.color import rgb2lab

            colors_rgb = np.clip(colors, 0, 1)
            colors_lab = rgb2lab(colors_rgb.reshape(1, -1, 3)).reshape(-1, 3)
            colors_lab[:, 0] /= 100.0
            colors_lab[:, 1:] /= 128.0
            feat_color = colors_lab * self.config.weight_color
        except ImportError:
            feat_color = colors * self.config.weight_color

        curvature = self.processor.estimate_curvature(k=8)
        curvature_norm = (curvature - curvature.min()) / (
            curvature.max() - curvature.min() + 1e-8
        )
        feat_curvature = curvature_norm.reshape(-1, 1) * self.config.weight_complexity

        normal_variance = self.processor.compute_normal_variance(k=8)
        nv_norm = (normal_variance - normal_variance.min()) / (
            normal_variance.max() - normal_variance.min() + 1e-8
        )
        feat_edge = nv_norm.reshape(-1, 1) * self.config.weight_normal * 0.5

        color_variance = self.processor.compute_color_variance(k=8)
        cv_norm = (color_variance - color_variance.min()) / (
            color_variance.max() - color_variance.min() + 1e-8
        )
        feat_texture = cv_norm.reshape(-1, 1) * self.config.weight_color * 0.5

        return np.hstack(
            [feat_xyz, feat_normal, feat_color, feat_curvature, feat_edge, feat_texture]
        )

    def _build_weighted_connectivity(self) -> csr_matrix:
        """Build connectivity graph with soft edge weights."""
        points = self.processor.points
        normals = self.processor.estimate_normals()
        complexity = self.processor.compute_local_complexity()
        colors = (
            self.processor.colors
            if self.processor.colors is not None
            else np.zeros_like(points)
        )

        adjacency = kneighbors_graph(
            points,
            n_neighbors=self.config.connectivity_k,
            mode="distance",
            include_self=False,
        )

        rows, cols = adjacency.nonzero()
        base_weights = np.array(adjacency[rows, cols]).flatten()

        max_dist = np.percentile(base_weights, 95)
        dist_weights = 1.0 - np.clip(base_weights / max_dist, 0, 1)

        norm_i = normals[rows]
        norm_j = normals[cols]
        dot_products = np.clip(np.sum(norm_i * norm_j, axis=1), -1.0, 1.0)
        normal_weights = (dot_products + 1) / 2

        comp_diff = np.abs(complexity[rows] - complexity[cols])
        complexity_weights = np.exp(-(comp_diff**2) / 0.09)

        col_i = colors[rows]
        col_j = colors[cols]
        color_diff = np.linalg.norm(col_i - col_j, axis=1)
        if self.config.edge_color_adaptive:
            sigma_color = np.percentile(color_diff, 5) + 1e-8
        else:
            sigma_color = 0.5
        color_weights = np.exp(-(color_diff**2) / sigma_color**2)

        combined_weights = (
            dist_weights**0.3
            * normal_weights**0.4
            * complexity_weights**0.2
            * color_weights**0.1
        )

        weighted_adjacency = csr_matrix(
            (combined_weights, (rows, cols)), shape=adjacency.shape
        )
        weighted_adjacency = 0.5 * (weighted_adjacency + weighted_adjacency.T)
        return weighted_adjacency


# =============================================================================
# Adaptive Resolution Calculator
# =============================================================================


class AdaptiveResolutionCalculator:
    """Calculates optimal grid resolution for a point cluster."""

    def __init__(self, config: SurfaceConfig):
        self.config = config

    def calculate(
        self, points: np.ndarray, complexity: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        """Compute resolution based on aspect ratio, point density, and complexity."""
        if not self.config.adaptive_resolution:
            return self.config.resolution_u, self.config.resolution_v

        n_points = len(points)
        if n_points < 12:
            return self.config.min_resolution, self.config.min_resolution

        try:
            pca = PCA(n_components=2)
            projected = pca.fit_transform(points - points.mean(axis=0))
            extents = projected.max(axis=0) - projected.min(axis=0)
            u_extent = max(extents[0], 1e-6)
            v_extent = max(extents[1], 1e-6)
        except Exception:
            u_extent = v_extent = 1.0

        aspect = np.clip(u_extent / v_extent, 0.25, 4.0)

        base_from_points = int(np.sqrt(n_points) * 0.5)
        base_res = int(
            np.clip(base_from_points, self.config.min_resolution, self.config.base_resolution)
        )

        if complexity is not None and len(complexity) > 0:
            avg_complexity = np.percentile(complexity, 80)
            complexity_factor = (
                1.0 + self.config.resolution_complexity_sensitivity * avg_complexity
            )
        else:
            complexity_factor = 1.0

        total_res = base_res * complexity_factor

        if aspect >= 1.0:
            res_u = int(total_res)
            res_v = int(total_res / aspect)
        else:
            res_u = int(total_res * aspect)
            res_v = int(total_res)

        res_u = int(np.clip(res_u, self.config.min_resolution, self.config.max_resolution))
        res_v = int(np.clip(res_v, self.config.min_resolution, self.config.max_resolution))

        return res_u, res_v


# =============================================================================
# NURBS Surface Fitter
# =============================================================================


class NURBSSurfaceFitter:
    """Fits NURBS surfaces to point cloud regions with adaptive resolution."""

    def __init__(self, config: SurfaceConfig):
        self.config = config
        self.res_calculator = AdaptiveResolutionCalculator(config)
        self.sampling_generator = AdaptiveSamplingGenerator(
            AdaptiveSamplingConfig(
                sampling_resolution_factor=config.sampling_resolution_factor,
                weight_curvature=config.weight_complexity,
                weight_color_variance=config.weight_color,
                weight_normal_variance=config.weight_normal,
            )
        )

    def _generate_adaptive_samples(
        self,
        surface_data: NURBSSurfaceData,
        sampling_resolution: Optional[Tuple[int, int]] = None,
    ) -> AdaptiveSamplingResult:
        if sampling_resolution is not None:
            self.sampling_generator.config.sampling_resolution_factor = (
                sampling_resolution[0]
                / surface_data.control_points.shape[0]
            )

        return self.sampling_generator.generate(
            control_points=surface_data.control_points,
            control_colors=surface_data.control_colors,
            knots_u=surface_data.knots_u,
            knots_v=surface_data.knots_v,
            degree=surface_data.degree_u,
        )

    # In NURBSSurfaceFitter, modify fit_surface signature and grid creation:
    # In NURBSSurfaceFitter class, add this import at the top of the file:
    from modules.fitting.mesh_patch import least_squares_bspline_surface

    # Then modify fit_surface to add a new path:

    def fit_surface(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            label: str = "surface",
            complexity: Optional[np.ndarray] = None,
            sampling_resolution: Optional[Tuple[int, int]] = None,
            override_resolution: Optional[Tuple[int, int]] = None,
            parameterization: Optional[str] = None,
            point_weights: Optional[np.ndarray] = None,
            cameras: Optional[List['CameraObject']] = None,
            normals: Optional[np.ndarray] = None,
            faces: Optional[np.ndarray] = None,  # NEW: triangle faces for mesh input
            use_least_squares: bool = True,  # NEW: toggle LS fitting
    ) -> NURBSSurfaceData:
        if len(points) < 16:
            raise ValueError(f"Need at least 16 points, got {len(points)}")

        # --- Determine resolution ---
        if override_resolution is not None:
            res_u = int(np.clip(override_resolution[0],
                                self.config.min_resolution, self.config.max_resolution))
            res_v = int(np.clip(override_resolution[1],
                                self.config.min_resolution, self.config.max_resolution))
        else:
            res_u, res_v = self.res_calculator.calculate(points, complexity)

        method = parameterization if parameterization else self.config.parameterization

        print(f"[Fit] Label: {label}, Res: {res_u}x{res_v}, Method: {method}")

        # ================================================================
        # NEW PATH: Least-Squares B-Spline Approximation
        # ================================================================
        if use_least_squares:
            # --- Step 1: Parameterize ---
            if faces is not None and len(faces) > 0:
                # Mesh input → harmonic parameterization (exploits topology)
                from modules.fitting.mesh_patch import harmonic_parameterization
                try:
                    uv_coords = harmonic_parameterization(points, faces)
                    print(f"[Fit] Using harmonic parameterization from mesh ({len(faces)} faces)")
                except (ValueError, Exception) as e:
                    print(f"[Fit] Harmonic param failed ({e}), falling back to spherical")
                    uv_coords = self._parameterize_spherical(points)
            else:
                # Point cloud → existing parameterization methods
                if method == "spherical":
                    uv_coords = self._parameterize_spherical(points)
                elif method == "geodesic":
                    uv_coords = self._parameterize_geodesic(points)
                else:
                    uv_coords = self._parameterize_pca(points)

            # --- Step 2: Least-Squares B-Spline Fit ---
            degree = min(self.config.degree_u, res_u - 1, res_v - 1)

            ctrl_pts, ctrl_colors, knots_u, knots_v = least_squares_bspline_surface(
                points=points,
                uv=uv_coords,
                n_ctrl_u=res_u,
                n_ctrl_v=res_v,
                degree=degree,
                smoothing=self.config.smoothing,
                colors=colors,
            )

            surface_data = NURBSSurfaceData(
                control_points=ctrl_pts.astype(np.float32),
                control_colors=np.clip(ctrl_colors, 0, 1).astype(np.float32)
                if ctrl_colors is not None
                else np.full((res_u, res_v, 3), 0.5, dtype=np.float32),
                knots_u=knots_u.astype(np.float32),
                knots_v=knots_v.astype(np.float32),
                degree_u=degree,
                degree_v=degree,
                label=label,
            )

        # ================================================================
        # LEGACY PATH: Grid scatter + Gaussian blur (existing behavior)
        # ================================================================
        else:
            if method == "pca" and self.config.use_geodesic_uv and len(points) > 12:
                method = "geodesic"
            if method == "spherical":
                uv_coords = self._parameterize_spherical(points)
            elif method == "geodesic":
                uv_coords = self._parameterize_geodesic(points)
            else:
                uv_coords = self._parameterize_pca(points)

            grid_xyz, grid_rgb = self._create_grid_samples(
                points, colors, uv_coords, res_u, res_v,
                point_weights=point_weights,
            )
            is_spherical = method == "spherical"
            surface_data = self._fit_bspline_to_grid(
                grid_xyz, grid_rgb, label, res_u, res_v, is_spherical
            )

        # --- Common post-processing ---
        surface_data.bounds = {
            "min": points.min(axis=0),
            "max": points.max(axis=0),
            "center": points.mean(axis=0),
        }

        if self.config.generate_adaptive_samples:
            sampling_result = self._generate_adaptive_samples(
                surface_data, sampling_resolution
            )
            surface_data.sampling_u_1D = sampling_result.intervals_u
            surface_data.sampling_v_1D = sampling_result.intervals_v
            surface_data.grid_samplings_u = sampling_result.grid_u
            surface_data.grid_samplings_v = sampling_result.grid_v
            surface_data.complexity_map = sampling_result.complexity_map

        return surface_data

    def fit_surface2(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            label: str = "surface",
            complexity: Optional[np.ndarray] = None,
            sampling_resolution: Optional[Tuple[int, int]] = None,
            override_resolution: Optional[Tuple[int, int]] = None,
            parameterization: Optional[str] = None,
            point_weights: Optional[np.ndarray] = None,  # NEW
            cameras: Optional[List['CameraObject']] = None,  # NEW
            normals: Optional[np.ndarray] = None,  # NEW
    ) -> NURBSSurfaceData:
        """Fit a NURBS surface to a point cloud region.

        Args:
            points: [N, 3] point positions
            colors: [N, 3] optional RGB colors
            label: Surface label string
            complexity: [N] per-point complexity scores
            sampling_resolution: Override for adaptive sampling resolution
            override_resolution: Override for control grid resolution
            parameterization: UV parameterization method override
            point_weights: [N] per-point importance weights from cameras
            cameras: Training cameras for normal orientation
            normals: [N, 3] pre-computed camera-consistent normals
        """
        if len(points) < 16:
            raise ValueError(f"Need at least 16 points, got {len(points)}")

        if override_resolution is not None:
            res_u = int(
                np.clip(
                    override_resolution[0],
                    self.config.min_resolution,
                    self.config.max_resolution,
                )
            )
            res_v = int(
                np.clip(
                    override_resolution[1],
                    self.config.min_resolution,
                    self.config.max_resolution,
                )
            )
        else:
            res_u, res_v = self.res_calculator.calculate(points, complexity)

        method = parameterization if parameterization else self.config.parameterization

        if method == "pca" and self.config.use_geodesic_uv and len(points) > 12:
            method = "geodesic"

        if method == "spherical":
            uv_coords = self._parameterize_spherical(points)
        elif method == "conformal":
            from modules.fitting.parametrization.conformal_uv import conformal_parameterize
            uv_coords = conformal_parameterize(points)

        elif method == "geodesic":
            uv_coords = self._parameterize_geodesic(points)
        else:
            uv_coords = self._parameterize_pca(points)

        print(f"[Fit] Label: {label}, Res: {res_u}x{res_v}, Method: {method}")

        # Use camera-weighted grid creation if weights available
        grid_xyz, grid_rgb = self._create_grid_samples(
            points, colors, uv_coords, res_u, res_v,
            point_weights=point_weights,  # Pass weights through
        )

        is_spherical = method == "spherical"
        surface_data = self._fit_bspline_to_grid(
            grid_xyz, grid_rgb, label, res_u, res_v, is_spherical
        )

        surface_data.bounds = {
            "min": points.min(axis=0),
            "max": points.max(axis=0),
            "center": points.mean(axis=0),
        }

        if self.config.generate_adaptive_samples:
            sampling_result = self._generate_adaptive_samples(
                surface_data, sampling_resolution
            )
            surface_data.sampling_u_1D = sampling_result.intervals_u
            surface_data.sampling_v_1D = sampling_result.intervals_v
            surface_data.grid_samplings_u = sampling_result.grid_u
            surface_data.grid_samplings_v = sampling_result.grid_v
            surface_data.complexity_map = sampling_result.complexity_map

        return surface_data

    def _parameterize_spherical(self, points: np.ndarray) -> np.ndarray:
        """Spherical UV parameterization for 360/object-centric scenes."""
        centroid = points.mean(axis=0)
        centered = points - centroid

        r = np.linalg.norm(centered, axis=1)
        r = np.maximum(r, 1e-8)

        phi = np.arccos(np.clip(centered[:, 2] / r, -1, 1))
        theta = np.arctan2(centered[:, 1], centered[:, 0])

        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi

        uv_coords = np.stack([u, v], axis=1)
        return np.clip(uv_coords, 0.001, 0.999)

    def _parameterize_pca(self, points: np.ndarray) -> np.ndarray:
        """Simple PCA-based UV parameterization."""
        center = points.mean(axis=0)
        centered = points - center

        pca = PCA(n_components=3)
        pca.fit(centered)
        projected = pca.transform(centered)[:, :2]

        uv_min = projected.min(axis=0)
        uv_max = projected.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range < 1e-6] = 1.0

        uv_coords = (projected - uv_min) / uv_range
        return np.clip(uv_coords, 0.01, 0.99)

    def _parameterize_geodesic(self, points: np.ndarray) -> np.ndarray:
        """UV parameterization using geodesic distances on the surface graph."""
        n_points = len(points)

        if n_points < 12:
            return self._parameterize_pca(points)

        tree = cKDTree(points)
        k = min(12, n_points - 1)
        distances, indices = tree.query(points, k=k)

        rows = np.repeat(np.arange(n_points), k)
        cols = indices.flatten()
        weights = distances.flatten()

        graph = csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))
        graph = 0.5 * (graph + graph.T)

        pca = PCA(n_components=1)
        proj = pca.fit_transform(points - points.mean(axis=0)).flatten()
        anchor1 = int(np.argmin(proj))
        anchor2 = int(np.argmax(proj))

        try:
            dist_from_1 = shortest_path(graph, indices=[anchor1], directed=False)[0]
            dist_from_2 = shortest_path(graph, indices=[anchor2], directed=False)[0]
        except Exception:
            return self._parameterize_pca(points)

        if np.isinf(dist_from_1).any() or np.isinf(dist_from_2).any():
            return self._parameterize_pca(points)

        total_dist = dist_from_1[anchor2]
        if total_dist < 1e-8:
            return self._parameterize_pca(points)

        u_coords = dist_from_1 / total_dist

        axis = points[anchor2] - points[anchor1]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return self._parameterize_pca(points)
        axis = axis / axis_norm

        centered = points - points[anchor1]
        along_axis = np.dot(centered, axis)[:, None] * axis
        perpendicular = centered - along_axis

        if np.std(perpendicular) > 1e-6:
            pca_perp = PCA(n_components=1)
            v_coords = pca_perp.fit_transform(perpendicular).flatten()
            v_range = v_coords.max() - v_coords.min()
            if v_range > 1e-8:
                v_coords = (v_coords - v_coords.min()) / v_range
            else:
                v_coords = np.zeros(n_points)
        else:
            v_coords = np.zeros(n_points)

        uv_coords = np.stack([u_coords, v_coords], axis=1)
        return np.clip(uv_coords, 0.001, 0.999)

    def _create_knot_vector(self, n_ctrl: int, degree: int) -> np.ndarray:
        """Create clamped uniform knot vector."""
        n_knots = n_ctrl + degree + 1
        n_internal = n_knots - 2 * (degree + 1)

        knots = np.zeros(n_knots)
        knots[-degree - 1 :] = 1.0

        if n_internal > 0:
            internal = np.linspace(0, 1, n_internal + 2)[1:-1]
            knots[degree + 1 : degree + 1 + n_internal] = internal

        return knots

    def _create_grid_samples(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray],
            uv_coords: np.ndarray,
            res_u: int,
            res_v: int,
            point_weights: Optional[np.ndarray] = None,  # NEW
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample points onto a regular grid with camera-weighted interpolation and robust depth rejection."""
        u_vals = np.linspace(0, 1, res_u)
        v_vals = np.linspace(0, 1, res_v)

        grid_xyz = np.full((res_u, res_v, 3), np.nan)
        grid_rgb = np.full((res_u, res_v, 3), np.nan)

        uv_tree = cKDTree(uv_coords)
        k = min(16, len(points))  # Increased K slightly for better statistical robustness

        # Adaptive hole threshold
        nn_dists, _ = uv_tree.query(uv_coords, k=min(4, len(points)))
        median_nn_dist = np.median(nn_dists[:, 1]) if nn_dists.shape[1] > 1 else 0.1
        hole_threshold = median_nn_dist * 3.0

        is_hole = np.zeros((res_u, res_v), dtype=bool)

        # Default weights: uniform
        if point_weights is None:
            point_weights = np.ones(len(points), dtype=np.float64)

        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                query = np.array([u, v])
                dists, idxs = uv_tree.query(query, k=k)

                if dists[0] > hole_threshold:
                    is_hole[i, j] = True
                else:
                    # --- CRITICAL FIX: Robust Depth Rejection in UV space ---
                    pts = points[idxs]

                    # 1. Find the spatial median of the queried points
                    median_pt = np.median(pts, axis=0)

                    # 2. Compute distances to the median
                    dists_to_median = np.linalg.norm(pts - median_pt, axis=1)
                    mad = np.median(dists_to_median) + 1e-6

                    # 3. Reject points that are geometrically far from the cluster's median (e.g. background points bleeding into foreground UVs)
                    valid_mask = dists_to_median <= (2.5 * mad)

                    # Fallback if too aggressive
                    if not np.any(valid_mask):
                        valid_mask = np.ones(k, dtype=bool)

                    valid_dists = dists[valid_mask]
                    valid_idxs = idxs[valid_mask]

                    if valid_dists[0] < 1e-10:
                        weights = np.zeros(len(valid_dists))
                        weights[0] = 1.0
                    else:
                        # Combine inverse-distance with camera observation weights
                        inv_dist = 1.0 / (valid_dists + 1e-8)
                        cam_w = point_weights[valid_idxs]
                        weights = inv_dist * cam_w
                        weights /= weights.sum()

                    grid_xyz[i, j] = np.sum(
                        points[valid_idxs] * weights[:, None], axis=0
                    )
                    if colors is not None:
                        grid_rgb[i, j] = np.sum(
                            colors[valid_idxs] * weights[:, None], axis=0
                        )
                    else:
                        grid_rgb[i, j] = 0.5

        grid_xyz = self._inpaint_holes(grid_xyz, is_hole)
        grid_rgb = self._inpaint_holes(grid_rgb, is_hole)
        grid_rgb = np.clip(grid_rgb, 0, 1)

        return grid_xyz, grid_rgb
    def _create_grid_samples2(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        uv_coords: np.ndarray,
        res_u: int,
        res_v: int,
        point_weights: Optional[np.ndarray] = None,  # NEW
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample points onto a regular grid with camera-weighted interpolation."""
        u_vals = np.linspace(0, 1, res_u)
        v_vals = np.linspace(0, 1, res_v)

        grid_xyz = np.full((res_u, res_v, 3), np.nan)
        grid_rgb = np.full((res_u, res_v, 3), np.nan)

        uv_tree = cKDTree(uv_coords)
        k = min(12, len(points))

        # Adaptive hole threshold
        nn_dists, _ = uv_tree.query(uv_coords, k=min(4, len(points)))
        median_nn_dist = np.median(nn_dists[:, 1]) if nn_dists.shape[1] > 1 else 0.1
        hole_threshold = median_nn_dist * 3.0

        is_hole = np.zeros((res_u, res_v), dtype=bool)

        # Default weights: uniform
        if point_weights is None:
            point_weights = np.ones(len(points), dtype=np.float64)

        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                query = np.array([u, v])
                dists, idxs = uv_tree.query(query, k=k)

                if dists[0] > hole_threshold:
                    is_hole[i, j] = True
                else:
                    if dists[0] < 1e-10:
                        weights = np.zeros(k)
                        weights[0] = 1.0
                    else:
                        # Combine inverse-distance with camera observation weights
                        inv_dist = 1.0 / (dists + 1e-8)
                        cam_w = point_weights[idxs]
                        weights = inv_dist * cam_w
                        weights /= weights.sum()

                    grid_xyz[i, j] = np.sum(
                        points[idxs] * weights[:, None], axis=0
                    )
                    if colors is not None:
                        grid_rgb[i, j] = np.sum(
                            colors[idxs] * weights[:, None], axis=0
                        )
                    else:
                        grid_rgb[i, j] = 0.5

        grid_xyz = self._inpaint_holes(grid_xyz, is_hole)
        grid_rgb = self._inpaint_holes(grid_rgb, is_hole)
        grid_rgb = np.clip(grid_rgb, 0, 1)

        return grid_xyz, grid_rgb

    def _inpaint_holes(self, grid: np.ndarray, is_hole: np.ndarray) -> np.ndarray:
        """Fill holes by iteratively averaging valid neighbors."""
        filled = grid.copy()
        remaining_holes = is_hole.copy()

        max_iterations = max(grid.shape[0], grid.shape[1])

        for _ in range(max_iterations):
            if not remaining_holes.any():
                break

            valid_mask = ~np.isnan(filled[..., 0])
            dilated_valid = binary_dilation(valid_mask)
            fillable = remaining_holes & dilated_valid

            if not fillable.any():
                global_mean = np.nanmean(filled, axis=(0, 1))
                for i, j in zip(*np.where(remaining_holes)):
                    filled[i, j] = global_mean
                break

            for i, j in zip(*np.where(fillable)):
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        if valid_mask[ni, nj]:
                            neighbors.append(filled[ni, nj])
                if neighbors:
                    filled[i, j] = np.mean(neighbors, axis=0)
                    remaining_holes[i, j] = False
        # Add tiny noise to in-painted regions to prevent mathematical zeroes in tangents
        noise = np.random.normal(0, 1e-4, size=grid.shape)
        grid[is_hole] += noise[is_hole]

        return filled

    def _fit_bspline_to_gridd(
            self,
            grid_xyz: np.ndarray,
            grid_rgb: np.ndarray,
            label: str,
            res_u: int,
            res_v: int,
            is_spherical: bool = False,
    ) -> NURBSSurfaceData:
        """Create B-spline surface from grid with robust despiking and smoothing."""
        from scipy.ndimage import median_filter, gaussian_filter  # Ensure median_filter is imported

        degree_u = min(self.config.degree_u, res_u - 1)
        degree_v = min(self.config.degree_v, res_v - 1)

        if self.config.smoothing > 0:
            # --- CRITICAL FIX: Eradicate spikes BEFORE Gaussian blurring ---
            # A 3x3 median filter destroys salt-and-pepper geometric outliers
            for c in range(3):
                grid_xyz[..., c] = median_filter(grid_xyz[..., c], size=3)
                grid_rgb[..., c] = median_filter(grid_rgb[..., c], size=3)

            sigma = self.config.smoothing * min(res_u, res_v) / 10.0
            # Use 'wrap' for U direction in spherical parameterization (theta wraps)
            mode_u = "wrap" if is_spherical else "nearest"
            mode_v = "nearest"

            for c in range(3):
                grid_xyz[..., c] = gaussian_filter(
                    grid_xyz[..., c], sigma=sigma, mode=(mode_v, mode_u)
                )
                grid_rgb[..., c] = gaussian_filter(
                    grid_rgb[..., c], sigma=sigma, mode=(mode_v, mode_u)
                )

        knots_u = self._create_knot_vector(res_u, degree_u)
        knots_v = self._create_knot_vector(res_v, degree_v)

        return NURBSSurfaceData(
            control_points=grid_xyz.astype(np.float32),
            control_colors=np.clip(grid_rgb, 0, 1).astype(np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=degree_u,
            degree_v=degree_v,
            label=label,
        )

    def _fit_bspline_to_grid(
        self,
        grid_xyz: np.ndarray,
        grid_rgb: np.ndarray,
        label: str,
        res_u: int,
        res_v: int,
        is_spherical: bool = False,
        ) -> NURBSSurfaceData:
        # """Fit a proper B-spline surface to the resampled grid."""
        from modules.fitting.bspline_fitting import BSplineSurfaceFitter
        from scipy.ndimage import median_filter

        degree_u = min(self.config.degree_u, res_u - 1)
        degree_v = min(self.config.degree_v, res_v - 1)

        # Pre-filter to remove outlier spikes (keep this — it's preprocessing)
        if self.config.smoothing > 0:
            for c in range(3):
                grid_xyz[..., c] = median_filter(grid_xyz[..., c], size=3)
                grid_rgb[..., c] = median_filter(grid_rgb[..., c], size=3)

        # === THE KEY CHANGE: Solve for control points via least-squares ===
        # The grid samples are DATA, not control points.
        # The smoothing parameter λ replaces the Gaussian blur.
        fitter = BSplineSurfaceFitter(
            n_ctrl_u=res_u,
            n_ctrl_v=res_v,
            degree_u=degree_u,
            degree_v=degree_v,
            smoothing=self.config.smoothing,  # λ in the regularized normal equations
            data_dependent_knots=False,  # Grid is uniform → uniform knots are fine
        )

        result = fitter.fit_from_grid(grid_xyz, grid_rgb)

        print(f"[BSpline Fit] {label}: RMS residual = {result.residual_rms:.6f}")

        return NURBSSurfaceData(
            control_points=result.control_points,
            control_colors=result.control_colors,
            knots_u=result.knots_u,
            knots_v=result.knots_v,
            degree_u=result.degree_u,
            degree_v=result.degree_v,
            label=label,
        )

"""
BSpline Surface Fitting from Point Clouds via Least-Squares Approximation.

Implements the classical surface approximation algorithm from:
  Piegl & Tiller, "The NURBS Book" (2nd ed.), Chapter 9.

Given N scattered 3D points and their UV parameterization,
solves for the (m+1)×(n+1) control points P_{ij} that minimize:

    min_{P} || A·P - D ||² + λ·|| L·P ||²

where:
    A  = [N_i(u_k) · N_j(v_k)]  is the (N × m·n) collocation matrix
    D  = [x_k]                   is the data vector
    L  = discrete Laplacian      regularization on the control grid
    λ  = smoothing weight

This is the CORRECT way to compute BSpline control points from data,
as opposed to treating grid-interpolated samples as control points.
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from scipy.sparse import csr_matrix, vstack as sparse_vstack, diags
from scipy.sparse.linalg import lsqr


# ---------------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------------

@dataclass
class BSplineFitResult:
    """Result of BSpline surface fitting."""
    control_points: np.ndarray    # [H, W, 3]
    control_colors: np.ndarray    # [H, W, 3]
    knots_u: np.ndarray           # [H + degree_u + 1]
    knots_v: np.ndarray           # [W + degree_v + 1]
    degree_u: int
    degree_v: int
    residual_rms: float           # RMS fitting error
    parameterization: np.ndarray  # [N, 2] UV coords used


# ---------------------------------------------------------------------------
#  B-spline basis function evaluator (NumPy)
# ---------------------------------------------------------------------------

class BSplineBasis:
    """
    Vectorized Cox-de Boor B-spline basis evaluation in NumPy.

    The recursion structure mirrors the PyTorch implementation in
    ``modules/utils/BSpline.py :: cox_de_boor_basis_and_derivative``
    so that basis values agree to floating-point precision.
    """

    @staticmethod
    def evaluate(
        params: np.ndarray,
        knots: np.ndarray,
        degree: int,
        n_ctrl: int,
    ) -> np.ndarray:
        """
        Evaluate all B-spline basis functions at given parameter values.

        Parameters
        ----------
        params : (M,) parameter values, typically in [0, 1].
        knots  : (n_ctrl + degree + 1,) clamped knot vector.
        degree : polynomial degree (usually 3).
        n_ctrl : number of control points.

        Returns
        -------
        B : (M, n_ctrl) basis matrix, B[k, i] = N_{i,p}(params[k]).
        """
        eps = 1e-12
        M = len(params)
        u_col = params[:, None]                             # (M, 1)

        # -- degree 0: indicator on half-open knot spans ----------------
        n_spans = n_ctrl + degree                           # = len(knots) - 1
        left = knots[:n_spans]                              # (n_spans,)
        right = knots[1 : n_spans + 1]                     # (n_spans,)
        N = ((u_col >= left) & (u_col < right)).astype(np.float64)

        # include right endpoint u == knots[-1]
        N[params == knots[-1], -1] = 1.0

        # -- recursion: degree 1 … p -----------------------------------
        for d in range(1, degree + 1):
            n_basis = n_ctrl + degree - d
            left_denom = np.maximum(knots[d : d + n_basis] - knots[:n_basis], eps)
            right_denom = np.maximum(
                knots[d + 1 : d + 1 + n_basis] - knots[1 : 1 + n_basis], eps
            )
            left_coeff = (u_col - knots[:n_basis]) / left_denom
            right_coeff = (knots[d + 1 : d + 1 + n_basis] - u_col) / right_denom

            N_shifted = np.zeros((M, n_basis))
            N_shifted[:, : min(n_basis, N.shape[1] - 1)] = N[:, 1 : 1 + n_basis]
            N = left_coeff * N[:, :n_basis] + right_coeff * N_shifted

        assert N.shape == (M, n_ctrl), (
            f"Basis shape mismatch: expected ({M}, {n_ctrl}), got {N.shape}"
        )
        return N

    # ---------------------------------------------------------------

    @staticmethod
    def create_knot_vector(
        n_ctrl: int,
        degree: int,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a clamped knot vector.

        When *params* is provided the internal knots are placed via
        the averaging method (Piegl & Tiller, Eq. 9.68).  The parameter
        array is automatically sub-sampled to ≤ 1 000 values to avoid
        artefacts from clustered distributions (e.g. spherical UV).

        Parameters
        ----------
        n_ctrl : number of control points.
        degree : polynomial degree.
        params : (N,) optional sorted parameter values in [0, 1].

        Returns
        -------
        knots : (n_ctrl + degree + 1,) clamped, non-decreasing.
        """
        n_knots = n_ctrl + degree + 1
        knots = np.zeros(n_knots, dtype=np.float64)
        knots[-degree - 1 :] = 1.0

        n_internal = n_ctrl - degree - 1
        if n_internal <= 0:
            return knots

        if params is not None and len(params) > 1:
            # Sub-sample to avoid clustering artefacts
            p_sorted = np.sort(params)
            max_samples = 1000
            if len(p_sorted) > max_samples:
                idx = np.linspace(0, len(p_sorted) - 1, max_samples).astype(int)
                p_sorted = p_sorted[idx]
            N_p = len(p_sorted)
            d_step = (N_p + 1) / (n_internal + 1)
            for j in range(1, n_internal + 1):
                fval = j * d_step
                i = int(fval)
                alpha = fval - i
                i = min(i, N_p - 2)
                knots[degree + j] = (1 - alpha) * p_sorted[i] + alpha * p_sorted[i + 1]
        else:
            knots[degree + 1 : degree + 1 + n_internal] = np.linspace(
                0, 1, n_internal + 2
            )[1:-1]

        # Enforce clamping invariants
        knots[: degree + 1] = 0.0
        knots[-degree - 1 :] = 1.0
        assert np.all(np.diff(knots) >= -1e-10), "Knot vector not non-decreasing"
        return knots



# ---------------------------------------------------------------------------
#  Surface fitter
# ---------------------------------------------------------------------------

class BSplineSurfaceFitter:
    """
    Regularised least-squares B-spline surface fitting.

    Two entry points:

    * ``fit()``           – scattered 3-D points with UV parameterisation.
    * ``fit_from_grid()`` – data already on a regular UV grid (fast
                            separable solver).

    Both solve for control points P that minimise

        Σ_k ‖S(u_k, v_k) − Q_k‖² + λ · ‖L P‖²

    where *L* is a discrete thin-plate regulariser.
    """

    def __init__(
        self,
        n_ctrl_u: int = 32,
        n_ctrl_v: int = 32,
        degree_u: int = 3,
        degree_v: int = 3,
        smoothing: float = 0.01,
        data_dependent_knots: bool = True,
    ):
        self.n_ctrl_u = n_ctrl_u
        self.n_ctrl_v = n_ctrl_v
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.smoothing = smoothing
        self.data_dependent_knots = data_dependent_knots

    # ===================================================================
    #  PUBLIC: scattered data
    # ===================================================================

    def fit(
        self,
        points: np.ndarray,
        uv_params: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_weights: Optional[np.ndarray] = None,
    ) -> BSplineFitResult:
        """
        Fit a B-spline surface to **scattered** 3-D points.

        Parameters
        ----------
        points        : (N, 3) world-space positions.
        uv_params     : (N, 2) parameter values in [0, 1]².
        colors        : (N, 3) optional RGB.
        point_weights : (N,)   optional per-point importance weights.
        """
        N = len(points)
        Hu, Wv = self.n_ctrl_u, self.n_ctrl_v
        du = min(self.degree_u, Hu - 1)
        dv = min(self.degree_v, Wv - 1)

        # --- Knot vectors -------------------------------------------------
        if self.data_dependent_knots:
            knots_u = BSplineBasis.create_knot_vector(Hu, du, uv_params[:, 0])
            knots_v = BSplineBasis.create_knot_vector(Wv, dv, uv_params[:, 1])
        else:
            knots_u = BSplineBasis.create_knot_vector(Hu, du)
            knots_v = BSplineBasis.create_knot_vector(Wv, dv)

        # --- Basis matrices -----------------------------------------------
        Bu = BSplineBasis.evaluate(uv_params[:, 0], knots_u, du, Hu)  # (N, Hu)
        Bv = BSplineBasis.evaluate(uv_params[:, 1], knots_v, dv, Wv)  # (N, Wv)

        # --- Sparse collocation matrix (vectorised chunks) ----------------
        A = self._build_collocation_sparse(Bu, Bv, Hu, Wv)

        # --- Normal equations LHS -----------------------------------------
        if point_weights is not None:
            W_diag = diags(point_weights.astype(np.float64))
            AtWA = A.T @ W_diag @ A
        else:
            AtWA = A.T @ A
            W_diag = None

        L = self._build_regularization_matrix_full(Hu, Wv)
        lhs = AtWA + self.smoothing * (L.T @ L)

        # --- Solve per coordinate -----------------------------------------
        n_total = Hu * Wv
        ctrl_pts = np.zeros((n_total, 3))
        for dim in range(3):
            d_vec = points[:, dim]
            rhs = (A.T @ (W_diag @ d_vec)) if W_diag is not None else (A.T @ d_vec)
            result = lsqr(lhs, rhs, atol=1e-10, btol=1e-10)
            ctrl_pts[:, dim] = result[0]

        ctrl_grid = ctrl_pts.reshape(Hu, Wv, 3)

        # --- Colours (same system) ----------------------------------------
        if colors is not None:
            ctrl_colors = np.zeros((n_total, 3))
            for dim in range(3):
                d_vec = colors[:, dim]
                rhs = (A.T @ (W_diag @ d_vec)) if W_diag is not None else (A.T @ d_vec)
                result = lsqr(lhs, rhs, atol=1e-10, btol=1e-10)
                ctrl_colors[:, dim] = result[0]
            ctrl_colors_grid = np.clip(ctrl_colors.reshape(Hu, Wv, 3), 0, 1)
        else:
            ctrl_colors_grid = np.full((Hu, Wv, 3), 0.5)

        # --- Residual -----------------------------------------------------
        fitted = A @ ctrl_pts
        residual = float(np.sqrt(np.mean(np.sum((fitted - points) ** 2, axis=1))))

        return BSplineFitResult(
            control_points=ctrl_grid.astype(np.float32),
            control_colors=ctrl_colors_grid.astype(np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=du,
            degree_v=dv,
            residual_rms=residual,
            parameterization=uv_params,
        )

    # ===================================================================
    #  PUBLIC: gridded data (fast separable solver)
    # ===================================================================

    def fit_from_grid(
        self,
        grid_xyz: np.ndarray,
        grid_rgb: np.ndarray,
        n_ctrl_u: Optional[int] = None,
        n_ctrl_v: Optional[int] = None,
    ) -> BSplineFitResult:
        """
        Fit a B-spline surface to **regularly-gridded** data.

        Exploits the tensor-product structure for an O(res · n²) separable
        solve instead of the O(res² · n²) general case.

        NOTE: uses axis-aligned regularisation only (no cross-derivative)
        because the separable solver requires L = L_u ⊗ I + I ⊗ L_v.

        Parameters
        ----------
        grid_xyz : (res_u, res_v, 3) gridded 3-D positions.
        grid_rgb : (res_u, res_v, 3) gridded RGB colours.
        n_ctrl_u : control points in U (defaults to ``self.n_ctrl_u``).
        n_ctrl_v : control points in V (defaults to ``self.n_ctrl_v``).
        """
        res_u, res_v, _ = grid_xyz.shape
        Hu = n_ctrl_u if n_ctrl_u is not None else min(self.n_ctrl_u, res_u)
        Wv = n_ctrl_v if n_ctrl_v is not None else min(self.n_ctrl_v, res_v)
        du = min(self.degree_u, Hu - 1)
        dv = min(self.degree_v, Wv - 1)

        # Uniform parameterisation on the grid
        u_params = np.linspace(0.0, 1.0, res_u)
        v_params = np.linspace(0.0, 1.0, res_v)

        knots_u = BSplineBasis.create_knot_vector(Hu, du, u_params)
        knots_v = BSplineBasis.create_knot_vector(Wv, dv, v_params)

        Bu = BSplineBasis.evaluate(u_params, knots_u, du, Hu)  # (res_u, Hu)
        Bv = BSplineBasis.evaluate(v_params, knots_v, dv, Wv)  # (res_v, Wv)

        # 1-D second-difference regularisation matrices
        D2u = self._second_diff_1d(Hu)
        D2v = self._second_diff_1d(Wv)

        lhs_u = Bu.T @ Bu + self.smoothing * (D2u.T @ D2u)  # (Hu, Hu)
        lhs_v = Bv.T @ Bv + self.smoothing * (D2v.T @ D2v)  # (Wv, Wv)

        ctrl_xyz = np.zeros((Hu, Wv, 3))
        ctrl_rgb = np.zeros((Hu, Wv, 3))

        for dim in range(3):
            # Position
            rhs = Bu.T @ grid_xyz[:, :, dim] @ Bv          # (Hu, Wv)
            Z = np.linalg.solve(lhs_u, rhs)                # lhs_u Z = rhs
            ctrl_xyz[:, :, dim] = np.linalg.solve(lhs_v, Z.T).T

            # Colour
            rhs_c = Bu.T @ grid_rgb[:, :, dim] @ Bv
            Z_c = np.linalg.solve(lhs_u, rhs_c)
            ctrl_rgb[:, :, dim] = np.linalg.solve(lhs_v, Z_c.T).T

        ctrl_rgb = np.clip(ctrl_rgb, 0.0, 1.0)

        # Residual
        residual = self._compute_grid_residual(grid_xyz, ctrl_xyz, Bu, Bv)

        # Parameterisation grid for the result container
        uu, vv = np.meshgrid(u_params, v_params, indexing="ij")
        uv_grid = np.stack([uu, vv], axis=-1).reshape(-1, 2)

        return BSplineFitResult(
            control_points=ctrl_xyz.astype(np.float32),
            control_colors=ctrl_rgb.astype(np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=du,
            degree_v=dv,
            residual_rms=residual,
            parameterization=uv_grid,
        )

    # ===================================================================
    #  PUBLIC: verify NumPy ↔ PyTorch basis consistency
    # ===================================================================

    def verify_basis_consistency(
        self,
        knots_u: np.ndarray,
        knots_v: np.ndarray,
        Hu: int,
        Wv: int,
        du: int,
        dv: int,
        n_test: int = 50,
        atol: float = 1e-5,
    ) -> bool:
        """
        Check that the NumPy basis agrees with the PyTorch backend.

        Returns True if they match, False (with a warning) otherwise.
        """
        try:
            import torch
            from modules.utils.BSpline import cox_de_boor_basis_and_derivative
        except ImportError:
            return True  # cannot verify without PyTorch / project on path

        u_test = np.linspace(0.01, 0.99, n_test)
        Bu_np = BSplineBasis.evaluate(u_test, knots_u, du, Hu)
        Bv_np = BSplineBasis.evaluate(
            np.linspace(0.01, 0.99, n_test), knots_v, dv, Wv
        )

        ku_t = torch.tensor(knots_u, dtype=torch.float64)
        kv_t = torch.tensor(knots_v, dtype=torch.float64)
        Bu_pt, _, _ = cox_de_boor_basis_and_derivative(
            torch.tensor(u_test, dtype=torch.float64), du, ku_t
        )
        Bv_pt, _, _ = cox_de_boor_basis_and_derivative(
            torch.tensor(np.linspace(0.01, 0.99, n_test), dtype=torch.float64),
            dv, kv_t,
        )

        ok_u = np.allclose(Bu_np, Bu_pt.numpy(), atol=atol)
        ok_v = np.allclose(Bv_np, Bv_pt.numpy(), atol=atol)
        if not (ok_u and ok_v):
            diff_u = np.max(np.abs(Bu_np - Bu_pt.numpy()))
            diff_v = np.max(np.abs(Bv_np - Bv_pt.numpy()))
            warnings.warn(
                f"[BSplineFitter] Basis mismatch! max|Δu|={diff_u:.2e}, "
                f"max|Δv|={diff_v:.2e}. Fitting ↔ rendering will diverge."
            )
            return False
        return True

    # ===================================================================
    #  PRIVATE helpers
    # ===================================================================

    @staticmethod
    def _build_collocation_sparse(
        Bu: np.ndarray,
        Bv: np.ndarray,
        Hu: int,
        Wv: int,
        chunk_size: int = 8192,
        threshold: float = 1e-15,
    ) -> csr_matrix:
        """
        Chunked, vectorised construction of the sparse tensor-product
        collocation matrix A[k, i*Wv + j] = Bu[k,i] · Bv[k,j].
        """
        N = Bu.shape[0]
        n_total = Hu * Wv
        all_rows, all_cols, all_vals = [], [], []

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            outer = Bu[s:e, :, None] * Bv[s:e, None, :]   # (C, Hu, Wv)
            flat = outer.reshape(e - s, -1)                 # (C, Hu*Wv)
            nz_k, nz_ij = np.nonzero(np.abs(flat) > threshold)
            all_rows.append(nz_k + s)
            all_cols.append(nz_ij)
            all_vals.append(flat[nz_k, nz_ij])

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_vals)
        return csr_matrix((vals, (rows, cols)), shape=(N, n_total))

    @staticmethod
    def _build_regularization_matrix_full(Hu: int, Wv: int) -> csr_matrix:
        """
        Full 2-D thin-plate regularisation:

            ‖d²P/du²‖² + ‖d²P/dv²‖² + 2·‖d²P/dudv‖²

        Includes the cross-derivative term that penalises diagonal
        oscillations — critical when UV axes don't align with principal
        curvature directions.
        """
        n = Hu * Wv
        _idx = lambda i, j: i * Wv + j  # noqa: E731

        # --- d²P/du² --------------------------------------------------
        ru, cu, vu = [], [], []
        eq = 0
        for i in range(1, Hu - 1):
            for j in range(Wv):
                ru.extend([eq, eq, eq])
                cu.extend([_idx(i - 1, j), _idx(i, j), _idx(i + 1, j)])
                vu.extend([1.0, -2.0, 1.0])
                eq += 1
        D2u = csr_matrix((vu, (ru, cu)), shape=(eq, n))

        # --- d²P/dv² --------------------------------------------------
        rv, cv, vv = [], [], []
        eq = 0
        for i in range(Hu):
            for j in range(1, Wv - 1):
                rv.extend([eq, eq, eq])
                cv.extend([_idx(i, j - 1), _idx(i, j), _idx(i, j + 1)])
                vv.extend([1.0, -2.0, 1.0])
                eq += 1
        D2v = csr_matrix((vv, (rv, cv)), shape=(eq, n))

        # --- d²P/dudv  (√2 weight because we square via LᵀL) ----------
        ruv, cuv, vuv = [], [], []
        eq = 0
        w = np.sqrt(2.0)
        for i in range(Hu - 1):
            for j in range(Wv - 1):
                ruv.extend([eq, eq, eq, eq])
                cuv.extend([
                    _idx(i, j), _idx(i, j + 1),
                    _idx(i + 1, j), _idx(i + 1, j + 1),
                ])
                vuv.extend([w, -w, -w, w])
                eq += 1
        D2uv = csr_matrix((vuv, (ruv, cuv)), shape=(eq, n))

        return sparse_vstack([D2u, D2v, D2uv], format="csr")

    @staticmethod
    def _second_diff_1d(n: int) -> np.ndarray:
        """1-D second-difference matrix  (n-2, n)."""
        if n < 3:
            return np.zeros((0, n))
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        return D

    @staticmethod
    def _compute_grid_residual(
        grid_xyz: np.ndarray,
        ctrl_pts: np.ndarray,
        Bu: np.ndarray,
        Bv: np.ndarray,
    ) -> float:
        """RMS fitting error on gridded data:  ‖Bu P Bv^T − Grid‖."""
        fitted = np.zeros_like(grid_xyz)
        for dim in range(3):
            fitted[:, :, dim] = Bu @ ctrl_pts[:, :, dim] @ Bv.T
        return float(np.sqrt(np.mean(np.sum((fitted - grid_xyz) ** 2, axis=-1))))
# =============================================================================
# Adaptive Sampling Generator
# =============================================================================


class AdaptiveSamplingGenerator:
    """
    Generates adaptive UV sampling coordinates based on surface complexity.

    Analyzes the fitted NURBS control grid to identify regions requiring
    higher sampling density and produces non-uniform UV coordinates that
    concentrate samples in complex regions.
    """

    def __init__(self, config: Optional[AdaptiveSamplingConfig] = None):
        self.config = config or AdaptiveSamplingConfig()

    def generate(
        self,
        control_points: np.ndarray,
        control_colors: np.ndarray,
        knots_u: np.ndarray,
        knots_v: np.ndarray,
        degree: int = 3,
    ) -> AdaptiveSamplingResult:
        """Generate adaptive UV sampling coordinates."""
        H, W, _ = control_points.shape
        Us = max(int(self.config.sampling_resolution_factor * H), 2)
        Vs = max(int(self.config.sampling_resolution_factor * W), 2)

        complexity_map = self._compute_complexity_map(control_points, control_colors)
        density_map = self._complexity_to_density(complexity_map)
        density_samples = self._upsample_density(density_map, Us, Vs)

        intervals_u, intervals_v = self._density_to_1d_intervals(
            density_samples, knots_u, knots_v, degree
        )

        if self.config.enforce_monotonic:
            intervals_u = np.sort(intervals_u)
            intervals_v = np.sort(intervals_v)

        return AdaptiveSamplingResult(
            intervals_u=intervals_u.astype(np.float32),
            intervals_v=intervals_v.astype(np.float32),
            grid_u=None,
            grid_v=None,
            complexity_map=complexity_map.astype(np.float32),
            density_map=density_samples.astype(np.float32),
        )

    def _density_to_1d_intervals(
        self,
        density: np.ndarray,
        knots_u: np.ndarray,
        knots_v: np.ndarray,
        degree: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 1D intervals by marginalizing the 2D density."""
        u_min = knots_u[degree]
        u_max = knots_u[-(degree + 1)]
        v_min = knots_v[degree]
        v_max = knots_v[-(degree + 1)]

        marginal_u = density.sum(axis=1)
        marginal_v = density.sum(axis=0)

        intervals_u = self._inverse_cdf_sample(marginal_u, u_min, u_max)
        intervals_v = self._inverse_cdf_sample(marginal_v, v_min, v_max)

        return intervals_u, intervals_v

    def _compute_complexity_map(
        self, control_points: np.ndarray, control_colors: np.ndarray
    ) -> np.ndarray:
        """Compute per-control-point complexity combining multiple metrics."""
        H, W, _ = control_points.shape

        # 1. Discrete curvature via second differences
        curvature = np.zeros((H, W))
        if H > 2 and W > 2:
            d2u = np.zeros_like(control_points)
            d2u[1:-1] = (
                control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
            )

            d2v = np.zeros_like(control_points)
            d2v[:, 1:-1] = (
                control_points[:, 2:]
                - 2 * control_points[:, 1:-1]
                + control_points[:, :-2]
            )

            d2uv = np.zeros_like(control_points)
            d2uv[1:-1, 1:-1] = (
                control_points[2:, 2:]
                - control_points[2:, :-2]
                - control_points[:-2, 2:]
                + control_points[:-2, :-2]
            ) / 4.0

            curvature = (
                np.linalg.norm(d2u, axis=-1)
                + np.linalg.norm(d2v, axis=-1)
                + 0.5 * np.linalg.norm(d2uv, axis=-1)
            )

        # 2. Color variance in local neighborhood
        color_var = self._compute_local_variance(control_colors, kernel_size=3)

        # 3. Normal variance
        normal_var = self._compute_normal_variance_grid(control_points)

        # 4. Edge proximity
        edge_prox = self._compute_edge_proximity(H, W)

        def safe_normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-8:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        curvature = safe_normalize(curvature)
        color_var = safe_normalize(color_var)
        normal_var = safe_normalize(normal_var)
        edge_prox = safe_normalize(edge_prox)

        cfg = self.config
        complexity = (
            cfg.weight_curvature * curvature
            + cfg.weight_color_variance * color_var
            + cfg.weight_normal_variance * normal_var
            + cfg.weight_edge_proximity * edge_prox
        )

        total_weight = (
            cfg.weight_curvature
            + cfg.weight_color_variance
            + cfg.weight_normal_variance
            + cfg.weight_edge_proximity
        )
        complexity /= total_weight + 1e-8

        return complexity.astype(np.float32)

    def _compute_local_variance(
        self, data: np.ndarray, kernel_size: int = 3
    ) -> np.ndarray:
        """Compute local variance using a sliding window."""
        H, W, C = data.shape
        pad = kernel_size // 2
        padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

        variance = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                patch = padded[i : i + kernel_size, j : j + kernel_size, :]
                variance[i, j] = np.var(patch)

        return variance

    def _compute_normal_variance_grid(self, control_points: np.ndarray) -> np.ndarray:
        """Compute variance of surface normals in local neighborhood."""
        H, W, _ = control_points.shape

        du = np.zeros_like(control_points)
        dv = np.zeros_like(control_points)

        du[:-1] = control_points[1:] - control_points[:-1]
        du[-1] = du[-2]

        dv[:, :-1] = control_points[:, 1:] - control_points[:, :-1]
        dv[:, -1] = dv[:, -2]

        normals = np.cross(du, dv)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms

        return self._compute_local_variance(normals, kernel_size=3)

    def _compute_edge_proximity(self, H: int, W: int) -> np.ndarray:
        """
        Compute proximity to edges (higher at boundaries).
        Helps ensure adequate sampling at surface boundaries.
        """
        u_dist = np.minimum(
            np.arange(H)[:, None],
            np.arange(H - 1, -1, -1)[:, None],
        )
        v_dist = np.minimum(
            np.arange(W)[None, :],
            np.arange(W - 1, -1, -1)[None, :],
        )

        edge_dist = np.minimum(u_dist, v_dist).astype(float)
        max_dist = min(H, W) / 2.0
        if max_dist < 1e-8:
            return np.ones((H, W), dtype=np.float32)

        edge_prox = 1.0 - (edge_dist / max_dist)
        return np.broadcast_to(edge_prox, (H, W)).copy()

    def _complexity_to_density(self, complexity: np.ndarray) -> np.ndarray:
        """
        Convert complexity map to target sampling density.
        Higher complexity -> higher density (more samples per unit UV area).
        """
        cfg = self.config

        if cfg.smoothing_sigma > 0:
            complexity = gaussian_filter(complexity, sigma=cfg.smoothing_sigma)

        density_ratio = (
                cfg.min_density_ratio
                + complexity * (cfg.max_density_ratio - cfg.min_density_ratio)
        )
        return density_ratio.astype(np.float32)

    def _upsample_density(
            self, density_ctrl: np.ndarray, Us: int, Vs: int
    ) -> np.ndarray:
        """Upsample density map from control grid resolution to sampling resolution."""
        from scipy.ndimage import zoom

        H, W = density_ctrl.shape
        if H == 0 or W == 0:
            return np.ones((Us, Vs), dtype=np.float32)

        zoom_factors = (Us / H, Vs / W)
        density_samples = zoom(density_ctrl, zoom_factors, order=1, mode="nearest")

        # Ensure exact shape
        density_samples = density_samples[:Us, :Vs]
        return density_samples

    def _inverse_cdf_sample(
            self, density: np.ndarray, val_min: float, val_max: float
    ) -> np.ndarray:
        """
        Generate non-uniform samples using inverse CDF of density.
        Higher density -> samples closer together.
        """
        N = len(density)
        if N == 0:
            return np.array([val_min])
        if N == 1:
            return np.array([(val_min + val_max) / 2.0])

        density = density + 1e-8
        pdf = density / density.sum()

        cdf = np.cumsum(pdf)
        cdf = np.concatenate([[0], cdf])

        uniform = np.linspace(0, 1, N)

        positions = np.zeros(N)
        for i, u in enumerate(uniform):
            idx = np.searchsorted(cdf, u, side="right") - 1
            idx = np.clip(idx, 0, N - 1)

            if idx < N - 1:
                denom = cdf[idx + 1] - cdf[idx] + 1e-8
                t = (u - cdf[idx]) / denom
                positions[i] = idx + t
            else:
                positions[i] = N - 1

        samples = val_min + (positions / (N - 1)) * (val_max - val_min)
        return samples

    def _renormalize_to_domain(
            self, samples: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        """Renormalize samples to [eps, 1-eps] domain."""
        s_min, s_max = samples.min(), samples.max()
        if s_max - s_min < 1e-8:
            return np.linspace(eps, 1 - eps, samples.shape[0])[:, None] * np.ones_like(
                samples
            )

        normalized = (samples - s_min) / (s_max - s_min)
        return eps + normalized * (1 - 2 * eps)

    # =============================================================================
    # Main Interface
    # =============================================================================

def extract_component_faces(faces, component_indices):
    """Extract faces where all 3 vertices belong to the component."""
    if faces is None:
        return None
    index_set = set(component_indices.tolist())
    # Remap global vertex indices to local [0..len(component_indices))
    global_to_local = {g: l for l, g in enumerate(component_indices)}
    local_faces = []
    for f in faces:
        if f[0] in index_set and f[1] in index_set and f[2] in index_set:
            local_faces.append([global_to_local[f[0]],
                                global_to_local[f[1]],
                                global_to_local[f[2]]])
    if len(local_faces) == 0:
        return None
    return np.array(local_faces, dtype=np.int64)
class NURBSFromPointCloud:
    """Main interface for creating NURBS surfaces from point clouds."""

    def __init__(self, config: Optional[SurfaceConfig] = None):
        self.config = config or SurfaceConfig()

    def create_surfaces(
            self,
            points: Union[np.ndarray, torch.Tensor],
            colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
            mode: Optional[DecompositionMode] = None,
            cameras: Optional[List] = None,  # NEW: training cameras
            **kwargs,
    ) -> MultiSurfaceResult:
        """Create NURBS surface(s) from point cloud.

        Args:
            points: [N, 3] point positions
            colors: [N, 3] optional RGB colors
            mode: Decomposition mode override
            cameras: List of training Camera objects (from cameras.py)
            **kwargs: Config overrides
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if colors is not None and isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

        config = self._update_config(**kwargs)
        if mode is not None:
            config.decomposition_mode = mode

        # --- Extract camera info ---
        camera_infos = None
        if cameras is not None:
            camera_infos = []
            for cam in cameras:
                try:
                    camera_infos.append(CameraObject.from_camera(cam))
                except Exception as e:
                    warnings.warn(f"Failed to extract camera info: {e}")
            if not camera_infos:
                camera_infos = None

        processor = PointCloudProcessor(points, colors)

        if config.outlier_removal:
            raw_n = processor.n_points
            clean_idx, _ = processor.remove_outliers(
                config.outlier_std_ratio, config.connectivity_k
            )
            print(f"[NURBS] Removed outliers: {raw_n} -> {len(clean_idx)} points")
            points = points[clean_idx]
            colors = colors[clean_idx] if colors is not None else None
            processor = PointCloudProcessor(points, colors)

        print(f"[NURBS] Processing {processor.n_points} points")

        # --- Camera-consistent normals ---
        normals = None
        if config.use_camera_normals and camera_infos:
            print("[NURBS] Computing camera-consistent normals")
            normals = compute_camera_consistent_normals(
                points, camera_infos, k=config.normal_estimation_k
            )
            processor._normals = normals

        # --- Camera observation weights ---
        point_weights = None
        if config.use_camera_weights and camera_infos:
            print("[NURBS] Computing camera observation weights")
            if normals is None:
                normals = processor.estimate_normals()
            point_weights = compute_observation_weights(
                points, camera_infos, normals
            )

        decomposer = SceneDecomposer(processor, config)
        fitter = NURBSSurfaceFitter(config)

        # --- Build post-fitter ---
        post_fitter = None
        if config.post_fit_enabled:
            post_fit_cfg = PostFitConfig(
                num_iterations=config.post_fit_iterations,
                learning_rate=config.post_fit_lr,
                chamfer_weight=config.post_fit_chamfer_weight,
                smoothness_weight=config.post_fit_smoothness_weight,
                normal_weight=config.post_fit_normal_weight,
                num_surface_samples=config.post_fit_num_samples,
                use_camera_weights=config.use_camera_weights,
                convergence_threshold=config.post_fit_convergence_threshold,
                patience=config.post_fit_patience,
            )
            post_fitter = BSplinePostFitter(post_fit_cfg)

        if config.decomposition_mode == DecompositionMode.SINGLE:
            surfaces, labels = self._create_single_surface(
                processor, fitter,
                point_weights=point_weights,
                cameras=camera_infos,
                normals=normals,
                post_fitter=post_fitter,
            )
        elif config.decomposition_mode == DecompositionMode.BACKGROUND_OBJECT:
            surfaces, labels = self._create_bg_object_surfaces(
                processor, decomposer, fitter,
                point_weights=point_weights,
                cameras=camera_infos,
                normals=normals,
                post_fitter=post_fitter,
            )
        elif config.decomposition_mode == DecompositionMode.K_COMPONENTS:
            surfaces, labels = self._create_k_component_surfaces(
                processor, decomposer, fitter,
                point_weights=point_weights,
                cameras=camera_infos,
                normals=normals,
                post_fitter=post_fitter,
            )
        else:
            raise ValueError(
                f"Unknown decomposition mode: {config.decomposition_mode}"
            )

        return MultiSurfaceResult(
            surfaces=surfaces,
            decomposition_mode=config.decomposition_mode,
            labels=labels,
            metadata={"n_input_points": len(points), "config": config},
        )

    def _update_config(self, **kwargs) -> SurfaceConfig:
        config_dict = {}
        for field_name in self.config.__dataclass_fields__:
            config_dict[field_name] = kwargs.get(
                field_name, getattr(self.config, field_name)
            )
        return SurfaceConfig(**config_dict)

    def _create_single_surface(
            self,
            processor: PointCloudProcessor,
            fitter: NURBSSurfaceFitter,
            point_weights: Optional[np.ndarray] = None,
            cameras: Optional[List[CameraObject]] = None,
            normals: Optional[np.ndarray] = None,
            post_fitter: Optional[BSplinePostFitter] = None,
    ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
        complexity = processor.compute_local_complexity()
        surface = fitter.fit_surface(
            processor.points,
            processor.colors,
            label="main",
            complexity=complexity,
            point_weights=point_weights,
            cameras=cameras,
            normals=normals,
        )
        surface.point_indices = np.arange(processor.n_points)

        # Post-fit Chamfer refinement
        if post_fitter is not None:
            print("[NURBS] Running post-fit Chamfer refinement")
            surface = post_fitter.refine(
                surface,
                target_points=processor.points,
                target_normals=normals,
                point_weights=point_weights,
                cameras=cameras,
            )

        labels = np.zeros(processor.n_points, dtype=np.int32)
        return [surface], labels

    def _create_bg_object_surfaces(
            self,
            processor: PointCloudProcessor,
            decomposer: SceneDecomposer,
            fitter: NURBSSurfaceFitter,
            point_weights: Optional[np.ndarray] = None,
            cameras: Optional[List[CameraObject]] = None,
            normals: Optional[np.ndarray] = None,
            post_fitter: Optional[BSplinePostFitter] = None,
    ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
        bg_indices, obj_indices = decomposer.decompose_background_object()

        surfaces = []
        labels = np.zeros(processor.n_points, dtype=np.int32)

        component_resolutions = {
            "background": self.config.bg_resolution,
            "object": self.config.object_resolution,
        }
        component_scales = {
            "background": self.config.bg_resolution_scale,
            "object": self.config.object_resolution_scale,
        }
        base_res = (self.config.base_resolution, self.config.base_resolution)

        for idx_array, label_val, name in [
            (bg_indices, 0, "background"),
            (obj_indices, 1, "object"),
        ]:
            if len(idx_array) < 16:
                continue

            pts = processor.points[idx_array]
            cols = (
                processor.colors[idx_array]
                if processor.colors is not None
                else None
            )
            pw = point_weights[idx_array] if point_weights is not None else None
            nrm = normals[idx_array] if normals is not None else None

            override_res = component_resolutions.get(name)
            if override_res is None:
                scale = component_scales.get(name, 1.0)
                override_res = (
                    int(base_res[0] * scale),
                    int(base_res[1] * scale),
                )

            print(
                f"[NURBS] {name}: {len(idx_array)} points, "
                f"resolution {override_res}"
            )

            try:
                surface = fitter.fit_surface(
                    pts,
                    cols,
                    label=name,
                    override_resolution=override_res,
                    point_weights=pw,
                    cameras=cameras,
                    normals=nrm,
                )
                surface.point_indices = idx_array

                # Post-fit per component
                if post_fitter is not None:
                    print(f"[NURBS] Post-fit refinement for {name}")
                    surface = post_fitter.refine(
                        surface,
                        target_points=pts,
                        target_normals=nrm,
                        point_weights=pw,
                        cameras=cameras,
                    )

                surfaces.append(surface)
                labels[idx_array] = label_val

            except Exception as e:
                warnings.warn(f"Failed to fit {name} surface: {e}")

        if len(surfaces) == 0:
            warnings.warn(
                "Failed to create separate surfaces, falling back to single"
            )
            return self._create_single_surface(
                processor, fitter, point_weights, cameras, normals, post_fitter
            )

        return surfaces, labels

    def _create_k_component_surfaces(
            self,
            processor: PointCloudProcessor,
            decomposer: SceneDecomposer,
            fitter: NURBSSurfaceFitter,
            point_weights: Optional[np.ndarray] = None,
            cameras: Optional[List[CameraObject]] = None,
            normals: Optional[np.ndarray] = None,
            post_fitter: Optional[BSplinePostFitter] = None,
    ) -> Tuple[List[NURBSSurfaceData], np.ndarray]:
        component_indices = decomposer.decompose_k_components()
        complexity = processor.compute_local_complexity()

        surfaces = []
        labels = np.full(processor.n_points, -1, dtype=np.int32)

        for i, indices in enumerate(component_indices):
            if len(indices) < 16:
                warnings.warn(
                    f"Component {i} has too few points ({len(indices)}), skipping"
                )
                continue

            pts = processor.points[indices]
            cols = (
                processor.colors[indices]
                if processor.colors is not None
                else None
            )
            comp_complexity = complexity[indices]
            pw = point_weights[indices] if point_weights is not None else None
            nrm = normals[indices] if normals is not None else None

            try:
                surface = fitter.fit_surface(
                    pts,
                    cols,
                    label=f"component_{i}",
                    complexity=comp_complexity,
                    point_weights=pw,
                    cameras=cameras,
                    normals=nrm,
                )
                surface.point_indices = indices

                if post_fitter is not None:
                    print(f"[NURBS] Post-fit refinement for component_{i}")
                    surface = post_fitter.refine(
                        surface,
                        target_points=pts,
                        target_normals=nrm,
                        point_weights=pw,
                        cameras=cameras,
                    )

                surfaces.append(surface)
                labels[indices] = len(surfaces) - 1

            except Exception as e:
                warnings.warn(f"Failed to fit surface for component {i}: {e}")

        if len(surfaces) == 0:
            warnings.warn(
                "Failed to create any component surfaces, falling back to single"
            )
            return self._create_single_surface(
                processor, fitter, point_weights, cameras, normals, post_fitter
            )

        return surfaces, labels
# =============================================================================
# Conversion Utilities
# =============================================================================

def nurbs_to_geomdl(surface_data: NURBSSurfaceData) -> BSpline.Surface:
    """Convert NURBSSurfaceData to geomdl BSpline.Surface."""
    surf = BSpline.Surface()
    surf.degree_u = surface_data.degree_u
    surf.degree_v = surface_data.degree_v

    H, W, _ = surface_data.control_points.shape
    ctrlpts = surface_data.control_points.reshape(-1, 3).tolist()

    surf.set_ctrlpts(ctrlpts, H, W)
    surf.knotvector_u = surface_data.knots_u.tolist()
    surf.knotvector_v = surface_data.knots_v.tolist()

    return surf

def surfaces_to_torch(
        result: MultiSurfaceResult, device: str = "cuda"
) -> Dict[str, Any]:
    """Convert MultiSurfaceResult to torch tensors for SplineModel initialization."""
    surfaces = result.surfaces

    cp_list = [
        torch.tensor(s.control_points, dtype=torch.float32, device=device)
        for s in surfaces
    ]
    cc_list = [
        torch.tensor(s.control_colors, dtype=torch.float32, device=device)
        for s in surfaces
    ]
    ku_list = [
        torch.tensor(s.knots_u, dtype=torch.float32, device=device) for s in surfaces
    ]
    kv_list = [
        torch.tensor(s.knots_v, dtype=torch.float32, device=device) for s in surfaces
    ]

    adaptive_u_list = []
    adaptive_v_list = []
    complexity_list = []

    for s in surfaces:
        if s.sampling_u_1D is not None:
            adaptive_u_list.append(
                torch.tensor(s.sampling_u_1D, dtype=torch.float32, device=device)
            )
            adaptive_v_list.append(
                torch.tensor(s.sampling_v_1D, dtype=torch.float32, device=device)
            )
        else:
            adaptive_u_list.append(None)
            adaptive_v_list.append(None)

        if s.complexity_map is not None:
            complexity_list.append(
                torch.tensor(s.complexity_map, dtype=torch.float32, device=device)
            )
        else:
            complexity_list.append(None)

    return {
        "control_points": cp_list,
        "control_colors": cc_list,
        "knots_u": ku_list,
        "knots_v": kv_list,
        "labels": torch.tensor(result.labels, dtype=torch.long, device=device),
        "surface_labels": [s.label for s in surfaces],
        "num_surfaces": len(surfaces),
        "adaptive_samples_u": adaptive_u_list,
        "adaptive_samples_v": adaptive_v_list,
        "complexity_maps": complexity_list,
    }


def create_nurbs_from_pointcloud(
        points: Union[np.ndarray, torch.Tensor],
        colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        resolution: Tuple[int, int] = (64, 64),
        mode: DecompositionMode = DecompositionMode.SINGLE,
        smoothing: float = 0.05,
        generate_adaptive_samples: bool = True,
        sampling_resolution_factor: float = 1.0,
        bg_resolution: Optional[Tuple[int, int]] = None,
        object_resolution: Optional[Tuple[int, int]] = None,
        bg_resolution_scale: float = 0.5,
        object_resolution_scale: float = 2.0,
        parameterization: str = "spherical",
        faces=None,
        use_least_squares=True,
        cameras: Optional[List] = None,
        **kwargs,
) -> MultiSurfaceResult:
    """Convenience function to create NURBS surface(s) from point cloud.

    Args:
        cameras: List of training Camera objects for view-aware fitting
        post_fit: Whether to run Chamfer post-optimization
        post_fit_iterations: Number of optimization iterations
        post_fit_smoothness_weight: Regularization strength
    """
    config = SurfaceConfig(
        resolution_u=resolution[0],
        resolution_v=resolution[1],

        smoothing=smoothing,
        decomposition_mode=mode,
        sampling_resolution_factor=sampling_resolution_factor,
        bg_resolution=bg_resolution,
        object_resolution=object_resolution,
        bg_resolution_scale=bg_resolution_scale,
        object_resolution_scale=object_resolution_scale,
        parameterization=parameterization,
        # Camera-aware fitting
        use_camera_weights=(cameras is not None),
        # use_camera_normals=(cameras is not None),
        # Post-fit
        # post_fit_enabled=post_fit,
        # post_fit_iterations=post_fit_iterations,
        # post_fit_smoothness_weight=post_fit_smoothness_weight,
        **{k: v for k, v in kwargs.items() if hasattr(SurfaceConfig, k)},
    )


    creator = NURBSFromPointCloud(config)
    return creator.create_surfaces(
        points, colors, mode,
        cameras=cameras,
        faces=faces,
        use_least_squares=use_least_squares,
        generate_adaptive_samples=generate_adaptive_samples,
    )


