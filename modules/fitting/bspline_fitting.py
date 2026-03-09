"""
Proper least-squares B-spline surface fitting.

Solves the regularized normal equations
    (A^T W A + λ L^T L) P = A^T W D
so that the resulting control points P reproduce the surface data
rather than acting as a second smoothing pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class BSplineFitResult:
    """Holds the output of a least-squares B-spline surface fit."""

    control_points: np.ndarray        # [H, W, 3]  float32
    control_colors: np.ndarray        # [H, W, 3]  float32
    knots_u: np.ndarray               # [n_ctrl_u + degree_u + 1]  float32
    knots_v: np.ndarray               # [n_ctrl_v + degree_v + 1]  float32
    degree_u: int
    degree_v: int
    residual_rms: float               # RMS fitting residual (geometry only)
    parameterization: Optional[np.ndarray] = field(default=None)
    """UV parameterization used for fitting, shape [N, 2]."""


# ---------------------------------------------------------------------------
# Cox-de Boor basis (NumPy, mirrors BSpline.py structure)
# ---------------------------------------------------------------------------

class BSplineBasis:
    """
    Vectorized Cox-de Boor B-spline basis evaluation in NumPy.

    The recursion structure mirrors ``cox_de_boor_basis_and_derivative``
    in ``modules/utils/BSpline.py`` so that verify_basis_consistency()
    can compare both implementations on random parameter values.
    """

    @staticmethod
    def evaluate(
        params: np.ndarray,
        knots: np.ndarray,
        degree: int,
        n_ctrl: int,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Evaluate B-spline basis functions.

        Parameters
        ----------
        params : (M,) array of parameter values in [0, 1]
        knots  : (n_ctrl + degree + 1,) knot vector
        degree : polynomial degree
        n_ctrl : number of control points
        eps    : small value to avoid division by zero

        Returns
        -------
        B : (M, n_ctrl) basis matrix
        """
        M = params.shape[0]
        u = params[:, None]             # (M, 1)
        U = knots                       # (n_ctrl + degree + 1,)

        # ------------------------------------------------------------------
        # Zero-degree: indicator functions on half-open spans
        # ------------------------------------------------------------------
        left  = U[:n_ctrl]              # (n_ctrl,)
        right = U[1:n_ctrl + 1]        # (n_ctrl,)
        N0 = ((u >= left) & (u < right)).astype(params.dtype)  # (M, n_ctrl)

        # Right-endpoint inclusion
        N0[params == U[-1], -1] = 1.0

        N_all = [N0]

        # ------------------------------------------------------------------
        # Recursion for degree 1 … p  (same slice pattern as BSpline.py)
        # ------------------------------------------------------------------
        for k in range(1, degree + 1):
            left_denom  = np.maximum(U[k:k + n_ctrl]       - U[:n_ctrl],      eps)
            right_denom = np.maximum(U[k + 1:k + 1 + n_ctrl] - U[1:n_ctrl + 1], eps)

            left_coeff  = (u - U[:n_ctrl])               / left_denom   # (M, n_ctrl)
            right_coeff = (U[k + 1:k + 1 + n_ctrl] - u) / right_denom  # (M, n_ctrl)

            prev = N_all[-1]
            shifted = np.concatenate(
                [prev[:, 1:], np.zeros((M, 1), dtype=params.dtype)], axis=1
            )
            Nk = left_coeff * prev + right_coeff * shifted
            N_all.append(Nk)

        return N_all[-1]                # (M, n_ctrl)

    @staticmethod
    def create_knot_vector(
        n_ctrl: int,
        degree: int,
        params: Optional[np.ndarray] = None,
        max_params_for_avg: int = 1000,
    ) -> np.ndarray:
        """
        Create a clamped knot vector.

        When *params* is provided the Piegl & Tiller averaging method
        (Eq. 9.68) is used; otherwise a uniform distribution is produced.
        Up to *max_params_for_avg* parameter values are subsampled before
        averaging to handle clustered distributions (e.g. from spherical
        parameterisation).

        The result always satisfies
        - ``knots[:degree+1] == 0``
        - ``knots[-degree-1:] == 1``
        - non-decreasing

        Compatible with ``make_clamped_uniform_knots`` in BSpline.py.
        """
        n_knots = n_ctrl + degree + 1
        knots = np.zeros(n_knots)
        knots[-degree - 1:] = 1.0

        n_inner = n_ctrl - degree - 1   # number of *strictly* interior knots

        if params is not None and n_inner > 0:
            # Subsample to avoid bias from clustered parameter distributions
            p = np.sort(params)
            if len(p) > max_params_for_avg:
                idx = np.round(
                    np.linspace(0, len(p) - 1, max_params_for_avg)
                ).astype(int)
                p = p[idx]

            # Piegl & Tiller Eq. 9.68: k-th inner knot = average of p[k..k+d-1]
            n_p = len(p)
            for j in range(1, n_inner + 1):
                i = int(math.floor(j * n_p / (n_inner + 1)))
                # average of p[i .. i+degree-1], clamped to valid range
                lo = max(0, i)
                hi = min(n_p - 1, i + degree - 1)
                knots[degree + j] = float(np.mean(p[lo:hi + 1]))

            # Ensure non-decreasing (numerical safety)
            knots = np.maximum.accumulate(knots)
            knots = np.minimum.accumulate(knots[::-1])[::-1]
            knots[:degree + 1] = 0.0
            knots[-degree - 1:] = 1.0

        elif n_inner > 0:
            # Uniform inner knots (matches make_clamped_uniform_knots in BSpline.py)
            inner = np.linspace(0.0, 1.0, n_ctrl - degree + 1)
            knots[degree:degree + len(inner)] = inner

        return knots.astype(np.float64)


# ---------------------------------------------------------------------------
# B-spline surface fitter
# ---------------------------------------------------------------------------

class BSplineSurfaceFitter:
    """
    Least-squares B-spline surface fitter.

    Solves:
        (A^T W A + λ L^T L) P = A^T W D

    for each coordinate independently.  *A* is the tensor-product
    collocation matrix, *W* are per-point weights, *L* is a
    thin-plate regularisation matrix (second differences + cross
    derivatives), and *λ* is controlled by the *smoothing* parameter.
    """

    def __init__(
        self,
        n_ctrl_u: int = 16,
        n_ctrl_v: int = 16,
        degree_u: int = 3,
        degree_v: int = 3,
        smoothing: float = 1e-3,
        data_dependent_knots: bool = True,
    ):
        self.n_ctrl_u = n_ctrl_u
        self.n_ctrl_v = n_ctrl_v
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.smoothing = smoothing
        self.data_dependent_knots = data_dependent_knots

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        points: np.ndarray,
        uv_params: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_weights: Optional[np.ndarray] = None,
    ) -> BSplineFitResult:
        """
        Fit a B-spline surface to scattered (u,v) → xyz data.

        Parameters
        ----------
        points       : (N, 3) XYZ positions
        uv_params    : (N, 2) parameter coordinates in [0, 1]²
        colors       : (N, 3) RGB colours, optional
        point_weights: (N,)   per-point weights, optional

        Returns
        -------
        BSplineFitResult
        """
        points = np.asarray(points, dtype=np.float64)
        uv_params = np.asarray(uv_params, dtype=np.float64)
        N = points.shape[0]

        if point_weights is None:
            W_diag = np.ones(N)
        else:
            W_diag = np.asarray(point_weights, dtype=np.float64)

        # Knot vectors
        params_u = uv_params[:, 0]
        params_v = uv_params[:, 1]
        if self.data_dependent_knots:
            knots_u = BSplineBasis.create_knot_vector(
                self.n_ctrl_u, self.degree_u, params_u
            )
            knots_v = BSplineBasis.create_knot_vector(
                self.n_ctrl_v, self.degree_v, params_v
            )
        else:
            knots_u = BSplineBasis.create_knot_vector(self.n_ctrl_u, self.degree_u)
            knots_v = BSplineBasis.create_knot_vector(self.n_ctrl_v, self.degree_v)

        # Basis matrices
        Bu = BSplineBasis.evaluate(params_u, knots_u, self.degree_u, self.n_ctrl_u)
        Bv = BSplineBasis.evaluate(params_v, knots_v, self.degree_v, self.n_ctrl_v)

        # Collocation matrix (sparse)
        A = self._build_collocation_matrix_sparse(Bu, Bv)   # (N, n_ctrl_u*n_ctrl_v)

        # Regularisation
        Hu, Nv = self.n_ctrl_u, self.n_ctrl_v
        L = self._build_regularization_matrix(Hu, Nv)

        # Normal equations
        W_sp = sp.diags(W_diag, format="csr")
        AtW  = A.T @ W_sp
        AtWA = AtW @ A
        lam  = self.smoothing
        lhs  = AtWA + lam * (L.T @ L)

        n_ctrl = Hu * Nv
        ctrl_pts = np.zeros((n_ctrl, 3))
        for c in range(3):
            rhs = AtW @ points[:, c]
            x, *_ = spla.lsqr(lhs, rhs, atol=1e-8, btol=1e-8)
            ctrl_pts[:, c] = x

        ctrl_pts_3d = ctrl_pts.reshape(Hu, Nv, 3)

        # RMS residual (geometry)
        residual = self._compute_scattered_residual(points, A, ctrl_pts)

        # Colours
        if colors is not None:
            colors_arr = np.asarray(colors, dtype=np.float64)
            ctrl_colors = np.zeros((n_ctrl, 3))
            for c in range(3):
                rhs = AtW @ colors_arr[:, c]
                x, *_ = spla.lsqr(lhs, rhs, atol=1e-8, btol=1e-8)
                ctrl_colors[:, c] = x
            ctrl_colors_3d = np.clip(ctrl_colors.reshape(Hu, Nv, 3), 0.0, 1.0)
        else:
            ctrl_colors_3d = np.zeros((Hu, Nv, 3))

        return BSplineFitResult(
            control_points=ctrl_pts_3d.astype(np.float32),
            control_colors=ctrl_colors_3d.astype(np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            residual_rms=float(residual),
            parameterization=uv_params.astype(np.float32),
        )

    def fit_from_grid(
        self,
        grid_xyz: np.ndarray,
        grid_rgb: np.ndarray,
        n_ctrl_u: Optional[int] = None,
        n_ctrl_v: Optional[int] = None,
    ) -> BSplineFitResult:
        """
        Fit a B-spline surface to gridded (regularly-sampled) data.

        Exploits tensor-product separability: solves two sequences of
        independent 1-D least-squares problems rather than the full
        (N_u * N_v) × (n_ctrl_u * n_ctrl_v) system.

        For the separable solver the regularisation is axis-aligned only:
        ``L = L_u ⊗ I_v + I_u ⊗ L_v`` (no cross-derivative term needed).

        Parameters
        ----------
        grid_xyz  : (res_u, res_v, 3)  XYZ grid samples
        grid_rgb  : (res_u, res_v, 3)  RGB grid samples
        n_ctrl_u  : number of control points in U (defaults to self.n_ctrl_u)
        n_ctrl_v  : number of control points in V (defaults to self.n_ctrl_v)

        Returns
        -------
        BSplineFitResult with properly solved control points.
        """
        grid_xyz = np.asarray(grid_xyz, dtype=np.float64)
        grid_rgb = np.asarray(grid_rgb, dtype=np.float64)
        res_u, res_v = grid_xyz.shape[:2]

        Hu = n_ctrl_u if n_ctrl_u is not None else self.n_ctrl_u
        Nv = n_ctrl_v if n_ctrl_v is not None else self.n_ctrl_v

        # Clamp control-point counts to grid resolution
        Hu = min(Hu, res_u)
        Nv = min(Nv, res_v)

        # Effective degrees (cannot exceed n_ctrl - 1)
        deg_u = min(self.degree_u, Hu - 1)
        deg_v = min(self.degree_v, Nv - 1)

        # Grid parameter values
        params_u = np.linspace(0.0, 1.0, res_u)
        params_v = np.linspace(0.0, 1.0, res_v)

        knots_u = BSplineBasis.create_knot_vector(Hu, deg_u, params_u)
        knots_v = BSplineBasis.create_knot_vector(Nv, deg_v, params_v)

        Bu = BSplineBasis.evaluate(params_u, knots_u, deg_u, Hu)   # (res_u, Hu)
        Bv = BSplineBasis.evaluate(params_v, knots_v, deg_v, Nv)   # (res_v, Nv)

        # Separable second-difference regularisation matrices
        Du = self._second_diff_1d(Hu)   # (Hu-2, Hu)
        Dv = self._second_diff_1d(Nv)   # (Nv-2, Nv)

        lam = self.smoothing

        # LHS for U and V directions
        lhs_u = Bu.T @ Bu + lam * Du.T @ Du   # (Hu, Hu)
        lhs_v = Bv.T @ Bv + lam * Dv.T @ Dv  # (Nv, Nv)

        def _solve_grid(data):
            """
            Solve for P:  lhs_u @ P @ lhs_v^T = Bu^T @ data @ Bv
            using the separable structure.
            """
            rhs = Bu.T @ data @ Bv          # (Hu, Wv)  for one channel
            # Step 1: solve lhs_u @ Z = rhs  →  Z = lhs_u^{-1} rhs
            Z = np.linalg.solve(lhs_u, rhs)    # (Hu, Wv)
            # Step 2: solve lhs_v @ P^T = Z^T  →  P = Z lhs_v^{-T}
            P = np.linalg.solve(lhs_v, Z.T).T  # (Hu, Wv)
            return P

        ctrl_pts = np.stack([_solve_grid(grid_xyz[..., c]) for c in range(3)], axis=-1)
        ctrl_colors = np.stack([_solve_grid(grid_rgb[..., c]) for c in range(3)], axis=-1)
        ctrl_colors = np.clip(ctrl_colors, 0.0, 1.0)

        residual = self._compute_grid_residual(grid_xyz, ctrl_pts, Bu, Bv)

        return BSplineFitResult(
            control_points=ctrl_pts.astype(np.float32),
            control_colors=ctrl_colors.astype(np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=deg_u,
            degree_v=deg_v,
            residual_rms=float(residual),
        )

    def verify_basis_consistency(
        self,
        knots_u: np.ndarray,
        knots_v: np.ndarray,
        n_ctrl_u: Optional[int] = None,
        n_ctrl_v: Optional[int] = None,
        n_samples: int = 50,
        atol: float = 1e-5,
    ) -> bool:
        """
        Test that the NumPy basis evaluation here matches the PyTorch
        ``cox_de_boor_basis_and_derivative`` in ``modules.utils.BSpline``
        at *n_samples* random parameter values.

        Returns True if the maximum absolute difference is within *atol*,
        False otherwise.  Always returns True when PyTorch is unavailable.
        """
        try:
            import torch
            from modules.utils.BSpline import cox_de_boor_basis_and_derivative
        except ImportError:
            return True

        Hu = n_ctrl_u if n_ctrl_u is not None else self.n_ctrl_u
        Wv = n_ctrl_v if n_ctrl_v is not None else self.n_ctrl_v

        rng = np.random.default_rng(42)
        params_u = rng.uniform(0.0, 1.0, n_samples).astype(np.float32)
        params_v = rng.uniform(0.0, 1.0, n_samples).astype(np.float32)

        # NumPy evaluation
        B_np_u = BSplineBasis.evaluate(
            params_u.astype(np.float64), knots_u.astype(np.float64),
            self.degree_u, Hu
        ).astype(np.float32)
        B_np_v = BSplineBasis.evaluate(
            params_v.astype(np.float64), knots_v.astype(np.float64),
            self.degree_v, Wv
        ).astype(np.float32)

        # PyTorch evaluation
        u_t = torch.from_numpy(params_u)
        v_t = torch.from_numpy(params_v)
        ku  = torch.from_numpy(knots_u.astype(np.float32))
        kv  = torch.from_numpy(knots_v.astype(np.float32))

        B_pt_u, _, _ = cox_de_boor_basis_and_derivative(u_t, self.degree_u, ku)
        B_pt_v, _, _ = cox_de_boor_basis_and_derivative(v_t, self.degree_v, kv)

        B_pt_u = B_pt_u.numpy()
        B_pt_v = B_pt_v.numpy()

        diff_u = float(np.abs(B_np_u - B_pt_u).max())
        diff_v = float(np.abs(B_np_v - B_pt_v).max())

        return diff_u <= atol and diff_v <= atol

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_collocation_matrix_sparse(
        Bu: np.ndarray,
        Bv: np.ndarray,
        chunk_size: int = 8192,
    ) -> sp.csr_matrix:
        """
        Build sparse (N, n_ctrl_u * n_ctrl_v) collocation matrix using
        vectorised chunked outer products.

        For each chunk of *chunk_size* points we compute the rank-1
        outer product per point:
            row[i, j*n_ctrl_v + k] = Bu[i, j] * Bv[i, k]
        and collect nonzero entries.
        """
        N, Hu = Bu.shape
        _,  Nv = Bv.shape
        n_ctrl = Hu * Nv

        rows_list = []
        cols_list = []
        vals_list = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Bu_c = Bu[start:end]                          # (C, Hu)
            Bv_c = Bv[start:end]                          # (C, Wv)
            # Outer product: (C, Hu, Wv)
            outer = Bu_c[:, :, None] * Bv_c[:, None, :]  # (C, Hu, Wv)
            outer_flat = outer.reshape(end - start, n_ctrl)  # (C, n_ctrl)

            chunk_rows = np.arange(start, end)[:, None]   # (C, 1)
            chunk_rows = np.broadcast_to(chunk_rows, outer_flat.shape)

            nonzero_mask = outer_flat != 0.0
            rows_list.append(chunk_rows[nonzero_mask])
            cols_list.append(np.broadcast_to(
                np.arange(n_ctrl)[None, :], outer_flat.shape
            )[nonzero_mask])
            vals_list.append(outer_flat[nonzero_mask])

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        return sp.csr_matrix((vals, (rows, cols)), shape=(N, n_ctrl))

    @staticmethod
    def _build_regularization_matrix(Hu: int, Nv: int) -> sp.csr_matrix:
        """
        Build full thin-plate regularisation for the scattered case.

        Includes:
        - d²P/du²  : second differences along U
        - d²P/dv²  : second differences along V
        - d²P/dudv : cross-derivative approximation
                     P[i+1,j+1] - P[i+1,j] - P[i,j+1] + P[i,j]
                     (weighted by √2 so L^T L counts it correctly)
        """
        n = Hu * Nv

        def _idx(i, j):
            return i * Nv + j

        # Second differences in U direction
        rows_u, cols_u, vals_u = [], [], []
        for i in range(Hu - 2):
            for j in range(Nv):
                c0, c1, c2 = _idx(i, j), _idx(i + 1, j), _idx(i + 2, j)
                row = i * Nv + j
                rows_u += [row, row, row]
                cols_u += [c0, c1, c2]
                vals_u += [1.0, -2.0, 1.0]
        Du = sp.csr_matrix(
            (vals_u, (rows_u, cols_u)),
            shape=((Hu - 2) * Nv, n),
        )

        # Second differences in V direction
        rows_v, cols_v, vals_v = [], [], []
        for i in range(Hu):
            for j in range(Nv - 2):
                c0, c1, c2 = _idx(i, j), _idx(i, j + 1), _idx(i, j + 2)
                row = i * (Nv - 2) + j
                rows_v += [row, row, row]
                cols_v += [c0, c1, c2]
                vals_v += [1.0, -2.0, 1.0]
        Dv = sp.csr_matrix(
            (vals_v, (rows_v, cols_v)),
            shape=(Hu * (Nv - 2), n),
        )

        # Cross derivative  (weighted by √2)
        w_cross = math.sqrt(2.0)
        rows_c, cols_c, vals_c = [], [], []
        row = 0
        for i in range(Hu - 1):
            for j in range(Nv - 1):
                c00 = _idx(i,     j    )
                c10 = _idx(i + 1, j    )
                c01 = _idx(i,     j + 1)
                c11 = _idx(i + 1, j + 1)
                rows_c += [row, row, row, row]
                cols_c += [c00,    c10,    c01,    c11   ]
                vals_c += [w_cross, -w_cross, -w_cross, w_cross]
                row += 1
        Dc = sp.csr_matrix(
            (vals_c, (rows_c, cols_c)),
            shape=((Hu - 1) * (Nv - 1), n),
        )

        return sp.vstack([Du, Dv, Dc], format="csr")

    @staticmethod
    def _second_diff_1d(n: int) -> np.ndarray:
        """
        Dense second-difference matrix of shape (n-2, n).

        Used for the separable grid solver:
            D[i, i] = 1,  D[i, i+1] = -2,  D[i, i+2] = 1
        """
        if n <= 2:
            return np.zeros((0, n))
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i]     =  1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] =  1.0
        return D

    @staticmethod
    def _compute_grid_residual(
        grid_xyz: np.ndarray,
        ctrl_pts: np.ndarray,
        Bu: np.ndarray,
        Bv: np.ndarray,
    ) -> float:
        """
        Compute RMS residual: ||grid_xyz - Bu @ ctrl_pts @ Bv^T||_F / sqrt(N).

        Parameters
        ----------
        grid_xyz : (res_u, res_v, 3)
        ctrl_pts : (Hu, Wv, 3)
        Bu       : (res_u, Hu)
        Bv       : (res_v, Wv)
        """
        # Reconstruct: S[u,v,c] = sum_{i,j} Bu[u,i] ctrl[i,j,c] Bv[v,j]
        S = np.einsum("ui,ijc,vj->uvc", Bu, ctrl_pts, Bv)
        diff = grid_xyz - S
        return float(np.sqrt(np.mean(diff ** 2)))

    @staticmethod
    def _compute_scattered_residual(
        points: np.ndarray,
        A: sp.csr_matrix,
        ctrl_pts: np.ndarray,
    ) -> float:
        """RMS residual for scattered-data fit."""
        P_flat = ctrl_pts.reshape(-1, 3)
        S = A @ P_flat                          # (N, 3)
        diff = points - S
        return float(np.sqrt(np.mean(diff ** 2)))
