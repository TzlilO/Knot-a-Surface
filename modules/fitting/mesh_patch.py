"""
Rhino Patch-equivalent: Mesh → B-Spline Surface via
harmonic parameterization + least-squares approximation.
"""
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from typing import Tuple, Optional
from dataclasses import dataclass


# ─── 1. Cotangent Laplacian (mesh-aware parameterization) ───────────────────

def _cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """
    Build the cotangent Laplacian matrix for a triangle mesh.
    L[i,j] = -cot(alpha_ij) - cot(beta_ij)  (opposite angles)
    L[i,i] = -sum_j L[i,j]
    """
    n = len(vertices)
    rows, cols, vals = [], [], []

    for tri in faces:
        for local in range(3):
            i, j, k = tri[local], tri[(local + 1) % 3], tri[(local + 2) % 3]
            # Angle at vertex k, opposite edge (i, j)
            e_ki = vertices[i] - vertices[k]
            e_kj = vertices[j] - vertices[k]
            cos_a = np.dot(e_ki, e_kj) / (np.linalg.norm(e_ki) * np.linalg.norm(e_kj) + 1e-12)
            cos_a = np.clip(cos_a, -0.999, 0.999)
            cot_a = cos_a / np.sqrt(1 - cos_a ** 2 + 1e-12)
            w = 0.5 * cot_a

            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])

    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    diag = np.array(-L.sum(axis=1)).flatten()
    L.setdiag(diag)
    return L


def _find_boundary(faces: np.ndarray) -> np.ndarray:
    """Find ordered boundary loop of an open mesh."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    edge_to_faces = defaultdict(list)

    for fi, tri in enumerate(faces):
        for local in range(3):
            e = tuple(sorted([tri[local], tri[(local + 1) % 3]]))
            edge_count[e] += 1
            edge_to_faces[e].append(fi)

    boundary_edges = {e for e, c in edge_count.items() if c == 1}

    if not boundary_edges:
        raise ValueError("Mesh has no boundary — closed surface. Use spherical param.")

    # Order boundary vertices
    adj = defaultdict(set)
    for (a, b) in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    start = next(iter(adj))
    loop = [start]
    visited = {start}
    current = start
    while True:
        neighbors = adj[current] - visited
        if not neighbors:
            break
        nxt = neighbors.pop()
        loop.append(nxt)
        visited.add(nxt)
        current = nxt

    return np.array(loop)


def harmonic_parameterization(
        vertices: np.ndarray,
        faces: np.ndarray,
) -> np.ndarray:
    """
    Tutte/harmonic parameterization: maps mesh to [0,1]^2.
    Boundary → unit circle, interior → Laplace solve.

    Returns: uv [N, 2] in [0, 1]
    """
    n = len(vertices)
    L = _cotangent_weights(vertices, faces)
    boundary = _find_boundary(faces)
    interior = np.setdiff1d(np.arange(n), boundary)

    # Map boundary to convex polygon (square for better aspect ratio)
    nb = len(boundary)
    t = np.linspace(0, 2 * np.pi, nb, endpoint=False)
    # Map to [0,1] square boundary (better than circle for tensor-product)
    boundary_uv = np.stack([0.5 + 0.49 * np.cos(t), 0.5 + 0.49 * np.sin(t)], axis=1)

    # Solve L_ii * uv_interior = -L_ib * uv_boundary
    L_ii = L[np.ix_(interior, interior)]
    L_ib = L[np.ix_(interior, boundary)]

    uv = np.zeros((n, 2))
    uv[boundary] = boundary_uv

    for dim in range(2):
        rhs = -L_ib @ boundary_uv[:, dim]
        uv[interior, dim] = spsolve(L_ii, rhs)

    # Normalize to [0.01, 0.99]
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    uv = 0.01 + 0.98 * (uv - uv_min) / (uv_max - uv_min + 1e-10)
    return uv


# ─── 2. B-Spline basis evaluation ──────────────────────────────────────────

def _bspline_basis_1d(t: np.ndarray, knots: np.ndarray, degree: int, n_ctrl: int) -> np.ndarray:
    """
    Evaluate B-spline basis functions for all t values.
    Returns: [len(t), n_ctrl] matrix.
    """
    N = np.zeros((len(t), n_ctrl))
    for i in range(n_ctrl):
        N[:, i] = _cox_de_boor(t, knots, i, degree)
    return N


def _cox_de_boor(t: np.ndarray, knots: np.ndarray, i: int, p: int) -> np.ndarray:
    """Evaluate single basis function N_{i,p}(t) for array of t."""
    if p == 0:
        return ((t >= knots[i]) & (t < knots[i + 1])).astype(float)

    result = np.zeros_like(t, dtype=float)

    d1 = knots[i + p] - knots[i]
    if abs(d1) > 1e-12:
        result += (t - knots[i]) / d1 * _cox_de_boor(t, knots, i, p - 1)

    d2 = knots[i + p + 1] - knots[i + 1]
    if abs(d2) > 1e-12:
        result += (knots[i + p + 1] - t) / d2 * _cox_de_boor(t, knots, i + 1, p - 1)

    return result


def _clamped_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector."""
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)
    knots[:degree + 1] = 0.0
    knots[-degree - 1:] = 1.0
    n_internal = n_knots - 2 * (degree + 1)
    if n_internal > 0:
        knots[degree + 1:degree + 1 + n_internal] = np.linspace(
            0, 1, n_internal + 2)[1:-1]
    return knots


# ─── 3. Least-Squares B-Spline Surface Approximation ───────────────────────

def least_squares_bspline_surface(
        points: np.ndarray,  # [N, 3]
        uv: np.ndarray,  # [N, 2]
        n_ctrl_u: int,
        n_ctrl_v: int,
        degree: int = 3,
        smoothing: float = 0.001,
        colors: Optional[np.ndarray] = None,  # [N, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Solve for B-spline control points that best approximate scattered data.

    This is the core of Rhino's Patch command.

    Returns:
        ctrl_pts: [n_ctrl_u, n_ctrl_v, 3]
        ctrl_colors: [n_ctrl_u, n_ctrl_v, 3] or None
        knots_u: [n_ctrl_u + degree + 1]
        knots_v: [n_ctrl_v + degree + 1]
    """
    N = len(points)
    knots_u = _clamped_uniform_knots(n_ctrl_u, degree)
    knots_v = _clamped_uniform_knots(n_ctrl_v, degree)

    # Clamp rightmost parameter to avoid basis function issues
    u = np.clip(uv[:, 0], knots_u[degree], knots_u[-degree - 1] - 1e-10)
    v = np.clip(uv[:, 1], knots_v[degree], knots_v[-degree - 1] - 1e-10)

    # Evaluate 1D basis functions
    Bu = _bspline_basis_1d(u, knots_u, degree, n_ctrl_u)  # [N, Hu]
    Bv = _bspline_basis_1d(v, knots_v, degree, n_ctrl_v)  # [N, Hv]

    # Build sparse approximation matrix A: [N, Hu*Hv]
    # A[k, i*Hv + j] = Bu[k, i] * Bv[k, j]
    M = n_ctrl_u * n_ctrl_v
    row_idx, col_idx, data = [], [], []
    for k in range(N):
        for i in range(n_ctrl_u):
            if abs(Bu[k, i]) < 1e-15:
                continue
            for j in range(n_ctrl_v):
                if abs(Bv[k, j]) < 1e-15:
                    continue
                row_idx.append(k)
                col_idx.append(i * n_ctrl_v + j)
                data.append(Bu[k, i] * Bv[k, j])

    A = csr_matrix((data, (row_idx, col_idx)), shape=(N, M))

    # Regularization: discrete Laplacian on control grid
    # Penalizes second differences → thin-plate-like fairness
    Lu = _control_grid_laplacian_1d(n_ctrl_u)
    Lv = _control_grid_laplacian_1d(n_ctrl_v)
    Iu = eye(n_ctrl_u)
    Iv = eye(n_ctrl_v)
    R = kron(Lu, Iv) + kron(Iu, Lv)  # 2D Laplacian on control grid

    # Normal equations: (A^T A + λ R^T R) P = A^T Q
    ATA = A.T @ A
    lhs = ATA + smoothing * (R.T @ R)

    # Solve for XYZ
    ctrl_flat = np.zeros((M, 3))
    for dim in range(3):
        rhs = A.T @ points[:, dim]
        ctrl_flat[:, dim] = spsolve(lhs, rhs)

    ctrl_pts = ctrl_flat.reshape(n_ctrl_u, n_ctrl_v, 3)

    # Solve for colors if provided
    ctrl_colors = None
    if colors is not None:
        ctrl_colors = np.zeros((M, 3))
        for dim in range(3):
            rhs = A.T @ colors[:, dim]
            ctrl_colors[:, dim] = spsolve(lhs, rhs)
        ctrl_colors = np.clip(ctrl_colors.reshape(n_ctrl_u, n_ctrl_v, 3), 0, 1)

    return ctrl_pts, ctrl_colors, knots_u, knots_v


def _control_grid_laplacian_1d(n: int) -> csr_matrix:
    """1D second-difference matrix for control grid regularization."""
    if n < 3:
        return eye(n) * 0.0
    diag = np.ones(n) * 2
    diag[0] = diag[-1] = 1
    off = -np.ones(n - 1)
    from scipy.sparse import diags
    return diags([off, diag, off], [-1, 0, 1], shape=(n, n), format='csr')


# ─── 4. Full Pipeline: Mesh → NURBSSurfaceData ─────────────────────────────

def mesh_to_bspline_surface(
        vertices: np.ndarray,  # [V, 3]
        faces: np.ndarray,  # [F, 3] triangle indices
        colors: Optional[np.ndarray],  # [V, 3] per-vertex RGB
        n_ctrl_u: int = 32,
        n_ctrl_v: int = 32,
        degree: int = 3,
        smoothing: float = 0.001,
) -> dict:
    """
    Rhino Patch equivalent: mesh → B-spline surface.

    Pipeline:
        1. Harmonic parameterization (mesh → UV)
        2. Least-squares B-spline approximation (UV + XYZ → control points)

    Returns dict compatible with NURBSSurfaceData.
    """
    # Step 1: Parameterize
    uv = harmonic_parameterization(vertices, faces)

    # Step 2: Least-squares fit
    ctrl_pts, ctrl_colors, knots_u, knots_v = least_squares_bspline_surface(
        vertices, uv, n_ctrl_u, n_ctrl_v,
        degree=degree,
        smoothing=smoothing,
        colors=colors,
    )

    return {
        'control_points': ctrl_pts.astype(np.float32),
        'control_colors': ctrl_colors.astype(np.float32) if ctrl_colors is not None else None,
        'knots_u': knots_u.astype(np.float32),
        'knots_v': knots_v.astype(np.float32),
        'degree_u': degree,
        'degree_v': degree,
    }