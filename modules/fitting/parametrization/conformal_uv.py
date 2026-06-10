"""
Conformal UV parameterization via Tutte embedding with cotangent weights.

Operates on a TRIANGLE MESH (from TSDF extraction), NOT on a raw point cloud.
This is the mathematically correct input for conformal parameterization:
the mesh provides the topology that a point cloud lacks.

When a mesh is unavailable, falls back to PCA projection (lossless—same as before).
"""

import numpy as np
import warnings
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, lsqr
from typing import Optional, Tuple


def conformal_parameterize_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_cot: float = 50.0,
    min_triangle_area: float = 1e-12,
) -> np.ndarray:
    """
    Conformal UV parameterization for a triangle mesh.

    This is the correct algorithm: the mesh provides connectivity,
    so we don't need to fabricate it from a 2D projection.

    Parameters
    ----------
    vertices : (V, 3) vertex positions
    triangles : (F, 3) integer face indices into vertices
    max_cot : clamp cotangent weights to prevent overflow
    min_triangle_area : skip degenerate faces

    Returns
    -------
    uv : (V, 2) in [0.001, 0.999]
    """
    V = len(vertices)
    F = len(triangles)

    if V < 4 or F < 1:
        return _fallback_pca(vertices)

    # ------------------------------------------------------------------
    # 1. Filter degenerate triangles
    # ------------------------------------------------------------------
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid_mask = areas > min_triangle_area
    triangles = triangles[valid_mask]

    if len(triangles) < 1:
        return _fallback_pca(vertices)

    # ------------------------------------------------------------------
    # 2. Find connected component with most vertices
    #    (TSDF meshes can have small disconnected fragments)
    # ------------------------------------------------------------------
    triangles, vert_map, V_new = _largest_connected_component(
        triangles, V
    )
    # vert_map[old_idx] = new_idx (or -1 if not in largest component)
    # We'll compute UV on the component, then scatter back

    if V_new < 4:
        return _fallback_pca(vertices)

    verts_component = np.zeros((V_new, 3))
    for old_idx, new_idx in enumerate(vert_map):
        if new_idx >= 0:
            verts_component[new_idx] = vertices[old_idx]

    # ------------------------------------------------------------------
    # 3. Build cotangent-weight Laplacian
    # ------------------------------------------------------------------
    L = _build_cotangent_laplacian(
        verts_component, triangles, V_new, max_cot
    )

    if L is None:
        return _fallback_pca(vertices)

    # ------------------------------------------------------------------
    # 4. Find mesh boundary (edges with exactly 1 adjacent face)
    # ------------------------------------------------------------------
    boundary_loop = _find_boundary_loop(triangles, V_new)

    if boundary_loop is None or len(boundary_loop) < 3:
        # Closed mesh (no boundary) → use virtual boundary
        # Pick vertices on the convex hull of the PCA projection
        boundary_loop = _convex_hull_boundary(verts_component)

    if boundary_loop is None or len(boundary_loop) < 3:
        return _fallback_pca(vertices)

    # ------------------------------------------------------------------
    # 5. Pin boundary to unit circle
    # ------------------------------------------------------------------
    n_bnd = len(boundary_loop)
    bnd_uv = np.zeros((n_bnd, 2))

    # Distribute boundary vertices proportional to their edge lengths
    # along the unit circle — much better than uniform spacing
    bnd_verts = verts_component[boundary_loop]
    edge_lens = np.linalg.norm(
        np.diff(np.vstack([bnd_verts, bnd_verts[:1]]), axis=0), axis=1
    )
    cumlen = np.concatenate([[0], np.cumsum(edge_lens)])
    cumlen /= cumlen[-1] + 1e-15  # normalize to [0, 1)

    for i in range(n_bnd):
        angle = 2.0 * np.pi * cumlen[i]
        bnd_uv[i] = [0.5 + 0.4 * np.cos(angle),
                      0.5 + 0.4 * np.sin(angle)]

    # ------------------------------------------------------------------
    # 6. Solve L_ii · uv_int = -L_ib · uv_bnd
    # ------------------------------------------------------------------
    bnd_set = set(boundary_loop)
    interior = np.array([v for v in range(V_new) if v not in bnd_set])

    if len(interior) == 0:
        uv_component = np.full((V_new, 2), 0.5)
        uv_component[boundary_loop] = bnd_uv
    else:
        bnd_arr = np.array(boundary_loop)

        L_ii = L[np.ix_(interior, interior)]
        L_ib = L[np.ix_(interior, bnd_arr)]

        uv_component = np.zeros((V_new, 2))
        uv_component[bnd_arr] = bnd_uv

        for dim in range(2):
            rhs = -L_ib @ bnd_uv[:, dim]

            try:
                sol = spsolve(L_ii.tocsc(), rhs)
            except Exception:
                # spsolve failed → try LSQR (handles rank-deficient)
                try:
                    sol = lsqr(L_ii, rhs, atol=1e-10, btol=1e-10)[0]
                except Exception as e:
                    warnings.warn(
                        f"[conformal_uv] Both solvers failed: {e}"
                    )
                    return _fallback_pca(vertices)

            if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
                warnings.warn(
                    "[conformal_uv] NaN/Inf in solution, "
                    "falling back to PCA"
                )
                return _fallback_pca(vertices)

            uv_component[interior, dim] = sol

    # ------------------------------------------------------------------
    # 7. Scatter back to original vertex ordering + normalize
    # ------------------------------------------------------------------
    uv_full = np.full((V, 2), 0.5)  # vertices not in component get 0.5
    for old_idx, new_idx in enumerate(vert_map):
        if new_idx >= 0:
            uv_full[old_idx] = uv_component[new_idx]

    # Normalize to [0, 1]
    uv_min = uv_full.min(axis=0)
    uv_max = uv_full.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range[uv_range < 1e-10] = 1.0
    uv_full = (uv_full - uv_min) / uv_range

    if np.any(np.isnan(uv_full)):
        return _fallback_pca(vertices)

    return np.clip(uv_full, 0.001, 0.999)


def conformal_parameterize(
    points: np.ndarray,
    mesh: "open3d.geometry.TriangleMesh" = None,
    **kwargs,
) -> np.ndarray:
    """
    Public API — dispatches to mesh-based or PCA fallback.

    Parameters
    ----------
    points : (N, 3) point positions
    mesh : optional Open3D TriangleMesh (from TSDF extraction).
           If provided, uses proper conformal parameterization.
           If None, falls back to PCA.
    **kwargs : forwarded to conformal_parameterize_mesh

    Returns
    -------
    uv : (N, 2) in [0.001, 0.999]
    """
    if mesh is not None:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int64)

        if len(tris) > 0 and len(verts) >= 4:
            uv_mesh = conformal_parameterize_mesh(verts, tris, **kwargs)

            # If input `points` differs from mesh vertices,
            # interpolate UV via nearest-neighbor
            if len(points) != len(verts) or not np.allclose(
                points, verts, atol=1e-6
            ):
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                _, nearest_idx = tree.query(points, k=1)
                return uv_mesh[nearest_idx]

            return uv_mesh

    # No mesh → PCA fallback
    return _fallback_pca(points)


# ======================================================================
#  Internal Helpers
# ======================================================================

def _build_cotangent_laplacian(
    vertices: np.ndarray,
    triangles: np.ndarray,
    V: int,
    max_cot: float,
) -> Optional[csr_matrix]:
    """
    Build symmetric cotangent-weight Laplacian.

    For each triangle (i, j, k):
      - Angle at vertex i → cotangent weights edge (j, k)
      - Angle at vertex j → cotangent weights edge (i, k)
      - Angle at vertex k → cotangent weights edge (i, j)

    The Laplacian L is negative semi-definite:
      L[a, b] = sum of cot weights for edge (a, b)  [off-diagonal]
      L[a, a] = -sum of L[a, b] for all b            [diagonal]
    """
    L = lil_matrix((V, V), dtype=np.float64)
    n_degenerate = 0

    for f in range(len(triangles)):
        i, j, k = int(triangles[f, 0]), int(triangles[f, 1]), int(triangles[f, 2])
        pi, pj, pk = vertices[i], vertices[j], vertices[k]

        cots = _triangle_cotangents(pi, pj, pk, max_cot)
        if cots is None:
            n_degenerate += 1
            continue

        cot_i, cot_j, cot_k = cots

        # cot at vertex i → weights edge (j, k)
        _add_symmetric(L, j, k, 0.5 * cot_i)
        # cot at vertex j → weights edge (i, k)
        _add_symmetric(L, i, k, 0.5 * cot_j)
        # cot at vertex k → weights edge (i, j)
        _add_symmetric(L, i, j, 0.5 * cot_k)

    if n_degenerate > 0.5 * len(triangles):
        warnings.warn(
            f"[conformal_uv] {n_degenerate}/{len(triangles)} "
            "degenerate triangles — mesh quality is poor"
        )
        return None

    return L.tocsr()


def _triangle_cotangents(
    pi: np.ndarray, pj: np.ndarray, pk: np.ndarray,
    max_cot: float,
) -> Optional[Tuple[float, float, float]]:
    """Compute clamped cotangent of each angle. Returns None if degenerate."""
    edges = [
        (pj - pi, pk - pi),  # angle at i
        (pi - pj, pk - pj),  # angle at j
        (pi - pk, pj - pk),  # angle at k
    ]
    cots = []
    for ea, eb in edges:
        la = np.linalg.norm(ea)
        lb = np.linalg.norm(eb)
        if la < 1e-15 or lb < 1e-15:
            return None
        cos_a = np.dot(ea, eb) / (la * lb)
        cos_a = np.clip(cos_a, -0.9999, 0.9999)
        sin_a = np.sqrt(1.0 - cos_a * cos_a)
        if sin_a < 1e-10:
            return None
        cot = np.clip(cos_a / sin_a, -max_cot, max_cot)
        cots.append(cot)
    return tuple(cots)


def _add_symmetric(L: lil_matrix, a: int, b: int, w: float):
    """Add weight w to both (a,b) and (b,a), subtract from diagonal."""
    L[a, b] += w
    L[b, a] += w
    L[a, a] -= w
    L[b, b] -= w


def _largest_connected_component(
    triangles: np.ndarray, V: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract the largest connected component from a triangle mesh.
    Returns remapped triangles, vertex mapping, and new vertex count.
    """
    from collections import defaultdict, deque

    # Build adjacency
    adj = defaultdict(set)
    vert_in_mesh = set()
    for f in triangles:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        adj[i].update([j, k])
        adj[j].update([i, k])
        adj[k].update([i, j])
        vert_in_mesh.update([i, j, k])

    # BFS to find components
    visited = set()
    components = []

    for start in vert_in_mesh:
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            component.add(v)
            for nb in adj[v]:
                if nb not in visited:
                    queue.append(nb)
        components.append(component)

    # Pick largest
    largest = max(components, key=len)

    # Build vertex remapping
    vert_map = np.full(V, -1, dtype=np.int64)
    new_idx = 0
    for old_idx in sorted(largest):
        vert_map[old_idx] = new_idx
        new_idx += 1
    V_new = new_idx

    # Remap triangles
    new_tris = []
    for f in triangles:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        if vert_map[i] >= 0 and vert_map[j] >= 0 and vert_map[k] >= 0:
            new_tris.append([vert_map[i], vert_map[j], vert_map[k]])

    return np.array(new_tris, dtype=np.int64), vert_map, V_new


def _find_boundary_loop(
    triangles: np.ndarray, V: int
) -> Optional[list]:
    """
    Find the longest boundary loop of the mesh.
    Boundary edges = edges appearing in exactly 1 triangle.
    Returns an ordered list of vertex indices, or None.
    """
    from collections import Counter, defaultdict

    edge_count = Counter()
    for f in triangles:
        verts = sorted([int(f[0]), int(f[1]), int(f[2])])
        edge_count[(verts[0], verts[1])] += 1
        edge_count[(verts[0], verts[2])] += 1
        edge_count[(verts[1], verts[2])] += 1

    boundary_edges = [(a, b) for (a, b), cnt in edge_count.items() if cnt == 1]

    if len(boundary_edges) < 3:
        return None

    # Build boundary adjacency
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Find all boundary loops
    visited_global = set()
    loops = []

    for start in adj:
        if start in visited_global:
            continue
        loop = [start]
        visited_global.add(start)
        current = start
        for _ in range(len(adj) + 1):
            neighbors = [n for n in adj[current] if n not in visited_global]
            if not neighbors:
                break
            nxt = neighbors[0]
            loop.append(nxt)
            visited_global.add(nxt)
            current = nxt
        if len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return None

    # Return the longest loop
    return max(loops, key=len)


def _convex_hull_boundary(vertices: np.ndarray) -> Optional[list]:
    """Fallback boundary from convex hull of PCA projection."""
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull

    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(vertices - vertices.mean(axis=0))
        hull = ConvexHull(proj)
        return hull.vertices.tolist()
    except Exception:
        return None


def _fallback_pca(points: np.ndarray) -> np.ndarray:
    """PCA-based UV parameterization as fallback."""
    from sklearn.decomposition import PCA

    if len(points) < 2:
        return np.full((len(points), 2), 0.5)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(points - points.mean(axis=0))
    uv = proj - proj.min(axis=0)
    rng = uv.max(axis=0)
    rng[rng < 1e-10] = 1.0
    uv = uv / rng
    return np.clip(uv, 0.001, 0.999)