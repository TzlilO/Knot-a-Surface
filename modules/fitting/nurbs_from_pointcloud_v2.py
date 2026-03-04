"""
Surface-Aware NURBS Initialization from Point Clouds

Enhanced clustering that respects surface boundaries and normal discontinuities.
Designed for scenes with multiple distinct objects (fruits on a plate, etc.)

Key Features:
- Normal-discontinuity-aware region growing
- Sharp edge detection and preservation
- Adaptive cluster merging with surface consistency
- Curvature-based seed selection

CORRECTED VERSION - Addresses:
1.Priority queue re-queuing logic
2.Normal orientation consistency via MST
3.Numerical stability in curvature estimation
4.Vectorized affinity matrix construction
5.Integration with existing DecompositionMode enum
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import deque, defaultdict
import heapq

# Core dependencies
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path, minimum_spanning_tree
from scipy.ndimage import gaussian_filter, binary_dilation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata

# NURBS library
from geomdl import BSpline


# =============================================================================
# Unified Enums (Backward Compatible)
# =============================================================================

class DecompositionMode(Enum):
    """Surface decomposition strategies - maintains backward compatibility."""
    SINGLE = "single"
    BACKGROUND_OBJECT = "bg_object"
    K_COMPONENTS = "k_components"
    # New surface-aware modes
    SURFACE_AWARE = "surface_aware"
    SURFACE_AWARE_GRAPH = "surface_aware_graph"


class ClusteringStrategy(Enum):
    """Surface-aware clustering strategies (internal use)."""
    REGION_GROWING = "region_growing"
    GRAPH_CUT = "graph_cut"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SurfaceAwareConfig:
    """Configuration for surface-aware clustering."""

    # --- Clustering Strategy ---
    strategy: ClusteringStrategy = ClusteringStrategy.REGION_GROWING
    target_num_clusters: int = 4
    min_cluster_size:  int = 50
    max_cluster_size:  Optional[int] = None

    # --- Normal Discontinuity Detection ---
    normal_angle_threshold_deg: float = 30.0
    normal_smoothness_sigma: float = 0.3
    use_local_normal_estimation: bool = True
    normal_estimation_k: int = 30

    # --- Region Growing Parameters ---
    seed_selection_method: str = "flatness"
    growth_cost_normal_weight: float = 5.0
    growth_cost_distance_weight: float = 1.0
    growth_cost_color_weight: float = 0.5
    max_growth_cost: float = 2.0

    # --- Boundary Detection ---
    detect_sharp_edges: bool = True
    sharp_edge_normal_threshold_deg: float = 45.0
    boundary_dilation_radius: int = 2

    # --- Merging Parameters ---
    allow_small_cluster_merge: bool = True
    merge_normal_threshold_deg: float = 20.0
    merge_size_threshold: int = 100

    # --- Grid Resolution ---
    adaptive_resolution: bool = True
    base_resolution: int = 64
    min_resolution: int = 32
    max_resolution: int = 256

    # --- NURBS Parameters ---
    degree_u: int = 3
    degree_v: int = 3
    smoothing:  float = 0.05

    # --- Preprocessing ---
    outlier_removal: bool = True
    outlier_std_ratio: float = 2.0

    # --- UV Parameterization ---
    use_geodesic_uv: bool = True

    # --- Backward Compatibility Aliases ---
    decomposition_mode: DecompositionMode = DecompositionMode.SURFACE_AWARE
    n_components: int = 4

    def __post_init__(self):
        # Sync aliases
        if hasattr(self, 'n_components'):
            self.target_num_clusters = self.n_components


# =============================================================================
# Surface Analyzer (Corrected)
# =============================================================================

class SurfaceAnalyzer:
    """
    Analyzes point cloud surface properties for clustering decisions.
    Computes normals, curvature, and detects sharp edges.

    CORRECTED:
    - MST-based normal orientation
    - Robust SVD for curvature
    - Validated k parameters
    """

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        self.points = np.asarray(points, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.float64) if colors is not None else None
        self.n_points = len(points)

        # Lazy computed
        self._kdtree:  Optional[cKDTree] = None
        self._normals: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._curvature: Optional[np.ndarray] = None
        self._flatness: Optional[np.ndarray] = None
        self._sharp_edge_mask: Optional[np.ndarray] = None
        self._normal_variation: Optional[np.ndarray] = None

    @property
    def kdtree(self) -> cKDTree:
        if self._kdtree is None:
            self._kdtree = cKDTree(self.points)
        return self._kdtree

    @property
    def normals(self) -> np.ndarray:
        """Lazy access to normals."""
        if self._normals is None:
            self.estimate_normals()
        return self._normals

    @property
    def edge_mask(self) -> np.ndarray:
        """Lazy access to edge mask."""
        if self._sharp_edge_mask is None:
            self.detect_sharp_edges()
        return self._sharp_edge_mask

    def estimate_normals(self, k: int = 30) -> np.ndarray:
        """
        Estimate surface normals using PCA on local neighborhoods.
        Orients normals consistently using MST propagation.

        CORRECTED:  Uses MST for robust orientation across disconnected regions.
        """
        if self._normals is not None:
            return self._normals

        # Validate k
        k = min(k, self.n_points - 1)
        if k < 3:
            self._normals = np.tile([0, 0, 1], (self.n_points, 1))
            self._eigenvalues = np.ones((self.n_points, 3))
            return self._normals

        normals = np.zeros_like(self.points)
        eigenvalues = np.zeros((self.n_points, 3))

        _, indices = self.kdtree.query(self.points, k=k)

        for i in range(self.n_points):
            neighbors = self.points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)

            try:
                # Use SVD for numerical stability
                _, s, Vt = np.linalg.svd(centered, full_matrices=False)
                normals[i] = Vt[-1]  # Smallest singular vector
                eigenvalues[i] = s ** 2 / (k - 1)  # Approximate eigenvalues
            except np.linalg.LinAlgError:
                normals[i] = np.array([0, 0, 1])
                eigenvalues[i] = np.array([1, 1, 1])

        # Orient normals consistently using MST
        normals = self._orient_normals_mst(normals, indices, k)

        self._normals = normals
        self._eigenvalues = eigenvalues
        return normals

    def _orient_normals_mst(
            self,
            normals: np.ndarray,
            neighbor_indices: np.ndarray,
            k: int = 16
    ) -> np.ndarray:
        """
        Orient normals consistently using minimum spanning tree propagation.
        Handles disconnected components properly.

        CORRECTED: Replaced simple BFS with MST-based approach.
        """
        n = len(normals)
        oriented = normals.copy()

        # Build weighted graph where weight = 1 - |dot(n_i, n_j)|
        rows, cols, weights = [], [], []

        for i in range(n):
            for j in neighbor_indices[i, 1:min(k, len(neighbor_indices[i]))]:
                dot = np.abs(np.dot(normals[i], normals[j]))
                weight = 1.0 - dot + 1e-6
                rows.append(i)
                cols.append(j)
                weights.append(weight)

        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        graph = (graph + graph.T) / 2  # Symmetrize

        # Find MST
        mst = minimum_spanning_tree(graph)

        # Find connected components
        n_components, component_labels = connected_components(mst, directed=False)

        # Orient each component separately
        for comp in range(n_components):
            comp_mask = component_labels == comp
            comp_indices = np.where(comp_mask)[0]

            if len(comp_indices) == 0:
                continue

            # Start from point with most consistent neighborhood
            best_seed = comp_indices[0]
            best_consistency = -1

            for idx in comp_indices[: min(100, len(comp_indices))]:
                neighbor_normals = normals[neighbor_indices[idx, 1:min(k, len(neighbor_indices[idx]))]]
                if len(neighbor_normals) == 0:
                    continue
                consistency = np.abs(np.dot(neighbor_normals, normals[idx])).mean()
                if consistency > best_consistency:
                    best_consistency = consistency
                    best_seed = idx

            # BFS within component using MST edges
            visited = set([best_seed])
            queue = deque([best_seed])

            while queue:
                current = queue.popleft()

                # Get MST neighbors
                mst_row = mst.getrow(current)
                mst_col = mst.getcol(current)
                mst_neighbors = list(set(mst_row.indices) | set(mst_col.indices))

                for neighbor in mst_neighbors:
                    if neighbor not in visited and component_labels[neighbor] == comp:
                        if np.dot(oriented[neighbor], oriented[current]) < 0:
                            oriented[neighbor] = -oriented[neighbor]
                        visited.add(neighbor)
                        queue.append(neighbor)

        return oriented

    def compute_curvature(self, k: int = 16) -> np.ndarray:
        """
        Compute surface curvature using eigenvalue ratios.

        CORRECTED: Uses SVD for numerical stability, handles edge cases.
        """
        if self._curvature is not None:
            return self._curvature

        # Validate k
        k = min(k, self.n_points - 1)
        if k < 4:
            self._curvature = np.zeros(self.n_points)
            return self._curvature

        # Ensure normals/eigenvalues are computed
        self.estimate_normals(k=k)

        # Curvature from eigenvalue ratios
        eigsum = self._eigenvalues.sum(axis=1) + 1e-10
        curvature = self._eigenvalues[: , 0] / eigsum

        # Normalize to [0, 1]
        curvature = np.clip(curvature, 0, 1)

        self._curvature = curvature
        return curvature

    def compute_flatness(self, k: int = 16) -> np.ndarray:
        """
        Compute local flatness score.
        High flatness = good seed candidate for region growing.
        """
        if self._flatness is not None:
            return self._flatness

        # Ensure eigenvalues are computed
        self.estimate_normals(k=k)

        e1 = self._eigenvalues[: , 0]
        e2 = self._eigenvalues[: , 1]
        e3 = self._eigenvalues[: , 2]

        # Flatness:  high when e1 << e2 ≈ e3
        flatness = 1.0 - (e1 / (e3 + 1e-10))
        flatness *= (e2 / (e3 + 1e-10))

        self._flatness = np.clip(flatness, 0, 1)
        return self._flatness

    def compute_normal_variation(self, k: int = 16) -> np.ndarray:
        """Compute local normal variation (high at edges/creases)."""
        if self._normal_variation is not None:
            return self._normal_variation

        k = min(k, self.n_points - 1)
        normals = self.estimate_normals(k=k)
        _, indices = self.kdtree.query(self.points, k=k)

        variation = np.zeros(self.n_points)
        for i in range(self.n_points):
            neighbor_normals = normals[indices[i]]
            dots = np.dot(neighbor_normals, normals[i])
            variation[i] = 1.0 - np.mean(np.abs(dots))

        self._normal_variation = variation
        return variation

    def detect_sharp_edges(
            self,
            angle_threshold_deg: float = 45.0,
            k: int = 16
    ) -> np.ndarray:
        """
        Detect points that lie on sharp edges (high normal discontinuity).
        Returns boolean mask.
        """
        if self._sharp_edge_mask is not None:
            return self._sharp_edge_mask

        k = min(k, self.n_points - 1)
        normals = self.estimate_normals(k=k)
        _, indices = self.kdtree.query(self.points, k=k)

        cos_threshold = np.cos(np.radians(angle_threshold_deg))

        edge_mask = np.zeros(self.n_points, dtype=bool)

        for i in range(self.n_points):
            neighbor_normals = normals[indices[i, 1:]]
            if len(neighbor_normals) == 0:
                continue
            dots = np.dot(neighbor_normals, normals[i])
            if np.any(dots < cos_threshold):
                edge_mask[i] = True

        self._sharp_edge_mask = edge_mask
        return edge_mask


# =============================================================================
# Surface-Aware Region Growing (Corrected)
# =============================================================================

class SurfaceAwareRegionGrowing:
    """
    Region growing algorithm that respects surface boundaries.

    CORRECTED:
    - Proper re-queuing for rejected points
    - Connectivity check before assignment
    - EMA for cluster statistics
    - Validated k parameters
    """

    def __init__(
            self,
            analyzer: SurfaceAnalyzer,
            config: SurfaceAwareConfig
    ):
        self.analyzer = analyzer
        self.config = config
        self.points = analyzer.points
        self.n_points = analyzer.n_points

        if self.n_points < config.min_cluster_size:
            raise ValueError(
                f"Too few points ({self.n_points}) for clustering."
                f"Minimum required:  {config.min_cluster_size}"
            )

        # Precompute with validated k
        k_normal = min(config.normal_estimation_k, self.n_points - 1)
        self.normals = analyzer.estimate_normals(k=k_normal)
        self.curvature = analyzer.compute_curvature(k=min(16, self.n_points - 1))
        self.flatness = analyzer.compute_flatness(k=min(16, self.n_points - 1))

        if config.detect_sharp_edges:
            self.edge_mask = analyzer.detect_sharp_edges(
                config.sharp_edge_normal_threshold_deg,
                k=min(16, self.n_points - 1)
            )
        else:
            self.edge_mask = np.zeros(self.n_points, dtype=bool)

        # Build neighbor graph with validated k
        self.k_neighbors = min(20, self.n_points - 1)
        self.neighbor_distances, self.neighbor_indices = analyzer.kdtree.query(
            self.points, k=self.k_neighbors
        )

        # Robust distance scale
        valid_dists = self.neighbor_distances[: , 1:].ravel()
        valid_dists = valid_dists[valid_dists > 0]
        self.distance_scale = np.percentile(valid_dists, 90) if len(valid_dists) > 0 else 1.0

    def select_seeds(self, num_seeds: int) -> List[int]:
        """
        Select seed points for region growing.
        Seeds should be in flat, non-edge regions and spatially distributed.
        """
        method = self.config.seed_selection_method

        if method == "flatness":
            scores = self.flatness * (1.0 - self.curvature) * (~self.edge_mask).astype(float)
            seeds = self._select_spread_seeds(scores, num_seeds)

        elif method == "curvature":
            scores = 1.0 - self.curvature
            scores[self.edge_mask] = 0
            seeds = self._select_spread_seeds(scores, num_seeds)

        else:  # random
            valid_indices = np.where(~self.edge_mask)[0]
            if len(valid_indices) < num_seeds:
                valid_indices = np.arange(self.n_points)
            seeds = np.random.choice(valid_indices, size=min(num_seeds, len(valid_indices)), replace=False).tolist()

        return seeds

    def _select_spread_seeds(self, scores: np.ndarray, num_seeds: int) -> List[int]:
        """
        Select seeds that are spread out spatially while having high scores.
        Uses greedy farthest-point sampling with score weighting.
        """
        seeds = []

        # Start with highest score point
        first_seed = int(np.argmax(scores))
        seeds.append(first_seed)

        # Track minimum distance to any seed
        min_dist_to_seeds = np.full(self.n_points, np.inf)

        for _ in range(num_seeds - 1):
            last_seed = seeds[-1]
            dists = np.linalg.norm(self.points - self.points[last_seed], axis=1)
            min_dist_to_seeds = np.minimum(min_dist_to_seeds, dists)

            # Combined score: original score * distance to existing seeds
            combined = scores * min_dist_to_seeds
            combined[seeds] = -np.inf

            next_seed = int(np.argmax(combined))
            if combined[next_seed] <= 0:
                break
            seeds.append(next_seed)

        return seeds

    def compute_growth_cost(
            self,
            candidate_idx: int,
            cluster_idx: int,
            cluster_normal: np.ndarray,
            cluster_color: Optional[np.ndarray] = None,
            cluster_points: Optional[set] = None
    ) -> float:
        """
        Compute cost of adding candidate point to cluster.

        CORRECTED: Added connectivity check.
        """
        # Check connectivity to cluster
        if cluster_points is not None:
            neighbors = set(self.neighbor_indices[candidate_idx, 1:])
            if not neighbors.intersection(cluster_points):
                return np.inf

        # Edge boundary check
        if self.edge_mask[candidate_idx]:
            candidate_normal = self.normals[candidate_idx]
            dot = np.abs(np.dot(candidate_normal, cluster_normal))
            angle = np.degrees(np.arccos(np.clip(dot, 0, 1)))

            if angle > self.config.normal_angle_threshold_deg:
                return np.inf

        # Normal cost with soft threshold
        candidate_normal = self.normals[candidate_idx]
        dot = np.abs(np.dot(candidate_normal, cluster_normal))
        dot = np.clip(dot, 0, 1)

        sigma_rad = np.radians(self.config.normal_smoothness_sigma * 45)
        angle_rad = np.arccos(dot)
        normal_cost = 1.0 - np.exp(-angle_rad ** 2 / (2 * sigma_rad ** 2 + 1e-10))

        # Color cost
        color_cost = 0.0
        if self.analyzer.colors is not None and cluster_color is not None:
            candidate_color = self.analyzer.colors[candidate_idx]
            color_diff = np.linalg.norm(candidate_color - cluster_color)
            color_cost = min(color_diff / (np.sqrt(3) + 1e-8), 1.0)

        total_cost = (
                self.config.growth_cost_normal_weight * normal_cost +
                self.config.growth_cost_color_weight * color_cost
        )

        return total_cost

    def grow_regions(self, seeds: List[int]) -> np.ndarray:
        """
        Grow regions from seeds with proper re-queuing and connectivity checks.

        CORRECTED: Points that exceed cost threshold are re-queued for other clusters.
        """
        num_clusters = len(seeds)
        labels = np.full(self.n_points, -1, dtype=np.int32)

        # Cluster state
        cluster_normals = [self.normals[seed].copy() for seed in seeds]
        cluster_colors = [
            self.analyzer.colors[seed].copy()
            if self.analyzer.colors is not None else None
            for seed in seeds
        ]
        cluster_sizes = [1] * num_clusters
        cluster_points = [set([seed]) for seed in seeds]

        # Assign seeds
        for cluster_idx, seed in enumerate(seeds):
            labels[seed] = cluster_idx

        # Priority queue with generation counter for re-queuing
        # (cost, generation, point_idx, cluster_idx)
        pq = []
        generation = 0
        point_generations = defaultdict(int)
        MAX_REQUEUE = 3

        # Initialize queue
        for cluster_idx, seed in enumerate(seeds):
            for neighbor_idx in self.neighbor_indices[seed, 1:]:
                if labels[neighbor_idx] == -1:
                    cost = self.compute_growth_cost(
                        neighbor_idx, cluster_idx,
                        cluster_normals[cluster_idx],
                        cluster_colors[cluster_idx],
                        cluster_points[cluster_idx]
                    )
                    if cost < np.inf:
                        heapq.heappush(pq, (cost, generation, neighbor_idx, cluster_idx))
                        generation += 1

        while pq:
            cost, _, point_idx, cluster_idx = heapq.heappop(pq)

            # Skip if already assigned
            if labels[point_idx] != -1:
                continue

            # Re-validate cost
            current_cost = self.compute_growth_cost(
                point_idx, cluster_idx,
                cluster_normals[cluster_idx],
                cluster_colors[cluster_idx],
                cluster_points[cluster_idx]
            )

            if current_cost > self.config.max_growth_cost:
                # Try to find alternative cluster
                if point_generations[point_idx] < MAX_REQUEUE:
                    point_generations[point_idx] += 1

                    best_alt_cost = np.inf
                    best_alt_cluster = -1

                    for alt_cluster in range(num_clusters):
                        if alt_cluster == cluster_idx:
                            continue
                        alt_cost = self.compute_growth_cost(
                            point_idx, alt_cluster,
                            cluster_normals[alt_cluster],
                            cluster_colors[alt_cluster],
                            cluster_points[alt_cluster]
                        )
                        if alt_cost < best_alt_cost:
                            best_alt_cost = alt_cost
                            best_alt_cluster = alt_cluster

                    if best_alt_cluster >= 0 and best_alt_cost < self.config.max_growth_cost:
                        heapq.heappush(pq, (best_alt_cost, generation, point_idx, best_alt_cluster))
                        generation += 1
                continue

            # Assign point
            labels[point_idx] = cluster_idx
            cluster_points[cluster_idx].add(point_idx)
            cluster_sizes[cluster_idx] += 1

            # Update cluster statistics (EMA for stability)
            n = cluster_sizes[cluster_idx]
            alpha = 2.0 / (n + 1)

            new_normal = (1 - alpha) * cluster_normals[cluster_idx] + alpha * self.normals[point_idx]
            norm = np.linalg.norm(new_normal)
            cluster_normals[cluster_idx] = new_normal / (norm + 1e-10)

            if cluster_colors[cluster_idx] is not None:
                cluster_colors[cluster_idx] = (
                        (1 - alpha) * cluster_colors[cluster_idx] +
                        alpha * self.analyzer.colors[point_idx]
                )

            # Add neighbors to queue
            for neighbor_idx in self.neighbor_indices[point_idx, 1:]:
                if labels[neighbor_idx] == -1:
                    neighbor_cost = self.compute_growth_cost(
                        neighbor_idx, cluster_idx,
                        cluster_normals[cluster_idx],
                        cluster_colors[cluster_idx],
                        cluster_points[cluster_idx]
                    )
                    if neighbor_cost < np.inf:
                        heapq.heappush(pq, (neighbor_cost, generation, neighbor_idx, cluster_idx))
                        generation += 1

        return labels

    def merge_small_clusters(
            self,
            labels: np.ndarray,
            min_size: int
    ) -> np.ndarray:
        """Merge small clusters into neighboring larger clusters with normal consistency."""
        unique_labels = np.unique(labels[labels >= 0])
        cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}

        small_clusters = [l for l, size in cluster_sizes.items() if size < min_size]

        if not small_clusters:
            return labels

        new_labels = labels.copy()

        for small_label in small_clusters:
            small_mask = labels == small_label
            small_indices = np.where(small_mask)[0]

            if len(small_indices) == 0:
                continue

            # Compute average normal of small cluster
            small_normal = self.normals[small_indices].mean(axis=0)
            norm = np.linalg.norm(small_normal)
            small_normal = small_normal / (norm + 1e-10)

            # Find neighboring clusters
            neighbor_labels = set()
            for idx in small_indices:
                for neighbor_idx in self.neighbor_indices[idx, 1:]:
                    if labels[neighbor_idx] >= 0 and labels[neighbor_idx] != small_label:
                        neighbor_labels.add(labels[neighbor_idx])

            # Find best merge candidate
            best_target = None
            best_score = -np.inf
            merge_threshold = np.cos(np.radians(self.config.merge_normal_threshold_deg))

            for target_label in neighbor_labels:
                target_mask = labels == target_label
                target_normal = self.normals[target_mask].mean(axis=0)
                norm = np.linalg.norm(target_normal)
                target_normal = target_normal / (norm + 1e-10)

                consistency = np.abs(np.dot(small_normal, target_normal))

                if consistency > merge_threshold and consistency > best_score:
                    best_score = consistency
                    best_target = target_label

            if best_target is not None:
                new_labels[small_mask] = best_target

        return self._relabel_consecutive(new_labels)

    def _relabel_consecutive(self, labels: np.ndarray) -> np.ndarray:
        """Relabel clusters to consecutive integers starting from 0."""
        new_labels = labels.copy()
        unique_labels = np.unique(labels[labels >= 0])

        for new_label, old_label in enumerate(sorted(unique_labels)):
            new_labels[labels == old_label] = new_label

        return new_labels

    def _assign_remaining(self, labels: np.ndarray) -> np.ndarray:
        """Assign remaining unassigned points to nearest cluster."""
        new_labels = labels.copy()
        unassigned = np.where(labels == -1)[0]
        assigned = np.where(labels >= 0)[0]

        if len(assigned) == 0:
            return new_labels

        assigned_tree = cKDTree(self.points[assigned])

        for idx in unassigned:
            _, nearest = assigned_tree.query(self.points[idx], k=1)
            new_labels[idx] = labels[assigned[nearest]]

        return new_labels

    def cluster(self) -> np.ndarray:
        """Main clustering method."""
        num_seeds = min(
            self.config.target_num_clusters,
            self.n_points // max(self.config.min_cluster_size, 1)
        )
        num_seeds = max(1, num_seeds)

        seeds = self.select_seeds(num_seeds)

        if len(seeds) == 0:
            return np.zeros(self.n_points, dtype=np.int32)

        print(f"[RegionGrowing] Selected {len(seeds)} seeds from {self.n_points} points")

        labels = self.grow_regions(seeds)

        unassigned = np.sum(labels == -1)
        print(f"[RegionGrowing] After growing: {unassigned} unassigned points")

        if unassigned > 0:
            labels = self._assign_remaining(labels)

        if self.config.allow_small_cluster_merge:
            labels = self.merge_small_clusters(labels, self.config.min_cluster_size)

        final_labels = self._relabel_consecutive(labels)
        num_final = len(np.unique(final_labels[final_labels >= 0]))
        print(f"[RegionGrowing] Final:  {num_final} clusters")

        return final_labels


# =============================================================================
# Graph-Cut Based Clustering (Vectorized)
# =============================================================================

class NormalAwareGraphClustering:
    """
    Graph-based clustering with normal-weighted edges.

    CORRECTED: Vectorized affinity matrix construction.
    """

    def __init__(
            self,
            analyzer: SurfaceAnalyzer,
            config: SurfaceAwareConfig
    ):
        self.analyzer = analyzer
        self.config = config
        self.points = analyzer.points
        self.n_points = analyzer.n_points

        k = min(config.normal_estimation_k, self.n_points - 1)
        self.normals = analyzer.estimate_normals(k)
        self.edge_mask = analyzer.detect_sharp_edges(config.sharp_edge_normal_threshold_deg)

    def build_affinity_matrix(self, k: int = 15) -> csr_matrix:
        """
        Build affinity matrix with vectorized operations.

        CORRECTED: Fully vectorized implementation.
        """
        n = self.n_points
        k = min(k, n - 1)

        tree = cKDTree(self.points)
        distances, indices = tree.query(self.points, k=k)

        # Flatten for vectorized operations
        rows = np.repeat(np.arange(n), k - 1)
        cols = indices[:, 1:].ravel()
        dists = distances[:, 1:].ravel()

        # Spatial weights
        dist_scale = np.percentile(dists, 90)
        spatial_weights = np.exp(-dists ** 2 / (2 * dist_scale ** 2))

        # Normal consistency (vectorized)
        normals_i = self.normals[rows]
        normals_j = self.normals[cols]
        dots = np.abs(np.einsum('ij,ij->i', normals_i, normals_j))

        cos_threshold = np.cos(np.radians(self.config.normal_angle_threshold_deg))

        # Edge mask
        edge_i = self.edge_mask[rows]
        edge_j = self.edge_mask[cols]
        sharp_boundary = (edge_i | edge_j) & (dots < cos_threshold)

        # Normal affinity
        sigma = self.config.normal_smoothness_sigma
        angles = np.arccos(np.clip(dots, 0, 1))
        normal_weights = np.exp(-angles ** 2 / (2 * sigma ** 2))
        normal_weights[sharp_boundary] = 0.01

        # Color consistency
        if self.analyzer.colors is not None:
            colors_i = self.analyzer.colors[rows]
            colors_j = self.analyzer.colors[cols]
            color_diffs = np.linalg.norm(colors_i - colors_j, axis=1)
            color_weights = np.exp(-color_diffs ** 2 / 0.5)
        else:
            color_weights = np.ones_like(spatial_weights)

        # Combined weights
        w_n = self.config.growth_cost_normal_weight
        w_d = self.config.growth_cost_distance_weight
        w_c = self.config.growth_cost_color_weight

        combined = (
                spatial_weights ** w_d *
                normal_weights ** w_n *
                color_weights ** w_c
        )

        # Build sparse matrix
        affinity = csr_matrix((combined, (rows, cols)), shape=(n, n))
        affinity = (affinity + affinity.T) / 2

        return affinity

    def cluster(self) -> np.ndarray:
        """Perform spectral clustering with custom affinity."""
        from sklearn.cluster import SpectralClustering

        print("[GraphClustering] Building affinity matrix...")
        affinity = self.build_affinity_matrix()

        print("[GraphClustering] Running spectral clustering...")
        clustering = SpectralClustering(
            n_clusters=self.config.target_num_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )

        labels = clustering.fit_predict(affinity.toarray())

        return labels


# =============================================================================
# Main Surface-Aware Decomposer
# =============================================================================

class SurfaceAwareDecomposer:
    """Main interface for surface-aware point cloud decomposition."""

    def __init__(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            config: Optional[SurfaceAwareConfig] = None
    ):
        self.config = config or SurfaceAwareConfig()
        self.analyzer = SurfaceAnalyzer(points, colors)
        self.points = points
        self.colors = colors

    def decompose(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decompose point cloud into surface-consistent clusters."""
        strategy = self.config.strategy

        if strategy == ClusteringStrategy.REGION_GROWING:
            clusterer = SurfaceAwareRegionGrowing(self.analyzer, self.config)
            labels = clusterer.cluster()

        elif strategy == ClusteringStrategy.GRAPH_CUT:
            clusterer = NormalAwareGraphClustering(self.analyzer, self.config)
            labels = clusterer.cluster()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        metadata = self._compute_metadata(labels)

        return labels, metadata

    def _compute_metadata(self, labels: np.ndarray) -> Dict[str, Any]:
        """Compute statistics about the clustering."""
        unique_labels = np.unique(labels[labels >= 0])

        cluster_stats = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = self.points[mask]
            cluster_normals = self.analyzer.normals[mask]

            avg_normal = cluster_normals.mean(axis=0)
            norm = np.linalg.norm(avg_normal)
            avg_normal = avg_normal / (norm + 1e-10)

            dots = np.dot(cluster_normals, avg_normal)
            normal_consistency = np.mean(np.abs(dots))

            bbox_min = cluster_points.min(axis=0)
            bbox_max = cluster_points.max(axis=0)

            cluster_stats.append({
                'label':  int(label),
                'size': int(np.sum(mask)),
                'normal_consistency': float(normal_consistency),
                'avg_normal': avg_normal.tolist(),
                'bbox_min': bbox_min.tolist(),
                'bbox_max':  bbox_max.tolist(),
                'centroid': cluster_points.mean(axis=0).tolist()
            })

        edge_count = 0
        if hasattr(self.analyzer, '_sharp_edge_mask') and self.analyzer._sharp_edge_mask is not None:
            edge_count = int(self.analyzer._sharp_edge_mask.sum())

        return {
            'num_clusters': len(unique_labels),
            'cluster_stats': cluster_stats,
            'total_points': len(labels),
            'edge_points': edge_count
        }

    def get_cluster_indices(self, labels: np.ndarray) -> List[np.ndarray]:
        """Convert labels to list of index arrays per cluster."""
        unique_labels = np.unique(labels[labels >= 0])
        return [np.where(labels == l)[0] for l in sorted(unique_labels)]


# =============================================================================
# Data Containers
# =============================================================================

@dataclass
class NURBSSurfaceData:
    """Container for a single NURBS surface."""
    control_points: np.ndarray
    control_colors: np.ndarray
    knots_u: np.ndarray
    knots_v: np.ndarray
    degree_u: int
    degree_v: int
    label: str = "surface"
    point_indices: Optional[np.ndarray] = None
    bounds: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict] = None


@dataclass
class MultiSurfaceResult:
    """Result container for multi-surface decomposition."""
    surfaces: List[NURBSSurfaceData]
    labels: np.ndarray
    decomposition_mode: DecompositionMode = DecompositionMode.SURFACE_AWARE
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# NURBS Surface Fitter
# =============================================================================

class SurfaceAwareNURBSFitter:
    """Fits NURBS surfaces using surface-aware clustering."""

    def __init__(self, config: Optional[SurfaceAwareConfig] = None):
        self.config = config or SurfaceAwareConfig()

    def fit(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None
    ) -> MultiSurfaceResult:
        """Main entry point:  decompose point cloud and fit NURBS to each cluster."""
        # Preprocessing
        if self.config.outlier_removal:
            points, colors, _ = self._remove_outliers(points, colors)

        print(f"[SurfaceAwareNURBS] Processing {len(points)} points")

        # Surface-aware decomposition
        decomposer = SurfaceAwareDecomposer(points, colors, self.config)
        labels, decomp_metadata = decomposer.decompose()

        cluster_indices = decomposer.get_cluster_indices(labels)

        print(f"[SurfaceAwareNURBS] Decomposed into {len(cluster_indices)} clusters")

        # Fit NURBS to each cluster
        surfaces = []
        for i, indices in enumerate(cluster_indices):
            if len(indices) < self.config.min_cluster_size:
                print(f"[SurfaceAwareNURBS] Skipping cluster {i} (too small:  {len(indices)} points)")
                continue

            cluster_points = points[indices]
            cluster_colors = colors[indices] if colors is not None else None

            try:
                # Get cluster metadata safely
                cluster_meta = {}
                if i < len(decomp_metadata.get('cluster_stats', [])):
                    cluster_meta = decomp_metadata['cluster_stats'][i]

                surface = self._fit_single_surface(
                    cluster_points,
                    cluster_colors,
                    label=f"surface_{i}",
                    cluster_metadata=cluster_meta
                )
                surface.point_indices = indices
                surfaces.append(surface)

                print(f"[SurfaceAwareNURBS] Cluster {i}:  {len(indices)} pts -> "
                      f"{surface.control_points.shape[0]}x{surface.control_points.shape[1]} grid")

            except Exception as e:
                print(f"[SurfaceAwareNURBS] Failed to fit cluster {i}: {e}")

        return MultiSurfaceResult(
            surfaces=surfaces,
            labels=labels,
            decomposition_mode=self.config.decomposition_mode,
            metadata={
                'decomposition':  decomp_metadata,
                'config': self.config
            }
        )

    def _remove_outliers(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Remove statistical outliers."""
        tree = cKDTree(points)
        k = min(20, len(points) - 1)
        distances, _ = tree.query(points, k=k)
        mean_distances = distances[: , 1:].mean(axis=1)

        threshold = mean_distances.mean() + self.config.outlier_std_ratio * mean_distances.std()
        inlier_mask = mean_distances < threshold

        clean_points = points[inlier_mask]
        clean_colors = colors[inlier_mask] if colors is not None else None

        print(f"[Outlier Removal] {len(points)} -> {len(clean_points)} points")

        return clean_points, clean_colors, inlier_mask

    def _fit_single_surface(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray],
            label: str,
            cluster_metadata: Dict
    ) -> NURBSSurfaceData:
        """Fit NURBS surface to a single cluster."""
        n_points = len(points)

        if self.config.adaptive_resolution:
            base = int(np.sqrt(n_points) * 0.5)
            res = int(np.clip(base, self.config.min_resolution, self.config.max_resolution))

            try:
                pca = PCA(n_components=2)
                projected = pca.fit_transform(points - points.mean(axis=0))
                extents = projected.max(axis=0) - projected.min(axis=0)
                aspect = max(extents[0], 1e-6) / max(extents[1], 1e-6)
                aspect = np.clip(aspect, 0.25, 4.0)

                if aspect >= 1.0:
                    res_u = res
                    res_v = int(res / aspect)
                else:
                    res_u = int(res * aspect)
                    res_v = res
            except:
                res_u = res_v = res
        else:
            res_u = self.config.base_resolution
            res_v = self.config.base_resolution

        res_u = max(res_u, 8)
        res_v = max(res_v, 8)

        # UV parameterization
        uv_coords = self._compute_uv(points, cluster_metadata)

        # Grid interpolation
        grid_xyz, grid_rgb = self._interpolate_to_grid(
            points, colors, uv_coords, res_u, res_v
        )

        # Smoothing
        if self.config.smoothing > 0:
            sigma = self.config.smoothing * min(res_u, res_v) / 10.0
            for c in range(3):
                grid_xyz[..., c] = gaussian_filter(grid_xyz[..., c], sigma=sigma, mode='nearest')
                if grid_rgb is not None:
                    grid_rgb[..., c] = gaussian_filter(grid_rgb[..., c], sigma=sigma, mode='nearest')

        # Create knot vectors
        degree_u = min(self.config.degree_u, res_u - 1)
        degree_v = min(self.config.degree_v, res_v - 1)
        knots_u = self._create_knot_vector(res_u, degree_u)
        knots_v = self._create_knot_vector(res_v, degree_v)

        return NURBSSurfaceData(
            control_points=grid_xyz.astype(np.float32),
            control_colors=grid_rgb.astype(np.float32) if grid_rgb is not None else np.full(grid_xyz.shape, 0.5,
                                                                                            dtype=np.float32),
            knots_u=knots_u.astype(np.float32),
            knots_v=knots_v.astype(np.float32),
            degree_u=degree_u,
            degree_v=degree_v,
            label=label,
            bounds={
                'min': points.min(axis=0),
                'max': points.max(axis=0),
                'center': points.mean(axis=0)
            },
            metadata=cluster_metadata
        )

    def _compute_uv(
            self,
            points: np.ndarray,
            cluster_metadata: Dict
    ) -> np.ndarray:
        """Compute UV parameterization using cluster's average normal."""
        n_points = len(points)

        if n_points < 8:
            return self._simple_pca_uv(points)

        avg_normal = np.array(cluster_metadata.get('avg_normal', [0, 0, 1]))

        # Find tangent vectors
        if np.abs(avg_normal[2]) < 0.9:
            tangent1 = np.cross(avg_normal, [0, 0, 1])
        else:
            tangent1 = np.cross(avg_normal, [1, 0, 0])

        tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-10)
        tangent2 = np.cross(avg_normal, tangent1)
        tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-10)

        # Project onto tangent plane
        centered = points - points.mean(axis=0)
        u_coords = np.dot(centered, tangent1)
        v_coords = np.dot(centered, tangent2)

        # Normalize to [0, 1]
        u_range = u_coords.max() - u_coords.min()
        v_range = v_coords.max() - v_coords.min()

        u_coords = (u_coords - u_coords.min()) / (u_range + 1e-10)
        v_coords = (v_coords - v_coords.min()) / (v_range + 1e-10)

        uv = np.stack([u_coords, v_coords], axis=1)
        return np.clip(uv, 0.001, 0.999)

    def _simple_pca_uv(self, points: np.ndarray) -> np.ndarray:
        """Simple PCA-based UV for small point sets."""
        centered = points - points.mean(axis=0)
        pca = PCA(n_components=2)
        projected = pca.fit_transform(centered)

        uv = projected - projected.min(axis=0)
        uv = uv / (uv.max(axis=0) + 1e-10)
        return np.clip(uv, 0.001, 0.999)

    def _interpolate_to_grid(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray],
            uv_coords: np.ndarray,
            res_u: int,
            res_v: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Interpolate points to regular grid with hole filling."""
        u_grid = np.linspace(0, 1, res_u)
        v_grid = np.linspace(0, 1, res_v)
        uu, vv = np.meshgrid(u_grid, v_grid, indexing='ij')
        grid_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

        # Interpolate XYZ
        grid_xyz = np.zeros((res_u * res_v, 3))
        for dim in range(3):
            grid_xyz[:, dim] = griddata(
                uv_coords, points[:, dim], grid_uv,
                method='linear', fill_value=points[: , dim].mean()
            )

        # Fill NaN with nearest
        nan_mask = np.isnan(grid_xyz).any(axis=1)
        if nan_mask.any():
            for dim in range(3):
                grid_xyz[nan_mask, dim] = griddata(
                    uv_coords, points[:, dim], grid_uv[nan_mask],
                    method='nearest'
                )

        # Interpolate colors
        if colors is not None:
            grid_rgb = np.zeros((res_u * res_v, 3))
            for dim in range(3):
                grid_rgb[:, dim] = griddata(
                    uv_coords, colors[:, dim], grid_uv,
                    method='linear', fill_value=colors[:, dim].mean()
                )

            nan_mask = np.isnan(grid_rgb).any(axis=1)
            if nan_mask.any():
                for dim in range(3):
                    grid_rgb[nan_mask, dim] = griddata(
                        uv_coords, colors[:, dim], grid_uv[nan_mask],
                        method='nearest'
                    )
            grid_rgb = np.clip(grid_rgb, 0, 1)
        else:
            grid_rgb = None

        return grid_xyz.reshape(res_u, res_v, 3), \
            grid_rgb.reshape(res_u, res_v, 3) if grid_rgb is not None else None

    def _create_knot_vector(self, n_ctrl: int, degree: int) -> np.ndarray:
        """Create clamped uniform knot vector."""
        n_knots = n_ctrl + degree + 1
        n_internal = n_knots - 2 * (degree + 1)

        knots = np.zeros(n_knots)
        knots[-degree - 1:] = 1.0

        if n_internal > 0:
            internal = np.linspace(0, 1, n_internal + 2)[1:-1]
            knots[degree + 1:degree + 1 + n_internal] = internal

        return knots


# =============================================================================
# Convenience Functions
# =============================================================================

def create_surface_aware_nurbs(
        points: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        num_clusters: int = 4,
        normal_threshold_deg: float = 30.0,
        strategy: ClusteringStrategy = ClusteringStrategy.REGION_GROWING,
        **kwargs
) -> MultiSurfaceResult:
    """
    Convenience function for surface-aware NURBS fitting.

    Args:
        points: [N, 3] point cloud
        colors: [N, 3] RGB colors (optional)
        num_clusters:  Target number of surface clusters
        normal_threshold_deg: Angle threshold for sharp edge detection
        strategy: Clustering strategy
        **kwargs: Additional config parameters

    Returns:
        MultiSurfaceResult with fitted surfaces

    Example:
        >>> result = create_surface_aware_nurbs(
        ...    points, colors,
        ...    num_clusters=6,
        ...    normal_threshold_deg=25.0
        ...)
        >>> for surf in result.surfaces:
        ...     print(f"{surf.label}: {surf.control_points.shape}")
    """
    # Convert tensors
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    if colors is not None and hasattr(colors, 'detach'):
        colors = colors.detach().cpu().numpy()

    # Filter valid config kwargs
    valid_fields = set(SurfaceAwareConfig.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    config = SurfaceAwareConfig(
        strategy=strategy,
        target_num_clusters=num_clusters,
        normal_angle_threshold_deg=normal_threshold_deg,
        **config_kwargs
    )

    fitter = SurfaceAwareNURBSFitter(config)
    return fitter.fit(points, colors)


# =============================================================================
# Integration with Existing Pipeline
# =============================================================================

def create_nurbs_from_pointcloud(
        points: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        resolution:  Tuple[int, int] = (64, 64),
        mode: DecompositionMode = DecompositionMode.K_COMPONENTS,
        smoothing: float = 0.05,
        **kwargs
) -> MultiSurfaceResult:
    """
    Main entry point - routes to appropriate implementation based on mode.

    For backward compatibility with existing codebase.
    """
    # Convert tensors
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    if colors is not None and hasattr(colors, 'detach'):
        colors = colors.detach().cpu().numpy()

    # Route based on mode
    if mode in [DecompositionMode.SURFACE_AWARE, DecompositionMode.SURFACE_AWARE_GRAPH]:
        strategy = (ClusteringStrategy.REGION_GROWING
                    if mode == DecompositionMode.SURFACE_AWARE
                    else ClusteringStrategy.GRAPH_CUT)

        return create_surface_aware_nurbs(
            points, colors,
            num_clusters=kwargs.get('n_components', 4),
            normal_threshold_deg=kwargs.get('normal_angle_threshold_deg', 30.0),
            strategy=strategy,
            base_resolution=resolution[0],
            smoothing=smoothing,
            **kwargs
        )

    # Legacy modes - use simpler clustering
    config = SurfaceAwareConfig(
        strategy=ClusteringStrategy.REGION_GROWING,
        target_num_clusters=kwargs.get('n_components', 4) if mode == DecompositionMode.K_COMPONENTS else 2,
        base_resolution=resolution[0],
        smoothing=smoothing,
        decomposition_mode=mode,
        **{k: v for k, v in kwargs.items() if hasattr(SurfaceAwareConfig, k)}
    )

    fitter = SurfaceAwareNURBSFitter(config)
    return fitter.fit(points, colors)


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
        result: MultiSurfaceResult,
        device: str = 'cuda'
) -> Dict[str, Any]:
    """Convert MultiSurfaceResult to torch tensors."""
    import torch

    surfaces = result.surfaces

    cp_list = [torch.tensor(s.control_points, dtype=torch.float32, device=device) for s in surfaces]
    cc_list = [torch.tensor(s.control_colors, dtype=torch.float32, device=device) for s in surfaces]
    ku_list = [torch.tensor(s.knots_u, dtype=torch.float32, device=device) for s in surfaces]
    kv_list = [torch.tensor(s.knots_v, dtype=torch.float32, device=device) for s in surfaces]

    return {
        'control_points': cp_list,
        'control_colors':  cc_list,
        'knots_u': ku_list,
        'knots_v':  kv_list,
        'labels': torch.tensor(result.labels, dtype=torch.long, device=device),
        'surface_labels': [s.label for s in surfaces],
        'num_surfaces': len(surfaces)
    }


# =============================================================================
# Visualization (Optional)
# =============================================================================

def visualize_clustering(
        points: np.ndarray,
        labels: np.ndarray,
        normals: Optional[np.ndarray] = None,
        edge_mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
):
    """Visualize clustering results with optional normal vectors and edge highlighting."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib required for visualization")
        return

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Clusters
    ax1 = fig.add_subplot(131, projection='3d')
    unique_labels = np.unique(labels[labels >= 0])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(
            points[mask, 0], points[mask, 1], points[mask, 2],
            c=[colors[i % len(colors)]], s=1, alpha=0.6, label=f'Cluster {label}'
        )
    ax1.set_title('Clustered Point Cloud')
    ax1.legend()

    # Plot 2: Edge points
    if edge_mask is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(
            points[~edge_mask, 0], points[~edge_mask, 1], points[~edge_mask, 2],
            c='blue', s=1, alpha=0.3, label='Interior'
        )
        ax2.scatter(
            points[edge_mask, 0], points[edge_mask, 1], points[edge_mask, 2],
            c='red', s=5, alpha=0.8, label='Edges'
        )
        ax2.set_title('Edge Detection')
        ax2.legend()

    # Plot 3: Normals (subsampled)
    if normals is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        subsample = np.random.choice(len(points), min(1000, len(points)), replace=False)
        ax3.quiver(
            points[subsample, 0], points[subsample, 1], points[subsample, 2],
            normals[subsample, 0], normals[subsample, 1], normals[subsample, 2],
            length=0.05, normalize=True, alpha=0.5
        )
        ax3.set_title('Surface Normals')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()