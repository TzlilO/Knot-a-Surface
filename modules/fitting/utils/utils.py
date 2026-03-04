"""
Surface Characteristic Scaler

Calculates adaptive learning rate multipliers for SplineModel surfaces based on:
1. Spatial Extent (Bounding Box Diagonal)
2. Geometric Complexity (Curvature / Normal Variance)
3. Control Point Density (Resolution vs. Volume)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from model.modules import SplineModel


class SurfaceCharacteristicScaler:
    def __init__(
            self,
            base_scale: float = 1.0,
            min_scale: float = 0.1,
            max_scale: float = 10.0,
            enable_logging: bool = True
    ):
        """
        Args:
            base_scale: Global multiplier for all surfaces.
            min_scale: Minimum allowed scaling factor.
            max_scale: Maximum allowed scaling factor.
        """
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.enable_logging = enable_logging

    def compute_scale_factors(self, surfaces: List[SplineModel]) -> List[float]:
        """
        Analyzes a list of SplineModels and returns a list of LR multipliers.
        """
        scales = []
        stats_list = []

        # 1. Gather raw statistics
        for i, surface in enumerate(surfaces):
            stats = self._analyze_surface(surface)
            stats_list.append(stats)

        # 2. Normalize and compute relative factors
        # We compute metrics relative to the *average* surface in the scene
        avg_extent = np.mean([s['extent'] for s in stats_list])
        avg_density = np.mean([s['density'] for s in stats_list])
        avg_curvature = np.mean([s['curvature'] for s in stats_list])

        if self.enable_logging:
            print(
                f"\n[SurfaceScaler] Scene Averages - Extent: {avg_extent:.4f}, Density: {avg_density:.4f}, Curvature: {avg_curvature:.4f}")

        for i, stats in enumerate(stats_list):
            # A. Extent Factor: Larger surfaces need larger updates to move same relative distance
            # Logic: If surface is 2x larger, standard gradients might be too small relative to scale.
            extent_factor = (stats['extent'] / (avg_extent + 1e-8)) ** 0.5

            # B. Curvature Factor: Highly curved surfaces are unstable; lower LR
            # Logic: High curvature means high frequency detail. High LR causes "exploding" control points.
            # We invert the relationship: Higher curvature -> Lower Scale
            curvature_ratio = stats['curvature'] / (avg_curvature + 1e-8)
            curvature_factor = 1.0 / (0.5 * curvature_ratio + 0.5)

            # C. Density Factor: Sparse control points covering large area need higher LR
            # Dense control points (high resolution for small area) need lower LR to remain smooth.
            density_ratio = stats['density'] / (avg_density + 1e-8)
            density_factor = 1.0 / (0.5 * density_ratio + 0.5)

            # Combine factors
            # You can tune weights here. Currently equal weight.
            raw_scale = self.base_scale * extent_factor * curvature_factor * density_factor

            # Clamp
            final_scale = np.clip(raw_scale, self.min_scale, self.max_scale)
            scales.append(final_scale)

            if self.enable_logging:
                print(f"[SurfaceScaler] Surf {i}: "
                      f"Ext={extent_factor:.2f}, Curv={curvature_factor:.2f}, Dens={density_factor:.2f} "
                      f"-> Final Scale: {final_scale:.4f}")

        return scales

    def _analyze_surface(self, surface: SplineModel) -> Dict[str, float]:
        """Compute geometric properties of a single surface."""
        # Access control points directly
        # Shape: (1, H, W, 3) -> (H*W, 3)
        cpts = surface.position.cpts.detach().cpu()
        if cpts.dim() > 2:
            cpts = cpts.view(-1, 3)

        # 1. Extent (Diagonal of bounding box)
        mins = cpts.min(dim=0).values
        maxs = cpts.max(dim=0).values
        extent = torch.norm(maxs - mins).item()

        # 2. Density (Control points per unit volume/area)
        # We approximate "area" by extent squared for robustness against flat surfaces
        area_proxy = extent ** 2
        num_points = cpts.shape[0]
        density = num_points / (area_proxy + 1e-6)

        # 3. Curvature Proxy (2nd derivative magnitude)
        # Using finite differences on the control grid
        H, W = surface.state.H, surface.state.W
        grid = cpts.view(H, W, 3)

        # Laplacians
        lap_u = torch.zeros_like(grid)
        lap_v = torch.zeros_like(grid)

        if H > 2:
            lap_u[1:-1] = grid[:-2] - 2 * grid[1:-1] + grid[2:]
        if W > 2:
            lap_v[:, 1:-1] = grid[:, :-2] - 2 * grid[:, 1:-1] + grid[:, 2:]

        # Mean magnitude of Laplacian indicates "bumpiness"
        curvature = (lap_u.norm(dim=-1).mean() + lap_v.norm(dim=-1).mean()).item()

        return {
            'extent': extent,
            'density': density,
            'curvature': curvature
        }