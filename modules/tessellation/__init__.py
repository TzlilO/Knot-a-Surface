# """
# Tessellation module for B-Spline/NURBS surfaces.
#
# Provides:
# - GridTessellator: Basic grid-to-triangle mesh conversion
# - ViewDependentTessellator: Chhugani-style adaptive UV sampling
# - TriangleMesh, TessellationConfig: Data structures
# """
#
# from .view_dependent import (
#     ViewDependentTessellator,
#     TessellationParams,
#     RefinementCriterion,
# )
#
# __all__ = [
#     'GridTessellator',
#     'TriangleMesh',
#     'TessellationConfig',
#     'mesh_to_obj',
#     'mesh_to_ply',
#     'ViewDependentTessellator',
#     'TessellationParams',
#     'RefinementCriterion',
# ]
#
# """
# Differentiable Triangle Mesh Tessellation for NURBS Surfaces.
#
# Exploits the regular grid structure of sampled surface points to create
# efficient, differentiable triangle meshes without explicit connectivity search.
#
# Key advantages:
# - O(1) per-quad connectivity (no search required)
# - Fully differentiable (gradients flow through vertices)
# - Memory efficient (indices are integer, only vertices need gradients)
# - Supports per-vertex attributes (colors, normals, UVs)
# """
#
# import torch
# import torch.nn as nn
# from typing import Tuple, Optional, Dict, NamedTuple, List
# from dataclasses import dataclass
#
#
# class Triangle(NamedTuple):
#     """Container for triangle mesh data."""
#     vertices: torch.Tensor  # [N, 3] vertex positions
#     faces: torch.Tensor  # [F, 3] face indices (int64)
#     vertex_normals: Optional[torch.Tensor] = None  # [N, 3]
#     vertex_colors: Optional[torch.Tensor] = None  # [N, 3] or [N, C]
#     vertex_uvs: Optional[torch.Tensor] = None  # [N, 2]
#     face_normals: Optional[torch.Tensor] = None  # [F, 3]
#
#     @property
#     def num_vertices(self) -> int:
#         return self.vertices.shape[0]
#
#     @property
#     def num_faces(self) -> int:
#         return self.faces.shape[0]
#
#     def to(self, device: str) -> 'TriangleMesh':
#         """Move mesh to device."""
#         return TriangleMesh(
#             vertices=self.vertices.to(device),
#             faces=self.faces.to(device),
#             vertex_normals=self.vertex_normals.to(device) if self.vertex_normals is not None else None,
#             vertex_colors=self.vertex_colors.to(device) if self.vertex_colors is not None else None,
#             vertex_uvs=self.vertex_uvs.to(device) if self.vertex_uvs is not None else None,
#             face_normals=self.face_normals.to(device) if self.face_normals is not None else None,
#         )
#
#
# @dataclass
# # class TessellationConfig:
# #     """Configuration for mesh tessellation."""
# #     # Diagonal split strategy
# #     split_strategy: str = 'shorter'  # 'shorter', 'longer', 'consistent', 'alternating'
# #
# #     # Normal computation
# #     compute_vertex_normals: bool = True
# #     compute_face_normals: bool = True
# #     normal_weighting: str = 'area'  # 'area', 'angle', 'uniform'
# #
# #     # Degenerate triangle handling
# #     min_triangle_area: float = 1e-10
# #     remove_degenerate: bool = False  # If True, filter out degenerate triangles
# #
# #     # UV computation
# #     compute_uvs: bool = True
#
#
# class GridTessellator:
#     """
#     Efficient tessellator for regular grid surfaces.
#
#     Given a [Us, Vs, 3] grid of surface points, creates a triangle mesh
#     by connecting adjacent quads into triangle pairs.
#
#     Quad layout (looking at UV plane):
#
#         v -->
#       u  (i,j)-----(i,j+1)
#       |    |    /    |
#       v    |   /     |
#          (i+1,j)---(i+1,j+1)
#
#     Each quad produces 2 triangles.  The diagonal choice affects mesh quality.
#     """
#
#     def __init__(self, config: Optional[TessellationConfig] = None):
#         self.config = config or TessellationConfig()
#
#         # Cache for face indices (regenerate only when grid size changes)
#         self._cached_faces: Optional[torch.Tensor] = None
#         self._cached_grid_shape: Optional[Tuple[int, int]] = None
#
#     def tessellate(
#             self,
#             vertices_grid: torch.Tensor,  # [Us, Vs, 3]
#             colors_grid: Optional[torch.Tensor] = None,  # [Us, Vs, 3]
#             normals_grid: Optional[torch.Tensor] = None,  # [Us, Vs, 3] pre-computed
#     ) -> TriangleMesh:
#         """
#         Convert a regular grid of vertices to a triangle mesh.
#
#         Args:
#             vertices_grid:  [Us, Vs, 3] surface point positions
#             colors_grid: [Us, Vs, C] optional per-vertex colors
#             normals_grid: [Us, Vs, 3] optional pre-computed normals
#
#         Returns:
#             TriangleMesh with vertices, faces, and optional attributes
#         """
#         Us, Vs, _ = vertices_grid.shape
#         device = vertices_grid.device
#
#         # Flatten vertices to [N, 3] where N = Us * Vs
#         vertices = vertices_grid.reshape(-1, 3)
#
#         # Get or compute face indices
#         faces = self._get_face_indices(Us, Vs, vertices_grid, device)
#
#         # Compute UVs if requested
#         vertex_uvs = None
#         if self.config.compute_uvs:
#             vertex_uvs = self._compute_vertex_uvs(Us, Vs, device)
#
#         # Compute face normals if requested
#         face_normals = None
#         if self.config.compute_face_normals:
#             face_normals = self._compute_face_normals(vertices, faces)
#
#         # Compute or use provided vertex normals
#         vertex_normals = None
#         if normals_grid is not None:
#             vertex_normals = normals_grid.reshape(-1, 3)
#         elif self.config.compute_vertex_normals:
#             if face_normals is None:
#                 face_normals = self._compute_face_normals(vertices, faces)
#             vertex_normals = self._compute_vertex_normals(
#                 vertices, faces, face_normals, Us, Vs
#             )
#
#         # Flatten colors if provided
#         vertex_colors = None
#         if colors_grid is not None:
#             vertex_colors = colors_grid.reshape(-1, colors_grid.shape[-1])
#
#         # Optionally filter degenerate triangles
#         if self.config.remove_degenerate and face_normals is not None:
#             faces, face_normals = self._filter_degenerate(
#                 vertices, faces, face_normals
#             )
#
#         return TriangleMesh(
#             vertices=vertices,
#             faces=faces,
#             vertex_normals=vertex_normals,
#             vertex_colors=vertex_colors,
#             vertex_uvs=vertex_uvs,
#             face_normals=face_normals,
#         )
#
#     def _get_face_indices(
#             self,
#             Us: int,
#             Vs: int,
#             vertices_grid: torch.Tensor,
#             device: torch.device
#     ) -> torch.Tensor:
#         """
#         Generate face indices for the grid.
#
#         Uses caching since indices only depend on grid dimensions.
#         """
#         # Check cache
#         if (self._cached_faces is not None and
#                 self._cached_grid_shape == (Us, Vs) and
#                 self._cached_faces.device == device):
#             return self._cached_faces
#
#         # Number of quads
#         num_quads_u = Us - 1
#         num_quads_v = Vs - 1
#         num_quads = num_quads_u * num_quads_v
#         num_faces = num_quads * 2
#
#         # Create index grids for quad corners
#         # For quad (i, j), the four corners are:
#         #   top-left:      i * Vs + j
#         #   top-right:    i * Vs + (j+1)
#         #   bottom-left:  (i+1) * Vs + j
#         #   bottom-right: (i+1) * Vs + (j+1)
#
#         i_indices = torch.arange(num_quads_u, device=device)
#         j_indices = torch.arange(num_quads_v, device=device)
#
#         # Create meshgrid of quad indices
#         ii, jj = torch.meshgrid(i_indices, j_indices, indexing='ij')
#         ii = ii.reshape(-1)  # [num_quads]
#         jj = jj.reshape(-1)  # [num_quads]
#
#         # Compute vertex indices for each quad corner
#         idx_tl = ii * Vs + jj  # top-left
#         idx_tr = ii * Vs + (jj + 1)  # top-right
#         idx_bl = (ii + 1) * Vs + jj  # bottom-left
#         idx_br = (ii + 1) * Vs + (jj + 1)  # bottom-right
#
#         # Choose diagonal based on strategy
#         if self.config.split_strategy == 'consistent':
#             # Always split along same diagonal:  TL-BR
#             # Triangle 1: TL, BL, BR
#             # Triangle 2: TL, BR, TR
#             faces = torch.stack([
#                 torch.stack([idx_tl, idx_bl, idx_br], dim=1),
#                 torch.stack([idx_tl, idx_br, idx_tr], dim=1),
#             ], dim=1).reshape(-1, 3)
#
#         elif self.config.split_strategy == 'alternating':
#             # Checkerboard pattern for more isotropic mesh
#             checker = ((ii + jj) % 2).bool()
#
#             # Diagonal 1 (TL-BR): TL-BL-BR, TL-BR-TR
#             # Diagonal 2 (TR-BL): TL-BL-TR, BL-BR-TR
#
#             faces_diag1_t1 = torch.stack([idx_tl, idx_bl, idx_br], dim=1)
#             faces_diag1_t2 = torch.stack([idx_tl, idx_br, idx_tr], dim=1)
#
#             faces_diag2_t1 = torch.stack([idx_tl, idx_bl, idx_tr], dim=1)
#             faces_diag2_t2 = torch.stack([idx_bl, idx_br, idx_tr], dim=1)
#
#             t1 = torch.where(checker.unsqueeze(1), faces_diag1_t1, faces_diag2_t1)
#             t2 = torch.where(checker.unsqueeze(1), faces_diag1_t2, faces_diag2_t2)
#
#             faces = torch.stack([t1, t2], dim=1).reshape(-1, 3)
#
#         elif self.config.split_strategy in ['shorter', 'longer']:
#             # Choose diagonal based on edge length
#             # Requires actual vertex positions
#             v_tl = vertices_grid[:-1, :-1].reshape(-1, 3)
#             v_tr = vertices_grid[:-1, 1:].reshape(-1, 3)
#             v_bl = vertices_grid[1:, :-1].reshape(-1, 3)
#             v_br = vertices_grid[1:, 1:].reshape(-1, 3)
#
#             # Diagonal lengths
#             diag1_len = (v_tl - v_br).norm(dim=-1)  # TL-BR
#             diag2_len = (v_tr - v_bl).norm(dim=-1)  # TR-BL
#
#             if self.config.split_strategy == 'shorter':
#                 use_diag1 = diag1_len <= diag2_len
#             else:
#                 use_diag1 = diag1_len > diag2_len
#
#             # Same logic as alternating but based on diagonal length
#             faces_diag1_t1 = torch.stack([idx_tl, idx_bl, idx_br], dim=1)
#             faces_diag1_t2 = torch.stack([idx_tl, idx_br, idx_tr], dim=1)
#
#             faces_diag2_t1 = torch.stack([idx_tl, idx_bl, idx_tr], dim=1)
#             faces_diag2_t2 = torch.stack([idx_bl, idx_br, idx_tr], dim=1)
#
#             t1 = torch.where(use_diag1.unsqueeze(1), faces_diag1_t1, faces_diag2_t1)
#             t2 = torch.where(use_diag1.unsqueeze(1), faces_diag1_t2, faces_diag2_t2)
#
#             faces = torch.stack([t1, t2], dim=1).reshape(-1, 3)
#
#             # Don't cache when using geometry-dependent splitting
#             return faces.long()
#
#         else:
#             raise ValueError(f"Unknown split_strategy: {self.config.split_strategy}")
#
#         # Cache result
#         self._cached_faces = faces.long()
#         self._cached_grid_shape = (Us, Vs)
#
#         return self._cached_faces
#
#     def _compute_face_normals(
#             self,
#             vertices: torch.Tensor,  # [N, 3]
#             faces: torch.Tensor,  # [F, 3]
#     ) -> torch.Tensor:
#         """
#         Compute per-face normals using cross product.
#
#         Differentiable with respect to vertices.
#         """
#         # Gather vertices for each face
#         v0 = vertices[faces[:, 0]]  # [F, 3]
#         v1 = vertices[faces[:, 1]]  # [F, 3]
#         v2 = vertices[faces[:, 2]]  # [F, 3]
#
#         # Edge vectors
#         e1 = v1 - v0
#         e2 = v2 - v0
#
#         # Cross product gives normal direction
#         normals = torch.cross(e1, e2, dim=-1)
#
#         # Normalize (handle degenerate triangles)
#         norms = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         normals = normals / norms
#
#         return normals
#
#     def _compute_vertex_normals(
#             self,
#             vertices: torch.Tensor,  # [N, 3]
#             faces: torch.Tensor,  # [F, 3]
#             face_normals: torch.Tensor,  # [F, 3]
#             Us: int,
#             Vs: int,
#     ) -> torch.Tensor:
#         """
#         Compute per-vertex normals by averaging adjacent face normals.
#
#         Exploits grid structure for efficient computation.
#         """
#         N = vertices.shape[0]
#         F = faces.shape[0]
#         device = vertices.device
#
#         if self.config.normal_weighting == 'uniform':
#             weights = torch.ones(F, device=device)
#         elif self.config.normal_weighting == 'area':
#             # Weight by triangle area (proportional to cross product magnitude)
#             v0 = vertices[faces[:, 0]]
#             v1 = vertices[faces[:, 1]]
#             v2 = vertices[faces[:, 2]]
#             e1 = v1 - v0
#             e2 = v2 - v0
#             weights = torch.cross(e1, e2, dim=-1).norm(dim=-1) * 0.5
#         elif self.config.normal_weighting == 'angle':
#             # Weight by angle at each vertex (more complex, skip for now)
#             weights = torch.ones(F, device=device)
#         else:
#             weights = torch.ones(F, device=device)
#
#         # Weighted normals
#         weighted_normals = face_normals * weights.unsqueeze(-1)  # [F, 3]
#
#         # Scatter-add to vertices
#         vertex_normals = torch.zeros(N, 3, device=device)
#
#         # Each face contributes to 3 vertices
#         for i in range(3):
#             vertex_normals.scatter_add_(
#                 0,
#                 faces[:, i: i + 1].expand(-1, 3),
#                 weighted_normals
#             )
#
#         # Normalize
#         norms = vertex_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         vertex_normals = vertex_normals / norms
#
#         return vertex_normals
#
#     def _compute_vertex_uvs(
#             self,
#             Us: int,
#             Vs: int,
#             device: torch.device
#     ) -> torch.Tensor:
#         """
#         Compute UV coordinates for grid vertices.
#         """
#         u_coords = torch.linspace(0, 1, Us, device=device)
#         v_coords = torch.linspace(0, 1, Vs, device=device)
#
#         uu, vv = torch.meshgrid(u_coords, v_coords, indexing='ij')
#         uvs = torch.stack([uu, vv], dim=-1).reshape(-1, 2)
#
#         return uvs
#
#     def _filter_degenerate(
#             self,
#             vertices: torch.Tensor,
#             faces: torch.Tensor,
#             face_normals: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Remove degenerate (zero-area) triangles.
#         """
#         # Compute areas
#         v0 = vertices[faces[:, 0]]
#         v1 = vertices[faces[:, 1]]
#         v2 = vertices[faces[:, 2]]
#         e1 = v1 - v0
#         e2 = v2 - v0
#         areas = torch.cross(e1, e2, dim=-1).norm(dim=-1) * 0.5
#
#         # Filter
#         valid_mask = areas > self.config.min_triangle_area
#
#         return faces[valid_mask], face_normals[valid_mask]
#
#     def invalidate_cache(self):
#         """Clear cached face indices."""
#         self._cached_faces = None
#         self._cached_grid_shape = None
#
#
# class SplineModelTessellator(nn.Module):
#     """
#     Tessellation module integrated with SplineModel.
#
#     Provides differentiable mesh generation from the surface representation.
#     """
#
#     def __init__(
#             self,
#             spline_model: 'SplineModel',
#             config: Optional[TessellationConfig] = None
#     ):
#         super().__init__()
#         self.spline_model = spline_model
#         self.tessellator = GridTessellator(config or TessellationConfig())
#
#     @property
#     def state(self):
#         return self.spline_model.state
#
#     def forward(
#             self,
#             camera=None,
#             include_colors: bool = True,
#             use_geometric_normals: bool = True,
#     ) -> TriangleMesh:
#         """
#         Generate triangle mesh from current surface state.
#
#         Args:
#             camera: Optional camera for normal orientation
#             include_colors: Include vertex colors from SH
#             use_geometric_normals:  Use cross-product normals vs rotation-based
#
#         Returns:
#             TriangleMesh with differentiable vertices
#         """
#         Us, Vs = self.state.Us, self.state.Vs
#
#         # Get surface points [Us*Vs, 3] -> [Us, Vs, 3]
#         xyz = self.spline_model.get_xyz.reshape(Us, Vs, 3)
#
#         # Get colors if requested
#         colors = None
#         if include_colors:
#             # Use DC component of SH as color
#             sh_features = self.spline_model.get_features  # [N, SH, 3]
#             colors = sh_features[:, 0, :].reshape(Us, Vs, 3)  # DC component
#             colors = torch.sigmoid(colors)  # Ensure [0, 1] range
#
#         # Get normals
#         normals = None
#         if use_geometric_normals:
#             # Compute from surface tangents
#             du = self.spline_model.position.du.reshape(Us, Vs, 3)
#             dv = self.spline_model.position.dv.reshape(Us, Vs, 3)
#             normals = torch.cross(du, dv, dim=-1)
#             normals = torch.nn.functional.normalize(normals, dim=-1)
#
#             # Orient toward camera if provided
#             if camera is not None:
#                 view_dirs = camera.camera_center - xyz
#                 dot = (normals * view_dirs).sum(dim=-1, keepdim=True)
#                 normals = torch.where(dot < 0, -normals, normals)
#
#         return self.tessellator.tessellate(xyz, colors, normals)
#
#     def get_mesh_at_resolution(
#             self,
#             resolution: Tuple[int, int],
#             camera=None,
#     ) -> TriangleMesh:
#         """
#         Generate mesh at specific UV resolution (different from control grid).
#
#         Useful for export or rendering at different detail levels.
#         """
#         Us, Vs = resolution
#         device = self.spline_model.device
#
#         # Create uniform UV grid at target resolution
#         u_samples = torch.linspace(0, 1, Us, device=device)
#         v_samples = torch.linspace(0, 1, Vs, device=device)
#         uu, vv = torch.meshgrid(u_samples, v_samples, indexing='ij')
#         uv_grid = torch.stack([uu, vv], dim=-1)  # [Us, Vs, 2]
#
#         # Evaluate surface at new resolution
#         # This requires recomputing basis functions
#         from modules.basis import BasisFunction
#
#         temp_basis = BasisFunction(self.state)
#         temp_basis.forward(
#             uv_grid.reshape(-1, 2),
#             self.spline_model.knot_u(),
#             self.spline_model.knot_v()
#         )
#
#         # Interpolate position
#         xyz = self.spline_model.position._interpolate_with_basis(temp_basis)
#         xyz = xyz.reshape(Us, Vs, 3)
#
#         # Interpolate colors (SH DC)
#         colors = self.spline_model.spherical_harmonics.sh_dc._interpolate_with_basis(temp_basis)
#         colors = torch.sigmoid(colors.reshape(Us, Vs, 3))
#
#         # Compute geometric normals from finite differences
#         du = torch.zeros_like(xyz)
#         dv = torch.zeros_like(xyz)
#         du[:-1] = xyz[1:] - xyz[:-1]
#         du[-1] = du[-2]
#         dv[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
#         dv[:, -1] = dv[:, -2]
#
#         normals = torch.cross(du, dv, dim=-1)
#         normals = torch.nn.functional.normalize(normals, dim=-1)
#
#         if camera is not None:
#             view_dirs = camera.camera_center - xyz
#             dot = (normals * view_dirs).sum(dim=-1, keepdim=True)
#             normals = torch.where(dot < 0, -normals, normals)
#
#         return self.tessellator.tessellate(xyz, colors, normals)
#
#
# # =============================================================================
# # Export Utilities
# # =============================================================================
#
# def mesh_to_obj(
#         mesh: TriangleMesh,
#         filepath: str,
#         include_normals: bool = True,
#         include_uvs: bool = True,
#         include_colors: bool = True,
# ):
#     """
#     Export TriangleMesh to OBJ format.
#
#     Args:
#         mesh: TriangleMesh to export
#         filepath: Output path (should end in .obj)
#         include_normals: Include vertex normals
#         include_uvs: Include texture coordinates
#         include_colors: Include vertex colors (as comments or MTL)
#     """
#     with open(filepath, 'w') as f:
#         f.write(f"# Triangle mesh with {mesh.num_vertices} vertices, {mesh.num_faces} faces\n")
#
#         # Write vertices (with optional colors as vertex colors extension)
#         vertices = mesh.vertices.detach().cpu().numpy()
#         if include_colors and mesh.vertex_colors is not None:
#             colors = mesh.vertex_colors.detach().cpu().numpy()
#             for i, (v, c) in enumerate(zip(vertices, colors)):
#                 f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
#         else:
#             for v in vertices:
#                 f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
#
#         # Write texture coordinates
#         if include_uvs and mesh.vertex_uvs is not None:
#             uvs = mesh.vertex_uvs.detach().cpu().numpy()
#             for uv in uvs:
#                 f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
#
#         # Write normals
#         if include_normals and mesh.vertex_normals is not None:
#             normals = mesh.vertex_normals.detach().cpu().numpy()
#             for n in normals:
#                 f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
#
#         # Write faces (OBJ uses 1-based indexing)
#         faces = mesh.faces.detach().cpu().numpy()
#         has_uvs = include_uvs and mesh.vertex_uvs is not None
#         has_normals = include_normals and mesh.vertex_normals is not None
#
#         for face in faces:
#             if has_uvs and has_normals:
#                 f.write(f"f {face[0] + 1}/{face[0] + 1}/{face[0] + 1} "
#                         f"{face[1] + 1}/{face[1] + 1}/{face[1] + 1} "
#                         f"{face[2] + 1}/{face[2] + 1}/{face[2] + 1}\n")
#             elif has_normals:
#                 f.write(f"f {face[0] + 1}//{face[0] + 1} "
#                         f"{face[1] + 1}//{face[1] + 1} "
#                         f"{face[2] + 1}//{face[2] + 1}\n")
#             elif has_uvs:
#                 f.write(f"f {face[0] + 1}/{face[0] + 1} "
#                         f"{face[1] + 1}/{face[1] + 1} "
#                         f"{face[2] + 1}/{face[2] + 1}\n")
#             else:
#                 f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
#
#
# def mesh_to_ply(
#         mesh: TriangleMesh,
#         filepath: str,
#         binary: bool = True,
# ):
#     """
#     Export TriangleMesh to PLY format.
#     """
#     import numpy as np
#
#     vertices = mesh.vertices.detach().cpu().numpy()
#     faces = mesh.faces.detach().cpu().numpy()
#
#     has_colors = mesh.vertex_colors is not None
#     has_normals = mesh.vertex_normals is not None
#
#     # Build header
#     header = [
#         "ply",
#         f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
#         f"element vertex {mesh.num_vertices}",
#         "property float x",
#         "property float y",
#         "property float z",
#     ]
#
#     if has_normals:
#         header.extend([
#             "property float nx",
#             "property float ny",
#             "property float nz",
#         ])
#
#     if has_colors:
#         header.extend([
#             "property uchar red",
#             "property uchar green",
#             "property uchar blue",
#         ])
#
#     header.extend([
#         f"element face {mesh.num_faces}",
#         "property list uchar int vertex_indices",
#         "end_header",
#     ])
#
#     with open(filepath, 'wb' if binary else 'w') as f:
#         header_str = '\n'.join(header) + '\n'
#
#         if binary:
#             f.write(header_str.encode('ascii'))
#
#             # Prepare vertex data
#             vertex_data = [vertices]
#             if has_normals:
#                 vertex_data.append(mesh.vertex_normals.detach().cpu().numpy())
#
#             vertex_array = np.hstack(vertex_data).astype(np.float32)
#
#             if has_colors:
#                 colors = (mesh.vertex_colors.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
#                 # Write vertex by vertex for mixed types
#                 for i in range(mesh.num_vertices):
#                     f.write(vertex_array[i].tobytes())
#                     f.write(colors[i].tobytes())
#             else:
#                 f.write(vertex_array.tobytes())
#
#             # Write faces
#             for face in faces:
#                 f.write(np.array([3], dtype=np.uint8).tobytes())
#                 f.write(face.astype(np.int32).tobytes())
#         else:
#             f.write(header_str)
#
#             normals = mesh.vertex_normals.detach().cpu().numpy() if has_normals else None
#             colors = (mesh.vertex_colors.detach().cpu().numpy() * 255).clip(0, 255).astype(int) if has_colors else None
#
#             for i in range(mesh.num_vertices):
#                 line = f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}"
#                 if has_normals:
#                     line += f" {normals[i, 0]:.6f} {normals[i, 1]:. 6f} {normals[i, 2]:.6f}"
#                 if has_colors:
#                     line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
#                 f.write(line + '\n')
#
#             for face in faces:
#                 f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
#
#
# # =============================================================================
# # Integration with SplineModel
# # =============================================================================
#
# def add_tessellation_to_spline_model(SplineModelClass):
#     """
#     Decorator/mixin to add tessellation methods to SplineModel.
#     """
#
#     def tessellate(self, camera=None, **kwargs) -> TriangleMesh:
#         """Generate triangle mesh from current surface state."""
#         if not hasattr(self, '_tessellator'):
#             self._tessellator = GridTessellator()
#
#         Us, Vs = self.state.Us, self.state.Vs
#         xyz = self.get_xyz.reshape(Us, Vs, 3)
#
#         # Colors from SH DC
#         colors = self.get_features[:, 0, :].reshape(Us, Vs, 3)
#         colors = torch.sigmoid(colors)
#
#         # Normals
#         normals = self.surface_normals_raw.reshape(Us, Vs, 3)
#         if camera is not None:
#             normals = self.get_normal(camera).reshape(Us, Vs, 3)
#
#         return self._tessellator.tessellate(xyz, colors, normals)
#
#     def export_mesh(self, filepath: str, camera=None, format: str = 'obj', **kwargs):
#         """Export current surface as triangle mesh."""
#         mesh = self.tessellate(camera, **kwargs)
#
#         if format.lower() == 'obj':
#             mesh_to_obj(mesh, filepath, **kwargs)
#         elif format.lower() == 'ply':
#             mesh_to_ply(mesh, filepath, **kwargs)
#         else:
#             raise ValueError(f"Unknown format: {format}")
#
#     SplineModelClass.tessellate = tessellate
#     SplineModelClass.export_mesh = export_mesh
#
#     return SplineModelClass