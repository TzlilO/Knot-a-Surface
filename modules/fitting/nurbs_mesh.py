"""
NURBS-native mesh generation for evaluation.

Provides direct tessellation of MultiSurfaceSplineModel without TSDF fusion.
Suitable for DTU benchmark evaluation pipeline.

Key optimizations:
- Chunked surface evaluation to avoid OOM
- Reuses existing cached samples when possible
- Memory-efficient triangle generation
"""

import torch
import torch.nn. functional as F
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
from tqdm import tqdm

# Avoid circular imports - these are used for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model.modules.fitting.multisurf import MultiSurfaceSplineModel
    from model.modules import SplineModel


@dataclass
class TessellationConfig:
    """Configuration for NURBS tessellation."""
    # Sampling density multiplier (relative to current Us, Vs)
    density_multiplier: float = 1.0

    # Maximum points to evaluate at once (prevents OOM)
    chunk_size: int = 4096

    # Mesh post-processing
    remove_degenerate:  bool = True
    compute_normals: bool = True

    # Multi-surface handling
    merge_surfaces: bool = True
    surface_gap_threshold: float = 0.001  # For stitching adjacent surfaces

    # Use existing cached samples if available (faster, less memory)
    use_cached_samples: bool = True


class NURBSMeshGenerator:
    """
    Generates triangle meshes directly from NURBS surfaces.

    Memory-efficient implementation using chunked evaluation.
    """

    def __init__(self, config: Optional[TessellationConfig] = None):
        self.config = config or TessellationConfig()

    @torch.no_grad()
    def generate_mesh(
        self,
        model: 'MultiSurfaceSplineModel',
        output_path: Optional[str] = None,
        return_components: bool = False,
        viewpoint_stack: Optional[List]=None,
    ) -> Union[o3d.geometry.TriangleMesh, Tuple[o3d.geometry. TriangleMesh, Dict]]:
        """
        Generate triangle mesh from MultiSurfaceSplineModel.

        Args:
            model: The NURBS model to tessellate
            output_path:  Optional path to save the mesh
            return_components: If True, also return per-surface meshes

        Returns:
            Combined mesh (and optionally per-surface components)
            :param viewpoint_stack:
        """
        surface_meshes = []
        surface_info = {}
        self.cameras=viewpoint_stack
        for surf_idx, (surface, active) in enumerate(
            zip(model.surfaces, model._active_surfaces)
        ):
            if not active:
                continue

            print(f"[NURBSMesh] Tessellating surface {surf_idx}...")

            # Generate mesh for this surface
            mesh, info = self._tessellate_surface(surface, surf_idx)

            if mesh is not None and len(mesh.vertices) > 0:
                surface_meshes. append(mesh)
                surface_info[surf_idx] = info

        # Combine surfaces
        if len(surface_meshes) == 0:
            combined = o3d.geometry.TriangleMesh()
        elif len(surface_meshes) == 1:
            combined = surface_meshes[0]
        else:
            combined = self._merge_surfaces(surface_meshes)

        # Post-process
        combined = self._post_process(combined)

        # Save if requested
        if output_path:
            o3d.io.write_triangle_mesh(
                output_path, combined,
                write_vertex_normals=True,
                write_vertex_colors=True
            )
            print(f"[NURBSMesh] Saved mesh to {output_path}")
            print(f"  Vertices: {len(combined.vertices)}")
            print(f"  Triangles: {len(combined.triangles)}")

        if return_components:
            return combined, {'surfaces': surface_meshes, 'info': surface_info}
        return combined

    @torch.no_grad()
    def _tessellate_surface(
        self,
        surface:  'SplineModel',
        surf_idx: int
    ) -> Tuple[o3d.geometry. TriangleMesh, Dict]:
        """
        Tessellate a single NURBS surface into triangles.

        Uses the existing cached samples to avoid expensive basis recomputation.
        """
        state = surface.state
        device = state.device
        return self._tessellate_from_basis(surface)
        # Option 1: Use existing cached/computed samples (memory efficient)
        # if self.config.use_cached_samples and surface.position. cache is not None:
        #     return self._tessellate_from_cache(surface, surf_idx)

        # Option 2: Evaluate surface on uniform grid (chunked to avoid OOM)
        # return self._tessellate_chunked(surface, surf_idx)

    def _tessellate_from_basis(
            self,
            surface: 'SplineModel',
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """
        Tessellate a single NURBS surface into triangles.

        Uses the UV grid structure to create a regular quad mesh,
        then triangulates each quad.
        """
        state = surface.state
        device = state.device

        # Determine tessellation density
        # if self.config.adaptive:
        #     Us, Vs = self._compute_adaptive_density(surface)
        # else:
        Us = int(state.Us * self.config.density_multiplier)
        Vs = int(state.Vs * self.config.density_multiplier)

        # # Generate UV samples
        # u_samples = torch.linspace(0, 1, Us, device=device)
        # v_samples = torch.linspace(0, 1, Vs, device=device)
        # uu, vv = torch.meshgrid(u_samples, v_samples, indexing='ij')
        # uv_grid = torch.stack([uu, vv], dim=-1).reshape(-1, 2)

        # Evaluate surface at UV points
        # Temporarily update basis for dense sampling
        # original_uv = surface.uv_sampler.get_uniform_grid()
        # surface.basis.forward(uv_grid, surface.knot_u(), surface.knot_v())
        surface.forward(self.cameras[0])
        xyz, normal, color = self._evaluate_surface(surface)
        # Get positions
        # xyz = surface.get_xyz.reshape(Us, Vs, 3)

        # Get normals from surface derivatives
        # normals = surface.surface_normals().reshape(Us, Vs, 3)

        # Get colors from SH (DC component)
        # from utils.sh_utils import SH2RGB
        # colors = SH2RGB(surface.get_features).reshape(Us, Vs, 3)

        # Restore original basis
        # surface.basis.forward(original_uv, surface.knot_u(), surface.knot_v())
        surface._invalidate_all_caches()

        # Create mesh
        mesh = self._create_quad_mesh(
            xyz,
            normal,
            color,
        )

        info = {
            'Us': Us,
            'Vs': Vs,
            'num_vertices': Us * Vs,
            'num_triangles': 2 * (Us - 1) * (Vs - 1)
        }

        return mesh, info
    @torch.no_grad()
    def _tessellate_from_cache(
        self,
        surface: 'SplineModel',
        surf_idx: int
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """
        Create mesh from already-computed surface samples.

        This is the most memory-efficient path - just uses existing data.
        """
        state = surface.state
        Us, Vs = state.Us, state.Vs

        # Get cached positions
        xyz = surface.get_xyz. reshape(Us, Vs, 3).cpu().numpy()

        # Get normals from rotation (smallest axis)
        normals = surface.surface_normals_raw.reshape(Us, Vs, 3).cpu().numpy()

        # Get colors from SH DC term
        colors = self._extract_colors_from_sh(surface, Us, Vs)

        # Create mesh
        mesh = self._create_quad_mesh(xyz, normals, colors)

        info = {
            'Us': Us,
            'Vs': Vs,
            'num_vertices': Us * Vs,
            'num_triangles': 2 * (Us - 1) * (Vs - 1),
            'method': 'cached'
        }

        return mesh, info

    @torch.no_grad()
    def _tessellate_chunked(
        self,
        surface: 'SplineModel',
        surf_idx: int
    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """
        Evaluate surface in chunks to avoid OOM.

        For when we need higher density than cached samples.
        """
        state = surface.state
        device = state.device

        # Target resolution
        base_Us, base_Vs = state. Us, state.Vs
        mult = self.config.density_multiplier
        Us = int(base_Us * mult)
        Vs = int(base_Vs * mult)

        # Clamp to reasonable size
        Us = min(Us, 512)
        Vs = min(Vs, 512)

        print(f"  Target grid: {Us} x {Vs} = {Us * Vs} points")

        # Generate uniform UV samples
        u_samples = torch.linspace(0, 1, Us, device=device)
        v_samples = torch. linspace(0, 1, Vs, device=device)

        # Allocate output arrays (on CPU to save GPU memory)
        xyz_grid = np.zeros((Us, Vs, 3), dtype=np.float32)
        normal_grid = np.zeros((Us, Vs, 3), dtype=np.float32)
        color_grid = np.zeros((Us, Vs, 3), dtype=np.float32)

        # Process in chunks (rows at a time)
        chunk_rows = max(1, self.config.chunk_size // Vs)

        for i_start in tqdm(range(0, Us, chunk_rows), desc="  Evaluating surface"):
            i_end = min(i_start + chunk_rows, Us)
            chunk_Us = i_end - i_start

            # Create UV grid for this chunk
            uu, vv = torch.meshgrid(
                u_samples[i_start:i_end],
                v_samples,
                indexing='ij'
            )
            uv_chunk = torch.stack([uu, vv], dim=-1).reshape(-1, 2)

            # Evaluate basis for this chunk
            # This is the memory-intensive part - we do it in small chunks
            xyz_chunk, normal_chunk, color_chunk = self._evaluate_surface_chunk(
                surface, uv_chunk, chunk_Us, Vs
            )

            # Store results
            xyz_grid[i_start:i_end] = xyz_chunk
            normal_grid[i_start:i_end] = normal_chunk
            color_grid[i_start: i_end] = color_chunk

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Create mesh
        mesh = self._create_quad_mesh(xyz_grid, normal_grid, color_grid)

        info = {
            'Us': Us,
            'Vs':  Vs,
            'num_vertices': Us * Vs,
            'num_triangles': 2 * (Us - 1) * (Vs - 1),
            'method': 'chunked'
        }

        return mesh, info
    @torch.no_grad()
    def _evaluate_surface_chunk(
        self,
        surface:  'SplineModel',
        uv_chunk: torch. Tensor,
        chunk_Us:  int,
        Vs: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate surface properties at UV coordinates.

        Returns numpy arrays to free GPU memory immediately.
        """
        device = uv_chunk.device
        state = surface.state
        H, W = state.H, state.W
        degree = state.degree

        # Get knot vectors
        knots_u = surface.knot_u()
        knots_v = surface.knot_v()

        # Compute basis functions for this chunk
        # Using the surface's basis computation
        from model.modules.basis import compute_1d_basis

        u_vals = surface.uv_sampler.interval_u # = uv_chunk[: , 0].reshape(chunk_Us, Vs)[:, 0]  # [chunk_Us]
        v_vals = surface.uv_sampler.interval_v # = uv_chunk[:, 1].reshape(chunk_Us, Vs)[0, :]  # [Vs]

        # Compute 1D basis functions
        Base_u = compute_1d_basis(u_vals, knots_u,  H, degree)  # [chunk_Us, H]
        Base_v = compute_1d_basis(v_vals, knots_v,  W, degree)  # [Vs, W]
        Bu = Base_u[0]
        Bv = Base_v[0]
        dBu = Base_u[1]
        dBv = Base_v[1]
        # Get control points
        ctrl_xyz = surface.position.cpts. reshape(H, W, 3)

        # Evaluate:  xyz[i,j] = sum_h sum_w Bu[i,h] * Bv[j,w] * ctrl[h,w]
        # Do this efficiently with einsum
        xyz = torch.einsum('ih,jw,hwc->ijc', Bu, Bv, ctrl_xyz)
        # Compute normals from tangent vectors
        # du = d/du S(u,v) = sum Bu'[i,h] * Bv[j,w] * ctrl[h,w]
        # dBu = compute_1d_basis(u_vals, knots_u, H, degree)[1]
        # dBv = compute_1d_basis(v_vals, knots_v, W, degree)[1]

        du = torch.einsum('ih,jw,hwc->ijc', dBu, Bv, ctrl_xyz)
        dv = torch.einsum('ih,jw,hwc->ijc', Bu, dBv, ctrl_xyz)

        normals = torch.cross(du, dv, dim=-1)
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        # Get colors from SH (evaluate SH control points)
        ctrl_sh = surface.spherical_harmonics. sh_dc. control_features.reshape(H, W, 3)
        sh_vals = torch.einsum('ih,jw,hwc->ijc', Bu, Bv, ctrl_sh)

        # Convert SH DC to RGB
        from utils.sh_utils import SH2RGB
        colors = SH2RGB(sh_vals).clamp(0, 1)

        # Move to CPU and convert to numpy
        xyz_np = xyz.cpu().numpy()
        normal_np = normals.cpu().numpy()
        color_np = colors.cpu().numpy()

        # Clean up GPU tensors
        del Bu, Bv, dBu, dBv, xyz, du, dv, normals, sh_vals, colors

        return xyz_np, normal_np, color_np
    @torch.no_grad()
    def _evaluate_surface(
        self,
        surface:  'SplineModel',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate surface properties at UV coordinates.

        Returns numpy arrays to free GPU memory immediately.
        """
        state = surface.state
        H, W = state.H, state.W
        degree = state.degree

        # Get knot vectors
        knots_u = surface.knot_u()
        knots_v = surface.knot_v()

        # Compute basis functions for this chunk
        # Using the surface's basis computation
        from model.modules.basis import compute_1d_basis

        u_vals = surface.uv_sampler.interval_u # = uv_chunk[: , 0].reshape(chunk_Us, Vs)[:, 0]  # [chunk_Us]
        v_vals = surface.uv_sampler.interval_v # = uv_chunk[:, 1].reshape(chunk_Us, Vs)[0, :]  # [Vs]

        # Compute 1D basis functions
        Base_u = compute_1d_basis(u_vals, knots_u,  H, degree)  # [chunk_Us, H]
        Base_v = compute_1d_basis(v_vals, knots_v,  W, degree)  # [Vs, W]
        Bu = Base_u[0]
        Bv = Base_v[0]
        dBu = Base_u[1]
        dBv = Base_v[1]
        # Get control points
        ctrl_xyz = surface.position.cpts. reshape(H, W, 3)

        # Evaluate:  xyz[i,j] = sum_h sum_w Bu[i,h] * Bv[j,w] * ctrl[h,w]
        # Do this efficiently with einsum
        xyz = torch.einsum('ih,jw,hwc->ijc', Bu, Bv, ctrl_xyz)
        # Compute normals from tangent vectors
        # du = d/du S(u,v) = sum Bu'[i,h] * Bv[j,w] * ctrl[h,w]
        # dBu = compute_1d_basis(u_vals, knots_u, H, degree)[1]
        # dBv = compute_1d_basis(v_vals, knots_v, W, degree)[1]

        du = torch.einsum('ih,jw,hwc->ijc', dBu, Bv, ctrl_xyz)
        dv = torch.einsum('ih,jw,hwc->ijc', Bu, dBv, ctrl_xyz)

        normals = torch.cross(du, dv, dim=-1)
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        # Get colors from SH (evaluate SH control points)
        ctrl_sh = surface.spherical_harmonics. sh_dc. control_features.reshape(H, W, 3)
        sh_vals = torch.einsum('ih,jw,hwc->ijc', Bu, Bv, ctrl_sh)

        # Convert SH DC to RGB
        from utils.sh_utils import SH2RGB
        colors = SH2RGB(sh_vals).clamp(0, 1)

        # Move to CPU and convert to numpy
        xyz_np = xyz.cpu().numpy()
        normal_np = normals.cpu().numpy()
        color_np = colors.cpu().numpy()

        # Clean up GPU tensors
        del Bu, Bv, dBu, dBv, xyz, du, dv, normals, sh_vals, colors

        return xyz_np, normal_np, color_np

    def _extract_colors_from_sh(
        self,
        surface: 'SplineModel',
        Us: int,
        Vs:  int
    ) -> np.ndarray:
        """
        Extract vertex colors from SH coefficients (DC term).
        """
        from utils.sh_utils import SH2RGB

        # Get SH features - use cached if available
        if surface. spherical_harmonics.sh_dc.cache is not None:
            sh_dc = surface.spherical_harmonics.sh_dc.cache
        else:
            sh_dc = surface.spherical_harmonics. sh_dc.interpolate_samples()

        sh_dc = sh_dc.reshape(Us, Vs, -1)

        # Convert DC to RGB
        colors = SH2RGB(sh_dc[..., : 3])
        colors = colors.clamp(0, 1).cpu().numpy()

        return colors

    def _create_quad_mesh(
        self,
        vertices: np.ndarray,  # [Us, Vs, 3]
        normals: np.ndarray,   # [Us, Vs, 3]
        colors:  np.ndarray     # [Us, Vs, 3]
    ) -> o3d.geometry.TriangleMesh:
        """
        Create triangle mesh from UV grid by triangulating quads.
        """
        Us, Vs, _ = vertices.shape

        # Flatten vertices
        verts_flat = vertices.reshape(-1, 3)
        norms_flat = normals.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)

        # Create triangle indices
        # Each quad (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j) becomes 2 triangles
        triangles = []
        for i in range(Us - 1):
            for j in range(Vs - 1):
                # Vertex indices for this quad
                v00 = i * Vs + j
                v01 = i * Vs + (j + 1)
                v10 = (i + 1) * Vs + j
                v11 = (i + 1) * Vs + (j + 1)

                # Two triangles per quad
                triangles.append([v00, v01, v11])
                triangles.append([v00, v11, v10])

        triangles = np.array(triangles, dtype=np.int32)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility. Vector3dVector(verts_flat. astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms_flat.astype(np.float64))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float64))

        return mesh

    def _merge_surfaces(
        self,
        meshes: List[o3d. geometry.TriangleMesh]
    ) -> o3d.geometry.TriangleMesh:
        """
        Merge multiple surface meshes.
        """
        if len(meshes) == 0:
            return o3d.geometry.TriangleMesh()

        if len(meshes) == 1:
            return meshes[0]

        # Simple concatenation
        combined = meshes[0]
        for m in meshes[1:]:
            combined += m

        # Optionally merge close vertices
        if self.config.surface_gap_threshold > 0:
            combined = combined.merge_close_vertices(self.config.surface_gap_threshold)

        return combined

    def _post_process(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """
        Post-process the mesh.
        """
        if self.config.remove_degenerate:
            mesh. remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()

        if self.config.compute_normals:
            mesh. compute_vertex_normals()

        return mesh


def compute_basis_1d(
    params: torch.Tensor,  # [N] parameter values
    knots: torch. Tensor,   # [K] knot vector
    degree: int,
    n_ctrl: int,
    derivative: int = 0
) -> torch.Tensor:
    """
    Compute 1D B-spline basis functions.

    Uses Cox-de Boor recursion.

    Args:
        params: Parameter values to evaluate at
        knots: Knot vector
        degree: Polynomial degree
        n_ctrl: Number of control points
        derivative: Derivative order (0 = value, 1 = first derivative)

    Returns:
        Basis matrix [N, n_ctrl]
    """
    N = len(params)
    device = params.device

    # Initialize basis (degree 0)
    # B_i,0(u) = 1 if knots[i] <= u < knots[i+1], else 0
    basis = torch.zeros(N, len(knots) - 1, device=device)

    for i in range(len(knots) - 1):
        mask = (params >= knots[i]) & (params < knots[i + 1])
        basis[:, i] = mask.float()

    # Handle right endpoint
    basis[:, -1] = (params == knots[-1]).float()

    # Cox-de Boor recursion
    for d in range(1, degree + 1):
        new_basis = torch.zeros(N, len(knots) - 1 - d, device=device)

        for i in range(len(knots) - 1 - d):
            # Left term
            denom1 = knots[i + d] - knots[i]
            if denom1 > 1e-10:
                left = (params - knots[i]) / denom1 * basis[:, i]
            else:
                left = torch.zeros(N, device=device)

            # Right term
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 > 1e-10:
                right = (knots[i + d + 1] - params) / denom2 * basis[:, i + 1]
            else:
                right = torch.zeros(N, device=device)

            new_basis[:, i] = left + right

        basis = new_basis

    # Handle derivative if requested
    if derivative > 0:
        # For derivative, use the formula:
        # B'_i,p(u) = p * (B_{i,p-1}(u) / (knots[i+p] - knots[i])
        #                 - B_{i+1,p-1}(u) / (knots[i+p+1] - knots[i+1]))

        # Recompute basis of degree-1
        basis_lower = torch.zeros(N, len(knots) - 1, device=device)
        for i in range(len(knots) - 1):
            mask = (params >= knots[i]) & (params < knots[i + 1])
            basis_lower[:, i] = mask.float()
        basis_lower[:, -1] = (params == knots[-1]).float()

        for d in range(1, degree):
            new_basis = torch.zeros(N, len(knots) - 1 - d, device=device)
            for i in range(len(knots) - 1 - d):
                denom1 = knots[i + d] - knots[i]
                if denom1 > 1e-10:
                    left = (params - knots[i]) / denom1 * basis_lower[:, i]
                else:
                    left = torch.zeros(N, device=device)

                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 > 1e-10:
                    right = (knots[i + d + 1] - params) / denom2 * basis_lower[:, i + 1]
                else:
                    right = torch.zeros(N, device=device)

                new_basis[:, i] = left + right
            basis_lower = new_basis

        # Now compute derivative
        deriv_basis = torch.zeros(N, n_ctrl, device=device)
        for i in range(n_ctrl):
            denom1 = knots[i + degree] - knots[i]
            denom2 = knots[i + degree + 1] - knots[i + 1]

            term1 = basis_lower[:, i] / denom1 if denom1 > 1e-10 else torch.zeros(N, device=device)
            term2 = basis_lower[: , i + 1] / denom2 if denom2 > 1e-10 and i + 1 < basis_lower.shape[1] else torch.zeros(N, device=device)

            deriv_basis[:, i] = degree * (term1 - term2)

        return deriv_basis

    # Return only the relevant columns
    return basis[:, : n_ctrl]


def extract_mesh_from_nurbs(
    model: 'MultiSurfaceSplineModel',
    output_path: str,
    density_multiplier: float = 1.0,
    use_cached: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Convenience function for mesh extraction.

    Args:
        model: MultiSurfaceSplineModel instance
        output_path: Where to save the mesh
        density_multiplier: Tessellation density relative to current sampling
        use_cached: Use cached surface samples (faster, more memory efficient)

    Returns:
        The generated mesh
    """
    config = TessellationConfig(
        density_multiplier=density_multiplier,
        use_cached_samples=use_cached,
        chunk_size=4096  # Adjust based on your GPU memory
    )
    generator = NURBSMeshGenerator(config)
    return generator.generate_mesh(model, output_path)