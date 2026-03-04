
import gc
import os
import numpy as _np

from model.schedulers.utils import plot_lr_schedule

# restore old aliases
_np.bool   = bool
_np.int    = int
_np.float  = float
_np.object = object
# etc., if you hit more
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from plyfile import PlyElement, PlyData
from torch import nn
from pytorch3d.loss.chamfer import chamfer_distance
from geomdl import BSpline, fitting
from geomdl import knotvector
import torch.nn.functional as F
from arguments import ModelParams, NurbsOptimizationParams
from model.Spline import SplineMLP
from model.spline_utils import Gaussians, stitch_control_features, grid_to_patches, grid_upscale, \
    get_basis_functions, generate_bspline_surface_grid, process_feature_grid, get_parameter_vectors, \
    normals_to_quaternions, subdivide_uv_params, subdivision, make_clamped_uniform_knots2, \
    subdivide_patch_batch, compare_split_unsplit, refine_mask_morphology, bernstein_basis_4, bernstein_derivative_4, \
    generate_global_grid, basis_eval, basis_eval_derivative, refine_mask_morphology_diff, clean_bad_faces, \
    insert_knot_surface_v, insert_knot_surface_u
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from utils.general_utils import (
    get_expon_lr_func, strip_symmetric, build_scaling_rotation, inverse_sigmoid,
)
from utils.sh_utils import SH2RGB, RGB2SH
import opt_einsum as oe

import torch


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.
    Assumes q1 and q2 are tensors of shape (..., 4) and w,x,y,z convention.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def quaternion_conjugate(q):
    """
    Computes the conjugate of a quaternion.
    Assumes q is a tensor of shape (..., 4) and w,x,y,z convention.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack((w, -x, -y, -z), dim=-1)


def rotate_vector_by_quaternion(v, q):
    """
    Rotates a vector v by a quaternion q.
    v: torch.Tensor of shape (N, 3)
    q: torch.Tensor of shape (N, 4) (w, x, y, z)
    """
    # Ensure q is normalized
    q = F.normalize(q, dim=-1)

    # Represent the vector v as a pure quaternion (0, vx, vy, vz)
    v_q = torch.cat([torch.zeros(v.shape[0], 1, device=v.device), v], dim=-1)

    # Compute the conjugate of q
    q_conj = quaternion_conjugate(q)

    # Rotate the vector: v' = q * v * q_conj
    v_rotated_q = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)

    # Return the vector part of the resulting quaternion
    return v_rotated_q[..., 1:]



def backface_cull(xyz, normals, camera):
    """
    Args:
        xyz:     [H, W, 3] or [N, 3] surface points (world coords)
        normals: [H, W, 3] or [N, 3] surface normals (world coords)
        camera:  Camera instance with .center attribute (world coords)
    Returns:
        mask:    [H, W] or [N] boolean tensor, True if front-facing
    """
    # Flatten if needed
    shape = xyz.shape
    points = xyz.reshape(-1, 3)
    norms = normals.reshape(-1, 3)
    # Vector from surface to camera
    cam_center = camera.camera_center.to(points.device)  # [3]
    view_vec = (cam_center - points)             # [N, 3]
    view_vec = F.normalize(view_vec, dim=-1)
    norms = F.normalize(norms, dim=-1)
    # Dot product > 0: front-facing
    dot = (view_vec * norms).sum(dim=-1)         # [N]
    mask = dot > 0                               # True: front-facing, False: back-facing
    return mask.reshape(shape[:-1])
def principal_curvatures(dSu, dSv, S_uu, S_uv, S_vv, n):
    E = (dSu * dSu).sum(dim=-1)
    F = (dSu * dSv).sum(dim=-1)
    G = (dSv * dSv).sum(dim=-1)
    L = (n * S_uu).sum(dim=-1)
    M = (n * S_uv).sum(dim=-1)
    N = (n * S_vv).sum(dim=-1)
    EG_F2 = E * G - F * F + 1e-12
    # Mean and Gaussian curvature
    H = (E * N - 2 * F * M + G * L) / (2 * EG_F2)
    K = (L * N - M * M) / EG_F2
    # Principal curvatures (k1 >= k2)
    tmp = torch.clamp(H ** 2 - K, min=0.0)
    sqrt_term = torch.sqrt(tmp)
    k1 = H + sqrt_term
    k2 = H - sqrt_term
    return k1, k2, K, H

def surface_curvatures(dSu, dSv, S_uu, S_uv, S_vv, n):
    """
    Compute mean and Gaussian curvature of a parametric surface.
    All inputs: shape (..., 3), except n (..., 3), and second derivatives (..., 3).
    Returns:
        K: Gaussian curvature (...,)
        H: Mean curvature (...,)
    """
    # First fundamental form
    E = (dSu * dSu).sum(dim=-1)
    F = (dSu * dSv).sum(dim=-1)
    G = (dSv * dSv).sum(dim=-1)
    # Second fundamental form
    L = (n * S_uu).sum(dim=-1)
    M = (n * S_uv).sum(dim=-1)
    N = (n * S_vv).sum(dim=-1)
    # Denominator
    EG_F2 = E * G - F * F
    # Gaussian curvature
    K = (L * N - M * M) / (EG_F2 + 1e-12)
    # Mean curvature
    H = (E * N - 2 * F * M + G * L) / (2 * (EG_F2 + 1e-12))
    return K, H


#     """
#     Convert an N×4 quaternion tensor to an N×3 normal vector.
#     Assumes w-xyz ordering and unit-length quaternions.
#
#     canonical – the vector to rotate; default (0,0,1).
#     """
#     if canonical is None:
#         canonical = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
#
#     w = q[..., 0:1]          # (N,1)
#     r = q[..., 1:]           # (N,3)
#
#     # v' = v + 2 r × (r × v + w v)
#     v = canonical.expand_as(r)
#     t = torch.cross(r, v, dim=-1) + w * v
#     n = v + 2.0 * torch.cross(r, t, dim=-1)
#     return F.normalize(n, dim=-1)          # (N,3)
def insert_knot_u(ctrl_pts, knot_u, u_star, p):
    """
    Insert a knot in the U direction (surface).
    ctrl_pts: (m, n, d)
    knot_u: (m+p+1,)
    u_star: float, the knot to insert
    p: degree in U
    Returns: new_ctrl_pts (m+1, n, d), new_knot_u (m+p+2,)
    """
    m, n, d = ctrl_pts.shape
    k = torch.searchsorted(knot_u, torch.tensor(u_star), right=False) - 1
    k = int(k.item())

    new_knot_u = torch.cat([knot_u[:k+1], torch.tensor([u_star], dtype=knot_u.dtype, device=knot_u.device), knot_u[k+1:]])
    new_ctrl_pts = torch.zeros((m+1, n, d), dtype=ctrl_pts.dtype, device=ctrl_pts.device)

    # Unaffected
    new_ctrl_pts[:k-p+1,:,:] = ctrl_pts[:k-p+1,:,:]
    new_ctrl_pts[k+2:,:,:] = ctrl_pts[k+1:,:,:]

    # Affected (de Boor update)
    for i in range(k-p+1, k+1):
        alpha = (u_star - knot_u[i]) / (knot_u[i+p] - knot_u[i])
        new_ctrl_pts[i,:,:] = (1-alpha) * ctrl_pts[i-1,:,:] + alpha * ctrl_pts[i,:,:]
    return new_ctrl_pts, new_knot_u

def compute_surface_feature_gradient(feature_map):
    # Simple finite difference for norm of gradient
    dx = feature_map[1:, :] - feature_map[:-1, :]
    dy = feature_map[:, 1:] - feature_map[:, :-1]
    grad = torch.zeros_like(feature_map)
    grad[:-1, :] += dx.abs()
    grad[:, :-1] += dy.abs()
    return grad
def compute_principal_curvatures(du, dv, duu, duv, dvv, normals, eps=1e-6):
    """
    Args:
        du, dv, duu, duv, dvv: (..., R, R, 3)
        normals: (..., R, R, 3)
    Returns:
        kmin, kmax: principal curvatures (..., R, R)
        dmin, dmax: principal directions (..., R, R, 2)  (as coefficients in [du, dv] tangent frame)
    """

    # First fundamental form coefficients
    E = (du * du).sum(-1) + eps
    F = (du * dv).sum(-1)
    G = (dv * dv).sum(-1) + eps

    # Second fundamental form coefficients (projected to normal)
    L = (duu * normals).sum(-1)
    M = (duv * normals).sum(-1)
    N = (dvv * normals).sum(-1)

    # Invert the metric tensor I (2x2 matrix at each point)
    # I = [[E, F], [F, G]]
    detI = E * G - F * F + eps
    invI00 = G / detI
    invI01 = -F / detI
    invI10 = -F / detI
    invI11 = E / detI

    # Shape operator S = I^{-1} II
    S00 = invI00 * L + invI01 * M
    S01 = invI00 * M + invI01 * N
    S10 = invI10 * L + invI11 * M
    S11 = invI10 * M + invI11 * N

    # Now S = [[S00, S01], [S10, S11]] at each (i,j)
    # Compute eigenvalues/eigenvectors analytically for 2x2 real symmetric matrix
    traceS = S00 + S11
    detS = S00 * S11 - S01 * S10
    temp = torch.sqrt((0.25 * (traceS)**2 - detS).clamp(min=0))
    kmin = 0.5 * traceS - temp
    kmax = 0.5 * traceS + temp

    # Principal directions (optional, if you want axes)
    # For 2x2 matrix [a b; b d], eigvec for lambda is [b, lambda-a]
    eigvec_min = torch.stack([S01, kmin - S00], dim=-1)  # (..., 2)
    eigvec_max = torch.stack([S01, kmax - S00], dim=-1)  # (..., 2)
    # Normalize
    eigvec_min = eigvec_min / (eigvec_min.norm(dim=-1, keepdim=True) + eps)
    eigvec_max = eigvec_max / (eigvec_max.norm(dim=-1, keepdim=True) + eps)

    return kmin, kmax, eigvec_min, eigvec_max

def insert_knot_v(ctrl_pts, knot_v, v_star, q):
    """
    Insert a knot in the V direction (surface).
    ctrl_pts: (m, n, d)
    knot_v: (n+q+1,)
    v_star: float, the knot to insert
    q: degree in V
    Returns: new_ctrl_pts (m, n+1, d), new_knot_v (n+q+2,)
    """
    m, n, d = ctrl_pts.shape
    l = torch.searchsorted(knot_v, torch.tensor(v_star), right=False) - 1
    l = int(l.item())

    new_knot_v = torch.cat([knot_v[:l+1], torch.tensor([v_star], dtype=knot_v.dtype, device=knot_v.device), knot_v[l+1:]])
    new_ctrl_pts = torch.zeros((m, n+1, d), dtype=ctrl_pts.dtype, device=ctrl_pts.device)

    # Unaffected
    new_ctrl_pts[:,:l-q+1,:] = ctrl_pts[:,:l-q+1,:]
    new_ctrl_pts[:,l+2:,:] = ctrl_pts[:,l+1:,:]

    # Affected (de Boor update)
    for j in range(l-q+1, l+1):
        alpha = (v_star - knot_v[j]) / (knot_v[j+q] - knot_v[j])
        new_ctrl_pts[:,j,:] = (1-alpha) * ctrl_pts[:,j-1,:] + alpha * ctrl_pts[:,j,:]
    return new_ctrl_pts, new_knot_v
def cox_de_boor_basis_and_derivative1(U, degree, u_samples):
    """
    Returns both basis and first derivative for all basis functions at all sample points.
    Args:
        U: (num_knots,) tensor
        degree: int
        u_samples: (N,) tensor in [0, 1]
    Returns:
        N: (N, num_ctrl)  # Basis values
        dN: (N, num_ctrl) # First derivatives
    """
    N = u_samples.shape[0]
    num_ctrl = U.shape[0] - degree - 1
    # Degree 0 basis
    basis = ((u_samples[:, None] >= U[:num_ctrl]) & (u_samples[:, None] < U[1:num_ctrl+1])).to(u_samples.dtype)
    dN = torch.zeros_like(basis)

    # Store lower-degree bases for derivatives
    bases = [basis]
    for d in range(1, degree + 1):
        left = (u_samples[:, None] - U[:num_ctrl]) / (U[d: d+num_ctrl] - U[:num_ctrl]).clamp(min=1e-8)
        left[torch.isnan(left)] = 0
        right = (U[d+1: d+1+num_ctrl] - u_samples[:, None]) / (U[d+1: d+1+num_ctrl] - U[1:1+num_ctrl]).clamp(min=1e-8)
        right[torch.isnan(right)] = 0
        left_basis = left * basis
        right_basis = right * torch.cat([basis[:, 1:], torch.zeros(N, 1, device=U.device, dtype=U.dtype)], dim=1)
        basis = left_basis + right_basis
        bases.append(basis)

    # Compute derivatives
    for i in range(num_ctrl):
        # First term
        if U[i+degree] != U[i]:
            term1 = degree / (U[i+degree] - U[i]) * bases[degree-1][:, i]
        else:
            term1 = 0.0
        # Second term
        if U[i+degree+1] != U[i+1]:
            if i+1 < num_ctrl:
                term2 = degree / (U[i+degree+1] - U[i+1]) * bases[degree-1][:, i+1]
            else:
                term2 = 0.0
        else:
            term2 = 0.0
        dN[:, i] = term1 - term2

    return bases[-1], dN

@torch.jit.script
def cox_de_boor_basis_and_derivative(
        u:       torch.Tensor,   # (N,)
        degree:  int,
        U:       torch.Tensor,   # (n_ctrl+degree+1,)
        eps:     float = 1e-12
):
    """
    Fully-vectorised, autograd-friendly Cox–de Boor evaluation up to 2nd
    derivative.  Works on any dtype / device supported by PyTorch.
    Returns
    -------
    N    : (N, n_ctrl)   – basis
    dN   : (N, n_ctrl)   – first derivative
    d2N  : (N, n_ctrl)   – second derivative
    """
    Np = u.shape[0]
    n_ctrl = U.numel() - degree - 1          # *always* consistent

    # ------------------------------------------------------------------
    # zero-degree
    # ------------------------------------------------------------------
    u_col  = u.unsqueeze(1)                  # (N,1)
    # Only the first n_ctrl knot spans generate zero-degree basis functions
    left   = U[:n_ctrl]                      # (n_ctrl,)
    right  = U[1:n_ctrl+1]                   # (n_ctrl,)
    N0 = ((u_col >= left) & (u_col < right)).to(u.dtype)  # (N, n_ctrl)

    # include right endpoint
    N0[u == U[-1], -1] = 1.0

    # tensor container for progressive degrees
    N_all = [N0]                             # list of (N,n_ctrl+k)

    # ------------------------------------------------------------------
    # recursion for degree 1 … p
    # ------------------------------------------------------------------
    for k in range(1, degree + 1):
        left_denom  = (U[k:    k+n_ctrl]   - U[:n_ctrl]).clamp_min(eps)   # (n_ctrl,)
        right_denom = (U[k+1: k+1+n_ctrl] - U[1:n_ctrl+1]).clamp_min(eps) # (n_ctrl,)

        left_coeff  = (u_col - U[:n_ctrl])           / left_denom         # (N,n_ctrl)
        right_coeff = (U[k+1:k+1+n_ctrl] - u_col)    / right_denom        # (N,n_ctrl)

        Nk   = left_coeff  * N_all[-1]                              \
             + right_coeff * torch.cat([N_all[-1][:, 1:],           # shift
                                         torch.zeros(Np, 1,
                                                     dtype=u.dtype,
                                                     device=u.device)],
                                        dim=1)
        N_all.append(Nk)

    N = N_all[-1]                              # (N,n_ctrl)

    # ------------------------------------------------------------------
    # 1st derivative
    # ------------------------------------------------------------------
    if degree == 0:
        dN = u.new_zeros(Np, n_ctrl)
    else:
        left_denom  = (U[degree:     degree+n_ctrl]   - U[:n_ctrl]).clamp_min(eps)
        right_denom = (U[degree+1:   degree+1+n_ctrl] - U[1:n_ctrl+1]).clamp_min(eps)

        term_left  = degree / left_denom  * N_all[degree-1]
        term_right = degree / right_denom * torch.cat([N_all[degree-1][:, 1:],
                                                       u.new_zeros(Np,1)], dim=1)
        dN = term_left - term_right                           # (N,n_ctrl)

    # ------------------------------------------------------------------
    # 2nd derivative
    # ------------------------------------------------------------------
    if degree <= 1:
        d2N = u.new_zeros(Np, n_ctrl)
    else:
        a = degree * (degree-1)
        A_denom = (U[degree] - U[:n_ctrl]).clamp_min(eps) * \
                  (U[degree-1] - U[:n_ctrl]).clamp_min(eps)
        B_denom = (U[degree] - U[1:n_ctrl+1]).clamp_min(eps) * \
                  (U[degree] - U[:n_ctrl]).clamp_min(eps)
        C_denom = (U[degree+1:degree+1+n_ctrl] - U[1:n_ctrl+1]).clamp_min(eps) * \
                  (U[degree+1:degree+1+n_ctrl] - U[2:n_ctrl+2]).clamp_min(eps)

        A = a / A_denom * N_all[degree-2]
        B = a / B_denom * torch.cat([N_all[degree-2][:, 1:], u.new_zeros(Np,1)], dim=1)
        C = a / C_denom * torch.cat([N_all[degree-2][:, 2:], u.new_zeros(Np,2)], dim=1)

        d2N = A - 2*B + C

    # ------------------------------------------------------------------
    return N, dN, d2N

def make_clamped_uniform_knots(n_ctrl, degree, device='cuda'):
    spans = n_ctrl - degree
    inner = torch.linspace(0, 1, spans + 1, device=device)
    return torch.cat([torch.zeros(degree, device=device),
                      inner,
                      torch.ones(degree,  device=device)])
def cox_de_boor_basis_and_derivative2(u, degree, U, num_ctrl):
    """
    Computes the basis functions and their first and second derivatives for B-splines.
    :param u: tensor of parameter values (shape: [N])
    :param degree: spline degree (int)
    :param U: knot vector (1D tensor, shape: [num_knots])
    :param num_ctrl: number of control points (int)
    :return: basis (N, num_ctrl), dN (N, num_ctrl), d2N (N, num_ctrl)
    """
    N = u.shape[0]
    basis = torch.zeros((degree + 1, N, num_ctrl), dtype=u.dtype, device=u.device)
    dN = torch.zeros((N, num_ctrl), dtype=u.dtype, device=u.device)
    d2N = torch.zeros((N, num_ctrl), dtype=u.dtype, device=u.device)

    # Zero-degree basis functions
    u_expand = u[:, None]
    left = U[:-1]
    right = U[1:]
    basis[0] = ((u_expand >= left) & (u_expand < right)).float()
    basis[0][u == U[-1], -1] = 1.0  # handle right edge

    for k in range(1, degree + 1):
        i = torch.arange(num_ctrl, device=u.device)
        for j in range(num_ctrl):
            left_den = (U[j + k] - U[j]).clamp(min=1e-8)
            right_den = (U[j + k + 1] - U[j + 1]).clamp(min=1e-8)
            left_coeff = (u - U[j]) / left_den
            right_coeff = (U[j + k + 1] - u) / right_den

            left = left_coeff * basis[k - 1, :, j]
            right = right_coeff * basis[k - 1, :, j + 1] if j + 1 < num_ctrl else 0.0
            basis[k, :, j] = left + right

    N_final = basis[degree]

    if degree > 0:
        for j in range(num_ctrl):
            left_den = (U[j + degree] - U[j]).clamp(min=1e-8)
            right_den = (U[j + degree + 1] - U[j + 1]).clamp(min=1e-8)

            left_term = degree / left_den * basis[degree - 1, :, j]
            right_term = degree / right_den * basis[degree - 1, :, j + 1] if j + 1 < num_ctrl else 0.0
            dN[:, j] = left_term - right_term

    if degree > 1:
        for j in range(num_ctrl):
            A1 = (U[j + degree] - U[j]).clamp(min=1e-8)
            A2 = (U[j + degree - 1] - U[j]).clamp(min=1e-8)
            B1 = (U[j + degree] - U[j]).clamp(min=1e-8)
            B2 = (U[j + degree] - U[j + 1]).clamp(min=1e-8)
            C1 = (U[j + degree + 1] - U[j + 1]).clamp(min=1e-8)
            C2 = (U[j + degree + 1] - U[j + 2]).clamp(min=1e-8)

            A = degree * (degree - 1) / (A1 * A2) * basis[degree - 2, :, j] if j < num_ctrl else 0.0
            B = degree * (degree - 1) / (B1 * B2) * basis[degree - 2, :, j + 1] if j + 1 < num_ctrl else 0.0
            C = degree * (degree - 1) / (C1 * C2) * basis[degree - 2, :, j + 2] if j + 2 < num_ctrl else 0.0

            d2N[:, j] = A - 2 * B + C

    dN = torch.nan_to_num(dN, nan=0.0, posinf=0.0, neginf=0.0)
    d2N = torch.nan_to_num(d2N, nan=0.0, posinf=0.0, neginf=0.0)

    return N_final, dN, d2N

class Spline(nn.Module):
    """
    A B-spline (or NURBS) surface model used for representing and optimizing a 3D surface
    using patches. It supports basis-function interpolation, patch subdivision,
    pruning, and a host of optimization methods.
    """
    gaussians: Gaussians

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    # ------------------------------------------------------------
    # 2.  helper functions  (add near end of class)
    # ------------------------------------------------------------
    def _latent_to_xyz(self, lat):  # lat (...,3*L)
        if not self.use_pos_enc: return lat
        L = self.pe_levels
        lat = lat.view(*lat.shape[:-1], 3, L)
        xyz = self.pe_scale * (torch.sin(lat) * self.pe_factors).sum(-1)
        return xyz  # (...,3)

    def _xyz_to_latent(self, xyz):  # inverse (loss-free init)
        if not self.use_pos_enc: return xyz
        L = self.pe_levels
        shp = list(xyz.shape[:-1]) + [3, L]
        lat = torch.zeros(*shp, device=xyz.device, dtype=xyz.dtype)
        lat[..., 0] = torch.asin(torch.clamp(xyz / self.pe_scale, -0.999, 0.999))
        return lat.view(*xyz.shape[:-1], 3 * L)
    def __init__(
            self,
            ctrl_pts: torch.Tensor,
            args: ModelParams,
            config: NurbsOptimizationParams,
            spatial_lr_scale,
            patch_res=4,
            scene_up=None
    ):
        super(Spline, self).__init__()
        if config.grid_densification_factor:
            with torch.no_grad():
                ctrl_pts = grid_upscale(ctrl_pts.detach().cpu()).clone().cuda()


        if ctrl_pts.ndim == 3:
            control_points_p = grid_to_patches(ctrl_pts)
            self.num_patches = control_points_p.shape[0]
            control_points = ctrl_pts

        else:
            self.num_patches = ctrl_pts.shape[0]
            control_points = stitch_control_features(ctrl_pts).requires_grad_(True)

        self.scene_up = torch.tensor([0, 1, 0], dtype=float, device='cuda') if scene_up is None else scene_up
        self.grid_levels = config.grid_levels   # total number of levels, e.g. 2
        self.active_grid_lvl = 0
        self.model_params = args
        #################### Initialization Flags and Placeholders ####################
        self.optimizer = None
        self.viewpoint_stack = None
        self.background = None
        self.pipe = None
        self.iteration = 0
        #################### Patch and Configuration Parameters ####################
        self.p, self.q = 3, 3
        self.percent_dense = config.percent_dense
        self.split_every = config.densification_interval
        self.config = config

        self.device = config.device
        self._res = config.sampling_density
        self._max_sh_degree = args.sh_degree
        self.active_sh_degree = 0
        #################### Activation Functions ####################
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize

        #################### Stage Flags ####################
        self._refine_scales = config.refine_scales
        self._refine_rotations = config.refine_rotations
        self._refine_opacities = config.refine_opacities
        self._refine_cp_weights = config.refine_weights
        self.use_app = config.exposure_compensation
        self.pe_levels = config.pe_levels
        self.spatial_lr_scale = spatial_lr_scale

        #################### Model Parameters ####################
        # Save original number of patches and patch resolution.
        self.patch_res = config.sampling_density

        self._param_names = [
            "control_points",
            "control_weights",
            "opacity",
            "scaling",
            "rotations",
            "features_dc",
            "features_rest",
        ]

        self.global_grids = {}
        for name in self._param_names:
            self.global_grids[name] = nn.ParameterList().requires_grad_()

        self.use_pos_enc = True #getattr(self.config, "use_pos_enc", False)
        self.pe_scale = float(getattr(self.config, "pe_scale", 1.0))  # e.g. 1.0 m →  ±1 m box


        self.patch_res = patch_res
        self.patch_stride = self.patch_res - 1
        ################################################################################
        # Stitch features into a global grid; this function should return a tensor of shape (H, W, D) and grid dimensions.
        bb_min = control_points.amin(dim=(0, 1))
        bb_max = control_points.amax(dim=(0, 1))
        diag = (bb_max - bb_min).norm()  # bounding-box diagonal
        self.pe_scale = diag.detach().item()

        control_features = self.init_control_grid(control_points)
        with torch.no_grad():
            if self.use_pos_enc and control_points.shape[-1] == 3:
                L = self.config.pe_levels
                fac = 0.5 ** torch.arange(L, device=self.device)  # (L,)
                self.register_buffer("pe_factors", fac.view(1, 1, 1, L))
                H, W, _ = control_points.shape
                lat = torch.zeros(H, W, 3, L, device=self.device)
                lat[..., 0] = torch.asin(torch.clamp(control_points.clone() / self.pe_scale, -0.999, 0.999))
                control_points = lat.view(H, W, 3 * L)
                control_points = nn.Parameter(control_points, requires_grad=True).contiguous()
            else:
                control_points = nn.Parameter(control_features[..., :3].clone(), requires_grad=True).contiguous()
            control_weights = nn.Parameter(control_features[..., 3:4].clone(),
                                           requires_grad=(self._refine_cp_weights))
            opacity = nn.Parameter(control_features[..., 4:5].clone().contiguous(), requires_grad=(self._refine_opacities))

            scaling = nn.Parameter(control_features[..., 5:8].clone(), requires_grad=self._refine_scales).contiguous()
            rotations = nn.Parameter(control_features[..., 8:12].clone(), requires_grad=self._refine_rotations).contiguous()
            features_dc = nn.Parameter(control_features[..., 12:15].clone(), requires_grad=True).contiguous()
            features_rest = nn.Parameter(control_features[..., 15:].clone(), requires_grad=True).contiguous()

            self.global_grids["control_points"].append(control_points)
            self.global_grids["control_weights"].append(control_weights.requires_grad_(self._refine_cp_weights))
            self.global_grids["opacity"].append(opacity)
            self.global_grids["scaling"].append(scaling)
            self.global_grids["rotations"].append(rotations)
            self.global_grids["features_dc"].append(features_dc)
            self.global_grids["features_rest"].append(features_rest)


        self.shc = (self._max_sh_degree + 1) ** 2
        #################### Computation Results and Scheduling ####################
        self.surface_normals = torch.empty(0, requires_grad=True, device=self.device)
        #################### NURBS / Training Setup ####################
        self.training_setup(config)
        #################### Mesh Tessellation ####################
        self._faces = None  # will hold (F,3) tensor
        self._cached_used = None  # will hold (V_used,) index map
        self.optimize_knots = False
        self.__compute = True
        #################### NURBS Interpolation Params ####################
        self.H, self.W, self.C = control_features.shape
        self.BU, self.dBU, self.dBUU, self.BV, self.dBV, self.dBVV = self.init_knots(self._res)
        # --- store knots per grid‑level ---------------------------------
        # each entry is a tensor on the correct device (detached copy)
        torch.cuda.empty_cache()

    def densify_sampling(self, set_to=None):
        if set_to is None:
            set_to = min(self._res + 1, self.config.max_resolution)
        with torch.no_grad():
            self._faces = None
            if self._res == self.config.max_resolution:
                return

            self._res = set_to
            if self.optimize_knots:
                self.update_knots()
            else:
                self.BU, self.dBU, self.dBUU, self.BV, self.dBV, self.dBVV = self.init_knots(self._res)


    def init_knots(self, res):
        # --- at top of __init__ after you know H, W, patch_res -----------------
        self.patch_stride = self.patch_res - 1  # 3 for bicubic
        self.num_patches_u = (self.H - self.patch_res) // self.patch_stride + 1
        self.num_patches_v = (self.W - self.patch_res) // self.patch_stride + 1
        delta_u = 1 / self.H
        delta_v = 1 / self.W
        self.u_samples = torch.linspace(delta_u, 1 - delta_u, res * (self.num_patches_u), device=self.device).detach()
        self.v_samples = torch.linspace(delta_v, 1 - delta_v, res * (self.num_patches_v), device=self.device).detach()

        self.knot_u = make_clamped_uniform_knots(self.H, self.p,
                                                 device=self.device).detach()
        self.knot_v = make_clamped_uniform_knots(self.W , self.q,
                                                 device=self.device).detach()
        BU, dBU, dBUU = cox_de_boor_basis_and_derivative(self.u_samples, self.p, self.knot_u)  # U knots, degree, u grid
        BV, dBV, dBVV = cox_de_boor_basis_and_derivative(self.v_samples, self.q, self.knot_v)  # V knots, degree, v grid
        return BU.contiguous(), dBU.contiguous(), dBUU, BV.contiguous(), dBV.contiguous(), dBVV

    def create_knots_by_res(self, res):

        # --- at top of __init__ after you know H, W, patch_res -----------------
        self.patch_stride = self.patch_res - 1  # 3 for bicubic
        self.num_patches_u = (self.H - self.patch_res) // self.patch_stride + 1
        self.num_patches_v = (self.W - self.patch_res) // self.patch_stride + 1
        delta_u = 1 / self.H
        delta_v = 1 / self.W
        knot_u = make_clamped_uniform_knots(self.H, self.p,
                                                device=self.device).detach()
        knot_v = make_clamped_uniform_knots(self.W, self.q,
                                                 device=self.device).detach()
        BU, dBU, dBUU = cox_de_boor_basis_and_derivative(torch.linspace(delta_u, 1 - delta_u,
                                                                        res * (self.num_patches_u),
                                        device=self.device).detach(), self.p,
                                                         knot_u)
        BV, dBV, dBVV = cox_de_boor_basis_and_derivative(torch.linspace(delta_v, 1 - delta_v,
                                                                        res * (self.num_patches_v),
                                        device=self.device).detach(), self.q,
                                                         knot_v)
        return BU, dBU, dBUU, BV, dBV, dBVV

    def update_knots(self):

        BU, dBU, dBUU = cox_de_boor_basis_and_derivative(self.u_samples.sigmoid(), self.p, self.knot_u)  # U knots, degree, u grid
        BV, dBV, dBVV = cox_de_boor_basis_and_derivative(self.v_samples.sigmoid(), self.q, self.knot_v)  # V knots, degree, v grid
        self.BU, self.dBU, self.dBUU = [t.contiguous() for t in (BU, dBU, dBUU)]
        self.BV, self.dBV, self.dBVV = [t.contiguous() for t in (BV, dBV, dBVV)]

    def learn_knots(self, eps=1e-4):
        if self.optimize_knots:
            self.update_knots()  # recompute basis; grads flow to knots

            return
        self.num_patches_u = (self.H - self.patch_res) // self.patch_stride + 1
        self.num_patches_v = (self.W - self.patch_res) // self.patch_stride + 1

        self.u_samples = nn.Parameter(torch.linspace(eps, 1 - eps, (self._res) * self.num_patches_u, device=self.device).logit(), requires_grad=True)
        self.v_samples = nn.Parameter(torch.linspace(eps, 1 - eps, (self._res) * self.num_patches_v, device=self.device).logit(), requires_grad=True)

        self.optimizer_uv = torch.optim.Adam([
            {'params': self.u_samples, "lr":self.config.uv_grid_lr, "name":"us"},
            {'params': self.v_samples, "lr":self.config.uv_grid_lr, "name":"vs"}
            ], lr=0.0, eps=1e-15)

        self.optimize_knots = True

    def insert_knot_u(self, u_star, level=0):
        """
        Insert knot u_star in the U direction at the given level,
        updating all relevant grids and parameters (and basis).
        """
        # Update all relevant parameter grids

        for param in self._param_names:
            grid = self.global_grids[param][level].data  # shape: (H, W, D)
            knot_u = self.knot_u  # old knots

            new_grid, new_knot_u = insert_knot_surface_u(grid, knot_u, u_star, self.p)
            # Update in place
            self.global_grids[param][level] = torch.nn.Parameter(new_grid)
        # Update knots
        self.knot_u = new_knot_u.detach()
        num_patches_u = (self.global_grids['opacity'][0].shape[0] - self._res) // 3 + 1
        u_samples = torch.linspace(0, 1 - 1e-12, (self._res) * num_patches_u, device=self.device).detach()

        self.BU, self.dBU, self.dBUU = cox_de_boor_basis_and_derivative(u_samples, self.p, self.knot_u)

    def insert_knots(self, u_star, v_star, level=0):
        """
        Insert knot v_star in the V direction at the given level,
        updating all relevant grids and parameters (and basis).
        """
        for param in self._param_names:
            param_grid = self.global_grids[param][level].data  # shape: (H, W, D)
            if param == 'control_weights':
                param_grid = param_grid.sigmoid()
                new_grid_tmp, new_knot_u, uidx = insert_knot_surface_u(param_grid, self.knot_u, v_star, self.p)
                new_grid, new_knot_v, vidx = insert_knot_surface_u(new_grid_tmp.permute(1, 0, 2), self.knot_v, u_star, self.q)
                self.global_grids[param][level] = torch.nn.Parameter(inverse_sigmoid(new_grid)).permute(1, 0, 2)
            else:
                new_grid_tmp, new_knot_u, uidx = insert_knot_surface_u(param_grid, self.knot_u, v_star, self.p)
                new_grid, new_knot_v, vidx = insert_knot_surface_u(new_grid_tmp.permute(1, 0, 2), self.knot_v, u_star, self.q)
                self.global_grids[param][level] = torch.nn.Parameter(new_grid).permute(1, 0, 2)
        self.knot_v = new_knot_v.detach()
        self.knot_u = new_knot_u.detach()
        self.H = self.global_grids['opacity'][0].shape[0]
        self.W = self.global_grids['opacity'][0].shape[1]
        # num_patches_u = (self.H - self._res) // 3 + 1
        # num_patches_v = (self.W - self._res) // 3 + 1
        num_patches_u = self.H // 3 + 1
        num_patches_v = self.W // 3 + 1
        self.u_samples = torch.linspace(0, 1 - 1e-3, (self._res) * num_patches_u, device=self.device).detach()
        self.v_samples = torch.linspace(0, 1 - 1e-3, (self._res) * num_patches_v, device=self.device).detach()
        self.BV, self.dBV, self.dBVV = cox_de_boor_basis_and_derivative(self.v_samples, self.q, self.knot_v)
        self.BU, self.dBU, self.dBUU = cox_de_boor_basis_and_derivative(self.u_samples, self.q, self.knot_u)

    def insert_tensors_to_optimizer(self, param, new_row, insert_row_idx):
        """
        Insert new_row at insert_row_idx along the first axis of param (shape [m, n, d]).
        Updates both the param tensor and optimizer state (exp_avg, exp_avg_sq, etc).

        Args:
            param: nn.Parameter or torch.Tensor (shape [m, n, d])
            optimizer: torch.optim.Optimizer (Adam, etc)
            new_row: torch.Tensor, shape (1, n, d)
            insert_row_idx: int, where to insert the row
        Returns:
            new_param: nn.Parameter, with row inserted
        """
        data = param.data
        # Insert new row into the data
        data_new = torch.cat([data[:insert_row_idx], new_row, data[insert_row_idx:]], dim=0)
        # Update param
        new_param = torch.nn.Parameter(data_new, requires_grad=True)

        # Copy optimizer state (e.g. for Adam)
        state = self.optimizer.state.get(param, {})
        if state:
            for key in state:
                tensor = state[key]
                # Insert a zero row in optimizer states
                tensor_new = torch.cat([tensor[:insert_row_idx], torch.zeros_like(new_row), tensor[insert_row_idx:]],
                                       dim=0)
                state[key] = tensor_new
            self.optimizer.state[new_param] = state
            # Remove old param from optimizer state if desired
            del self.optimizer.state[param]
        return new_param
    def update_knot_and_basis(self, direction='u', level=0):
        knots = self.knot_u[level] if direction == 'u' else self.knot_v[level]
        degree = self.p if direction == 'u' else self.q
        num_patches = len(knots) - degree - 1
        samples = torch.linspace(0, 1 - 1e-3, (self._res) * num_patches, device=self.device)
        BU, dBU, dBUU = cox_de_boor_basis_and_derivative(samples, degree, knots)
        if direction == 'u':
            self.BU, self.dBU, self.dBUU = BU, dBU, dBUU
        else:
            self.BV, self.dBV, self.dBVV = BU, dBU, dBUU

    def init_control_grid(self, control_points, device='cuda'):
        with torch.no_grad():
            H, W, _ = control_points.shape
            control_points = control_points.requires_grad_()

            # init_w = 1                                              # target weight in (0,1)
            # w_logit = inverse_sigmoid(torch.full_like(control_points[..., 0:1],
            #                                           fill_value=init_w))
            # w_logit = w_logit #+ 0.05 * torch.randn_like(w_logit)          # ±0.05 jitter
            # control_weights = w_logit.requires_grad_(True)
            init_w = 0.5  # <--- CHANGE THIS VALUE

            w_logit = inverse_sigmoid(torch.full_like(control_points[..., 0:1],
                                                      fill_value=init_w))
            # Optional: Add a small amount of noise to break symmetry
            w_logit = w_logit + 0.01 * torch.randn_like(w_logit)
            control_weights = w_logit.contiguous().requires_grad_(self.config.refine_weights)
            opacity = inverse_sigmoid(
                0.05 * torch.ones(H, W, 1, device=self.device, dtype=torch.float32)).requires_grad_(True)
            features_dc = torch.zeros(H, W, 3, device=self.device).requires_grad_()
            features_rest = torch.zeros(H, W, ((self._max_sh_degree + 1) ** 2 - 1) * 3,
                                        device=self.device).requires_grad_()
            scaling = self.scaling_inverse_activation(
                (torch.full((H, W, 3), fill_value=self.config.scaling_init_factor, dtype=torch.float32, device=self.device) * torch.tensor([1, 1, 1e-3], device=device).reshape(1, 1, 3)).requires_grad_(self.config.refine_scales)
            )
            # rots = torch.ones((H, W, 4), dtype=torch.float32, device=self.device).requires_grad_(self.config.refine_rotations)
            # rots = self.rotation_activation(torch.ones((H, W, 4), dtype=torch.float32, device=self.device)).requires_grad_(self.config.refine_rotations)

            # Instead of torch.ones for rots:
            rots = torch.zeros((H, W, 4), dtype=torch.float32, device=self.device)
            rots[..., 0] = 1.0  # Set w component to 1 for an identity quaternion (w, x, y, z)
            rots = rots.requires_grad_(self.config.refine_rotations)

            return torch.cat([control_points.contiguous(),
                                         control_weights,
                                         opacity,
                                         scaling,
                                         rots,
                                         features_dc,
                                         features_rest],
                                        dim=-1).to(self.device)

    def render_nurbs(self, viewpoint_cam):
        with torch.no_grad():
            valid_points = backface_cull(self.xyz.detach(), self.Sn.detach(), viewpoint_cam)
            self.gaussians = Gaussians(
                xyz=self.xyz[valid_points].reshape(-1, 3),
                opacity=self.opacities[valid_points].reshape(-1, 1),
                scaling=self.scalings[valid_points].reshape(-1, 3),
                rotation=self.rots[valid_points].reshape(-1, 4),
                features=self.spherical_harmonics[valid_points].reshape(-1,  (self._max_sh_degree + 1) ** 2, 3),
                active_sh_degree=self.active_sh_degree,
            )

    @property
    def surface_geo(self):
        return torch.cat([self.xyz, self.dSu, self.dSv, self.Sn], dim=-1)

    @property
    def surface_ft(self):
        return torch.cat([self.opacities, self.scalings, self.rots, self.spherical_harmonics], dim=-1)

    def compute_second_derivatives(self, cp_w, BU, dBU, d2BU, BV, dBV, d2BV, denom):
        """
        Analytic computation of S_uu, S_uv, S_vv for NURBS surfaces.
        cp_w: (H, W, 3)
        BU, dBU, d2BU: (res_u, H)
        BV, dBV, d2BV: (res_v, W)
        denom: (res_u, res_v, 1)
        Returns:
            S_uu, S_uv, S_vv: (res_u, res_v, 3)
        """
        # S_uu
        temp_num_uu = self.contract('ui,ijk->ujk', d2BU, cp_w)
        temp_wdot_uu = self.contract('ui,ij->uj', d2BU, self.c_wts)
        d2Num_uu = self.contract('ujk,vj->uvk', temp_num_uu, BV)
        d2Den_uu = self.contract('uj,vj->uv', temp_wdot_uu, BV).unsqueeze(-1)
        S_uu = (d2Num_uu - 2 * self.xyz * d2Den_uu) / denom  # adjust as needed for full rational

        # S_vv
        temp_num_vv = self.contract('ui,ijk->ujk', BU, cp_w)
        temp_wdot_vv = self.contract('ui,ij->uj', BU, self.c_wts)
        d2Num_vv = self.contract('ujk,vj->uvk', temp_num_vv, d2BV)
        d2Den_vv = self.contract('uj,vj->uv', temp_wdot_vv, d2BV).unsqueeze(-1)
        S_vv = (d2Num_vv - 2 * self.xyz * d2Den_vv) / denom

        # Mixed partial S_uv
        temp_num_uv = self.contract('ui,ijk->ujk', dBU, cp_w)
        temp_wdot_uv = self.contract('ui,ij->uj', dBU, self.c_wts)
        d2Num_uv = self.contract('ujk,vj->uvk', temp_num_uv, dBV)
        d2Den_uv = self.contract('uj,vj->uv', temp_wdot_uv, dBV).unsqueeze(-1)
        S_uv = (d2Num_uv - self.xyz * d2Den_uv) / denom

        return S_uu, S_uv, S_vv
    def evaluate_surface(self, viewpoint_cam):

        if not self.__compute:
            self.gaussians = Gaussians(
                xyz=self.xyz,
                opacity=self.opacities,
                scaling=self.scalings,
                rotation=self.rots,
                features=self.spherical_harmonics,
                active_sh_degree=self.active_sh_degree,
            )
        BU, dBU, dBUU = self.BU, self.dBU, self.dBUU
        BV, dBV, dBVV = self.BV, self.dBV, self.dBVV
        grid_features = self.get_texture_features(0)

        if not self.config.refine_weights:
            cpts = self._latent_to_xyz(self.c_pts) if self.use_pos_enc else self.c_pts
            temp_ft = self.contract('ui,ijk->ujk', BU, grid_features)  # (res_u, W, 3)
            feature_samples = self.contract('ujk,vj->uvk', temp_ft, BV).contiguous()

            temp_surf = self.contract('ui,ijk->ujk', BU, cpts)  # (res_u, W, 3)
            S =  self.contract('ujk,vj->uvk', temp_surf, BV).contiguous()
            temp_num_u = self.contract('ui,ijk->ujk', dBU, cpts)  # N' * wP
            dSu = self.contract('ujk,vj->uvk', temp_num_u, BV).contiguous()

            temp_num_v = self.contract('ui,ijk->ujk', BU, cpts)  # N' * wP
            dSv = self.contract('ujk,vj->uvk', temp_num_v, dBV).contiguous()

        else:
            cp_w = self.control_weighted_enc if self.use_pos_enc else self.c_ptsw

            temp_denom = self.contract('ui,ij->uj', BU, self.c_wts)
            denom = self.contract('uj,vj->uv', temp_denom, BV).unsqueeze(-1).clamp(min=1e-6)

            temp_ft = self.contract('ui,ijk->ujk', BU, grid_features)  # (res_u, W, 3)
            feature_samples = (self.contract('ujk,vj->uvk', temp_ft, BV) / denom).contiguous()

            temp_surf = self.contract('ui,ijk->ujk', BU, cp_w)  # (res_u, W, 3)
            S = (self.contract('ujk,vj->uvk', temp_surf, BV) / denom).contiguous()
            temp_num_u = self.contract('ui,ijk->ujk', dBU, cp_w)  # N' * wP
            temp_wdot_u = self.contract('ui,ij -> uj', dBU, self.c_wts)  # N' * w
            dNum_u = self.contract('ujk,vj->uvk', temp_num_u, BV)
            dDen_u = self.contract('uj ,vj->uv', temp_wdot_u, BV).unsqueeze(-1)

            temp_num_v = self.contract('ui,ijk->ujk', BU, cp_w)  # N' * wP
            temp_wdot_v = self.contract('ui,ij -> uj', BU, self.c_wts)  # N' * w
            dNum_v = self.contract('ujk,vj->uvk', temp_num_v, dBV)
            dDen_v = self.contract('uj ,vj->uv', temp_wdot_v, dBV).unsqueeze(-1)

            dSu = ((dNum_u - S * dDen_u) / denom ).contiguous() # textbook Eq.(4.11)
            dSv = ((dNum_v - S * dDen_v) / denom ).contiguous() # textbook Eq.(4.11)

        self.delta_u = 1.0 / S.shape[0]
        self.delta_v = 1.0 / S.shape[1]
        self.Sn = torch.cross(dSu, dSv, dim=-1)
        self.surface_normals = torch.nn.functional.normalize(self.Sn, dim=-1)
        # --- 2. orthonormal frame --------------------------------------------
        t1 = torch.nn.functional.normalize(dSu, dim=-1)
        t2 = torch.nn.functional.normalize(torch.cross(self.surface_normals, t1, dim=-1), dim=-1)


        # --- 3. principal sigmas with safety clamps --------------------------
        sig_u = dSu.norm(dim=-1, keepdim=True) * self.delta_u * 0.5
        sig_v = dSv.norm(dim=-1, keepdim=True) * self.delta_v * 0.5
        sig_n = 1/self.Sn.norm(dim=-1, keepdim=True) ** 2

        # --- 4. log-space parameter (inverse activation in your pipeline) ----
        R = torch.stack([t1, t2, F.normalize(self.surface_normals, dim=-1)], dim=-1)  # (P,3,3)
        self.feature_samples_ = feature_samples.view(-1, feature_samples.shape[-1])

        self.xyz = S.view(-1, 3)
        self.dSu = dSu
        self.dSv = dSv
        self.opacities = feature_samples[..., :1].view(-1, 1)

        self.scalings = (torch.log(torch.cat([sig_u, sig_v, sig_n], dim=-1)) + feature_samples[..., 1:4]).view(-1, 3)
        # self.rots = matrix_to_quaternion(R.view(-1, 3, 3)).view(-1, 4) +feature_samples[..., 4:8].view(-1, 4)

        # --- CORRECTED ROTATION ---
        # Convert the rotation matrix to a quaternion
        q_surface = matrix_to_quaternion(R.view(-1, 3, 3)).view(-1, 4)

        # Get the learned residual quaternion
        q_residual = feature_samples[..., 4:8].view(-1, 4)

        # Ensure the residual is normalized (important as it's a learned parameter)
        q_residual_normalized = F.normalize(q_residual, dim=-1)

        # Compose the rotations using quaternion multiplication
        # The order matters: q2 * q1 applies q1 first, then q2.
        combined_q = quaternion_multiply(q_residual_normalized, q_surface)

        # Final normalization is a good practice to prevent accumulating errors
        self.rots = F.normalize(combined_q, dim=-1)

        self.spherical_harmonics = feature_samples[..., 8:].view(-1,  self.shc, 3)
        self.__compute = False

        with torch.no_grad():
            front_facing_inds = backface_cull(self.xyz.detach(), quat_to_normal(F.normalize(self.rots, dim=-1)).detach(), viewpoint_cam).view(-1)
        if self.config.cull_backfaces:
            self.feature_samples = self.feature_samples_[front_facing_inds]
            self.gaussians = Gaussians(
                xyz= self.xyz[front_facing_inds],
                opacity=self.opacities[front_facing_inds],
                scaling=self.scalings[front_facing_inds],
                rotation=self.rots[front_facing_inds],
                features=self.spherical_harmonics[front_facing_inds],
                active_sh_degree=self.active_sh_degree,
            )
            return
        self.feature_samples = self.feature_samples_

        self.gaussians = Gaussians(
            xyz= self.xyz,
            opacity=self.opacities,
            scaling=self.scalings,
            rotation=self.rots,
            features=self.spherical_harmonics,
            active_sh_degree=self.active_sh_degree,
        )

    def compute_normal_regularization_loss(self, weight=1.0):
        """
        Computes a regularization loss that encourages the normal implied by `self.rots`
        to align with the true `self.surface_normals`.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        if weight == 0.0:
            return torch.tensor(0.0, device=self.device)

        # 1. Get the true surface normals, already computed and normalized in `evaluate_surface`
        # Reshape to match the flattened gaussian list
        true_normals = self.surface_normals.view(-1, 3)

        # 2. Get the canonical "up" vector. This should correspond to the axis
        # that you consider the "normal" of an un-rotated Gaussian.
        # Your code initializes this as self.scene_up = [0, 1, 0]
        canonical_up_vector = self.scene_up.expand_as(true_normals).to(self.device)

        # 3. Rotate the canonical vector by each quaternion in self.rots
        # The result is the normal direction for each Gaussian according to its rotation
        quaternion_normals = rotate_vector_by_quaternion(F.normalize(canonical_up_vector, dim=-1), F.normalize(self.rots.clone(), dim=-1))

        # 4. Compute the cosine similarity.
        # Since both `true_normals` and `quaternion_normals` are unit vectors,
        # the cosine similarity is simply their dot product.
        cosine_similarity = torch.sum(true_normals * quaternion_normals, dim=-1)

        # 5. Formulate the loss. We want to MAXIMIZE the cosine similarity,
        # which means we MINIMIZE (1 - cosine_similarity).
        # This loss is 0 for perfect alignment, 1 for orthogonal vectors, and 2 for opposite vectors.
        loss = 1.0 - cosine_similarity

        # Return the mean loss over all Gaussians
        return torch.mean(loss) * weight

    def reinit_optimizer(self):
        """
        Re-initializes the optimizer after changing spline parameters (e.g., after knot insertion).
        Keeps learning rates and param group names as set in training_setup.
        """
        # Example: Adam optimizer with parameter groups for each type.
        # The names and lrs here must match those in training_setup!
        param_groups = {}
        for name in self._param_names:
            try:
                new_param_dict = self.replace_tensor_to_optimizer(self.global_grids[name][0], name)
                self.global_grids[name][0] = new_param_dict[name]
            except: # Triggered if current param is not requires grad
                continue
        print("[Spline] Optimizer re-initialized with", len(param_groups), "parameter groups.")
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def adaptive_refine(self, lambda1=1.0, lambda2=1.0, lambda3=1.0, thresh=None):
        """
        Refine spline grid adaptively by inserting knots at locations of high composite score.
        Composite score: weighted sum of normalized |curvature|, reconstruction error, and feature gradient.

        Args:
            lambda1, lambda2, lambda3: Weights for curvature, error, feature gradient.
            thresh: Optional; if set, only refine where the max score exceeds this.
        """
        # --- Gather/compute relevant per-grid maps ---
        curvature_score = torch.abs(self.curvature_mean)  # (H, W)
        error_score = getattr(self, "reconstruction_error_map", None)  # (H, W)
        if error_score is None:
            error_score = torch.zeros_like(curvature_score)
        feature_grad_score = self.c_pts.grad.norm(dim=-1)

        # --- Normalize maps robustly ---
        def normalize(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        score_map = (
                # normalize(curvature_score) * lambda1 +
                # normalize(error_score) * lambda2 +
                normalize(feature_grad_score) * lambda3
        )
        # --- Optionally threshold, otherwise pick global maxima per axis
        row_scores = score_map.mean(dim=1)
        col_scores = score_map.mean(dim=0)
        row_idx = torch.argmax(row_scores).item()
        col_idx = torch.argmax(col_scores).item()

        # You can do more sophisticated local-max search if desired.
        u_star = row_idx / self.H
        v_star = col_idx / self.W

        if thresh is not None:
            if score_map[row_idx, col_idx] < thresh:
                print("[Spline] Adaptive refine: No region exceeds threshold; skipping.")
                return

        print(f"[Spline] Adaptive refine at u={u_star:.3f} (row {row_idx}), v={v_star:.3f} (col {col_idx})")

        # self.insert_knot_u(u_star)
        self.insert_knots(u_star, v_star)
        self.reinit_optimizer()

    # ------------------------------------------------------------------
    #   NEW  –  geomdl‑based evaluator
    # ------------------------------------------------------------------

    # --- in model/KnotSurface.py  -------------------------------
    def orientation_barrier(self, weight: float = 1.0,
                            eps: float = 2e-2) -> torch.Tensor:
        """
        Penalise negative Jacobian determinant  det J = (S_u × S_v)·n  .
        Args
        ----
        weight : multiplier for the loss
        eps    : small positive margin; only wrong-sign areas incur loss
        Returns
        -------
        torch.Tensor   scalar loss (requires_grad=True)
        """
        loss = torch.tensor(0., device='cuda')

        if weight == 0.0:  # quick exit if caller disabled it
            return loss
        for g_lvl in range(self.active_grid_lvl + 1):
            # signed = torch.zeros_like(self._dus[g_lvl][:, 0])
            norm = torch.cross(self.dSu[g_lvl], self.dSv[g_lvl], dim=-1)
            self._surf_n = torch.nn.functional.normalize(
                norm, dim=-1)
            # 1. signed area density
            signed = (norm
                       * self._surf_n).sum(-1)  # sh1

        # 2. hinge barrier:  (max(0, ε − signed))²
            loss += torch.relu(eps - signed).pow(2).mean()

        return loss * weight

    def contract(self, equation: str, *args):
        """
        Cached opt_einsum contraction with PyTorch backend.

        * Builds the optimal path once (no backend arg).
        * Re‑uses it every call, specifying backend='torch' at call time
          (this is the API expected by opt_einsum ≥3.4).
        """
        if not hasattr(self, "_oe_cache"):
            self._oe_cache = {}

        key = (equation, tuple(t.shape for t in args))
        fn = self._oe_cache.get(key)
        if fn is None:
            fn = oe.contract_expression(
                equation, *[t.shape for t in args], optimize="optimal"
            )
            self._oe_cache[key] = fn
        # call with explicit backend each time
        return fn(*args, backend="torch")

    def dense_interp(self, res=4):

        BU, dBU, dBUU, BV, dBV, dBVV = self.create_knots_by_res(res)
        with torch.no_grad():
            cp_w = self.control_weighted_enc if self.use_pos_enc else self.c_ptsw
            grid_features = torch.cat([self.opacity, self.c_featute_dc], dim=-1)
            temp_denom = self.contract('ui,ij->uj', BU, self.c_wts)
            denom = self.contract('uj,vj->uv', temp_denom, BV).unsqueeze(-1).clamp(min=1e-6)

            temp_surf = self.contract('ui,ijk->ujk', BU, cp_w)  # (res_u, W, 3)
            dense_S = torch.einsum('ujk,vj->uvk', temp_surf, BV) / denom

            temp_ft = self.contract('ui,ijk->ujk', BU, grid_features)  # (res_u, W, 3)
            feature_samples = (self.contract('ujk,vj->uvk', temp_ft, BV) / denom)
            opacity = feature_samples[..., :1]
            albedo = feature_samples[..., 1:]
            temp_num_u = self.contract('ui,ijk->ujk', dBU, cp_w)  # N' * wP
            temp_wdot_u = self.contract('ui,ij -> uj', dBU, self.c_wts)  # N' * w
            dNum_u = self.contract('ujk,vj->uvk', temp_num_u, BV)
            dDen_u = self.contract('uj ,vj->uv', temp_wdot_u, BV).unsqueeze(-1)

            temp_num_v = self.contract('ui,ijk->ujk', BU, cp_w)  # N' * wP
            temp_wdot_v = self.contract('ui,ij -> uj', BU, self.c_wts)  # N' * w
            dNum_v = self.contract('ujk,vj->uvk', temp_num_v, dBV)
            dDen_v = self.contract('uj ,vj->uv', temp_wdot_v, dBV).unsqueeze(-1)

            dSu = (dNum_u - dense_S * dDen_u) / denom  # textbook Eq.(4.11)
            dSv = (dNum_v - dense_S * dDen_v) / denom  # textbook Eq.(4.11)

            # --- 2. orthonormal frame --------------------------------------------
            n_raw = torch.cross(dSu, dSv, dim=-1)
            return dense_S, dSu, dSv, n_raw, opacity, albedo

    def set_knot_u(self, knot_u):
        with torch.no_grad():
            self.knot_u = torch.tensor(knot_u, device=self.device)

    def set_knot_v(self, knot_v):
        with torch.no_grad():
            self.knot_v = torch.tensor(knot_v, device=self.device)

    def grid_scaling(self, du, dv,
                     feature_samples: torch.Tensor, device='cuda') -> torch.Tensor:

        """
        dus, dvs: (P,3) partial derivatives at each sample.
        sampled_scales: previous learned log‐scales (unused here, or you can add them).
        Returns: (P,3) log‐(sigma_u, sigma_v, sigma_norm).
        """

        adj_dist = torch.stack([ torch.full_like(du, fill_value=1e-20, device=device), du , dv], dim=-1).detach().log()

        return adj_dist + feature_samples

    def dU(self, S):
        du = S[1:, :, :] - S[:-1, :, :]
        du = (torch.cat([du, du[-1:, :]], dim=0))
        return du

    def dS(self):
        return self.dU, self.dV

    def dV(self, S):
        dv = S[:, 1:, :] - S[:, :-1, :]
        dv = (torch.cat([dv, dv[:, -1:]], dim=1))  # .clamp(min=1e-6, max=.005)
        return dv

    def visualize_all_grid_features(self):
        """
        Visualize the global grid features for all parameters in a single figure with subplots.
        For each parameter, the corresponding grid is displayed and the parameter name is shown above the subplot.
        """
        # List of parameter names that we wish to visualize.
        # This list should match the keys in self.global_grids.
        param_names = [p for p in self._param_names if self.global_grids[p][0].requires_grad ]  # For example: ["control_points", "control_weights", "opacity", "scaling", "rotations", "features_dc", "features_rest"]
        num_params = len(param_names)
        n_rows = int(np.ceil(np.sqrt(num_params)))
        n_cols = int(np.ceil(num_params / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

        # Iterate over each parameter and plot its grid representation.
        for idx, pname in enumerate(param_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            if not self.global_grids[pname][0].requires_grad:
                continue
            # Retrieve the grid from the global grids; here we use the first element for visualization.
            grid_feature = self.global_grids[pname][self.active_grid_lvl].detach().cpu()
            # If the parameter is a feature that needs conversion (for example SH coefficients)
            if pname == "features_dc" or pname == "features_rest":
                # Convert spherical harmonics coefficients to RGB.
                # if self.global_grids[pname][self.active_grid_lvl].grad is not None:
                #     img = F.normalize(self.global_grids[pname][self.active_grid_lvl], dim=-1).cpu().numpy()
                #     pname += " grad"
                img = SH2RGB(grid_feature.numpy())
            elif pname in ["control_weights", "opacity"]:
                img = grid_feature.sigmoid().numpy()
            elif pname == "control_points":
                # if self.use_pos_enc:
                #     grid_feature = self._latent_to_xyz(grid_feature.cuda()).detach().cpu()

                # img = (self.compute_normal_map(grid_feature)).numpy()
                img = F.normalize(self.Sn.detach(), dim=-1).cpu().numpy()
                img = (img - img.min())/(img.max() - img.min())
            else:
                # For other parameters, visualize the activated version (after sigmoid if desired).
                img = grid_feature.numpy()


            img = process_feature_grid(torch.from_numpy(img))
            # Optionally, rotate the image 90° clockwise.
            img = np.rot90(img, k=-3)
            ax.imshow(img)
            ax.set_title(pname, fontsize=10)
            ax.axis("off")

        # Hide any unused subplots if the grid is larger than the number of parameters.
        for idx in range(num_params, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)  # add this
        del fig, axes
        gc.collect()
        torch.cuda.empty_cache()  # optional but handy

    # =========================================================================
    #   NURBS PARAMETERS: SETUP, SPLIT, PRUNE, AND REINITIALIZATION
    # =========================================================================
    def oneupSHdegree(self):
        if self.active_sh_degree < self._max_sh_degree:
            self.active_sh_degree += 1

# =========================================================================
#   LOSSES REGULARIZATION
# =========================================================================
    def uv_laplacian_regularization(self, weight=1.0, param_name='control_points', lvl=0) -> torch.Tensor:
        """
        Compute the Laplacian regularization on a uv grid to enforce surface consistency.

        Args:
            uv_grid (torch.Tensor): A tensor of shape (H, W, C) containing the uv grid values.
                                    Typically, C=2 for (u, v) coordinates.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization loss.
        """
        # Define a base Laplacian filter kernel (2D, float32, device-correct).
        base_kernel = torch.tensor([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=torch.float32, device=self.device)

        loss = torch.tensor(0., device=self.device)
        if weight > 0:
        # for lvl in range(self.active_grid_lvl + 1):
            uv_grid = self.global_grids[f'{param_name}'][lvl].permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            C = uv_grid.size(1)
            laplacian_filter = base_kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # (C,1,3,3)
            laplacian_response = F.conv2d(uv_grid, laplacian_filter, padding=1, groups=C)
            loss = laplacian_response.pow(2).mean() * weight
        return loss

    def grid_consistency(self, weight=1.0) -> torch.Tensor:
        """
        Compute the Laplacian regularization on a uv grid to enforce surface consistency.

        Args:
            uv_grid (torch.Tensor): A tensor of shape (H, W, C) containing the uv grid values.
                                    Typically, C=2 for (u, v) coordinates.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization loss.
        """
        # Define a Laplacian filter kernel.
        # if no active levels, return zero
        loss =  torch.tensor(0.0, device=self.device)
        if weight == 0.0:
            return loss
        # compute normals map: shape H x W x 3
        normals = F.normalize(self.Sn, dim=-1)

        cos_u = F.cosine_similarity(normals[1:, :, :], normals[:-1, :, :], dim=-1)
        cos_v = F.cosine_similarity(normals[:, 1:, :], normals[:, :-1, :], dim=-1)

        # accumulate loss = mean(1 - cosine) over valid neighbor pairs
        u_loss = (1.0 - cos_u).mean()
        v_loss = (1.0 - cos_v).mean()


        return (u_loss + v_loss) * weight


    def step(self, iteration, update_knots=False, radii=None):
        """
        One optimization step.
        """

        if update_knots and iteration > self.config.refine_knot_from:
            self.optimizer_uv.step()
            self.optimizer_uv.zero_grad(set_to_none=True)
            self.update_knots()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.__compute = True

        with torch.no_grad():
            if iteration % 2000 == 1:
                self.visualize_all_grid_features()

            lr = self.xyz_scheduler_args(iteration)
            for group in self.optimizer.param_groups:
                if group['name'] in ['control_points_0', 'control_points_1', 'control_points_2', 'control_points_3', 'control_points_4']:
                    group['lr'] = lr


            reduce_batch_every = self.config.batch_until_iter // max(int(np.log(self.config.batch_size)), 1)
            if not (iteration % reduce_batch_every) and iteration <= self.config.batch_until_iter:
                self.densify_sampling()
                self.config.batch_size = max(self.config.batch_size // 2, 1)


            if iteration == self.config.refine_knot_from:
                self.learn_knots()

            if iteration % self.config.opacity_reset_interval == 0 and iteration < 10000:
                self.reset_opacity()

    def normal_discrepancy_loss(self, viewpoint_cam, w=1.0):
        return (1-((self.surface_normals / torch.norm(self.surface_normals, dim=-1, keepdim=True)).clamp_min(min=1e-6) * self.get_normal(viewpoint_cam))).mean() * w

    def eikonal_loss(self, mask=None, eps=1e-6):
        """
        Stable and expressive eikonal loss for the Spline surface.
        Args:
            mask (Tensor, optional): (N,) or (N,H,W) bool tensor indicating valid surface points for regularization.
            eps (float): small value to avoid division by zero.
        Returns:
            Scalar eikonal loss.
        """
        # Compute gradients (already shaped as (N, H, W, 3))
        dus = self.dSu.reshape(-1, 3).abs()  # (N*H*W, 3)
        dvs = self.dSv.reshape(-1, 3).abs()
        grad_norms = torch.sqrt((dus ** 2).sum(dim=-1) + (dvs ** 2).sum(dim=-1) + eps)  # (N*H*W,)

        # Optionally, use mean of both du and dv gradients
        eikonal_term = (grad_norms - 1).abs()

        # Optional: mask out unreliable/occluded points
        if mask is not None:
            mask = mask.view(-1).bool()
            eikonal_term = eikonal_term[mask]
            if eikonal_term.numel() == 0:
                return torch.tensor(0.0, device=self.dSu.device)

        return eikonal_term.mean()

    def loss_curvature(self, weight=1e-4):
        """
        Penalize high mean curvature over the surface (smoothness prior).
        """
        if weight == 0:
            return torch.tensor(0.0, device='cuda')
        # self.curvature_mean: (H, W) or (N,) tensor, computed during evaluate_surface
        if not hasattr(self, "curvature_mean"):
            raise RuntimeError("curvature_mean not computed. Run evaluate_surface first.")
        return weight * (self.curvature_mean.abs()).mean()

    def loss_curvature_fair(self, weight=1e-4):
        """
        Penalize the squared differences in mean curvature between neighboring points.
        Promotes fair, naturally varying curvature.
        """
        # Assume self.curvature_mean is (H, W)
        if not hasattr(self, "curvature_mean"):
            raise RuntimeError("curvature_mean not computed. Run evaluate_surface first.")

        # Compute curvature differences along u and v axes
        diff_u = self.curvature_mean[1:, :] - self.curvature_mean[:-1, :]
        diff_v = self.curvature_mean[:, 1:] - self.curvature_mean[:, :-1]

        loss_u = diff_u.abs().mean()
        loss_v = diff_v.abs().mean()

        return 0 * (loss_u + loss_v)

    def eikonal_term(self, weight=1.0):
        """
        Compute an Eikonal loss on the spline surface by enforcing that the local area element,
        given by ||S_u x S_v||, is close to 1.

        Here, S(u,v) is the spline surface; S_u and S_v denote its partial derivatives with respect
        to the parameters u and v. For a perfect signed distance function the norm of the gradient is 1,
        and here we analogously enforce that the mapping from parameter space to 3D is locally isometric.

        Returns:
            loss (torch.Tensor): A scalar tensor representing the mean squared error of the area element from 1.
        """
        loss = torch.tensor(weight, device='cuda')
        if weight == 0.0:
            return loss
        for lvl in range(self.active_grid_lvl+1):
            loss += ((self.get_normals.norm(dim=-1, keepdim=True) - 1).abs()).mean()
        return loss * weight

    def edge_aware_loss(self, weight=100.0, alpha=10.0):
        r"""
        Compute an edge-aware regularization loss over the spline surface.

        Let \(X \in \mathbb{R}^{N \times R \times R \times 3}\) denote the predicted
        surface normals (or positions) per patch, where \(N\) is the number of patches
        and \(R\) is the resolution per patch. We compute horizontal and vertical
        finite differences:

        \[
        \begin{aligned}
        \Delta_h X &= X[:, :, 1:, :] - X[:, :, :-1, :] \\
        \Delta_v X &= X[:, 1:, :, :] - X[:, :-1, :, :]
        \end{aligned}
        \]

        The edge-aware weights are defined as
        \[
        w_h = \exp(-\alpha \|\Delta_h X\|), \quad w_v = \exp(-\alpha \|\Delta_v X\|),
        \]
        so that regions with large differences (strong edges) are penalized less.

        Finally, the loss is given by:

        \[
        \mathcal{L}_{\text{edge}} = \lambda \cdot \frac{1}{|\Omega|} \sum_{p \in \Omega} \Bigl( w_h(p) \|\Delta_h X(p)\|^2 + w_v(p) \|\Delta_v X(p)\|^2 \Bigr)
        \]

        where \(\Omega\) indexes the spatial grid and \(\lambda\) is a weighting factor.

        Returns:
            loss (torch.Tensor): A scalar tensor representing the edge-aware loss.
        """
        # Reshape the predicted surface normals to (N, R, R, 3)
        N = self.num_patches
        R = self._res
        # Here we use the computed surface normals; you could also use surface positions.
        normals = self.surface_normals.view(N, R, R, 3)

        # Compute finite differences along horizontal (u) and vertical (v) directions.
        diff_h = normals[:, :, 1:, :] - normals[:, :, :-1, :]  # shape: (N, R, R-1, 3)
        diff_v = normals[:, 1:, :, :] - normals[:, :-1, :, :]  # shape: (N, R-1, R, 3)

        # Compute the norm of these differences.
        norm_h = torch.norm(diff_h, dim=-1, keepdim=True)  # (N, R, R-1, 1)
        norm_v = torch.norm(diff_v, dim=-1, keepdim=True)  # (N, R-1, R, 1)

        # Compute edge-aware weights: if the difference is high, weight is low.
        w_h = torch.exp(-alpha * norm_h)
        w_v = torch.exp(-alpha * norm_v)

        # Compute the weighted squared differences.
        loss_h = (w_h * (diff_h ** 2)).mean()
        loss_v = (w_v * (diff_v ** 2)).mean()

        loss = weight * (loss_h + loss_v)
        return loss

    # ---------------------------------------------------------------------
    #   SURFACE DISCREPANCY LOSS  (Gaussian ↔ NURBS, global grid)
    # ---------------------------------------------------------------------

    def surface_discrepancy_loss(
            self,
            K: int = 4,
            lvl: int = 0,
            weight: float = 1.0,
            w_norm: float = 0.05,
    ) -> torch.Tensor:
        """
        Memory-safe, fully-vectorised loss that measures how well each Gaussian
        aligns with its local NURBS patch (position + normal).

        Steps
        -----
        1.  Build K random planar samples for every Gaussian.
        2.  Convert each 3-D displacement to Δ(u,v) via J†  (J detached).
        3.  Fetch surface position & normal with one `grid_sample`.
        4.  Loss = ‖p−S‖² + w_norm·(1−⟨n_plane,n_surf⟩)²;  O(G·K) memory.
        """
        if weight == 0.0:
            return self.xyz[0].new_zeros([])

        # --- Gaussian parameters (need grads) -----------------------------
        mu = self.xyz.reshape(-1, 3)  # (G,3)
        rot = quaternion_to_matrix(self.rots.reshape(-1, 4))  # (G,3,3)
        scale = self.scalings.reshape(-1, 3)  # (G,≥2)
        G = mu.shape[0]
        Hs, Ws = self.xyz.shape[:2]
        e1, e2 = rot[..., 0], rot[..., 1]
        su, sv = scale[:, 0], scale[:, 1]

        rnd = torch.rand(G, K, 2, device=mu.device) * 2 - 1
        # rnd = torch.rand(Hs, Ws, K, 2, device=mu.device) * 2 - 1
        p = mu[:, None, :] + rnd[..., 0, None] * su[:, None, None] * e1[:, None, :] \
            + rnd[..., 1, None] * sv[:, None, None] * e2[:, None, :]  # (G,K,3)

        n_plane = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-6)
        n_plane = n_plane[:, None, :].expand(G, K, 3)
        S, dSu, dSv, _, _, _ = self.dense_interp()

        # --- Detached geometry for UV mapping & sampling -----------------
        with torch.no_grad():
            H, W = self.xyz.shape[:2]

            # UV grid for every Gaussian centre
            u_lin = torch.linspace(0.0, 1.0, H, device=mu.device)
            v_lin = torch.linspace(0.0, 1.0, W, device=mu.device)
            uv0 = torch.stack(torch.meshgrid(u_lin, v_lin, indexing="ij"), -1).view(-1, 2)  # (G,2)

            du = self.dSu.reshape(-1, 3)
            dv = self.dSv.reshape(-1, 3)
            J = torch.stack([du, dv], dim=-1)  # (G,3,2)
            J_pinv = torch.linalg.pinv(J)  # (G,2,3)

            disp = (p.detach() - mu[:, None, :])
            d_uv = (J_pinv[:, None] @ disp[..., None]).squeeze(-1)  # (G,K,2)
            uv = (uv0[:, None, :] + d_uv).clamp(0, 1)  # (G,K,2)

            # normalise to [-1,1] for grid_sample
            grid = torch.stack([
                2 * uv[..., 1] / (W - 1) - 1,
                2 * uv[..., 0] / (H - 1) - 1
            ], dim=-1).view(1, G * K, 1, 2)


            normals = torch.cross(dSu, dSv, dim=-1)
            # --- grid_sample (positions & normals) ---------------------------
            pos_img = S.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            nrm_img = normals.permute(2, 0, 1).unsqueeze(0)

            surf_pos = F.grid_sample(pos_img, grid, mode='bilinear',
                                     align_corners=True, padding_mode='border') \
                .squeeze(3).permute(0, 2, 1).reshape(G, K, 3)
            surf_nrm = F.grid_sample(nrm_img, grid, mode='bilinear',
                                     align_corners=True, padding_mode='border') \
                .squeeze(3).permute(0, 2, 1).reshape(G, K, 3)
            surf_nrm = F.normalize(surf_nrm, dim=-1, eps=1e-6)

        # --- Loss --------------------------------------------------------
        chamfer, cossim = chamfer_distance(
            p, surf_pos,
            x_normals=n_plane, y_normals=surf_nrm,
            batch_reduction=None,
            point_reduction=None,
            single_directional=True
        )
        return chamfer.mean().pow(2) * weight, cossim.norm(dim=-1).mean() * w_norm

    def tessellate_tri_mesh(self, viewpoint_cam, force=True):
        """
        Efficient, fully vectorized tessellation of the Spline surface at grid level `lvl`.
        Returns:
            verts:  (V, 3) float Tensor of vertex positions.
            faces:  (F, 3) long Tensor of triangle indices.
            colors: (V, 3) uint8 Tensor of per-vertex RGB colors.
        """
        # 1) Gather per-patch samples and colors
        verts, dSu, dSv, normals, opacity, albedo = self.dense_interp(res=self._res*2)
        Hs, Ws, D = verts.shape
        # normal_grid = self.surface_normals[lvl].reshape(-1, 3)
        normal_grid = F.normalize(normals, dim=-1).reshape(-1, 3)
        albedo = SH2RGB(albedo.squeeze()).reshape(-1, 3)
        opacity_grid = opacity.reshape(-1, 1)
        # sh = self.sh_feat.transpose(1, 2).reshape(Hs * Ws, -1)
        # ------------------------------------------------------------------
        # 2) Compute face indices with a sliding 2×2 window (stride = 1)
        # ------------------------------------------------------------------
        if self._faces is None or force:

            idx_map = torch.arange(Hs * Ws, device=verts.device).view(Hs, Ws)
            rr, cc = torch.meshgrid(
                torch.arange(Hs - 1, device=self.device),
                torch.arange(Ws - 1, device=self.device),
                indexing="xy",
            )

            v0 = idx_map[rr, cc].reshape(-1)
            v1 = idx_map[rr + 1, cc].reshape(-1)
            v2 = idx_map[rr, cc + 1].reshape(-1)
            v3 = idx_map[rr + 1, cc + 1].reshape(-1)

            faces = torch.cat(
                [torch.stack([v0, v1, v3], dim=1), torch.stack([v0, v3, v2], dim=1)],
                dim=0,
            )  # ← last existing line
            self._faces = faces

        faces = self._faces.detach()
        valid_points = backface_cull(verts, normals, viewpoint_cam)
        valid_points_flat = valid_points.flatten()  # (Hs*Ws,)
        is_face_valid = valid_points_flat[faces].all(dim=1)  # [n_faces], True if all 3 verts valid
        faces_valid = faces[is_face_valid]
        return {
                "verts": verts,
                "faces": faces_valid,
                "normals": normal_grid,
                "opacity": opacity_grid.sigmoid(),
                "albedo": albedo,
                # "sh": sh,
            }

    def compute_curvature(self):
        control_features = torch.cat([self.control_points * self.control_weights,
                                      self.control_weights,
                                      ], dim=-1)

        patch_features = grid_to_patches(control_features, patch_size=4, stride=self.patch_stride).contiguous()
        duus  = (torch.einsum('nsi,nijc,nhj->nshc', self._d2UM, patch_features[..., :3].contiguous(), self._VM)).view(self.num_patches, self._res, self._res, 3)
        dvvs  = (torch.einsum('nsi,nijc,nhj->nshc', self._UM, patch_features[..., :3].contiguous(), self._d2VM)).view(self.num_patches, self._res, self._res, 3)

        E = torch.sum(self.dSu * self.dSu, dim=-1).view(self.num_patches, self._res, self._res)
        F = torch.sum(self.dSu * self.dSv, dim=-1).view(self.num_patches, self._res, self._res)
        G = torch.sum(self.dSv * self.dSv, dim=-1).view(self.num_patches, self._res, self._res)

        normals = self.get_normals()
        normal_magnitudes = torch.norm(normals, dim=-1, keepdim=True)
        unit_normals = normals / (normal_magnitudes + 1e-6)
        L = torch.sum(duus * unit_normals, dim=-1)
        M = torch.sum(normals * unit_normals, dim=-1)
        N = torch.sum(dvvs * unit_normals, dim=-1)
        mean_curvature = (E * N - 2 * F * M + G * L) / (2 * (E * G - F ** 2) + 1e-6)
        return mean_curvature

    def construct_list_of_attributes(self, scaling_dim=2):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        features_dc = self.gaussians.features[...,:1,:]
        features_rest = self.gaussians.features[...,1:,:]
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(scaling_dim):
            l.append(f'scale_{i}')
        for i in range(self.gaussians.rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, outdir='output', pcd_name='point_cloud.ply', save_3d=False):
        gaussians = self.gaussians
        outdir = os.path.join(outdir, pcd_name)
        os.makedirs(os.path.dirname(outdir), exist_ok=True)

        features, opacity, rotations, scaling, xyz = (
            gaussians.features,
            gaussians.opacity,
            gaussians.rotation,
            gaussians.scaling,
            gaussians.xyz,
        )
        f_dc = features[:, :1, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features[:, 1:, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = opacity.detach().cpu().numpy()
        rotation = rotations.detach().cpu().numpy()
        if not save_3d:
            scale = scaling.detach().cpu().numpy()
        else:
            try:
                scale = torch.cat(
                    (scaling.detach().cpu(), -8 * torch.ones((scaling.shape[0], 1), dtype=torch.float32, device='cpu')),
                    dim=-1
                ).detach().cpu().numpy()
            except:
                print("Failed to save 3D scales")
                scale = scaling.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(scaling_dim=scale.shape[-1])]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        if os.path.isfile(outdir):
            os.remove(outdir)
        print(f"Saving outdir: {outdir}")
        PlyData([el]).write(outdir)


    def discrepancy(self, weight=1.0, w_norm=1.0) -> torch.Tensor:
        """
        For each patch, for each Gaussian (each grid cell), sample a local tangent plane
        using the Gaussian's own scale in u and v. For each Gaussian, we generate a grid
        of R_sample x R_sample points. Thus, for a patch with self._res = R,
        we sample R² * R_sample² points per patch.

        Then, we compute the Chamfer distance between these adaptively–sampled points and an
        upsampled version of the actual surface points of the patch.

        Returns:
            A scalar loss (mean Chamfer distance over patches) multiplied by the given weight.
        """
        if weight == 0.0:
            return torch.tensor(0.0, device=self.device),  torch.tensor(0.0, device=self.device)

        chamfer, cossim = self.surface_discrepancy_loss(weight=weight, w_norm=w_norm)

        return chamfer, cossim

    def activate_rots(self):
        self._refine_rotations=True
        for i in range(self.active_grid_lvl):
            self.global_grids['rotations'][i].requires_grad_(self._refine_rotations)

    def activate_weights(self):
        self._refine_cp_weights = True
        for i in range(self.active_grid_lvl):
            self.global_grids['control_weights'][i].requires_grad_(self._refine_cp_weights)

    def activate_scales(self):
        self._refine_scales = True
        for i in range(self.active_grid_lvl):
            self.global_grids['scaling'][i].requires_grad_(self._refine_scales)

    def activate_opacities(self):
        self._refine_opacities = True
        for i in range(self.active_grid_lvl):
            self.global_grids['opacity'][i].requires_grad_(self._refine_opacities)

    def set_background(self, background):
        self.background = background

    def set_pipe(self, pipe):
        self.pipe = pipe
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale / 2) - 1, 0))
        depth_view = depth.squeeze()[st::scale, st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts - T) @ R.transpose(-1, -2)
        return pts
    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale / 2) - 1, 0)
        depth_view = depth[None, :, st::scale, st::scale]
        W, H = int(fov_camera.image_width / scale), int(fov_camera.image_height / scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
            [points_in_camera_space[:, 0] * fov_camera.Fx / points_in_camera_space[:, 2] + fov_camera.Cx,
             points_in_camera_space[:, 1] * fov_camera.Fy / points_in_camera_space[:, 2] + fov_camera.Cy],
            -1).float() / scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) & \
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:, 2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self.gaussians.xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    @property
    def get_features(self):
        return self.gaussians.features

    @property
    def get_device(self):
        return self.device

    @property
    def get_scaling(self):
        return self.scaling_activation(self.gaussians.scaling)

    @property
    def get_learnable_scaling(self):
        return (self.scaling_activation(self.feature_samples[..., 1:4]))

    @property
    def get_rotation(self):
        return self.rotation_activation(self.gaussians.rotation)

    @property
    def get_xyz(self):
        return self.gaussians.xyz


    @property
    def get_opacity(self):
        return self.opacity_activation(self.gaussians.opacity)

    @property
    def get_pts3d(self):
        return self.xyz.view(-1, 3)

    @property
    def get_normals(self):
        return self.Sn.view(-1, 3)

    def refine_texture(self):
        self._refine_scales = True
        self.global_grids['rotations'][0].requires_grad_()
        self.global_grids['scaling'][0].requires_grad_(self._refine_scales)

    def reset_opacity(self):
        for g_lvl in range(self.active_grid_lvl+1):
            opacities_new = self.inverse_opacity_activation(
                torch.min(self.opacity_activation(self.global_grids['opacity'][g_lvl]), torch.ones_like(self.global_grids['opacity'][g_lvl]) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, f"opacity")
            self.global_grids['opacity'][g_lvl] = optimizable_tensors[f"opacity"].requires_grad_(True)

    def set_opacity(self, new_val=0.5):
        for g_lvl in range(self.active_grid_lvl+1):
            opacities_new = self.inverse_opacity_activation(
                torch.min(self.opacity_activation(self.global_grids['opacity'][g_lvl]), torch.ones_like(self.global_grids['opacity'][g_lvl]) * new_val))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, f"opacity_{g_lvl}")
            self.global_grids['opacity'][g_lvl] = optimizable_tensors[f"opacity_{g_lvl}"].requires_grad_(True)

    def training_setup(self, training_args):
        #################### Optimization Hyperparameters ####################
        self.spatial_lr_scale = self.spatial_lr_scale
        self.position_lr = training_args.position_lr_init
        self.feature_lr = training_args.feature_lr
        self.scale_params_lr = training_args.scaling_lr
        self.position_lr_final = training_args.position_lr_final
        self.opacity_lr = training_args.opacity_lr
        self.uv_lr = training_args.uv_grid_lr
        self.param_groups_tex = []
        g_lvl = 0
        param_groups = [
            {'params': self.global_grids['control_points'][g_lvl], 'lr': self.position_lr * self.spatial_lr_scale,  "name": f"control_points"},
            {'params': self.global_grids['features_dc'][g_lvl], 'lr': self.feature_lr, "name": f"features_dc"},
            {'params': self.global_grids['features_rest'][g_lvl], 'lr': self.feature_lr / 20, "name": f"features_rest"},
            {'params': self.global_grids['opacity'][g_lvl], 'lr': self.opacity_lr, "name": f"opacity"},
            {'params': self.global_grids['scaling'][g_lvl], 'lr': self.scale_params_lr, "name": f"scaling"},
            {'params': self.global_grids['control_weights'][g_lvl], 'lr': self.config.nurbs_weight_lr,
             "name": f"control_weights"},
            {'params': self.global_grids['rotations'][g_lvl], 'lr': self.config.rotation_lr, "name": f"rotations"}
            ]
        #
        # if self._refine_scales:
        #     param_groups.extend([{'params': self.global_grids['scaling'][g_lvl], 'lr': self.scale_params_lr, "name": f"scaling"}])
        # if self._refine_cp_weights:
        #     param_groups.extend([{'params': self.global_grids['control_weights'][g_lvl], 'lr': self.config.nurbs_weight_lr, "name": f"control_weights"}])
        # if self._refine_rotations:
        #     param_groups.extend([{'params': self.global_grids['rotations'][g_lvl], 'lr':  self.config.rotation_lr, "name": f"rotations"}])

        self.optimizer = torch.optim.Adam(
            param_groups, lr=0.0, eps=1e-15)

        self.optimizer_uv = None

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.position_lr * self.spatial_lr_scale,
            lr_final=self.position_lr_final * self.spatial_lr_scale,
            # lr_delay_mult=0.05,
            # lr_delay_steps=2000,
            max_steps=self.config.iterations,
        )
        self.scheduler_args = {"control_points": self.xyz_scheduler_args}
        plot_lr_schedule(self.xyz_scheduler_args, num_epochs=30_000)

    def replace_tensor_to_optimizer(self, tensor, name: str):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def initialize_materials(self, shape, device):
        return (
            nn.Parameter(torch.full(shape, 0.7, device=device), requires_grad=True),
            nn.Parameter(torch.full(shape[:-1] + (1,), 0.3, device=device), requires_grad=True),
            nn.Parameter(torch.full(shape[:-1] + (1,), 0.0, device=device), requires_grad=True),
        )

    # ====================== 2.  NEW method in SplineModel ====================
    def get_all_features(self, lvl):
        return torch.cat([
            self.global_grids["control_points"][lvl], # 0-3
            self.global_grids["control_weights"][lvl], # 3-4
            self.global_grids["opacity"][lvl], # 4-5
            self.global_grids["scaling"][lvl], # 5-8
            self.global_grids["rotations"][lvl], # 8-12
            self.global_grids["features_dc"][lvl], # 12-15
            self.global_grids["features_rest"][lvl], # 15-End
        ], dim=-1)

    def get_texture_features(self, lvl=0, pnames=None):
        return torch.cat([
            self.global_grids["opacity"][lvl], # 4-5
            self.global_grids["scaling"][lvl], # 5-8
            self.global_grids["rotations"][lvl], # 8-12
            self.global_grids["features_dc"][lvl], # 12-15
            self.global_grids["features_rest"][lvl], # 15-End
        ], dim=-1)

    @classmethod
    def restore(cls, state_dict):
        """
        Restore a Spline instance from a previously captured state dictionary.

        Args:
            state_dict (dict): State dictionary as produced by capture().
            device (str or torch.device, optional): Device to move tensors to. If None, use state_dict['device'].

        Returns:
            Spline: The reconstructed Spline instance, ready for interp().
        """
        from argparse import ArgumentParser, Namespace
        grids = state_dict['global_grids']
        # Use control_points from first level as dummy patches
        ctrl_pts = grids['control_points'][0]
        if hasattr(ctrl_pts, 'cuda'):
            ctrl_pts = ctrl_pts.cuda()
        device = 'cuda'

        config = NurbsOptimizationParams(ArgumentParser(description="Training script parameters"))
        # try to get colors (not used)
        spline = cls(
            ctrl_pts=ctrl_pts.cuda(),
            args=state_dict.get('model_params'),
            config=config,
            spatial_lr_scale=state_dict.get('spatial_lr_scale', 1.0),
        )
        # Restore grid_levels, active_grid_lvl, etc.
        spline.grid_levels = state_dict.get('grid_levels', spline.grid_levels)
        spline.active_grid_lvl = state_dict.get('active_grid_lvl', spline.active_grid_lvl)
        spline.p = state_dict.get('pe_levels', spline.pe_levels)
        spline.p = state_dict.get('pe_scale', spline.pe_scale)
        spline.pe_factors = 0.5 ** torch.arange(spline.pe_levels, device=spline.device)
        spline.p = state_dict.get('use_pos_enc', spline.use_pos_enc)
        spline.p = state_dict.get('p', spline.p)
        spline.q = state_dict.get('q', spline.q)
        spline._res = state_dict.get('_res', spline._res)
        spline._max_sh_degree = state_dict.get('_max_sh_degree', spline._max_sh_degree)
        spline.active_sh_degree = state_dict.get('active_sh_degree', spline.active_sh_degree)
        spline.num_patches = state_dict.get('num_patches', spline.num_patches)
        spline.patch_res = state_dict.get('patch_res', spline.patch_res)
        spline.patch_stride = state_dict.get('patch_stride', spline.patch_stride)
        # Restore config if possible
        if hasattr(spline, 'config') and state_dict.get('config', None) is not None:
            if isinstance(state_dict['config'], dict):
                for k, v in state_dict['config'].items():
                    setattr(spline.config, k, v)
        # Restore global_grids
        for pname, plist in grids.items():
            for i, tensor in enumerate(plist):
                # Move to device and make nn.Parameter
                param = nn.Parameter(tensor.to(device))
                spline.global_grids[pname][i] = param
        # Restore knot vectors
        if state_dict.get('knot_u', None) is not None:
            spline.knot_u = state_dict['knot_u'].to(device)
        if state_dict.get('knot_v', None) is not None:
            spline.knot_v = state_dict['knot_v'].to(device)
        # Restore basis functions if present, else recompute
        if state_dict.get('BU', None) is not None and state_dict.get('BV', None) is not None:
            spline.BU = state_dict['BU'].to(device)
            spline.dBU = state_dict['dBU'].to(device) if state_dict.get('dBU', None) is not None else None
            spline.dBU = state_dict['dBUU'].to(device) if state_dict.get('dBUU', None) is not None else None
            spline.BV = state_dict['BV'].to(device)
            spline.dBV = state_dict['dBV'].to(device) if state_dict.get('dBV', None) is not None else None
            spline.dBV = state_dict['dBVV'].to(device) if state_dict.get('dBVV', None) is not None else None
        else:
            # Recompute basis
            spline.BU, spline.dBU, spline.dBUU, spline.BV, spline.dBV, spline.dBVV = spline.init_knots(spline._res)

        spline.xyz = state_dict['xyz'].to(device)
        spline.dSu = state_dict['dSu'].to(device) if state_dict.get('dSu', None) is not None else None
        spline.dSv = state_dict['dSv'].to(device)
        spline.Sn = state_dict['Sn'].to(device) if state_dict.get('dSu', None) is not None else None

        spline.opacities = state_dict['opacities'].to(device)
        spline.scalings = state_dict['scalings'].to(device) if state_dict.get('dSu', None) is not None else None
        spline.rots = state_dict['rots'].to(device)
        spline.spherical_harmonics = state_dict['sh_feat'].to(device) if state_dict.get('sh_feat', None) is not None else None

        spline.use_pos_enc = state_dict.get('use_pos_enc', False)
        if ctrl_pts.shape[-1] > 3:
            spline.use_pos_enc = True
        # Restore optimizer state if present and optimizer exists
        if state_dict.get('optimizer_state_dict', None) is not None and spline.optimizer is not None:
            try:
                spline.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            except Exception:
                pass
        # Restore iteration if present
        if 'iteration' in state_dict:
            spline.iteration = state_dict['iteration']
        return spline
    def capture(self):
        """
        Capture the current state of the Spline model, including all parameter tensors,
        optimizer state (if present), scalar attributes, configuration, knot vectors,
        and basis functions.

        Returns:
            dict: A dictionary containing all information required to fully restore this Spline instance.
        """
        state = {}
        # Capture all grids (all levels for each param)
        state['global_grids'] = {}
        for pname, plist in self.global_grids.items():
            state['global_grids'][pname] = [p.detach().cpu() for p in plist]
        # Optimizer state (if exists)
        if self.optimizer is not None:
            try:
                state['optimizer_state_dict'] = self.optimizer.state_dict()
            except Exception:
                state['optimizer_state_dict'] = None
        else:
            state['optimizer_state_dict'] = None

        state['use_pos_enc'] = self.use_pos_enc
        # Scalar attributes and settings
        state['grid_levels'] = self.grid_levels
        state['model_params'] = self.model_params
        state['active_grid_lvl'] = self.active_grid_lvl
        state['p'] = self.p
        state['q'] = self.q
        state['device'] = str(self.device)
        state['patch_res'] = self.patch_res
        state['patch_stride'] = self.patch_stride
        state['_res'] = self._res
        state['_max_sh_degree'] = self._max_sh_degree
        state['active_sh_degree'] = self.active_sh_degree
        state['num_patches'] = self.num_patches
        # Config as dict if possible
        if hasattr(self.config, '__dict__'):
            state['config'] = dict(self.config.__dict__)
        else:
            state['config'] = self.config

        state['xyz'] = self.xyz.detach().cpu() if hasattr(self, 'xyz') else None
        state['dSu'] = self.dSu.detach().cpu() if hasattr(self, 'dSu') else None
        state['dSv'] = self.dSv.detach().cpu() if hasattr(self, 'dSv') else None
        state['Sn'] = self.Sn.detach().cpu() if hasattr(self, 'Sn') else None

        state['sh_feat'] = self.spherical_harmonics.detach().cpu() if hasattr(self, 'spherical_harmonics') else None
        state['rots'] = self.rots.detach().cpu() if hasattr(self, 'rots') else None
        state['scalings'] = self.scalings.detach().cpu() if hasattr(self, 'scalings') else None
        state['opacities'] = self.opacities.detach().cpu() if hasattr(self, 'opacities') else None

        state['knot_u'] = self.knot_u.detach().cpu() if hasattr(self, 'knot_u') else None
        state['knot_v'] = self.knot_v.detach().cpu() if hasattr(self, 'knot_v') else None
        # Basis functions (if not recomputed at load)
        state['BU'] = self.BU.detach().cpu() if hasattr(self, 'BU') else None
        state['dBU'] = self.dBU.detach().cpu() if hasattr(self, 'dBU') else None
        state['dBUU'] = self.dBU.detach().cpu() if hasattr(self, 'dBUU') else None
        state['BV'] = self.BV.detach().cpu() if hasattr(self, 'BV') else None
        state['dBV'] = self.dBV.detach().cpu() if hasattr(self, 'dBV') else None
        state['dBVV'] = self.dBV.detach().cpu() if hasattr(self, 'dBVV') else None
        # Optionally, iteration and other relevant state
        state['iteration'] = getattr(self, 'iteration', 0)
        # Add any additional scalar attributes here as needed
        return state

    @property
    def c_pts(self, lvl=0):
        return self.global_grids["control_points"][lvl]

    @property
    def c_wts(self, lvl=0):
        return self.global_grids["control_weights"][lvl].sigmoid().squeeze()

    @property
    def control_weighted_enc(self, lvl=0):
        return self._latent_to_xyz(self.global_grids["control_points"][lvl]).view(self.H, self.W, 3) * self.global_grids["control_weights"][lvl].sigmoid()

    @property
    def cpts_enc(self):
        return self._latent_to_xyz(self.global_grids["control_points"][0]).view(self.H, self.W, 3)

    @property
    def c_ptsw(self, lvl=0):
        return self.global_grids["control_points"][lvl] * self.global_grids["control_weights"][lvl].sigmoid()

    @property
    def opacity(self, lvl=0):
        return self.global_grids["opacity"][lvl]

    @property
    def c_scale(self, lvl=0):
        return self.global_grids["scaling"][lvl]

    @property
    def c_rotations(self, lvl=0):
        return self.global_grids["rotations"][lvl]

    @property
    def c_featute_dc(self, lvl=0):
        return self.global_grids["features_dc"][lvl]
    @property
    def c_featute_rest(self, lvl=0):
        return self.global_grids["features_rest"][lvl]


    @classmethod
    def from_pointcloud(cls, xyz, uv, args, config, scene_extent,
                        grid_res=(128,128), degree=(3,3), device='cuda'):
        from model.utils.spline_local_init import seed_control_grid  # top of file

        ctrl, ku, kv = seed_control_grid(
            xyz.to(device), uv.to(device),
            grid_res=grid_res, degree=degree)
        spline = cls(ctrl_pts=ctrl, args=args, config=config, spatial_lr_scale=scene_extent)
        spline.init_control_grid(ctrl.to(device))
        spline.set_knot_u(ku)
        spline.set_knot_v(kv)
        return spline