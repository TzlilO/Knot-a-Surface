"""
Differentiable basis function computation from arbitrary UV samples.
"""
import opt_einsum as oe
import torch
from typing import Tuple, Optional, Dict

from torch import nn as nn

from modules.basis.basis_matrix import DifferentiableBSplineBasis
from modules.basis.cox_de_boor import compute_all_derivatives
from modules.spline_utils import BasisFuncs


def grid_bspline_basis_matrices(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    # This implementation is optimized for the cubic (p=3) case.
    # For other degrees, you would need a general-purpose Cox-de-Boor implementation.
    if degree != 3:
        raise NotImplementedError("This fast implementation is optimized for cubic (degree=3) splines. "
                                  "Use a general Cox-de-Boor implementation for other degrees.")

    # Capture original shape to restore later
    original_shape = samples.shape
    # Flatten samples to 1D for vectorized processing
    samples = samples.reshape(-1)

    # --- 1. Find the segment index and local parameter t for each sample u ---
    # Scale samples to the range of the interior knots
    # knots = knots.T
    u = samples * (knots[num_control_points] - knots[degree]) + knots[degree]

    # This is differentiable and works for non-uniform knots.
    segment_idx = torch.searchsorted(knots, u, right=True).clamp(min=degree, max=num_control_points) - 1

    # Normalize u to a local parameter t_local within [0, 1] for its segment
    knot_start = knots[segment_idx]
    knot_end = knots[segment_idx + 1]
    segment_length = knot_end - knot_start
    # Avoid division by zero for stacked knots
    t_local = (u - knot_start) / (segment_length + 1e-9)

    # --- 2. Compute Blending Functions and their Derivatives ---
    t2 = t_local * t_local
    t3 = t2 * t_local

    # Basis functions (B-spline blending functions for a cubic segment)
    b0 = (1 - 3 * t_local + 3 * t2 - t3) / 6
    b1 = (4 - 6 * t2 + 3 * t3) / 6
    b2 = (1 + 3 * t_local + 3 * t2 - 3 * t3) / 6
    b3 = t3 / 6
    weights = torch.stack([b0, b1, b2, b3], dim=-1)

    # First derivatives of blending functions w.r.t. t_local
    db0 = (-3 + 6 * t_local - 3 * t2) / 6
    db1 = (-12 * t_local + 9 * t2) / 6
    db2 = (3 + 6 * t_local - 9 * t2) / 6
    db3 = 3 * t2 / 6
    d_weights = torch.stack([db0, db1, db2, db3], dim=-1)

    # Second derivatives of blending functions w.r.t. t_local
    d2b0 = (6 - 6 * t_local) / 6
    d2b1 = (-12 + 18 * t_local) / 6
    d2b2 = (6 - 18 * t_local) / 6
    d2b3 = 6 * t_local / 6
    d2_weights = torch.stack([d2b0, d2b1, d2b2, d2b3], dim=-1)

    # --- 3. Apply Chain Rule for non-uniform knot intervals ---
    # The derivative w.r.t. u is (d/dt_local) * (dt_local/du)
    # dt_local_du = 1.0 / (segment_length + 1e-9)

    # d_weights_scaled = d_weights * dt_local_du.unsqueeze(-1)
    # d2_weights_scaled = d2_weights * (dt_local_du.unsqueeze(-1) ** 2)

    # --- 4. Construct the final sparse basis matrices ---
    N = torch.zeros(u.shape[0], num_control_points, device=device)
    dN = torch.zeros(u.shape[0], num_control_points, device=device)
    d2N = torch.zeros(u.shape[0], num_control_points, device=device)

    # The 4 influencing control points for each sample start at segment_idx - degree
    indices = (segment_idx - degree).unsqueeze(-1) + torch.arange(4, device=device).unsqueeze(0)

    # Scatter the weights and their derivatives into the correct locations
    # N.scatter(-1, indices, weights)
    N = torch.scatter(N, -1, indices, weights)
    dN = torch.scatter(dN, -1, indices, d_weights)
    d2N = torch.scatter(d2N, -1, indices, d2_weights)

    # Reshape back to original dimensions + control points dim
    # E.g. [Us, Vs, num_control_points]
    output_shape = original_shape + (num_control_points,)
    return N.reshape(output_shape), dN.reshape(output_shape), d2N.reshape(output_shape)


def generate_bspline_basis_matrices_d(
        samples_u: torch.Tensor,  # [Us] with requires_grad=True
        samples_v: torch.Tensor,  # [Vs] with requires_grad=True
        knots_u: torch.Tensor,
        knots_v: torch.Tensor,
        num_control_u: int,  # H
        num_control_v: int,  # W
        degree: int = 3
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate B-spline basis matrices WITH derivatives for grid control points.

    CRITICAL: Fully differentiable for interval optimization.

    Returns:
        Dict with keys:
            'basis': (bu, bv) - 0th order basis
            'deriv1': (dbu, dbv) - 1st derivatives
            'deriv2': (dbuu, dbvv) - 2nd derivatives
    """
    from modules.basis.basis_matrix import DifferentiableBSplineBasis

    basis_computer = DifferentiableBSplineBasis(degree=degree, device='cuda')

    # Ensure gradient tracking
    if not samples_u.requires_grad or not samples_v.requires_grad:
        print("[WARNING] Sample points should have requires_grad=True for gradient flow")

    # Compute 0th order (positions)
    bu, bv = basis_computer.forward(
        samples_u, samples_v,
        knots_u, knots_v,
        num_control_u, num_control_v
    )

    # Compute 1st derivatives (velocities)
    dbu, dbv = basis_computer.compute_derivatives(
        samples_u, samples_v,
        knots_u, knots_v,
        num_control_u, num_control_v,
        order=1
    )

    # Compute 2nd derivatives (accelerations)
    dbuu, dbvv = basis_computer.compute_derivatives(
        samples_u, samples_v,
        knots_u, knots_v,
        num_control_u, num_control_v,
        order=2
    )


    return {
        'basis': (bu, bv),
        'deriv1': (dbu, dbv),
        'deriv2': (dbuu, dbvv)
    }

def bspline_basis_and_derivs_1d(
    u: torch.Tensor,        # [M] parameter values
    knots: torch.Tensor,    # [n_ctrl + degree + 1] knot vector
    degree: int,
    max_deriv: int = 2,
) -> Tuple[torch.Tensor, ...]:
    """
    Exact B-spline basis values and derivatives (triangular-table algorithm,
    The NURBS Book A2.3). Fully vectorized over samples, differentiable
    w.r.t. both `u` and `knots`, valid for non-uniform knot vectors.

    Returns (N, dN, ..., d^k N), each [M, n_ctrl].
    """
    device = u.device
    dtype = u.dtype
    m = knots.shape[0] - 1
    n = m - degree
    M = u.shape[0]

    u = u.reshape(-1)
    u_col = u.unsqueeze(1)

    # ── Build the full triangular table ──────────────────────────────────────
    # table[d] stores N_{i,d}(u) for all i, shape [M, m-d]
    table = {}

    # Degree 0
    left_edges = knots[:-1].unsqueeze(0)
    right_edges = knots[1:].unsqueeze(0)
    N0 = ((u_col >= left_edges) & (u_col < right_edges)).to(dtype)

    # Right endpoint fix
    last_span_mask = (u == knots[m])
    if last_span_mask.any():
        for j in range(m - 1, -1, -1):
            if knots[j] < knots[j + 1]:
                N0[last_span_mask, j] = 1.0
                break

    table[0] = N0[:, :m]  # [M, m]

    # Degrees 1 through p
    for d in range(1, degree + 1):
        n_d = m - d  # number of basis functions at degree d
        prev = table[d - 1]

        t_left = knots[:n_d].unsqueeze(0)
        t_right = knots[d:d + n_d].unsqueeze(0)
        denom_l = t_right - t_left

        t_left_r = knots[1:n_d + 1].unsqueeze(0)
        t_right_r = knots[d + 1:d + 1 + n_d].unsqueeze(0)
        denom_r = t_right_r - t_left_r

        alpha = torch.where(
            denom_l.abs() > 1e-14,
            (u_col - t_left) / denom_l,
            torch.zeros(1, device=device, dtype=dtype)
        )

        beta = torch.where(
            denom_r.abs() > 1e-14,
            (t_right_r - u_col) / denom_r,
            torch.zeros(1, device=device, dtype=dtype)
        )

        table[d] = alpha * prev[:, :n_d] + beta * prev[:, 1:n_d + 1]

    # ── Extract basis values ─────────────────────────────────────────────────
    B = table[degree][:, :n]  # [M, n]

    results = [B]

    for k in range(1, max_deriv + 1):
        if degree - k < 0:
            # Derivative order exceeds degree → result is zero
            results.append(torch.zeros(M, n, device=device, dtype=dtype))
            continue

        # Start from degree-(p-k) table
        source_degree = degree - k
        n_source = m - source_degree  # number of functions at source degree

        # Clone the source to avoid modifying the table
        D = table[source_degree][:, :n_source].clone()

        current_n = n_source

        for step in range(k):

            p_eff = source_degree + step + 1  # = (p-k) + step + 1
            new_n = current_n - 1

            # Vectorized computation over all i at once
            # Knot spans for left term: t_{i+p_eff} - t_i, for i = 0..new_n-1
            t_lo = knots[:new_n]                    # [new_n]
            t_hi = knots[p_eff:p_eff + new_n]      # [new_n]
            denom_left = (t_hi - t_lo).unsqueeze(0) # [1, new_n]

            # Knot spans for right term: t_{i+1+p_eff} - t_{i+1}, for i = 0..new_n-1
            t_lo_r = knots[1:new_n + 1]                     # [new_n]
            t_hi_r = knots[p_eff + 1:p_eff + 1 + new_n]    # [new_n]
            denom_right = (t_hi_r - t_lo_r).unsqueeze(0)    # [1, new_n]

            left_term = torch.where(
                denom_left.abs() > 1e-14,
                D[:, :new_n] / denom_left,
                torch.zeros(1, device=device, dtype=dtype)
            )

            right_term = torch.where(
                denom_right.abs() > 1e-14,
                D[:, 1:new_n + 1] / denom_right,
                torch.zeros(1, device=device, dtype=dtype)
            )

            D = p_eff * (left_term - right_term)  # [M, new_n]
            current_n = new_n

        assert D.shape[1] == n, (
            f"Derivative shape mismatch: got {D.shape[1]}, expected {n}. "
            f"degree={degree}, k={k}, m={m}, source_degree={source_degree}"
        )
        results.append(D)

    return tuple(results)


def compute_all_derivatives_2d(
    uv: torch.Tensor,  # [Us x Vs x 2] or [N x 2] evaluation parameters for uv space
    knots_u: torch.Tensor,  # [nu + pu + 1] knot vector for u
    knots_v: torch.Tensor,  # [nv + pv + 1] knot vector for v
    degree_u: int,  # Polynomial degree pu for u
    degree_v: int,  # Polynomial degree pv for v
    max_deriv: int = 2  # Maximum derivative order
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    # Handle input shape: assume [Us, Vs, 2] or [N, 2]
    if uv.dim() == 3:
        # [Us, Vs, 2] -> flatten to [Us*Vs]
        u = uv[..., 0].reshape(-1)
        v = uv[..., 1].reshape(-1)
    elif uv.dim() == 2:
        # [N, 2]
        u = uv[:, 0]
        v = uv[:, 1]
    else:
        raise ValueError(f"uv must be of shape [Us, Vs, 2] or [N, 2]. Got {uv.shape}")

    results_u = bspline_basis_and_derivs_1d(u, knots_u, degree_u, max_deriv)
    results_v = bspline_basis_and_derivs_1d(v, knots_v, degree_v, max_deriv)
    return results_u, results_v


def compute_bases_uv_diff(
        samples_u: torch.Tensor,  # [Us]
        samples_v: torch.Tensor,  # [Vs]
        knots_u: torch.Tensor,
        knots_v: torch.Tensor,
        num_control_u: int,  # H
        num_control_v: int,  # W
        degree: int = 3,
        device='cuda') -> BasisFuncs:
    """
    Separable basis matrices with values + first/second derivatives,
    differentiable w.r.t. samples and knots (exact, non-uniform-safe).
    """
    bu, dbu, dbuu = bspline_basis_and_derivs_1d(
        samples_u.reshape(-1), knots_u, degree, max_deriv=2
    )
    bv, dbv, dbvv = bspline_basis_and_derivs_1d(
        samples_v.reshape(-1), knots_v, degree, max_deriv=2
    )
    assert bu.shape[-1] == num_control_u and bv.shape[-1] == num_control_v, (
        f"Basis/control mismatch: bu {tuple(bu.shape)} vs H={num_control_u}, "
        f"bv {tuple(bv.shape)} vs W={num_control_v}"
    )
    return BasisFuncs(
        bu_data={'basis': bu, 'deriv1': dbu, 'deriv2': dbuu},
        bv_data={'basis': bv, 'deriv1': dbv, 'deriv2': dbvv},
    )

def compute_basis_u(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    return compute_bases_uv_diff(samples, knots, num_control_points, degree, device)

def compute_basis_v(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    return compute_bases_uv_diff(samples, knots, num_control_points, degree, device)
def compute_bases_uv(samples_u, samples_v, knots_u, knots_v, num_control_u: int, num_control_v: int, degree=3,
                     device='cuda', uv_grid=None) -> BasisFuncs:
    return compute_bases_uv_diff(
        samples_u, samples_v, knots_u, knots_v,
        num_control_u, num_control_v, degree=degree, device=device,
    )
@torch.jit.script
def cox_de_boor_basis_and_derivative(
        u:       torch.Tensor,   # (N,)
        U:       torch.Tensor,   # (n_ctrl+degree+1,)
        degree:  int,
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
def generate_bspline_basis_matrices(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    """
    Computes the B-spline basis matrix and its derivatives for a given knot vector.
    It uses a highly efficient, vectorized matrix formulation for the common cubic case (p=3)
    and can fall back to a general recursive algorithm for other degrees.

    Args:
        samples (torch.Tensor): Parameter values for evaluation, assumed to be in [0, 1].
        num_control_points (int): The number of control points (e.g., self.H or self.W).
        degree (int): The polynomial degree of the spline.
        knots (torch.Tensor): The learnable or fixed knot vector.
        device (str): The device to create tensors on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The basis matrix B, and its
        first (dB) and second (d2B) derivatives.
    """
    # This implementation is optimized for the cubic (p=3) case.
    # For other degrees, you would need a general-purpose Cox-de-Boor implementation.
    if degree != 3:
        raise NotImplementedError("This fast implementation is optimized for cubic (degree=3) splines. "
                                  "Use a general Cox-de-Boor implementation for other degrees.")

    # --- 1. Find the segment index and local parameter t for each sample u ---
    # Scale samples to the range of the interior knots
    # knots = knots.T
    u = samples * (knots[num_control_points] - knots[degree]) + knots[degree]

    # This is differentiable and works for non-uniform knots.
    segment_idx = torch.searchsorted(knots, u, right=True).clamp(min=degree, max=num_control_points) - 1

    # Normalize u to a local parameter t_local within [0, 1] for its segment
    knot_start = knots[segment_idx]
    knot_end = knots[segment_idx + 1]
    segment_length = knot_end - knot_start
    # Avoid division by zero for stacked knots
    t_local = (u - knot_start) / (segment_length + 1e-9)

    # --- 2. Compute Blending Functions and their Derivatives ---
    t2 = t_local * t_local
    t3 = t2 * t_local

    # Basis functions (B-spline blending functions for a cubic segment)
    b0 = (1 - 3 * t_local + 3 * t2 - t3) / 6
    b1 = (4 - 6 * t2 + 3 * t3) / 6
    b2 = (1 + 3 * t_local + 3 * t2 - 3 * t3) / 6
    b3 = t3 / 6
    weights = torch.stack([b0, b1, b2, b3], dim=-1)

    # First derivatives of blending functions w.r.t. t_local
    db0 = (-3 + 6 * t_local - 3 * t2) / 6
    db1 = (-12 * t_local + 9 * t2) / 6
    db2 = (3 + 6 * t_local - 9 * t2) / 6
    db3 = 3 * t2 / 6
    d_weights = torch.stack([db0, db1, db2, db3], dim=-1)

    # Second derivatives of blending functions w.r.t. t_local
    d2b0 = (6 - 6 * t_local) / 6
    d2b1 = (-12 + 18 * t_local) / 6
    d2b2 = (6 - 18 * t_local) / 6
    d2b3 = 6 * t_local / 6
    d2_weights = torch.stack([d2b0, d2b1, d2b2, d2b3], dim=-1)

    # --- 3. Apply Chain Rule for non-uniform knot intervals ---
    # The derivative w.r.t. u is (d/dt_local) * (dt_local/du)
    # dt_local_du = 1.0 / (segment_length + 1e-9)

    # d_weights_scaled = d_weights * dt_local_du.unsqueeze(-1)
    # d2_weights_scaled = d2_weights * (dt_local_du.unsqueeze(-1) ** 2)

    # --- 4. Construct the final sparse basis matrices ---
    N = torch.zeros(u.shape[0], num_control_points, device=device)
    dN = torch.zeros(u.shape[0], num_control_points, device=device)
    d2N = torch.zeros(u.shape[0], num_control_points, device=device)

    # The 4 influencing control points for each sample start at segment_idx - degree
    indices = ((segment_idx - degree).unsqueeze(-1) + torch.arange(4, device=device).unsqueeze(0))#.clamp(min=degree, max=num_control_points - 1)

    # Scatter the weights and their derivatives into the correct locations
    # N.scatter(-1, indices, weights)
    N = torch.scatter(N, -1, indices, weights)
    dN = torch.scatter(dN, -1, indices, d_weights)
    d2N = torch.scatter(d2N, -1, indices, d2_weights)
    return N, dN, d2N


def find_span_vectorized(
        u: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        n_ctrl: int
) -> torch.Tensor:
    """
    Find knot span index for each parameter value (vectorized).
    """
    # Clamp u to valid range (non-inplace)
    u_clamped = u.clamp(knots[degree].item(), knots[n_ctrl].item() - 1e-10)

    # Use searchsorted to find spans
    span = torch.searchsorted(knots[degree: n_ctrl + 1].contiguous(), u_clamped.contiguous(), right=True)
    span = span + degree - 1

    # Clamp to valid range (non-inplace)
    span = span.clamp(degree, n_ctrl - 1)

    return span


def evaluate_surface(
        Bu: torch.Tensor,
        Bv: torch.Tensor,
        control_points: torch.Tensor,
        is_rational: bool = False
) -> torch.Tensor:
    """
    Evaluate B-spline/NURBS surface given basis functions and control points.

    Args:
        Bu: [... , n_ctrl_u] U basis values
        Bv:  [..., n_ctrl_v] V basis values
        control_points: [n_ctrl_u, n_ctrl_v, dim] control point grid
                       For NURBS, dim includes weight as first channel
        is_rational: If True, treat as NURBS with weights

    Returns:
        surface_points: [... , dim] or [... , dim-1] for NURBS
    """
    # Use einsum for efficient contraction
    # Bu:  [... , H], Bv: [..., W], P: [H, W, C] -> [..., C]

    result = torch.einsum('...h,...w,hwc->...c', Bu, Bv, control_points)

    if is_rational:
        # First channel is weight, project back
        weights = result[..., : 1].clamp(min=1e-6)
        result = result[..., 1:] / weights

    return result


def make_clamped_uniform_knots(
        n_ctrl: int,
        degree: int,
        device: str = 'cuda'
) -> torch.Tensor:
    """
    Create clamped uniform knot vector.

    Args:
        n_ctrl: Number of control points
        degree: Polynomial degree
        device:  Torch device

    Returns:
        knots: [n_ctrl + degree + 1] knot vector
    """
    n_knots = n_ctrl + degree + 1
    n_internal = n_knots - 2 * (degree + 1)

    # Clamped:  degree+1 zeros, internal, degree+1 ones
    zeros = torch.zeros(degree + 1, device=device)
    ones = torch.ones(degree + 1, device=device)

    if n_internal > 0:
        internal = torch.linspace(0, 1, n_internal + 2, device=device)[1:-1]
    else:
        internal = torch.tensor([], device=device)

    knots = torch.cat([zeros, internal, ones])

    return knots


class BasisFunction(nn.Module):
    """
    Abstract base module for spline basis functions.
    Subclasses must implement _compute_1d_basis_and_derivs.
    """
    # _basis_funcs: BasisFuncs
    _uniform_basis_funcs: BasisFuncs
    _optimal_path: Optional[str] = 'auto'
    def __init__(
            self,
            state:'ModelState',
            sampling_module: 'SamplerUV' = None,
            knot_u:'KnotVector' = None,
            knot_v:'KnotVector' = None,
            # init_basis_u=Optional[Dict[str, Tuple[torch.Tensor]]],
            # init_basis_v=Optional[Dict[str, Tuple[torch.Tensor]]],
    ):
        super(BasisFunction, self).__init__()
        self.uv_sampler = sampling_module
        self.ndim_interp = state.full_basis
        self.state = state
        self.knot_u = knot_u
        self.knot_v = knot_v
        self._basis_funcs = BasisFuncs()
        # if init_basis_u is not None and init_basis_v is not None:
        # self.recompute()
    def clear(self):
        # if hasattr(self, '_basis_funcs'):
        try:
            self._basis_funcs.clear()
        except:
            pass




    @property
    def basis_layout_v(self):
        if self.state.full_basis:
            UV = (self.state.Us * self.state.Vs, ) if self.state.flatten_uv else (self.state.Us, self.state.Vs)
            return (*UV, self.state.W) if self.state.full_basis else (self.state.Vs, self.state.W)

        return self.state.Vs, self.state.W

    @property
    def basis_layout_u(self):
        if self.state.full_basis:
            UV = (self.state.Us * self.state.Vs, ) if self.state.flatten_uv else (self.state.Us, self.state.Vs)
            return (*UV, self.state.H) if self.state.full_basis else (self.state.Us, self.state.H)
        else:
            return self.state.Us, self.state.H

    def calc_basis_layout_u(self, Us, Vs):
        UV = (Us * Vs,) if self.state.flatten_uv else (Us, Vs)
        return (*UV, self.state.H) if self.state.full_basis else (Us, self.state.H)


    def calc_basis_layout_v(self, Us, Vs):
        UV = (Us * Vs,) if self.state.flatten_uv else (Us, Vs)

        return(*UV, self.state.W) if self.state.full_basis else (Vs, self.state.W)

    @property
    def contract_path(self):
        if self.state.full_basis:
            contract_path = 'fh,hwc,fw -> fc' if self.bu.ndim == 2 else 'uvh,hwc,uvw->uvc'# 'uvh,uvw,hwc->uvc'
            return contract_path
        else:
            return 'uh,hwc,vw->uvc'
    @property
    def uniform_contract_path(self):
        return 'uh,vw,hwc->uvc'

    @property
    def optimal_path(self):
        return self._optimal_path if self._optimal_path is not None else 'auto'

    def set_knot_u(self, knot_u: 'KnotVector'):
        self.knot_u = knot_u

    def set_basis_u(self, u_basis: Tuple[torch.Tensor]):
        if u_basis is None:
            raise ValueError("u_basis cannot be None")
        self._basis_funcs.set_bu(u_basis)

    def set_basis_v(self, v_basis: Tuple[torch.Tensor]):
        if v_basis is None:
            raise ValueError("v_basis cannot be None")
        self._basis_funcs.set_bv(v_basis)

    def replace_funcs(self, buv: BasisFuncs):
        if not hasattr(self, '_basis_funcs') or self._basis_funcs is None:
            self._basis_funcs = buv
            return
        self._basis_funcs.bu = buv.bu
        self._basis_funcs.dbu = buv.dbu
        self._basis_funcs.dbuu = buv.dbuu

        self._basis_funcs.bv = buv.bv
        self._basis_funcs.dbv = buv.dbv
        self._basis_funcs.dbvv = buv.dbvv
    def set_knot_v(self, knot_v: 'KnotVector'):
        self.knot_v = knot_v
    def forward(self, samples_uv, knot_u, knot_v, **kwargs):
        self.clear()
        compute = compute_bases_uv_diff if self.state.opt.optimize_intervals else compute_bases_uv
        # compute = compute_bases_uv_diff
        buv = compute(
            *samples_uv,
            knot_u,
            knot_v,
            self.state.H,
            self.state.W,
            degree=3,
        )
        self.replace_funcs(buv)
        return buv
    
    def uniform_knots_case(self, samples, knots=None, num_ctrl=None, degree=3):
        return compute_bases_uv_diff(samples, knots=knots, num_control_points=num_ctrl, degree=degree)
    def set_optimal_path(self):
        if not hasattr(self, '_optimal_path'):
            try:
                self._optimal_path, _ = oe.contract_path(self.contract_path,
                                                         self.bu.shape,
                                                         self.bv.shape,
                                                         (self.state.H, self.state.W, 3),
                                                         shapes=True)

            except Exception:
                # Fall back to a valid opt_einsum strategy name — the
                # contract STRING is not a valid `optimize=` argument.
                self._optimal_path = 'auto'


    @property
    def Us(self):
        return self.state.Us

    @property
    def Vs(self):
        return self.state.Vs

    @property
    def sampling_layout(self):
        return (self.state.Us, self.state.Vs, -1)


    @property
    def bug(self):
        return self._basis_funcs.bu.view(self.state.Bu_layout)

    @property
    def bvg(self):
        return self._basis_funcs.bv.view(self.state.Bv_layout)

    @property
    def dbug(self):
        return self._basis_funcs.dbu.view(self.state.Bu_layout)

    @property
    def dbvg(self):
        return self._basis_funcs.dbv.view(self.state.Bv_layout)
    @property
    def dbuug(self):
        return self._basis_funcs.dbuu.view(self.state.Bu_layout)
    @property
    def dbvvg(self):
        return self._basis_funcs.dbvv.view(self.state.Bv_layout)
    @property
    def Bdv(self):
        if self._Bdv is None:
            self._Bdv = oe.contract(self.bug, self.dbvg)
        return self._Bdv  # Cache hit: Direct return, no autograd overhead
    @property
    def Bdu(self):
        if self._Bdu is None:
            self._Bdu = oe.contract(self.state.hw2uv, self.dbug, self.bvg)  # , optimize='dp')
        return self._Bdu

    @property
    def Buv(self):
        if self._Buv is None:
            self._Buv = oe.contract(self.state.hw2uv, self.bug, self.bvg) #, optimize='dp')

        return self._Buv

    def recompute(self):
        compute = compute_bases_uv_diff #if self.state.opt.optimize_intervals else compute_bases_uv
        # compute = compute_bases_uv
        # self.clear()
        # u = torch.linspace(0.0, 1.0, self.state.Us, device=self.state.device)
        # v = torch.linspace(0.0, 1.0, self.state.Vs, device=self.state.device)
        basis_data = compute(
            self.uv_sampler.interval_u,
            self.uv_sampler.interval_v,
            self.knot_u(),
            self.knot_v(),
            self.state.H,
            self.state.W,
            degree=3,
        )
        self.replace_funcs(basis_data)

    @property
    def buv(self):
        if self._basis_funcs.bu is None:
            self.recompute()
        return oe.contract('...h,...w->...hw', self.bu.T, self.bv.T)
    @property
    def bu(self):
        if self._basis_funcs.bu is None:
            self.recompute()
        return self._basis_funcs.bu

    @property
    def bv(self):
        if self._basis_funcs.bv is None:
            self.recompute()
        return self._basis_funcs.bv


    @property
    def dbu(self):
        if self._basis_funcs.dbu is None:
            self.recompute()
        return self._basis_funcs.dbu


    @property
    def dbv(self):
        if self._basis_funcs.dbv is None:
            self.recompute()
        return self._basis_funcs.dbv

    @property
    def dbuu(self):
        if self._basis_funcs.dbuu is None:
            self.recompute()
        return self._basis_funcs.dbuu

    @property
    def dbvv(self):
        if self._basis_funcs.dbvv is None:
            self.recompute()
        return self._basis_funcs.dbvv


    def capture_state(self) -> dict:
        """Capture basis state (mostly configuration, computed on restore)."""
        state = {
            'degree': self.state.degree,
            'deriv_order': getattr(self, 'deriv_order', 1),
        }

        # Cache current basis matrices if computed
        if hasattr(self, '_Buv') and self._Buv is not None:
            state['bu'] = self.bu.clone().cpu()
            state['bv'] = self.bv.clone().cpu()

        if hasattr(self, '_Bdu') and self._Bdu is not None:
            state['dbu'] = self.dbu.clone().cpu()
            state['dbv'] = self.dbv.clone().cpu()

        # Contract path for einsum optimization
        state['contract_path'] = self.contract_path

        return state

    @classmethod
    def from_state(
            cls,
            state: dict,
            model_state: 'ModelState',
            knot_u: 'KnotVector' = None,
            knot_v: 'KnotVector' = None,
            device: str = 'cuda'
    ) -> 'BasisFunction':
        """Restore BasisFunction."""

        instance = cls(model_state)

        # Restore cached basis if available
        if 'bu' in state and 'bv' in state:
            instance._bu = state['bu'].to(device)
            instance._bv = state['bv'].to(device)
            instance._Buv = True  # Mark as computed

        if 'dbu' in state and 'dbv' in state:
            instance._dbu = state['dbu'].to(device)
            instance._dbv = state['dbv'].to(device)
            instance._Bdu = True
            instance._Bdv = True

        return instance