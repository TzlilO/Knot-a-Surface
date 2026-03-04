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

def compute_all_derivatives_2d(
    uv: torch.Tensor,  # [Us x Vs x 2] or [N x 2] evaluation parameters for uv space
    knots_u: torch.Tensor,  # [nu + pu + 1] knot vector for u
    knots_v: torch.Tensor,  # [nv + pv + 1] knot vector for v
    degree_u: int,  # Polynomial degree pu for u
    degree_v: int,  # Polynomial degree pv for v
    max_deriv: int = 2  # Maximum derivative order
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:

    # Helper function for 1D computation (copied from the provided 1D implementation)
    def compute_1d(u: torch.Tensor, knots: torch.Tensor, degree: int, max_deriv: int) -> Tuple[torch.Tensor, ...]:
        device = u.device
        dtype = u.dtype
        m = knots.shape[0] - 1
        n = m - degree
        M = u.shape[0]

        u_col = u.unsqueeze(1)
        u_shape = u.shape
        u = u.reshape(-1)

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

    # Compute for u and v directions
    results_u = compute_1d(u, knots_u, degree_u, max_deriv)
    results_v = compute_1d(v, knots_v, degree_v, max_deriv)
    # results_u = [r_u.reshape(uv.shape[:-1] + (r_u.shape[-1],)) for r_u in results_u]
    # results_v = [r_v.reshape(uv.shape[:-1] + (r_v.shape[-1],)) for r_v in results_v]
    return results_u, results_v
def compute_bases_uv_diff(
        samples_u: torch.Tensor,  # [Us] with requires_grad=True
        samples_v: torch.Tensor,  # [Vs] with requires_grad=True
        knots_u: torch.Tensor,
        knots_v: torch.Tensor,
        num_control_u: int,  # H
        num_control_v: int,  # W
        degree: int = 3
, device='cuda') -> BasisFuncs:
    """
    Generate basis matrices WITH full gradient support.

    CRITICAL: All operations preserve gradients w.r.t. samples_u/samples_v.

    Returns:
        Dict with:
            'basis': (bu, bv) - 0th order
            'deriv1': (dbu, dbv) - 1st derivatives
            'deriv2': (dbuu, dbvv) - 2nd derivatives
    """
    from modules.basis.basis_matrix import DifferentiableBSplineBasis

    basis_computer = DifferentiableBSplineBasis(degree=degree, device=device)

    # Ensure gradient tracking
    # assert samples_u.requires_grad, "samples_u must have requires_grad=True"
    # assert samples_v.requires_grad, "samples_v must have requires_grad=True"
    #
    # 0th order (positions)
    bu, bv = basis_computer.forward(
        samples_u, samples_v,
        knots_u, knots_v,
        num_control_u, num_control_v
    )

    # 1st derivatives
    dbu = basis_computer._compute_dbasis_matrix_form(
        samples_u,
        knots_u,
        num_control_u,
        degree=3
    )
    dbv = basis_computer._compute_dbasis_matrix_form(
        samples_v,
        knots_v,
        num_control_v,
        degree=3
    )

    # 2nd derivatives
    dbuu = basis_computer._compute_d2basis_matrix_form(
        samples_u,
        knots_u,
        num_control_u,
        degree=3
    )
    dbvv = basis_computer._compute_d2basis_matrix_form(
        samples_v,
        knots_v,
        num_control_v,
        degree=3
    )

    # Verify gradient flow
    # assert bu.requires_grad, "Basis lost gradients!"
    # assert dbu.requires_grad, "Derivative basis lost gradients!"
    bu_data = {
        'basis': bu, 'deriv1': dbu, 'deriv2': dbuu
    }
    bv_data = {
        'basis': bv, 'deriv1': dbv, 'deriv2': dbvv
    }
    basis_funcs = BasisFuncs(
        bu_data,
        bv_data
    )
    return basis_funcs
    # return {
    #     'basis': (bu, bv),
    #     'deriv1': (dbu, dbv),
    #     'deriv2': (dbuu, dbvv)}

def compute_basis_u(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    return compute_bases_uv_diff(samples, knots, num_control_points, degree, device)

def compute_basis_v(samples, knots=None, num_control_points=4, degree=3, device='cuda'):
    return compute_bases_uv_diff(samples, knots, num_control_points, degree, device)
def compute_bases_uv(samples_u, samples_v, knots_u, knots_v, num_control_u: int, num_control_v: int, degree=3,
                     device='cuda', uv_grid=None) -> BasisFuncs:
    # bu, dbu, dbuu = generate_bspline_basis_matrices(samples_u, knots_u, num_control_u, degree, device)
    # bv, dbv, dbvv = generate_bspline_basis_matrices(samples_v, knots_v, num_control_v, degree, device)
    # if uv_grid is not None:
    #     BU, BV = compute_all_derivatives_2d(uv_grid, knots_u, knots_v, degree_u=degree, degree_v=degree, max_deriv=2)
    #     bu, dbu, dbuu = BU
    #     bv, dbv, dbvv = BV
    #     bu, dbu, dbuu = bu, dbu, dbuu
    #     bv, dbv, dbvv = bv, dbv, dbvv
    # else:
    bu, dbu, dbuu = compute_all_derivatives(samples_u, knots_u, degree, max_deriv=2)
    bv, dbv, dbvv = compute_all_derivatives(samples_v, knots_v, degree, max_deriv=2)

    return BasisFuncs(
        bu_data={'basis': bu, 'deriv1': dbu, 'deriv2': dbuu},
        bv_data={'basis': bv, 'deriv1': dbv, 'deriv2': dbvv}
    )
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


def basis_functions_and_derivatives(
        u: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        n_derivs: int = 2
) -> Tuple[torch.Tensor, ...]:
    """
    Compute basis functions and their derivatives (vectorized, differentiable).
    Fixed to avoid all inplace operations.

    Args:
        u: [... ] parameter values in [0, 1]
        knots: [n_knots] knot vector
        degree: polynomial degree
        n_derivs: number of derivatives to compute (0, 1, or 2)

    Returns:
        N: [..., n_ctrl] basis function values
        dN: [... , n_ctrl] first derivatives (if n_derivs >= 1)
        d2N: [..., n_ctrl] second derivatives (if n_derivs >= 2)
    """
    n_knots = len(knots)
    n_ctrl = n_knots - degree - 1

    batch_shape = u.shape
    device = u.device
    dtype = u.dtype

    # Find spans
    span = find_span_vectorized(u, knots, degree, n_ctrl)

    # Flatten for processing
    u_flat = u.reshape(-1)
    span_flat = span.reshape(-1)
    n_samples = u_flat.shape[0]

    # Use lists to collect results (avoid inplace tensor modifications)
    # Build basis using De Boor-Cox recursion WITHOUT inplace ops

    # Initialize basis values for degree 0
    # N[i,j] = basis function j at sample i
    N_prev = torch.zeros(n_samples, degree + 1, device=device, dtype=dtype)
    N_prev[:, 0] = 1.0

    # Store all intermediate N values for derivative computation
    ndu_list = [N_prev.clone()]

    # Build up the triangular table
    for j in range(1, degree + 1):
        N_curr = torch.zeros(n_samples, degree + 1, device=device, dtype=dtype)

        # Compute left and right differences for this level
        # left[r] = u - knots[span - j + 1 + r]
        # right[r] = knots[span + 1 + r] - u

        left_list = []
        right_list = []

        for r in range(j):
            idx_left = (span_flat - j + 1 + r).clamp(0, n_knots - 1).long()
            idx_right = (span_flat + 1 + r).clamp(0, n_knots - 1).long()
            left_list.append(u_flat - knots[idx_left])
            right_list.append(knots[idx_right] - u_flat)

        # Stack for vectorized computation
        left = torch.stack(left_list, dim=1) if left_list else torch.zeros(n_samples, 0, device=device, dtype=dtype)
        right = torch.stack(right_list, dim=1) if right_list else torch.zeros(n_samples, 0, device=device, dtype=dtype)

        # Compute new basis values
        saved = torch.zeros(n_samples, device=device, dtype=dtype)

        N_vals = []
        for r in range(j):
            denom = right[:, r] + left[:, j - 1 - r]
            denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
            temp = N_prev[:, r] / denom

            new_val = saved + right[:, r] * temp
            N_vals.append(new_val)
            saved = left[:, j - 1 - r] * temp

        N_vals.append(saved)

        # Stack results (no inplace modification)
        for r in range(j + 1):
            N_curr = N_curr.clone()
            N_curr[:, r] = N_vals[r]

        ndu_list.append(N_curr.clone())
        N_prev = N_curr

    # Extract final basis functions
    N_local = N_prev  # [n_samples, degree+1]

    # Scatter to full control point size
    N = torch.zeros(n_samples, n_ctrl, device=device, dtype=dtype)
    for i in range(degree + 1):
        idx = (span_flat - degree + i).clamp(0, n_ctrl - 1).long()
        # Use index_add for non-inplace accumulation
        N = N.scatter_add(1, idx.unsqueeze(-1), N_local[:, i:i + 1])

    N = N.reshape(*batch_shape, n_ctrl)
    results = [N]

    if n_derivs >= 1:
        # Compute first derivatives using finite differences (stable approach)
        eps = 1e-5
        u_plus = (u + eps).clamp(0, 1)
        u_minus = (u - eps).clamp(0, 1)

        # Recursively compute basis at perturbed points
        N_plus = _basis_functions_only(u_plus, knots, degree, n_ctrl)
        N_minus = _basis_functions_only(u_minus, knots, degree, n_ctrl)

        dN = (N_plus - N_minus) / (2 * eps)
        results.append(dN)

    if n_derivs >= 2:
        # Second derivatives via finite differences
        eps = 1e-4
        u_plus = (u + eps).clamp(0, 1)
        u_minus = (u - eps).clamp(0, 1)

        N_plus = _basis_functions_only(u_plus, knots, degree, n_ctrl)
        N_center = _basis_functions_only(u, knots, degree, n_ctrl)
        N_minus = _basis_functions_only(u_minus, knots, degree, n_ctrl)

        d2N = (N_plus - 2 * N_center + N_minus) / (eps * eps)
        results.append(d2N)

    return tuple(results)


def _basis_functions_only(
        u: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        n_ctrl: int
) -> torch.Tensor:
    """
    Compute basis functions only (no derivatives).
    Optimized helper that avoids inplace operations.
    """
    n_knots = len(knots)
    batch_shape = u.shape
    device = u.device
    dtype = u.dtype

    # Find spans
    span = find_span_vectorized(u, knots, degree, n_ctrl)

    u_flat = u.reshape(-1)
    span_flat = span.reshape(-1)
    n_samples = u_flat.shape[0]

    # Initialize
    N_prev = torch.zeros(n_samples, degree + 1, device=device, dtype=dtype)
    N_prev = N_prev.clone()
    N_prev[:, 0] = 1.0

    for j in range(1, degree + 1):
        # Compute differences
        left_list = []
        right_list = []

        for r in range(j):
            idx_left = (span_flat - j + 1 + r).clamp(0, n_knots - 1).long()
            idx_right = (span_flat + 1 + r).clamp(0, n_knots - 1).long()
            left_list.append(u_flat - knots[idx_left])
            right_list.append(knots[idx_right] - u_flat)

        left = torch.stack(left_list, dim=1)
        right = torch.stack(right_list, dim=1)

        # Build new level
        saved = torch.zeros(n_samples, device=device, dtype=dtype)
        N_vals = []

        for r in range(j):
            denom = right[:, r] + left[:, j - 1 - r]
            denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
            temp = N_prev[:, r] / denom

            new_val = saved + right[:, r] * temp
            N_vals.append(new_val)
            saved = left[:, j - 1 - r] * temp

        N_vals.append(saved)

        # Create new tensor (no inplace)
        N_curr = torch.stack(N_vals + [torch.zeros(n_samples, device=device, dtype=dtype)] * (degree - j), dim=1)
        N_prev = N_curr

    # Scatter to full size
    N_local = N_prev[:, :degree + 1]
    N = torch.zeros(n_samples, n_ctrl, device=device, dtype=dtype)

    for i in range(degree + 1):
        idx = (span_flat - degree + i).clamp(0, n_ctrl - 1).long()
        N = N.scatter_add(1, idx.unsqueeze(-1), N_local[:, i:i + 1])

    return N.reshape(*batch_shape, n_ctrl)


def create_basis_from_uv_grid(
        uv_grid: torch.Tensor,
        knots_u: torch.Tensor,
        knots_v: torch.Tensor,
        degree: int,
        n_derivs: int = 2,
        target_shape_u = None,
        target_shape_v = None,
        normalize: bool = True,
        is_grid: bool = True
) -> Tuple[torch.Tensor, ...]:
    """
    Create basis functions from a UV sample grid.

    This is the main entry point for creating differentiable basis functions
    from arbitrary (potentially learned) UV coordinates.

    Args:
        uv_grid: [Us, Vs, 2] or [N, 2] UV coordinates in [0, 1]
        knots_u: [n_knots_u] knot vector for U direction
        knots_v: [n_knots_v] knot vector for V direction
        degree: polynomial degree (same for both directions)
        n_derivs: number of derivatives (0, 1, or 2)

    Returns:
        Bu: [Us, Vs, n_ctrl_u] or [N, n_ctrl_u] U basis values
        Bv: [Us, Vs, n_ctrl_v] or [N, n_ctrl_v] V basis values
        dBu: First derivative of U basis (if n_derivs >= 1)
        dBv: First derivative of V basis (if n_derivs >= 1)
        d2Bu: Second derivative of U basis (if n_derivs >= 2)
        d2Bv: Second derivative of V basis (if n_derivs >= 2)
    """
    # Handle both grid and flat inputs
    # is_grid = uv_grid.dim() == 3

    H, W = knots_u.shape[0] - degree - 1, knots_v.shape[0] - degree - 1
    if is_grid:
        u_samples = uv_grid[..., 0]  # [Us, Vs]
        v_samples = uv_grid[..., 1]  # [Us, Vs]
        Us, Vs, = u_samples.shape[0:2] if is_grid else uv_grid.shape[0]
    else:

        u_samples = uv_grid[0]
        v_samples = uv_grid[1]
        Us, Vs = u_samples.shape[0], v_samples.shape[0]

    # Compute basis for U
    u_results = basis_functions_and_derivatives(
        u_samples, knots_u, degree, n_derivs
    )

    # Compute basis for V
    v_results = basis_functions_and_derivatives(
        v_samples, knots_v, degree, n_derivs
    )
    norm_u = H if normalize else 1.0
    norm_v = W if normalize else 1.0
    u_results = [u.reshape(target_shape_u) for u in u_results]
    v_results = [v.reshape(target_shape_v) for v in v_results]


    return BasisFuncs(  bu=u_results[0],
                        dbu=u_results[1] / (Us * Vs) if n_derivs >= 1 else None,
                        dbuu=u_results[2] if n_derivs >= 2 else None,
                        bv=v_results[0],
                        dbv=v_results[1] / (Us * Vs) if n_derivs >= 1 else None,
                        dbvv=v_results[2] if n_derivs >= 2 else None
                        )


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


class DifferentiableBasisModule(torch.nn.Module):
    """
    PyTorch module wrapper for differentiable basis computation.

    This can be used as a drop-in replacement for BasisFunction when
    you need fully differentiable UV-to-basis computation.
    """

    def __init__(
            self,
            n_ctrl_u: int,
            n_ctrl_v: int,
            degree: int = 3,
            device: str = 'cuda'
    ):
        super().__init__()
        self.n_ctrl_u = n_ctrl_u
        self.n_ctrl_v = n_ctrl_v
        self.degree = degree
        self.device = device

        # Initialize with clamped uniform knots
        knots_u = make_clamped_uniform_knots(n_ctrl_u, degree, device=device)
        knots_v = make_clamped_uniform_knots(n_ctrl_v, degree, device=device)

        self.register_buffer('knots_u', knots_u)
        self.register_buffer('knots_v', knots_v)

        # Cache
        self._Bu = None
        self._Bv = None
        self._dBu = None
        self._dBv = None

    def forward(
            self,
            uv_grid: torch.Tensor,
            compute_derivs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute basis functions for given UV grid.

        Args:
            uv_grid: [Us, Vs, 2] UV coordinates
            compute_derivs: If True, also compute derivatives

        Returns:
            Bu, Bv: Basis function values
        """
        n_derivs = 2 if compute_derivs else 0

        results = create_basis_from_uv_grid(
            uv_grid,
            self.knots_u,
            self.knots_v,
            self.degree,
            n_derivs=n_derivs
        )

        self._Bu = results[0]
        self._Bv = results[1]

        if compute_derivs and len(results) >= 4:
            self._dBu = results[2]
            self._dBv = results[3]

        return self._Bu, self._Bv

    @property
    def Bu(self) -> torch.Tensor:
        if self._Bu is None:
            raise RuntimeError("Call forward() first")
        return self._Bu

    @property
    def Bv(self) -> torch.Tensor:
        if self._Bv is None:
            raise RuntimeError("Call forward() first")
        return self._Bv

    @property
    def dBu(self) -> Optional[torch.Tensor]:
        return self._dBu

    @property
    def dBv(self) -> Optional[torch.Tensor]:
        return self._dBv

    def evaluate(
            self,
            control_points: torch.Tensor,
            is_rational: bool = False
    ) -> torch.Tensor:
        """Evaluate surface using cached basis."""
        return evaluate_surface(self.Bu, self.Bv, control_points, is_rational)

    def update_knots(self, knots_u: torch.Tensor, knots_v: torch.Tensor):
        """Update knot vectors (e.g., after refinement)."""
        self.knots_u = knots_u.to(self.device)
        self.knots_v = knots_v.to(self.device)
        self.n_ctrl_u = len(knots_u) - self.degree - 1
        self.n_ctrl_v = len(knots_v) - self.degree - 1

        # Invalidate cache
        self._Bu = None
        self._Bv = None
        self._dBu = None
        self._dBv = None


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




    def compute_uniform_basis(self, knot_u, knot_v):
        u = torch.linspace(0.0, 1.0, self.state.Us, device=self.state.device)
        v = torch.linspace(0.0, 1.0, self.state.Vs, device=self.state.device)
        if self.state.full_basis:
            samples_uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).view(-1, 2)

        bu, dbu, dbuu = create_basis_from_uv_grid(
            samples_uv,
            knot_u,
            knot_v,
            degree=3,
            n_derivs=self.state.deriv_order,
            target_shape=self.basis_layout_u)#((self.state.Vs * self.state.Us, self.state.H) if self.state.flatten_uv else (self.state.Us, self.state.H)))

        # bv shape: (H, W, n_ctrl_v)
        bv, dbv, dbvv = create_basis_from_uv_grid(
            v,
            knot_v,
            degree=3,
            deriv_order=self.state.deriv_order,
            target_shape=self.basis_layout_v)#((self.state.Vs * self.state.Us, self.state.W) if self.state.flatten_uv else (self.state.Vs, self.state.W)))


        self._uniform_basis_funcs = BasisFuncs(bu,
                                               dbu,
                                               dbuu,
                                               bv,
                                               dbv,
                                               dbvv)
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
        return self._optimal_path if self._optimal_path is not None else self.contract_path

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
    def update_basis(self, precomp_u, precomp_v):
        self._basis_funcs.bu = precomp_u[0],
        self._basis_funcs.dbu = precomp_u[1] if self.state.deriv_order >= 1 else None,
        self._basis_funcs.dbuu = precomp_u[2] if self.state.deriv_order >= 2 else None,
        self._basis_funcs.bv = precomp_v[0],
        self._basis_funcs.dbv = precomp_v[1] if self.state.deriv_order >= 1 else None,
        self._basis_funcs.dbvv = precomp_v[2] if self.state.deriv_order >= 2 else None
        if self.optimal_path is None:
            self.set_optimal_path()

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

            except Exception as e:
                self._optimal_path = self.contract_path


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
        compute = compute_bases_uv_diff if self.state.opt.optimize_intervals else compute_bases_uv
        self.clear()
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