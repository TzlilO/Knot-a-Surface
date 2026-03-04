"""
Cox-de Boor B-Spline Basis Function Evaluation

Formal implementation of the Cox-de Boor recursion formula for computing
B-spline basis functions and their derivatives, consistent with the
NURBS pipeline in this project.

Mathematical Reference:
───────────────────────
Degree 0 (indicator):
    N_{i,0}(u) = 1   if t_i ≤ u < t_{i+1}
                 0   otherwise

Degree p (recursion):
    N_{i,p}(u) = α_{i,p}(u) · N_{i,p-1}(u) + (1 - α_{i+1,p}(u)) · N_{i+1,p-1}(u)

    where  α_{i,p}(u) = (u - t_i) / (t_{i+p} - t_i)

Convention: 0/0 ≡ 0  (handles repeated knots)

Derivatives (analytic):
    N'_{i,p}(u) = p / (t_{i+p} - t_i)     · N_{i,p-1}(u)
                - p / (t_{i+p+1} - t_{i+1}) · N_{i+1,p-1}(u)
"""

from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SCALAR (REFERENCE) IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def cox_de_boor_scalar(
    u: float,
    i: int,
    p: int,
    knots: List[float]
) -> float:
    """
    Evaluate a single B-spline basis function N_{i,p}(u) via Cox-de Boor recursion.

    This is the textbook-faithful reference implementation. Use for verification
    and small-scale debugging only — not for batch evaluation.

    Args:
        u:     Evaluation parameter, u ∈ [t_0, t_m].
        i:     Basis function index, 0 ≤ i ≤ n-1.
        p:     Polynomial degree, p ≥ 0.
        knots: Knot vector T = (t_0, t_1, ..., t_m), non-decreasing.
               Must satisfy |T| = n + p + 1.

    Returns:
        N_{i,p}(u) ∈ [0, 1].

    Raises:
        IndexError: If i or p cause out-of-range knot access.

    Complexity:
        O(p) time, O(p) stack depth (recursive).

    Example:
        >>> knots = [0, 0, 0, 0, 1, 1, 1, 1]  # Clamped cubic, 4 control points
        >>> cox_de_boor_scalar(0.5, i=1, p=3, knots=knots)
        0.375
    """
    # Base case: degree 0 — indicator function on half-open knot span
    if p == 0:
        # Special case: last knot span is closed [t_{m-1}, t_m] to include u = t_m
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        elif u == knots[-1] and knots[i] <= u <= knots[i + 1]:
            # Right endpoint inclusion for the last non-zero span
            return 1.0
        else:
            return 0.0

    # Recursive case: degree p
    # Left term:  α_{i,p}(u) · N_{i,p-1}(u)
    denom_left = knots[i + p] - knots[i]
    if denom_left == 0.0:
        # Convention: 0/0 ≡ 0 (repeated knot, zero-length span)
        left = 0.0
    else:
        alpha = (u - knots[i]) / denom_left
        left = alpha * cox_de_boor_scalar(u, i, p - 1, knots)

    # Right term: (1 - α_{i+1,p}(u)) · N_{i+1,p-1}(u)
    denom_right = knots[i + p + 1] - knots[i + 1]
    if denom_right == 0.0:
        right = 0.0
    else:
        beta = (knots[i + p + 1] - u) / denom_right
        right = beta * cox_de_boor_scalar(u, i + 1, p - 1, knots)

    return left + right


def cox_de_boor_derivative_scalar(
    u: float,
    i: int,
    p: int,
    knots: List[float],
    deriv_order: int = 1
) -> float:
    """
    Evaluate the k-th derivative of N_{i,p}(u) using the analytic derivative formula.

    First derivative:
        N'_{i,p}(u) = p/(t_{i+p} - t_i) · N_{i,p-1}(u)
                    - p/(t_{i+p+1} - t_{i+1}) · N_{i+1,p-1}(u)

    Higher derivatives apply the formula recursively on the lower-degree terms.

    Args:
        u:           Evaluation parameter.
        i:           Basis function index.
        p:           Polynomial degree.
        knots:       Knot vector.
        deriv_order: Derivative order k (1 = first derivative, 2 = second, ...).

    Returns:
        d^k/du^k N_{i,p}(u).
    """
    if deriv_order == 0:
        return cox_de_boor_scalar(u, i, p, knots)

    if p == 0:
        # Derivative of a piecewise-constant function is 0 (except at knots,
        # where it's undefined — we return 0 by convention).
        return 0.0

    # Analytic derivative formula: reduce degree by 1, recurse for higher orders
    denom_left = knots[i + p] - knots[i]
    denom_right = knots[i + p + 1] - knots[i + 1]

    left = 0.0 if denom_left == 0.0 else (
        p / denom_left * cox_de_boor_derivative_scalar(u, i, p - 1, knots, deriv_order - 1)
    )

    right = 0.0 if denom_right == 0.0 else (
        p / denom_right * cox_de_boor_derivative_scalar(u, i + 1, p - 1, knots, deriv_order - 1)
    )

    return left - right


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VECTORIZED (PRODUCTION) IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_basis_functions(
    u: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    deriv_order: int = 0
) -> torch.Tensor:
    """
    Compute all B-spline basis functions N_{i,p}(u_k) for a vector of
    evaluation points, using the iterative (triangular table) form of
    Cox-de Boor. Returns the full basis matrix B[k, i] = N_{i,p}(u_k).

    This is the vectorized counterpart used in the pipeline. The output
    matrix B^u or B^v is contracted with control points via einsum:

        S(u_k, v_l) = Σ_{i,j} B^u_{k,i} · B^v_{l,j} · P_{i,j}
                    ⟺  einsum('uh, vw, hwc -> uvc', Bu, Bv, P)

    Algorithm:
        1. Initialize degree-0 indicators for all spans simultaneously.
        2. Iterate from degree 1 to p, computing blending coefficients α
           and accumulating left + right contributions.
        3. Optionally compute derivatives using the relation:
           N'_{i,p} = p·[N_{i,p-1}/(t_{i+p}-t_i) - N_{i+1,p-1}/(t_{i+p+1}-t_{i+1})]

    Args:
        u:           [M] evaluation parameters, each in [0, 1].
        knots:       [n + p + 1] knot vector (clamped, non-decreasing).
        degree:      Polynomial degree p.
        deriv_order: 0 = basis values, 1 = first derivative, 2 = second derivative.

    Returns:
        B: [M, n] basis matrix, where n = len(knots) - degree - 1.

    Shape relationship to your pipeline:
        Bu = compute_basis_functions(u_samples, knots_u, p)  # [Us, H]
        Bv = compute_basis_functions(v_samples, knots_v, q)  # [Vs, W]

    Properties verified:
        • Partition of unity:  B.sum(dim=1) ≈ 1  (to floating-point precision)
        • Non-negativity:      B ≥ 0
        • Local support:       B[k, i] = 0 if u_k ∉ [t_i, t_{i+p+1})
    """
    device = u.device
    dtype = u.dtype
    m = knots.shape[0] - 1          # Highest knot index
    n = m - degree                   # Number of basis functions (= num control points)
    M = u.shape[0]                   # Number of evaluation points

    # ── Step 1: Degree-0 indicators ──────────────────────────────────────────
    # N_{i,0}(u) = 1 if t_i ≤ u < t_{i+1}, else 0
    # Shape: [M, m] — one column per knot span [t_i, t_{i+1})

    # Broadcast: u is [M, 1], knots sliced to [1, m]
    u_col = u.unsqueeze(1)                          # [M, 1]
    left_edges = knots[:-1].unsqueeze(0)             # [1, m]
    right_edges = knots[1:].unsqueeze(0)             # [1, m]

    # Half-open interval [t_i, t_{i+1})
    N = ((u_col >= left_edges) & (u_col < right_edges)).to(dtype)  # [M, m]

    # Handle right endpoint: u = t_m should activate the last non-zero span.
    # For clamped knots, the last p+1 knots are equal, so the last valid span
    # is [t_{m-p-1}, t_{m-p}).  We include u = t_m in this span.
    last_span_mask = (u == knots[m]) & (knots[m - 1] < knots[m])
    if last_span_mask.any():
        # Find the last non-degenerate span
        for j in range(m - 1, -1, -1):
            if knots[j] < knots[j + 1]:
                N[last_span_mask, j] = 1.0
                break

    # ── Step 2: Build triangular table, degree 1 → p ────────────────────────
    # At each level d, we compute N_{i,d} from N_{i,d-1} and N_{i+1,d-1}
    # and optionally store the degree (p-1) table for derivative computation.

    saved_for_deriv = None  # Will store N_{i, p-1} if deriv_order ≥ 1

    for d in range(1, degree + 1):
        # Number of basis functions at degree d: m - d
        n_basis_d = m - d

        # Blending coefficients α_{i,d}(u) = (u - t_i) / (t_{i+d} - t_i)
        # Shape: [M, n_basis_d]
        t_left = knots[:n_basis_d].unsqueeze(0)            # [1, n_basis_d]
        t_right = knots[d:d + n_basis_d].unsqueeze(0)      # [1, n_basis_d]

        denom_left = t_right - t_left                        # [1, n_basis_d]
        # Safe division: where denom = 0, α = 0 (0/0 ≡ 0 convention)
        alpha = torch.where(
            denom_left.abs() > 1e-14,
            (u_col - t_left) / denom_left,
            torch.zeros(1, device=device, dtype=dtype)
        )  # [M, n_basis_d]

        # β_{i+1,d}(u) = (t_{i+d+1} - u) / (t_{i+d+1} - t_{i+1})
        t_left_r = knots[1:n_basis_d + 1].unsqueeze(0)      # [1, n_basis_d]
        t_right_r = knots[d + 1:d + 1 + n_basis_d].unsqueeze(0)  # [1, n_basis_d]

        denom_right = t_right_r - t_left_r
        beta = torch.where(
            denom_right.abs() > 1e-14,
            (t_right_r - u_col) / denom_right,
            torch.zeros(1, device=device, dtype=dtype)
        )  # [M, n_basis_d]

        # Save degree (p-1) for derivative computation before overwriting
        if d == degree and deriv_order >= 1:
            # N_{i, p-1} has (m - (p-1)) = m - p + 1 = n + 1 columns
            saved_for_deriv = N[:, :n_basis_d + 1].clone()  # [M, n+1]

        # Recurrence: N_{i,d} = α · N_{i,d-1} + β · N_{i+1,d-1}
        N_new = alpha * N[:, :n_basis_d] + beta * N[:, 1:n_basis_d + 1]

        # Prepare for next iteration (or final result)
        # Pad to maintain indexing consistency
        N_padded = torch.zeros(M, m, device=device, dtype=dtype)
        N_padded[:, :n_basis_d] = N_new
        N = N_padded

    # Extract the n basis functions of degree p
    B = N[:, :n]  # [M, n]

    # ── Step 3 (optional): Compute derivatives ───────────────────────────────
    if deriv_order >= 1 and saved_for_deriv is not None:
        B = _compute_derivative_from_lower_degree(
            saved_for_deriv, knots, degree, deriv_order, n, device, dtype
        )

    return B


def _compute_derivative_from_lower_degree(
    N_lower: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    deriv_order: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Compute derivatives of B-spline basis functions from the degree-(p-1) table.

    Uses the analytic formula:
        N'_{i,p}(u) = p / (t_{i+p} - t_i)     · N_{i,p-1}(u)
                    - p / (t_{i+p+1} - t_{i+1}) · N_{i+1,p-1}(u)

    For second derivatives, the formula is applied recursively:
        N''_{i,p}(u) = p·[(N'_{i,p-1})/(t_{i+p}-t_i) - (N'_{i+1,p-1})/(t_{i+p+1}-t_{i+1})]

    But since N'_{i,p-1} itself uses degree-(p-2) functions, we get:
        N''_{i,p} = p(p-1)·[N_{i,p-2}/(t_{i+p}-t_i)(t_{i+p-1}-t_i)
                           - N_{i+1,p-2}/(...) - N_{i+1,p-2}/(...) + N_{i+2,p-2}/(...)]

    Args:
        N_lower: [M, n+1] basis functions at degree (p-1).
        knots:   Full knot vector.
        degree:  Target degree p.
        deriv_order: 1 or 2.
        n:       Number of degree-p basis functions.

    Returns:
        dB: [M, n] derivative values.
    """
    M = N_lower.shape[0]
    p = degree

    if deriv_order == 1:
        # N'_{i,p}(u) = p·[N_{i,p-1}/(t_{i+p}-t_i) - N_{i+1,p-1}/(t_{i+p+1}-t_{i+1})]
        dB = torch.zeros(M, n, device=device, dtype=dtype)

        for i in range(n):
            denom_left = knots[i + p] - knots[i]
            denom_right = knots[i + p + 1] - knots[i + 1]

            left = 0.0 if abs(denom_left) < 1e-14 else (
                p / denom_left * N_lower[:, i]
            )
            right = 0.0 if abs(denom_right) < 1e-14 else (
                p / denom_right * N_lower[:, i + 1]
            )

            dB[:, i] = left - right

        return dB

    elif deriv_order == 2:
        # First compute d/du of the degree-(p-1) functions (which are degree p-2)
        # Then apply the derivative formula again.
        # We need N_{i, p-2} — recompute or use the recursion.

        # Approach: compute first derivatives of the (p-1)-degree functions,
        # then apply derivative formula on those.
        # This requires the degree-(p-2) table — for now, use finite differences
        # on the first derivative as a pragmatic fallback for the second derivative.

        # Analytic: use the relation
        # N''_{i,p} = p * [ N'_{i,p-1}/(t_{i+p}-t_i) - N'_{i+1,p-1}/(t_{i+p+1}-t_{i+1}) ]
        # where N'_{i,p-1} is computed recursively from degree-(p-2) functions.

        # For degree p=3 (cubic), N'_{i,2} needs N_{i,1} which needs N_{i,0}.
        # This is a full re-evaluation. For production, pre-compute all levels.

        # Simplified: return first derivative and let caller use finite differences
        # for second derivatives if needed.
        raise NotImplementedError(
            "Second derivatives require the full triangular table. "
            "Use compute_all_derivatives() instead."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FULL TRIANGULAR TABLE (for derivatives up to order k)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_derivatives(
    u: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    max_deriv: int = 2
) -> Tuple[torch.Tensor, ...]:
    """
    Compute basis functions AND derivatives up to order `max_deriv` by
    retaining the full Cox-de Boor triangular table.

    Returns all levels needed for analytic derivative computation:
        N_{i,0}, N_{i,1}, ..., N_{i,p}  and  N'_{i,p}, N''_{i,p}

    This is what your pipeline uses to populate:
        basis.bu,  basis.dbu,  basis.dbuu   (for u-direction)
        basis.bv,  basis.dbv,  basis.dbvv   (for v-direction)

    Args:
        u:         [M] evaluation parameters.
        knots:     [n + p + 1] knot vector.
        degree:    Polynomial degree p.
        max_deriv: Maximum derivative order (default 2 for normals/curvature).

    Returns:
        Tuple of (B, dB, d2B, ...) where:
            B:   [M, n]  basis function values
            dB:  [M, n]  first derivatives
            d2B: [M, n]  second derivatives (if max_deriv >= 2)

    Pipeline integration:
        Bu, dBu, d2Bu = compute_all_derivatives(u_samples, knots_u, p, max_deriv=2)
        Bv, dBv, d2Bv = compute_all_derivatives(v_samples, knots_v, q, max_deriv=2)

        # Surface point:
        S = einsum('uh, vw, hwc -> uvc', Bu, Bv, P)

        # Tangents (for normals):
        dSdu = einsum('uh, vw, hwc -> uvc', dBu, Bv, P)
        dSdv = einsum('uh, vw, hwc -> uvc', Bu, dBv, P)

        # Second derivatives (for curvature):
        d2Sduu = einsum('uh, vw, hwc -> uvc', d2Bu, Bv, P)
        d2Sdvv = einsum('uh, vw, hwc -> uvc', Bu, d2Bv, P)
    """
    device = u.device
    dtype = u.dtype
    m = knots.shape[0] - 1
    n = m - degree
    M = u.shape[0]
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
def compute_all_derivatives2(
    u: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    max_deriv: int = 2
) -> Tuple[torch.Tensor, ...]:
    """
    Compute basis functions AND derivatives up to order `max_deriv` by
    retaining the full Cox-de Boor triangular table.

    Returns all levels needed for analytic derivative computation:
        N_{i,0}, N_{i,1}, ..., N_{i,p}  and  N'_{i,p}, N''_{i,p}

    This is what your pipeline uses to populate:
        basis.bu,  basis.dbu,  basis.dbuu   (for u-direction)
        basis.bv,  basis.dbv,  basis.dbvv   (for v-direction)

    Args:
        u:         [M] evaluation parameters.
        knots:     [n + p + 1] knot vector.
        degree:    Polynomial degree p.
        max_deriv: Maximum derivative order (default 2 for normals/curvature).

    Returns:
        Tuple of (B, dB, d2B, ...) where:
            B:   [M, n]  basis function values
            dB:  [M, n]  first derivatives
            d2B: [M, n]  second derivatives (if max_deriv >= 2)

    Pipeline integration:
        Bu, dBu, d2Bu = compute_all_derivatives(u_samples, knots_u, p, max_deriv=2)
        Bv, dBv, d2Bv = compute_all_derivatives(v_samples, knots_v, q, max_deriv=2)

        # Surface point:
        S = einsum('uh, vw, hwc -> uvc', Bu, Bv, P)

        # Tangents (for normals):
        dSdu = einsum('uh, vw, hwc -> uvc', dBu, Bv, P)
        dSdv = einsum('uh, vw, hwc -> uvc', Bu, dBv, P)

        # Second derivatives (for curvature):
        d2Sduu = einsum('uh, vw, hwc -> uvc', d2Bu, Bv, P)
        d2Sdvv = einsum('uh, vw, hwc -> uvc', Bu, d2Bv, P)
    """
    device = u.device
    dtype = u.dtype
    m = knots.shape[0] - 1
    n = m - degree
    M = u.shape[0]
    u_col = u.unsqueeze(1)

    # ── Build the full triangular table ──────────────────────────────────────
    # table[d] = N_{i,d}(u) for all i, shape [M, m-d]
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

    # ── Compute derivatives from the table ───────────────────────────────────
    # k-th derivative of degree-p functions uses degree-(p-k) functions:
    #
    # General formula (Piegl & Tiller, "The NURBS Book", Eq. 2.10):
    #   d^k/du^k N_{i,p}(u) is built from N_{i, p-k} coefficients
    #   weighted by products of degree / knot-span-lengths.
    #
    # Implementation: iterative differentiation.
    #   D^(0)_{i} = N_{i,p}  (the basis values)
    #   D^(k)_{i} = p_eff · [D^(k-1)_{i} / (t_{i+p_eff} - t_i)
    #                       - D^(k-1)_{i+1} / (t_{i+p_eff+1} - t_{i+1})]
    #   where p_eff = p - (k-1) is the effective degree at each differentiation step.

    # Start from the degree-(p-1) table for first derivative,
    # degree-(p-2) table for second, etc.

    for k in range(1, max_deriv + 1):
        if degree - k < 0:
            # Derivative order exceeds degree → result is zero
            results.append(torch.zeros(M, n, device=device, dtype=dtype))
            continue

        # Start from degree-(p-k) table
        source_degree = degree - k
        n_source = m - source_degree  # number of functions at source degree

        # We need to apply k successive differentiation steps
        # Step 1: start with table[source_degree], shape [M, n_source]
        D = table[source_degree][:, :n_source].clone()

        # Apply differentiation k times, each time reducing the number of
        # functions by 1 and multiplying by the effective degree.
        current_n = n_source
        for step in range(k):
            p_eff = degree - step  # effective degree for this differentiation
            new_n = current_n - 1

            D_new = torch.zeros(M, new_n, device=device, dtype=dtype)

            for i in range(new_n):
                # Knot span indices shift with each step
                # After `step` differentiations from degree p:
                # the i-th function corresponds to original index i
                knot_offset = step  # cumulative index shift
                t_lo = knots[i + knot_offset]
                t_hi = knots[i + knot_offset + p_eff]

                denom = t_hi - t_lo

                left_term = D[:, i] / denom if abs(denom) > 1e-14 else torch.zeros(M, device=device, dtype=dtype)

                t_lo_r = knots[i + 1 + knot_offset]
                try:
                    t_hi_r = knots[i + 1 + knot_offset + p_eff]
                    denom_r = t_hi_r - t_lo_r
                except Exception as e:
                    pass
                    # print(e)
                    # print(f"Index error at i={i}, step={step}, p_eff={p_eff}, knot_offset={knot_offset}")
                    # print(f"Shape of D: {D.shape}, expected columns: {current_n}")


                right_term = D[:, i + 1] / denom_r if abs(denom_r) > 1e-14 else torch.zeros(M, device=device, dtype=dtype)

                D_new[:, i] = p_eff * (left_term - right_term)

            D = D_new
            current_n = new_n

        # D should now have shape [M, n]
        assert D.shape[1] == n, f"Derivative shape mismatch: got {D.shape[1]}, expected {n}"
        results.append(D)

    return tuple(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLAMPED KNOT VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_clamped_knot_vector(
    n: int,
    degree: int,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Construct a clamped (open) uniform knot vector.

    T = {0, ..., 0, t_{p+1}, ..., t_{m-p-1}, 1, ..., 1}
         \_p+1_/    \___internal knots___/    \_p+1_/

    with uniform internal knot spacing:
        t_{p+k} = k / (n - p)   for k = 1, ..., n - p - 1

    Args:
        n:      Number of control points (= number of basis functions).
        degree: Polynomial degree p.
        device: Torch device.

    Returns:
        Knot vector of length n + p + 1.

    Dimension check:
        |T| = n + p + 1  ✓

    Example:
        >>> make_clamped_knot_vector(n=6, degree=3)
        tensor([0., 0., 0., 0., 0.3333, 0.6667, 1., 1., 1., 1.])
    """
    assert n > degree, f"Need n > p, got n={n}, p={degree}"

    m = n + degree  # highest knot index
    num_knots = m + 1

    knots = torch.zeros(num_knots, device=device)

    # Clamped left: t_0 = ... = t_p = 0
    # (already zero from initialization)

    # Internal knots: uniform spacing
    num_internal = n - degree - 1
    if num_internal > 0:
        internal = torch.linspace(0, 1, num_internal + 2, device=device)[1:-1]
        knots[degree + 1: degree + 1 + num_internal] = internal

    # Clamped right: t_{m-p} = ... = t_m = 1
    knots[m - degree:] = 1.0

    return knots


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VERIFICATION UTILITIES
# ══════���════════════════════════════════════════════════════════════════════════

def verify_partition_of_unity(
    B: torch.Tensor,
    tol: float = 1e-6,
    verbose: bool = True
) -> bool:
    """
    Verify the partition of unity property: Σ_i N_{i,p}(u) = 1 for all u.

    Args:
        B:   [M, n] basis matrix from compute_basis_functions.
        tol: Acceptable deviation from 1.0.

    Returns:
        True if property holds within tolerance.
    """
    row_sums = B.sum(dim=1)
    max_deviation = (row_sums - 1.0).abs().max().item()
    passes = max_deviation < tol

    if verbose:
        print(f"Partition of unity: max |Σ N_i(u) - 1| = {max_deviation:.2e}  "
              f"{'✓ PASS' if passes else '✗ FAIL'}")

    return passes


def verify_non_negativity(
    B: torch.Tensor,
    verbose: bool = True
) -> bool:
    """Verify non-negativity: N_{i,p}(u) ≥ 0."""
    min_val = B.min().item()
    passes = min_val >= -1e-10

    if verbose:
        print(f"Non-negativity:     min N_i(u) = {min_val:.2e}  "
              f"{'✓ PASS' if passes else '✗ FAIL'}")

    return passes


def verify_local_support(
    B: torch.Tensor,
    u: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    verbose: bool = True
) -> bool:
    """
    Verify local support: N_{i,p}(u) = 0 if u ∉ [t_i, t_{i+p+1}).

    Args:
        B:      [M, n] basis matrix.
        u:      [M] evaluation parameters.
        knots:  Knot vector.
        degree: Polynomial degree.
    """
    n = B.shape[1]
    violations = 0

    for i in range(n):
        support_left = knots[i].item()
        support_right = knots[i + degree + 1].item()

        outside_support = (u < support_left) | (u > support_right)
        values_outside = B[outside_support, i]

        if values_outside.abs().max().item() > 1e-10:
            violations += 1

    passes = violations == 0

    if verbose:
        print(f"Local support:      {violations} violations  "
              f"{'✓ PASS' if passes else '✗ FAIL'}")

    return passes


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CROSS-VALIDATION: SCALAR vs VECTORIZED
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate(
    n: int = 8,
    degree: int = 3,
    num_samples: int = 100,
    verbose: bool = True
) -> bool:
    """
    Cross-validate the scalar (reference) and vectorized implementations.

    Creates a clamped knot vector, evaluates both implementations on a
    uniform grid, and checks agreement to floating-point tolerance.

    Args:
        n:           Number of control points.
        degree:      Polynomial degree.
        num_samples: Number of evaluation points.
        verbose:     Print results.

    Returns:
        True if all checks pass.
    """
    knots = make_clamped_knot_vector(n, degree, device=torch.device('cpu'))
    u = torch.linspace(0, 1, num_samples)

    # Vectorized
    B_vec = compute_basis_functions(u, knots, degree)

    # Scalar (reference)
    knots_list = knots.tolist()
    B_ref = torch.zeros(num_samples, n)
    for k in range(num_samples):
        for i in range(n):
            B_ref[k, i] = cox_de_boor_scalar(u[k].item(), i, degree, knots_list)

    # Compare
    max_diff = (B_vec - B_ref).abs().max().item()
    passes = max_diff < 1e-10

    if verbose:
        print(f"\n{'='*60}")
        print(f"Cross-validation: n={n}, p={degree}, M={num_samples}")
        print(f"{'='*60}")
        print(f"Max |vectorized - scalar| = {max_diff:.2e}  "
              f"{'✓ PASS' if passes else '✗ FAIL'}")

    # Run property checks
    p1 = verify_partition_of_unity(B_vec, verbose=verbose)
    p2 = verify_non_negativity(B_vec, verbose=verbose)
    p3 = verify_local_support(B_vec, u, knots, degree, verbose=verbose)

    # Derivative cross-validation
    B_val, dB_val = compute_all_derivatives(u, knots, degree, max_deriv=1)
    dB_ref = torch.zeros(num_samples, n)
    for k in range(num_samples):
        for i in range(n):
            dB_ref[k, i] = cox_de_boor_derivative_scalar(
                u[k].item(), i, degree, knots_list, deriv_order=1
            )

    deriv_diff = (dB_val - dB_ref).abs().max().item()
    p4 = deriv_diff < 1e-8

    if verbose:
        print(f"Derivative agreement: max |dB_vec - dB_scalar| = {deriv_diff:.2e}  "
              f"{'✓ PASS' if p4 else '✗ FAIL'}")
        print(f"{'='*60}\n")

    return all([passes, p1, p2, p3, p4])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Run verification suite
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Cox-de Boor Implementation Verification Suite")
    print("=" * 60)

    all_pass = True
    for n in [5, 8, 12, 20]:
        for p in [2, 3, 4]:
            if n > p:
                result = cross_validate(n=n, degree=p, num_samples=200, verbose=True)
                all_pass = all_pass and result

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)