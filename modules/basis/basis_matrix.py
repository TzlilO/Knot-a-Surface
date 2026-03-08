# File: model/modules/basis/basis_matrix.py

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SparseBasis:
    """
    Sparse local representation of B-spline basis functions.

    Stores only the p+1 non-zero values per sample point rather than
    scattering to a dense [N, num_control] matrix.  For cubic B-splines
    (p=3) this reduces basis storage by ~96 % when num_control is large.

    Attributes:
        values:      [N, p+1]  local non-zero basis values (carries gradients).
        indices:     [N, p+1]  corresponding control-point indices (int64, no grad).
        num_control: Total number of control points (needed for to_dense()).
    """

    def __init__(self, values: torch.Tensor, indices: torch.Tensor, num_control: int):
        self.values = values
        self.indices = indices
        self.num_control = num_control

    # ------------------------------------------------------------------
    # Dense conversion (backward-compatibility fallback)
    # ------------------------------------------------------------------

    def to_dense(self, num_control: int = None) -> torch.Tensor:
        """Scatter to dense [N, num_control] matrix.

        Args:
            num_control: Override the stored num_control when provided.

        Returns:
            Dense [N, num_control] basis matrix with gradient flow preserved.
        """
        nc = num_control if num_control is not None else self.num_control
        N = self.values.shape[0]
        device = self.values.device
        dtype = self.values.dtype
        basis = torch.zeros(N, nc, device=device, dtype=dtype)
        return basis.scatter_add(1, self.indices, self.values)

    # ------------------------------------------------------------------
    # Duck-typing helpers so existing code that checks .shape / .ndim
    # / .isnan() continues to work without dense conversion.
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """Logical dense shape (N, num_control)."""
        return (self.values.shape[0], self.num_control)

    @property
    def ndim(self) -> int:
        return 2

    def isnan(self) -> torch.Tensor:
        """Return a boolean tensor indicating NaN entries in the values."""
        return self.values.isnan()

    @property
    def T(self) -> torch.Tensor:
        """Transpose — converts to dense first (needed for legacy oe.contract paths)."""
        return self.to_dense().T

    def __repr__(self) -> str:
        return (
            f"SparseBasis(N={self.values.shape[0]}, p+1={self.values.shape[1]}, "
            f"num_control={self.num_control}, device={self.values.device})"
        )


class DifferentiableBSplineBasis(nn.Module):
    """
    Matrix-form B-spline basis evaluation with full gradient support.

    Uses the matrix formulation where basis functions are computed as:
        N(u) = U^T · M · C
    where:
        U = [1, u, u², ..., u^p]^T (power basis)
        M = basis matrix for degree p
        C = coefficient matrix derived from knot intervals

    For derivatives:
        dN/du = U^T · D^T · M · C
        d²N/du² = U^T · (D²)^T · M · C

    CRITICAL: Works with grid control point layout [H, W, C].
    """

    def __init__(self, degree: int = 3, device: str = 'cuda'):
        super().__init__()
        self.degree = degree
        self.device = device

        # Precompute derivative matrices
        # self.register_buffer('D', self._make_derivative_matrix(degree))
        # self.register_buffer('D2', self._make_second_derivative_matrix(degree))
        self.init_basis_matrix()
        self.non_uniform_correction = False
    def init_basis_matrix(self):
        self.M = (1 / 6) * torch.tensor([
            [1, 4, 1, 0],
            [-3, 0, 3, 0],
            [3, -6, 3, 0],
            [-1, 3, -3, 1]
        ], device='cuda')
        self.D = (1 / 2) * torch.tensor([
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0]], device='cuda')
        self.D2 = (1 / 2) * torch.tensor([
            [0, 0, 2, 0],
            [0, 0, 0, 6],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], device='cuda')
        # Precompute derivative basis matrices
        self.M_prime = self.D @ self.M  # First derivative basis
        self.M_double_prime = self.D2 @ self.M  # Second derivative basis

    def _make_derivative_matrix(self, degree: int) -> torch.Tensor:
        """
        Create derivative matrix D for power basis differentiation.

        For cubic (degree=3): d/du [1, u, u², u³]^T = D · [1, u, u², u³]^T
        D = [[0, 1, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 3],
             [0, 0, 0, 0]]
        """
        p = degree + 1
        D = torch.zeros(p, p, dtype=torch.float32)
        for i in range(p - 1):
            D[i, i + 1] = i + 1
        return D

    def _make_second_derivative_matrix(self, degree: int) -> torch.Tensor:
        """
        Create second derivative matrix D² for power basis.

        For cubic: d²/du² [1, u, u², u³]^T = D² · [1, u, u², u³]^T
        D² = [[0, 0, 2, 0],
              [0, 0, 0, 6],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
        """
        p = degree + 1
        D2 = torch.zeros(p, p, dtype=torch.float32)
        for i in range(p - 2):
            D2[i, i + 2] = (i + 1) * (i + 2)
        return D2

    def _get_uniform_basis_matrix(self, degree: int) -> torch.Tensor:
        """
        Get the standard basis matrix for uniform B-splines.

        For cubic (degree=3):
        M = (1/6) * [[ 1,  4,  1,  0],
                     [-3,  0,  3,  0],
                     [ 3, -6,  3,  0],
                     [-1,  3, -3,  1]]
        """
        return self.M
        if degree == 1:
            # Linear
            return torch.tensor([
                [1.0, 0.0],
                [-1.0, 1.0]
            ], dtype=torch.float32)
        elif degree == 2:
            # Quadratic
            return torch.tensor([
                [0.5, 0.5, 0.0],
                [-1.0, 1.0, 0.0],
                [0.5, -1.0, 0.5]
            ], dtype=torch.float32)
        elif degree == 3:
            # Cubic
            return (1.0 / 6.0) * torch.tensor([
                [1.0, 4.0, 1.0, 0.0],
                [-3.0, 0.0, 3.0, 0.0],
                [3.0, -6.0, 3.0, 0.0],
                [-1.0, 3.0, -3.0, 1.0]
            ], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Matrix form not implemented for degree {degree}")

    def forward(
            self,
            u: torch.Tensor,  # [Us] sample points in U
            v: torch.Tensor,  # [Vs] sample points in V
            knots_u: torch.Tensor,  # [M_u] knot vector
            knots_v: torch.Tensor,  # [M_v] knot vector
            num_control_u: int,  # H
            num_control_v: int  # W
    ) -> Tuple['SparseBasis', 'SparseBasis']:
        """
        Compute separable basis matrices for grid control points using matrix form.

        Returns:
            bu: SparseBasis in U — values [Us, p+1], indices [Us, p+1]
            bv: SparseBasis in V — values [Vs, p+1], indices [Vs, p+1]
        """
        bu = self._compute_basis_matrix_form(u, knots_u, num_control_u, self.degree)
        bv = self._compute_basis_matrix_form(v, knots_v, num_control_v, self.degree)

        return bu, bv


    # Instead of loops, process all derivatives at once
    def compute_derivatives_vectorized(self, u, v, knots_u, knots_v, num_control_u, num_control_v, order=1):
        # Compute basis in batch
        bu = self._compute_basis_matrix_form_vectorized(u, knots_u, num_control_u, self.degree)
        bv = self._compute_basis_matrix_form_vectorized(v, knots_v, num_control_v, self.degree)

        if order == 1:
            dbu = self._compute_deriv_vectorized(u, knots_u, num_control_u, self.M_prime)
            dbv = self._compute_deriv_vectorized(v, knots_v, num_control_v, self.M_prime)
        else:
            dbu = self._compute_deriv_vectorized(u, knots_u, num_control_u, self.M_double_prime)
            dbv = self._compute_deriv_vectorized(v, knots_v, num_control_v, self.M_double_prime)

        return dbu, dbv

    def _adjust_basis_matrix_nonuniform(
            self,
            M: torch.Tensor,
            knots: torch.Tensor,
            span: torch.Tensor,  # [N] - batched span indices
            degree: int
    ) -> torch.Tensor:
        """
        Adjust basis matrix for non-uniform knots using Oslo algorithm.

        The Oslo algorithm computes the exact transformation between uniform
        and non-uniform B-spline basis functions using knot insertion matrices.

        References:
            - Cohen, E., Lyche, T., & Riesenfeld, R. (1980).
              "Discrete B-splines and subdivision techniques in CAGD"
            - Prautzsch, H., Boehm, W., & Paluszny, M. (2002).
              "Bézier and B-Spline Techniques"

        Args:
            M: Uniform basis matrix [degree+1, degree+1]
            knots: Knot vector [M]
            span: Knot span indices [N] - VECTORIZED
            degree: B-spline degree

        Returns:
            M_adjusted: Adjusted basis matrices [N, degree+1, degree+1]
        """
        p = degree
        N = span.shape[0]
        device = knots.device
        dtype = M.dtype

        # === Step 1: Extract local knot vectors for each sample ===
        # For span i, we need knots[i-p : i+p+2] (total 2p+2 knots)

        # Create offset indices: [-p, -p+1, ..., p+1]
        offsets = torch.arange(-p, p + 2, device=device)  # [2p+2]

        # Broadcast to get indices for all spans: [N, 2p+2]
        indices = span.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 2p+2]

        # Clamp to valid range
        indices = indices.clamp(0, len(knots) - 1)

        # Gather local knot vectors: [N, 2p+2]
        knot_local = knots[indices]

        # === Step 2: Compute Oslo transformation matrix ===
        Oslo = self._compute_oslo_coefficients(knot_local, degree)  # [N, p+1, p+1]

        # === Step 3: Apply Oslo transformation to uniform basis matrix ===
        M_batch = M.unsqueeze(0).expand(N, p + 1, p + 1)  # [N, p+1, p+1]

        # Apply Oslo transformation: M_adjusted = M @ Oslo
        # This transforms the uniform basis to non-uniform
        M_adjusted = torch.bmm(M_batch, Oslo)  # [N, p+1, p+1]

        return M_adjusted

    def _compute_oslo_coefficients(
            self,
            knots_local: torch.Tensor,  # [N, 2p+2]
            degree: int
    ) -> torch.Tensor:
        """
        Compute Oslo transformation coefficients using the discrete B-spline algorithm.

        This is the core of the Oslo algorithm: computing the transformation matrix
        that converts from uniform to non-uniform basis functions.

        The Oslo algorithm works by iteratively computing knot insertion coefficients
        (alpha values) at each refinement level, building up the full transformation.

        Args:
            knots_local: Local knot vectors [N, 2p+2]
            degree: B-spline degree

        Returns:
            Oslo matrix [N, p+1, p+1] - transformation coefficients
        """
        p = degree
        N = knots_local.shape[0]
        device = knots_local.device
        dtype = knots_local.dtype

        # Initialize Oslo matrix as identity
        Oslo = torch.eye(p + 1, device=device, dtype=dtype).unsqueeze(0).expand(N, p + 1, p + 1).clone()

        # The parameter value we're evaluating at (center of local knots)
        # For matrix-based evaluation, this is the normalized parameter in [0,1]
        t_eval = knots_local[:, p + 1]  # [N] - evaluation parameter

        # Apply Oslo recursion for each degree level
        # At each level k, we refine from degree k-1 to degree k
        for k in range(1, p + 1):
            # At degree k, we update basis functions from right to left
            # This maintains the recurrence relation property

            for i in range(p - k + 1):
                # For basis function N_{i,k}, we need knots from support interval
                # Left knot of support
                idx_left = i
                # Right knot of support (k+1 knots away)
                idx_right = i + k + 1

                # Safety clamp (shouldn't be needed with proper local knot extraction)
                idx_left = max(idx_left, 0)
                idx_right = min(idx_right, 2 * p + 1)

                # Extract knots for this basis function
                t_left = knots_local[:, idx_left]  # [N]
                t_right = knots_local[:, idx_right]  # [N]

                # Compute Oslo alpha coefficient
                # alpha = (t - t_left) / (t_right - t_left)
                denominator = (t_right - t_left).clamp(min=1e-10)
                alpha = ((t_eval - t_left) / denominator).clamp(0.0, 1.0)  # [N]

                # Update Oslo matrix using the recurrence relation:
                # N_{i,k}(t) = alpha * N_{i,k-1}(t) + (1-alpha) * N_{i+1,k-1}(t)
                #
                # This updates row i of the Oslo matrix
                if i + 1 < p + 1:
                    Oslo[:, i, :] = (
                            alpha.unsqueeze(1) * Oslo[:, i, :] +
                            (1 - alpha).unsqueeze(1) * Oslo[:, i + 1, :]
                    )
                else:
                    # Edge case: no i+1 term
                    Oslo[:, i, :] = alpha.unsqueeze(1) * Oslo[:, i, :]

        return Oslo

    def _compute_basis_matrix_form(
            self,
            t: torch.Tensor,  # [N]
            knots: torch.Tensor,  # [M]
            num_control: int,
            degree: int
    ) -> 'SparseBasis':
        """Fully vectorized B-spline basis evaluation — returns SparseBasis."""
        N = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Find knot spans for ALL samples at once [N]
        spans = torch.searchsorted(knots[degree:-degree], t, right=False) + degree - 1
        spans = spans.clamp(degree, len(knots) - degree - 2)

        # Normalize ALL parameters at once [N]
        u_normalized = (t - knots[spans]) / (knots[spans + 1] - knots[spans] + 1e-10)
        u_normalized = u_normalized.clamp(0.0, 1.0)

        # Compute power basis for ALL samples [N, degree+1]
        U = self._power_basis(u_normalized, degree)

        # Get basis matrix
        M = self.M.to(device=device, dtype=dtype)

        if self.non_uniform_correction:
            # Apply Oslo algorithm for exact non-uniform transformation
            M_batch = self._adjust_basis_matrix_nonuniform(M, knots, spans, degree)

            # Batched matrix multiplication: [N, 1, degree+1] @ [N, degree+1, degree+1]
            N_local = torch.bmm(U.unsqueeze(1), M_batch).squeeze(1)  # [N, degree+1]
        else:
            # Uniform case: single matrix for all samples
            N_local = U @ M  # [N, degree+1]

        # Compute local control-point indices [N, degree+1]
        control_indices = (
            spans.unsqueeze(1) - degree + torch.arange(degree + 1, device=device)
        ).clamp(0, num_control - 1)

        return SparseBasis(N_local, control_indices, num_control)


    def _compute_basis_matrix_formm(
            self,
            t: torch.Tensor,  # [N]
            knots: torch.Tensor,  # [M]
            num_control: int,
            degree: int
    ) -> torch.Tensor:
        """Fully vectorized B-spline basis evaluation."""
        N = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Find knot spans for ALL samples at once [N]
        spans = torch.searchsorted(knots[degree:-degree], t, right=False) + degree - 1
        spans = spans.clamp(degree, len(knots) - degree - 2)

        # Normalize ALL parameters at once [N]
        u_normalized = (t - knots[spans]) / (knots[spans + 1] - knots[spans] + 1e-10)
        u_normalized = u_normalized.clamp(0.0, 1.0)

        # Compute power basis for ALL samples [N, degree+1]
        U = self._power_basis(u_normalized, degree)

        # Get basis matrix
        M = self.M.to(device=device, dtype=dtype)
        if self.non_uniform_correction:
            M = self._adjust_basis_matrix_nonuniform(M, knots, spans, degree)
        # Compute ALL local basis values [N, degree+1]
        N_local = U @ M

        # Scatter to global basis matrix [N, num_control]
        basis = torch.zeros(N, num_control, device=device, dtype=dtype)

        # Vectorized scatter using advanced indexing
        sample_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, degree + 1)
        control_indices = (spans.unsqueeze(1) - degree + torch.arange(degree + 1, device=device)).clamp(0,
                                                                                                        num_control - 1)

        basis = basis.scatter_add(1, sample_indices, N_local)

        return basis


    def _compute_dbasis_matrix_form(
            self,
            t: torch.Tensor,  # [N]
            knots: torch.Tensor,  # [M]
            num_control: int,
            degree: int
    ) -> 'SparseBasis':
        """Fully vectorized first-derivative B-spline basis evaluation — returns SparseBasis."""
        N = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Find knot spans for ALL samples at once [N]
        spans = torch.searchsorted(knots[degree:-degree], t, right=False) + degree - 1
        spans = spans.clamp(degree, len(knots) - degree - 2)

        # Normalize ALL parameters at once [N]
        u_normalized = (t - knots[spans]) / (knots[spans + 1] - knots[spans] + 1e-10)
        u_normalized = u_normalized.clamp(0.0, 1.0)

        # Compute power basis for ALL samples [N, degree+1]
        U = self._power_basis(u_normalized, degree)

        # Get basis matrix
        M_prime = self.M_prime.to(device=device, dtype=dtype)

        # Compute ALL local basis values [N, degree+1]
        N_local = U @ M_prime

        # Compute local control-point indices [N, degree+1]
        control_indices = (
            spans.unsqueeze(1) - degree + torch.arange(degree + 1, device=device)
        ).clamp(0, num_control - 1)

        return SparseBasis(N_local, control_indices, num_control)


    def _compute_d2basis_matrix_form(
            self,
            t: torch.Tensor,  # [N]
            knots: torch.Tensor,  # [M]
            num_control: int,
            degree: int
    ) -> 'SparseBasis':
        """Fully vectorized second-derivative B-spline basis evaluation — returns SparseBasis."""
        N = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Find knot spans for ALL samples at once [N]
        spans = torch.searchsorted(knots[degree:-degree], t, right=False) + degree - 1
        spans = spans.clamp(degree, len(knots) - degree - 2)

        # Normalize ALL parameters at once [N]
        u_normalized = (t - knots[spans]) / (knots[spans + 1] - knots[spans] + 1e-10)
        u_normalized = u_normalized.clamp(0.0, 1.0)

        # Compute power basis for ALL samples [N, degree+1]
        U = self._power_basis(u_normalized, degree)

        # Get basis matrix
        M_prime = self.M_double_prime.to(device=device, dtype=dtype)

        # Compute ALL local basis values [N, degree+1]
        N_local = U @ M_prime

        # Compute local control-point indices [N, degree+1]
        control_indices = (
            spans.unsqueeze(1) - degree + torch.arange(degree + 1, device=device)
        ).clamp(0, num_control - 1)

        return SparseBasis(N_local, control_indices, num_control)
    def _power_basis(self, u: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Compute power basis vectors [1, u, u², u³, ...].

        Args:
            u: [N] normalized parameters
            degree: polynomial degree

        Returns:
            [N, degree+1] power basis matrix
        """
        N = u.shape[0]
        U = torch.ones(N, degree + 1, device=u.device, dtype=u.dtype)

        for i in range(1, degree + 1):
            U[:, i] = u ** i

        return U

    def _is_uniform_knot_span(
            self,
            knots: torch.Tensor,
            start: int,
            end: int,
            tol: float = 1e-6
    ) -> bool:
        """Check if a knot span has uniform spacing."""
        if start < 0 or end >= len(knots):
            return False

        diffs = knots[start + 1:end + 1] - knots[start:end]
        return torch.allclose(diffs, diffs[0:1], atol=tol)

    def _adjust_basis_matrix_nonuniform(
            self,
            M: torch.Tensor,
            knots: torch.Tensor,
            span: torch.Tensor,  # Now a tensor [N] instead of scalar
            degree: int
    ) -> torch.Tensor:
        """
        Adjust the basis matrix for non-uniform knot spacing (vectorized version).

        This implements the transformation from uniform to non-uniform
        B-splines using knot interval ratios for ALL samples at once.

        Args:
            M: Basis matrix [degree+1, degree+1]
            knots: Knot vector [M]
            span: Knot span indices [N] - VECTORIZED
            degree: B-spline degree

        Returns:
            M_adjusted: Adjusted basis matrices [N, degree+1, degree+1]
        """
        p = degree
        N = span.shape[0]
        device = knots.device

        # Broadcast M to batch dimension [N, p+1, p+1]
        M_adjusted = M.unsqueeze(0).expand(N, p + 1, p + 1).clone()

        # Extract local knot subsequences for ALL spans at once
        # We need knots[span[i] - p : span[i] + p + 2] for each i

        # Create index offsets: [-p, -p+1, ..., p+1] (length 2p+2)
        offsets = torch.arange(-p, p + 2, device=device)  # [2p+2]

        # Broadcast to get indices for all spans: [N, 2p+2]
        indices = span.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 2p+2]

        # Clamp indices to valid range
        indices = indices.clamp(0, len(knots) - 1)

        # Gather knot values: [N, 2p+2]
        knot_local = knots[indices]

        # Compute knot intervals: [N, 2p+1]
        intervals = knot_local[:, 1:] - knot_local[:, :-1]
        intervals = intervals.clamp(min=1e-10)  # Avoid division by zero

        # Apply interval-based scaling (simplified version)
        # Full implementation would use Oslo algorithm or knot insertion matrix

        # For each basis function column, scale by relative interval
        # This is a simplified heuristic - for production, consider:
        # 1. Oslo algorithm for exact non-uniform conversion
        # 2. Knot insertion-based refinement
        # 3. Recursive Cox-de Boor evaluation

        for i in range(p + 1):
            if i < intervals.shape[1]:
                # Scale column i by ratio of interval i to first interval
                # [N, 1]
                scale = intervals[:, i] / (intervals[:, 0] + 1e-10)
                # Apply scaling to column i for all rows
                M_adjusted[:, :, i] *= scale.unsqueeze(1)

        return M_adjusted

    def compute_derivatives(
            self,
            u: torch.Tensor,
            v: torch.Tensor,
            knots_u: torch.Tensor,
            knots_v: torch.Tensor,
            num_control_u: int,
            num_control_v: int,
            order: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute derivative basis matrices using matrix form.

        Uses the derivative formula:
            dN/du = U^T · D^T · M · C
            d²N/du² = U^T · (D²)^T · M · C

        This is fully differentiable and efficient.
        """
        if order == 1:
            dbu = self._compute_derivative_basis_matrix(
                u, knots_u, num_control_u, self.degree, order=1
            )
            dbv = self._compute_derivative_basis_matrix(
                v, knots_v, num_control_v, self.degree, order=1
            )
        elif order == 2:
            dbu = self._compute_derivative_basis_matrix(
                u, knots_u, num_control_u, self.degree, order=2
            )
            dbv = self._compute_derivative_basis_matrix(
                v, knots_v, num_control_v, self.degree, order=2
            )
        else:
            raise ValueError(f"Derivative order {order} not supported")

        return dbu, dbv

    def _compute_derivative_basis_matrix(
            self,
            t: torch.Tensor,
            knots: torch.Tensor,
            num_control: int,
            degree: int,
            order: int = 1
    ) -> torch.Tensor:
        """
        Compute derivative basis using matrix formulation.

        Key insight: dN/du = U^T · M' where M' = D^T · M

        For the kth derivative: N^(k) = U^T · (D^k)^T · M
        """
        N = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Initialize output
        deriv_basis = torch.zeros(N, num_control, device=device, dtype=dtype)

        # Get basis matrix
        M = self._get_uniform_basis_matrix(degree).to(device=device, dtype=dtype)

        # Apply derivative operator
        if order == 1:
            M_deriv = self.D @ M  # First derivative basis matrix
        elif order == 2:
            M_deriv = self.D2 @ M  # Second derivative basis matrix
        else:
            raise ValueError(f"Order {order} not supported")

        # Process all sample points
        for i in range(num_control):
            t_start = knots[i]
            t_end = knots[i + degree + 1]

            if t_end - t_start < 1e-10:
                continue

            # Find samples in this span's influence range
            mask = (t >= knots[i + degree]) & (t < knots[i + degree + 1])

            if not mask.any():
                continue

            t_local = t[mask]
            span_idx = i + degree

            # Normalize to [0, 1]
            span_length = knots[span_idx + 1] - knots[span_idx]
            u_normalized = (t_local - knots[span_idx]) / span_length
            u_normalized = u_normalized.clamp(0.0, 1.0)

            # Build power basis
            U = self._power_basis(u_normalized, degree)  # [N_local, p+1]

            # Apply derivative basis matrix
            dN_local = U @ M_deriv  # [N_local, degree+1]

            # Scale by chain rule: d/dt = d/du · du/dt = d/du · 1/span_length
            # For kth derivative: (1/span_length)^k
            scale_factor = (1.0 / span_length) ** order
            dN_local = dN_local * scale_factor

            # Adjust for non-uniform knots if needed
            if not self._is_uniform_knot_span(knots, span_idx - degree, span_idx + 1):
                # Apply non-uniform correction
                dN_local = self._adjust_derivative_nonuniform(
                    dN_local, knots, span_idx, degree, order
                )

            # Place in global matrix
            start_idx = span_idx - degree
            for j in range(degree + 1):
                if start_idx + j < num_control:
                    deriv_basis[mask, start_idx + j] = dN_local[:, j]

        return deriv_basis

    def _adjust_derivative_nonuniform(
            self,
            dN: torch.Tensor,
            knots: torch.Tensor,
            span: int,
            degree: int,
            order: int
    ) -> torch.Tensor:
        """
        Adjust derivative values for non-uniform knot spacing.
        """
        # For derivatives, the knot spacing affects the scaling
        # This is handled primarily by the chain rule scaling above,
        # but additional corrections may be needed for highly non-uniform cases

        return dN  # Simplified - full implementation would apply interval corrections

    def compute_mixed_derivative(
            self,
            u: torch.Tensor,
            v: torch.Tensor,
            knots_u: torch.Tensor,
            knots_v: torch.Tensor,
            num_control_u: int,
            num_control_v: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute basis matrices for mixed partial derivative ∂²/∂u∂v.

        For surfaces S(u,v) = U^T · M_u · P · M_v^T · V:
            ∂²S/∂u∂v = U^T · M'_u · P · M'_v^T · V

        Returns:
            dbu: First derivative basis in U [Us, H]
            dbv: First derivative basis in V [Vs, W]
        """
        dbu = self._compute_derivative_basis_matrix(
            u, knots_u, num_control_u, self.degree, order=1
        )
        dbv = self._compute_derivative_basis_matrix(
            v, knots_v, num_control_v, self.degree, order=1
        )

        return dbu, dbv

#
# # Example usage function
# def example_usage():
#     """Demonstrate matrix-form B-spline evaluation and derivatives."""
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # Create basis module
#     basis = DifferentiableBSplineBasis(degree=3, device=device)
#
#     # Setup parameters
#     num_control_u = 10
#     num_control_v = 10
#     degree = 3
#
#     # Create uniform knot vectors
#     knots_u = torch.linspace(0, 1, num_control_u + degree + 1, device=device)
#     knots_v = torch.linspace(0, 1, num_control_v + degree + 1, device=device)
#
#     # Sample points
#     u = torch.linspace(0, 1, 50, device=device, requires_grad=True)
#     v = torch.linspace(0, 1, 50, device=device, requires_grad=True)
#
#     # Compute basis matrices
#     bu, bv = basis(u, v, knots_u, knots_v, num_control_u, num_control_v)
#     print(f"Basis matrix U shape: {bu.shape}")  # [50, 10]
#     print(f"Basis matrix V shape: {bv.shape}")  # [50, 10]
#
#     # Compute derivatives
#     dbu, dbv = basis.compute_derivatives(
#         u, v, knots_u, knots_v, num_control_u, num_control_v, order=1
#     )
#     print(f"First derivative U shape: {dbu.shape}")
#
#     # Test autograd
#     loss = bu.sum()
#     loss.backward()
#     print(f"Gradient computed: {u.grad is not None}")
#
