"""
Quaternion utilities for rotation interpolation and conversion.

Used by RotationControl for SLERP blending during knot insertion/removal,
and by any module that needs rotation matrix ↔ quaternion conversion.
"""

import torch
import torch.nn.functional as F


def quaternion_mean(quats: torch.Tensor) -> torch.Tensor:
    """
    Compute average quaternion via Markley's eigenvector method.

    Args:
        quats: Quaternions [N, 4] in (w, x, y, z) format.

    Returns:
        Mean quaternion [4], unit-normalized.
    """
    Q = (quats.unsqueeze(-1) * quats.unsqueeze(-2)).sum(dim=0) / quats.shape[0]
    _, eigenvectors = torch.linalg.eigh(Q)
    mean_quat = eigenvectors[:, -1]
    return F.normalize(mean_quat, dim=0)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions using Shepperd's method.

    Args:
        R: Rotation matrices [N, 3, 3].

    Returns:
        Quaternions [N, 4] in (w, x, y, z) format, unit-normalized.
    """
    batch_dim = R.shape[0]
    device = R.device

    q = torch.zeros(batch_dim, 4, device=device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask0 = trace > 0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2
        q[mask0, 0] = 0.25 * s
        q[mask0, 1] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / s
        q[mask0, 2] = (R[mask0, 0, 2] - R[mask0, 2, 0]) / s
        q[mask0, 3] = (R[mask0, 1, 0] - R[mask0, 0, 1]) / s

    # Case 2: R[0,0] largest diagonal
    mask1 = (~mask0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s

    # Case 3: R[1,1] largest diagonal
    mask2 = (~mask0) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s

    # Case 4: R[2,2] largest diagonal
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        q[mask3, 3] = 0.25 * s

    return F.normalize(q, dim=-1, eps=1e-6)


def slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Spherical Linear Interpolation between quaternions.

    Ensures constant angular velocity interpolation on the rotation manifold.
    Falls back to normalized lerp for near-parallel quaternions.

    Args:
        q0: Start quaternions [N, 4] or [4].
        q1: End quaternions [N, 4] or [4].
        t:  Interpolation factor in [0, 1].
        eps: Numerical stability constant.

    Returns:
        Interpolated quaternions, same shape as input.
    """
    device = q0.device
    original_shape = q0.shape

    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)
        q1 = q1.unsqueeze(0)

    q0 = F.normalize(q0, dim=-1, eps=eps)
    q1 = F.normalize(q1, dim=-1, eps=eps)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # Take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    linear_threshold = 0.9995
    use_linear = dot > linear_threshold

    if use_linear.all():
        q_interp = (1 - t) * q0 + t * q1
        q_interp = F.normalize(q_interp, dim=-1, eps=eps)
    else:
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        w0 = torch.sin((1 - t) * theta) / (sin_theta + eps)
        w1 = torch.sin(t * theta) / (sin_theta + eps)

        q_slerp = w0 * q0 + w1 * q1

        q_linear = (1 - t) * q0 + t * q1
        q_linear = F.normalize(q_linear, dim=-1, eps=eps)

        q_interp = torch.where(use_linear.expand_as(q_slerp), q_linear, q_slerp)
        q_interp = F.normalize(q_interp, dim=-1, eps=eps)

    if len(original_shape) == 1:
        q_interp = q_interp.squeeze(0)

    return q_interp


def batch_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized SLERP with per-element interpolation factors.

    Args:
        q0: Start quaternions [N, 4].
        q1: End quaternions [N, 4].
        t:  Per-element interpolation factors [N, 1] or [N].

    Returns:
        Interpolated quaternions [N, 4].
    """
    if t.ndim == 1:
        t = t.unsqueeze(-1)

    q0 = F.normalize(q0, dim=-1, eps=1e-6)
    q1 = F.normalize(q1, dim=-1, eps=1e-6)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w0 = torch.sin((1 - t) * theta) / (sin_theta + 1e-6)
    w1 = torch.sin(t * theta) / (sin_theta + 1e-6)

    q_interp = w0 * q0 + w1 * q1
    return F.normalize(q_interp, dim=-1, eps=1e-6)