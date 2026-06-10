"""
Stubs for CUDA-only extension modules so the math in `modules/` can be
imported and tested on a CPU-only machine.

Import this module FIRST in any test file:

    import tests.conftest_stubs  # noqa: F401
"""
import sys
import types

import torch


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _distCUDA2_cpu(points: torch.Tensor) -> torch.Tensor:
    """CPU fallback: mean squared distance to 3 nearest neighbors."""
    d2 = torch.cdist(points, points).pow(2)
    d2.fill_diagonal_(float("inf"))
    k = min(3, points.shape[0] - 1)
    nn3 = d2.topk(k, largest=False).values
    return nn3.mean(dim=-1)


# simple_knn (CUDA K-nearest-neighbour)
_simple_knn = _stub("simple_knn")
_stub("simple_knn._C", distCUDA2=_distCUDA2_cpu)
_simple_knn._C = sys.modules["simple_knn._C"]

# diff_plane_rasterization (PGSR CUDA rasterizer)
class _FakeSettings:  # pragma: no cover - never actually rasterized on CPU
    def __init__(self, *a, **k):
        raise RuntimeError("Rasterizer unavailable in CPU tests")


class _FakeRasterizer:
    def __init__(self, *a, **k):
        raise RuntimeError("Rasterizer unavailable in CPU tests")


_diff = _stub(
    "diff_plane_rasterization",
    GaussianRasterizationSettings=_FakeSettings,
    GaussianRasterizer=_FakeRasterizer,
)

# pytorch3d (only matrix_to_quaternion / quaternion utils are used)
try:
    import pytorch3d  # noqa: F401
except ImportError:
    _p3d = _stub("pytorch3d")

    def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """Rotation matrix [..., 3, 3] -> quaternion [..., 4] (w, x, y, z)."""
        batch = R.shape[:-2]
        m = R.reshape(-1, 3, 3)
        w = torch.sqrt(torch.clamp(1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], min=1e-12)) / 2
        x = (m[:, 2, 1] - m[:, 1, 2]) / (4 * w)
        y = (m[:, 0, 2] - m[:, 2, 0]) / (4 * w)
        z = (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)
        return torch.stack([w, x, y, z], dim=-1).reshape(*batch, 4)

    def quaternion_to_matrix(quats: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quats.unbind(-1)
        two_s = 2.0 / (quats * quats).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
                two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
                two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y),
            ),
            -1,
        )
        return o.reshape(quats.shape[:-1] + (3, 3))

    def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ),
            -1,
        )

    _stub(
        "pytorch3d.transforms",
        matrix_to_quaternion=matrix_to_quaternion,
        quaternion_to_matrix=quaternion_to_matrix,
        quaternion_multiply=quaternion_multiply,
    )
    _p3d.transforms = sys.modules["pytorch3d.transforms"]
