"""
uv_tangent must produce orthonormal, right-handed (det=+1) frames and
let gradients flow back to the tangents.

Run:  .venv/bin/python tests/test_rotation_frame.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tests.conftest_stubs  # noqa: F401

import torch

from modules.spline_formulas import uv_tangent
from modules.control_feature.quaternion_utils import quaternion_to_matrix

torch.manual_seed(0)


def main():
    H, W = 12, 10
    # Random, non-unit, non-orthogonal tangents incl. downward-facing normals
    tu = torch.randn(H, W, 3, dtype=torch.float64) * 3.0
    tv = torch.randn(H, W, 3, dtype=torch.float64) * 0.3
    tu.requires_grad_(True)
    tv.requires_grad_(True)

    quats = uv_tangent(tu, tv)
    R = quaternion_to_matrix(quats.reshape(-1, 4)).reshape(H, W, 3, 3)

    eye = torch.eye(3, dtype=R.dtype).expand(H, W, 3, 3)
    ortho_err = (R @ R.transpose(-1, -2) - eye).abs().max().item()
    det = torch.linalg.det(R)
    det_err = (det - 1.0).abs().max().item()

    # Normal column (3rd) must align with +/- normalized(tu x tv) and have nz > 0
    n = R[..., 2]
    nz_min = n[..., 2].min().item()

    # Gradient flow
    quats.sum().backward()
    grad_ok = (
        tu.grad is not None and torch.isfinite(tu.grad).all().item()
        and tu.grad.abs().sum().item() > 0
        and torch.isfinite(tv.grad).all().item()
    )

    checks = {
        "orthonormal (R R^T = I)": ortho_err < 1e-9,
        "proper rotation (det = +1)": det_err < 1e-9,
        "consistent orientation (n_z >= 0)": nz_min >= -1e-12,
        "gradients flow to tangents": grad_ok,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    ortho_err={ortho_err:.2e}  det_err={det_err:.2e}  nz_min={nz_min:.2e}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
