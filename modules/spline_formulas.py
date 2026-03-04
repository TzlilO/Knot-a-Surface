import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion

def uv_tangent(tangents_u, tangents_v, dudv=None, lambda_curv=0.1):
    # Compute basis
    normals = torch.cross(tangents_u, tangents_v, dim=-1)
    eps = 1e-10
    normals = normals / (normals.norm(dim=-1, keepdim=True) + eps)
    tu_norm = tangents_u / (tangents_u.norm(dim=-1, keepdim=True) + eps)
    tv_norm = tangents_v / (tangents_v.norm(dim=-1, keepdim=True) + eps)
    tv_ortho = tv_norm - (torch.einsum('hwi,hwi->hw', tv_norm, tu_norm).unsqueeze(-1) * tu_norm)
    tv_ortho = tv_ortho / (tv_ortho.norm(dim=-1, keepdim=True) + eps)

    # Rotation
    vec = normals[..., 2]#.mean()
    normals[vec <= 0] *= -1
    R = torch.stack([tangents_u, tv_ortho, normals], dim=-1)
    # R = torch.stack([normals.cross(tv_ortho), tv_ortho, normals], dim=-1)
    rotation = F.normalize(matrix_to_quaternion(R.view(-1, 3, 3)).reshape(*tangents_u.shape[:2], 4), dim=-1)
    return rotation
