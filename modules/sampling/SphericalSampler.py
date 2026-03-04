import torch
from torch import nn

from model.modules import ModelState, SamplerUV
from utils.sh_utils import SH2RGB, eval_sh


class SphericalSamplerUV(SamplerUV):  # Subclass SamplerUV
    def __init__(self, state: ModelState, **kwargs):
        super().__init__(state, **kwargs)
        self.spherical_scale = torch.tensor([2 * torch.pi, torch.pi], device=self.device)  # For θ, φ scaling

    def uv_to_spherical(self, uv_grid: torch.Tensor) -> torch.Tensor:
        """
        Map UV grid (HxW, 2) to unit sphere Cartesian (HxW, 3).
        uv_grid: Tensor of shape (H*W, 2) or (H, W, 2).
        Returns: (H, W, 3) Cartesian points on unit sphere.
        """
        if uv_grid.dim() == 3:
            uv_grid = uv_grid.view(-1, 2)  # Flatten to (H*W, 2)

        theta = uv_grid[:, 0] * self.spherical_scale[0]  # [0,1] -> [0, 2π]
        phi = uv_grid[:, 1] * self.spherical_scale[1]  # [0,1] -> [0, π]

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        spherical_points = torch.stack([x, y, z], dim=-1)  # (H*W, 3)
        return spherical_points.view(self.state.Us, self.state.Vs, 3)  # Reshape to grid

class ViewConditionedMLP(nn.Module):
    def __init__(self, input_dim: int = 3 + 11, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LogSigmoid()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LogSigmoid()]
        layers += [nn.Linear(hidden_dim, 2 * (2) ** 2), nn.Sigmoid()]  # Output [0,1] UV
        self.net = nn.Sequential(*layers)

    def forward(self, spherical_points: torch.Tensor, view_emb: torch.Tensor) -> torch.Tensor:
        # spherical_points: (N, 3), view_emb: (1, emb_dim) repeated to (N, emb_dim)
        inputs = torch.cat([spherical_points, view_emb.expand(spherical_points.size(0), -1)], dim=-1)
        return self.net(inputs)


# Helper to get view embedding (add to SplineModel or viewpoint_cam)
def get_view_embedding(viewpoint_cam) -> torch.Tensor:
    pos = viewpoint_cam.camera_center  # (3,)
    dir = viewpoint_cam.get_forward_dir()  # Assume method to get forward; or compute from transform
    up = viewpoint_cam.get_up_vector()     # Similarly
    fov = torch.tensor([viewpoint_cam.FoVx, viewpoint_cam.FoVy])  # (2,)
    features = torch.cat([pos, dir, up, fov], dim=0)  # (10,)
    # Simple embedding: linear projection or pos-enc (from utils.sh_utils or similar)
    emb = F.linear(features.unsqueeze(0), torch.randn(64, 10, device=features.device)).squeeze(0)  # (64,)
    return emb