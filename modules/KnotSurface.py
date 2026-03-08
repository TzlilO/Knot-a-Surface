from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch

from ply_export import save_ply_single_surface
from .control_feature import ControlFeature, PositionControl, WeightControl, ScalingControl, RotationControl, OpacityControl, SHControl, SHControlWrapper

from modules.tessellation.chhugani import (
    attach_chhugani_to_surface,
    ForwardContext,
)


from .basis import BasisFunction, compute_bases_uv_diff
from .knotvector import KnotVector
from .sampling.SamplerUV import SamplerUV
from .ModelState import ModelState, SamplingMode

from modules.spline_formulas import uv_tangent
from pytorch3d.transforms import quaternion_to_matrix
from utils.general_utils import (
    get_expon_lr_func, strip_symmetric, build_scaling_rotation, inverse_sigmoid
)
import opt_einsum as oe

from utils.sh_utils import RGB2SH
from .sampling.ray_utils import compute_ray_info, compute_oriented_normals
from .tessellation import (
    GridTessellator,
    TriangleMesh,
    TessellationConfig,
    mesh_to_obj,
    mesh_to_ply,
)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def _downsample_to_ctrl(dense_values, feature_channels, Hs, Ws, normalize=False, target_u=None, target_v=None):
    grid = dense_values.view(Hs, Ws, feature_channels).permute(2, 0, 1).unsqueeze(0)
    ctrl_grid = F.interpolate(grid, size=(target_u, target_v), mode='bilinear', align_corners=True).squeeze(0).permute(
        1, 2, 0).view(-1, feature_channels)
    if normalize:
        ctrl_grid = F.normalize(ctrl_grid, dim=-1)
    return ctrl_grid

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def get_optimal_path(subscript, *tensors):
    """Preview einsum path for debugging/mem estimation."""
    shapes = [t.shape for t in tensors]
    path, info = oe.contract_path(subscript, *shapes)
    print(f"Path: {path}, Peak Mem: {info.opt_cost / 1e9:.2f} GiB")  # Use before einsum
    return path


def latent_to_feature(lat, pe_levels, min_val=0.0, max_val=0.9999):
    """Synthesizes a feature value using a DESCENDING basis."""
    fac = 0.5 ** torch.arange(pe_levels, device=lat.device)
    factors = fac.view(1, 1, 1, pe_levels)
    synthesis = (torch.sin(lat) * factors).sum(-1)
    scale = (max_val - min_val) / 2.0
    bias = (max_val + min_val) / 2.0
    return synthesis * scale + bias


def feature_to_latent(feat, pe_levels, min_val=0.0, max_val=1.0):
    """Finds the latent representation using the fast 'asin' trick (for descending basis)."""
    lat_shape = list(feat.shape) + [pe_levels]
    lat = torch.zeros(lat_shape, device=feat.device, dtype=feat.dtype)
    scale = (max_val - min_val) / 2.0
    bias = (max_val + min_val) / 2.0
    normalized_feat = (feat - bias) / scale
    lat[..., 0] = torch.asin(torch.clamp(normalized_feat, -0.99999, 0.99999))
    return lat.flatten(start_dim=-2)


def insert_knot_u_midpoint(controls: torch.Tensor, knots: torch.Tensor, degree: int, u_bar: float,
                           k: int = None):
    H, W, C = controls.shape
    device = controls.device
    if not isinstance(u_bar, torch.Tensor):
        u_bar_t = torch.tensor([u_bar], device=device, dtype=knots.dtype)
    else:
        u_bar_t = u_bar.to(device=device, dtype=knots.dtype).view(1)

    # 1. Determine span index k: knots[k] <= u_bar < knots[k+1]
    if k is None:
        k = torch.searchsorted(knots, u_bar_t, side='right').item() - 1

    max_k = len(knots) - degree - 2
    k = max(degree, min(k, max_k))

    insert_pos = k - degree + 1
    insert_pos = max(1, min(insert_pos, H - 1))

    left_idx = insert_pos - 1
    right_idx = insert_pos

    # 3. Build new control grid [H+1, W, C]
    new_controls = torch.zeros(H + 1, W, C, device=device, dtype=controls.dtype)

    # Copy rows before the insertion point (unchanged)
    new_controls[:insert_pos] = controls[:insert_pos]

    # Insert midpoint row
    new_controls[insert_pos] = 0.5 * (controls[left_idx] + controls[right_idx])

    # Copy rows from insert_pos onward (shifted by 1)
    new_controls[insert_pos + 1:] = controls[insert_pos:]

    # 4. Update knot vector
    new_knots = torch.cat([knots[:k + 1], u_bar_t, knots[k + 1:]])

    return new_controls, new_knots
def insert_knot_u(controls: torch.Tensor, knots: torch.Tensor, degree: int, u_bar: float, k: int = None, device='cuda'):

    # Ensure u_bar is a 1D tensor for concatenation
    if not isinstance(u_bar, torch.Tensor):
        u_bar = torch.tensor([u_bar], device=device).view(1)
    else:
        u_bar = u_bar.view(1)

    if k is None:
        k = torch.searchsorted(knots, u_bar, side='right').item() - 1

    # and enough on the right to define the basis.
    max_k = len(knots) - degree - 2
    k = max(degree, min(k, max_k))

    # 2. Prefix Copy: Indices [0, k - degree]
    # These control points are unaffected by the insertion.
    prefix_len = k - degree + 1
    H, W, C = controls.shape
    new_controls = torch.zeros(H + 1, W, C, device=device)

    if prefix_len > 0:
        new_controls[:prefix_len] = controls[:prefix_len]

    # 3. Suffix Copy: Indices [k, H-1] -> [k+1, H]
    # In single knot insertion, Q_{k+1} = P_k. So we copy P_k and onwards
    # to the new array starting at k+1.
    suffix_start = k
    if suffix_start < H:
        new_controls[suffix_start + 1:] = controls[suffix_start:]

    # 4. Interpolation Loop: Indices [k - degree + 1, k]
    # Boehm's recursion: Q_i = (1-a) * P_{i-1} + a * P_i
    i = torch.arange(k - degree + 1, k + 1, device=device)

    # Calculate Alphas
    # Denominator handles safe division; u_bar is broadcasted
    denom = knots[i + degree] - knots[i] + 1e-10
    alpha = (u_bar - knots[i]) / denom
    alpha = alpha.clamp(0.0, 1.0)  # Numerical stability for convexity

    # Interpolate
    # Given strict k bounds, 'i' and 'i-1' are guaranteed valid indices in 'controls'.
    new_controls[i] = (1.0 - alpha[:, None, None]) * controls[i - 1] + \
                      alpha[:, None, None] * controls[i]

    # 5. Update Knot Vector
    new_knots = torch.cat((knots[:k + 1], u_bar, knots[k + 1:]))

    return new_controls, new_knots
def get_forward_dir(viewpoint_cam) -> torch.Tensor:
    """
    Returns the forward direction vector in world space (normalized).
    - Vectorized: Direct matrix slice.
    - Differentiable: Composed of torch ops.
    Returns: torch.Tensor of shape (3,) on self.data_device.
    """
    c2w = viewpoint_cam.c2w  # (4,4) tensor
    forward = -c2w[:3, 2]  # Negate Z-basis for -Z forward convention
    return F.normalize(forward, dim=0)  # Normalize for unit vector (optional but recommended for embeddings)

def get_up_vector(viewpoint_cam) -> torch.Tensor:
    """
    Returns the up direction vector in world space (normalized).
    - Vectorized: Direct matrix slice.
    - Differentiable: Composed of torch ops.
    Returns: torch.Tensor of shape (3,) on self.data_device.
    """
    c2w = viewpoint_cam.c2w  # (4,4) tensor
    up = c2w[:3, 1]  # Y-basis
    return F.normalize(up, dim=0)  # Normalize for consistency

class SplineModel(nn.Module):
    """
    Modular Spline model containing all feature modules.
    Shares a single BasisFunction across all features.
    """
    _uv_recompute_interval: int = -1
    _last_uv_recompute_iter: int = -1
    _snapshot_iteration: int = -1
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_weights_activation = inverse_sigmoid
        self.weights_activation = torch.sigmoid #torch.erf
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
            self,
            **kwargs,
    ):
        super(SplineModel, self).__init__()
        self.setup_functions()
        surf = kwargs.get('surf', None)
        surf_rgb = kwargs.get('surf_rgb', None)
        args = kwargs.get('args', None)
        config = kwargs.get('config', None)
        spatial_lr_scale = kwargs.get('spatial_lr_scale', 1.0)
        use_app = kwargs.get('use_app', False)
        late_init = kwargs.get('late_init', False)
        self.surf_uid = kwargs.get('surf_uid', None)
        self.force_recompute = False
        self.label = kwargs.get('label', 'surface')
        attach_chhugani_to_surface(self)

        if not late_init:
            ctrl_pts, ctrl_rgb, knots_u, knots_v = self.extract_surface(surf, surf_rgb)
            H, W = ctrl_pts.shape[:2]
            self.state = ModelState(opt=config,
                                    H=H,
                                    W=W,
                                    device=config.device,
                                    degree=config.spline_degree[0],
                                    active_sh_degree=0,
                                    max_sh_degree=args.sh_degree,
                                    sampling_density=config.sampling_density, #if not self.is_background is None else 1,
                                    surf_uid=self.surf_uid,
                                    args=args,
                                    label=self.label)


            self.use_app = use_app
            self.device = config.device
            self.spatial_lr_scale = spatial_lr_scale  # or 1.0
            self.res_v = self.res_u = self.state.sampling_density
            self.iteration = 0

            knots_u = torch.sort(knots_u)[0].clone().detach()
            knots_v = torch.sort(knots_v)[0].clone().detach()
            self.knot_u = KnotVector(self.state, direction='u', initial_knots=knots_u, name=f"knot_u_{self.surf_uid}")
            self.knot_v = KnotVector(self.state, direction='v', initial_knots=knots_v, name=f"knot_v_{self.surf_uid}")
            # self._chhugani_aggregation = AggregationState()
            fused_color = RGB2SH((ctrl_rgb)).reshape(-1, 3).clone()
            features = (torch.zeros((H * W, 3, (self.state.max_sh_degree + 1) ** 2)).float().cuda().clone())
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            self.uv_sampler = SamplerUV(
                state=self.state,
                mode='single',
            )
            self.basis = BasisFunction(self.state, self.uv_sampler, knot_u=self.knot_u, knot_v=self.knot_v)
            self.cameras = kwargs.get('cameras', None)
            pos_init = ctrl_pts.reshape(H * W, -1).squeeze().contiguous()
            weights_init = None
            if self.refine_weights_active:
                # 1. Generate Noise
                noise = torch.rand(H * W, 1, device=self.device) * 0.05 #- 0.025  # Small noise in [-0.025, 0.025]
                weights_init = torch.full((H * W, 1), fill_value=0.95, device=self.device) + noise  # Base value + noise
                weights_init = self.inverse_weights_activation(weights_init)  # Inverse activation to get initial params

            scaling_ch = self.state.scaling_dims
            self.update_sampling_density(1)

            self.basis.recompute()
            contract_path = self.basis.contract_path
            from modules.basis.basis_matrix import SparseBasis
            dbu = self.basis.dbu
            bv = self.basis.bv
            bu = self.basis.bu
            dbv = self.basis.dbv
            dbu_dense = dbu.to_dense() if isinstance(dbu, SparseBasis) else dbu
            bv_dense = bv.to_dense() if isinstance(bv, SparseBasis) else bv
            bu_dense = bu.to_dense() if isinstance(bu, SparseBasis) else bu
            dbv_dense = dbv.to_dense() if isinstance(dbv, SparseBasis) else dbv
            dSu = oe.contract(contract_path, dbu_dense, ctrl_pts, bv_dense).reshape(H, W, -1)# / self.Us

            dSv = oe.contract(contract_path, bu_dense, ctrl_pts, dbv_dense).reshape(H, W, -1)#  / self.Vs

            self._init_position(pos_init, weights_init)
            self._init_opacity(H, W)
            self._init_sh_features(features)
            self._init_rots(dSu, dSv, H, W)
            self._init_scaling(H, W, dSu, dSv, scaling_ch)

            self.cameras = kwargs.get('cameras', list())
            if kwargs['train_cam_uids'] is not None:
                self.train_cam_uids = kwargs['train_cam_uids']
            self.num_train_views = len(self.train_cam_uids)

            self._uv_recompute_interval = np.inf # self.num_train_views if (uv_recompute <= 0 and self._sampling_mode != SamplingMode.STATIC) else uv_recompute
            self._last_uv_recompute_iter = -1
            self._snapshot_iteration = None

            self._last_subdivision_step = 0
            self.update_sampling_density(self.state.opt.sampling_density)
            self._forward_context = ForwardContext()
            self._forward_context = ForwardContext()
            torch.cuda.empty_cache()
            self.invalidate_all_caches(force=True)


    def _init_position(self, pos_init, weights_init, pos_name=None, **kwargs):
        if pos_name is None:
            pos_name = f"xyz_{self.surf_uid}" if self.surf_uid is not None else "xyz"
        weights_name = kwargs.get('weights_name', None)
        if weights_name is None:
            weights_name = f"weights_{self.surf_uid}" if self.surf_uid is not None else "weights"
        self.position = PositionControl(
            self.state,
            pos_init,
            self.basis,
            name=pos_name,
            use_pe=self.state.opt.use_pe,  # New config option
            pe_levels=self.state.opt.pe_levels,  # e.g., 6 for positions
        )
        self.weights = WeightControl(self.state, weights_init, self.basis,
                                     name=weights_name)
        self.position.set_weights(self.weights)
    def update_sampling_density(self, new_density: float) -> None:
        self.state.update_sampling_density(new_density)
        self.uv_sampler.create_uv_grid(self.state.Us, self.state.Vs)
    @property
    def is_background(self):
        return self.state.label == 'background'
    def _init_opacity(self, H, W, opacity_name=None,  **kwargs):

        if opacity_name is None:
            opacity_name = f"opacity_{self.surf_uid}" if self.surf_uid is not None else "opacity"
        opac_param = None
        target_opacity = 0.1
        if self.refine_opacity_active:
            initial_opac = torch.full((H * W, 1),
                                      fill_value=target_opacity,
                                      device=self.device)  # .squeeze(0)
            # if self.is_background:
            #     target_opacity = 0.85  # Background should start HIGH opacity
            #     initial_opac = torch.full((H * W, 1), fill_value=target_opacity, device=self.device)
            # else:
            #     target_opacity = 0.5
            #     initial_opac = torch.full((H * W, 1), fill_value=target_opacity, device=self.device)

            opac_param = self.inverse_opacity_activation(initial_opac)  # Convert to raw parameter space
        self.opacity = OpacityControl(self.state, opac_param, self.basis, name=opacity_name)

    def _init_sh_features(self, features, dc_name=None, rest_name=None, **kwargs):
        if dc_name is None:
            dc_name = f"f_dc_{self.surf_uid}" if self.surf_uid is not None else "f_dc"
        if rest_name is None:
            rest_name = f"f_rest_{self.surf_uid}" if self.surf_uid is not None else "f_rest"

        sh_dc = SHControl(
            self.state, features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True), self.basis,
            sh_component='dc',
            name=dc_name,
            **kwargs
        )
        sh_rest = SHControl(
            self.state, features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True), self.basis,
            sh_component='rest',
            name=rest_name,
            **kwargs
        )
        self.spherical_harmonics = SHControlWrapper(self.state, sh_dc, sh_rest)

    def _init_scaling(self, H, W, dSu, dSv, scaling_ch, scaling_name=None, apply_sqrt=False, **kwargs):
        if scaling_name is None:
            scaling_name = f"scaling_{self.surf_uid}" if self.surf_uid is not None else "scaling"
        scaling_init = None
        if self.state.opt.refine_scales:
            if self.state.opt.residual_scaling:
                scaling_init = self.scaling_inverse_activation(
                    torch.ones((H * W, scaling_ch), device=self.device)).requires_grad_(True).contiguous()
            else:
                scale_u = dSu.reshape(self.state.H, self.state.W, 3).norm(
                    dim=-1) * self.uv_sampler.delta_u / 2
                scale_v = dSv.reshape(self.state.H, self.state.W, 3).norm(
                    dim=-1) * self.uv_sampler.delta_v / 2
                if apply_sqrt:
                    scale_u, scale_v = scale_u.sqrt(), scale_v.sqrt()
                if scaling_ch == 2:
                    scaling_init = torch.stack([scale_u, scale_v], dim=-1).reshape(self.state.H * self.state.W,
                                                                                   scaling_ch)
                else:
                    scale_n = 1e-8 * torch.ones_like(scale_u)  # Placeholder for normal-based scaling
                    scaling_init = torch.stack([scale_u, scale_v, scale_n], dim=-1).reshape(
                        self.state.H * self.state.W, scaling_ch)

        # scaling_init = self.scaling_inverse_activation((torch.ones((H * W, scaling_ch), device=self.device) / (self.spatial_lr_scale * (self.H * self.W))).sqrt()).requires_grad_(True).contiguous()
        scaling_init = self.scaling_inverse_activation(scaling_init.clamp_min(1e-10)).requires_grad_(True).contiguous()
        self.scaling = ScalingControl(self.state, scaling_init, self.basis, name=scaling_name)

    def _init_rots(self, dSu, dSv, H, W, rotation_name=None, **kwargs):
        if rotation_name is None:
            rotation_name = f"rotation_{self.surf_uid}" if self.surf_uid is not None else "rotation"
        if self.state.opt.refine_rotations:
            if not self.state.opt.residual_rots:
                rotation_init = F.normalize(uv_tangent(dSu, dSv).detach().clone().reshape(H * W, 4).contiguous(),
                                            dim=-1)
            else:
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(H * W, -1)
                rotation_init = F.normalize(
                    identity_quat + uv_tangent(dSu, dSv).detach().clone().reshape(H * W, 4).contiguous(), dim=-1)

        else:
            rotation_init = None
        self.rotation = RotationControl(self.state, rotation_init, self.basis, name=rotation_name)

    def extract_surface(self, surf, surf_rgb, device='cuda', transposed=False, flipped=False):
        ctrl2d = torch.stack([torch.tensor(ctrl2d, device='cuda')[..., :3] for ctrl2d in surf.ctrlpts2d],
                             dim=0)
        knots_u = torch.tensor(surf.knotvector_u, device=device)  # torch.stack(knot_u_list, dim=0)
        knots_v = torch.tensor(surf.knotvector_v, device=device)  # torch.stack(knot_u_list, dim=0)

        if surf_rgb is not None:
            ctrl2d_rgb = torch.stack(
                [torch.tensor(ctrl2d, device='cuda')[..., :3] for ctrl2d in surf_rgb.ctrlpts2d], dim=0)

        else:
            ctrl2d_rgb = torch.full_like(ctrl2d, fill_value=0.5)
        ctrl_pts = ctrl2d.detach().clone()
        ctrl_rgb = ctrl2d_rgb.detach().clone()
        if transposed:# := (knots_u[1] - knots_u[0]) < 0:
            ctrl_pts = ctrl_pts.transpose(0, 1)
            ctrl_rgb = ctrl_rgb.transpose(0, 1)
            knots_u, knots_v = knots_v, knots_u

        if flipped:# := (knots_u[1] - knots_u[0]) < 0:
            ctrl_pts = torch.flip(ctrl_pts, dims=[0])
            ctrl_rgb = torch.flip(ctrl_rgb, dims=[0])
            knots_u = torch.flip(knots_u, dims=[0])
        return ctrl_pts, ctrl_rgb, knots_u, knots_v


    def recompute(self):
        if self.should_recompute():
            self.basis.recompute()

    def should_recompute(self):
        if self.force_recompute:
            return True
        if self.uv_sampler.should_optimize or self.knot_u.should_optimize or self.knot_v.should_optimize:
            return True
        if self.iteration - self._last_uv_recompute_iter >= self._uv_recompute_interval:
            return True
        if self.basis.bu is None or self.basis.bv is None:
            return True
        return False

    @property
    def refine_weights_active(self):
        return self.state.opt.refine_weights

    @property
    def refine_opacity_active(self):
        return self.state.opt.refine_opacities

    @property
    def refine_scales_active(self):
        return self.state.opt.refine_scales

    @property
    def residual_rots(self):
        return self.state.opt.residual_rots

    @property
    def refine_rotations_active(self):
        return self.state.opt.refine_rotations

    def save_ply(self, path: str, viewpoint_cam=None):
        """
        Save model to PLY format compatible with GaussianSplatting viewers.
        """
        xyz = self.get_xyz
        if viewpoint_cam is not None:
            normals = self.get_normal(viewpoint_cam)
        elif hasattr(self, 'surface_normals_raw'):
            normals = self.surface_normals().view(-1, 3)
        else:
            normals = self.get_smallest_axis()

        features = self.get_features
        if features.dim() == 3:
            features_dc = features[:, :1, :]
            features_rest = features[:, 1:, :]
        else:
            features_dc = features.unsqueeze(1)
            features_rest = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device)

        # opacities = self.opacity.cache if self.opacity.cache is not None else \
        opacities = self.get_opacity

        scaling = self.get_scaling


        rotation = self.get_rotation

        save_ply_single_surface(
            path=path,
            xyz=xyz.reshape(-1, 3),
            normals=normals.reshape(-1, 3),
            features_dc=features_dc,
            features_rest=features_rest,
            opacities=opacities.reshape(-1, 1),
            scaling=scaling.reshape(-1, 3),
            rotation=rotation.reshape(-1, 4),
        )

    def compute_grid_normals(self, cam):
        return self.get_normal(cam)
    @torch.no_grad()
    def derive_scale(self):
        scale_u = (self.position.dSu.reshape(self.state.Us, self.state.Vs, 3).norm(dim=-1)) * self.uv_sampler.delta_u #* 1.6  #* self.spatial_lr_scale  #* 0.5# / (self.state.Us * 2)#
        scale_v = (self.position.dSv.reshape(self.state.Us, self.state.Vs, 3).norm(dim=-1)) * self.uv_sampler.delta_v #* 1.6  #* self.spatial_lr_scale # * 0.5# /(self.state.Vs * 2)#* self.uv_sampler.delta_v


        scale_normal = torch.full_like(scale_u, fill_value=1e-6)

        return torch.stack([scale_u, scale_v, scale_normal], dim=-1).reshape(-1, 3)#.clamp(min=1e-20, max=1e-1 / self.spatial_lr_scale)

    @torch.no_grad()
    def derive_rotation(self, eps=1e-8):
        rotation = uv_tangent(self.position.dSu, self.position.dSv)
        return rotation #/ rotation.norm(dim=-1, keepdim=True)# F.normalize(rotation, dim=-1)

    @property
    def dSu(self):
        if self.position.dsu_cache is not None:
            return self.position.dsu_cache
        else:
            return self.position.dSu

    @property
    def dSv(self):
        if self.position.dsv_cache is not None:
            return self.position.dsv_cache
        else:
            return self.position.dSv

    @property
    def dSuu(self):
        if self.position.dsuu_cache is not None:
            return self.position.dsuu_cache
        else:
            return self.position.dSuu
    @property
    def dSvv(self):
        if self.position.dsvv_cache is not None:
            return self.position.dsvv_cache
        else:
            return self.position.dSvv
    @property
    def Sn(self):
        return torch.cross(self.dSu, self.dSv, dim=-1).contiguous()

    @property
    def total_gaussians(self):
        return self.state.Us * self.state.Vs
    @property
    def grid_shape(self):
        return (self.state.Us, self.state.Vs, -1)

    @property
    def control_grid_shape(self):
        return (self.state.H, self.state.W, -1)
    @property
    def active_sh_degree(self):
        return self.state.active_sh_degree

    @property
    def features_activated(self):
        # Stack all control-features into single tensor:
        control_list = [
            self.position.features,
            self.spherical_harmonics.sh_dc.control_features,
            self.spherical_harmonics.sh_rest.control_features,
            self.scaling_activation(self.scaling.control_features),
            self.rotation_activation(self.rotation.control_features, dim=-1),
            self.opacity_activation(self.opacity.control_features),
        ]
        return torch.cat(control_list, dim=-1).view(self.state.control_layout)  # Shape: (H, W, total_feat_dim)
    @property
    def features(self):
        # Stack all control-features into single tensor:
        control_list = [
            self.position.features.view(-1, self.position.feature_channels),
            self.spherical_harmonics.sh_dc.control_features,
            self.spherical_harmonics.sh_rest.control_features,
            self.scaling.control_features,
            self.rotation.control_features,
            self.opacity.control_features,
        ]
        return torch.cat(control_list, dim=-1).view(self.state.control_layout)  # Shape: (H, W, total_feat_dim)
    @property
    def control_list(self):
        controls = [
            self.position,
            self.spherical_harmonics.sh_dc,
            self.spherical_harmonics.sh_rest,
            self.scaling,
            self.rotation,
            self.opacity
        ]
        if self.weights.control_features is not None:
            controls.append(self.weights)

        return controls


    def invalidate_control_features(self, hard: bool = False):
        """
        Invalidate feature caches after basis change.
        This ensures all properties (xyz, normals, SH, etc.) use the same basis.
        """
        self.position.invalidate(hard)
        self.rotation.invalidate(hard)
        self.scaling.invalidate(hard)
        self.opacity.invalidate(hard)
        self.spherical_harmonics.sh_dc.invalidate(hard)
        self.spherical_harmonics.sh_rest.invalidate(hard)

    @property
    def spherical_uv(self):
        return False


    def oneupSHdegree(self):
        if self.state.active_sh_degree < self.state.max_sh_degree:
            self.state.active_sh_degree += 1

    def get_mod_reg_loss(self, weight: float = 0.01) -> torch.Tensor:
        return weight * self.sampling_mod.control_features.norm(dim=1).mean()  # Encourage subtle warps

    @property
    def parameters_count(self):

        baseline =  (self.position.total_parameters +
                self.spherical_harmonics.sh_dc.total_parameters +
                self.spherical_harmonics.sh_rest.total_parameters)
        if self.refine_opacity_active:
            baseline += self.opacity.total_parameters

        if self.refine_rotations_active:
            baseline += self.rotation.total_parameters
        if self.refine_scales_active:
            baseline += self.scaling.total_parameters
        return baseline

    def invalidate_all_caches(self, force=False):
        self.invalidate_control_features(force)

        if (self.state.opt.optimize_knots or self.state.opt.optimize_intervals or force):
            self._forward_context = ForwardContext()
            self.uv_sampler.invalidate()

            self.basis.clear()

    def update_learning_rate(self, iteration, optimizer=None, lr_scheduler=None):
        """Update learning rates based on schedulers."""
        optimizer = optimizer if optimizer is not None else self.optimizer
        if lr_scheduler is None:
            lr_scheduler = self.scheduler
        for param_group in optimizer.param_groups:

            if param_group["name"] == self.position.name:
                lr = lr_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    @property
    def H(self):
        return self.state.H

    @property
    def W(self):
        return self.state.W

    @property
    def sampling_grid_shape(self):
        return self.Us, self.Vs, -1

    @property
    def num_patches_u(self):
        return self.state.num_patches_u

    @property
    def num_patches_v(self):
        return self.state.num_patches_v

    @property
    def Us(self):
        return self.state.Us

    @property
    def Vs(self):
        return self.state.Vs

    @property
    def feat_layout(self):
        return (self.Us, self.Vs, -1)

    def patch_sampling_view(self):
        return (self.num_patches_u, self.num_patches_v, self.res_u, self.res_v, -1)

    @property
    def shc(self):
        return self.state.shc

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        if self.refine_scales_active:
            smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        else:
            smallest_axis_idx = torch.ones(self.Us * self.Vs, 3, 1, dtype=torch.int64,
                                       device=self.state.device) * 2
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

    @property
    def get_xyz(self) -> torch.Tensor:
        """Sample positions with lazy caching."""
        return self.position().view(-1, 3)


    @property
    def get_opacity(self) -> torch.Tensor:
        """Sample opacities with lazy caching."""

        return self.opacity.forward().view(-1, 1)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Sample rotations."""

        if not self.state.opt.refine_rotations:
            rots = self.derive_rotation()
            return rots.reshape(-1, 4)

        rots = self.rotation.forward().reshape(-1, 4)

        if self.state.opt.residual_rots:
            rots = quaternion_multiply(rots,
                                       self.derive_rotation().reshape(-1, 4))
        return rots.reshape(-1, 4)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Sample scaling with lazy caching."""

        scaling = self.scaling.forward().reshape(-1, 3)
        if self.state.opt.residual_scaling:
            scaling = self.derive_scale().view(-1, 3) * scaling
        return scaling
    @property
    def get_features(self) -> torch.Tensor:
        """Sample SH features with lazy caching."""
        return self.spherical_harmonics.forward().view(-1, self.shc, 3)
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

    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self.get_xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def feature_modules(self) -> List[ControlFeature]:
        """Return list of all feature modules."""
        modules = [self.position, self.rotation, self.scaling, self.opacity, self.spherical_harmonics.sh_dc]
        return modules  # Add others if exist

    def capture(self) -> Tuple:
        """Capture the current state of the model for saving."""
        state = self.state  # ModelState instance
        captured = {
            'position_state': self.position.capture_state(),
            'sh_dc_state': self.spherical_harmonics.sh_dc.capture_state(),
            'sh_rest_state': self.spherical_harmonics.sh_rest.capture_state(),
            'scaling_state':self.scaling.capture_state(),
            'rotation_state':self.rotation.capture_state(),
            'opacity_state': self.opacity.capture_state(),
            'uv_sampler_state': self.uv_sampler.capture_state(),
            'knot_u_state': self.knot_u.capture_state(),
            'knot_v_state': self.knot_v.capture_state(),

            'sampling_mode': self.sampling_mode,
            'spatial_lr_scale': self.spatial_lr_scale,

            'state': {
                'opt': state.opt,
                'H': state.H,
                'W': state.W,
                'base_v': state.base_v,
                'base_u': state.base_u,
                'degree': state.degree,
                'sampling_density': state.sampling_density,
                'active_sh_degree': state.active_sh_degree,
                'max_sh_degree': state.max_sh_degree,
                'scaling_dims': state.scaling_dims,
            },

        }
        if self.state.opt.refine_weights:
            captured.update({'weights_state': self.weights.capture_state()})

        if hasattr(self, 'optimizer'):
            captured['optimizer'] = self.optimizer.state_dict()

        captured['context'] = {
            'warped_us': self._forward_context.warped_us,
            'warped_vs': self._forward_context.warped_vs,
            'warped_uvs': self._forward_context.warped_uvs,
            'uv_basis': self._forward_context.uv_basis,
        }
        return captured

    def restore(self, model_args: dict, train_model=False):
        """Restore the model state from the captured tuple."""
        self.state = ModelState(**model_args['state'])
        self.knot_u = KnotVector.from_state(
            model_args['knot_u_state'],
            self.state,
            evaluate_mode=True

        )
        self.knot_v = KnotVector.from_state(
            model_args['knot_v_state'],
            self.state,
            evaluate_mode=True

        )
        self.spatial_lr_scale = model_args.get('spatial_lr_scale', 1.0)
        uv_state = model_args['uv_sampler_state']
        self.uv_sampler = SamplerUV.from_state(uv_state, self.state, device='cuda')
        self.iteration = model_args.get('iteration', torch.inf)
        self.basis = BasisFunction(self.state,
                                   self.uv_sampler,
                                   self.knot_u,
                                   self.knot_v)

        self.position = PositionControl.from_state(
            model_args['position_state'],
            self.state,
            self.basis,
            device='cuda'
        )

        self.weights = self.position.weights
        self.position.set_weights(self.weights)
        self.scaling = ScalingControl.from_state(
            model_args['scaling_state'],
            self.state,
            self.basis,
            device='cuda'
        ) if self.state.opt.refine_scales else ScalingControl(self.state, None, self.basis)
        self.rotation = RotationControl.from_state(
            model_args['rotation_state'],
            self.state,
            self.basis,
            device='cuda'
        )
        self.opacity = OpacityControl.from_state(
            model_args['opacity_state'],
            self.state,
            self.basis,
            device='cuda'
        )
        self.spherical_harmonics = SHControlWrapper(self.state, None, None)
        self.spherical_harmonics.sh_dc = SHControl.from_state(
            model_args['sh_dc_state'],
            self.state,
            self.basis,
            device='cuda'
        )
        self.spherical_harmonics.sh_rest = SHControl.from_state(
            model_args['sh_rest_state'],
            self.state,
            self.basis,
            device='cuda'
        )

        self._forward_context = ForwardContext()
        self._forward_context.warped_us = model_args.get('context', {}).get('warped_us', {})
        self._forward_context.warped_vs = model_args.get('context', {}).get('warped_vs', {})
        self._forward_context.warped_uvs = model_args.get('context', {}).get('warped_uvs', {})
        self._forward_context.uv_basis =  model_args.get('context', {}).get('uv_basis', {})
        self._uv_recompute_interval = 1  # Recompute every N iterations
        self._last_uv_recompute_iter = -1
        self._snapshot_iteration = None

        # if not train_model:
        #     if self.sampling_mode == SamplingMode.ADAPTIVE:
        #         self.sampling_mode = SamplingMode.ADAPTIVE
        #     elif self.sampling_mode == SamplingMode.STATIC:
        #         self.sampling_mode = SamplingMode.STATIC
        #
        #     else:
        #         self.sampling_mode = SamplingMode.EVALUATION
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            if group['name'] == 'opacity':
                torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def background_coverage_loss(self, render_pkg, gt_image, background_mask):
        """Penalize incorrect background rendering explicitly."""
        if not any(s.is_background for s in self.surfaces):
            return torch.tensor(0.0)

        bg_render = render_pkg['render'] * background_mask
        bg_gt = gt_image * background_mask

        # L1 in background region, weighted by brightness
        brightness_weight = bg_gt.mean(dim=0, keepdim=True)  # Bright regions matter more
        return (F.l1_loss(bg_render, bg_gt, reduction='none') * brightness_weight).mean()
    def reset_opacity(self, optimizer=None):
        if not self.refine_opacity_active:
            return
        if optimizer is None:
            optimizer = self.optimizer
        reset_val = 0.01# if self.is_background else 0.01

        opacities_new = self.inverse_opacity_activation(
            torch.min(self.opacity_activation(self.opacity.control_features),
                      torch.ones_like(self.opacity.control_features) * reset_val)) #.clamp(0.0001, 0.9999))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, self.opacity.name, optimizer=optimizer)
        self.opacity.control_features = optimizable_tensors


    def reset_scaling(self, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        if self.refine_scales_active:
            scaling_param = self.scaling_inverse_activation(
                self.scaling.activation(self.scaling.control_features) * self.state.opt.scaling_reset_factor)
            optimizable_tensors = self.replace_tensor_to_optimizer(scaling_param, self.scaling.name, optimizer=optimizer)
            self.scaling.control_features = optimizable_tensors

    def insert_tensors_to_optimizer(self, tensors_dict, direction='u', degree=None, u_bar=None, insert_idx=None, optimizer=None,
                                    momentum_strategy='zero' # 'zero', 'neighbor_avg', 'neighbor_max', 'copy_nearest'
                                    ):
        optimizable_tensors = {}
        if optimizer is None:
            optimizer = self.optimizer
        is_v = direction == 'v'

        for group in optimizer.param_groups:
            value = tensors_dict.get(group["name"], None)
            if value is None:
                continue

            new_grid, a = value
            name = group["name"]
            if  name.startswith('f_dc'):
                ch = self.spherical_harmonics.sh_dc.control_features.shape[1:]
            elif name.startswith('f_rest'):
                ch = self.spherical_harmonics.sh_rest.control_features.shape[1:]
            else:
                ch = new_grid.shape[-1:]

            params = group['params'][0]
            stored_state = optimizer.state.get(params, None)
            H, W = self.state._H, self.state._W
            # --- 2. Optimizer State Update (Interpolation) ---
            if stored_state is not None:
                exp_avg = stored_state["exp_avg"].view(H, W, -1)
                exp_avg_sq = stored_state["exp_avg_sq"].view(H, W, -1)

                if is_v:
                    knots = self.knot_v()
                    exp_avg = exp_avg.permute(1, 0, 2)
                    exp_avg_sq = exp_avg_sq.permute(1, 0, 2)
                else:
                    knots = self.knot_u()

                new_exp_avg_reshaped, _ = insert_knot_u(exp_avg, knots, degree, u_bar, insert_idx)
                new_exp_avg_sq_reshaped, _ = insert_knot_u(exp_avg_sq, knots, degree, u_bar, insert_idx)
                if is_v:
                    new_exp_avg_reshaped = new_exp_avg_reshaped.permute(1, 0, 2)
                    new_exp_avg_sq_reshaped = new_exp_avg_sq_reshaped.permute(1, 0, 2)

                new_exp_avg = new_exp_avg_reshaped.reshape(-1, *ch)
                new_exp_avg_sq = new_exp_avg_sq_reshaped.reshape(-1, *ch)
                stored_state["exp_avg"] = new_exp_avg.contiguous()
                stored_state["exp_avg_sq"] = new_exp_avg_sq.contiguous()

            # --- 3. Finalize and Replace ---
            new_param = new_grid.reshape(-1, *ch)
            if stored_state is not None:
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(new_param.contiguous(), requires_grad=True)
                optimizer.state[group['params'][0]] = stored_state

            else:
                group["params"][0] = nn.Parameter(new_param.contiguous(), requires_grad=True)

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def _remove_tensors_from_optimizer(
            self,
            tensors_dict: Dict[str, torch.Tensor],
            removed_idx: int,
            direction: str,
            optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, nn.Parameter]:
        """
        Remove rows/columns from optimizer state after knot removal.

        Properly handles Adam momentum (exp_avg, exp_avg_sq) by removing
        the corresponding entries.
        """
        optimizer = optimizer if optimizer is not None else self.optimizer
        optimizable_tensors = {}

        H, W = self.state._H, self.state._W
        is_v = (direction == 'v')

        for group in optimizer.param_groups:
            name = group['name']

            # Handle suffix for multi-surface case
            base_name = name.split('_')[0] if '_' in name else name

            if name not in tensors_dict and base_name not in tensors_dict:
                continue

            new_tensor = tensors_dict.get(name, tensors_dict.get(base_name))
            if new_tensor is None:
                continue

            name = group["name"]
            if  name.startswith('f_dc'):
                ch = self.spherical_harmonics.sh_dc.control_features.shape[1:]
            elif name.startswith('f_rest'):
                ch = self.spherical_harmonics.sh_rest.control_features.shape[1:]
            else:
                ch = new_tensor.shape[-1:]

            old_param = group['params'][0]
            stored_state = optimizer.state.get(old_param, None)


            if stored_state is not None:
                # Reshape momentum to grid
                try:
                    exp_avg = stored_state['exp_avg'].view(H, W, -1)
                    exp_avg_sq = stored_state['exp_avg_sq'].view(H, W, -1)
                except RuntimeError:
                    # Shape mismatch - reinitialize
                    print(f"Warning: Momentum shape mismatch for {name}. Reinitializing momentum to zeros.")
                    new_param_flat = new_tensor.reshape(-1, *ch)
                    group['params'][0] = nn.Parameter(new_param_flat.contiguous().requires_grad_(True))
                    optimizable_tensors[name] = group['params'][0]
                    continue

                if is_v:
                    exp_avg = exp_avg.permute(1, 0, 2)  # [W, H, ch]
                    exp_avg_sq = exp_avg_sq.permute(1, 0, 2)

                # Remove row
                new_exp_avg = torch.cat([
                    exp_avg[:removed_idx],
                    exp_avg[removed_idx + 1:]
                ], dim=0)
                new_exp_avg_sq = torch.cat([
                    exp_avg_sq[:removed_idx],
                    exp_avg_sq[removed_idx + 1:]
                ], dim=0)

                if is_v:
                    new_exp_avg = new_exp_avg.permute(1, 0, 2)  # Back to [H, W-1, ch]
                    new_exp_avg_sq = new_exp_avg_sq.permute(1, 0, 2)

                # Flatten
                new_exp_avg = new_exp_avg.reshape(-1, *ch)
                new_exp_avg_sq = new_exp_avg_sq.reshape(-1, *ch)
                new_param_flat = new_tensor.reshape(-1, *ch)
                stored_state['exp_avg'] = new_exp_avg
                stored_state['exp_avg_sq'] = new_exp_avg_sq

                # Update optimizer
                del optimizer.state[old_param]
                new_param = nn.Parameter(new_param_flat.contiguous().requires_grad_(True))
                group['params'][0] = new_param
                optimizer.state[group['params'][0]] = stored_state

            else:
                new_param_flat = new_tensor.reshape(-1, *ch)
                group['params'][0] = nn.Parameter(new_param_flat.contiguous().requires_grad_(True))

            optimizable_tensors[name] = group['params'][0]

        return optimizable_tensors


    def replace_tensor_to_optimizer(self, tensor, name, uid=None, optimizer=None):
        optimizable_tensors = {}
        if optimizer is None:
            optimizer = self.optimizer

        uid = 0 if uid is None else uid  # For ParameterList
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][uid], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group['params'][uid]]
                group["params"][uid] = nn.Parameter(tensor.contiguous().requires_grad_(True))
                # optimizer.state[group['params'][uid]] = stored_state
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors[name]
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def surface_normals(self):
        t_u, t_v = self.position.dSu, self.position.dSv
        return torch.cross(t_u, t_v, dim=-1)

    def eikonal_loss(self, weight: torch.Tensor = None, reduction='mean') -> torch.Tensor:
        """
        Computes the eikonal regularization loss based on the norm of the cross product of partial derivatives.

        Args:
            weight: Scaling factor for the loss (default: 1.0).

        Returns:
            Scalar tensor representing the eikonal loss.
        """
        t_u, t_v = self.dSu, self.dSv
        normals = torch.cross(t_u, t_v, dim=-1)

        # Compute norm of each normal vector
        norms = torch.linalg.norm(normals, dim=-1)  # [N]

        # Eikonal loss:  penalize deviation from unit length
        # Using (||n|| - 1)^2 for smooth gradients
        eikonal_error = (norms - 1.0).abs()
        eikonal_error = eikonal_error if weight is None else eikonal_error.reshape(-1) * weight.reshape(-1)

        if reduction == 'mean':
            loss = eikonal_error.mean()
        elif reduction == 'sum':
            loss = eikonal_error.sum()
        elif reduction == 'none':
            loss = eikonal_error
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return loss




    @property
    def gaussian_curvature(self):
        """Compute Gaussian curvature K at each sampled point (Us, Vs). Vectorized and differentiable."""
        # First partials
        Su = self.dSu  # (Us, Vs, 3)
        Sv = self.dSv  # (Us, Vs, 3)

        # First fundamental form
        E = torch.einsum('hwc,hwc->hw', Su, Su)  # (Us, Vs)
        F = torch.einsum('hwc,hwc->hw', Su, Sv)
        G = torch.einsum('hwc,hwc->hw', Sv, Sv)
        denom_first = E * G - F ** 2  # (Us, Vs)
        denom_first = denom_first.clamp(min=1e-8)  # Stability

        # Unit normal
        cross = torch.cross(Su, Sv, dim=-1)  # (Us, Vs, 3)
        norm_cross = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        N = cross / norm_cross  # (Us, Vs, 3)

        # Second partials
        Suu = self.d2Suu  # (Us, Vs, 3)
        Suv = self.d2Suv
        Svv = self.d2Svv

        # Second fundamental form
        L = torch.einsum('hwc,hwc->hw', Suu, N)  # (Us, Vs)
        M = torch.einsum('hwc,hwc->hw', Suv, N)
        N_coeff = torch.einsum('hwc,hwc->hw', Svv, N)  # Rename to avoid conflict
        numer = L * N_coeff - M ** 2  # (Us, Vs)

        # Gaussian K
        K = numer / denom_first  # (Us, Vs)

        return K







    @property
    def surface_normals_raw(self) -> torch.Tensor:
        """Raw surface normals from rotation matrix (may point either direction)."""
        return F.normalize(self.dSu, dim=-1).cross(F.normalize(self.dSv, dim=-1), dim=-1) #self.get_smallest_axis()

    def surface_normals_oriented(self, camera) -> torch.Tensor:
        """Surface normals oriented to face the given camera."""
        raw = self.surface_normals_raw
        return compute_oriented_normals(raw, self.get_xyz, camera.camera_center)

    def get_normals(self, camera) -> torch.Tensor:
        """Alias for surface_normals_oriented for backward compatibility."""
        return self.surface_normals_oriented(camera)



    def local_planar_deviation_loss(
            self,
            step: int = 1,  # Grid steps for a/b (1=adjacent; larger for broader check)
            dist_type: str = 'l1',  # 'sq' (MSE), 'l2' (mean dist), 'l1' (mean abs)
    ) -> torch.Tensor:
        """
        Computes deviation loss between actual surface points at uv-perturbed corners
        and their linear approximations using tangents. Encourages local planarity.

        Args:
            step: Int steps in grid for perturbations (a = step * avg_delta_u).
            dist_type: How to measure deviation per corner.

        Returns:
            Scalar loss (mean over centers and corners).
        """
        # Get grids: assume pre-computed and reshaped
        Us, Vs = self.state.Us, self.state.Vs
        xyz_grid = self.get_xyz.view(Us, Vs, -1)  # [Us, Vs, 3]
        Su_grid = self.position.du.view(Us, Vs, -1)  # tangents u
        Sv_grid = self.position.dv.view(Us, Vs, -1)  # tangents v

        # Get sample positions (handles optimized intervals)
        samples_u = self.uv_sampler.interval_u.unsqueeze(1).expand(-1, self.Vs).reshape(self.Us, self.Vs, -1)
        samples_v = self.uv_sampler.interval_v.unsqueeze(0).expand(self.Us, -1).reshape(self.Us, self.Vs, -1)

        if Us <= 2 * step or Vs <= 2 * step:
            return torch.tensor(0.0, device=self.state.device)  # Skip if grid too small

        center_slice_u = slice(step, -step)
        center_slice_v = slice(step, -step)
        xyz_center = xyz_grid[center_slice_u, center_slice_v, :]
        Su_center = Su_grid[center_slice_u, center_slice_v, :]
        Sv_center = Sv_grid[center_slice_u, center_slice_v, :]

        # Local deltas for + and - directions (vectorized)
        uv = torch.cat([samples_u, samples_v], dim=-1).squeeze()  # [Us, Vs, 2]
        pos_delta_uv = (uv[2 * step:, 2 * step:] - uv[step:-step, step:-step])#.unsqueeze(1).unsqueeze(-1)  # [Us-2*step, 1, 1]
        neg_delta_uv = (uv[step:-step, step:-step] - uv[:-2 * step, :-2 * step])#.unsqueeze(0).unsqueeze(-1)  # [1, Vs-2*step, 1]
        devs = []
        pos_delta_u = pos_delta_uv[..., 0].unsqueeze(-1)  # [Us-2*step, Vs-2*step, 1]
        pos_delta_v = pos_delta_uv[..., 1].unsqueeze(-1)
        delta_u_neg = neg_delta_uv[..., 0].unsqueeze(-1)
        delta_v_neg = neg_delta_uv[..., 1].unsqueeze(-1)
        # ++
        actual_pp = xyz_grid[2 * step:, 2 * step:, :]
        planar_pp = xyz_center + pos_delta_u * Su_center + pos_delta_v * Sv_center
        dev_pp = actual_pp - planar_pp
        devs.append(dev_pp)

        # +-
        actual_pm = xyz_grid[2 * step:, :-2 * step, :]
        planar_pm = xyz_center + pos_delta_u * Su_center + delta_v_neg * Sv_center
        dev_pm = actual_pm - planar_pm
        devs.append(dev_pm)

        # -+
        actual_mp = xyz_grid[:-2 * step, 2 * step:, :]
        planar_mp = xyz_center + delta_u_neg * Su_center + pos_delta_v * Sv_center
        dev_mp = actual_mp - planar_mp
        devs.append(dev_mp)

        # --
        actual_mm = xyz_grid[:-2 * step, :-2 * step, :]
        planar_mm = xyz_center + delta_u_neg * Su_center + delta_v_neg * Sv_center
        dev_mm = actual_mm - planar_mm
        devs.append(dev_mm)

        # Aggregate deviations
        devs = torch.stack(devs, dim=0)  # [4, Us-2*step, Vs-2*step, 3]
        if dist_type == 'sq':
            loss = devs.pow(2).mean()
        elif dist_type == 'l2':
            loss = devs.norm(dim=-1).mean()
        elif dist_type == 'l1':
            loss = devs.abs().mean()
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        return loss

    @property
    def sampling_mode(self) -> 'SamplingMode':
        return self.state.sampling_mode


    def get_snapshot_intervals(self, uid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get activated snapshot intervals for a specific view."""
        # if self._interval_u is None:
        #     raise RuntimeError("Snapshot parameters not initialized")

        idx = self.uid_to_idx.get(uid, 0)
        u = self.uv_sampler._decode(self.uv_sampler._interval_u[idx])
        v = self.uv_sampler._decode(self.uv_sampler._interval_v[idx])
        return u, v



    def _compute_geometric_normals(self, xyz_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute normals from surface geometry using finite differences.
        More reliable than rotation-based normals for orientation checks.
        """
        Us, Vs, _ = xyz_grid.shape

        # Tangent in U direction
        du = torch.zeros_like(xyz_grid)
        du[:-1] = xyz_grid[1:] - xyz_grid[:-1]
        du[-1] = du[-2]  # Replicate boundary

        # Tangent in V direction
        dv = torch.zeros_like(xyz_grid)
        dv[:, :-1] = xyz_grid[:, 1:] - xyz_grid[:, :-1]
        dv[:, -1] = dv[:, -2]

        # Normal as cross product
        normals = torch.cross(du, dv, dim=-1)
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        return normals

    # =========================================================================
    # MAIN FORWARD PASS (Refactored)
    # =========================================================================

    def forward(self, viewpoint_cam, alpha: float = .9, beta: float = 0.9):
        """
        Main forward pass with proper cache management.

        Args:
            viewpoint_cam:  Camera object
            alpha: EMA factor for visibility updates
            beta:  Blend factor for UV warping
        """
        # uid = str(viewpoint_cam.uid)
        # self.uv_sampler.active_uid = uid
        self.uv_sampler.active_uid = str(viewpoint_cam.uid % self.uv_sampler.num_channels)  # Use first view for static tessellation

        if self.sampling_mode == SamplingMode.STATIC:
            self._forward_static()
        elif self.sampling_mode == SamplingMode.OPTIMIZABLE:
            self._forward_optimizable(viewpoint_cam)
        elif self.sampling_mode == SamplingMode.ADAPTIVE:
            self._forward_adaptive(viewpoint_cam, alpha, beta)
        elif self.sampling_mode == SamplingMode.EVALUATION:
            self.recompute()

        setattr(self, 'current_viewpoint_cam', viewpoint_cam)  # Store for ray_info access
    #
    def ray_info(self):
        self.ray = compute_ray_info(
            camera=getattr(self, 'current_viewpoint_cam', None),
            surface_points=self.get_xyz.reshape(-1, 3),
            scale=1.0,
            depth_min=0.0,
            border_margin=self.spatial_lr_scale * 2,
            shape=(self.state.Us, self.state.Vs)
        )

    def uv_depth(self):
        try:
            depths = self.ray.depths.reshape(self.state.Us, self.state.Vs, 1)
            final = torch.full(depths.shape, fill_value=torch.nan, device=depths.device)
            final = torch.where((depths > 0) & (~depths.isnan()) & (depths != torch.nan), depths, final)
            # final = torch.where((depths > 0) & (~depths.isnan()) & (depths < torch.inf), depths, final)
            return final
        except Exception as e:
            return torch.ones((self.state.Us, self.state.Vs, 1), device='cuda')

    def weights_map(self):
        uv_depth = self.uv_depth()
        if uv_depth is not None:
            return 1 / self.uv_depth().reshape(self.state.Us, self.state.Vs, 1)
        else:
            return torch.ones((self.state.Us, self.state.Vs, 1), device='cuda')

    def weights_diff(self):
        weight_maps = self.weights_map()
        w_u = weight_maps[1:, :] * weight_maps[:-1, :]
        w_v = weight_maps[:, 1:] * weight_maps[:, :-1]

        return torch.stack((w_u, w_v), dim=-1)


    def _forward_static(self):
        """Forward pass with static (uniform) UV sampling."""
        # Check if we should recompute UV this iteration
        should_recompute = self.should_recompute()
        if should_recompute:
            self.basis.recompute()

    def _forward_adaptive(self, camera, alpha: float, beta: float):
        """
        Forward pass with adaptive UV recomputation.

        WARMUP PROTOCOL:
        - For the first _warmup_iterations steps, use STATIC uniform
          intervals. This lets the surface geometry converge before we
          start adapting the sampling distribution.
        - After warmup, switch to Chhugani view-dependent tessellation.
        """
        # --- Phase 1: Warmup (static uniform sampling) ---
        if not self.state._warmup_complete:
            if self.iteration < self.state._warmup_iterations:
                self._forward_static()
                return

            self.state._warmup_complete = True
            self.invalidate_all_caches(force=True)
            self._forward_context = ForwardContext()
            # self._forward_context.warped_us.clear()
            # self._forward_context.warped_vs.clear()
            # self._forward_context.warped_uvs.clear()
            # self._forward_context.uv_basis.clear()

            self.basis.clear()
            print(f"[Surface {self.surf_uid}] Warmup complete at iter "
                  f"{self.iteration}. Switching to adaptive tessellation.")

        # --- Phase 2: Adaptive (Chhugani view-dependent) ---
        uid = camera.uid

        # Periodically clear caches to allow re-tessellation

        if self.iteration % self.state._ada_sampling_interval == 1:
            self.invalidate_all_caches(force=True)
            self._forward_context.warped_us.clear()
            self._forward_context.warped_vs.clear()
            self._forward_context.warped_uvs.clear()
            self._forward_context.uv_basis.clear()
            self.basis.clear()

        # Try to use cached intervals for this view
        try:
            buv = self._forward_context.uv_basis[uid]
            self.basis.replace_funcs(buv)
        except KeyError:

            nearest = None
            if hasattr(self, 'cameras') and self.cameras:
                rand_id = camera.nearest_id[
                    torch.randint(0, len(camera.nearest_id), (1,)).item()
                ]
                nearest = [self.cameras[rand_id]]

            self.update_uv_distribution_chhugani(
                camera, neighbor_cameras=nearest
            )

        self.invalidate_control_features()

    def _forward_optimizable(self, camera):
        """Forward pass with optimizable UV parameters."""
        self.recompute()#(self.uv_sampler.interval_u, self.uv_sampler.interval_v, self.knot_u(), self.knot_v())


    def get_real_normal(self, view_cam):
        normal_global = F.normalize(self.dSu, dim=-1).cross(F.normalize(self.dSv, dim=-1), dim=-1).view(-1, 3) #self.get_smallest_axis()

        # Compute view direction for each point
        xyz = self.get_xyz
        view_dirs = view_cam.camera_center - xyz  # Points toward camera

        # Flip normals that point away from camera
        dot_product = (normal_global * view_dirs).sum(dim=-1)
        flip_mask = dot_product < 0.0

        # Apply flip (creates new tensor, doesn't modify in place)
        oriented_normals = normal_global.clone()
        oriented_normals[flip_mask] = -oriented_normals[flip_mask]
        return oriented_normals

    # =========================================================================
    # PARAMETER UPDATE (Extended)
    # =========================================================================

    def update_parameters(self, iteration: int):
        """
        Update iteration-dependent parameters.
        Extended to handle sampling mode transitions.
        """
        self.iteration = iteration
        self.state.iteration = iteration
        # if self.iteration % 500 ==1:
        #     self.invalidate_all_caches(force=True)
        #     self.basis.clear()



    @torch.no_grad()
    def compute_span_scores(self, metric_type='residual'):
        """
        Computes subdivision scores per knot span using intrinsic surface metrics
        (e.g., Eikonal/Area distortion or Fitting Residuals).
        Handles separable UV sampling (1D intervals).

        Returns:
            scores_u: (Num_U_Spans,) Score for splitting U (based on worst V-strip error)
            scores_v: (Num_V_Spans,) Score for splitting V (based on worst U-strip error)
            unique_u: Knot vector boundaries for U
            unique_v: Knot vector boundaries for V
        """
        # 1. Compute Dense Metric Map (Us, Vs)
        # ---------------------------------------------------------------------
        if metric_type == 'spatial':
            # Spatial Metric: Magnitude of the normal vector (proxy for surface distortion)
            metric_map = None
            for cam in self.cameras:
                ray = compute_ray_info(
                    camera=cam,
                    surface_points=self.get_xyz.reshape(-1, 3).detach(),
                    scale=1.0,
                    depth_min=0.0,
                    border_margin=self.spatial_lr_scale * 2,
                    shape=(self.state.Us, self.state.Vs)
                )
                geodesic_depth = ray.depths.reshape(self.state.Us, self.state.Vs)
                geodesic_distortion = 1/(geodesic_depth + 1e-6)
                # Higher depth = more distortion/need for refinement

                residual_u_append = geodesic_distortion[-1:] - geodesic_distortion[-2:-1]
                geodesic_distortion_du = torch.diff(geodesic_distortion, dim=0, append=residual_u_append)
                residual_v_append = geodesic_distortion[:, -1:] - geodesic_distortion[:, -2:-1]
                geodesic_distortion_dv = torch.diff(geodesic_distortion, dim=1, append=residual_v_append)
                    # Compute normal magnitude as a proxy for distortion
                if metric_map is None:
                    # metric_map = torch.sqrt(geodesic_depth_du ** 2 + geodesic_depth_dv ** 2)
                    metric_map = torch.max(geodesic_distortion_du.abs(), geodesic_distortion_dv.abs())
                else:
                    # metric_map += torch.sqrt(geodesic_depth_du ** 2 + geodesic_depth_dv ** 2)
                    metric_map += torch.max(geodesic_distortion_du.abs(), geodesic_distortion_dv.abs())
            metric_map = metric_map / len(self.cameras)  # Average over views if multiple

        elif metric_type == 'eikonal':
            # Eikonal Term: Distortion of the area element (deviation from isometry)
            # metric = | ||Su x Sv|| - 1 |
            Su = self.dSu.detach()  # (Us, Vs, 3)
            Sv = self.dSv.detach()  # (Us, Vs, 3)
            normals = torch.cross(Su, Sv, dim=-1)
            area_element = torch.linalg.norm(normals, dim=-1)
            # We target unit speed/area. High deviation = high stress on the basis.
            metric_map = (area_element - 1.0).abs()
        elif metric_type == 'residual':
            # Fitting Error: Magnitude of the positional gradient (proxy for residual)
            grad_map = self.state.get_grid_grads().reshape(self.state.Us, self.state.Vs, -1)
            metric_map = grad_map.norm(dim=-1)
        elif metric_type == 'hybrid':
            geodesic_distortion = self.uv_depth().reshape(self.state.Us, self.state.Vs)
            residual_u_append = geodesic_distortion[-1:] - geodesic_distortion[-2:-1]
            geodesic_distortion_du = torch.diff(geodesic_distortion, dim=0, append=residual_u_append)
            residual_v_append = geodesic_distortion[:, -1:] - geodesic_distortion[:, -2:-1]
            geodesic_distortion_dv = torch.diff(geodesic_distortion, dim=1, append=residual_v_append)
                # Compute normal magnitude as a proxy for distortion
            distortion = torch.sqrt(geodesic_distortion_du ** 2 + geodesic_distortion_dv ** 2).reshape(self.state.Us, self.state.Vs)


            grad_map = self.state.get_grid_grads().reshape(self.state.Us, self.state.Vs, -1)
            residual = grad_map.norm(dim=-1)

            # Weighted combination
            metric_map = 0.99 * residual + 0.01 * distortion
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        # 2. Map Samples to Knot Spans
        # ---------------------------------------------------------------------
        # Get active knot vectors (unique values define spans)
        knots_u = self.knot_u.forward().detach()
        knots_v = self.knot_v.forward().detach()

        unique_u = torch.unique_consecutive(knots_u)
        unique_v = torch.unique_consecutive(knots_v)

        # Use separable intervals as provided by the user
        samples_u = self.uv_sampler.interval_u.detach().view(-1)
        samples_v = self.uv_sampler.interval_v.detach().view(-1)

        # Bucketize samples into spans
        # indices give the right-boundary index, so we subtract 1 to get span index
        u_span_idx = torch.bucketize(samples_u, unique_u, right=True) - 1
        v_span_idx = torch.bucketize(samples_v, unique_v, right=True) - 1

        num_spans_u = len(unique_u) - 1
        num_spans_v = len(unique_v) - 1

        u_span_idx = u_span_idx.clamp(0, num_spans_u - 1)
        v_span_idx = v_span_idx.clamp(0, num_spans_v - 1)

        # 3. Aggregate Metric per Span (Max-Pooling)
        # ---------------------------------------------------------------------
        # We want to find the worst error within each physical patch [u_i, u_{i+1}] x [v_j, v_{j+1}]

        # Create a dense tensor for span errors
        span_errors = torch.zeros((num_spans_u, num_spans_v), device=self.device)

        # BROADCASTING FIX: Create grid indices from 1D span indices
        # u_span_idx is (Us,), v_span_idx is (Vs,)
        # We need to map every point (i, j) in the (Us, Vs) metric map to a span
        u_grid = u_span_idx.unsqueeze(1)  # (Us, 1)
        v_grid = v_span_idx.unsqueeze(0)  # (1, Vs)

        # Calculate flattened index into the (NumSpansU, NumSpansV) error grid
        # Resulting grid is (Us, Vs)
        flat_span_idx_grid = u_grid * num_spans_v + v_grid

        # Flatten everything for scatter operation
        flat_span_idx = flat_span_idx_grid.view(-1)
        flat_metric = metric_map.view(-1)

        # Verify shapes match (crucial check)
        if flat_span_idx.numel() != flat_metric.numel():
            # Warning fallback if dimensions are out of sync
            print(f"Warning: Metric shape {flat_metric.shape} mismatch with indices {flat_span_idx.shape}")
            return None, None, unique_u, unique_v

        # Aggregate using max-pooling (scatter_reduce_)
        try:
            # 'amax' reduction finds the maximum metric value falling into each span bucket
            span_errors.reshape(-1).scatter_reduce_(0, flat_span_idx, flat_metric, reduce='amax', include_self=False)
        except AttributeError:
            # Fallback for older PyTorch versions using average instead of max
            count_grid = torch.zeros_like(span_errors)
            span_errors.reshape(-1).index_add_(0, flat_span_idx, flat_metric)
            count_grid.reshape(-1).index_add_(0, flat_span_idx, torch.ones_like(flat_metric))
            span_errors = span_errors / (count_grid + 1e-8)

        # 4. Directional Scoring (The "Orthogonal" Logic)
        # ---------------------------------------------------------------------
        # Score for splitting U-span i = Max error across all V-spans in that row
        scores_u = span_errors.max(dim=1).values  # (NumSpansU,)
        scores_v = span_errors.max(dim=0).values  # (NumSpansV,)

        return scores_u, scores_v, unique_u, unique_v

    def get_subdivision_candidates(self, use_partitioning: bool = False, num_partitions: int = -1):
        """
        Calculates candidates using Intrinsic Geometric Metrics (Eikonal/Residual).

        Args:
            use_partitioning: If True, divide grid into k partitions and select one candidate per partition.
                             If None, uses self.state.opt.use_spatial_partitioning
            num_partitions: Number of partitions (default: min(H, W) // 2)

        Returns:
            List of subdivision candidates with spatial diversity
        """
        # 1. Compute Geometric Scores
        scores_u, scores_v, unique_u, unique_v = self.compute_span_scores(
            metric_type=self.state.opt.subdiv_critertia
        )

        if scores_u is None:
            return []

        # 2. Anisotropy Weighting (Rich-get-richer prevention)
        curr_H, curr_W = self.state._H, self.state._W
        eps = 1e-6
        weight_u = (curr_W / (curr_H + eps)) ** 0.5
        weight_v = (curr_H / (curr_W + eps)) ** 0.5


        if not use_partitioning:
            # Original behavior: all candidates above threshold
            return self._get_all_subdivision_candidates(
                scores_u, scores_v, unique_u, unique_v, weight_u, weight_v
            )

        print(f"Using partitioned candidate selection with {num_partitions} partitions.")
        # Partitioned approach
        num_partitions = num_partitions if num_partitions is not None else \
            getattr(self.state.opt, 'num_partitions', min(curr_H, curr_W) // 2)

        return self._get_partitioned_subdivision_candidates(
            scores_u, scores_v, unique_u, unique_v, weight_u, weight_v, num_partitions
        )

    def _get_all_subdivision_candidates(
            self,
            scores_u, scores_v, unique_u, unique_v,
            weight_u, weight_v
    ):
        """Original behavior: return all candidates above threshold."""
        candidates = []

        # U Candidates
        for i in range(len(scores_u)):
            score = scores_u[i].item() * weight_u
            if score > self.state.opt.densify_grad_threshold and score > self.state.opt.densify_abs_grad_threshold:
                val = (unique_u[i] + unique_u[i + 1]) / 2.0
                candidates.append({
                    'score': score,
                    'val': val.item(),
                    'type': 'u',
                    'index': i,
                    'partition': None  # No partition info
                })

        # V Candidates
        for i in range(len(scores_v)):
            score = scores_v[i].item() * weight_v
            if score > self.state.opt.densify_grad_threshold and score > self.state.opt.densify_abs_grad_threshold:
                val = (unique_v[i] + unique_v[i + 1]) / 2.0
                candidates.append({
                    'score': score,
                    'val': val.item(),
                    'type': 'v',
                    'index': i,
                    'partition': None
                })

        return candidates

    def _get_partitioned_subdivision_candidates(
            self,
            scores_u, scores_v, unique_u, unique_v,
            weight_u, weight_v,
            num_partitions: int
    ):
        """
        Spatially-aware candidate selection using k-partitions.

        Strategy:
        - Divide the grid into k non-overlapping regions
        - Select the best candidate from each partition
        - Ensures spatial diversity across the surface
        """
        candidates = []

        # ========== U-Direction Partitioning ==========
        if len(scores_u) > 0:
            # Create partitions along U
            partition_size_u = max(1, len(scores_u) // num_partitions)

            for p in range(num_partitions):
                start_idx = p * partition_size_u
                end_idx = min((p + 1) * partition_size_u, len(scores_u))

                if start_idx >= len(scores_u):
                    break

                # Find best candidate in this partition
                partition_scores = scores_u[start_idx:end_idx] * weight_u

                if partition_scores.numel() == 0:
                    continue

                # Get local maximum
                local_max_score, local_max_idx = partition_scores.max(dim=0)
                global_idx = start_idx + local_max_idx.item()

                # Only add if above threshold
                if local_max_score.item() > self.state.opt.densify_grad_threshold:
                    val = (unique_u[global_idx] + unique_u[global_idx + 1]) / 2.0
                    candidates.append({
                        'score': local_max_score.item(),
                        'val': val.item(),
                        'type': 'u',
                        'index': global_idx,
                        'partition': p,
                        'partition_type': 'u'
                    })

        # ========== V-Direction Partitioning ==========
        if len(scores_v) > 0:
            partition_size_v = max(1, len(scores_v) // num_partitions)

            for p in range(num_partitions):
                start_idx = p * partition_size_v
                end_idx = min((p + 1) * partition_size_v, len(scores_v))

                if start_idx >= len(scores_v):
                    break

                partition_scores = scores_v[start_idx:end_idx] * weight_v

                if partition_scores.numel() == 0:
                    continue

                local_max_score, local_max_idx = partition_scores.max(dim=0)
                global_idx = start_idx + local_max_idx.item()

                if local_max_score.item() > self.state.opt.densify_grad_threshold:
                    val = (unique_v[global_idx] + unique_v[global_idx + 1]) / 2.0
                    candidates.append({
                        'score': local_max_score.item(),
                        'val': val.item(),
                        'type': 'v',
                        'index': global_idx,
                        'partition': p,
                        'partition_type': 'v'
                    })

        return candidates


    def _compute_pruning_metrics(self, H, W, Us, Vs, device):
        """Extract and compute all pruning metrics."""
        # Opacity (control point level)
        opacity_ctrl = self.opacity.features

        # Radii
        if hasattr(self.state, 'max_radii2D') and self.state.max_radii2D.numel() > 0:
            radii_sampling = self.state.max_radii2D.reshape(Us, Vs)
            radii_ctrl = F.adaptive_avg_pool2d(
                radii_sampling.unsqueeze(0).unsqueeze(0), (H, W)
            ).squeeze()
        else:
            radii_sampling = torch.zeros(H, W, device=device)
        max_scale_ctrl, visibility_ctrl, grad_ctrl = torch.zeros((3, H, W), device=device).unbind(0) # Placeholder tensors
        return opacity_ctrl, max_scale_ctrl, grad_ctrl, visibility_ctrl, radii_ctrl


    def _get_partitioned_pruning_candidates(
            self, H, W, degree,
            opacity_ctrl, max_scale_ctrl, visibility_ctrl, grad_ctrl, radii_ctrl,
            min_opacity, max_screen_size, max_world_scale_factor,
            min_visibility_ratio, extent, num_partitions
    ):
        """
        Partitioned pruning using 3DGS-compliant criteria.
        Selects highest-scoring candidate per partition for spatial diversity.
        """
        candidates = []
        min_size = degree + 2

        total_vis = visibility_ctrl.sum().item()
        total_grad = grad_ctrl.sum().item()

        # ========== U-Direction ==========
        if H > min_size:
            valid_rows = torch.arange(degree, H - degree, device=opacity_ctrl.device)
            num_valid = len(valid_rows)

            if num_valid > 0:
                all_row_opacity = opacity_ctrl[valid_rows, :].mean(dim=1)
                all_row_scale = max_scale_ctrl[valid_rows, :].max(dim=1).values
                all_row_visibility = visibility_ctrl[valid_rows, :].mean(dim=1)
                all_row_gradient = grad_ctrl[valid_rows, :].mean(dim=1)
                all_row_radii = radii_ctrl[valid_rows, :].max(dim=1).values

                all_removal_scores, _ = self._compute_removal_scores_vectorized(
                    all_row_opacity, all_row_scale, all_row_visibility, all_row_gradient, all_row_radii,
                    W, min_opacity, max_screen_size, max_world_scale_factor,
                    min_visibility_ratio, extent, total_vis, total_grad
                )

                partition_size = max(1, num_valid // num_partitions)
                partition_indices = torch.arange(num_valid, device=opacity_ctrl.device) // partition_size
                partition_indices = partition_indices.clamp(max=num_partitions - 1)

                best_scores_per_partition = torch.full((num_partitions,), -1.0, device=opacity_ctrl.device)
                best_indices_per_partition = torch.full((num_partitions,), -1, dtype=torch.long,
                                                        device=opacity_ctrl.device)

                for p in range(num_partitions):
                    partition_mask = partition_indices == p
                    if partition_mask.any():
                        partition_scores = all_removal_scores[partition_mask]
                        partition_local_indices = torch.where(partition_mask)[0]
                        if partition_scores.numel() > 0:
                            best_local_idx = partition_scores.argmax()
                            best_scores_per_partition[p] = partition_scores[best_local_idx]
                            best_indices_per_partition[p] = partition_local_indices[best_local_idx]

                knots_u = self.knot_u.forward()
                valid_partition_mask = (best_scores_per_partition > 0) & (best_indices_per_partition >= 0)
                valid_partitions = torch.where(valid_partition_mask)[0]

                for p in valid_partitions.tolist():
                    local_idx = best_indices_per_partition[p].item()
                    i = valid_rows[local_idx].item()
                    score = best_scores_per_partition[p].item()
                    knot_val = knots_u[i]

                    reasons = []
                    if all_row_opacity[local_idx].item() < min_opacity:
                        reasons.append(f"transparent(opacity={all_row_opacity[local_idx].item():.4f})")
                    if max_screen_size is not None and all_row_radii[local_idx].item() > max_screen_size:
                        reasons.append(f"too_large(radii={all_row_radii[local_idx].item():.1f})")

                    candidates.append({
                        'score': score,
                        'type': 'u',
                        'index': i,
                        'val': knot_val,
                        'reasons': reasons,
                        'metrics': {
                            'opacity': all_row_opacity[local_idx].item(),
                            'scale': all_row_scale[local_idx].item(),
                            'visibility': all_row_visibility[local_idx].item(),
                            'gradient': all_row_gradient[local_idx].item(),
                            'radii': all_row_radii[local_idx].item()
                        },
                        'partition': p,
                        'partition_type': 'u'
                    })

        # ========== V-Direction ==========
        if W > min_size:
            valid_cols = torch.arange(degree, W - degree, device=opacity_ctrl.device)
            num_valid = len(valid_cols)

            if num_valid > 0:
                all_col_opacity = opacity_ctrl[:, valid_cols].mean(dim=0)
                all_col_scale = max_scale_ctrl[:, valid_cols].max(dim=0).values
                all_col_visibility = visibility_ctrl[:, valid_cols].mean(dim=0)
                all_col_gradient = grad_ctrl[:, valid_cols].mean(dim=0)
                all_col_radii = radii_ctrl[:, valid_cols].max(dim=0).values

                all_removal_scores, _ = self._compute_removal_scores_vectorized(
                    all_col_opacity, all_col_scale, all_col_visibility, all_col_gradient, all_col_radii,
                    H, min_opacity, max_screen_size, max_world_scale_factor,
                    min_visibility_ratio, extent, total_vis, total_grad
                )

                partition_size = max(1, num_valid // num_partitions)
                partition_indices = torch.arange(num_valid, device=opacity_ctrl.device) // partition_size
                partition_indices = partition_indices.clamp(max=num_partitions - 1)

                best_scores_per_partition = torch.full((num_partitions,), -1.0, device=opacity_ctrl.device)
                best_indices_per_partition = torch.full((num_partitions,), -1, dtype=torch.long,
                                                        device=opacity_ctrl.device)

                for p in range(num_partitions):
                    partition_mask = partition_indices == p
                    if partition_mask.any():
                        partition_scores = all_removal_scores[partition_mask]
                        partition_local_indices = torch.where(partition_mask)[0]
                        if partition_scores.numel() > 0:
                            best_local_idx = partition_scores.argmax()
                            best_scores_per_partition[p] = partition_scores[best_local_idx]
                            best_indices_per_partition[p] = partition_local_indices[best_local_idx]

                knots_v = self.knot_v.forward()
                valid_partition_mask = (best_scores_per_partition > 0) & (best_indices_per_partition >= 0)
                valid_partitions = torch.where(valid_partition_mask)[0]

                for p in valid_partitions.tolist():
                    local_idx = best_indices_per_partition[p].item()
                    j = valid_cols[local_idx].item()
                    score = best_scores_per_partition[p].item()
                    knot_val = knots_v[j]

                    reasons = []
                    if all_col_opacity[local_idx].item() < min_opacity:
                        reasons.append(f"transparent(opacity={all_col_opacity[local_idx].item():.4f})")
                    if max_screen_size is not None and all_col_radii[local_idx].item() > max_screen_size:
                        reasons.append(f"too_large(radii={all_col_radii[local_idx].item():.1f})")

                    candidates.append({
                        'score': score,
                        'type': 'v',
                        'index': j,
                        'val': knot_val,
                        'reasons': reasons,
                        'metrics': {
                            'opacity': all_col_opacity[local_idx].item(),
                            'scale': all_col_scale[local_idx].item(),
                            'visibility': all_col_visibility[local_idx].item(),
                            'gradient': all_col_gradient[local_idx].item(),
                            'radii': all_col_radii[local_idx].item()
                        },
                        'partition': p,
                        'partition_type': 'v'
                    })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def get_pruning_candidates(
            self,
            min_opacity: float = 0.005,
            max_screen_size: float = 20.0,
            max_world_scale_factor: float = 0.1,
            min_visibility_ratio: float = 0.1,
            extent: Optional[float] = None,
            use_partitioning: bool = None,
            num_partitions: int = None
    ) -> List[Dict]:
        """
        Get candidates for knot removal (pruning) using 3DGS-compliant criteria.

        3DGS prunes Gaussians that are:
          1. Transparent (opacity < min_opacity)
          2. Too large in screen-space (max_radii2D > max_screen_size)

        For NURBS surfaces, we evaluate these criteria per row/column.

        Args:
            min_opacity: Opacity threshold below which to consider removal
            max_screen_size: Maximum allowed screen-space radius
            max_world_scale_factor: Unused (kept for API compat)
            min_visibility_ratio: Unused (kept for API compat)
            extent: Scene extent (unused in 3DGS-style pruning)
            use_partitioning: If True, use k-partition strategy
            num_partitions: Number of partitions

        Returns:
            List of removal candidates with scores and metadata
        """
        H, W = self.state._H, self.state._W
        Us, Vs = self.state.Us, self.state.Vs
        device = self.device
        degree = self.state.degree

        min_size = degree + 2
        if H <= min_size and W <= min_size:
            return []

        # Compute per-sampling-point metrics
        opacity_ctrl, max_scale_ctrl, grad_ctrl, visibility_ctrl, radii_ctrl = \
            self._compute_pruning_metrics(H, W, Us, Vs, device)

        # Determine if using partitioning
        use_partitioning = use_partitioning if use_partitioning is not None else \
            getattr(self.state.opt, 'use_spatial_partitioning_prune', False)

        if not use_partitioning:
            return self._get_all_pruning_candidates(
                H, W, degree, opacity_ctrl, max_scale_ctrl, visibility_ctrl,
                grad_ctrl, radii_ctrl, min_opacity, max_screen_size,
                max_world_scale_factor, min_visibility_ratio, extent
            )

        num_partitions = num_partitions if num_partitions is not None else \
            getattr(self.state.opt, 'num_partitions_prune', min(H, W) // 2)

        return self._get_partitioned_pruning_candidates(
            H, W, degree, opacity_ctrl, max_scale_ctrl, visibility_ctrl,
            grad_ctrl, radii_ctrl, min_opacity, max_screen_size,
            max_world_scale_factor, min_visibility_ratio, extent, num_partitions
        )

    def _get_all_pruning_candidates(
            self, H, W, degree,
            opacity_ctrl, max_scale_ctrl, visibility_ctrl, grad_ctrl, radii_ctrl,
            min_opacity, max_screen_size, max_world_scale_factor,
            min_visibility_ratio, extent
    ):
        """
        3DGS-compliant pruning: check all rows/columns against transparency
        and screen-space size criteria only.
        """
        candidates = []
        min_size = degree + 2

        # Unused in 3DGS-style pruning but kept for compatibility
        total_vis = visibility_ctrl.sum().item()
        total_grad = grad_ctrl.sum().item()

        # U direction (rows)
        if H > min_size:
            valid_rows = torch.arange(degree, H - degree, device=opacity_ctrl.device)

            if len(valid_rows) > 0:
                row_opacity = opacity_ctrl[valid_rows, :].mean(dim=1)
                row_scale = max_scale_ctrl[valid_rows, :].max(dim=1).values
                row_visibility = visibility_ctrl[valid_rows, :].mean(dim=1)
                row_gradient = grad_ctrl[valid_rows, :].mean(dim=1)
                row_radii = radii_ctrl[valid_rows, :].max(dim=1).values

                removal_scores, _ = self._compute_removal_scores_vectorized(
                    row_opacity, row_scale, row_visibility, row_gradient, row_radii,
                    W, min_opacity, max_screen_size, max_world_scale_factor,
                    min_visibility_ratio, extent, total_vis, total_grad
                )

                positive_mask = removal_scores > 0
                positive_indices = valid_rows[positive_mask]
                positive_scores = removal_scores[positive_mask]

                if positive_indices.numel() > 0:
                    knots_u = self.knot_u.forward()

                    for idx, (i, score) in enumerate(zip(positive_indices.tolist(), positive_scores.tolist())):
                        knot_val = knots_u[i].item()
                        # Determine reason(s) for this candidate
                        reasons = []
                        if row_opacity[valid_rows == i].item() < min_opacity:
                            reasons.append(
                                f"transparent(opacity={row_opacity[valid_rows == i].item():.4f}<{min_opacity})")
                        if max_screen_size is not None and row_radii[valid_rows == i].item() > max_screen_size:
                            reasons.append(
                                f"too_large(radii={row_radii[valid_rows == i].item():.1f}>{max_screen_size})")

                        candidates.append({
                            'score': score,
                            'type': 'u',
                            'index': i,
                            'val': knot_val,
                            'reasons': reasons,
                            'metrics': {
                                'opacity': row_opacity[valid_rows == i].item(),
                                'scale': row_scale[valid_rows == i].item(),
                                'visibility': row_visibility[valid_rows == i].item(),
                                'gradient': row_gradient[valid_rows == i].item(),
                                'radii': row_radii[valid_rows == i].item()
                            },
                            'partition': None
                        })

        # V direction (columns)
        if W > min_size:
            valid_cols = torch.arange(degree, W - degree, device=opacity_ctrl.device)

            if len(valid_cols) > 0:
                col_opacity = opacity_ctrl[:, valid_cols].mean(dim=0)
                col_scale = max_scale_ctrl[:, valid_cols].max(dim=0).values
                col_visibility = visibility_ctrl[:, valid_cols].mean(dim=0)
                col_gradient = grad_ctrl[:, valid_cols].mean(dim=0)
                col_radii = radii_ctrl[:, valid_cols].max(dim=0).values

                removal_scores, _ = self._compute_removal_scores_vectorized(
                    col_opacity, col_scale, col_visibility, col_gradient, col_radii,
                    H, min_opacity, max_screen_size, max_world_scale_factor,
                    min_visibility_ratio, extent, total_vis, total_grad
                )

                positive_mask = removal_scores > 0
                positive_indices = valid_cols[positive_mask]
                positive_scores = removal_scores[positive_mask]

                if positive_indices.numel() > 0:
                    knots_v = self.knot_v.forward()

                    for idx, (j, score) in enumerate(zip(positive_indices.tolist(), positive_scores.tolist())):
                        knot_val = knots_v[j].item()
                        reasons = []
                        if col_opacity[valid_cols == j].item() < min_opacity:
                            reasons.append(
                                f"transparent(opacity={col_opacity[valid_cols == j].item():.4f}<{min_opacity})")
                        if max_screen_size is not None and col_radii[valid_cols == j].item() > max_screen_size:
                            reasons.append(
                                f"too_large(radii={col_radii[valid_cols == j].item():.1f}>{max_screen_size})")

                        candidates.append({
                            'score': score,
                            'type': 'v',
                            'index': j,
                            'val': knot_val,
                            'reasons': reasons,
                            'metrics': {
                                'opacity': col_opacity[valid_cols == j].item(),
                                'scale': col_scale[valid_cols == j].item(),
                                'visibility': col_visibility[valid_cols == j].item(),
                                'gradient': col_gradient[valid_cols == j].item(),
                                'radii': col_radii[valid_cols == j].item()
                            },
                            'partition': None
                        })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def _compute_removal_scores_vectorized(
            self,
            opacity: torch.Tensor,  # [N] mean opacity along row/col
            scale: torch.Tensor,  # [N] max scale along row/col
            visibility: torch.Tensor,  # [N] mean visibility count (unused in 3DGS-style)
            gradient: torch.Tensor,  # [N] mean gradient magnitude (unused in 3DGS-style)
            radii: torch.Tensor,  # [N] max screen-space radii
            other_dim: int,  # Size of orthogonal dimension
            min_opacity: float,
            max_screen_size: float,
            max_world_scale_factor: float,
            min_visibility_ratio: float,
            extent: float,
            total_vis: float,
            total_grad: float,
            curvature: torch.Tensor = None,
            neighbor_similarity: torch.Tensor = None,
    ):
        """
        3DGS-compliant removal score computation.

        Original 3DGS prunes Gaussians that are:
          1. Transparent (opacity < min_opacity)
          2. Too large in screen-space (max_radii2D > max_screen_size)

        For NURBS, we apply these criteria to rows/columns:
          - A row/col is a pruning candidate if its mean opacity is below
            min_opacity OR its max screen-space radius exceeds max_screen_size.
          - The score reflects how strongly the criterion is violated
            (for ranking when multiple candidates exist), but any candidate
            with score > 0 is eligible for removal.

        Returns:
            scores: [N] tensor of removal scores (> 0 means eligible)
            reasons: None (computed on-demand for selected candidates)
        """
        N = opacity.shape[0]
        device = opacity.device

        scores = torch.zeros(N, device=device)

        # ===================================================================
        # CRITERION 1: Transparent (opacity < min_opacity)
        #   Original 3DGS:  prune_mask = (opacities < min_opacity).squeeze()
        #   NURBS adaptation: mean opacity of row/col < min_opacity
        # ===================================================================
        transparent_mask = opacity.squeeze() < min_opacity
        # Score proportional to how far below threshold (for ranking)
        # Debug prints for shapes and values
        scores[transparent_mask] += (min_opacity - opacity.squeeze()[transparent_mask]) / (min_opacity + 1e-8)

        # ===================================================================
        # CRITERION 2: Too large in screen-space (radii > max_screen_size)
        #   Original 3DGS:  big_points_ss = max_radii2D > max_screen_size
        #   NURBS adaptation: max radii in row/col > max_screen_size
        # ===================================================================
        if max_screen_size is not None and max_screen_size > 0:
            too_large_mask = radii > max_screen_size
            # Score proportional to how much it exceeds the threshold
            scores[too_large_mask] += (radii[too_large_mask] - max_screen_size) / (max_screen_size + 1e-8)

        return scores, None

    def depth_discontinuities(self):
        # Spatial Metric: Magnitude of the normal vector (proxy for surface distortion)
        geodesic_depth = self.uv_depth().reshape(self.state.Us, self.state.Vs)
        # Higher depth = more distortion/need for refinement

        residual_u_append = geodesic_depth[-1:] - geodesic_depth[-2:-1]
        geodesic_depth_du = torch.diff(geodesic_depth, dim=0, append=residual_u_append)
        residual_v_append = geodesic_depth[:, -1:] - geodesic_depth[:, -2:-1]
        geodesic_depth_dv = torch.diff(geodesic_depth, dim=1, append=residual_v_append)
        # deviation = torch.stack([geodesic_depth_du.abs(), geodesic_depth_dv.abs()], dim=-1).reshape(self.state.Us, self.state.Vs).norm(dim=-1)
        # deviation = (deviation - deviation.min()) / (deviation.max() - deviation.min() + 1e-8)
        return geodesic_depth_du, geodesic_depth_dv

    def apply_subdivision(self, cands=None, optimizer=None, cand=None):
        """
        Executes a batch of subdivisions based on the candidates.
        Maintains the invariant:  Us == int(H * D), Vs == int(W * D)
        """
        # Backward compatibility for 'cand' single-item kwargs
        if cands is None and cand is not None:
            cands = cand

        if not cands:
            return

        if not isinstance(cands, list):
            cands = [cands]

        optimizer = optimizer if optimizer is not None else self.optimizer

        # Separate U and V candidates to avoid indexing collision issues
        # and sort them strictly in reverse order by 'val' so earlier insertions
        # do not affect later indices.
        u_cands = sorted([c for c in cands if c['type'] == 'u'], key=lambda x: x['val'], reverse=True)
        v_cands = sorted([c for c in cands if c['type'] == 'v'], key=lambda x: x['val'], reverse=True)

        self._last_subdivision_step = self.iteration

        for cand_item in u_cands + v_cands:
            val = cand_item['val']
            is_u_split = (cand_item['type'] == 'u')

            # Current state
            curr_H, curr_W = self.state._H, self.state._W
            density_u = self.state.sampling_density
            density_v = self.state.sampling_density

            # Compute how many interval samples to insert
            if is_u_split:
                num_interval_insertions = int((curr_H + 1) * density_u) - int(curr_H * density_u)
                direction = 'u'
                curr_knots = self.knot_u.forward()
                curr_dim_size = curr_H
            else:
                num_interval_insertions = int((curr_W + 1) * density_v) - int(curr_W * density_v)
                direction = 'v'
                curr_knots = self.knot_v.forward()
                curr_dim_size = curr_W

            # A. Calculate Insertion Index for control points
            k = torch.searchsorted(curr_knots, val, side='right').item() - 2
            max_k = len(curr_knots) - self.state.degree - 2
            k = max(self.state.degree, min(k, max_k))
            insert_idx = max(self.state.degree, min(k - self.state.degree + 1, curr_dim_size - (self.state.degree + 1)))

            # B. Compute New Control Point Tensors
            tensors_dict = {}
            for module in self.control_list:
                if module.control_features is None:
                    continue
                new_grid, insert_idx = module.compute_inserted_grid(
                    direction=direction,
                    knots=curr_knots,
                    degree=self.state.degree,
                    val=val,
                    insert_idx=insert_idx,
                    insertion_fn=insert_knot_u,
                    use_blend=False
                )

                tensors_dict[module.name] = (new_grid, insert_idx)

            # C. Update Optimizer (Adam States)
            opt_tensors = self.insert_tensors_to_optimizer(
                tensors_dict, direction=direction, degree=self.state.degree, u_bar=val, insert_idx=insert_idx,
                optimizer=optimizer,
                momentum_strategy=self.state.optimizer_blend_strategy
            )

            # D. Apply to Modules
            for module in self.control_list:
                if module.name in opt_tensors:
                    module.control_features = opt_tensors[module.name]

            self.position.set_weights(self.weights)
            dummy_shape = curr_H if is_u_split else curr_W
            _, new_knots_full = insert_knot_u(
                torch.zeros(dummy_shape, 1, 1, device=self.device),
                curr_knots, self.state.degree, val
            )

            if self.sampling_mode != SamplingMode.ADAPTIVE or hasattr(self.uv_sampler, '_interval_u_global'):
                self.uv_sampler.subdivide(
                    direction=direction,
                    val=val,
                    optimizer=optimizer)

            if is_u_split:
                internal_knots = new_knots_full[self.state.degree + 1: -(self.state.degree + 1)]
                self.knot_u.update_knot_vector(self, torch.sort(internal_knots)[0], u_bar=val, optimizer=optimizer)
                self.state._H += 1
                self.state.base_u += 1
            else:
                internal_knots = new_knots_full[self.state.degree + 1: -(self.state.degree + 1)]
                self.knot_v.update_knot_vector(self, torch.sort(internal_knots)[0], u_bar=val, optimizer=optimizer)
                self.state._W += 1
                self.state.base_v += 1

        # DONE WITH ALL CANDIDATES. NOW FINALIZE.
        torch.cuda.empty_cache()
        self.basis.uv_sampler = self.uv_sampler
        self.basis.knot_u = self.knot_u
        self.basis.knot_v = self.knot_v

        if hasattr(self, '_chhugani_tessellator'):
            self._chhugani_tessellator.reset()
        if hasattr(self, '_forward_context'):
            from modules.tessellation.chhugani import ForwardContext
            self._forward_context = ForwardContext()
        if hasattr(self, '_chhugani_params'):
            del self._chhugani_params

        self.invalidate_all_caches(force=True)
    def subdivide_surface(self, cand, optimizer=None):
        """
        Executes a single subdivision based on the candidate.
        Maintains the invariant:  Us == int(H * D), Vs == int(W * D)
        """
        optimizer = optimizer if optimizer is not None else self.optimizer

        val = cand['val']
        is_u_split = (cand['type'] == 'u')

        # Current state
        # curr_Us, curr_Vs = self.state.Us, self.state.Vs
        curr_H, curr_W = self.state._H, self.state._W
        density_u = self.state.sampling_density
        density_v = self.state.sampling_density
        self._last_subdivision_step = self.iteration

        # Compute how many interval samples to insert
        if is_u_split:
            num_interval_insertions = int((curr_H + 1) * density_u) - int(curr_H * density_u)
            direction = 'u'
            curr_knots = self.knot_u.forward()
            curr_dim_size = curr_H
        else:
            num_interval_insertions = int((curr_W + 1) * density_v) - int(curr_W * density_v)
            direction = 'v'
            curr_knots = self.knot_v.forward()
            curr_dim_size = curr_W

        # A.  Calculate Insertion Index for control points
        k = torch.searchsorted(curr_knots, val, side='right').item() - 2
        max_k = len(curr_knots) - self.state.degree - 2
        k = max(self.state.degree, min(k, max_k))
        insert_idx = max(self.state.degree, min(k - self.state.degree + 1, curr_dim_size - (self.state.degree + 1)))

        # B. Compute New Control Point Tensors
        tensors_dict = {}
        for module in self.control_list:
            if module.control_features is None:
                continue
            # use_blend = module.name == f'dc_{self.surf_uid}' or module.name == f'rest_{self.surf_uid}'
            new_grid, insert_idx = module.compute_inserted_grid(
                direction=direction,
                knots=curr_knots,
                degree=self.state.degree,
                val=val,
                insert_idx=insert_idx,
                insertion_fn = insert_knot_u, #if module.name in [f'xyz_{self.surf_uid}', f'scaling_{self.surf_uid}'] else insert_knot_u_midpoint,
                use_blend=False
            )

            tensors_dict[module.name] = (new_grid, insert_idx)

        # C. Update Optimizer (Adam States)
        opt_tensors = self.insert_tensors_to_optimizer(
            tensors_dict, direction=direction, degree=self.state.degree, u_bar=val, insert_idx=insert_idx,
            optimizer=optimizer,
            momentum_strategy=self.state.optimizer_blend_strategy
        )

        # D. Apply to Modules
        for module in self.control_list:
            if module.name in opt_tensors:
                module.control_features = opt_tensors[module.name]


        self.position.set_weights(self.weights)
        dummy_shape = curr_H if is_u_split else curr_W
        _, new_knots_full = insert_knot_u(
            torch.zeros(dummy_shape, 1, 1, device=self.device),
            curr_knots, self.state.degree, val
        )

        if self.sampling_mode != SamplingMode.ADAPTIVE or hasattr(self.uv_sampler, '_interval_u_global'):
            self.uv_sampler.subdivide(
                direction=direction,
                val=val,
                optimizer=optimizer)

        if is_u_split:
            internal_knots = new_knots_full[self.state.degree + 1: -(self.state.degree + 1)]
            self.knot_u.update_knot_vector(self, torch.sort(internal_knots)[0], u_bar=val, optimizer=optimizer)
            self.state._H += 1
            self.state.base_u += 1 #self.state.sampling_density

        else:
            internal_knots = new_knots_full[self.state.degree + 1: -(self.state.degree + 1)]
            self.knot_v.update_knot_vector(self, torch.sort(internal_knots)[0], u_bar=val, optimizer=optimizer)
            self.state._W += 1
            self.state.base_v += 1 #self.state.sampling_density

            # self.state.update_samples(num_interval_insertions, direction='v')

        torch.cuda.empty_cache()
        self.basis.uv_sampler = self.uv_sampler
        self.basis.knot_u = self.knot_u
        self.basis.knot_v = self.knot_v

        if hasattr(self, '_chhugani_tessellator'):
            self._chhugani_tessellator.reset()
        if hasattr(self, '_forward_context'):
            from modules.tessellation.chhugani import ForwardContext
            self._forward_context = ForwardContext()
        if hasattr(self, '_chhugani_params'):
            del self._chhugani_params
        self.invalidate_all_caches(force=True)


    def get_multi_view_trim_candidates(
            self,
            observe_cnt: torch.Tensor,
            min_observations: int = 2,
            row_threshold: float = 0.7,
            col_threshold: float = 0.7
    ) -> Dict[str, List[int]]:
        """
        Identify rows/columns that are consistently unobserved across views.

        For NURBS grids, we can't prune individual points - we identify entire
        rows or columns where the majority of points are under-observed.

        Args:
            observe_cnt: [Us*Vs, 1] observation counts
            min_observations: Minimum observations to be considered "observed"
            row_threshold:  Fraction of row that must be under-observed to consider removal
            col_threshold: Fraction of column that must be under-observed to consider removal

        Returns:
            Dict with 'u' and 'v' keys containing lists of candidate indices for removal
        """
        Us, Vs = self.state.Us, self.state.Vs
        H, W = self.state._H, self.state._W
        degree = self.state.degree

        # Reshape to grid
        observe_grid = observe_cnt.view(Us, Vs)
        under_observed = (observe_grid < min_observations).float()

        candidates = {'u': [], 'v': []}

        # Analyze rows (U direction)
        # Don't consider boundary rows (clamped knot region)
        sampling_density_u = 1# int(self.state.sampling_density)
        for ctrl_idx in range(degree, H - degree):
            # Map control point row to sampling rows
            sample_start = ctrl_idx * sampling_density_u
            sample_end = min((ctrl_idx + 1) * sampling_density_u, Us)

            if sample_end <= sample_start:
                continue

            # Check fraction of under-observed points in this row band
            row_band = under_observed[sample_start:sample_end, :]
            under_obs_fraction = row_band.mean().item()

            if under_obs_fraction >= row_threshold:
                candidates['u'].append({
                    'index': ctrl_idx,
                    'under_observed_fraction': under_obs_fraction,
                    'sample_range': (sample_start, sample_end)
                })

        # Analyze columns (V direction)
        sampling_density_v = 1 #int(self.state.sampling_density)
        for ctrl_idx in range(degree, W - degree):
            # Map control point column to sampling columns
            sample_start = ctrl_idx * sampling_density_v
            sample_end = min((ctrl_idx + 1) * sampling_density_v, Vs)

            if sample_end <= sample_start:
                continue

            # Check fraction of under-observed points in this column band
            col_band = under_observed[:, sample_start:sample_end]
            under_obs_fraction = col_band.mean().item()

            if under_obs_fraction >= col_threshold:
                candidates['v'].append({
                    'index': ctrl_idx,
                    'under_observed_fraction': under_obs_fraction,
                    'sample_range': (sample_start, sample_end)
                })

        return candidates

    def prune_surface(
            self,
            cands: Optional[List[Dict]] = None,
            cand: Optional[Dict] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            error_tolerance: float = 1e-3
    ) -> int:
        """
        Apply a batch of pruning (knot removal) operations.
        Returns the number of successful removals.
        """
        if cands is None and cand is not None:
            cands = [cand]

        if not cands:
            return 0

        optimizer = optimizer if optimizer is not None else self.optimizer

        # Separate U and V candidates, sort in strictly reverse order by index
        # This ensures that removal of a higher index doesn't invalidate lower indices
        u_cands = sorted([c for c in cands if c['type'] == 'u'], key=lambda x: x['index'], reverse=True)
        v_cands = sorted([c for c in cands if c['type'] == 'v'], key=lambda x: x['index'], reverse=True)

        success_count = 0

        for cand_item in u_cands + v_cands:
            is_u = (cand_item['type'] == 'u')
            direction = 'u' if is_u else 'v'
            remove_idx = cand_item['index']

            H, W = self.state._H, self.state._W
            degree = self.state.degree

            # Safety checks
            min_size = degree + 2
            if (is_u and H <= min_size) or (not is_u and W <= min_size):
                continue

            if is_u:
                if remove_idx < degree or remove_idx >= H - degree:
                    continue
            else:
                if remove_idx < degree or remove_idx >= W - degree:
                    continue

            # Get current knot vectors
            knots_u = self.knot_u.forward()
            knots_v = self.knot_v.forward()
            curr_knots = knots_u if is_u else knots_v

            # ==========================================================================
            # 1. Remove control point rows/columns from all feature modules
            # ==========================================================================
            tensors_dict = {}

            for module in self.control_list:
                if module.control_features is None:
                    continue
                new_ctrl = module.compute_removed_grid(direction=direction, remove_idx=remove_idx)
                tensors_dict[module.name] = new_ctrl

            # ==========================================================================
            # 2. Update knot vector
            # ==========================================================================
            n_internal = len(curr_knots) - 2 * (degree + 1)
            if n_internal <= 0:
                continue

            internal_knot_idx = remove_idx - degree
            internal_knot_idx = max(0, min(internal_knot_idx, n_internal - 1))
            abs_knot_idx = degree + 1 + internal_knot_idx

            new_knots = torch.cat([
                curr_knots[:abs_knot_idx],
                curr_knots[abs_knot_idx + 1:]
            ])

            # ==========================================================================
            # 3. Update optimizer state for control features
            # ==========================================================================
            opt_tensors = self._remove_tensors_from_optimizer(
                tensors_dict,
                remove_idx,
                direction='u' if is_u else 'v',
                optimizer=optimizer
            )

            # ==========================================================================
            # 4. Apply to modules
            # ==========================================================================
            for module in self.control_list:
                if module.name in opt_tensors:
                    module.control_features = opt_tensors[module.name]

            self.position.set_weights(self.weights)

            # ==========================================================================
            # 5. Update sampler intervals BEFORE updating state dimensions
            # ==========================================================================
            self.uv_sampler.prune_uv(direction, removed_idx=remove_idx, optimizer=optimizer)

            new_internal_knots = new_knots[degree + 1:-(degree + 1)]
            if is_u:
                self.knot_u.update_knot_vector(self, new_internal_knots, u_bar=None, optimizer=optimizer)
                self.state._H -= 1
                self.state.base_u -= 1
            else:
                self.knot_v.update_knot_vector(self, new_internal_knots, u_bar=None, optimizer=optimizer)
                self.state._W -= 1
                self.state.base_v -= 1

            success_count += 1

        if success_count == 0:
            return 0

        # DONE WITH ALL BATCH CANDIDATES. NOW FINALIZE.
        self.basis.uv_sampler = self.uv_sampler
        self.basis.knot_u = self.knot_u
        self.basis.knot_v = self.knot_v
        torch.cuda.empty_cache()

        if hasattr(self, '_chhugani_tessellator'):
            self._chhugani_tessellator.reset()
        if hasattr(self, '_forward_context'):
            from modules.tessellation.chhugani import ForwardContext
            self._forward_context = ForwardContext()
        if hasattr(self, '_chhugani_params'):
            del self._chhugani_params

        self.invalidate_all_caches(force=True)
        self.recompute()

        return success_count
    def prune_surface2(
            self,
            cand: Dict,
            optimizer: Optional[torch.optim.Optimizer] = None,
            error_tolerance: float = 1e-3
    ) -> bool:
        """
        Apply a single pruning (knot removal) operation.
        """
        optimizer = optimizer if optimizer is not None else self.optimizer

        is_u = (cand['type'] == 'u')
        direction = 'u' if is_u else 'v'
        remove_idx = cand['index']

        H, W = self.state._H, self.state._W
        degree = self.state.degree
        # Safety checks
        min_size = degree + 2
        if (is_u and H <= min_size) or (not is_u and W <= min_size):
            return False

        if is_u:
            if remove_idx < degree or remove_idx >= H - degree:
                return False
        else:
            if remove_idx < degree or remove_idx >= W - degree:
                return False

        # Get current knot vectors
        knots_u = self.knot_u.forward()
        knots_v = self.knot_v.forward()
        curr_knots = knots_u if is_u else knots_v

        # ==========================================================================
        # 1. Remove control point rows/columns from all feature modules
        # ==========================================================================

        tensors_dict = {}

        for module in self.control_list:
            if module.control_features is None:
                continue
            new_ctrl = module.compute_removed_grid(direction=direction, remove_idx=remove_idx)
            tensors_dict[module.name] = new_ctrl

        # ==========================================================================
        # 2. Update knot vector
        # ==========================================================================

        n_internal = len(curr_knots) - 2 * (degree + 1)
        if n_internal <= 0:
            return False

        internal_knot_idx = remove_idx - degree
        internal_knot_idx = max(0, min(internal_knot_idx, n_internal - 1))
        abs_knot_idx = degree + 1 + internal_knot_idx

        new_knots = torch.cat([
            curr_knots[:abs_knot_idx],
            curr_knots[abs_knot_idx + 1:]
        ])

        # ==========================================================================
        # 3. Update optimizer state for control features
        # ==========================================================================

        opt_tensors = self._remove_tensors_from_optimizer(
            tensors_dict,
            remove_idx,
            direction='u' if is_u else 'v',
            optimizer=optimizer
        )

        # ==========================================================================
        # 4. Apply to modules
        # ==========================================================================

        for module in self.control_list:
            if module.name in opt_tensors:
                module.control_features = opt_tensors[module.name]

        self.position.set_weights(self.weights)
        # ==========================================================================
        # 5. Update sampler intervals BEFORE updating state dimensions
        # ==========================================================================
        #
        self.uv_sampler.prune_uv(direction, removed_idx=remove_idx, optimizer=optimizer)

        new_internal_knots = new_knots[degree + 1:-(degree + 1)]
        if is_u:
            self.knot_u.update_knot_vector(self, new_internal_knots, u_bar=None, optimizer=optimizer)
            self.state._H -= 1
            self.state.base_u -= 1

        else:
            self.knot_v.update_knot_vector(self, new_internal_knots, u_bar=None, optimizer=optimizer)
            self.state._W -= 1
            self.state.base_v -= 1# self.state.sampling_density

        self.basis.uv_sampler = self.uv_sampler
        self.basis.knot_u = self.knot_u
        self.basis.knot_v = self.knot_v
        torch.cuda.empty_cache()
        if hasattr(self, '_chhugani_tessellator'):
            self._chhugani_tessellator.reset()
        if hasattr(self, '_forward_context'):
            from tessellation.chhugani import ForwardContext
            self._forward_context = ForwardContext()
        if hasattr(self, '_chhugani_params'):
            del self._chhugani_params
        self.invalidate_all_caches(force=True)
        self.recompute()
        return True

    @staticmethod
    def evaluate_points(einsum_path:str, control_points:torch.Tensor, basis_u:torch.Tensor, basis_v:torch.Tensor, optimal_path:str = 'auto') -> torch.Tensor:
        return oe.contract(
            einsum_path,
            basis_u,
            control_points,
            basis_v,
            optimize=optimal_path
            )

    def optimize_intervals(self, num_steps=200, lr=0.005, verbose=True, render_fn=None, **kwargs):
        """
        Optimize intervals with VERIFIED gradient flow.
        """
        original_u = self.uv_sampler.interval_u.clone().clamp(1e-6, 1 - 1e-6)
        original_v = self.uv_sampler.interval_v.clone().clamp(1e-6, 1 - 1e-6)

        interval_u = nn.Parameter(
            inverse_sigmoid(original_u),
            requires_grad=True
        )
        interval_v = nn.Parameter(
            inverse_sigmoid(original_v),
            requires_grad=True
        )
        cameras = kwargs.get('camera', None)
        render_pkg = None
        if cameras is not None:
            pipe = kwargs.get('pipe', None)
            background = kwargs.get('background', None)
            app_model = kwargs.get('app_model', None)

        optimizer = torch.optim.Adam([interval_u, interval_v], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()
            if cameras is not None:
                viewpoint_stack = cameras.copy()
                import random
                cam_idx = random.randint(0, len(viewpoint_stack)-1)
                cam = viewpoint_stack.pop(cam_idx)
                render_pkg = render_fn(cam, self, pipe, background, app_model=app_model,
                                       return_plane=False, return_depth_normal=False)
                if cam is not None:
                    gt_img = cam.get_images()[0].cuda()
                    loss_recon = self._rendering_reconstruction_loss(cam, gt_img)
            # Activate and sort
            interval_u_active = torch.sigmoid(interval_u)#.clamp(1e-6, 1 - 1e-6)
            interval_v_active = torch.sigmoid(interval_v)#.clamp(1e-6, 1 - 1e-6)

            interval_u_sorted, _ = torch.sort(interval_u_active.squeeze(), dim=0)
            interval_v_sorted, _ = torch.sort(interval_v_active.squeeze(), dim=0)

            # === CRITICAL: Use differentiable basis generation ===
            buv = compute_bases_uv_diff(
                interval_u_sorted,  # Has requires_grad=True via chain rule
                interval_v_sorted,
                self.knot_u(),
                self.knot_v(),
                self.state.H,
                self.state.W,
                degree=3
            )


            # Compute loss (with gradient flow)
            total_loss = self._curvature_based_density_loss_wasserstein(
                interval_u_sorted, interval_v_sorted, buv=buv
            )


            total_loss = total_loss + loss_recon #+ self.eikonal(buv)


            # Check for NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[WARNING] NaN/Inf loss at step {step}")
                continue

            # Backprop
            total_loss.backward()

            # === CRITICAL CHECK: Verify gradients exist ===
            if interval_u.grad is None or interval_v.grad is None:
                print(f"[ERROR] No gradients at step {step}!")
                print(f"  interval_u.requires_grad: {interval_u.requires_grad}")
                print(f"  interval_u_sorted.requires_grad: {interval_u_sorted.requires_grad}")
                print(f"  bu.requires_grad: {buv.bu.requires_grad}")
                print(f"  total_loss.requires_grad: {total_loss.requires_grad}")

                # Debug: Try simple loss
                if step == 0:
                    print("\n=== Testing simple gradient flow ===")
                    test_loss = interval_u_sorted.sum()
                    test_loss.backward()
                    print(f"Simple test gradient: {interval_u.grad is not None}")
                    optimizer.zero_grad()

                continue

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_([interval_u, interval_v], max_norm=1.0)

            optimizer.step()

            if verbose and step % 10 == 0:
                grad_norm_u = interval_u.grad.norm().item()
                grad_norm_v = interval_v.grad.norm().item()
                print(f"[Step {step}] Loss: {total_loss.item():.6f} | "
                      f"Grad: U={grad_norm_u:.4f}, V={grad_norm_v:.4f}")

        # Final update
        interval_u_final = torch.sigmoid(interval_u).detach().clone()
        interval_v_final = torch.sigmoid(interval_v).detach().clone()
        self.uv_sampler._interval_u['0'] = interval_u_final
        self.uv_sampler._interval_v['0'] = interval_v_final
        # Generate final basis
        final_basis_data = compute_bases_uv_diff(
            interval_u_final.sort()[0],
            interval_v_final.sort()[0],
            self.knot_u(),
            self.knot_v(),
            self.state.H,
            self.state.W,
            degree=3
        )
        self.basis.replace_funcs(final_basis_data)

        if verbose:
            improvement = (original_u - interval_u_final).abs().mean() + (original_v - interval_v_final).abs().mean()
            print(f"\n[Interval Optimization] Complete | Interval Change: {improvement.item():.6f}")

    def prune_grid(
            self,
            min_opacity: float = 0.005,
            max_screen_size: float = 20.0,
            max_world_scale_factor: float = 0.1,
            extent: Optional[float] = None,
            max_removals: int = 3,
            error_tolerance: float = 1e-4,
            optimizer: Optional[torch.optim.Optimizer] = None,
            verbose: bool = True
    , candidates=None) -> int:
        """
        Main pruning entry point - removes knots based on 3DGS-like criteria.

        Similar to 3DGS densify_and_prune but for NURBS surfaces.

        Args:
            min_opacity: Remove rows/cols with mean opacity below this
            max_screen_size: Remove rows/cols with max radii above this
            max_world_scale_factor: Remove if scale > factor * extent
            extent: Scene extent (auto-computed if None)
            max_removals:  Maximum number of removals per call
            error_tolerance: Max surface deviation allowed for removal
            optimizer: Optimizer to update
            verbose: Print progress

        Returns:
            Number of successful removals
        """
        optimizer = optimizer if optimizer is not None else self.optimizer
        if not candidates:
            if verbose:
                print("[Pruning] No candidates found")
            return 0

        if verbose:
            print(f"[Pruning] Found {len(candidates)} candidates, attempting up to {max_removals} removals")

        successful_removals = 0
        attempted = 0

        for cand in candidates[: max_removals * 2]:  # Try more than max to account for failures
            if successful_removals >= max_removals:
                break

            attempted += 1
            success = self.prune_surface2(
                cand,
                optimizer=optimizer,
                error_tolerance=error_tolerance
            )

            if success:
                successful_removals += 1
                if verbose:
                    print(f"  [Pruning] Removed {cand['type'].upper()} at idx={cand['index']}, "
                          f"val={cand['val']:.4f}, reasons:  {', '.join(cand['reasons'])}")
            else:
                if verbose:
                    print(f"  [Pruning] Failed to remove {cand['type'].upper()} at idx={cand['index']} "
                          f"(exceeded error tolerance)")

        if verbose:
            print(f"[Pruning] Completed:  {successful_removals}/{attempted} successful, "
                  f"grid now {self.state._H}x{self.state._W}")

        return successful_removals


    def tessellate(
            self,
            camera=None,
            config: Optional[TessellationConfig] = None,
    ) -> TriangleMesh:
        """
        Generate differentiable triangle mesh from current surface.

        Args:
            camera: Optional camera for normal orientation
            config:  Tessellation configuration

        Returns:
            TriangleMesh with differentiable vertices
        """
        if not hasattr(self, '_tessellator') or config is not None:
            self._tessellator = GridTessellator(config or TessellationConfig())

        Us, Vs = self.state.Us, self.state.Vs

        # Get surface points
        xyz = self.get_xyz.reshape(Us, Vs, 3)

        # Get colors from SH DC component
        colors = self.get_features[:, 0, :].reshape(Us, Vs, 3)
        colors = torch.sigmoid(colors)  # Map to [0, 1]

        # Get normals (use geometric for better quality)
        du = self.position.du.reshape(Us, Vs, 3)
        dv = self.position.dv.reshape(Us, Vs, 3)
        normals = torch.cross(du, dv, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        # Orient toward camera if provided
        if camera is not None:
            view_dirs = camera.camera_center - xyz
            dot = (normals * view_dirs).sum(dim=-1, keepdim=True)
            normals = torch.where(dot < 0, -normals, normals)

        return self._tessellator.tessellate(xyz, colors, normals)

    def export_mesh(
            self,
            filepath: str,
            camera=None,
            format: str = 'obj',
            resolution: Optional[Tuple[int, int]] = None,
            **kwargs
    ):
        """
        Export surface as triangle mesh file.

        Args:
            filepath: Output file path
            camera: Optional camera for normal orientation
            format: 'obj' or 'ply'
            resolution: Optional (Us, Vs) for export resolution
            **kwargs: Additional export options
        """
        if resolution is not None and resolution != (self.state.Us, self.state.Vs):
            # Generate at different resolution
            mesh = self._tessellate_at_resolution(resolution, camera)
        else:
            mesh = self.tessellate(camera)

        if format.lower() == 'obj':
            mesh_to_obj(mesh, filepath, **kwargs)
        elif format.lower() == 'ply':
            mesh_to_ply(mesh, filepath, **kwargs)
        else:
            raise ValueError(f"Unknown format:  {format}")

        print(f"Exported mesh to {filepath}:  {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    def _tessellate_at_resolution(
            self,
            resolution: Tuple[int, int],
            camera=None,
    ) -> TriangleMesh:
        """Generate mesh at specific resolution."""
        Us, Vs = resolution
        device = self.device

        # Create UV grid at target resolution
        u_samples = torch.linspace(0, 1, Us, device=device)
        v_samples = torch.linspace(0, 1, Vs, device=device)
        uu, vv = torch.meshgrid(u_samples, v_samples, indexing='ij')
        uv_grid = torch.stack([uu, vv], dim=-1).reshape(-1, 2)

        # Temporarily compute basis at new resolution
        with torch.no_grad():
            self.basis.forward(uv_grid, self.knot_u(), self.knot_v())
            xyz = self.position.forward().reshape(Us, Vs, 3)
            colors = self.spherical_harmonics.sh_dc.forward()
            colors = torch.sigmoid(colors.reshape(Us, Vs, 3))

            # Restore original basis
            self.invalidate_all_caches()

        # Compute normals from finite differences
        du = torch.zeros_like(xyz)
        dv = torch.zeros_like(xyz)
        du[:-1] = xyz[1:] - xyz[:-1]
        du[-1] = du[-2]
        dv[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
        dv[:, -1] = dv[:, -2]

        normals = torch.cross(du, dv, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        if camera is not None:
            view_dirs = camera.camera_center - xyz
            dot = (normals * view_dirs).sum(dim=-1, keepdim=True)
            normals = torch.where(dot < 0, -normals, normals)

        tessellator = GridTessellator(TessellationConfig())
        return tessellator.tessellate(xyz, colors, normals)


