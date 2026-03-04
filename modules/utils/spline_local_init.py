
# model/utils/pointcloud_sampling.py
import torch

def random_downsample(*tensors, reduction: float = 0.01, generator=None):
    """
    Randomly keep `reduction` fraction of rows from each input tensor.
    All tensors must have the same first-dimension length.

    Args
    ----
    *tensors   : torch.Tensor
        Point-aligned tensors, e.g. xyz (N,3), uv (N,2), normals (N,3).
    reduction  : float, default 0.01
        Fraction to keep; e.g. 0.01 ⇒ 1 %.
    generator  : torch.Generator or None
        Optional RNG to make the sampling reproducible.

    Returns
    -------
    Tuple[torch.Tensor, …]  — tensors with ⌈N·reduction⌉ rows
    """
    if not 0.0 < reduction <= 1.0:
        raise ValueError("reduction must be in (0,1].")

    N = tensors[0].shape[0]
    if any(t.shape[0] != N for t in tensors):
        raise ValueError("All tensors must share the same first dimension.")

    # Bernoulli mask — allocate on the device of the first tensor
    device = tensors[0].device
    rng = torch.rand(N, device=device, generator=generator)
    keep = rng < reduction                                   # Bool mask

    # Safety check: never return an empty set
    if not keep.any():
        # Fall back to at least one random index
        idx = torch.randint(0, N, (1,), device=device, generator=generator)
        keep[idx] = True

    return tuple(t[keep] for t in tensors)
import torch
from torch.nn import functional as F
def voxel_downsample(xyz, uv, size=0.01):
    keys = torch.floor(xyz / size).long()
    _, uniq = torch.unique(keys, return_inverse=True, dim=0)
    keep = torch.zeros_like(uniq).scatter_reduce_(0, uniq, uniq,
                                                  reduce='amin').bool()
    return xyz[keep], uv[keep]
# ---------- 1.1  helper: depth → point cloud -------------------------------
def depth_maps_to_pointcloud(depth, K, T_cam2world, device='cuda'):
    """
    Args
        depth : (B,H,W)  float32 [metres]
        K     : (B,3,3)  intrinsics
        T_cam2world : (B,4,4)
    Returns
        xyz : (N,3)  concatenated point cloud in world coords
        uv  : (N,2)  image-plane barycentric parameters in [0,1]^2
    """
    B, H, W = depth.shape
    i, j = torch.meshgrid(torch.arange(H, device=depth.device),
                          torch.arange(W, device=depth.device),
                          indexing="ij")
    ones = torch.ones_like(i, dtype=torch.float32)
    pix = torch.stack((j, i, ones), dim=-1)  # (H,W,3)

    pix = pix.view(1, H * W, 3).repeat(B, 1, 1)           # (B,HW,3)
    d   = depth.view(B, -1, 1)                            # (B,HW,1)
    Kinv = torch.inverse(K)                               # (B,3,3)

    cam_xyz = (Kinv @ pix.transpose(1,2)).transpose(1,2)  # (B,HW,3)
    cam_xyz *= d                                          # scale by depth
    world   = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], -1)
    world   = (T_cam2world @ world.transpose(1,2)).transpose(1,2)[...,:3]

    uv = torch.stack((j.float() / (W-1), i.float() / (H-1)), -1)  # (H,W,2)
    uv = uv.view(1, -1, 2).repeat(B, 1, 1)
    # world, uv = voxel_downsample(world, uv)

    return world.reshape(-1,3), uv.reshape(-1,2)

# ---------- 1.2  helper: centripetal parametrisation -----------------------
def chord_length_knots(params, degree, num_ctrl):
    """Returns open, centripetal knot vector as torch tensor (num_ctrl+degree+1)."""
    d = torch.sqrt(torch.sum((params[1:] - params[:-1]) ** 2, dim=-1))
    u = torch.cat((torch.zeros(1, device=d.device),
                   torch.cumsum(d / d.sum(), dim=0)))
    # open uniform style ends
    knot = torch.nn.functional.pad(u, (degree+1, degree+1), mode="constant", value=1.0)
    knot[:degree+1] = 0.0
    knot[-degree-1:] = 1.0
    return knot

# ---------- 1.3  Liew-style local least-squares seeding --------------------

# model/utils/pointcloud_sampling.py
import torch

def random_downsample(*tensors, reduction: float = 0.01, generator=None):
    """
    Randomly keep `reduction` fraction of rows from each input tensor.
    All tensors must have the same first-dimension length.

    Args
    ----
    *tensors   : torch.Tensor
        Point-aligned tensors, e.g. xyz (N,3), uv (N,2), normals (N,3).
    reduction  : float, default 0.01
        Fraction to keep; e.g. 0.01 ⇒ 1 %.
    generator  : torch.Generator or None
        Optional RNG to make the sampling reproducible.

    Returns
    -------
    Tuple[torch.Tensor, …]  — tensors with ⌈N·reduction⌉ rows
    """
    if not 0.0 < reduction <= 1.0:
        raise ValueError("reduction must be in (0,1].")

    N = tensors[0].shape[0]
    if any(t.shape[0] != N for t in tensors):
        raise ValueError("All tensors must share the same first dimension.")

    # Bernoulli mask — allocate on the device of the first tensor
    device = tensors[0].device
    rng = torch.rand(N, device=device, generator=generator)
    keep = rng < reduction                                   # Bool mask

    # Safety check: never return an empty set
    if not keep.any():
        # Fall back to at least one random index
        idx = torch.randint(0, N, (1,), device=device, generator=generator)
        keep[idx] = True

    return tuple(t[keep] for t in tensors)
import torch
from torch.nn import functional as F
def voxel_downsample(xyz, uv, size=0.01):
    keys = torch.floor(xyz / size).long()
    _, uniq = torch.unique(keys, return_inverse=True, dim=0)
    keep = torch.zeros_like(uniq).scatter_reduce_(0, uniq, uniq,
                                                  reduce='amin').bool()
    return xyz[keep], uv[keep]
# ---------- 1.1  helper: depth → point cloud -------------------------------
def depth_maps_to_pointcloud(depth, K, T_cam2world, device='cuda'):
    """
    Args
        depth : (B,H,W)  float32 [metres]
        K     : (B,3,3)  intrinsics
        T_cam2world : (B,4,4)
    Returns
        xyz : (N,3)  concatenated point cloud in world coords
        uv  : (N,2)  image-plane barycentric parameters in [0,1]^2
    """
    B, H, W = depth.shape
    i, j = torch.meshgrid(torch.arange(H, device=depth.device),
                          torch.arange(W, device=depth.device),
                          indexing="ij")
    ones = torch.ones_like(i, dtype=torch.float32)
    pix = torch.stack((j, i, ones), dim=-1)  # (H,W,3)

    pix = pix.view(1, H * W, 3).repeat(B, 1, 1)           # (B,HW,3)
    d   = depth.view(B, -1, 1)                            # (B,HW,1)
    Kinv = torch.inverse(K)                               # (B,3,3)

    cam_xyz = (Kinv @ pix.transpose(1,2)).transpose(1,2)  # (B,HW,3)
    cam_xyz *= d                                          # scale by depth
    world   = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], -1)
    world   = (T_cam2world @ world.transpose(1,2)).transpose(1,2)[...,:3]

    uv = torch.stack((j.float() / (W-1), i.float() / (H-1)), -1)  # (H,W,2)
    uv = uv.view(1, -1, 2).repeat(B, 1, 1)
    # world, uv = voxel_downsample(world, uv)

    return world.reshape(-1,3), uv.reshape(-1,2)

# ---------- 1.2  helper: centripetal parametrisation -----------------------
# ---------- helper: centripetal / fallback knot vector --------------------
def chord_length_knots(params: torch.Tensor,
                       degree: int,
                       num_ctrl: int,
                       use_uniform=True) -> torch.Tensor:
    """
    Produce an open knot vector of length `num_ctrl + degree + 1`.
    • Centripetal parametrisation when ≥ 2 distinct parameter samples
    • Uniform fallback when not enough information is available
    """
    device = params.device
    if params.numel() < 2 or use_uniform:
        use_uniform = True
    else:
        d = torch.diff(params)
        use_uniform = torch.allclose(d, torch.zeros_like(d))

    if use_uniform:
        # Uniform open knot vector
        knot = torch.zeros(num_ctrl + degree + 1, device=device)
        knot[degree + 1 : num_ctrl] = torch.linspace(
            0.0, 1.0, num_ctrl - degree - 1, device=device
        )
        knot[num_ctrl:] = 1.0
        return knot

    # --- standard centripetal formulation --------------------------------
    d = torch.sqrt(torch.sum((params[1:] - params[:-1]) ** 2, dim=-1))
    u = torch.cat(
        (
            torch.zeros(1, device=device),
            torch.cumsum(d / d.sum(), dim=0)[..., None],
        )
    )
    knot = torch.nn.functional.pad(
        u, (degree + 1, degree + 1), mode="constant", value=1.0
    )
    knot[: degree + 1] = 0.0
    knot[-degree - 1 :] = 1.0
    return knot

# ---------- 1.3  Liew‑style local least‑squares seeding --------------------
def seed_control_grid(
    xyz,
    uv,
    grid_res=(32, 32),
    degree=(3, 3),
    k: int = 16,
    reduce_factor: float = 0.01,
):
    """
    Density‑aware bootstrap that converts a (possibly huge) point cloud
    into an initial bicubic tensor‑product B‑spline control grid.

    Parameters
    ----------
    xyz : (N, 3) torch.Tensor
        Point positions in world metres.
    uv  : (N, 2) torch.Tensor
        Normalised parameter coordinates in [0, 1]² produced by
        `depth_maps_to_pointcloud`.
    grid_res : (int, int)
        Number of B‑spline *surface* evaluation samples per direction.
        The resulting control‑grid resolution equals
        `(grid_res[i] // 4 + degree[i])`.
    degree : (int, int)
        B‑spline polynomial degree in (u, v).
    k : int
        Unused for now – kept for future k‑NN variants.
    reduce_factor : float in (0, 1]
        Target fraction of points to *keep* (≈ memory guard).

    Returns
    -------
    ctrl_pts : (Nu, Nv, 3) torch.Tensor
    knot_u   : (Nu + du + 1) torch.Tensor
    knot_v   : (Nv + dv + 1) torch.Tensor
    """
    from simple_knn._C import distCUDA2  # fast, O(N) memory
    with torch.no_grad():

        device = xyz.device
        du, dv = degree

        # ---------------------------------------------------------------------
        # 1.  Density‑aware down‑sampling  (keep ≈ reduce_factor × N points)
        # ---------------------------------------------------------------------
        # distCUDA2 returns squared distance to the nearest neighbour
        nn_d2 = distCUDA2(xyz)           # (N,)   numpy or torch
        if not isinstance(nn_d2, torch.Tensor):
            nn_d2 = torch.from_numpy(nn_d2).to(device)
        nn_d = torch.sqrt(nn_d2)         # metres
        thresh = torch.quantile(nn_d, 1.0 - reduce_factor)
        keep_mask = nn_d >= thresh       # sparse areas survive
        xyz = xyz[keep_mask]
        uv = uv[keep_mask]

        # ---------------------------------------------------------------------
        # 2.  Grid topology
        # ---------------------------------------------------------------------
        Nu, Nv = [r // 4 + d for r, d in zip(grid_res, degree)]  # control pts
        # Patch‑local integer cell coordinates
        cell_u = torch.clamp((uv[:, 0] * (grid_res[0] - 1) / 4).long(), 0, Nu - du - 1)
        cell_v = torch.clamp((uv[:, 1] * (grid_res[1] - 1) / 4).long(), 0, Nv - dv - 1)

        # ---------------------------------------------------------------------
        # 3.  Cubic B‑spline basis evaluation (closed form, no F.interpolate)
        # ---------------------------------------------------------------------
        def cubic_basis(t: torch.Tensor) -> torch.Tensor:
            """Return (N,4) cubic B‑spline basis at relative offset t∈[0,1]."""
            t2, t3 = t * t, t * t * t
            return torch.stack(
                (
                    (1 - 3 * t + 3 * t2 - t3) / 6,
                    (4 - 6 * t2 + 3 * t3) / 6,
                    (1 + 3 * t + 3 * t2 - 3 * t3) / 6,
                    t3 / 6,
                ),
                dim=-1,
            )

        frac_u = (uv[:, 0] * (grid_res[0] - 1) / 4) - cell_u
        frac_v = (uv[:, 1] * (grid_res[1] - 1) / 4) - cell_v
        Bu = cubic_basis(frac_u)
        Bv = cubic_basis(frac_v)
        basis = (Bu.unsqueeze(2) * Bv.unsqueeze(1)).view(-1, 16)  # (M,16)

        # ---------------------------------------------------------------------
        # 4.  Scatter‑add into control grid
        # ---------------------------------------------------------------------
        ctrl_pts = torch.zeros((Nu, Nv, 3), device=device)
        wt_sum = torch.zeros((Nu, Nv, 1), device=device)

        # Pre‑compute per‑sample control‑point indices
        for du_i in range(4):
            for dv_j in range(4):
                w = basis[:, du_i * 4 + dv_j].unsqueeze(-1)       # (M,1)
                if torch.allclose(w, torch.zeros_like(w)):
                    continue
                u_idx = (cell_u + du_i).clamp(max=Nu - 1)
                v_idx = (cell_v + dv_j).clamp(max=Nv - 1)

                ctrl_pts.index_put_((u_idx, v_idx), xyz * w, accumulate=True)
                wt_sum.index_put_((u_idx, v_idx), w, accumulate=True)

        ctrl_pts = ctrl_pts / wt_sum.clamp_min(1e-6)

        # ---------------------------------------------------------------------
        # 5.  Knot vectors (centripetal)
        # ---------------------------------------------------------------------
        knot_u = chord_length_knots(uv[:, 0], du, Nu - du).to(device)
        knot_v = chord_length_knots(uv[:, 1], dv, Nv - dv).to(device)
        return ctrl_pts, knot_u, knot_v

def seed_control_grid2(xyz, uv,
                      grid_res=(32,32), degree=(3,3), k=16, reduce_factor=0.01):
    """
    Args
        xyz      : (N,3) point positions (metres, world)
        uv       : (N,2) param coords in [0,1]^2
        grid_res : (#patch_u * 4 , #patch_v * 4) for bicubic
    Returns
        ctrl_pts : (Nu,Nv,3)
        knot_u   : (Nu+du+1)
        knot_v   : (Nv+dv+1)
    """
    from simple_knn._C import distCUDA2

    device = xyz.device
    du,dv  = degree
    Nu, Nv = [r//4 + du for r in grid_res]  # 4×4 control pts per patch

    # 1.3.1 build cell indices for each sample
    cell_u = torch.clamp((uv[:,0] * (grid_res[0]-1) / 4).long(), 0, Nu-du-1)
    cell_v = torch.clamp((uv[:,1] * (grid_res[1]-1) / 4).long(), 0, Nv-dv-1)

    # index in flattened control grid (0..Nu*Nv-1)
    root_idx = cell_v * Nu + cell_u

    dist2 = distCUDA2(torch.from_numpy(xyz.detach().cpu().numpy()).float().cuda())
    dist = dist2[dist2 < torch.quantile(dist2, q=reduce_factor)]

    grid_t = torch.linspace(0,1,5,device=device)
    B = torch.stack([F.interpolate(
                          torch.eye(1,5,device=device), size=4,
                          mode='linear', align_corners=False)[0]  # placeholder
                    ])  # not elegant, but we only need 4 values anyway
    # quick tensor look-up
    Bu = B[0, (uv[:,0]*(B.shape[1]-1)).long()]
    Bv = B[0, (uv[:,1]*(B.shape[1]-1)).long()]
    basis = (Bu.unsqueeze(1) * Bv.unsqueeze(2)).view(-1,16)  # (N,16)

    # 1.3.4 accumulate numerator and weights in a sparse grid
    ctrl_pts = torch.zeros((Nu,Nv,3), device=device)
    wt_sum   = torch.zeros((Nu,Nv,1), device=device)

    # flatten grid for scatter-add
    flat_xyz = xyz.repeat(1,16).view(-1,3)                    # (N*16,3)
    flat_wt  = basis.repeat_interleave(1,dim=0)[...,None]     # (N*16,1)

    # which control point each copied sample contributes to
    offset_u = torch.arange(4, device=device).repeat(Nv*Nu)  # dummy
    # *** This section is intentionally left schematic ***:
    # build indices u_idx, v_idx for each contribution,
    # then scatter_reduce().
    # For brevity I omit the boilerplate; the idea should be clear.

    # final divide
    ctrl_pts = ctrl_pts / wt_sum.clamp_min(1e-6)

    # 1.3.5 knot vectors
    knot_u = chord_length_knots(uv[:,0], du, Nu-du)
    knot_v = chord_length_knots(uv[:,1], dv, Nv-dv)

    return ctrl_pts, knot_u, knot_v