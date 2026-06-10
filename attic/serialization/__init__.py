import torch
from typing import Dict, Any, Optional
from modules.KnotSurface import ModelState, SplineModel, BasisFunction, PositionControl, RotationControl, ScalingControl, \
    OpacityControl, SHControl, SHControlWrapper, SamplingMode
from modules.sampling.SamplerUV import SamplerUV#, SoftmaxIntervalSampler
from modules.knotvector import KnotVector
from modules.multisurf import MultiSurfaceSplineModel
from modules.fitting.nurbs_from_pointcloud import DecompositionMode
from utils.general_utils import inverse_sigmoid


def capture_spline_model(model) -> Dict[str, Any]:
    """
    Capture complete model state for serialization.
    All tensors are detached and moved to CPU.
    """
    state = model.state

    def to_cpu(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().clone()
        return tensor

    def to_cpu_dict(d):
        if d is None:
            return None
        return {k: to_cpu(v) for k, v in d.items()}

    # CRITICAL FIX: Save raw parameters if they exist, otherwise save activated values
    # This prevents precision loss from sigmoid -> inverse_sigmoid cycle

    # 1. Capture Knots
    if hasattr(model.knot_u, '_internal_knots') and isinstance(model.knot_u._internal_knots, torch.nn.Parameter):
        knot_u_raw = to_cpu(model.knot_u._internal_knots)
    else:
        # Fallback for static mode: ensure we map back to raw space later if needed
        knot_u_raw = to_cpu(inverse_sigmoid(
            model.knot_u.internal_knots) if model.knot_u.should_optimize else model.knot_u.internal_knots)

    if hasattr(model.knot_v, '_internal_knots') and isinstance(model.knot_v._internal_knots, torch.nn.Parameter):
        knot_v_raw = to_cpu(model.knot_v._internal_knots)
    else:
        knot_v_raw = to_cpu(inverse_sigmoid(
            model.knot_v.internal_knots) if model.knot_v.should_optimize else model.knot_v.internal_knots)

    captured = {
        'position': to_cpu(model.position.control_features.data),
        'sh_dc': to_cpu(model.spherical_harmonics.sh_dc.control_features.data),
        'sh_rest': to_cpu(model.spherical_harmonics.sh_rest.control_features.data),
        'scaling': to_cpu(model.scaling.control_features.data),
        'rotation': to_cpu(model.rotation.control_features.data),
        'opacity': to_cpu(model.opacity.control_features.data),
        'knot_u': knot_u_raw,
        'knot_v': knot_v_raw,
        'uv_sampler': {
            'mode': model.uv_sampler.mode,
            'num_channels': model.uv_sampler.num_channels,
            'interval_u': export_interval_dict(model.uv_sampler._interval_u),
            'interval_v': export_interval_dict(model.uv_sampler._interval_v),
            'vis_probs': to_cpu_dict(getattr(model.uv_sampler, 'vis_probs', {})),
        },

        'state': {
            'H': state.H,
            'W': state.W,
            'degree': state.degree,
            'sampling_density_u': state.sampling_density,
            'sampling_density_v': state.sampling_density,
            'active_sh_degree': state.active_sh_degree,
            'max_sh_degree': state.max_sh_degree,
            'opt_dict': serialize_opt(state.opt),
        },

        # Optimizer state
        'optimizer': model.optimizer.state_dict() if hasattr(model,
                                                             'optimizer') and model.optimizer is not None else None,

        # Training state
        'iteration': getattr(model, 'iteration', 0),
        'spatial_lr_scale': getattr(model, 'spatial_lr_scale', 1.0),
        '_version': '2.1',  # Bumped version
    }

    return captured


def export_interval_dict(interval_dict) -> Dict:
    """Export interval dictionary handling both dict and tensor formats."""
    if interval_dict is None:
        return {}

    if isinstance(interval_dict, dict):
        return {
            str(k): v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in interval_dict.items()
        }
    elif isinstance(interval_dict, torch.Tensor):
        return {'_tensor': interval_dict.detach().cpu().clone()}
    elif isinstance(interval_dict, torch.nn.ParameterList):
        return {
            str(i): p.detach().cpu().clone()
            for i, p in enumerate(interval_dict)
        }
    return {}


def serialize_opt(opt) -> Dict:
    """Convert optimization params to serializable dict."""
    if hasattr(opt, '__dict__'):
        return {k: v for k, v in opt.__dict__.items() if not k.startswith('_')}
    return {}


def restore_spline_model(model, state_dict: Dict, train_mode: bool = False, device: str = 'cuda'):
    """
    Restore model from saved state.
    """

    def to_device(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        return tensor

    # Restore state configuration
    state_config = state_dict['state']

    # Reconstruct opt
    from arguments.nurbs_params import NurbsOptimizationParams
    opt = NurbsOptimizationParams()
    if 'opt_dict' in state_config:
        for k, v in state_config['opt_dict'].items():
            if hasattr(opt, k):
                setattr(opt, k, v)

    # Force optimize flags based on train_mode
    if not train_mode:
        opt.optimize_knots = False
        opt.optimize_intervals = False

    # Create ModelState
    model.state = ModelState(
        opt=opt,
        H=state_config['H'],
        W=state_config['W'],
        device=device,
        num_surfaces=state_config['num_surfaces'],
        degree=state_config['degree'],
        active_sh_degree=state_config['active_sh_degree'],
        max_sh_degree=state_config['max_sh_degree'],
    )
    model.state.sampling_density = state_config['sampling_density_u']
    model.state.sampling_density = state_config['sampling_density_v']

    # --- CRITICAL FIX FOR KNOTS ---
    # We must handle the raw vs activated loading carefully.

    loaded_knot_u = to_device(state_dict['knot_u'])
    loaded_knot_v = to_device(state_dict['knot_v'])

    # If we are training, we load the RAW values directly into the parameter
    # The KnotVector module usually expects 'initial_knots' to be in Euclidean (0-1) space.
    # We need to bypass the initialization logic if we have raw parameters.

    model.knot_u = KnotVector(
        model.state, direction='u',
        initial_knots=None,  # Don't init yet
        evaluate_mode=not train_mode
    )
    _restore_knot_data(model.knot_u, loaded_knot_u, train_mode and opt.optimize_knots)

    model.knot_v = KnotVector(
        model.state, direction='v',
        initial_knots=None,
        evaluate_mode=not train_mode
    )
    _restore_knot_data(model.knot_v, loaded_knot_v, train_mode and opt.optimize_knots)

    uv_state = state_dict['uv_sampler']
    model.uv_sampler = create_sampler_from_state(model.state, uv_state, device, train_mode and opt.optimize_intervals)

    # Restore basis
    model.basis = BasisFunction(model.state, model.knot_u, model.knot_v)

    # Restore control features
    model.position = PositionControl(model.state, to_device(state_dict['position']), model.basis)
    model.scaling = ScalingControl(model.state, to_device(state_dict['scaling']), model.basis)
    model.rotation = RotationControl(model.state, to_device(state_dict['rotation']), model.basis)
    model.opacity = OpacityControl(model.state, to_device(state_dict['opacity']), model.basis)
    model.sh_dc = SHControl(model.state, to_device(state_dict['sh_dc']), model.basis)
    model.sh_rest = SHControl(model.state, to_device(state_dict['sh_rest']), model.basis)

    # Restore
    model.spherical_harmonics = SHControlWrapper(model.state, model.sh_dc, model.sh_rest)
    # model.spherical_harmonics.sh_dc.name = 'f_dc'
    # model.spherical_harmonics.sh_rest = SHControl(model.state, to_device(state_dict['sh_rest']), model.basis)
    # model.spherical_harmonics.sh_rest.name = 'f_rest'

    # Restore training state
    model.iteration = state_dict.get('iteration', 0)
    model.spatial_lr_scale = state_dict.get('spatial_lr_scale', 1.0)

    # Re-setup training to register parameters with optimizer
    if train_mode and state_dict.get('optimizer') is not None:
        model.training_setup(model.config)
        try:
            model.optimizer.load_state_dict(state_dict['optimizer'])
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}")

    # FORCE CACHE INVALIDATION & PRE-COMPUTE
    # This ensures consistency immediately after load
    _initialize_surface_after_restore(model, device)

    return model


def _restore_knot_data(knot_module, loaded_data, should_optimize):
    """
    Helper to inject loaded data into KnotVector, bypassing standard init if optimizing.
    """
    # Initialize basic properties that __init__ would have done
    # n_internal = len(loaded_data)
    # knot_module._num_control = ... (derived)

    if should_optimize:
        # We loaded RAW logits. Directly create parameter.
        knot_module._internal_knots = torch.nn.Parameter(loaded_data.contiguous(), requires_grad=True)
        knot_module.should_optimize = True
        knot_module._decode = torch.sigmoid
        knot_module._encode = inverse_sigmoid
        # Update control points count based on loaded data
        knot_module._num_control = len(loaded_data) + knot_module.degree + 1
    else:
        # We loaded RAW logits (likely), but we want static mode.
        # OR we loaded static values (0-1).
        # Check range to be safe. If data is outside [0,1] substantially, it's probably logits.

        is_logits = (loaded_data.min() < 0.0) or (loaded_data.max() > 1.0)

        if is_logits:
            final_knots = torch.sigmoid(loaded_data)
        else:
            final_knots = loaded_data

        knot_module.register_buffer('_internal_knots_buffer', final_knots.contiguous())
        knot_module._internal_knots = knot_module._internal_knots_buffer
        knot_module.should_optimize = False
        knot_module._decode = lambda x: x
        knot_module._encode = lambda x: x
        knot_module._num_control = len(final_knots) + knot_module.degree + 1


def create_sampler_from_state(state, uv_state: Dict, device: str, should_optimize: bool):
    """Create appropriate sampler and correctly wrap parameters."""

    mode = uv_state.get('mode', 'single')
    num_channels = uv_state.get('num_channels', 1)

    # Parse intervals - these are currently stored as Tensors
    interval_u_data = parse_interval_dict(uv_state.get('interval_u', {}), device)
    interval_v_data = parse_interval_dict(uv_state.get('interval_v', {}), device)

    # Instantiate

    sampler = SamplerUV(state, num_channels=num_channels, evaluate_mode=not should_optimize)

    # Handle Single View
    u_val = interval_u_data.get('_tensor', interval_u_data)
    v_val = interval_v_data.get('_tensor', interval_v_data)

    if should_optimize:
        sampler._interval_u = torch.nn.Parameter(u_val, requires_grad=True)
        sampler._interval_v = torch.nn.Parameter(v_val, requires_grad=True)
    else:
        # Check logits
        is_logits = (u_val.min() < 0) or (u_val.max() > 1)
        if is_logits:
            sampler._interval_u = torch.sigmoid(u_val)
            sampler._interval_v = torch.sigmoid(v_val)
        else:
            sampler._interval_u = u_val
            sampler._interval_v = v_val

    # Restore visibility probs
    if 'vis_probs' in uv_state and uv_state['vis_probs']:
        sampler.vis_probs = {
            int(k) if k.isdigit() else k: v.to(device)
            for k, v in uv_state['vis_probs'].items()
        }

    return sampler


def parse_interval_dict(d: Dict, device: str) -> Dict:
    """Parse interval dictionary from saved format."""
    if not d:
        return {}

    result = {}
    for k, v in d.items():
        if k == '_tensor':
            return v.to(device)

        key = int(k) if k.isdigit() else k
        if isinstance(v, torch.Tensor):
            result[key] = v.to(device)
        else:
            result[key] = v

    return result


def load_model(path: str, eval_mode: bool = True, device: str = 'cuda'):
    """
    Load model from disk with proper initialization.

    Args:
        path:  Path to saved model
        eval_mode:  If True, set model to evaluation mode
        device: Target device

    Returns:
        MultiSurfaceSplineModel or SplineModel
    """
    # state_dict, i = torch.load(path)
    state_dict, i = torch.load(path,
               weights_only=False,
               map_location='cuda')

    # Handle tuple format (iteration, state_dict)
    if isinstance(state_dict, tuple):
        iteration, state_dict = state_dict

    surfaces = []
    for surf_state in state_dict['surfaces']:
        model = SplineModel(late_init=True)
        model.restore(surf_state, train_model=not eval_mode)
        surfaces.append(model)

    multi_model = MultiSurfaceSplineModel(
        surfaces=surfaces,
        labels=state_dict['labels'],
        decomposition_mode=DecompositionMode(state_dict['decomposition_mode']),
        point_labels=state_dict.get('point_labels'),
        eval=eval_mode,
        setup_training=False
    )

    multi_model._active_surfaces = state_dict['active_surfaces']
    multi_model._surface_weights = state_dict['surface_weights']

    # Update surface offsets
    multi_model._update_surface_offsets()

    if not eval_mode:
        multi_model.training_setup()

    return multi_model


def _initialize_surface_after_restore(model: SplineModel, device: str = 'cuda'):
    """
    Initialize basis functions and caches after restoring from checkpoint.
    """
    model.to(device)
    model.device = device
    model.state.device = device
    if  model.sampling_mode != SamplingMode.ADAPTIVE:

        # 1. Force Sampling Mode (important for get_xyz to pick correct path)
        model.sampling_mode = SamplingMode.EVALUATION if model.uv_sampler.evaluate_mode else SamplingMode.OPTIMIZABLE

        # 2. Get UV grid (will use parameters if optimizable)
        uv_grid = model.uv_sampler()
            # UVshape=(model.state.Us, model.state.Vs)
        # )
        model.invalidate_all_caches()

        # 3. Compute Basis
        model.basis.forward(
            uv_grid,
            model.knot_u(),
            model.knot_v()
        )

        # 4. Invalidate and Compute
        _ = model.get_xyz  # Force computation
        _ = model.get_scaling
        _ = model.get_rotation
        _ = model.get_opacity
        _ = model.get_features
        print(f"[load_model] Initialized surface:  "
              f"H={model.state.H}, W={model.state.W}, "
              f"Us={model.state.Us}, Vs={model.state.Vs}, "
              f"Gaussians={model.state.Us * model.state.Vs}")