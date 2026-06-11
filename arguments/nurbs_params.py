from math import inf

from arguments import OptimizationParams

class NurbsOptimizationParams(OptimizationParams):

    def __init__(self, parser):
        self.device = 'cuda'


        self.scaling_reset_factor = 0.5
        self.nurbs_weight_lr = 0.01
        self.background_lr_scale_factor = 1.0
        self.pe_lr_factor = 1.0
        self.batch_size = 1
        self.uv_lr_factor = 1e-4
        self.knot_lr = 1e-4

        # --- Weights for loss terms ---
        self.lambda_eikonal = 0.0
        self.eikonal_from_iter = 0
        self.quat_smoothness_weight = 0.0
        self.scale_deviation_weight = 0.0
        self.scale_smoothness_weight = 0.0
        self.normal_smoothness_weight = 0.0
        self.normal_global_smoothness_weight = 0.0
        self.normal_dev_weight = 0.0
        self.chamfer_weight = 0.0
        self.cossim_weight = 0.0
        self.kl_div_weight = 0.0

        # --- Stage thresholds for loss activation ---
        self.scale_consistency_from = 0
        self.quat_smooth_from = 0
        self.normal_smooth_from = 0
        self.normal_dev_from = 0
        self.local_planar_deviation_weight = 0.
        self.local_planar_deviation_from = 70000


        self.use_pe_sampling = False
        self.pe_num_frequencies = 2
        self.pe_lr_base = 0.01
        self.use_pe = False  # Enable PE mode
        self.pe_lr_scale = 0.01
        self.pe_log_sampling = False
        self.pe_max_freq = 8  # Enable PE mode
        self.pe_levels = 4  # Number of frequency levels
        self.pe_include_input = False  # Include identity term
        self.pe_learnable_freqs = False  # Make frequencies learnable
        self.discrepancy_from_iter = 70_000
        self.lambda_uv_consistency = 0.0
        self.uv_update_interval = inf
        self.set_uv_adaptive_from = 1000000
        self.set_uv_optimizable_from = inf
        self.random_sampling = False
        # Adaptive tessellation: periodically re-place UV samples by
        # accumulated visibility (same sample budget, no optimizer surgery).
        self.adaptive_sampling = True
        self.resample_start = 1000
        self.resample_every = 500

        # self.sampling_strategy = None
        self.freeze_uv_iter = 160_000
        self.subdiv_critertia = 'residual' #'hybrid'
        self.sampling_strategy = 'static' #'hybrid'
        self.decomposition_mode = 'single'
        self.encode_points = 'spherical' # 'geodesic' #'spherical' 'geodesic', 'pca
        # Initial-surface estimation: ONE knob — 'fast' | 'balanced' | 'fine'
        # | 'raw' (raw = LS fit only, no Chamfer post-fit). Resolution, LS
        # smoothing and post-fit budget are derived from the data; see
        # modules/fitting/simple_init.py. The legacy post_fit_* knobs below
        # no longer drive the live init path.
        self.post_fit_iterations = 500
        self.post_fit_enabled = False
        self.base_res = 256
        self.max_res = 512
        self.min_res = 128
        self.spline_degree = [3, 3]
        self.max_k_subdiv = 0.025
        self.max_k_prune = 0.0
        self.max_k_prune_vis = 0.0

        # Spline Model Parameters
        self.target_density_per_unit = 1
        self.sampling_density = 1.0
        self.refine_scales = True
        self.refine_rotations = True
        self.refine_opacities = True
        self.refine_weights = True
        self.optimize_intervals = False
        self.optimize_knots = False

        # Residual mode = the paper's formulation: splat scale/rotation are
        # DERIVED differentiably from surface tangents (Eqs. 5-6), with the
        # learned control features acting as a multiplicative/compositional
        # residual on top. With residual_* = False the geometry only
        # initializes free Gaussian attributes (ablation mode).
        self.residual_scaling = False and self.refine_scales
        self.residual_rots = False and self.refine_rotations
        self.residual_opacity = False and self.refine_opacities

        self.use_spatial_partitioning = False  # For subdivision
        self.use_spatial_partitioning_prune = False  # For pruning
        self.num_partitions = 8  # Auto-adjust based on grid size
        self.num_partitions_prune = 4  # Different for pruning
        super().__init__(parser)


# class NurbsOptimizationParams2(OptimizationParams):
#     """
#     YAML-backed NURBS optimization parameters with argparse compatibility.
#
#     Adds a single CLI flag:
#         --nurbs_config <path.yaml>   (scene-specific override YAML)
#
#     Individual params can still be overridden via CLI:
#         --nurbs_weight_lr 0.002
#
#     All NURBS-specific defaults live in configs/nurbs_default.yaml.
#     """
#
#     def __init__(self, parser: ArgumentParser):
#         # --- Load YAML defaults into self.* so ParamGroup registers them ---
#         self._nurbs_cfg = NurbsConfig()
#
#         # Expose every leaf as self.<leaf_name> for ParamGroup auto-registration
#         for dotted_key, value in self._nurbs_cfg.to_flat_dict().items():
#             leaf = dotted_key.rsplit(".", 1)[-1]
#             # Skip keys already defined by OptimizationParams (they'll be set by super())
#             if not hasattr(self, leaf):
#                 setattr(self, leaf, value)
#
#         # Add the --nurbs_config flag itself
#         # (We inject it after super().__init__ so it doesn't collide)
#         self._parser = parser
#
#         # Call parent — this registers all self.* as argparse arguments
#         super().__init__(parser)
#
#         # Now add the config path flag
#         parser.add_argument(
#             "--nurbs_config",
#             type=str,
#             default=None,
#             help=os.getenv('NURBS_CONFIG', "Path to scene-specific NURBS config YAML (overrides defaults)"),
#         )
#
#     def extract(self, args) -> GroupParams:
#         """
#         Override extract to:
#         1. Re-load NurbsConfig with the scene YAML if --nurbs_config was given.
#         2. Merge CLI overrides on top.
#         3. Return a GroupParams that has all flat attributes.
#         """
#         # Get the base extraction (handles OptimizationParams fields)
#         group = super().extract(args) ,
#
#         # Collect CLI overrides: any parsed arg that differs from the YAML default
#         cli_overrides: Dict[str, Any] = {}
#         flat_defaults = self._nurbs_cfg.to_flat_dict()
#         for dotted_key, default_val in flat_defaults.items():
#             leaf = dotted_key.rsplit(".", 1)[-1]
#             cli_val = getattr(args, leaf, None)
#             if cli_val is not None and cli_val != default_val:
#                 cli_overrides[dotted_key] = cli_val
#
#         # Rebuild config with scene YAML + CLI overrides
#         scene_yaml = getattr(args, "nurbs_config", None)
#         final_cfg = NurbsConfig(
#             config_path=scene_yaml,
#             cli_overrides=cli_overrides,
#         )
#
#         # Stamp all final values onto the GroupParams
#         for dotted_key, value in final_cfg.to_flat_dict().items():
#             leaf = dotted_key.rsplit(".", 1)[-1]
#             setattr(group, leaf, value)
#
#
#         # Attach the full config for anyone who wants structured access
#         group._nurbs_config = final_cfg
#
#         return group