# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
#
# # This software is free for non-commercial, research and evaluation use
# # under the terms of the LICENSE.md file.
# # For inquiries contact  george.drettakis@inria.fr
#
#
# from argparse import ArgumentParser, Namespace
# import sys
# import os
#
# class GroupParams:
#     pass
#
# class ParamGroup:
#     def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
#         group = parser.add_argument_group(name)
#         for key, value in vars(self).items():
#             shorthand = False
#             if key.startswith("_"):
#                 shorthand = True
#                 key = key[1:]
#             t = type(value)
#             value = value if not fill_none else None
#             if shorthand:
#                 if t == bool:
#                     group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
#                 else:
#                     group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
#             else:
#                 if t == bool:
#                     group.add_argument("--" + key, default=value, action="store_true")
#                 else:
#                     group.add_argument("--" + key, default=value, type=t)
#
#     def extract(self, args):
#         group = GroupParams()
#         for arg in vars(args).items():
#             if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
#                 setattr(group, arg[0], arg[1])
#         return group
# class ModelParams(ParamGroup):
#     def __init__(self, parser, sentinel=False):
#         self.sh_degree = 3
#         self._source_path = ""
#         self._model_path = ""
#         self._images = "images"
#         self._resolution = -1
#         self._white_background = False
#         self.data_device = "cuda"
#         self.eval = False
#         self.preload_img = True
#         self.ncc_scale = 1.0
#         self.multi_view_num = 8
#         self.multi_view_max_angle = 30
#         self.multi_view_min_dis = 0.01
#         self.multi_view_max_dis = 1.5
#         super().__init__(parser, "Loading Parameters", sentinel)
#
#     def extract(self, args):
#         g = super().extract(args)
#         g.source_path = os.path.abspath(g.source_path)
#         return g
#
# class PipelineParams(ParamGroup):
#     def __init__(self, parser):
#         self.convert_SHs_python = False
#         self.compute_cov3D_python = False
#         self.debug = False
#         super().__init__(parser, "Pipeline Parameters")
#
#
# class OptimizationParams(ParamGroup):
#     def __init__(self, parser):
#         self.iterations = 30_000
#         self.position_lr_init = 0.00016
#         self.position_lr_final = 0.0000016
#         self.position_lr_delay_mult = 1.
#         self.position_lr_max_steps = 1
#         self.feature_lr = 0.0025
#         self.opacity_lr = 0.025
#         self.scaling_lr = 0.005
#         self.rotation_lr = 0.001
#         self.percent_dense = 0.001
#         self.lambda_dssim = 0.2
#         self.densification_interval = 100
#         self.opacity_reset_interval = 3000
#         self.densify_from_iter = 500
#         self.densify_until_iter = 15_000
#         self.densify_grad_threshold = 0.0002
#         self.scale_loss_weight = 100.0
#
#         self.wo_image_weight = False
#         self.single_view_weight = 0.015
#         # self.single_view_weight_from_iter = 10
#         self.single_view_weight_from_iter = 7000
#
#         self.use_virtul_cam = False
#         self.virtul_cam_prob = 0.5
#         self.use_multi_view_trim = True
#         self.multi_view_ncc_weight = 0.15
#         self.multi_view_geo_weight = 0.03
#         # self.multi_view_weight_from_iter = 10
#         self.multi_view_weight_from_iter = 7000
#         # self.manifold_consistency_weight = 0.05
#         self.multi_view_patch_size = 3
#         self.multi_view_sample_num = 102400
#         self.multi_view_pixel_noise_th = 1.0
#         self.wo_use_geo_occ_aware = False
#
#         self.opacity_cull_threshold = 0.005
#         self.densify_abs_grad_threshold = 0.0008
#         self.abs_split_radii2D_threshold = 20
#         self.max_abs_split_points = 50_000
#         self.max_all_points = 6000_000
#         self.exposure_compensation = False
#         self.random_background = False
#         super().__init__(parser, "Optimization Parameters")
#
# class NurbsOptimizationParams(OptimizationParams):
#     def __init__(self, parser):
#         self.use_analytic_curvature = True
#         self.nurbs_weight_lr = 0.005
#         self.uv_grid_lr = 0.00005
#         self.knot_lr = 0.0005
#         self.batch_until_iter = 1200000
#         self.batch_size = 1
#         self.uv_update_every = 1
#
#         self.refine_knot_from = 300000
#         self.refine_knot_until = 30000000
#
#         # --- Weights for loss terms ---
#         self.scaling_init_factor = .5
#         self.lambda_eikonal = 1e-3
#         self.lambda_arap = 1e-3
#         self.lambda_q2q = 1e-3
#         self.lambda_quat = 1e-3
#         self.lambda_s2s = 1e-3
#         self.lambda_depth_discrepancy = (0.0, 1e-3)   # (Chamfer Distance, Cossim)
#         self.lambda_multiview_geo = 0.2
#
#         # --- Stage thresholds for loss activation ---
#         self.eikonal_from_iter = 90000
#         self.refine_opacity_from = 1
#         self.refine_scaling_from = 100
#         self.refine_weights_from = 70_000
#         self.refine_rotations_from = 7_000
#         self.entropy_patch_size = 3
#         self.lambda_opacity_entropy = 0.0001
#         self.opacity_curvature_sensitivity = 5.0
#         self.opacity_curvature_offset = 3.
#         self.opacity_set_value = 0.01
#         self.opacity_base_level = .0
#         self.use_hybrid_sampler = True
#         self.num_random_views = 1
#         self.camera_dropout = .1
#         self.spline_degree = [3, 3]
#         self.use_pos_enc = False
#         self.pe_scale =  1.0
#         self.pe_levels =  12
#
#         self.lambda_grid_consistency   = 1e-3
#         self.lambda_orientation_barrier   = 1e-2
#         self.consistency_from = 3000000
#         self.orientation_barrier_from   = 300000
#
#
#         # Spline Model Parameters
#         self.patch_density = 2
#         self.grid_densification_factor = 20
#
#         self.device = 'cuda'
#
#         self.refine_scales = False
#         self.refine_rotations = False
#         self.refine_opacities = False
#         self.refine_weights = False
#         self.fix_rotation = False
#         self.cull_backfaces = False
#
#         self.pe_basis = 'descending'
#         # ---------- Dense-manifold NCC loss -------------------------------------------------
#         self.lambda_dense_manifold_ncc        = 0.0
#         self.lambda_features_consistency      = 1e-3
#         self.dense_manifold_ncc_from_iter     = 700000
#         self.dense_manifold_ncc_patch_size    = 3
#         self.dense_manifold_ncc_num_samples   = 102_400
#
#         # ---------- Grid-consistency loss ---------------------------------------------------
#         self.lambda_normal_consistency    = .0
#         self.lambda_opacity_consistency   = .0
#         self.lambda_rots_consistency      = .0
#         self.lambda_sh_consistency        = .0
#         self.grid_consistency_patch_size  = 3
#         self.grid_consistency_from   = 1000000
#
#         self.continuous_resolution_from = 100000
#         self.use_multi_resolution = 100000
#         self.use_multi_resolution_from = 100000
#         self.continuous_resolution = False
#         self.num_resolution_levels = 3
#         self.resolution_scale_factor = 1.5
#         self.lod_policy = 'nyquist' #'complexity'
#         self.enable_multi_res_from = 1
#
#         self.lod_distance_thresholds = None
#         self.lod_scale_thresholds = None #, [2.16, 2.23]
#         self.lod_area_thresholds = None #, [2.16, 2.23]
#         self.lod_complexity_thresholds = None #, [2.16, 2.23]
#
#         self.dynamic_resolution = False
#         self.resolution_update_every = 500
#         self.nyquist_to_res_scale = 200.0
#         self.min_resolution = 8
#         self.max_resolution = 64
#         # ---------- MLP-Spline Parameters ---------------------------------------------------
#         self.use_mlp_interpolation = False
#         self.use_mlp_basis_from = 3000
#         self.mlp_pretrain_until = 2000
#         self.mlp_hidden_dim = 64
#         self.mlp_num_layers = 4
#         self.lambda_mlp_basis_loss = 1.0
#         super().__init__(parser)
#
# def get_combined_args(parser : ArgumentParser):
#     cmdlne_string = sys.argv[1:]
#     cfgfile_string = "Namespace()"
#     args_cmdline = parser.parse_args(cmdlne_string)
#
#     try:
#         cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
#         print("Looking for config file in", cfgfilepath)
#         with open(cfgfilepath) as cfg_file:
#             print("Config file found: {}".format(cfgfilepath))
#             cfgfile_string = cfg_file.read()
#     except TypeError:
#         print("Config file not found at")
#         pass
#     args_cfgfile = eval(cfgfile_string)
#
#     merged_dict = vars(args_cfgfile).copy()
#     for k,v in vars(args_cmdline).items():
#         if v != None:
#             merged_dict[k] = v
#     return Namespace(**merged_dict)