# Knot-a-Surface — Fixes Applied (2026-06-10)

All changes verified by CPU unit tests against `geomdl` ground truth
(`tests/test_basis_math.py`, `tests/test_position_rational.py`,
`tests/test_rotation_frame.py` — all PASS) and by full static import of every
touched module with CUDA extensions stubbed (`tests/conftest_stubs.py`).

## Mathematical correctness (these change results)

1. **B-spline basis derivatives were wrong** (`modules/basis/`):
   the live training path (`compute_bases_uv_diff` →
   `DifferentiableBSplineBasis`) computed derivatives via a power-basis
   matrix with (a) a spurious global ×0.5 factor on D and D², (b) no
   span-width chain rule, (c) no Oslo non-uniform correction on derivative
   matrices. **Replaced with the exact triangular-table algorithm**
   (`bspline_basis_and_derivs_1d`, The NURBS Book A2.3) — machine-precision
   match vs geomdl for uniform AND non-uniform knots, differentiable w.r.t.
   samples and knots. `compute_bases_uv` (non-interval path) routed through
   the same function. Deleted finite-difference fallback
   (`basis_functions_and_derivatives`) and dead/broken helpers
   (`compute_uniform_basis`, `update_basis`, `create_basis_from_uv_grid`,
   `DifferentiableBasisModule`).
2. **Rational (NURBS) derivative quotient rule**
   (`modules/control_feature/position.py`): `_deriv_interpolate` returned
   `A'/W`; correct is `(A' − S·W')/W` (first order) and
   `(A'' − 2·S'·W' − S·W'')/W` (second order). Measured error of the old
   formula vs ground truth: up to **8.4** absolute in tangent components.
   Tangents/normals/curvatures/eikonal were all corrupted whenever
   `refine_weights=True` (the default).
3. **Cache detach killed gradients** (`modules/control_feature/base.py`):
   `set_cache` stored `tensor.detach()`, so the first consumer in an
   iteration got gradients and every later consumer (e.g. the min-scale loss
   reading `get_scaling` after render) silently got a constant. Cache now
   keeps the graph; serialization detaches explicitly.
4. **Invalid rotation frames** (`modules/spline_formulas.py::uv_tangent`):
   used un-normalized `tangents_u` as a rotation column and flipped normals
   without flipping a tangent → non-orthonormal matrices and reflections
   (det = −1) → garbage quaternions. Now a Gram-Schmidt frame
   [t̂u, t̂v⊥, n̂] with a handedness-consistent flip; verified orthonormal,
   det=+1, differentiable.
5. **Geometry-derived splats restored** (`modules/KnotSurface.py`):
   `derive_scale` (was: `no_grad` + kNN `distCUDA2`, i.e. vanilla-3DGS init)
   now computes ‖Sᵤ‖·Δu, ‖Sᵥ‖·Δv differentiably (paper Eq. 5);
   `derive_rotation` (was: `no_grad` + detached normals) now uses the fixed
   `uv_tangent` differentiably (Eq. 6). Defaults flipped to
   `residual_scaling = residual_rots = True` (paper formulation: learned
   features are residuals on the geometric base); residual rotation init
   fixed to identity quaternion. Ablation (free attributes) still available
   by setting `residual_* = False`.
6. **Activation order matched to Eq. 7**
   (`modules/control_feature/base.py`): interpolation now happens in raw
   parameter space with activation after (σ(ΣN·x̃)), which also makes
   rendering consistent with Boehm knot insertion (done in raw space).
   ScalingControl's appended normal-axis channel fixed from −9.0 (negative
   scale in activated space) to e⁻⁹.
7. **"Eikonal" loss made scale-invariant** (`modules/KnotSurface.py`):
   penalized (‖Sᵤ×Sᵥ‖−1) — a scene-unit-dependent prior fighting the derived
   scales. Now penalizes deviation of the area element from its mean.
8. **NaN propagation stopped**: `uv_depth` had the always-true
   `depths != torch.nan` comparison; `weights_map` returned 1/NaN for missed
   rays and fed NaNs into the normal/scale losses. Missed rays now get
   weight 0.

## Loss / training-loop correctness

9. **Loss weights were silently ignored** (`utils/loss_utils.py`):
   `cossim_loss_multisurf` and `param_surf_deviation` shadowed the weight
   argument `w` in their `zip(...)` loops — every configured weight acted as
   an on/off switch at strength 1.0. Fixed (and weights actually applied).
10. **Weight maps were destroyed** (`cosine_similarity_loss`,
    `cosine_similarity_geodesic_loss`): `F.normalize(weight_map, dim=-1)` on
    an (H,W,1) map turns every weight into ±1/NaN. Now: NaN-safe,
    mean-normalized multiplicative weighting.
11. **Swapped weights** in `process_view`: `normal_smoothness` got the
    global weight and vice versa. Fixed.
12. **Gradient accumulation**: per-view `backward()` summed gradients →
    effective LR ∝ batch size, AND `BATCH_SIZE` decayed by 1 every 1k iters
    → an undocumented LR schedule. Per-view loss now scaled by 1/B; decay
    removed.
13. **Camera ordering**: batched mode consumed cameras in fixed
    forward/reverse order (random choice commented out) → correlated
    batches. Now shuffled per epoch.
14. **LR schedule off-by-one**: `update_learning_rate(iteration)` ran after
    `optimizer.step()`. Moved before the forward pass.
15. `local_planar_deviation_loss` (multisurf) returned inside its loop —
    only surface 0 contributed. Fixed.
16. `_invalidate_cache` early-returned inside the per-surface loop —
    order-dependent stale caches. Fixed.
17. `is_pruned`/`is_split` labels in `subdivide_and_cull` were swapped
    (cosmetic). `SHControl.forward` never set `cache_valid` (always
    re-interpolated). Fixed.

## Showstoppers / API bugs

18. Checkpoint restore was a no-op (loaded then discarded) — now calls
    `nurbs.restore(...)`. `gpu_id` could be undefined → defaults to 0.
    `run_evaluation` returned `None` on the render-only path → KeyError in
    the callback; callback also hardened.
19. `prepare_img_log` returned inside its loop (only first image logged).
    `axs[i].sampling_grid(...)` → `.grid(...)` (3 sites + tnt plot script) —
    bad global rename. `evaluation/checkpoint/save_iterations` aliased one
    list object. `--use_wandb`/`--include_eval` were force-overridden after
    parsing — CLI now honored (**pass `--include_eval --use_wandb`
    explicitly on the cluster**). `--seed` now actually seeds.
    `set_optimal_path` fallback passed an einsum string as an opt_einsum
    strategy (KeyError). `wandb.save` pointed at deleted `model/` paths.
20. Removed dead code: duplicate trailing `setup_batched_optimizer` /
    `build_controller` (the live duplicates used `getattr` on a dict —
    always returned defaults), shadowed local `plot_render_outputs` (the
    real one lives in `utils/image_utils.py`), commented
    `log_qualitative_results2` block. **Dropped the `pytorch3d` dependency**
    from live modules (local `matrix_to_quaternion`/`quaternion_to_matrix`
    in `modules/control_feature/quaternion_utils.py`).

## Evaluation hygiene

21. Held-out evaluation (`log_qualitative_results`) was only reachable while
    `iteration < densify_until_iter` — moved out; test PSNR now reported for
    the entire run.
22. Periodic matplotlib diagnostics called `plt.show()` every 500 iterations
    (blocking; figures never closed). Now opt-in via `--show_plots`, saves
    PNGs to the debug dir, interactive only under a debugger, and always
    closes figures.

## Performance

23. `dSu/dSv/dSuu/dSvv` now cache-guarded (were recomputed on every access).
24. Cached pixel grid + per-camera CUDA R/T tensors
    (`utils/graphics_utils.py::get_pixel_grid/get_cam_RT_cuda`) replace
    per-iteration `meshgrid` + CPU→GPU `torch.tensor(cam.R).cuda()`.
25. SSIM Gaussian window cached per (size, channels, device, dtype).
26. Basis: one exact O(p²) pass replaces matrix-form + FD paths.

## Still open (deliberate, needs your call / a GPU)

- **Checkpoint selection by test-set Chamfer** (`run_evaluation` keeps the
  best-CD model): for publication, report a fixed final iteration or select
  on held-out *validation* views — reviewers will flag test-set selection.
- Densification statistics see only the last view of each batch
  (`process_batch_views` frees the others). Matters only when
  `batch_size > 1`.
- `modules/tessellation/chhugani_v2.py` imports nonexistent
  `model.modules.basis` (dead module; v1 is the live one).
- Cluster smoke test: 200–500 iterations on one DTU scan to confirm losses
  decrease and no NaNs — CUDA rasterizer not available locally.
- The custom CUDA rasterizer with B-spline forward/backward: now unblocked —
  the exact basis/derivative functions these Jacobians need are tested.
