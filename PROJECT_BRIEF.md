# Knot-a-Surface — Session Warm-Up Brief

Paste-ready context for starting a new working session on this repo.

## What the project is

**Knot-a-Surface** (MSc thesis, targeting a top-tier venue): inverts 3D
Gaussian Splatting by replacing the unstructured point cloud with a
**single learnable B-spline surface**. Control points (and optionally NURBS
weights and knots) are the trainable parameters; Gaussian splat attributes
are **derived from the surface's differential geometry** — positions from
S(u,v), scales from ‖Sᵤ‖·Δu, ‖Sᵥ‖·Δv (Eq. 5), rotations from the tangent
frame [t̂u, t̂v⊥, n̂] (Eq. 6), attributes interpolated via basis functions
(Eq. 7). Rendering uses the PGSR planar rasterizer
(`diff_plane_rasterization`). Output: photorealistic renders + a compact,
editable, watertight parametric surface (DTU Chamfer benchmark).

## Architecture (live code path)

```
optimize_nurbs.py                    training loop, losses, wandb, eval
└─ spline_scene/  (SplineScene)      COLMAP load → fit_initial_surface
   └─ modules/fitting/simple_init.py one-knob init: MBA fit + Chamfer post-fit
      └─ modules/fitting/mba.py      Multilevel B-spline Approximation
   └─ modules/KnotSurface.py         SplineModel — THE model (single surface)
      ├─ control_feature/            position/scaling/rotation/opacity/SH/weights
      │    base.py: activate-then-interpolate, fused CUDA eval, NO caching
      ├─ basis/                      exact triangular-table basis (tested vs geomdl)
      ├─ sampling/SamplerUV.py       UV samples (margin-bounded sigmoid,
      │                              visibility-driven adaptive redistribution)
      └─ knotvector/                 clamped knots (margin-bounded sigmoid)
gaussian_renderer/                   PGSR render wrapper
submodules/                          simple-knn | diff-plane-rasterization |
                                     bspline-eval (custom fused kernel)
attic/                               retired code (multisurf, decompose, …)
```

Key design decisions in force:
- **Single surface** — `MultiSurfaceSplineModel` is retired to `attic/`.
- **No lazy caching** — every access recomputes; the fused CUDA kernel
  (`bspline_eval`: S, Sᵤ, Sᵥ, Sᵤᵤ, Sᵥᵥ in one local-support pass, exact
  backward, 30× einsum) makes recomputation cheap. einsum is CPU fallback.
- **Activate-then-interpolate** (Σ N·σ(x̃)); rotation renormalizes after
  interpolation; one activation per attribute, applied exactly once.
- **MBA initialization** (Lee–Wolberg–Shin): coarse-to-fine residual
  fitting, closed-form local updates, Greville collocation projection.
  One knob: `init_quality` ∈ raw/fast/balanced/fine (128/128/192/256 grid).
- **Margin-bounded sigmoids** keep UV samples and internal knots strictly
  inside (0,1) — no saturation freeze, no degenerate boundary basis rows.
- **Adaptive tessellation**: per-sample visibility accumulated from
  `out_observe`; every `resample_every` iters samples are re-placed by
  inverse-CDF of visibility marginals (density floor 0.2, same budget).
- **Subdivision = knot insertion** (whole rows/cols) with Adam-state
  splicing; knot-multiplicity guard prevents repeated same-val insertion
  (was the source of 0/0→NaN Boehm alphas).

## Environments

- **Local Mac (M2 Pro)**: `.venv`, CPU torch — unit tests only
  (`tests/test_*.py`, CUDA stubs in `tests/conftest_stubs.py`).
- **GPU server**: `ssh user@40.142.110.216` (RTX 3090), conda env
  `ml_env`, repo at `~/Knots`, DTU at `~/datasets/DTU` (15 scans,
  2DGS-preprocessed, masks included), eval GT at
  `~/datasets/dtu_eval/MVSDATA`. Server has NO GitHub auth — sync via
  `git bundle` over scp. Fresh env: `bash setup_conda_env.sh` (encodes the
  CUDA-version-pinning lessons; see header comments).
- **GitHub**: `TzlilO/Knots`, branch `fix/formulation-and-training-correctness`
  ≡ `main` (kept in lockstep).

Run command:
```bash
SCAN_ID=scan24 CUDA_VISIBLE_DEVICES=0 python optimize_nurbs.py \
  -s ~/datasets/DTU -m ~/output_dtu/<run> -r 2 --ncc_scale 0.5 \
  --use_wandb [--eval] [--include_eval]
```
`SCAN_ID` env var is appended to both `-s` and `-m`. ~35 it/s on the 3090
(full 30k ≈ 15 min). Eval snapshots (render|GT PNGs) land in
`eval_vis/<scan>/` every 500 iters. wandb: project NURBS, entity
sw7gynvgmn-hebrew-university-of-jerusalem.

## Current state (as of 2026-06-11)

- Core math validated: basis/derivatives vs geomdl to machine precision,
  rational quotient rule, rotation frames (det=+1), fused kernel forward/
  backward to fp32 rounding. CPU suite: `tests/test_*.py` all pass.
- MBA init on scan24: 206×178 grid, init dense-CD 0.113→0.056 after
  Chamfer post-fit; PSNR 14.2 @ 500 → 18.3 @ 1500 iters.
- Current config (`arguments/nurbs_params.py`) is an **ablation setting**:
  `residual_scaling/rots=False` (free attributes, geometric init only),
  `refine_weights=False` (polynomial B-spline, not NURBS),
  `optimize_knots=False`, `optimize_intervals=False`, eikonal/planar
  weights 0. The paper formulation = flipping residual_* to True.

## Open / not done

1. **Full 30k diagnostic run** (`scan24_full30k_v2`): launched to answer
   "PSNR used to be 40" — verify the converged number; if it plateaus low,
   suspects are the ablation toggles, gaussian budget growth, or the
   activate-then-interpolate switch.
2. **No converged DTU Chamfer number yet** on the rebuilt codebase — run
   with `--include_eval` and compare against PGSR's 0.47 table.
3. **Checkpoint selection uses test-set Chamfer** (`run_evaluation` keeps
   best-CD) — publication risk; switch to fixed-final-iter or validation
   views before submitting numbers.
4. **Batch>1 densification stats** see only the last view of each batch.
5. **Full in-rasterizer B-spline fusion**: the fused eval kernel exists
   and is integrated *before* the rasterizer; folding it INTO the
   rasterizer's preprocess (ctrl-points → pixels in one graph) is designed
   but not built.
6. **Hyperparameter sweep**: config + agent script exist
   (`configs/sweep_nurbs.yaml`, `scripts/run_sweep_agent.sh`); the agent
   is not currently running.
7. Known warts: per-span densification scores depend on camera coverage;
   `freeze_uv_iter` would reinterpret raw sampler params if it ever fires
   (dormant at 30k); scaling insertion uses pre-insertion positions.

## Gotchas that cost time before (don't rediscover)

- Conda CUDA packages MUST come from `nvidia/label/cuda-X.Y.Z` (bare
  `nvidia` channel mixes versions → cub/cccl header chaos).
- Extension builds need `CUDA_HOME=$CONDA_PREFIX` + `--no-build-isolation`.
- `cuda.Event.elapsed_time` needs `iter_end.synchronize()` first.
- `distCUDA2` returns SQUARED distances — take sqrt.
- Optimizer param groups are name-keyed (`opacity_0`, `xyz_0`, …) — match
  via `module.name`, never literals.
- The SSH key needs `ssh-add` after Mac sleep; server auth then works.
