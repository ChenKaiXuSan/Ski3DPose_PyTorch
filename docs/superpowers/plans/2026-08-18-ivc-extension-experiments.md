# IVC Extension Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build reproducible view-angle, sampling-rate, front-end generalization, and masking-summary experiment entry points for the IVC extension.

**Architecture:** Pure experiment logic is centralized in a small helper module and exercised by unit tests. Four scripts reuse the repository's existing Unity evaluation stack and write explicit CSV/JSON/figure artifacts without altering the CrossViewFusion architecture.

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, Hydra/OmegaConf, NumPy, Matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-08-18-ivc-extension-experiments-design.md`

## Global Constraints

- Preserve the current checkpoint and `fold_00` evaluation protocol by default.
- Do not download or run third-party pose-estimator code.
- Alternative front-end inputs use validated local `.npy`/`.npz` manifests.
- Do not smooth or fabricate uncertainty for the existing one-run masking data.
- Preserve all pre-existing uncommitted files and results.

---

### Task 1: Shared experiment helpers and view-angle evaluation

**Files:**
- Create: `dual2pose/eval/extension_experiment_utils.py`
- Create: `dual2pose/eval/eval_unity_view_angle.py`
- Modify: `dual2pose/trainer/train_crossview_fusion.py`
- Test: `tests/test_ivc_extension_experiments.py`

**Interfaces:**
- Produces: `parse_unity_camera_id(str) -> tuple[int, float]`, `circular_angle_distance(float, float) -> float`, `assign_angle_bin(float, Sequence[float]) -> str`, `summarize_outputs_by_angle(list[dict], float, Sequence[float]) -> list[dict]`.
- Adds `meta` to each test output without changing existing tensor keys.

- [x] Write failing tests for camera parsing, circular separation, bin boundaries, and metadata-aware grouped metrics.
- [x] Run `python3 -m unittest tests.test_ivc_extension_experiments.ViewAngleTest -v` and confirm missing-symbol failures.
- [x] Implement the pure helpers and metadata propagation.
- [x] Implement the Hydra entry point and CSV/JSON output.
- [x] Re-run the focused tests and confirm they pass.

### Task 2: Sampling-rate drift evaluation

**Files:**
- Modify: `dual2pose/eval/extension_experiment_utils.py`
- Create: `dual2pose/eval/eval_unity_sampling_rate.py`
- Modify: `tests/test_ivc_extension_experiments.py`

**Interfaces:**
- Produces: `resample_pose_rate(pose, rate_error, anchor='center') -> Tensor` and a datamodule that applies it to left, right, or both streams.

- [x] Write failing tests with literal expected samples for positive and negative drift and invalid rates.
- [x] Run the focused sampling-rate tests and confirm missing-symbol failures.
- [x] Implement center-anchored linear resampling and the wrapped datamodule.
- [x] Implement summary CSV/JSON generation including relative degradation from zero drift.
- [x] Re-run the focused tests and confirm they pass.

### Task 3: Front-end estimator manifest adapter

**Files:**
- Modify: `dual2pose/eval/extension_experiment_utils.py`
- Create: `dual2pose/eval/eval_unity_frontend_generalization.py`
- Modify: `tests/test_ivc_extension_experiments.py`

**Interfaces:**
- Produces: `FrontEndManifest.load(Path)`, `FrontEndPoseDataset`, and `replace_frontend_inputs(sample, manifest) -> dict`.
- Consumes sequence arrays shaped `T x J x 3`, with optional `.npz` key `pose` and optional manifest `joint_indices`.

- [x] Write failing tests for valid replacement, duplicate entries, missing coverage, shape mismatch, and non-finite arrays.
- [x] Run the focused front-end tests and confirm missing-symbol failures.
- [x] Implement manifest parsing, array loading, uniform temporal resampling, and strict validation.
- [x] Implement the SAM3D baseline/no-manifest path and comparison summary artifacts.
- [x] Re-run focused tests and confirm they pass.

### Task 4: Masking result publication summary

**Files:**
- Modify: `dual2pose/eval/extension_experiment_utils.py`
- Create: `dual2pose/eval/summarize_unity_masking.py`
- Modify: `tests/test_ivc_extension_experiments.py`
- Modify: `paper/ivc_extension_experiments.md`

**Interfaces:**
- Produces: `summarize_masking_rows(rows, selected_ratios) -> tuple[list[dict], list[dict]]` and a CLI that writes selected points, AUC, trend, figure, and Markdown artifacts.

- [x] Write failing tests using a hand-derived two-curve CSV fixture for degradation and trapezoidal AUC.
- [x] Run the focused masking-summary tests and confirm missing-symbol failures.
- [x] Implement deterministic aggregation and plotting.
- [x] Document all four experiment commands, inputs, and outputs in the IVC plan.
- [x] Re-run focused tests and confirm they pass.

### Task 5: Integrated verification

**Files:**
- Test: `tests/test_ivc_extension_experiments.py`
- Test: `tests/test_temporal_offset_eval.py`

**Interfaces:**
- Consumes all prior task outputs; produces verification evidence only.

- [x] Run `python3 -m unittest tests.test_ivc_extension_experiments tests.test_temporal_offset_eval -v`.
- [x] Run `python3 -m py_compile` on every new or modified Python module.
- [x] Run each new module with `--help` or a summary-only fixture command where available.
- [x] Inspect `git diff --check` and `git status --short` without modifying unrelated changes.
