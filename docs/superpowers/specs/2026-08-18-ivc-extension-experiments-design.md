# IVC Extension Experiments Design

## Scope

Add four reproducible experiment paths around the existing Unity dual-view
evaluation stack:

1. Camera-pair azimuth separation analysis.
2. Sampling-rate drift sensitivity analysis.
3. Pluggable front-end estimator generalization evaluation.
4. Publication-oriented summarization of the completed masking sweep.

The work must preserve the current CrossViewFusion checkpoint and test split,
must not redesign the model, and must not download or execute a new third-party
pose estimator. Alternative front-end predictions are supplied through a
validated manifest when they become available.

## Architecture

Shared, pure helpers live in `dual2pose/eval/extension_experiment_utils.py` so
camera parsing, interpolation, manifest validation, and tabular aggregation can
be tested without loading a checkpoint. Four focused command-line entry points
reuse the existing Unity datamodule, model builder, trainer builder, and metric
functions from `eval_unity_masking.py`.

The view-angle evaluator retains batch metadata in test outputs, computes the
circular azimuth separation `min(|a-b|, 360-|a-b|)`, and reports per-bin MPJPE,
velocity error, acceleration error, failure rate, sample count, and fusion gain
over canonical averaging. Default bins are `[0,30)`, `[30,60)`, `[60,90)`,
`[90,120)`, `[120,150)`, and `[150,181)` degrees.

The sampling-rate evaluator perturbs only one view. For a fractional rate error
`r`, it samples the original sequence at
`center + (t-center)/(1+r)`. Anchoring at the sequence center isolates clock-rate
drift from a constant temporal offset. Linear interpolation and edge clamping
match the existing sub-frame offset study. Default errors are -2%, -1%, -0.5%,
0%, +0.5%, +1%, and +2%.

The front-end evaluator accepts a JSON manifest. Each entry identifies
`person_id`, `action_id`, `camera_id`, and a `.npy` or `.npz` pose sequence of
shape `T x J x 3`. Optional top-level `joint_indices` reorder/select joints.
An evaluation dataset wrapper replaces `kpt3d_sam` per sample, uniformly
resamples predictions to the base sequence length, and refuses incomplete,
duplicate, non-finite, or shape-incompatible manifests. With no manifest, the
existing SAM3D stream is evaluated as the baseline.

The masking summarizer consumes an existing `occlusion_summary_*.csv` and emits
a selected-ratio table, a per-pattern/view robustness-AUC table, a long-form
trend CSV, a PNG/PDF figure, and a Markdown report. It never smooths, invents
replicates, or adds error bars to the existing single-run sweep.

## Reproducibility and Safety

- Every output records checkpoint, input path, fold, seed, settings, and units.
- Commands expose environment-variable overrides consistent with the existing
  evaluation scripts.
- Pure logic is developed test-first; full checkpoint evaluation is separate
  from unit verification.
- Existing uncommitted files and logged results are preserved.
- No external data, model, or unpublished artifact is uploaded.

## Acceptance Criteria

- Camera IDs from the Unity index map are parsed and binned correctly across
  the 0/360-degree boundary.
- Positive and negative sampling-rate errors produce hand-checkable linear-ramp
  resampling around the center anchor.
- Front-end manifests reject missing coverage and malformed arrays before model
  evaluation.
- Masking summaries are deterministic and expose selected degradation values
  and trapezoidal AUC derived from fixture data.
- New and existing temporal-offset tests pass together.

