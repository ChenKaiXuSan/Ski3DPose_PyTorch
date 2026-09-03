# IVC P1 Automatic Temporal Alignment Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: ivc_p1_temporal_alignment_v1

## Experiment Overview

- Objective: estimate and correct small camera offsets using pose streams only,
  without camera calibration or ground-truth timestamps.
- Hypothesis: validation-gated velocity correlation recovers short offsets and
  approaches oracle-corrected MPJPE on sufficiently dynamic sequences without
  degrading synchronized inputs.
- Type: controlled perturbation and deterministic alignment evaluation.

## Inputs and Perturbations

- Use the full 64,440-sample fold-0 test split and the MMSports checkpoint.
- Inject right-stream offsets at `-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5`
  frames using the established clamped linear interpolation convention.
- Use the fold-0 validation split only to choose the confidence threshold.
- Search candidate corrections from -5 to +5 frames in 0.5-frame increments.

## Estimator

1. Canonicalize left and right streams with the same function used by the
   trainer.
2. Compute first-order joint velocities and remove the per-frame across-joint
   mean to suppress residual translation.
3. Normalize the flattened velocity descriptor for each valid candidate
   overlap.
4. Score every candidate lag with cosine cross-correlation over only frames
   valid under that shift.
5. Select the highest-scoring lag. Apply it only if its improvement over the
   zero-lag score exceeds a threshold selected on validation data; otherwise
   return zero.
6. Correct the right stream before CanonFuse3D inference.

The method does not use Unity ground truth. The oracle condition applies the
known inverse injected offset and is reported only as an upper bound.

## Compared Conditions

- `uncorrected`: injected offset with no alignment.
- `automatic`: validation-gated estimated correction.
- `oracle`: known inverse offset.
- `zero_reference`: unperturbed synchronized input.

## Metrics and Analysis

- Offset mean absolute error, signed bias, exact-within-0.5-frame accuracy, and
  correction activation rate.
- MPJPE, acceleration error, gate-error correlation, and view-preference
  accuracy for all three correction conditions.
- Stratify offset-estimation and pose results by motion-speed quartile so the
  paper can distinguish low-motion ambiguity from fast-sport behavior.
- Report degradation relative to `zero_reference` and recovery relative to the
  `uncorrected` to `oracle` gap.

## Outputs

- `logs/ivc_p1/temporal_alignment/validation_threshold.json`
- `logs/ivc_p1/temporal_alignment/per_sample_offsets.csv`
- `logs/ivc_p1/temporal_alignment/temporal_alignment_summary.csv`
- `logs/ivc_p1/temporal_alignment/temporal_alignment_provenance.json`
- `logs/ivc_p1/temporal_alignment/temporal_alignment.pdf`

## Acceptance Criteria

- Unit fixtures recover integer and half-frame shifts with declared sign
  conventions.
- A constant-pose fixture returns zero rather than an arbitrary lag.
- No test sample influences the confidence threshold.
- The zero-offset automatic condition is reported even if it degrades.
- The summary includes every injected offset x correction condition row.

