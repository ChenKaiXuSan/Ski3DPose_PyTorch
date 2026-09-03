# IVC P1 Front-End Adaptation Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: ivc_p1_frontend_adaptation_v1

## Experiment Overview

- Objective: determine whether small, validation-controlled adaptation removes
  the observed negative transfer for VideoPose3D and PoseFormer and improves
  MotionBERT transfer.
- Hypothesis: head-only adaptation recovers part of the distribution mismatch;
  full and mixed-front-end fine-tuning improve more but may trade native SAM3D
  accuracy for cross-estimator robustness.
- Type: transfer learning and cross-front-end evaluation.

## Data and Leakage Rules

- Use fold 0 only for this experiment.
- Export VideoPose3D, PoseFormer, and MotionBERT predictions separately for the
  existing train, validation, and test action splits.
- Reuse the completed test manifests when their hashes and split metadata match.
- Train and validation prediction exports must use the same official repository
  commits, checkpoints, H36M-17 ground-truth 2D input, mapping, and coordinate
  conventions as the existing test exports.
- The test split is never used for epoch selection, learning-rate selection,
  freezing decisions, or corruption normalization.

## Training Conditions

All adapted models initialize model weights from the MMSports checkpoint but
start a fresh optimizer and scheduler.

For each of VideoPose3D, PoseFormer, and MotionBERT:

1. `frozen`: existing zero-shot evaluation; no training.
2. `heads_only`: freeze joint encoders, cross-view attention, temporal refiner,
   and Sim(3) feature path; train only gate and residual heads for 20 epochs.
3. `full_finetune`: train all fusion-model parameters for 20 epochs.

Add one `mixed_full` model trained for 20 epochs on a balanced concatenation of
native SAM3D, VideoPose3D, PoseFormer, and MotionBERT training samples. Its
validation loader is the balanced concatenation of the four validation views.

Adaptation uses AdamW with learning rate `1e-4`, weight decay `0.01`, seed 42,
batch size 4096, cosine annealing, and best validation-MPJPE selection. The
runner records trainable parameter count for each condition.

## Cross-Front-End Matrix

Evaluate these trained checkpoints on all four test front ends:

- original MMSports;
- three `heads_only` models;
- three `full_finetune` models; and
- one `mixed_full` model.

This produces an 8 trained-model x 4 test-front-end matrix. Front-end-specific
models must be tested on non-matching front ends as well as their matching
front end so the paper can distinguish adaptation from general robustness.

## Metrics and Analysis

- Common-13 canonical-average and fused MPJPE.
- Absolute and relative fusion gain.
- Acceleration error, gate-error correlation, and gate preference accuracy.
- Native-SAM3D retention: change from the original MMSports checkpoint when
  tested on SAM3D.
- Negative-transfer recovery: change from the frozen result for VideoPose3D
  and PoseFormer.
- Trainable parameters, epochs, wall-clock time, and checkpoint SHA-256.

## Outputs

- `logs/ivc_p1/frontend_adaptation/manifests/<frontend>/<split>.json`
- `logs/ivc_p1/frontend_adaptation/checkpoints/<condition>/...`
- `logs/ivc_p1/frontend_adaptation/run_manifest.json`
- `logs/ivc_p1/frontend_adaptation/frontend_adaptation_matrix.csv`
- `logs/ivc_p1/frontend_adaptation/frontend_adaptation_summary.json`
- `logs/ivc_p1/frontend_adaptation/frontend_adaptation_matrix.pdf`

## Acceptance Criteria

- Every non-native front end has complete, hash-identified train, validation,
  and test manifests with disjoint action splits.
- Adaptation loads model weights but not the MMSports optimizer or scheduler
  state.
- `heads_only` checkpoints change only declared gate/residual parameters.
- The final matrix has exactly 8 x 4 unique condition rows.
- Best epochs are selected only from validation metrics.
- Negative results and SAM3D regressions remain in the final artifact.

