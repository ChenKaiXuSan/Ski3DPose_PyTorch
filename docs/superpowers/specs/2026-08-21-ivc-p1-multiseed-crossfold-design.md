# IVC P1 Multi-Seed Cross-Fold Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: ivc_p1_multiseed_v1

## Experiment Overview

- Objective: quantify training variability and action-split variability of the
  native SAM3D CanonFuse3D pipeline.
- Hypothesis: CanonFuse3D retains a positive gain over canonical averaging for
  every seed and both legacy action folds, although absolute MPJPE varies.
- Type: repeated training and evaluation.

## Protocol Boundary

The existing data root has exactly two legacy action-disjoint index files,
`fold_00.json` and `fold_01.json`. Their 12 train / 8 validation / 4 test action
allocation is a two-fold protocol. The five-fold value currently present in
`configs/dual2pose.yaml` does not describe those files. This experiment uses
the two real legacy folds and must be called a two-fold repeated experiment,
not five-fold cross-validation.

## Training Matrix

- Folds: 0 and 1.
- Seeds: 13, 42, and 73.
- Total new runs: 6, all trained from scratch.
- Backbone and losses: active `crossview_fusion` configuration matching the
  MMSports checkpoint.
- Training budget: 100 epochs, AdamW learning rate 0.001, weight decay 0.01,
  batch size 4096, and cosine annealing.
- Checkpoint rule: choose the lowest validation MPJPE checkpoint; also retain
  `last.ckpt` for audit.
- Device: GPU 0 only.

The original MMSports checkpoint is evaluated as a separate legacy anchor and
is not counted as one of the six new controlled runs.

## Required Training Changes

- Replace hard-coded `seed_everything(42)` with `train.seed`, defaulting to 42.
- Require the supplied index JSON metadata fold to equal `train.fold`.
- Write fold and seed into the log directory, checkpoint metadata, CSV rows,
  and final test summary.
- Ensure validation and test dataloaders retain their final partial batches for
  the statistical protocol.
- Provide a non-interactive matrix runner that refuses to overwrite a completed
  run and emits an explicit pending/running/complete/failed status manifest.

## Metrics and Statistical Summary

- Per run: canonical-average MPJPE, fused MPJPE, absolute/relative gain,
  acceleration error, best epoch, and training duration.
- Per action: the same spatial metrics, using action identifiers retained in
  test outputs.
- Aggregate: mean, sample standard deviation, median, minimum, maximum, and
  95% t-interval over the six independently trained models.
- Report fold and seed effects descriptively. Do not use camera-pair count as
  the degrees of freedom for a significance test.

## Outputs

- `logs/ivc_p1/multiseed/run_manifest.json`
- `logs/ivc_p1/multiseed/fold_<fold>/seed_<seed>/...`
- `logs/ivc_p1/multiseed/per_run_metrics.csv`
- `logs/ivc_p1/multiseed/per_action_metrics.csv`
- `logs/ivc_p1/multiseed/multiseed_summary.json`
- `logs/ivc_p1/multiseed/multiseed_summary.tex`

## Monitoring

- Monitor process liveness and the run's CSV metrics file.
- Flag non-finite loss, missing validation epochs, and no metric improvement for
  20 consecutive epochs. Do not automatically restart a failed run.
- A hard timeout may stop a run only when explicitly configured at launch.

## Acceptance Criteria

- Six unique fold/seed runs complete from fresh initialization.
- Every selected checkpoint has a recorded validation metric and SHA-256.
- Every test run uses the matching fold index and complete test loader.
- Aggregation refuses duplicate fold/seed keys or missing matrix cells.
- The legacy checkpoint is clearly separated from the controlled six-run
  statistics.

