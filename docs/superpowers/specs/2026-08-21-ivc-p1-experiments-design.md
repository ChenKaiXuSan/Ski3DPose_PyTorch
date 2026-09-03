# IVC P1 Experiment Program Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: ivc_p1_design_v1

## Objective

Implement and execute the four P1 experiment tracks selected for the IVC
journal extension:

1. compositional 1/2/3/4-view evaluation;
2. automatic pose-space temporal alignment;
3. multi-seed, cross-fold training statistics; and
4. front-end-specific and mixed-front-end adaptation.

The program extends the accepted MMSports CanonFuse3D checkpoint and evaluation
stack. It must not silently replace the MMSports-reported results, change the
original checkpoint, or tune decisions on the test split.

## Verified Repository State

- The active MMSports checkpoint is
  `logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt`.
- The current Unity fold-0 index contains 193,320 train, 128,880 validation,
  and 64,440 test camera-pair samples.
- Despite `configs/dual2pose.yaml` declaring five folds, the data directory
  contains only `fold_00.json` and `fold_01.json`. Their 12/8/4 action counts
  show that they are the two complementary folds of the legacy protocol, not
  the first two files of the currently configured five-fold generator.
- Every fold-0 test action has 180 cameras: 36 azimuths in each of five layers.
- VideoPose3D, PoseFormer, and MotionBERT test manifests each contain 720 unique
  camera streams. Train and validation manifests have not yet been exported.
- GPU 0 is available for this program. GPU 1 is occupied by another workload
  and is outside the experiment allocation.

## Global Scientific Rules

- Use the current Unity data root
  `/home/kaixu_chen/skiing/data/skiing_unity_dataset` and rewrite stale index
  paths without editing the original index JSON files.
- Store all new outputs under `logs/ivc_p1/`; never overwrite MMSports logs,
  checkpoints, existing front-end manifests, or the current IVC evidence.
- Use the complete test loader with `drop_last=False`.
- Record checkpoint SHA-256, index-file SHA-256, seed, fold, sample count,
  joint subset, coordinate units, command, Git revision, and configuration in
  every final JSON/CSV artifact.
- Use normalized canonical-coordinate MPJPE and acceleration error as the main
  metrics. Front-end comparisons use the established common-13 joint subset.
- Select thresholds, checkpoints, and adaptation hyperparameters on validation
  data only. The test split is evaluated once per frozen, predeclared condition.
- Treat camera-pair samples and frames as correlated observations. Statistical
  summaries report training-run variation and action-level summaries; they do
  not claim that 64,440 camera pairs are independent subjects.
- Do not download new models or upload data. Reuse the locally cached official
  front-end repositories and checkpoints.

## Chosen Architecture

### N-view composition

Keep the published two-view network unchanged. For an N-view group, run all
`N choose 2` view pairs through the frozen CanonFuse3D checkpoint, transform all
pair outputs into the shared body-centric canonical representation, and average
the pair predictions. This gives a reproducible N-view extension with quadratic
inference cost while preserving the conference method. A native N-view
attention redesign was rejected because it would constitute a new architecture
and would no longer isolate the scalability of the published model.

### Automatic temporal alignment

Estimate the right-stream offset without ground truth by maximizing normalized
cross-correlation between canonical joint-velocity descriptors over a declared
half-frame candidate grid. A validation-selected confidence threshold prevents
low-motion sequences from being shifted when the correlation peak is weak.
Compare uncorrected input, automatic correction, and oracle correction.

### Training uncertainty

Run seeds 13, 42, and 73 on both existing legacy folds. All six statistical
runs are trained from scratch under one controlled 100-epoch protocol and use
the best validation-MPJPE checkpoint. The original MMSports checkpoint remains
a separate legacy anchor. Missing folds 2--4 are not fabricated or regenerated
because doing so would change the fold construction and the MMSports baseline.

### Front-end adaptation

Export train and validation predictions for VideoPose3D, PoseFormer, and
MotionBERT using the same local official checkpoints as the completed test
study. Starting from the MMSports checkpoint, compare frozen transfer,
head-only adaptation, full-model fine-tuning, and one balanced mixed-front-end
model. All adaptation checkpoint selection uses validation MPJPE.

## Subproject Specifications

- `docs/superpowers/specs/2026-08-21-ivc-p1-nview-design.md`
- `docs/superpowers/specs/2026-08-21-ivc-p1-temporal-alignment-design.md`
- `docs/superpowers/specs/2026-08-21-ivc-p1-multiseed-crossfold-design.md`
- `docs/superpowers/specs/2026-08-21-ivc-p1-frontend-adaptation-design.md`

## Execution Order

1. Implement and smoke-test the pure N-view and temporal-alignment evaluators.
2. Make seed/fold/checkpoint selection explicit and run the six native-SAM3D
   training replicates on GPU 0.
3. Export non-test front-end manifests, then run the adaptation matrix on GPU 0.
4. Validate artifact provenance and aggregate statistics.
5. Generate paper tables/figures and revise the IVC manuscript only after all
   rows have passed the provenance and sample-count gates.

## Program Acceptance Criteria

- Focused unit tests fail before implementation and pass afterward.
- A CPU fixture or limited-data smoke test completes for each new entry point.
- N-view results contain paired 1/2/3/4-view rows for every accepted group.
- Temporal results contain injected offset, estimated offset, estimation error,
  uncorrected metrics, automatic-correction metrics, and oracle metrics.
- The training matrix contains exactly 3 seeds x 2 legacy folds, with no reused
  result mislabeled as a new run.
- The adaptation matrix contains every declared train condition x test front-end
  combination and records the initialization checkpoint.
- Final manuscript values are generated from CSV/JSON outputs rather than typed
  manually.

