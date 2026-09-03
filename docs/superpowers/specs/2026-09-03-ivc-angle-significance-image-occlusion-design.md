# IVC Angle Significance and Image-Occlusion Extension Design

## Purpose

Close the two experiment requests recorded by the supervisor for the IVC
extension:

1. add a statistically defensible significance analysis to the Unity
   camera-angle experiment (E4); and
2. add an image-level occlusion experiment (E5) that complements the existing
   corruption of precomputed 3D pose streams.

The extension must preserve every already reported primary number. It uses the
same fold-0 Unity test actions, the same fixed CanonFuse3D checkpoint, and the
same dataset-coordinate MPJPE definitions as the existing journal draft.

## Scope and non-goals

The work adds analysis and evaluation paths; it does not retrain CanonFuse3D,
redesign its architecture, or alter the existing keypoint-masking results.
Image occlusion is a controlled Unity/SAM3D stress test, not a claim that the
chosen synthetic occluder reproduces every real obstruction from snow, motion
blur, equipment, or self-occlusion.

The image experiment uses the two severity levels already displayed in the
paper, 0.5 and 1.0. Crossing three corruption patterns, three affected-view
modes, and two severities produces 18 result cells. The unoccluded native-SAM3D
result remains the shared reference rather than being counted as an additional
cell.

## E4: camera-angle statistical analysis

### Questions

The analysis answers two distinct questions:

1. Within every angle bin, does CanonFuse3D improve over canonical averaging?
2. Does the size of the fusion gain vary across angle bins?

These questions must not be conflated. A significant within-bin improvement
does not imply a significant or practically important angle effect.

### Analysis unit and exported data

The view-angle evaluator will export one row per held-out action and unordered
camera pair. Each row will include the action ID, both camera IDs, circular
azimuth separation, angle bin, sample/frame count, fused MPJPE,
canonical-average MPJPE, absolute paired gain, and relative paired gain.

The unordered camera pair is the clustering unit because the same pair is
evaluated over multiple test actions. Metrics must first be averaged across
actions for the same unordered pair before inferential tests. Joint-frame
observations and repeated action records must not be treated as independent
replicates.

### Statistical procedure

All tests are two-sided with alpha 0.05. The seed is 42.

- For each of the six angle bins, compare pair-averaged fused and
  canonical-average MPJPE using a paired Wilcoxon signed-rank test. Report the
  paired median absolute improvement, a rank-biserial effect size, a 95%
  camera-pair bootstrap confidence interval for the mean improvement, and a
  Holm-adjusted p-value across the six bins.
- Test the angle-bin effect on pair-averaged relative fusion gain using a
  Kruskal-Wallis omnibus test. Report epsilon-squared effect size.
- If the omnibus test is significant, report all 15 pairwise angle-bin
  Mann-Whitney contrasts with Holm correction and a rank-biserial effect size.
  These post-hoc contrasts are secondary; practical interpretation is based on
  effect sizes and confidence intervals, not p-values alone.
- Use 10,000 deterministic bootstrap resamples of unordered camera pairs
  within each angle bin. Empty bins, non-finite metrics, duplicate action-pair
  records, or incomplete six-bin coverage are hard validation failures.

The manuscript may state that fusion is significantly better than canonical
averaging within a bin only when the adjusted test supports that statement.
It may state that angle has an inferential effect only when the omnibus test
supports it. Even then, wording must distinguish statistical evidence from
practical magnitude and must not label 120--150 degrees as an optimum without
a pre-specified optimization study.

### E4 outputs

Source artifacts under `logs/ivc_mmsports_extension/view_angle/`:

- `view_angle_per_pair_last.csv`
- `view_angle_significance_last.csv`
- `view_angle_pairwise_contrasts_last.csv`
- `view_angle_statistics_last.json`

The existing aggregate `view_angle_summary_last.csv` remains unchanged. The
statistics artifacts record checkpoint hash, fold, seed, bootstrap count,
analysis unit, multiplicity correction, software versions, and row counts.

## E5: image-level occlusion through SAM3D

### Controlled image occluder

Image masks use the Unity ground-truth 2D character joints only to place
synthetic occluders; ground-truth positions are never passed to SAM3D or
CanonFuse3D. For each frame, a robust person height is computed from the valid
projected character joints. Every selected joint is covered by an axis-aligned
solid square centered on that joint. The square side length is 12% of the
person height, clipped to at least 16 pixels and to the image boundary. The fill
color is the per-image RGB channel mean so that the experiment removes local
appearance without introducing a fixed black-color cue.

The experiment is deterministic with seed 42 and uses the same filtered
15-joint convention as the current pose-stream masking experiment. Random
choices are derived from a stable hash of the seed, person, action, camera,
pattern, ratio, joint, and source frame ID. They therefore do not depend on
worker count, traversal order, or resume position.

- **Random:** independently select each of the 15 joints in each frame with
  probability equal to the severity ratio.
- **Distal:** independently select the same eight filtered indices used verbatim by
  the archived pose-stream masking code (`[2,3,5,6,8,9,11,12]`) in each frame,
  with probability equal to the severity ratio.
- **Temporal:** select `round(ratio * 15)` joints for each camera stream and
  cover each selected joint during one independently positioned contiguous
  interval of 10 source-frame IDs. Defining this physical-time interval per
  camera stream, rather than per camera pair, makes the image corruption
  invariant to which partner camera is later used.

Only severity ratios 0.5 and 1.0 are evaluated. The affected-view modes are
left, right, and both. A camera stream's deterministic masked SAM3D prediction
is generated once per pattern and ratio and reused wherever that stream occurs
in a camera pair. Left/right/both composition therefore requires six masked
front-end variants, not separate SAM3D inference for all 18 result cells.

### Exact test coverage

The runner derives the four held-out actions and all 180 camera streams per
action from the executed fold-0 index. It records the exact 30 source frame
positions selected for every action-camera-pair sequence, including repeated
positions introduced when fewer than 30 valid source frames are available.
The image inference manifest is the union of those pair-specific requirements.
For the archived fold-0 data this gives 720 camera streams and 17,089 unique
images per masked front-end variant, or 102,534 SAM3D image inferences across
the six variants. The validator derives and records this count again at run
time instead of assuming that every camera has 30 distinct required images.

No favorable subset may be selected after observing inference results. The
manifest must prove coverage of all four test actions, all 180 cameras per
action, every pair-specific 30-position sequence, the 17,089-image union, and
all six pattern-ratio variants before fusion evaluation starts.

### SAM3D inference and failure policy

The image is masked in memory immediately before SAM3D inference; masked image
copies are not persisted. The existing local SAM3D checkpoint and assets are
used, and their SHA-256 hashes are recorded. The runner initializes each SAM3D
estimator once per worker, supports deterministic action-level sharding, skips
already verified outputs on resume, and writes each result atomically.

The evaluated sample set must remain fixed across conditions. A custom
evaluation wrapper first obtains the native dataset's already selected frame
indices and then replaces the corresponding pose values; it must not rerun the
dataset's missing-detection filter on the masked predictions. If SAM3D returns
no person for a required frame, the exported 15-joint pose for that frame is an
all-zero missing observation and the failure is recorded in a separate boolean
manifest. The runner must never substitute the unoccluded prediction. Each
condition reports frame-level detection failure rate in addition to pose and
fusion metrics. Any non-finite downstream metric is a hard failure and remains
visible rather than being silently dropped.

### Fusion evaluation

CanonFuse3D remains frozen. For each pattern and severity:

- **left:** use masked-SAM3D predictions for the left stream and native SAM3D
  for the right stream;
- **right:** use native SAM3D for the left stream and masked-SAM3D predictions
  for the right stream;
- **both:** use masked-SAM3D predictions for both streams, generated
  independently for their respective camera streams.

Every one of the 18 cells evaluates the complete 64,440-pair test split with
`drop_last=False`. Report fused MPJPE, canonical-average MPJPE, acceleration
error, relative fusion gain, gate-error correlation, gate-preference accuracy,
and SAM3D detection failure rate. Results are compared descriptively with the
matched pose-stream masking cells; the two perturbations are not treated as
identical interventions.

### E5 outputs

Source artifacts under `logs/ivc_mmsports_extension/image_occlusion/`:

- `required_frames_manifest.json`
- `mask_protocol.json`
- `inference/<pattern>/<ratio>/...` condition-specific pose files and failure
  manifests
- `image_occlusion_summary_last.csv`
- `image_occlusion_summary_last.json`
- `image_vs_pose_occlusion_last.csv`
- `image_occlusion_robustness.pdf`

The summary JSON records all coverage counts, checkpoint hashes, SAM3D hashes,
fold, seed, mask geometry, failure policy, exact commands, and runtime/software
information.

## Code organization

Pure mask geometry, deterministic sampling, frame-manifest validation, and
statistical helpers will be separate from GPU inference so they can be tested
without loading SAM3D or CanonFuse3D.

Planned files:

- add per-pair export support to
  `dual2pose/eval/eval_unity_view_angle.py` and shared aggregation helpers;
- add `dual2pose/eval/analyze_unity_view_angle.py` for E4 inference;
- add `dual2pose/eval/image_occlusion.py` for pure mask and manifest logic;
- add `dual2pose/eval/run_unity_image_occlusion_frontend.py` for resumable
  SAM3D inference;
- add `dual2pose/eval/eval_unity_image_occlusion.py` for the 18-cell frozen
  fusion evaluation;
- add focused unit and artifact tests under `tests/`;
- update the IVC tables, figures, Results/Discussion text, and evidence
  manifest only after validated result artifacts exist.

Existing user changes and existing result files must be preserved. New output
directories are condition-specific and may not overwrite native SAM3D data.

## Validation gates

Implementation is accepted only when all of the following hold:

1. unit tests fail before implementation and pass after implementation;
2. a tiny image-occlusion smoke run proves deterministic masks, atomic resume,
   failure recording, and SAM3D-to-fusion compatibility;
3. E4 exports exactly six nonempty bins with unique action-pair rows and
   reproducible inferential outputs;
4. the E5 manifest proves 720 streams, the archived 17,089 required unique
   images per variant, six complete variants, and 18 complete 64,440-pair
   evaluation cells;
5. the clean native-SAM3D reference reproduced through the new evaluation path
   agrees with the archived reference within numerical tolerance;
6. manuscript values are generated from source artifacts rather than entered
   independently;
7. packaged copies and source files have matching SHA-256 hashes;
8. the manuscript builds successfully and the rendered PDF is checked for
   table/figure overflow and claim-result consistency.

## Resource and execution safeguards

The full SAM3D stage is long-running and disk-intensive. Before launch, the
runner reports the exact pending frame count and estimated output size. It uses
only explicitly selected GPUs, never kills unrelated processes, and can stop
and resume at verified frame boundaries. A full condition is not labeled
complete until its coverage validator passes.

## Paper interpretation

The E4 addition supplies formal uncertainty and significance evidence without
turning small angle-dependent differences into an unsupported camera-placement
claim. The E5 addition distinguishes robustness to corrupted pose inputs from
robustness propagated through an image-based front end. Negative results,
detection failures, and cases where canonical averaging outperforms learned
fusion remain in the paper.

The final Discussion must continue to state that the synthetic image masks do
not reproduce every real-world occlusion process and that the systematic
extension remains primarily a Unity evaluation.
