# IVC P1 N-View Composition Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: ivc_p1_nview_v2

## Experiment Overview

- Objective: test whether the frozen dual-view CanonFuse3D model can scale to
  three and four cameras through pairwise composition.
- Hypothesis: aggregating complementary pair predictions reduces MPJPE from
  one to multiple views, but improvement saturates and inference cost grows as
  `N choose 2`.
- Type: controlled inference evaluation.

## Inputs

- Unity fold-0 test index:
  `/home/kaixu_chen/skiing/data/skiing_unity_dataset/index_mapping/use_layer_camera_filter_disabled/camera_pairs_by_action_folds/fold_00.json`
- MMSports checkpoint:
  `logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt`
- Native SAM3D pose streams and Unity avatar skeletons under the current Unity
  data root.

## Paired Group Construction

For each of the four fold-0 test actions, each elevation layer L0--L4, and each
unique 90-degree cyclic anchor, construct one ordered four-camera group
`[A, A+90, A+180, A+270] mod 360`. Canonicalize cyclic rotations so the same
camera set is emitted once. This yields 4 actions x 5 layers x 9 anchors = 180
base groups.

Evaluate nested subsets from each base group:

- N=1: `[A]`;
- N=2: `[A, A+180]`;
- N=3: `[A, A+90, A+180]`; and
- N=4: `[A, A+90, A+180, A+270]`.

The nested construction keeps action, layer, anchor, and target motion paired
across N. For each group, intersect valid frame indices across all four cameras
and the avatar ground truth before uniform selection of the common 30-frame
window. Reject the entire base group if fewer than 30 common valid frames
remain; report rejected-group counts and reasons.

## Compared Methods

- `single_view`: the canonicalized N=1 input.
- `nview_canonical_mean`: arithmetic mean of the N independently canonicalized
  inputs.
- `pairwise_canonfuse_mean`: run all unique view pairs through the frozen model
  and average their fused canonical predictions.
- `pairwise_oracle_select`: select, for analysis only, the pair prediction with
  the lowest ground-truth MPJPE. This is an upper bound and must never be called
  a deployable method.

N=1 has only `single_view`; N=2/3/4 report the applicable baselines. No model is
trained or fine-tuned in this experiment.

## Metrics and Analysis

- MPJPE and acceleration error for every method.
- Absolute and relative gain over `nview_canonical_mean`.
- Per-action metrics and aggregate metrics over the same accepted base groups.
- Number of pair forwards, synchronized serial wall-clock inference time, throughput, and peak allocated GPU memory. Timing uses ten four-view warm-ups followed by all 44 accepted 30-frame groups on one idle NVIDIA RTX A6000; inputs are transferred before timing, and the monocular front end, data loading, and oracle selection are excluded.
- Paired bootstrap confidence intervals over the 180 base groups, resampling
  groups rather than individual frames or joints.

## Outputs

- `logs/ivc_p1/nview/nview_group_manifest.json`
- `logs/ivc_p1/nview/nview_per_group.csv`
- `logs/ivc_p1/nview/nview_summary.csv`
- `logs/ivc_p1/nview/nview_efficiency.csv`
- `logs/ivc_p1/nview/nview_provenance.json`
- `logs/ivc_p1/nview/nview_scaling.pdf`

## Acceptance Criteria

- Exactly 180 base groups are proposed before missing-data filtering.
- Accepted groups have one row for every N=1,2,3,4 condition.
- Every pair within a group uses identical common frame indices and ground truth.
- N=2 `pairwise_canonfuse_mean` agrees with direct dual-view inference within
  floating-point tolerance.
- Oracle rows are visibly labeled as upper bounds.

