# IVC P1 Temporal Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate and correct half-frame pose-stream offsets without ground truth, then quantify recovery against uncorrected and oracle conditions.

**Architecture:** Pure tensor helpers build canonical velocity descriptors and score a fixed lag grid. A validation calibration step chooses the confidence threshold. One evaluator produces uncorrected, automatic, and oracle predictions per injected offset while reusing the established offset sign and interpolation implementation.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Lightning, NumPy, Matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-08-21-ivc-p1-temporal-alignment-design.md`

## Global Constraints

- Search -5 to +5 frames in 0.5-frame increments.
- Select the confidence threshold on fold-0 validation data only.
- Evaluate all 64,440 fold-0 test samples and every declared injected offset.
- Preserve the established convention: positive injected offset means the right stream lags.
- Write only under `logs/ivc_p1/temporal_alignment/`.

---

### Task 1: Pose-velocity lag estimator

**Files:**
- Create: `dual2pose/eval/temporal_alignment.py`
- Test: `tests/test_temporal_alignment.py`

**Interfaces:**
- Produces: `velocity_descriptor(pose: Tensor) -> Tensor`.
- Produces: `lag_score(left: Tensor, right: Tensor, correction_frames: float) -> Tensor`.
- Produces: `estimate_temporal_correction(left: Tensor, right: Tensor, candidates: Sequence[float], confidence_threshold: float) -> AlignmentEstimate`.
- `AlignmentEstimate` contains `correction_frames`, `best_score`, `zero_score`, and `confidence` tensors of shape `[B]`.

- [ ] **Step 1: Write failing tests for sign, half-frame recovery, and constant motion**

```python
def test_positive_injected_lag_needs_negative_correction():
    left = nonlinear_pose_fixture(batch=2, frames=30)
    right = shift_pose_sequence(left, offset_frames=2.0)
    result = estimate_temporal_correction(left, right, CANDIDATES, 0.0)
    assert torch.equal(result.correction_frames, torch.tensor([-2.0, -2.0]))

def test_constant_pose_returns_zero():
    pose = torch.zeros(2, 30, 15, 3)
    result = estimate_temporal_correction(pose, pose, CANDIDATES, 0.0)
    assert torch.equal(result.correction_frames, torch.zeros(2))
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python3 -m unittest tests.test_temporal_alignment.TemporalEstimatorTest -v`

- [ ] **Step 3: Implement canonical velocity descriptors and overlap-only cosine scoring**

Use `canonicalize_pose_torch` and `_shift_pose_sequence`. Candidate scoring must exclude clamped boundary positions so repeated edge frames cannot improve a lag score.

- [ ] **Step 4: Implement deterministic tie-breaking**

Break equal scores by smallest absolute correction and then negative-before-positive ordering. Non-finite or zero-energy descriptors return correction 0 and confidence 0.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_temporal_alignment.TemporalEstimatorTest -v`

- [ ] **Step 6: Commit the estimator**

```bash
git add dual2pose/eval/temporal_alignment.py tests/test_temporal_alignment.py
git commit -m "feat: estimate pose-space temporal corrections"
```

### Task 2: Validation-only threshold calibration

**Files:**
- Create: `dual2pose/eval/calibrate_temporal_alignment.py`
- Modify: `tests/test_temporal_alignment.py`

**Interfaces:**
- Produces: `choose_confidence_threshold(rows: Sequence[CalibrationRow], candidates: Sequence[float]) -> dict[str, Any]`.
- Writes: `logs/ivc_p1/temporal_alignment/validation_threshold.json`.

- [ ] **Step 1: Add a literal calibration fixture**

```python
def test_threshold_minimizes_validation_offset_mae_without_test_rows():
    rows = [CalibrationRow("val", 0.1, True), CalibrationRow("val", 0.6, False)]
    result = choose_confidence_threshold(rows, [0.0, 0.2, 0.5, 0.8])
    assert result["selected_threshold"] == 0.2
```

- [ ] **Step 2: Implement threshold sweep with split assertion**

Reject any calibration row whose split is not exactly `val`. Select the lowest
offset MAE; break ties by lower zero-offset false-correction rate, then higher
threshold.

- [ ] **Step 3: Run tests and a limited validation calibration**

Run: `python3 -m unittest tests.test_temporal_alignment -v`

Run: `CUDA_VISIBLE_DEVICES=0 ALIGN_LIMIT_BATCHES=2 python3 -m dual2pose.eval.calibrate_temporal_alignment data.num_workers=0`

- [ ] **Step 4: Commit calibration code**

```bash
git add dual2pose/eval/calibrate_temporal_alignment.py tests/test_temporal_alignment.py
git commit -m "feat: calibrate temporal alignment confidence"
```

### Task 3: Full automatic/oracle evaluation

**Files:**
- Create: `dual2pose/eval/eval_unity_temporal_alignment.py`
- Test: `tests/test_temporal_alignment_eval.py`

**Interfaces:**
- Consumes: validation threshold JSON and MMSports checkpoint.
- Produces: one per-sample row and one summary row per injected offset x correction condition.

- [ ] **Step 1: Write a fake-model test for all three correction branches**
- [ ] **Step 2: Implement batch evaluation that emits uncorrected, automatic, and oracle outputs**

For each injected offset, perform one data pass. Compute the automatic correction once per sample, then invoke the existing model forward path on the three right-stream variants. Record known injected offset, estimated correction, target correction, and motion speed.

- [ ] **Step 3: Implement speed-quartile and aggregate summaries**

Motion quartile cut points must come from the fold-0 validation distribution and be stored beside the confidence threshold.

- [ ] **Step 4: Run unit and two-batch smoke tests**

Run: `python3 -m unittest tests.test_temporal_alignment tests.test_temporal_alignment_eval -v`

Run: `CUDA_VISIBLE_DEVICES=0 ALIGN_LIMIT_BATCHES=2 python3 -m dual2pose.eval.eval_unity_temporal_alignment data.num_workers=0`

- [ ] **Step 5: Commit evaluator code**

```bash
git add dual2pose/eval/eval_unity_temporal_alignment.py tests/test_temporal_alignment_eval.py
git commit -m "feat: evaluate automatic temporal alignment"
```

### Task 4: Full run and paper artifacts

**Files:**
- Create: `paper/ivc_draft_20260821/scripts/generate_p1_temporal_alignment_artifacts.py`
- Test: `tests/test_p1_temporal_alignment_artifacts.py`

- [ ] **Step 1: Launch validation calibration on the allocated GPU**

Run: `CUDA_VISIBLE_DEVICES=<allocated> python3 -m dual2pose.eval.calibrate_temporal_alignment`

- [ ] **Step 2: Launch the full test matrix after calibration succeeds**

Run: `CUDA_VISIBLE_DEVICES=<allocated> python3 -m dual2pose.eval.eval_unity_temporal_alignment`

- [ ] **Step 3: Test and generate the LaTeX table and PDF figure**

Run: `python3 -m unittest tests.test_p1_temporal_alignment_artifacts -v`

Run: `python3 paper/ivc_draft_20260821/scripts/generate_p1_temporal_alignment_artifacts.py`

- [ ] **Step 4: Verify provenance and row completeness**

Expected: 11 offsets x 3 correction conditions, with per-sample estimates and validation-only threshold hashes.

- [ ] **Step 5: Commit generated artifacts**

```bash
git add paper/ivc_draft_20260821/scripts/generate_p1_temporal_alignment_artifacts.py \
  paper/ivc_draft_20260821/tables/temporal_alignment.tex \
  paper/ivc_draft_20260821/figures/extension/temporal_alignment.pdf \
  tests/test_p1_temporal_alignment_artifacts.py
git commit -m "exp: add automatic temporal alignment results"
```

