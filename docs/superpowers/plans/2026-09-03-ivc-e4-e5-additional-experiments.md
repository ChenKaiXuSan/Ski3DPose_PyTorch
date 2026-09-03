# IVC E4/E5 Additional Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add statistically defensible camera-angle inference and a complete 18-cell RGB-occlusion-through-SAM3D evaluation, then synchronize validated results into the IVC manuscript and evidence package.

**Architecture:** E4 uses a standalone runner that reuses the frozen view-angle evaluation path but exports action-camera-pair rows and analyzes camera-pair clusters. E5 separates deterministic image-mask and frame-manifest logic from resumable SAM3D inference and frozen CanonFuse3D evaluation. GPU-free helpers are test driven; full evaluation runs only after smoke and coverage gates pass.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Lightning, NumPy, SciPy, OpenCV, Hydra, unittest, LaTeX.

**Spec:** `docs/superpowers/specs/2026-09-03-ivc-angle-significance-image-occlusion-design.md`

## Global Constraints

- Preserve all existing primary manuscript numbers and the fixed CanonFuse3D checkpoint SHA-256 `869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f`.
- Use fold 0, seed 42, the four archived Unity test actions, all 64,440 camera pairs, all 15 model joints, and `drop_last=False`.
- Do not overwrite native SAM3D predictions or remove condition failures.
- Do not treat joint-frame observations as independent statistical replicates.
- Run image masking only at ratios 0.5 and 1.0 for random, distal, and temporal patterns; compose left, right, and both modes downstream.
- Preserve unrelated dirty-worktree files and do not stage them in experiment commits.

---

### Task 1: E4 per-pair extraction and statistical helpers

**Files:**
- Create: `dual2pose/eval/view_angle_significance.py`
- Create: `tests/test_view_angle_significance.py`

**Interfaces:**
- Consumes: trainer `test_outputs` dictionaries containing fused/canonical/ground-truth tensors and batch metadata.
- Produces: `extract_angle_pair_rows(test_outputs, bin_edges) -> list[dict[str, object]]`, `collapse_action_rows(rows) -> list[dict[str, object]]`, `holm_adjust(p_values) -> list[float]`, and `analyze_angle_rows(rows, bootstrap_resamples, seed) -> dict[str, object]`.

- [ ] **Step 1: Write failing extraction tests**

```python
def test_extract_angle_pair_rows_keeps_action_and_unordered_pair():
    rows = extract_angle_pair_rows([fixture_output()], [0, 30, 60, 90, 120, 150, 180])
    assert rows[0]["action_id"] == "turn"
    assert rows[0]["camera_pair_id"] == "capture_L0_A010|capture_L0_A350"
    assert rows[0]["angle_bin"] == "0-30"
    assert rows[0]["fused_mpjpe"] == 1.0

def test_collapse_action_rows_rejects_duplicate_action_pair_records(self):
    with self.assertRaisesRegex(ValueError, "duplicate"):
        collapse_action_rows([pair_row(), pair_row()])
```

- [ ] **Step 2: Verify extraction tests fail for missing module**

Run: `python3 -m unittest tests.test_view_angle_significance -v`

Expected: import failure for `dual2pose.eval.view_angle_significance`.

- [ ] **Step 3: Implement per-pair extraction and action collapse**

```python
def extract_angle_pair_rows(test_outputs, bin_edges):
    rows = []
    for output in test_outputs:
        fused, left, right, gt, meta = _validated_output(output)
        for index in range(fused.shape[0]):
            canonical = 0.5 * (left[index] + right[index])
            fused_error = torch.norm(fused[index] - gt[index], dim=-1).mean().item()
            baseline_error = torch.norm(canonical - gt[index], dim=-1).mean().item()
            rows.append(_pair_row(meta, index, fused_error, baseline_error, bin_edges))
    return rows
```

Require finite values, exact metadata lengths, canonical unordered camera IDs,
one row per action-camera pair, and all six bins.

- [ ] **Step 4: Run extraction tests to green**

Run: `python3 -m unittest tests.test_view_angle_significance -v`

Expected: extraction and duplicate-validation tests pass.

- [ ] **Step 5: Add failing statistical tests**

```python
def test_holm_adjust_preserves_original_order_and_is_monotone():
    assert holm_adjust([0.04, 0.01, 0.03]) == [0.06, 0.03, 0.06]

def test_analysis_reports_six_adjusted_within_bin_tests():
    result = analyze_angle_rows(six_bin_fixture(), bootstrap_resamples=200, seed=42)
    assert len(result["within_bin"]) == 6
    assert all(0.0 <= row["p_holm"] <= 1.0 for row in result["within_bin"])
    assert result["omnibus"]["test"] == "kruskal_wallis"
```

- [ ] **Step 6: Verify statistical tests fail for missing behavior**

Run: `python3 -m unittest tests.test_view_angle_significance -v`

Expected: failures identify unimplemented Holm/bootstrap/statistical outputs.

- [ ] **Step 7: Implement deterministic statistical analysis**

Use `scipy.stats.wilcoxon`, `scipy.stats.kruskal`, and
`scipy.stats.mannwhitneyu`. Compute paired rank-biserial effect size from
positive and negative signed-rank sums, epsilon-squared as
`max(0, (H-k+1)/(n-k))`, and 10,000 within-bin camera-pair bootstrap resamples
for production. Run all 15 post-hoc contrasts only when the omnibus p-value is
below 0.05; adjust them with Holm.

- [ ] **Step 8: Run E4 helper tests and syntax checks**

Run: `python3 -m unittest tests.test_view_angle_significance -v`

Run: `python3 -m py_compile dual2pose/eval/view_angle_significance.py tests/test_view_angle_significance.py`

Expected: all tests pass and both files compile.

- [ ] **Step 9: Commit Task 1 files only**

```bash
git add -- dual2pose/eval/view_angle_significance.py tests/test_view_angle_significance.py
git commit -m "feat: add clustered view-angle statistics"
```

### Task 2: E4 full runner and artifact validation

**Files:**
- Create: `dual2pose/eval/run_unity_view_angle_significance.py`
- Create: `tests/test_view_angle_significance_artifacts.py`

**Interfaces:**
- Consumes: the existing Unity datamodule, frozen checkpoint, and Task 1 helpers.
- Produces: four E4 artifacts under `logs/ivc_mmsports_extension/view_angle/`.

- [ ] **Step 1: Write failing artifact-serialization tests**

```python
def test_write_angle_artifacts_requires_64440_action_pair_rows(self):
    with self.assertRaisesRegex(ValueError, "64440"):
        write_angle_artifacts(self.output_root, pair_rows=[], statistics=fixture_statistics())

def test_write_angle_artifacts_emits_declared_files(self):
    paths = write_angle_artifacts(self.output_root, full_pair_fixture(), fixture_statistics())
    assert {path.name for path in paths} == {
        "view_angle_per_pair_last.csv",
        "view_angle_significance_last.csv",
        "view_angle_pairwise_contrasts_last.csv",
        "view_angle_statistics_last.json",
    }
```

- [ ] **Step 2: Verify artifact tests fail**

Run: `python3 -m unittest tests.test_view_angle_significance_artifacts -v`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement the standalone E4 runner**

The runner mirrors `eval_unity_view_angle.py`: patch archived paths, retain the
final batch, load the same checkpoint, run one test pass, extract 64,440 rows,
collapse them to 16,110 unordered camera-pair clusters, analyze, validate, and
write CSV/JSON atomically. Store SciPy, NumPy, PyTorch, CUDA, GPU, checkpoint,
fold, seed, and command provenance.

- [ ] **Step 4: Run artifact tests and existing angle tests**

Run: `python3 -m unittest tests.test_view_angle_significance_artifacts tests.test_ivc_extension_experiments.ViewAngleTest -v`

Expected: all tests pass.

- [ ] **Step 5: Execute full E4 analysis**

Run:

```bash
EVAL_OUTPUT_ROOT=logs/ivc_mmsports_extension/view_angle \
EVAL_SEED=42 \
python3 -m dual2pose.eval.run_unity_view_angle_significance \
  data.num_workers=32 data.batch_size=4096 train.gpu=0
```

Expected: exit 0, 64,440 action-pair rows, 16,110 camera-pair clusters, six
within-bin rows, one omnibus result, and either zero or 15 post-hoc rows as
specified by the omnibus result.

- [ ] **Step 6: Re-run the pure analyzer and compare hashes**

Run:

```bash
python3 -m dual2pose.eval.view_angle_significance \
  --input logs/ivc_mmsports_extension/view_angle/view_angle_per_pair_last.csv \
  --output-root /tmp/ivc-e4-recheck --bootstrap-resamples 10000 --seed 42
```

Expected: statistical CSV contents match the full-run artifacts byte for byte.

- [ ] **Step 7: Commit Task 2 files only**

```bash
git add -- dual2pose/eval/run_unity_view_angle_significance.py tests/test_view_angle_significance_artifacts.py
git commit -m "feat: run full view-angle significance analysis"
```

### Task 3: E5 frame manifest and deterministic image masks

**Files:**
- Create: `dual2pose/eval/image_occlusion.py`
- Create: `tests/test_image_occlusion.py`

**Interfaces:**
- Consumes: fold-0 index rows, native SAM3D availability, RGB frames, and Unity 2D character joints.
- Produces: `build_required_frames_manifest(...)`, `ImageOcclusionSetting`, `selected_joint_mask(...)`, and `apply_image_occlusion(...)`.

- [ ] **Step 1: Write failing frame-manifest tests**

```python
def test_required_manifest_preserves_pair_specific_repeated_positions(self):
    result = build_required_frames_manifest(fixture_index(self.output_root), target_length=5)
    assert result["pair_sequences"][0]["frame_indices"] == [0, 0, 1, 1, 2]
    assert result["unique_required_frame_count"] == 3

def test_manifest_rejects_missing_rgb_or_2d_joint_file(self):
    with self.assertRaises(FileNotFoundError):
        build_required_frames_manifest(incomplete_fixture_index(self.output_root), target_length=5)
```

- [ ] **Step 2: Verify frame-manifest tests fail**

Run: `python3 -m unittest tests.test_image_occlusion -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement exact frame requirement derivation**

Match `UnityDatasetDualView` frame discovery: intersect ground truth with both
native SAM3D streams, remove native `none_detected_frames`, and apply
the loader's `torch.round(torch.linspace(...)).long()` subsampling. Persist pair-specific positions plus
the per-stream union. Validate person/action/camera identities and source paths.

- [ ] **Step 4: Run frame-manifest tests to green**

Run: `python3 -m unittest tests.test_image_occlusion -v`

Expected: manifest tests pass.

- [ ] **Step 5: Add failing mask-geometry and determinism tests**

```python
def test_occluder_uses_twelve_percent_body_height_and_image_mean():
    masked, record = apply_image_occlusion(image_fixture(), joints_fixture(), setting(), frame_key())
    assert record["side_px"] == 24
    assert np.all(masked[88:112, 88:112] == image_fixture().mean((0, 1)).round())

def test_mask_is_invariant_to_iteration_and_resume_order():
    assert selected_joint_mask(setting(), frame_key()) == selected_joint_mask(setting(), frame_key())
```

- [ ] **Step 6: Verify mask tests fail for missing behavior**

Run: `python3 -m unittest tests.test_image_occlusion -v`

Expected: failures identify the missing mask geometry and stable-hash logic.

- [ ] **Step 7: Implement mask protocol**

Use BLAKE2b-derived random values keyed by seed/person/action/camera/pattern/
ratio/joint/frame. Clip valid 2D joints to the image; compute robust person
height from finite joints; fill selected 12%-height squares with rounded
per-image RGB channel means. Temporal selection is per stream and uses a
contiguous 10-source-frame interval.

- [ ] **Step 8: Run E5 pure tests and compile**

Run: `python3 -m unittest tests.test_image_occlusion -v`

Run: `python3 -m py_compile dual2pose/eval/image_occlusion.py tests/test_image_occlusion.py`

Expected: all tests pass and both files compile.

- [ ] **Step 9: Commit Task 3 files only**

```bash
git add -- dual2pose/eval/image_occlusion.py tests/test_image_occlusion.py
git commit -m "feat: add deterministic image occlusion protocol"
```

### Task 4: Resumable SAM3D image-occlusion inference

**Files:**
- Create: `dual2pose/eval/run_unity_image_occlusion_frontend.py`
- Create: `tests/test_image_occlusion_frontend.py`

**Interfaces:**
- Consumes: Task 3 required-frame manifest and mask helpers; a predictor callable for tests or the local SAM3D estimator in production.
- Produces: one `.npz` per action-camera stream with `pose`, `frame_indices`, and `detection_failed`, plus a validated 720-entry manifest per pattern-ratio condition.

- [ ] **Step 1: Write failing inference-core tests with a fake predictor**

```python
def test_infer_stream_records_detection_failure_as_zero_pose(self):
    path = infer_stream(stream_fixture(), predictor=lambda image: None, output_root=self.output_root)
    data = np.load(path)
    assert data["detection_failed"].tolist() == [True]
    assert np.count_nonzero(data["pose"]) == 0

def test_infer_stream_resumes_only_after_npz_validation(self):
    predictor = CountingPredictor()
    path = infer_stream(stream_fixture(), predictor=predictor, output_root=self.output_root)
    infer_stream(stream_fixture(), predictor=predictor, output_root=self.output_root)
    assert predictor.calls == 1
    assert validate_stream_npz(path, expected_frames=[3])
```

- [ ] **Step 2: Verify inference tests fail**

Run: `python3 -m unittest tests.test_image_occlusion_frontend -v`

Expected: missing-function failures.

- [ ] **Step 3: Implement dependency-injected inference core**

Load each RGB frame and 2D joint file, apply the Task 3 mask, call the supplied
predictor, filter SAM3D output to the model's 15 joints, encode failure as a
zero 15x3 array, and atomically replace a temporary `.npz`. Validate shapes,
finite values, exact unique frame IDs, and boolean failure length before resume.

- [ ] **Step 4: Run inference-core tests to green**

Run: `python3 -m unittest tests.test_image_occlusion_frontend -v`

Expected: fake-predictor failure and resume tests pass.

- [ ] **Step 5: Implement production SAM3D adapter and CLI**

Initialize `SAM3Dbody.infer.setup_sam_3d_body` once per worker, call
`process_one_image`, select the largest detected person with
`SAM3Dbody.infer.select_best_person`, and return `pred_keypoints_3d`. Expose
explicit manifest, output root, pattern, ratio, GPU, shard count/index,
`--dry-run`, and `--max-streams` arguments. Record checkpoint and MHR hashes.

- [ ] **Step 6: Run a one-stream SAM3D smoke test on GPU 0**

Run:

```bash
python3 -m dual2pose.eval.run_unity_image_occlusion_frontend \
  --index-path /home/kaixu_chen/skiing/data/skiing_unity_dataset/index_mapping/use_layer_camera_filter_disabled/camera_pairs_by_action_folds/fold_00.json \
  --data-root /home/kaixu_chen/skiing/data/skiing_unity_dataset \
  --output-root logs/ivc_mmsports_extension/image_occlusion/smoke \
  --pattern random --ratio 0.5 --gpu 0 --max-streams 1
```

Expected: one validated stream NPZ, no native prediction overwritten, and a
manifest reporting the attempted and failed detection counts.

- [ ] **Step 7: Commit Task 4 files only**

```bash
git add -- dual2pose/eval/run_unity_image_occlusion_frontend.py tests/test_image_occlusion_frontend.py
git commit -m "feat: add resumable masked-image SAM3D runner"
```

### Task 5: E5 mixed-view fusion evaluation and summaries

**Files:**
- Create: `dual2pose/eval/eval_unity_image_occlusion.py`
- Create: `tests/test_image_occlusion_eval.py`

**Interfaces:**
- Consumes: six complete Task 4 front-end manifests, the native Unity dataset, and the frozen CanonFuse3D checkpoint.
- Produces: 18 complete result rows plus JSON provenance and detection-failure metrics.

- [ ] **Step 1: Write failing selected-view replacement tests**

```python
def test_left_mode_replaces_only_cam1_and_aligns_duplicate_frame_ids():
    sample = base_sample(frame_indices=[2, 2, 7])
    actual = replace_image_occlusion_inputs(sample, fixture_manifest(), "left")
    assert actual["kpt3d_sam"]["cam1"][:, 0, 0].tolist() == [2.0, 2.0, 7.0]
    assert torch.equal(actual["kpt3d_sam"]["cam2"], sample["kpt3d_sam"]["cam2"])

def test_both_mode_exposes_pair_aligned_detection_failure_flags():
    actual = replace_image_occlusion_inputs(base_sample(), fixture_manifest(), "both")
    assert actual["image_occlusion_failed"]["cam1"].dtype == torch.bool
```

- [ ] **Step 2: Verify replacement tests fail**

Run: `python3 -m unittest tests.test_image_occlusion_eval -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement selected-view dataset/data-module wrapper**

Reuse `FrontEndManifest.load_pose` for exact `frame_indices` alignment, add a
matching failure-array loader, replace only the requested view mode, preserve
the native selection and complete final batch, and reject incomplete 720-stream
coverage before model evaluation.

- [ ] **Step 4: Add failing 18-cell and summary tests**

```python
def test_build_image_occlusion_study_has_exactly_18_cells():
    cells = build_image_occlusion_study(patterns=("random", "distal", "temporal"), ratios=(0.5, 1.0))
    assert len(cells) == 18
    assert {cell.view_mode for cell in cells} == {"left", "right", "both"}

def test_summary_keeps_negative_fusion_gain_and_failure_rate():
    row = summarize_cell(worse_fusion_outputs(), failure_flags())
    assert row["fusion_gain_percent"] < 0
    assert row["sam3d_detection_failure_rate"] == 0.5
```

- [ ] **Step 5: Verify summary tests fail**

Run: `python3 -m unittest tests.test_image_occlusion_eval -v`

Expected: failures identify missing study-grid and summary behavior.

- [ ] **Step 6: Implement evaluation and artifact writer**

Reuse `_flatten_test_outputs`, `_summarize_outputs`, and
`_summarize_gate_error_relationship` from the established masking/temporal
evaluators. For every cell validate 64,440 samples, finite outputs, all-15
joints, checkpoint hash, and matched manifest hashes. Write
`image_occlusion_summary_last.csv`, `image_occlusion_summary_last.json`, and
`image_vs_pose_occlusion_last.csv` atomically.

- [ ] **Step 7: Run E5 evaluation tests and compile**

Run: `python3 -m unittest tests.test_image_occlusion_eval -v`

Run: `python3 -m py_compile dual2pose/eval/eval_unity_image_occlusion.py tests/test_image_occlusion_eval.py`

Expected: all tests pass and files compile.

- [ ] **Step 8: Commit Task 5 files only**

```bash
git add -- dual2pose/eval/eval_unity_image_occlusion.py tests/test_image_occlusion_eval.py
git commit -m "feat: evaluate image-level occlusion robustness"
```

### Task 6: Full E5 GPU run and reproducibility validation

**Files:**
- Generate only under: `logs/ivc_mmsports_extension/image_occlusion/`

**Interfaces:**
- Consumes: Tasks 3--5 commands.
- Produces: six complete front-end variants and 18 complete fusion cells.

- [ ] **Step 1: Generate and validate the required-frame manifest**

Run the Task 4 CLI with `--dry-run` for each pattern-ratio pair and confirm each
reports 720 streams and 17,089 unique frames.

- [ ] **Step 2: Run six resumable SAM3D conditions on explicitly free GPUs**

Launch random/distal/temporal at ratios 0.5 and 1.0 with disjoint condition
output roots. Before every launch inspect `nvidia-smi`; never use or terminate a
GPU occupied by an unrelated process. Monitor process-alive status, completed
NPZ count, detection-failure count, and disk use at least once per hour.

- [ ] **Step 3: Validate six condition manifests**

Run the Task 4 validator for each condition. Expected per condition: 720 stream
NPZs, exact 17,089-frame union, finite 15x3 poses, exact frame IDs, and recorded
failure booleans.

- [ ] **Step 4: Run all 18 frozen fusion cells**

Run:

```bash
python3 -m dual2pose.eval.eval_unity_image_occlusion \
  --manifest-root logs/ivc_mmsports_extension/image_occlusion/inference \
  --output-root logs/ivc_mmsports_extension/image_occlusion \
  --checkpoint logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt \
  --gpu 0 --batch-size 4096 --num-workers 32
```

Expected: 18 rows, each with sample count 64,440; negative cells retained.

- [ ] **Step 5: Re-run one representative cell and compare**

Re-run `both/random/1.0` into `/tmp/ivc-e5-recheck`. Expected metric relative
differences below `1e-6` because inference inputs and checkpoint are fixed.

### Task 7: Manuscript, figure, and evidence synchronization

**Files:**
- Create: `dual2pose/eval/render_e4_e5_artifacts.py`
- Create: `tests/test_e4_e5_paper_artifacts.py`
- Create: `paper/ivc_draft_20260821/tables/view_angle_significance.tex`
- Create: `paper/ivc_draft_20260821/tables/image_occlusion_summary.tex`
- Create: `paper/ivc_draft_20260821/figures/extension/image_occlusion_robustness.pdf`
- Modify: `paper/ivc_draft_20260821/main.tex`
- Modify: `paper/ivc_draft_20260821/evidence/evidence_manifest.md`
- Package: E4/E5/E7/E8 source summaries under `paper/ivc_draft_20260821/evidence/results/`

**Interfaces:**
- Consumes: validated E4 and E5 CSV/JSON artifacts plus existing E7/E8 source summaries.
- Produces: generated LaTeX tables, comparison figure, manuscript wording, and hash-checked evidence package.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_rendered_tables_contain_only_source_csv_values(self):
    render_all(e4_fixture(), e5_fixture(), output_root=self.output_root)
    assert "Holm-adjusted" in (self.output_root / "view_angle_significance.tex").read_text()
    assert "Image-level" in (self.output_root / "image_occlusion_summary.tex").read_text()

def test_evidence_manifest_lists_e7_e8_and_new_e4_e5_sources():
    text = EVIDENCE_MANIFEST.read_text()
    for name in ("view_angle_statistics", "image_occlusion_summary", "temporal_alignment_summary", "frontend_adaptation_matrix"):
        assert name in text
```

- [ ] **Step 2: Verify artifact tests fail before generation**

Run: `python3 -m unittest tests.test_e4_e5_paper_artifacts -v`

Expected: missing renderer/artifacts fail.

- [ ] **Step 3: Implement deterministic renderer**

Read only validated source CSV/JSON, render E4 significance and E5 image-mask
tables, generate the image-vs-pose robustness figure, and refuse row-count,
checkpoint-hash, or non-finite-value mismatches.

- [ ] **Step 4: Update manuscript claims from validated artifacts**

In E4, report within-bin adjusted significance, the omnibus angle result,
effect size, and confidence intervals without calling an angle optimum. In E5,
separate pose-stream corruption from image-level front-end occlusion, report
detection failures, preserve negative cells, and retain the synthetic-mask
limitation. Correct the normalization explanation so body-centered
canonicalization is not said to divide skeleton scale.

- [ ] **Step 5: Package and hash E4/E5/E7/E8 evidence**

Copy only summary/provenance artifacts, compute SHA-256 hashes, and add exact
source and packaged paths to `evidence_manifest.md`. Verify every packaged copy
matches its source.

- [ ] **Step 6: Run manuscript artifact tests**

Run: `python3 -m unittest tests.test_e4_e5_paper_artifacts tests.test_p1_artifact_rendering -v`

Expected: all tests pass.

- [ ] **Step 7: Build and inspect the manuscript**

Run: `latexmk -pdf -interaction=nonstopmode main.tex` from
`paper/ivc_draft_20260821`.

Run: `pdftotext main.pdf /tmp/ivc-main.txt` and inspect E4/E5 wording, table
values, figure captions, disclosure text, and unresolved placeholders.

Expected: successful PDF build, no table overflow warnings attributable to new
artifacts, no unsupported significance/end-to-end claims, and source-matched
numbers.

- [ ] **Step 8: Run the focused regression suite**

Run:

```bash
python3 -m unittest \
  tests.test_view_angle_significance \
  tests.test_view_angle_significance_artifacts \
  tests.test_image_occlusion \
  tests.test_image_occlusion_frontend \
  tests.test_image_occlusion_eval \
  tests.test_e4_e5_paper_artifacts \
  tests.test_ivc_extension_experiments \
  tests.test_p1_artifact_rendering -v
```

Expected: zero failures and zero errors.

- [ ] **Step 9: Commit only newly created code/tests; leave pre-existing untracked paper ownership visible**

```bash
git add -- dual2pose/eval/render_e4_e5_artifacts.py tests/test_e4_e5_paper_artifacts.py
git commit -m "docs: add E4 and E5 journal evidence"
```

Report all modified manuscript/evidence paths separately because the existing
`paper/` tree was already untracked before this work and must not be silently
claimed as a clean isolated change.
