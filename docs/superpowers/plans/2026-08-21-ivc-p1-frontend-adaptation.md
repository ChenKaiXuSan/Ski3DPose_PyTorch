# IVC P1 Front-End Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export leakage-safe train/validation front-end predictions, adapt CanonFuse3D, and produce the complete 8-model x 4-front-end transfer matrix.

**Architecture:** Existing manifest replacement becomes split-agnostic and can wrap train, validation, and test datasets. Adaptation loads MMSports model weights into a fresh optimizer run and supports heads-only, full, and balanced mixed-front-end modes. A matrix runner and summarizer keep training and test provenance separate.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Lightning, Hydra/OmegaConf, local VideoPose3D/PoseFormer/MotionBERT repositories, unittest.

**Spec:** `docs/superpowers/specs/2026-08-21-ivc-p1-frontend-adaptation-design.md`

## Global Constraints

- Use fold 0 and disjoint train/validation/test action splits.
- Never select an epoch or hyperparameter from test metrics.
- Initialize weights from the MMSports checkpoint but reset optimizer, scheduler, epoch, and global step.
- Use common-13 metrics for the final matrix.
- Write only under `logs/ivc_p1/frontend_adaptation/`.

---

### Task 1: Split-complete front-end exports

**Files:**
- Modify: `dual2pose/eval/export_unity_frontend_predictions.py`
- Create: `dual2pose/experiments/export_frontend_splits.py`
- Test: `tests/test_frontend_split_export.py`

**Interfaces:**
- Extends: `discover_unity_streams(..., split: str)` to accept `train`, `val`, `test`, or `all`.
- Produces: `merge_split_manifests(paths: Sequence[Path], output: Path) -> Path` with split membership metadata.

- [ ] **Step 1: Write failing tests for all-split de-duplication and action-disjoint metadata**
- [ ] **Step 2: Implement `all` as the union of the three named split lists**

Reject a camera stream assigned to more than one split. Preserve per-entry
`split` in the manifest and record split counts in metadata.

- [ ] **Step 3: Implement the three-front-end export orchestrator**

It must reuse an existing hash-matching test manifest, export missing train and
validation streams to new directories, and never pass `--overwrite` unless the
destination is an explicitly incomplete P1 directory.

- [ ] **Step 4: Run export tests and one-stream smoke exports**

Run: `python3 -m unittest tests.test_frontend_split_export tests.test_frontend_lifters -v`

Run: `CUDA_VISIBLE_DEVICES=0 python3 -m dual2pose.experiments.export_frontend_splits --frontends videopose3d --splits train --limit-streams 1`

- [ ] **Step 5: Commit export changes**

```bash
git add dual2pose/eval/export_unity_frontend_predictions.py \
  dual2pose/experiments/export_frontend_splits.py tests/test_frontend_split_export.py
git commit -m "feat: export leakage-safe front-end splits"
```

### Task 2: Train/validation/test manifest datamodule

**Files:**
- Create: `dual2pose/dataloader/frontend_pose_data.py`
- Test: `tests/test_frontend_pose_data.py`

**Interfaces:**
- Produces: `FrontEndDataModule(base_dm, train_manifest, val_manifest, test_manifest)`.
- Produces: `MixedFrontEndDataset(base_dataset, sources: Sequence[FrontEndManifest | None])`.
- `None` source denotes native SAM3D.

- [ ] **Step 1: Write tests for split-specific replacement and balanced mixed length**

```python
def test_mixed_dataset_contains_each_source_once_per_base_sample():
    mixed = MixedFrontEndDataset(base, [None, video, poseformer, motionbert])
    assert len(mixed) == 4 * len(base)
    assert [mixed[i * len(base)]["_frontend_name"] for i in range(4)] == [
        "sam3d", "videopose3d", "poseformer", "motionbert"]
```

- [ ] **Step 2: Implement wrappers without mutating the base sample**
- [ ] **Step 3: Validate manifest coverage before constructing any DataLoader**
- [ ] **Step 4: Run focused and existing manifest tests**

Run: `python3 -m unittest tests.test_frontend_pose_data tests.test_frontend_generalization_eval -v`

- [ ] **Step 5: Commit the datamodule**

```bash
git add dual2pose/dataloader/frontend_pose_data.py tests/test_frontend_pose_data.py
git commit -m "feat: train CanonFuse3D on manifest front ends"
```

### Task 3: Fresh-state adaptation trainer

**Files:**
- Create: `dual2pose/train_frontend_adaptation.py`
- Create: `configs/frontend_adaptation.yaml`
- Modify: `dual2pose/trainer/train_crossview_fusion.py`
- Test: `tests/test_frontend_adaptation.py`

**Interfaces:**
- Produces: `load_model_weights_only(module, checkpoint: Path) -> LoadReport`.
- Produces: `configure_trainable_scope(module, scope: Literal["heads_only", "full"]) -> dict[str, int]`.
- Adds configuration fields `adaptation.scope`, `adaptation.frontend`, `adaptation.epochs`, and split manifest paths.

- [ ] **Step 1: Write tests proving optimizer state is not restored and only heads change in `heads_only`**

```python
def test_heads_only_trainable_names_are_gate_and_residual():
    model = CrossViewFusionTrainer(config)
    report = configure_trainable_scope(model, "heads_only")
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names
    assert all("gate_head" in n or "residual_head" in n for n in names)
    assert report["trainable"] < report["total"]
```

- [ ] **Step 2: Implement weights-only initialization with strict state-key checking**
- [ ] **Step 3: Implement trainable scopes and fresh optimizer scheduling**
- [ ] **Step 4: Add common-13 validation metric and checkpoint selection**

The original all-15 training loss remains unchanged. The new validation metric
`val/common13_mpjpe` selects adaptation checkpoints.

- [ ] **Step 5: Run tests and one-batch heads-only/full smoke fits**

Run: `python3 -m unittest tests.test_frontend_adaptation -v`

- [ ] **Step 6: Commit the adaptation trainer**

```bash
git add dual2pose/train_frontend_adaptation.py configs/frontend_adaptation.yaml \
  dual2pose/trainer/train_crossview_fusion.py tests/test_frontend_adaptation.py
git commit -m "feat: add front-end adaptation training"
```

### Task 4: Eight-model cross-front-end matrix runner

**Files:**
- Create: `dual2pose/experiments/run_frontend_adaptation_matrix.py`
- Create: `dual2pose/eval/summarize_frontend_adaptation.py`
- Test: `tests/test_frontend_adaptation_matrix.py`

**Interfaces:**
- Produces 7 adaptation training conditions plus the original MMSports model.
- Produces exactly 8 trained-model x 4 test-front-end rows.
- Writes atomic training/evaluation status to `run_manifest.json`.

- [ ] **Step 1: Write tests for the seven training runs and 32 evaluation cells**
- [ ] **Step 2: Implement two-GPU non-overwriting training scheduling**
- [ ] **Step 3: Implement evaluation commands using existing front-end manifests**
- [ ] **Step 4: Implement matrix validation, recovery deltas, and SAM3D-retention summaries**
- [ ] **Step 5: Run runner/summarizer tests with fake subprocess outputs**

Run: `python3 -m unittest tests.test_frontend_adaptation_matrix -v`

- [ ] **Step 6: Commit matrix tooling**

```bash
git add dual2pose/experiments/run_frontend_adaptation_matrix.py \
  dual2pose/eval/summarize_frontend_adaptation.py \
  tests/test_frontend_adaptation_matrix.py
git commit -m "feat: run front-end adaptation matrix"
```

### Task 5: Export, train, evaluate, and generate paper artifacts

**Files:**
- Create: `paper/ivc_draft_20260821/scripts/generate_p1_frontend_adaptation_artifacts.py`
- Test: `tests/test_p1_frontend_adaptation_artifacts.py`

- [ ] **Step 1: Export complete train and validation manifests on an allocated GPU**

Run: `CUDA_VISIBLE_DEVICES=<allocated> python3 -m dual2pose.experiments.export_frontend_splits --frontends videopose3d poseformer motionbert --splits train val`

- [ ] **Step 2: Verify split disjointness, entry counts, repository commits, and checkpoint hashes**
- [ ] **Step 3: Launch the seven adaptation trainings across the two-GPU scheduler**

Run: `python3 -m dual2pose.experiments.run_frontend_adaptation_matrix --gpus 0 1 --phase train`

- [ ] **Step 4: Evaluate all 32 frozen cells**

Run: `python3 -m dual2pose.experiments.run_frontend_adaptation_matrix --gpus 0 1 --phase evaluate`

- [ ] **Step 5: Summarize and generate LaTeX/PDF artifacts**

Run: `python3 -m dual2pose.eval.summarize_frontend_adaptation`

Run: `python3 paper/ivc_draft_20260821/scripts/generate_p1_frontend_adaptation_artifacts.py`

- [ ] **Step 6: Verify exactly 32 unique rows and retain negative results**

