# IVC P1 Multi-Seed Cross-Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run six controlled native-SAM3D trainings across seeds 13/42/73 and the two real legacy action folds, then report training-run uncertainty.

**Architecture:** Training configuration becomes seed- and fold-explicit. A matrix runner creates isolated log directories and status records without overwriting runs. A separate summarizer verifies every checkpoint and test artifact before computing run-level and action-level statistics.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Lightning, Hydra/OmegaConf, CSV/JSON, unittest.

**Spec:** `docs/superpowers/specs/2026-08-21-ivc-p1-multiseed-crossfold-design.md`

## Global Constraints

- Use only `fold_00.json` and `fold_01.json`; call the result a two-fold repeated experiment.
- Train seeds 13, 42, and 73 from scratch for 100 epochs.
- Use GPU 0/1 only when each device is free at launch; never kill or preempt an unrelated process.
- Select best validation MPJPE and retain `last.ckpt`.
- Write only under `logs/ivc_p1/multiseed/`.

---

### Task 1: Seed/fold-explicit training configuration

**Files:**
- Modify: `configs/dual2pose.yaml`
- Modify: `dual2pose/train_unity.py`
- Test: `tests/test_train_unity_matrix_config.py`

**Interfaces:**
- Adds: `train.seed: int = 42`.
- Adds: `resolve_fold_index_path(data_root: Path, fold: int) -> Path`.
- Adds: `validate_fold_metadata(index_path: Path, fold: int) -> None`.

- [ ] **Step 1: Write tests proving the hard-coded seed and fixed fold path are removed**

```python
def test_fold_one_resolves_fold_01():
    assert resolve_fold_index_path(Path("/data"), 1).name == "fold_01.json"

def test_metadata_fold_mismatch_fails(tmp_path):
    path = write_fold(tmp_path, metadata_fold=0)
    with pytest.raises(ValueError, match="metadata fold 0"):
        validate_fold_metadata(path, fold=1)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python3 -m unittest tests.test_train_unity_matrix_config -v`

- [ ] **Step 3: Implement seed/fold resolution and provenance logging**

`train_unity.py` must call `seed_everything(int(config.train.seed), workers=True)` and set the index mapping from `train.fold` before constructing `UnityDataModule`.

- [ ] **Step 4: Make validation and test loaders complete**

Modify `UnityDataModule.val_dataloader()` and `test_dataloader()` to use `drop_last=False`; keep training `drop_last=True`.

- [ ] **Step 5: Run focused and existing loader tests**

Run: `python3 -m unittest tests.test_train_unity_matrix_config tests.test_ivc_extension_experiments -v`

- [ ] **Step 6: Commit configuration changes**

```bash
git add configs/dual2pose.yaml dual2pose/train_unity.py \
  dual2pose/dataloader/data_loader.py tests/test_train_unity_matrix_config.py
git commit -m "feat: make Unity training seed and fold explicit"
```

### Task 2: Non-overwriting training matrix runner

**Files:**
- Create: `dual2pose/experiments/run_multiseed_crossfold.py`
- Test: `tests/test_multiseed_runner.py`

**Interfaces:**
- Produces: `RunKey(fold: int, seed: int)` and `RunRecord(status, command, log_dir, pid, started_at, completed_at, return_code)`.
- Produces: `build_training_command(run: RunKey, gpu: int, repo_root: Path) -> list[str]`.
- Writes: `logs/ivc_p1/multiseed/run_manifest.json` atomically.

- [ ] **Step 1: Write tests for the exact six-cell matrix and command overrides**

```python
def test_matrix_is_three_seeds_by_two_folds():
    assert set(build_run_matrix()) == {
        RunKey(0, 13), RunKey(0, 42), RunKey(0, 73),
        RunKey(1, 13), RunKey(1, 42), RunKey(1, 73)}
```

- [ ] **Step 2: Implement atomic state transitions and overwrite refusal**

Valid transitions are `pending -> running -> complete|failed`. A directory with a complete record is skipped; a directory with files but no valid record causes a hard error.

- [ ] **Step 3: Implement two-GPU scheduling**

Maintain at most one child process per allocated GPU. Poll process status every 30 seconds, append metric-file heartbeat information to the manifest, and start the next pending run when that GPU becomes free. Do not retry a failed run automatically.

- [ ] **Step 4: Run runner tests with fake subprocesses**

Run: `python3 -m unittest tests.test_multiseed_runner -v`

- [ ] **Step 5: Commit the runner**

```bash
git add dual2pose/experiments/run_multiseed_crossfold.py tests/test_multiseed_runner.py
git commit -m "feat: schedule multi-seed cross-fold training"
```

### Task 3: Per-run evaluation and statistical summarizer

**Files:**
- Create: `dual2pose/eval/summarize_multiseed_crossfold.py`
- Test: `tests/test_multiseed_summary.py`

**Interfaces:**
- Produces: `collect_run_metrics(root: Path) -> list[RunMetrics]`.
- Produces: `summarize_training_runs(rows: Sequence[RunMetrics]) -> dict[str, Any]`.
- Writes the declared per-run, per-action, JSON, and TeX outputs.

- [ ] **Step 1: Write fixture tests for duplicate/missing cells and literal mean/std/t-interval values**
- [ ] **Step 2: Implement checkpoint hash, best-epoch, split, and sample-count validation**
- [ ] **Step 3: Implement descriptive run-level and action-level statistics**

Use sample standard deviation with `ddof=1`. Compute the 95% t-interval with five degrees of freedom for the six-run aggregate. Do not compute a frame-level p-value.

- [ ] **Step 4: Run summary tests**

Run: `python3 -m unittest tests.test_multiseed_summary -v`

- [ ] **Step 5: Commit the summarizer**

```bash
git add dual2pose/eval/summarize_multiseed_crossfold.py tests/test_multiseed_summary.py
git commit -m "feat: summarize repeated CanonFuse3D training"
```

### Task 4: Launch, monitor, and aggregate six runs

**Files:**
- Runtime output only: `logs/ivc_p1/multiseed/`

- [ ] **Step 1: Run a one-epoch fold-1/seed-13 smoke training on GPU 0**

Run: `CUDA_VISIBLE_DEVICES=0 python3 dual2pose/train_unity.py train.gpu=0 train.fold=1 train.seed=13 train.max_epochs=1 data.num_workers=0 data.batch_size=64 log_path=logs/ivc_p1/multiseed_smoke/fold_1/seed_13`

- [ ] **Step 2: Verify smoke checkpoint metadata and complete validation/test batches**
- [ ] **Step 3: Recheck both GPUs and launch the non-overwriting matrix runner**

Run: `python3 -m dual2pose.experiments.run_multiseed_crossfold --gpus 0 1`

- [ ] **Step 4: Monitor manifest and metric heartbeats until terminal status**
- [ ] **Step 5: Run the summarizer after all six cells are complete**

Run: `python3 -m dual2pose.eval.summarize_multiseed_crossfold --root logs/ivc_p1/multiseed`

- [ ] **Step 6: Verify the six-run matrix and output hashes**

