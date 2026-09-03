# MMSports-Centered IVC Extension Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified Elsevier LaTeX draft for *Image and Vision Computing* that preserves the MMSports CanonFuse3D paper as its baseline and adds consistently evaluated robustness, interpretation, and front-end transfer experiments.

**Architecture:** The work has two boundaries. The evaluation boundary standardizes every journal-only Unity experiment on the MMSports checkpoint and complete fold-0 test split, emitting provenance-rich CSV/JSON artifacts. The paper boundary consumes only those final artifacts plus the read-only MMSports manuscript to generate tables, figures, an evidence manifest, and a compiled journal draft.

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, Hydra/OmegaConf, pytest, Matplotlib, LaTeX/BibTeX, Elsevier `elsarticle`.

**Spec:** `docs/superpowers/specs/2026-08-21-ivc-mmsports-centered-paper-design.md`

## Global Constraints

- Do not modify any file under `paper/mmsports_reference/`.
- Use `logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt` for every final Unity extension result.
- Use fold 0, seed 42, 30-frame windows, and `drop_last=False` for journal re-evaluation.
- Preserve the MMSports-reported Unity `0.3705 -> 0.2711` result as a labeled conference result; do not compare it silently with a different checkpoint or sample policy.
- Use the exact common-13 joint intersection for the front-end comparison.
- Record exact sample count, checkpoint path, checkpoint SHA-256, data root, fold, seed, joint subset, and metric units in final result metadata.
- Do not invent bibliographic metadata, author contributions, ethics claims, conflicts, or conference DOI information.
- Preserve unrelated working-tree changes and do not create Git commits unless the user explicitly asks for commits.

---

### Task 1: Complete-test-split dataloader boundary

**Files:**
- Modify: `dual2pose/eval/extension_experiment_utils.py`
- Modify: `tests/test_ivc_extension_experiments.py`

**Interfaces:**
- Consumes: an existing `torch.utils.data.DataLoader` and an optional replacement `collate_fn`.
- Produces: `complete_test_dataloader(base_loader: DataLoader, collate_fn: Callable | None = None, dataset: Dataset | None = None) -> DataLoader`, preserving supported loader settings while forcing `shuffle=False` and `drop_last=False`.

- [ ] **Step 1: Add a failing complete-split test**

```python
def test_complete_test_dataloader_retains_partial_batch():
    base = DataLoader(TensorDataset(torch.arange(10)), batch_size=4, drop_last=True)
    loader = complete_test_dataloader(base)
    batches = list(loader)
    assert [len(batch[0]) for batch in batches] == [4, 4, 2]
    assert loader.drop_last is False
```

- [ ] **Step 2: Run the focused test and confirm the missing interface**

Run: `python3 -m pytest tests/test_ivc_extension_experiments.py -q`

Expected: failure because `complete_test_dataloader` is not defined.

- [ ] **Step 3: Implement the loader clone**

```python
def complete_test_dataloader(base_loader, collate_fn=None, dataset=None):
    kwargs = {
        "dataset": dataset if dataset is not None else base_loader.dataset,
        "batch_size": base_loader.batch_size,
        "shuffle": False,
        "num_workers": base_loader.num_workers,
        "pin_memory": base_loader.pin_memory,
        "drop_last": False,
        "collate_fn": collate_fn or base_loader.collate_fn,
        "worker_init_fn": base_loader.worker_init_fn,
    }
    if base_loader.num_workers > 0:
        kwargs["persistent_workers"] = base_loader.persistent_workers
        kwargs["prefetch_factor"] = base_loader.prefetch_factor
    return DataLoader(**kwargs)
```

- [ ] **Step 4: Run the focused test**

Run: `python3 -m pytest tests/test_ivc_extension_experiments.py -q`

Expected: pass, including a final two-sample batch.

### Task 2: Apply the complete-split and provenance contract

**Files:**
- Modify: `dual2pose/eval/eval_unity_masking.py`
- Modify: `dual2pose/eval/eval_unity_temporal_offset.py`
- Modify: `dual2pose/eval/eval_unity_sampling_rate.py`
- Modify: `dual2pose/eval/eval_unity_view_angle.py`
- Modify: `dual2pose/eval/eval_unity_frontend_generalization.py`
- Modify: `tests/test_masking_summary.py`
- Modify: `tests/test_temporal_offset_eval.py`
- Modify: `tests/test_ivc_extension_experiments.py`
- Modify: `tests/test_frontend_generalization_eval.py`

**Interfaces:**
- Consumes: `complete_test_dataloader`, the configured checkpoint, and each evaluator's flattened tensors.
- Produces: every final CSV/JSON row with `sample_count`, `checkpoint`, `checkpoint_sha256`, `fold`, `seed`, `joint_subset`, and `units` fields.

- [ ] **Step 1: Add failing wrapper tests**

For masking, temporal offset, and sampling-rate wrappers, construct a ten-item
base loader with `batch_size=4, drop_last=True`, call each wrapper's
`test_dataloader()`, and assert that ten samples are emitted. For the front-end
row, assert the provenance fields are present and that the checkpoint hash is
64 hexadecimal characters.

- [ ] **Step 2: Run evaluator tests before implementation**

Run:

```bash
python3 -m pytest \
  tests/test_ivc_extension_experiments.py \
  tests/test_masking_summary.py \
  tests/test_temporal_offset_eval.py \
  tests/test_frontend_generalization_eval.py -q
```

Expected: failures on inherited `drop_last=True` and missing provenance fields.

- [ ] **Step 3: Replace local DataLoader reconstruction**

Each wrapper obtains the base loader, defines only its perturbation collate
function, and returns:

```python
return complete_test_dataloader(base_loader, collate_fn=_collate)
```

The front-end wrapper passes the manifest dataset through the `dataset`
parameter. The view-angle evaluator uses the complete base loader directly so
the final metadata-bearing partial batch is retained.

- [ ] **Step 4: Add deterministic provenance fields**

Use streaming SHA-256 calculation for the checkpoint and serialize the
resolved checkpoint path, fold, seed, common joint subset name, unit string,
and `int(flat["fused"].shape[0])`. Temporal, sampling, and masking summary
writers must not infer the count from batch count.

- [ ] **Step 5: Run evaluator tests after implementation**

Run the command from Step 2.

Expected: all focused tests pass.

### Task 3: Re-evaluate the MMSports checkpoint under the journal protocol

**Files:**
- Create through evaluator output: `logs/ivc_mmsports_extension/temporal_offset/**`
- Create through evaluator output: `logs/ivc_mmsports_extension/view_angle/**`
- Create through evaluator output: `logs/ivc_mmsports_extension/sampling_rate/**`
- Create through evaluator output: `logs/ivc_mmsports_extension/masking/**`

**Interfaces:**
- Consumes: the standardized evaluators and the MMSports checkpoint.
- Produces: complete-split CSV/JSON artifacts for the zero condition and every perturbation condition.

- [ ] **Step 1: Run the temporal-offset and gate study**

```bash
EVAL_CKPT_PATH=logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt \
EVAL_OUTPUT_ROOT=logs/ivc_mmsports_extension/temporal_offset \
TEMPORAL_OFFSETS=-2,-1,-0.5,0,0.5,1,2 \
TEMPORAL_OFFSET_VIEW=right \
conda run -n dual2pose python -m dual2pose.eval.eval_unity_temporal_offset \
  data.batch_size=256
```

Expected: seven rows with identical sample counts and an offset-zero row.

- [ ] **Step 2: Run the view-angle study**

```bash
EVAL_CKPT_PATH=logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt \
EVAL_OUTPUT_ROOT=logs/ivc_mmsports_extension/view_angle \
VIEW_ANGLE_BIN_EDGES=0,30,60,90,120,150,180 \
conda run -n dual2pose python -m dual2pose.eval.eval_unity_view_angle \
  data.batch_size=256
```

Expected: one row for every non-empty angle bin and total sample count equal to the complete test split.

- [ ] **Step 3: Run the sampling-rate study**

```bash
EVAL_CKPT_PATH=logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt \
EVAL_OUTPUT_ROOT=logs/ivc_mmsports_extension/sampling_rate \
SAMPLING_RATE_ERRORS=-0.02,-0.01,-0.005,0,0.005,0.01,0.02 \
SAMPLING_RATE_VIEW=right \
SAMPLING_RATE_ANCHOR=center \
conda run -n dual2pose python -m dual2pose.eval.eval_unity_sampling_rate \
  data.batch_size=256
```

Expected: seven rows with degradation columns anchored to zero drift.

- [ ] **Step 4: Run the selected masking grid**

Run the existing masking evaluator with the MMSports checkpoint and output
root `logs/ivc_mmsports_extension/masking` for view modes `left`, `right`, and
`both`; patterns `random`, `distal`, and `temporal`; and ratios `0`, `0.1`,
`0.2`, `0.3`, and `0.5`. Use seed 42 and temporal span 10.

Expected: 45 condition rows; the nine zero-ratio rows are numerically equal.

- [ ] **Step 5: Generate the masking publication summary**

```bash
conda run -n dual2pose python -m dual2pose.eval.summarize_unity_masking \
  --input logs/ivc_mmsports_extension/masking/occlusion_summary_last.csv \
  --output-dir logs/ivc_mmsports_extension/masking/paper_summary \
  --selected-ratios 0,0.1,0.2,0.3,0.5 \
  --max-ratio 0.5
```

Expected: selected-points CSV, normalized-AUC CSV, metadata JSON, Markdown,
PNG, and PDF outputs.

### Task 4: Re-evaluate front-end transfer with the MMSports checkpoint

**Files:**
- Create: `configs/frontend_generalization.mmsports_ivc.yaml`
- Create through evaluator output: `logs/ivc_mmsports_extension/frontend_generalization/**`
- Modify: `tests/test_frontend_comparison.py`

**Interfaces:**
- Consumes: existing validated prediction manifests for MotionBERT, PoseFormer,
  and VideoPose3D plus the MMSports checkpoint.
- Produces: a four-row common-13 comparison table and JSON provenance record.

- [ ] **Step 1: Add a failing comparison guard**

Extend the comparison test so rows with different checkpoint SHA-256 values,
different sample counts, or a non-common-13 joint subset raise `ValueError`.

- [ ] **Step 2: Run the comparison test before implementation**

Run: `python3 -m pytest tests/test_frontend_comparison.py -q`

Expected: failure until the comparison guard checks the complete provenance tuple.

- [ ] **Step 3: Implement the comparison guard and IVC suite specification**

The YAML names the MMSports checkpoint, fold 0, evaluation batch size 256,
output root `logs/ivc_mmsports_extension/frontend_generalization`, and the
three existing manifest paths. It does not re-export front-end predictions.

- [ ] **Step 4: Run all four front-end evaluations**

```bash
conda run -n dual2pose python -m dual2pose.eval.run_unity_frontend_suite \
  --spec configs/frontend_generalization.mmsports_ivc.yaml
```

Expected: SAM3D, MotionBERT, PoseFormer, and VideoPose3D rows use the MMSports
checkpoint, the same sample count, and common-13 metrics.

- [ ] **Step 5: Run the comparison tests**

Run: `python3 -m pytest tests/test_frontend_comparison.py -q`

Expected: pass.

### Task 5: Build deterministic paper tables, figures, and evidence manifest

**Files:**
- Create: `paper/ivc_draft_20260821/scripts/generate_extension_figures.py`
- Create: `paper/ivc_draft_20260821/evidence_manifest.md`
- Create: `paper/ivc_draft_20260821/tables/table_mmsports_results.tex`
- Create: `paper/ivc_draft_20260821/tables/table_feature_ablation.tex`
- Create: `paper/ivc_draft_20260821/tables/table_temporal_gate.tex`
- Create: `paper/ivc_draft_20260821/tables/table_view_angle.tex`
- Create: `paper/ivc_draft_20260821/tables/table_sampling_rate.tex`
- Create: `paper/ivc_draft_20260821/tables/table_frontend_generalization.tex`
- Create: `paper/ivc_draft_20260821/tables/table_masking_selected.tex`
- Create: `tests/test_ivc_paper_artifacts.py`

**Interfaces:**
- Consumes: final CSV/JSON files under `logs/ivc_mmsports_extension/` and MMSports report values.
- Produces: LaTeX tables, PDF/PNG plots, and a one-to-one number-to-artifact evidence map.

- [ ] **Step 1: Add failing artifact tests**

The tests parse all extension CSV files, assert a single checkpoint hash and
sample count, compare zero conditions within `1e-6` on the same joint subset,
and assert that every generated table contains its source filename in a LaTeX
comment.

- [ ] **Step 2: Run the artifact test before generation**

Run: `python3 -m pytest tests/test_ivc_paper_artifacts.py -q`

Expected: failure because the paper artifact package does not exist.

- [ ] **Step 3: Implement deterministic generation**

The script reads explicit input paths, refuses mixed provenance, converts
metres to millimetres only for the front-end table, uses a colorblind-safe
palette, writes both PDF and PNG figures, and writes table-source comments.

- [ ] **Step 4: Generate paper artifacts**

Run:

```bash
conda run -n dual2pose python \
  paper/ivc_draft_20260821/scripts/generate_extension_figures.py
```

Expected: temporal/gate, view-angle, sampling-rate, front-end, and masking
figures plus seven LaTeX tables.

- [ ] **Step 5: Complete and test the evidence manifest**

Map each abstract/table/conclusion number to the exact CSV row or MMSports
report line, including protocol label, checkpoint, sample count, joint subset,
and unit conversion. Then run `python3 -m pytest tests/test_ivc_paper_artifacts.py -q`.

Expected: pass.

### Task 6: Write the IVC manuscript package

**Files:**
- Create: `paper/ivc_draft_20260821/main.tex`
- Create: `paper/ivc_draft_20260821/references.bib`
- Create: `paper/ivc_draft_20260821/highlights.txt`
- Create: `paper/ivc_draft_20260821/cover_letter.md`
- Create: `paper/ivc_draft_20260821/revision_log.md`
- Copy binary assets into: `paper/ivc_draft_20260821/figures/`

**Interfaces:**
- Consumes: the read-only MMSports manuscript, final tables/figures, verified bibliography records, and evidence manifest.
- Produces: a self-contained English Elsevier preprint and submission-support files.

- [ ] **Step 1: Create the self-contained Elsevier structure**

Use `\documentclass[preprint,12pt]{elsarticle}`, numeric citations, six authors
in MMSports order, and separate inputs for every table. Copy the MMSports
figures without modifying the originals.

- [ ] **Step 2: Write title, abstract, introduction, and contributions**

Define the output as canonical root-relative 3D pose, disclose the MMSports
origin, and enumerate journal-only contributions. Abstract numbers come only
from the evidence manifest.

- [ ] **Step 3: Revise related work and method**

Add verified unsynchronized-camera and front-end sources. Match the method to
the active checkpoint: sequence-level Sim(3), 15-dimensional joint features,
hidden size 128, four heads, temporal kernel 5, 30-frame windows, and loss
weights `0.05`, `0.02`, `0.005`, and `0.01`.

- [ ] **Step 4: Write experimental protocol and results**

Separate MMSports-reported core results from complete-split IVC re-evaluation.
Describe each additional experiment, report negative transfer honestly, and
avoid interpreting the gate as calibrated uncertainty.

- [ ] **Step 5: Write discussion, limitations, conclusion, and declarations**

Cover synchronization, sampling drift, view geometry, blur, N-view scaling,
front-end domain shift, single-sport scope, and fused-reference limitations.
Use explicit author-confirmation markers only in the declarations sidecar when
the source package does not establish the fact.

- [ ] **Step 6: Write highlights, cover letter, and revision log**

The cover letter identifies the MMSports version, lists the new experiments,
and states that the journal submission is a substantial extension rather than
duplicate publication.

### Task 7: Citation, source, and rendered-PDF verification

**Files:**
- Modify if verification finds issues: `paper/ivc_draft_20260821/main.tex`
- Modify if verification finds issues: `paper/ivc_draft_20260821/references.bib`
- Create through compilation: `paper/ivc_draft_20260821/main.pdf`

**Interfaces:**
- Consumes: the complete IVC package.
- Produces: a compiling PDF with no undefined citations/references and a final verification report in `revision_log.md`.

- [ ] **Step 1: Verify citation-key closure**

Extract `\cite{}` keys from `main.tex`, compare them with BibTeX keys, and
fail on missing or unused journal-added records. Verify each new record through
its DOI, official proceedings page, arXiv identifier, or official repository.

- [ ] **Step 2: Compile the manuscript**

Run from `paper/ivc_draft_20260821/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 and no undefined reference or citation warnings.

- [ ] **Step 3: Inspect the rendered artifact**

Use `pdftotext main.pdf -` to check title, abstract, tables, declarations, and
page ordering. Rasterize representative pages and inspect title page, method
figure, wide result tables, and the final declarations for clipping or overlap.

- [ ] **Step 4: Run the full automated verification suite**

```bash
python3 -m pytest \
  tests/test_ivc_extension_experiments.py \
  tests/test_masking_summary.py \
  tests/test_temporal_offset_eval.py \
  tests/test_frontend_generalization_eval.py \
  tests/test_frontend_comparison.py \
  tests/test_ivc_paper_artifacts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Confirm source-package preservation**

Run: `git -C paper/mmsports_reference status --short`

Expected: no output. Record the compilation result, test count, unresolved
author-confirmation items, and source-preservation check in `revision_log.md`.
