# IVC P1 two-GPU runbook

All commands run from the repository root with:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python
```

The protocol uses one process per physical GPU. A scheduler does not preempt unrelated compute processes, does not overwrite non-empty pending directories, and never retries a failed cell automatically.

## Lane A: GPU 0

1. Run strict N-view evaluation:

   ```bash
   CUDA_VISIBLE_DEVICES=0 UNITY_DATA_ROOT=/home/kaixu_chen/skiing/data/skiing_unity_dataset NVIEW_WARMUP_ITERATIONS=10 /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.eval_unity_nview
   ```

   This writes `nview_efficiency.csv` with synchronized serial latency, throughput, peak allocated GPU memory, GPU/software metadata, and the 2/3/4-view accuracy--cost ratios. Use one otherwise idle GPU; do not co-schedule another workload on that device during timing.

2. After both front-end export commands in Lane B step 3 finish, run the six fresh seed-fold trainings on GPU 0:

   ```bash
   /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.experiments.run_multiseed_crossfold --gpus 0
   ```

3. Summarize after every cell is terminal and complete:

   ```bash
   /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.summarize_multiseed_crossfold --root logs/ivc_p1/multiseed
   ```

## Lane B: GPU 1

1. Calibrate temporal alignment on validation data:

   ```bash
   CUDA_VISIBLE_DEVICES=1 /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.calibrate_temporal_alignment
   ```

2. Only after calibration exits successfully, run the frozen test matrix:

   ```bash
   CUDA_VISIBLE_DEVICES=1 /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.eval_unity_temporal_alignment
   ```

3. After the N-view and temporal jobs finish, and before either training queue starts, export train/validation predictions for all three alternative front ends. Two export processes can share the two physical GPUs because their output directories are disjoint:

   GPU 0:

   ```bash
   CUDA_VISIBLE_DEVICES=0 /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.experiments.export_frontend_splits --frontends videopose3d motionbert --splits train val --device cuda:0
   ```

   GPU 1:

   ```bash
   CUDA_VISIBLE_DEVICES=1 /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.experiments.export_frontend_splits --frontends poseformer --splits train val --device cuda:0
   ```

4. Run the seven adaptation fits and then the 32 frozen evaluation cells on GPU 1:

   ```bash
   /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.experiments.run_frontend_adaptation_matrix --phase train --gpus 1
   /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.experiments.run_frontend_adaptation_matrix --phase evaluate --gpus 1
   /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.summarize_frontend_adaptation
   ```

If GPU 1 becomes free while the multi-seed queue still has pending cells, a fresh matrix must not be started against the same output root. Let the existing single-GPU scheduler finish to preserve one authoritative manifest.

## Publication artifacts

Generate artifacts only after the corresponding validator succeeds:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python paper/ivc_draft_20260821/scripts/generate_p1_nview_artifacts.py
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python paper/ivc_draft_20260821/scripts/generate_p1_temporal_alignment_artifacts.py
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python paper/ivc_draft_20260821/scripts/generate_p1_multiseed_artifacts.py
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python paper/ivc_draft_20260821/scripts/generate_p1_frontend_adaptation_artifacts.py
```

Then compile the manuscript:

```bash
cd paper/ivc_draft_20260821
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Monitoring

- Multi-seed: `logs/ivc_p1/multiseed/run_manifest.json`
- Front-end adaptation: `logs/ivc_p1/frontend_adaptation/run_manifest.json`
- N-view: `logs/ivc_p1/nview/nview_group_manifest.json`
- Temporal alignment: `logs/ivc_p1/temporal_alignment/temporal_alignment_provenance.json`
- Per-job logs are stored inside each job directory as `run.log`.

Any `failed` status is terminal. Inspect the recorded command and log, report the cause, and obtain explicit approval before starting a replacement run.
