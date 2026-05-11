#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch visualize all frames for all runs.
Usage:
    python batch_visualize_all_frames.py [--num-workers 4] [--show-joint-labels]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool
import numpy as np


def _enumerate_person_frames() -> List[Tuple[str, int]]:
    """Enumerate all (person_id, frame_index) pairs with valid dual-view npz files."""
    sam_root = Path('/workspace/data/sam3d_body_results/person')
    person_dirs = sorted([d for d in sam_root.iterdir() if d.is_dir()])
    
    tasks: List[Tuple[str, int]] = []
    for person_dir in person_dirs:
        person_id = person_dir.name
        left_npz = person_dir / 'osmo_1_sam_3d_body_outputs.npz'
        right_npz = person_dir / 'osmo_2_sam_3d_body_outputs.npz'
        if not left_npz.exists() or not right_npz.exists():
            continue
        
        data = np.load(left_npz, allow_pickle=True)
        outputs = data['outputs']
        n_frames = len(outputs)
        
        for frame_idx in range(n_frames):
            tasks.append((person_id, frame_idx))
    
    return tasks


def _visualize_one_frame(task: Tuple[str, int, bool, str]) -> Tuple[str, int, bool]:
    """Visualize one frame. Returns (person_id, frame_idx, success)."""
    person_id, frame_idx, show_labels, script_path = task
    
    cmd = [
        sys.executable,
        script_path,
        '--run-id', person_id,
        '--frame-index', str(frame_idx),
        '--device', 'cpu',
    ]
    
    if show_labels:
        cmd.append('--show-joint-labels')
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
        success = result.returncode == 0
        if not success:
            print(f"[FAIL] {person_id} frame {frame_idx}: {result.stderr[:200]}")
        else:
            print(f"[OK] {person_id} frame {frame_idx}")
        return (person_id, frame_idx, success)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {person_id} frame {frame_idx}")
        return (person_id, frame_idx, False)
    except Exception as e:
        print(f"[ERROR] {person_id} frame {frame_idx}: {e}")
        return (person_id, frame_idx, False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch visualize all frames for all available runs."
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=32,
        help='Number of parallel workers for visualization.'
    )
    parser.add_argument(
        '--show-joint-labels',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to show joint labels in output figures (default: True).'
    )
    parser.add_argument(
        '--persons',
        type=str,
        nargs='*',
        default=None,
        help='Optional list of specific person/run ids, e.g. run_3 run_4 (default: all)'
    )
    args = parser.parse_args()
    
    script_path = Path(__file__).parent / 'visualize_side_raw_main_frame.py'
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    tasks = _enumerate_person_frames()
    
    if args.persons:
        allowed = set(args.persons)
        tasks = [(r, f) for r, f in tasks if r in allowed]
    
    print(f"[INFO] Processing {len(set(r for r, _ in tasks))} runs with {len(tasks)} total frames, joint labels enabled")
    
    print(f"[INFO] Total tasks: {len(tasks)} frames across runs")
    
    extended_tasks = [(r, f, args.show_joint_labels, str(script_path)) for r, f in tasks]
    
    successes = 0
    failures = 0
    
    if args.num_workers == 1:
        for task in extended_tasks:
            _, _, success = _visualize_one_frame(task)
            if success:
                successes += 1
            else:
                failures += 1
    else:
        with Pool(args.num_workers) as pool:
            for _, _, success in pool.imap_unordered(_visualize_one_frame, extended_tasks):
                if success:
                    successes += 1
                else:
                    failures += 1
    
    print(f"\n[DONE] Successes: {successes}, Failures: {failures}")
    print("Output saved to: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/eval_true_data/side_raw_main_frame_vis/")


if __name__ == '__main__':
    main()
