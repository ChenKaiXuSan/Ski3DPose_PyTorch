#!/usr/bin/env python3
"""Run the 3-seed x 2-fold CanonFuse3D matrix without overwriting runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


DEFAULT_DATA_ROOT = Path("/home/kaixu_chen/skiing/data/skiing_unity_dataset")
DEFAULT_PYTHON = Path("/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python")
SEEDS = (13, 42, 73)
FOLDS = (0, 1)
VALID_TRANSITIONS = {
    "pending": {"running"},
    "running": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
}


@dataclass(frozen=True, order=True)
class RunKey:
    fold: int
    seed: int

    @property
    def manifest_key(self) -> str:
        return f"fold_{self.fold}_seed_{self.seed}"


@dataclass
class ActiveRun:
    key: RunKey
    gpu: int
    process: subprocess.Popen[Any]
    log_handle: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_matrix() -> list[RunKey]:
    return [RunKey(fold, seed) for fold in FOLDS for seed in SEEDS]


def run_directory(repo_root: Path, run: RunKey) -> Path:
    return Path(repo_root) / "logs/ivc_p1/multiseed" / f"fold_{run.fold}" / f"seed_{run.seed}"


def build_training_command(
    run: RunKey,
    gpu: int,
    repo_root: Path,
    python_executable: Path = DEFAULT_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    num_workers: int = 16,
    batch_size: int = 4096,
) -> list[str]:
    del gpu  # Physical allocation is expressed through CUDA_VISIBLE_DEVICES.
    repo_root = Path(repo_root)
    return [
        str(python_executable),
        str(repo_root / "dual2pose/train_unity.py"),
        "train.gpu=0",
        f"train.fold={run.fold}",
        f"train.seed={run.seed}",
        "train.max_epochs=100",
        "train.test_ckpt_path=best",
        f"data.unity.root_path={Path(data_root)}",
        f"data.num_workers={int(num_workers)}",
        f"data.batch_size={int(batch_size)}",
        f"log_path={run_directory(repo_root, run)}",
    ]


def transition_status(current: str, target: str) -> str:
    if target not in VALID_TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid run transition: {current} -> {target}")
    return target


def prepare_run_directory(
    run_dir: Path, known_record: Mapping[str, Any] | None
) -> None:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and known_record is None:
        raise RuntimeError(f"Refusing run directory with untracked files: {run_dir}")
    if known_record is not None and known_record.get("status") == "complete":
        return
    run_dir.mkdir(parents=True, exist_ok=True)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "experiment": "ivc_p1_multiseed_crossfold",
            "created_at": utc_now(),
            "runs": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"Invalid run manifest: {path}")
    return payload


def _gpu_uuid_map() -> dict[int, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    output: dict[int, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        raw_index, raw_uuid = [part.strip() for part in line.split(",", maxsplit=1)]
        output[int(raw_index)] = raw_uuid
    return output


def gpu_compute_processes(gpu: int) -> list[dict[str, str]]:
    uuid = _gpu_uuid_map().get(int(gpu))
    if uuid is None:
        raise ValueError(f"GPU index {gpu} does not exist")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) == 4 and parts[0] == uuid:
            rows.append(
                {"gpu_uuid": parts[0], "pid": parts[1], "name": parts[2], "memory_mib": parts[3]}
            )
    return rows


def _metric_heartbeat(run_dir: Path) -> dict[str, Any] | None:
    metric_files = sorted(run_dir.glob("csv_logs/**/metrics.csv"))
    if not metric_files:
        return None
    path = metric_files[-1]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def run_matrix(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    root = repo_root / "logs/ivc_p1/multiseed"
    manifest_path = root / "run_manifest.json"
    manifest = _load_manifest(manifest_path)
    records: dict[str, Any] = manifest["runs"]
    matrix = build_run_matrix()
    for run in matrix:
        records.setdefault(
            run.manifest_key,
            {
                "fold": run.fold,
                "seed": run.seed,
                "status": "pending",
                "log_dir": str(run_directory(repo_root, run)),
            },
        )
    stale_running = [key for key, value in records.items() if value.get("status") == "running"]
    if stale_running:
        raise RuntimeError(
            "Manifest contains running records; inspect their PIDs before resuming: "
            + ", ".join(stale_running)
        )
    _atomic_json_write(manifest_path, manifest)

    active: dict[int, ActiveRun] = {}
    while True:
        changed = False
        for gpu, active_run in list(active.items()):
            return_code = active_run.process.poll()
            record = records[active_run.key.manifest_key]
            heartbeat = _metric_heartbeat(run_directory(repo_root, active_run.key))
            if heartbeat is not None:
                record["metric_heartbeat"] = heartbeat
            if return_code is None:
                continue
            active_run.log_handle.close()
            target = "complete" if return_code == 0 else "failed"
            record["status"] = transition_status(record["status"], target)
            record["completed_at"] = utc_now()
            record["return_code"] = int(return_code)
            del active[gpu]
            changed = True

        pending = [
            run for run in matrix if records[run.manifest_key].get("status") == "pending"
        ]
        for gpu in args.gpus:
            gpu = int(gpu)
            if gpu in active or not pending:
                continue
            blockers = gpu_compute_processes(gpu)
            if blockers:
                manifest.setdefault("gpu_blockers", {})[str(gpu)] = {
                    "observed_at": utc_now(),
                    "processes": blockers,
                }
                continue
            run = pending.pop(0)
            record = records[run.manifest_key]
            directory = run_directory(repo_root, run)
            prepare_run_directory(directory, known_record=record)
            command = build_training_command(
                run,
                gpu=gpu,
                repo_root=repo_root,
                python_executable=Path(args.python),
                data_root=Path(args.data_root),
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )
            log_path = directory / "train.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(
                {
                    "status": transition_status(record["status"], "running"),
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "started_at": utc_now(),
                    "log_path": str(log_path),
                }
            )
            active[gpu] = ActiveRun(run, gpu, process, log_handle)
            changed = True

        manifest["updated_at"] = utc_now()
        if changed or active or pending:
            _atomic_json_write(manifest_path, manifest)
        terminal = all(
            records[run.manifest_key].get("status") in {"complete", "failed"}
            for run in matrix
        )
        if terminal:
            break
        time.sleep(float(args.poll_seconds))
    return 1 if any(records[run.manifest_key]["status"] == "failed" for run in matrix) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique device indices")
    if args.num_workers < 0 or args.batch_size <= 0 or args.poll_seconds <= 0:
        parser.error("worker count, batch size, and poll interval must be valid")
    return args


def main() -> None:
    raise SystemExit(run_matrix(parse_args()))


if __name__ == "__main__":
    main()
