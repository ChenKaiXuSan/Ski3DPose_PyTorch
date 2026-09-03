#!/usr/bin/env python3
"""Schedule seven adaptation fits or the 8x4 frozen transfer evaluation matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from dual2pose.experiments.export_frontend_splits import (
    DATA_ROOT,
    EXISTING_TEST_MANIFESTS,
    OUTPUT_ROOT as PREDICTION_ROOT,
)
from dual2pose.experiments.run_multiseed_crossfold import (
    DEFAULT_PYTHON,
    _atomic_json_write,
    gpu_compute_processes,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "logs/ivc_p1/frontend_adaptation"
SOURCE_CHECKPOINT = REPO_ROOT / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
FRONTENDS = ("videopose3d", "poseformer", "motionbert")
TEST_FRONTENDS = ("sam3d",) + FRONTENDS


@dataclass(frozen=True, order=True)
class AdaptationRun:
    frontend: str
    scope: str

    @property
    def name(self) -> str:
        return f"{self.frontend}_{self.scope}"


@dataclass(frozen=True, order=True)
class EvaluationCell:
    model_name: str
    test_frontend: str

    @property
    def name(self) -> str:
        return f"{self.model_name}__on__{self.test_frontend}"


def build_training_runs() -> list[AdaptationRun]:
    return [
        *(AdaptationRun(frontend, "heads_only") for frontend in FRONTENDS),
        *(AdaptationRun(frontend, "full") for frontend in FRONTENDS),
        AdaptationRun("mixed", "full"),
    ]


def build_evaluation_cells() -> list[EvaluationCell]:
    model_names = ["mmsports"] + [run.name for run in build_training_runs()]
    return [
        EvaluationCell(model_name, test_frontend)
        for model_name in model_names
        for test_frontend in TEST_FRONTENDS
    ]


def _split_manifest(frontend: str, split: str) -> Path:
    return PREDICTION_ROOT / frontend / split / f"{frontend}_manifest.json"


def training_directory(run: AdaptationRun) -> Path:
    return ROOT / "training" / run.name


def build_training_command(run: AdaptationRun, python: Path) -> list[str]:
    command = [
        str(python),
        str(REPO_ROOT / "dual2pose/train_frontend_adaptation.py"),
        "train.gpu=0",
        "train.fold=0",
        "train.seed=42",
        f"adaptation.frontend={run.frontend}",
        f"adaptation.scope={run.scope}",
        "adaptation.epochs=20",
        "adaptation.learning_rate=0.0001",
        f"adaptation.source_checkpoint={SOURCE_CHECKPOINT}",
        f"data.unity.root_path={DATA_ROOT}",
        "data.num_workers=16",
        "data.batch_size=4096",
        f"log_path={training_directory(run)}",
    ]
    for frontend in FRONTENDS:
        for split in ("train", "val", "test"):
            path = (
                EXISTING_TEST_MANIFESTS[frontend]
                if split == "test"
                else _split_manifest(frontend, split)
            )
            command.append(f"adaptation.manifests.{frontend}.{split}={path}")
    return command


def _checkpoint_for_model(model_name: str) -> Path:
    if model_name == "mmsports":
        return SOURCE_CHECKPOINT
    completion = ROOT / "training" / model_name / "adaptation_complete.json"
    if not completion.is_file():
        raise FileNotFoundError(f"Adaptation completion record is missing: {completion}")
    payload = json.loads(completion.read_text(encoding="utf-8"))
    path = Path(str(payload.get("best_model_path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Best adaptation checkpoint is missing: {path}")
    return path


def evaluation_directory(cell: EvaluationCell) -> Path:
    return ROOT / "matrix" / cell.model_name / cell.test_frontend


def build_evaluation_command(
    cell: EvaluationCell,
    python: Path,
) -> tuple[list[str], dict[str, str]]:
    checkpoint = _checkpoint_for_model(cell.model_name)
    command = [
        str(python),
        "-m",
        "dual2pose.eval.eval_unity_frontend_generalization",
        "train.gpu=0",
        "train.fold=0",
        f"data.unity.root_path={DATA_ROOT}",
        "data.num_workers=8",
        "data.batch_size=256",
    ]
    env = {
        "EVAL_CKPT_PATH": str(checkpoint),
        "EVAL_OUTPUT_ROOT": str(evaluation_directory(cell) / "evaluation"),
        "DATA_PATH_REWRITE_FROM": "/home/kaixu_chen/data/skiing/skiing_unity_dataset",
    }
    if cell.test_frontend != "sam3d":
        env["FRONTEND_MANIFEST"] = str(EXISTING_TEST_MANIFESTS[cell.test_frontend])
    return command, env


def _load_manifest() -> dict[str, Any]:
    path = ROOT / "run_manifest.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "experiment": "ivc_p1_frontend_adaptation",
            "created_at": utc_now(),
            "training": {},
            "evaluation": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("training"), dict) or not isinstance(payload.get("evaluation"), dict):
        raise ValueError(f"Invalid front-end adaptation manifest: {path}")
    return payload


def _prepare_directory(path: Path, record: dict[str, Any]) -> None:
    if path.exists() and any(path.iterdir()) and record.get("status") == "pending":
        raise RuntimeError(f"Refusing non-empty pending directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _schedule(
    manifest: dict[str, Any],
    jobs: list[Any],
    records: dict[str, Any],
    job_name: Callable[[Any], str],
    job_directory: Callable[[Any], Path],
    command_builder: Callable[[Any], tuple[list[str], dict[str, str]]],
    gpus: list[int],
    poll_seconds: float,
    allow_shared_gpu: bool = False,
) -> int:
    for job in jobs:
        records.setdefault(job_name(job), {"status": "pending"})
    stale = [name for name, record in records.items() if record.get("status") == "running"]
    if stale:
        raise RuntimeError("Inspect running records before resume: " + ", ".join(stale))
    active: dict[int, tuple[Any, subprocess.Popen[Any], Any]] = {}
    while True:
        changed = False
        for gpu, (job, process, log_handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            record = records[job_name(job)]
            record["status"] = "complete" if return_code == 0 else "failed"
            record["return_code"] = int(return_code)
            record["completed_at"] = utc_now()
            del active[gpu]
            changed = True
        pending = [job for job in jobs if records[job_name(job)].get("status") == "pending"]
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            blockers = [] if allow_shared_gpu else gpu_compute_processes(gpu)
            if blockers:
                manifest.setdefault("gpu_blockers", {})[str(gpu)] = {
                    "observed_at": utc_now(), "processes": blockers
                }
                continue
            job = pending.pop(0)
            record = records[job_name(job)]
            directory = job_directory(job)
            _prepare_directory(directory, record)
            command, extra_env = command_builder(job)
            log_path = directory / "run.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update(extra_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(
                {
                    "status": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "environment": extra_env,
                    "directory": str(directory),
                    "started_at": utc_now(),
                }
            )
            active[gpu] = (job, process, log_handle)
            changed = True
        manifest["updated_at"] = utc_now()
        if changed or active or pending:
            _atomic_json_write(ROOT / "run_manifest.json", manifest)
        if all(records[job_name(job)].get("status") in {"complete", "failed"} for job in jobs):
            break
        time.sleep(poll_seconds)
    return 1 if any(records[job_name(job)]["status"] == "failed" for job in jobs) else 0


def run_phase(args: argparse.Namespace) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    if args.phase == "train":
        jobs = build_training_runs()
        for frontend in FRONTENDS:
            for split in ("train", "val"):
                path = _split_manifest(frontend, split)
                if not path.is_file():
                    raise FileNotFoundError(f"Export front-end split first: {path}")
        return _schedule(
            manifest,
            jobs,
            manifest["training"],
            lambda run: run.name,
            training_directory,
            lambda run: (build_training_command(run, Path(args.python)), {}),
            args.gpus,
            args.poll_seconds,
            args.allow_shared_gpu,
        )
    jobs = build_evaluation_cells()
    return _schedule(
        manifest,
        jobs,
        manifest["evaluation"],
        lambda cell: cell.name,
        evaluation_directory,
        lambda cell: build_evaluation_command(cell, Path(args.python)),
        args.gpus,
        args.poll_seconds,
        args.allow_shared_gpu,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Launch even when unrelated compute processes are already using the GPU.",
    )
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or args.poll_seconds <= 0:
        parser.error("GPU indices must be unique and poll interval positive")
    return args


def main() -> None:
    raise SystemExit(run_phase(parse_args()))


if __name__ == "__main__":
    main()
