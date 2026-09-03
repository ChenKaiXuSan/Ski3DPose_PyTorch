#!/usr/bin/env python3
"""Export and evaluate the complete Unity front-end generalization suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
import yaml

from dual2pose.eval.export_unity_frontend_predictions import (
    DEFAULT_STALE_DATA_ROOT,
    export_predictions,
)
from dual2pose.eval.frontend_comparison import build_comparison_rows
from dual2pose.eval.frontend_manifest import FrontEndManifest


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FRONTENDS = ("videopose3d", "poseformer", "motionbert")


def _read_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Suite specification must be a YAML object")
    for field in ("data_root", "fusion_checkpoint", "output_root", "frontends"):
        if field not in payload:
            raise ValueError(f"Suite specification requires {field}")
    frontends = payload["frontends"]
    if not isinstance(frontends, Mapping):
        raise ValueError("frontends must be an object keyed by estimator name")
    missing = [name for name in REQUIRED_FRONTENDS if name not in frontends]
    if missing:
        raise ValueError(f"Suite specification is missing: {', '.join(missing)}")
    return payload


def _manifest_for(
    name: str,
    estimator: Mapping[str, Any],
    spec: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    explicit_manifest = estimator.get("manifest")
    if explicit_manifest is not None and not overwrite:
        manifest_path = Path(str(explicit_manifest)).resolve()
        manifest = FrontEndManifest.load(manifest_path)
        if manifest.frontend_name.lower() != name.lower():
            raise ValueError(
                f"Configured manifest for {name} identifies {manifest.frontend_name}"
            )
        return manifest_path
    output_dir = Path(spec["output_root"]).resolve() / "predictions" / name
    manifest_path = output_dir / f"{name}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        FrontEndManifest.load(manifest_path)
        return manifest_path
    args = argparse.Namespace(
        frontend=name,
        frontend_repo=Path(estimator["repo"]),
        checkpoint=Path(estimator["checkpoint"]),
        data_root=Path(spec["data_root"]),
        fold_json=Path(spec["fold_json"]) if spec.get("fold_json") else None,
        split=str(spec.get("split", "test")),
        output_dir=output_dir,
        rewrite_from=Path(spec.get("rewrite_from", DEFAULT_STALE_DATA_ROOT)),
        device=str(spec.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")),
        batch_size=int(estimator.get("batch_size", spec.get("export_batch_size", 64))),
        poseformer_frames=int(estimator.get("frames", 81)),
        motionbert_config=(
            Path(estimator["config"]) if estimator.get("config") else None
        ),
        limit_streams=None,
        overwrite=overwrite,
        allow_unsafe_checkpoint=bool(
            estimator.get("allow_unsafe_checkpoint", False)
        ),
        allow_numpy_checkpoint_state=bool(
            estimator.get("allow_numpy_checkpoint_state", False)
        ),
    )
    return export_predictions(args)


def _run_evaluation(
    spec: Mapping[str, Any], manifest: Path | None
) -> None:
    data_root = Path(spec["data_root"]).resolve()
    fold_json = (
        Path(spec["fold_json"]).resolve()
        if spec.get("fold_json")
        else data_root
        / "index_mapping/use_layer_camera_filter_disabled/"
        "camera_pairs_by_action_folds/fold_00.json"
    )
    command = [
        sys.executable,
        "-m",
        "dual2pose.eval.eval_unity_frontend_generalization",
        f"data.unity.root_path={data_root}",
        f"data.unity.index_mapping_path={fold_json}",
        f"data.batch_size={int(spec.get('eval_batch_size', 256))}",
        f"data.num_workers={int(spec.get('num_workers', 8))}",
        f"train.fold={int(spec.get('fold', 0))}",
    ]
    env = os.environ.copy()
    env["EVAL_CKPT_PATH"] = str(Path(spec["fusion_checkpoint"]).resolve())
    env["EVAL_OUTPUT_ROOT"] = str(Path(spec["output_root"]).resolve() / "evaluation")
    env["DATA_PATH_REWRITE_FROM"] = str(
        Path(spec.get("rewrite_from", DEFAULT_STALE_DATA_ROOT))
    )
    if manifest is None:
        env.pop("FRONTEND_MANIFEST", None)
    else:
        env["FRONTEND_MANIFEST"] = str(manifest.resolve())
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _write_comparison(spec: Mapping[str, Any]) -> Path:
    output_root = Path(spec["output_root"]).resolve()
    checkpoint_stem = Path(spec["fusion_checkpoint"]).stem
    summary_path = (
        output_root
        / "evaluation"
        / f"frontend_generalization_summary_{checkpoint_stem}.csv"
    )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(REQUIRED_FRONTENDS)
    if bool(spec.get("include_sam3d", True)):
        requested.add("sam3d")
    rows = [row for row in rows if row.get("frontend_name") in requested]
    present = {str(row.get("frontend_name")) for row in rows}
    missing = sorted(requested - present)
    if missing:
        raise ValueError(f"Evaluation summary is missing rows: {', '.join(missing)}")
    comparison = build_comparison_rows(rows)
    comparison_path = output_root / "frontend_generalization_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    comparison_path.with_suffix(".json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.export_only and args.evaluate_only:
        parser.error("--export-only and --evaluate-only are mutually exclusive")
    spec = _read_spec(args.spec)
    manifests: dict[str, Path] = {}
    for name in REQUIRED_FRONTENDS:
        estimator = spec["frontends"][name]
        if not isinstance(estimator, Mapping):
            raise ValueError(f"frontends.{name} must be an object")
        if args.evaluate_only:
            explicit_manifest = estimator.get("manifest")
            path = (
                Path(str(explicit_manifest)).resolve()
                if explicit_manifest is not None
                else Path(spec["output_root"]).resolve()
                / "predictions"
                / name
                / f"{name}_manifest.json"
            )
            manifest = FrontEndManifest.load(path)
            if manifest.frontend_name.lower() != name.lower():
                raise ValueError(
                    f"Configured manifest for {name} identifies {manifest.frontend_name}"
                )
            manifests[name] = path
        else:
            manifests[name] = _manifest_for(
                name, estimator, spec, overwrite=args.overwrite_predictions
            )
    if args.export_only:
        for name, path in manifests.items():
            print(f"{name}: {path}")
        return
    if bool(spec.get("include_sam3d", True)):
        _run_evaluation(spec, manifest=None)
    for name in REQUIRED_FRONTENDS:
        print(f"Evaluating {name}: {manifests[name]}", flush=True)
        _run_evaluation(spec, manifest=manifests[name])
    print(_write_comparison(spec))


if __name__ == "__main__":
    main()
