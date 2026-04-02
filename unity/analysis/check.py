#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RE_FRAME = re.compile(r"^frame_(\d+)\.png$")
RE_KPT2D_PER_FRAME = re.compile(r"^kpt2d_(\d+)\.npy$")
RE_GENERIC_PER_FRAME_NPY = re.compile(r"^(.+?)_(\d+)\.npy$")
RE_HELPER_JOINT = re.compile(r"(twist|roll|end|helper|ik|pole|weapon|prop|socket)", re.IGNORECASE)
KPT2D_VARIANT_NAMES = ("character", "pole", "ski")


def resolve_default_dataset_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "SkiDataset",
        script_dir.parent / "SkiDataset" / "female",
        script_dir.parent / "SkiDataset" / "male",
        Path.cwd() / "SkiDataset",
        Path.cwd() / "unity" / "SkiDataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return script_dir.parent / "SkiDataset"


@dataclass
class Issue:
    severity: str  # ERROR / WARN / INFO
    scope: str
    message: str


@dataclass
class ActionNode:
    action_name: str
    action_path: Path
    character_name: str = ""


@dataclass
class CameraReport:
    camera_id: str
    has_intrinsics: bool = False
    has_extrinsics: bool = False
    camera_meta_source: str = ""

    frame_count: int = 0
    frame_indices_min: int | None = None
    frame_indices_max: int | None = None
    frame_index_gaps: int = 0

    has_kpt2d_npy: bool = False
    kpt2d_shape: list[int] | None = None

    has_kpt2d_npz: bool = False
    kpt2d_npz_arrays: dict[str, list[int] | None] = field(default_factory=dict)

    kpt2d_per_frame_count: int = 0
    kpt2d_per_frame_gaps: int = 0
    kpt2d_per_frame_shape_ok: bool | None = None

    frame_vs_expected_delta: int | None = None
    overlay_svg: str = ""
    overlay_all_count: int = 0
    overlay_point_source: str = ""

    issues: list[Issue] = field(default_factory=list)


@dataclass
class ActionReport:
    action_name: str
    action_path: str
    character_name: str = ""

    has_meta_sequence: bool = False
    sequence: dict[str, Any] | None = None

    has_kpt3d_npy: bool = False
    kpt3d_shape: list[int] | None = None

    has_kpt3d_npz: bool = False
    kpt3d_npz_arrays: dict[str, list[int] | None] = field(default_factory=dict)
    kpt3d_visuals: list[str] = field(default_factory=list)

    camera_reports: list[CameraReport] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


@dataclass
class Summary:
    checked_at_utc: str
    dataset_root: str

    action_count: int = 0
    camera_count: int = 0

    error_count: int = 0
    warn_count: int = 0
    info_count: int = 0

    passed: bool = True


@dataclass
class FullReport:
    summary: Summary
    actions: list[ActionReport]
    visuals: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ski dataset, NPZ content, and generate visual reports.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=resolve_default_dataset_root(),
        help="Dataset root path (default: auto-detected SkiDataset location)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "reports"),
        help="Directory to write report files",
    )
    parser.add_argument(
        "--sample-per-frame-shape-check",
        type=int,
        default=32,
        help="Check shape for first N per-frame kpt2d npy files per camera (0=all)",
    )
    parser.add_argument(
        "--overlay-per-action",
        type=int,
        default=2,
        help="How many camera overlay examples to generate per action",
    )
    parser.add_argument(
        "--overlay-all-frames",
        action="store_true",
        default=True,
        help="Generate overlays for all actions, all cameras, all frames (default: enabled)",
    )
    parser.add_argument(
        "--no-overlay-all-frames",
        dest="overlay_all_frames",
        action="store_false",
        help="Disable full-frame overlay generation and only keep sample overlays",
    )
    parser.add_argument(
        "--viz-conf-threshold",
        type=float,
        default=0.0,
        help="Only draw keypoints with confidence >= threshold (default: 0.0)",
    )
    parser.add_argument(
        "--viz-main-joints",
        type=str,
        default="",
        help="Comma-separated joint indices to visualize, e.g. '0,1,2,5,8'. Empty means visualize all",
    )
    parser.add_argument(
        "--viz-auto-filter-helper-joints",
        action="store_true",
        default=True,
        help="Auto-filter helper/twist/end joints in visualization using meta/joint_names.json (default: enabled)",
    )
    parser.add_argument(
        "--no-viz-auto-filter-helper-joints",
        dest="viz_auto_filter_helper_joints",
        action="store_false",
        help="Disable automatic helper-joint filtering",
    )
    parser.add_argument(
        "--kpt3d-viz-frames",
        type=int,
        default=3,
        help="How many frames to visualize for action-level kpt3d when not using --kpt3d-viz-all-frames (default: 3)",
    )
    parser.add_argument(
        "--kpt3d-viz-all-frames",
        action="store_true",
        default=False,
        help="Visualize all kpt3d frames for each action",
    )
    parser.add_argument(
        "--kpt3d-3d-backend",
        type=str,
        choices=["matlab", "svg"],
        default="matlab",
        help="Renderer for 3D perspective image (default: matlab)",
    )
    parser.add_argument(
        "--matlab-command",
        type=str,
        default="matlab",
        help="MATLAB executable command name/path (default: matlab)",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def auto_select_main_joint_indices(joint_names: list[str]) -> set[int]:
    selected: set[int] = set()
    for idx, name in enumerate(joint_names):
        if not name:
            continue
        if RE_HELPER_JOINT.search(name):
            continue
        selected.add(idx)
    return selected


def parse_npy_header_from_bytes(raw: bytes) -> tuple[str | None, list[int] | None, int | None]:
    try:
        if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
            return None, None, None
        major = raw[6]
        if major == 1:
            hlen = struct.unpack("<H", raw[8:10])[0]
            header_start = 10
        else:
            hlen = struct.unpack("<I", raw[8:12])[0]
            header_start = 12

        header = raw[header_start:header_start + hlen].decode("latin1")
        m_dtype = re.search(r"'descr':\s*'([^']+)'", header)
        m_shape = re.search(r"'shape':\s*\((.*?)\)", header)
        if not m_dtype or not m_shape:
            return None, None, None

        dtype = m_dtype.group(1)
        shape_raw = m_shape.group(1).strip()
        shape = [int(p.strip()) for p in shape_raw.split(",") if p.strip()] if shape_raw else []
        data_offset = header_start + hlen
        return dtype, shape, data_offset
    except Exception:
        return None, None, None


def parse_npy_shape(path: Path) -> list[int] | None:
    try:
        raw = path.read_bytes()
        _, shape, _ = parse_npy_header_from_bytes(raw)
        return shape
    except Exception:
        return None


def parse_npy_float32_data(path: Path) -> tuple[list[int] | None, list[float] | None]:
    try:
        raw = path.read_bytes()
        return parse_npy_float32_data_from_bytes(raw)
    except Exception:
        return None, None


def parse_npy_float32_data_from_bytes(raw: bytes) -> tuple[list[int] | None, list[float] | None]:
    try:
        dtype, shape, data_offset = parse_npy_header_from_bytes(raw)
        if dtype not in ("<f4", "|f4") or shape is None or data_offset is None:
            return shape, None
        count = 1
        for d in shape:
            count *= d
        data = struct.unpack("<" + "f" * count, raw[data_offset:data_offset + 4 * count])
        return shape, list(data)
    except Exception:
        return None, None


def parse_npz_shapes(path: Path) -> dict[str, list[int] | None]:
    out: dict[str, list[int] | None] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                raw = zf.read(name)
                _, shape, _ = parse_npy_header_from_bytes(raw)
                out[name] = shape
    except Exception:
        return {}
    return out


def parse_npz_float32_arrays(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    out: dict[str, tuple[list[int], list[float]]] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                shape, data = parse_npy_float32_data_from_bytes(zf.read(name))
                if shape is None or data is None:
                    continue
                out[name] = (shape, data)
    except Exception:
        return {}
    return out


def extract_indices(file_paths: list[Path], regex: re.Pattern[str]) -> list[int]:
    out: list[int] = []
    for p in file_paths:
        m = regex.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def detect_kpt2d_files(kpt2d_dir: Path) -> tuple[Path | None, Path | None, list[Path], str | None]:
    if not kpt2d_dir.exists():
        return None, None, [], None

    candidate_dirs: list[Path] = [kpt2d_dir / name for name in KPT2D_VARIANT_NAMES if (kpt2d_dir / name).is_dir()]
    if not candidate_dirs:
        candidate_dirs = [kpt2d_dir]

    selected_dir: Path | None = None
    npy_files: list[Path] = []
    npz_files: list[Path] = []
    for candidate_dir in candidate_dirs:
        candidate_npy = sorted([p for p in candidate_dir.glob("*.npy") if p.is_file()])
        candidate_npz = sorted([p for p in candidate_dir.glob("*.npz") if p.is_file()])
        if candidate_npy or candidate_npz:
            selected_dir = candidate_dir
            npy_files = candidate_npy
            npz_files = candidate_npz
            break

    if selected_dir is None:
        selected_dir = candidate_dirs[0]
        npy_files = sorted([p for p in selected_dir.glob("*.npy") if p.is_file()])
        npz_files = sorted([p for p in selected_dir.glob("*.npz") if p.is_file()])

    # Prefer canonical names if present.
    main_npy = selected_dir / "kpt2d.npy"
    if not main_npy.exists():
        main_npy = None

    main_npz = selected_dir / "kpt2d.npz"
    if not main_npz.exists():
        main_npz = None

    per_frame_groups: dict[str, list[Path]] = {}
    for p in npy_files:
        m = RE_GENERIC_PER_FRAME_NPY.match(p.name)
        if not m:
            continue
        prefix = m.group(1)
        per_frame_groups.setdefault(prefix, []).append(p)

    # If canonical names are missing, infer from files.
    if main_npy is None:
        non_per_frame = [p for p in npy_files if RE_GENERIC_PER_FRAME_NPY.match(p.name) is None]
        if non_per_frame:
            main_npy = non_per_frame[0]

    if main_npz is None and npz_files:
        main_npz = npz_files[0]

    selected_prefix: str | None = None
    if main_npy is not None:
        selected_prefix = main_npy.stem

    if selected_prefix is None and per_frame_groups:
        # Pick the prefix with most files to avoid selecting unrelated npy groups.
        selected_prefix = max(per_frame_groups.items(), key=lambda kv: len(kv[1]))[0]

    per_frame_files: list[Path] = []
    if selected_prefix is not None and selected_prefix in per_frame_groups:
        per_frame_files = sorted(per_frame_groups[selected_prefix])

    return main_npy, main_npz, per_frame_files, selected_prefix


def resolve_kpt2d_frame_file(kpt2d_dir: Path, frame_index: int, prefix: str | None = None) -> Path | None:
    candidate_dirs: list[Path] = [kpt2d_dir / name for name in KPT2D_VARIANT_NAMES if (kpt2d_dir / name).is_dir()]
    if not candidate_dirs:
        candidate_dirs = [kpt2d_dir]

    candidate_names: list[str] = []
    if prefix:
        candidate_names.append(f"{prefix}_{frame_index:06d}.npy")
    candidate_names.append(f"kpt2d_{frame_index:06d}.npy")

    for candidate_dir in candidate_dirs:
        for candidate_name in candidate_names:
            candidate = candidate_dir / candidate_name
            if candidate.exists():
                return candidate
    return None


def count_gaps(indices: list[int]) -> int:
    if len(indices) <= 1:
        return 0
    return sum(1 for prev, cur in zip(indices, indices[1:]) if cur != prev + 1)


def is_action_dir(path: Path) -> bool:
    return path.is_dir() and (path / "frames").exists() and (path / "kpt2d").exists()


def discover_action_nodes(dataset_root: Path) -> list[ActionNode]:
    if not dataset_root.exists():
        return []

    direct = [d for d in dataset_root.iterdir() if is_action_dir(d)]
    if direct:
        return [ActionNode(action_name=d.name, action_path=d) for d in sorted(direct)]

    nested: list[ActionNode] = []
    for maybe_char in sorted(dataset_root.iterdir()):
        if not maybe_char.is_dir() or maybe_char.name in {"cameras", "reports"}:
            continue
        for maybe_action in sorted(maybe_char.iterdir()):
            if is_action_dir(maybe_action):
                nested.append(ActionNode(action_name=maybe_action.name, action_path=maybe_action, character_name=maybe_char.name))
    if nested:
        return nested

    legacy = dataset_root / "actions"
    if legacy.exists():
        cand = [d for d in legacy.iterdir() if is_action_dir(d)]
        if cand:
            return [ActionNode(action_name=d.name, action_path=d) for d in sorted(cand)]

    return []


def resolve_shared_camera_root(dataset_root: Path, node: ActionNode) -> Path:
    if node.character_name:
        return dataset_root / node.character_name / "cameras"
    return dataset_root / "cameras"


def reshape_kpt2d_points(shape: list[int], data: list[float], frame_idx: int | None = None) -> list[tuple[float, float, float]]:
    if len(shape) == 2 and shape[1] == 3:
        pts = []
        for j in range(shape[0]):
            base = j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    if len(shape) == 3 and shape[2] == 3:
        t = shape[0]
        jn = shape[1]
        idx = 0 if frame_idx is None else max(0, min(frame_idx, t - 1))
        start = idx * jn * 3
        pts = []
        for j in range(jn):
            base = start + j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    return []


def svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_points_to_pixels(points: list[tuple[float, float, float]], width: int, height: int) -> list[tuple[float, float, float]]:
    if not points:
        return points

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_abs = max(max(abs(v) for v in xs), max(abs(v) for v in ys))

    # normalized [0,1] or [-1,1] range
    if max_abs <= 2.0:
        normalized: list[tuple[float, float, float]] = []
        for x, y, c in points:
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and (x < 0.0 or y < 0.0):
                x = (x + 1.0) * 0.5
                y = (y + 1.0) * 0.5
            normalized.append((x * (width - 1), y * (height - 1), c))
        return normalized

    return points


def reshape_kpt3d_points(shape: list[int], data: list[float], frame_idx: int) -> list[tuple[float, float, float]]:
    if len(shape) != 3 or shape[2] != 3:
        return []
    t = shape[0]
    jn = shape[1]
    if t <= 0 or jn <= 0:
        return []

    idx = max(0, min(frame_idx, t - 1))
    start = idx * jn * 3
    pts: list[tuple[float, float, float]] = []
    for j in range(jn):
        base = start + j * 3
        pts.append((data[base], data[base + 1], data[base + 2]))
    return pts


def pick_frame_indices(total_frames: int, max_count: int) -> list[int]:
    if total_frames <= 0 or max_count <= 0:
        return []
    if total_frames <= max_count:
        return list(range(total_frames))

    if max_count == 1:
        return [total_frames // 2]

    out: list[int] = []
    for i in range(max_count):
        idx = round(i * (total_frames - 1) / (max_count - 1))
        out.append(idx)
    return sorted(set(out))


def project_to_panel(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> tuple[float, float]:
    xr = x_max - x_min
    yr = y_max - y_min
    nx = 0.5 if abs(xr) < 1e-9 else (x - x_min) / xr
    ny = 0.5 if abs(yr) < 1e-9 else (y - y_min) / yr
    px = panel_x + nx * panel_w
    py = panel_y + (1.0 - ny) * panel_h
    return px, py


def save_kpt3d_three_views_svg(out_path: Path, points: list[tuple[float, float, float]], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        out_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="980" height="360"><text x="20" y="40" font-size="16">No kpt3d points</text></svg>', encoding="utf-8")
        return

    width, height = 980, 360
    margin = 24
    gap = 18
    panel_w = (width - margin * 2 - gap * 2) / 3
    panel_h = height - 90
    panel_y = 50

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    panels = [
        ("X-Y", xs, ys, x_min, x_max, y_min, y_max),
        ("X-Z", xs, zs, x_min, x_max, z_min, z_max),
        ("Y-Z", ys, zs, y_min, y_max, z_min, z_max),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for i, (label, a_vals, b_vals, a_min, a_max, b_min, b_max) in enumerate(panels):
        panel_x = margin + i * (panel_w + gap)
        lines.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#f9f9fb" stroke="#d8d8de"/>')
        lines.append(f'<text x="{panel_x + panel_w/2}" y="{panel_y - 10}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')

        for j, (av, bv) in enumerate(zip(a_vals, b_vals)):
            x, y = project_to_panel(av, bv, a_min, a_max, b_min, b_max, panel_x + 8, panel_y + 8, panel_w - 16, panel_h - 16)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="#1f77b4" fill-opacity="0.9"/>')
            if j % 10 == 0:
                lines.append(f'<text x="{x + 3:.2f}" y="{y - 3:.2f}" font-size="8" fill="#444" font-family="Arial">{j}</text>')

        lines.append(f'<text x="{panel_x + 6}" y="{panel_y + panel_h + 16}" font-size="10" fill="#666" font-family="Arial">min/max A: {a_min:.3f} / {a_max:.3f}</text>')
        lines.append(f'<text x="{panel_x + 6}" y="{panel_y + panel_h + 30}" font-size="10" fill="#666" font-family="Arial">min/max B: {b_min:.3f} / {b_max:.3f}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_kpt3d_perspective_svg(out_path: Path, points: list[tuple[float, float, float]], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        out_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420"><text x="20" y="40" font-size="16">No kpt3d points</text></svg>', encoding="utf-8")
        return

    width, height = 420, 420
    cx, cy = width / 2, height / 2 + 10
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    mz = sum(zs) / len(zs)

    rx = 25.0 * 3.141592653589793 / 180.0
    ry = -35.0 * 3.141592653589793 / 180.0
    cosx, sinx = math.cos(rx), math.sin(rx)
    cosy, siny = math.cos(ry), math.sin(ry)
    dist = 4.0

    def to_view(x: float, y: float, z: float) -> tuple[float, float, float]:
        x -= mx
        y -= my
        z -= mz

        xr = x * cosy + z * siny
        zr = -x * siny + z * cosy
        yr = y * cosx - zr * sinx
        zr2 = y * sinx + zr * cosx

        denom = dist + zr2 * 0.35
        if denom < 0.2:
            denom = 0.2
        ux = xr / denom
        uy = yr / denom
        return zr2, ux, uy

    transformed: list[tuple[float, float, float, int]] = []
    for idx, (x, y, z) in enumerate(points):
        depth, ux, uy = to_view(x, y, z)
        transformed.append((depth, ux, uy, idx))

    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    z_span = max(zs) - min(zs)
    axis_world_len = max(x_span, y_span, z_span) * 0.35
    if axis_world_len < 1e-4:
        axis_world_len = 0.2

    axis_specs = [
        ("X", axis_world_len, 0.0, 0.0, "#ff4d4f"),
        ("Y", 0.0, axis_world_len, 0.0, "#22c55e"),
        ("Z", 0.0, 0.0, axis_world_len, "#3b82f6"),
    ]

    axis_proj: list[tuple[str, str, float, float, float, float]] = []
    o_depth, o_ux, o_uy = to_view(mx, my, mz)
    for label, dx, dy, dz, color in axis_specs:
        e_depth, e_ux, e_uy = to_view(mx + dx, my + dy, mz + dz)
        axis_proj.append((label, color, o_ux, o_uy, e_ux, e_uy))

    u_vals = [p[1] for p in transformed]
    v_vals = [p[2] for p in transformed]
    for _, _, su, sv, eu, ev in axis_proj:
        u_vals.extend([su, eu])
        v_vals.extend([sv, ev])
    u_min, u_max = min(u_vals), max(u_vals)
    v_min, v_max = min(v_vals), max(v_vals)
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)

    range_u = max(1e-6, u_max - u_min)
    range_v = max(1e-6, v_max - v_min)
    fit_scale_x = (width * 0.78) / range_u
    fit_scale_y = (height * 0.72) / range_v
    scale = min(fit_scale_x, fit_scale_y)

    projected: list[tuple[float, float, float, int]] = []
    for depth, ux, uy, idx in transformed:
        px = cx + (ux - u_mid) * scale
        py = cy - (uy - v_mid) * scale
        projected.append((depth, px, py, idx))

    axis_px: list[tuple[str, str, float, float, float, float]] = []
    for label, color, su, sv, eu, ev in axis_proj:
        sx = cx + (su - u_mid) * scale
        sy = cy - (sv - v_mid) * scale
        ex = cx + (eu - u_mid) * scale
        ey = cy - (ev - v_mid) * scale
        axis_px.append((label, color, sx, sy, ex, ey))

    projected.sort(key=lambda item: item[0])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for label, color, sx, sy, ex, ey in axis_px:
        lines.append(f'<line x1="{sx:.2f}" y1="{sy:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<text x="{ex + 4:.2f}" y="{ey - 4:.2f}" font-size="12" fill="{color}" font-family="Arial">{label}</text>')

    for depth, px, py, idx in projected:
        tone = max(70, min(220, int(140 + depth * 35)))
        color = f"rgb({tone},{min(255, tone + 20)},255)"
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.0" fill="{color}" fill-opacity="0.9"/>')
        if idx % 10 == 0:
            lines.append(f'<text x="{px + 3:.2f}" y="{py - 3:.2f}" font-size="8" fill="#444" font-family="Arial">{idx}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_kpt3d_perspective_matlab(
    out_png_path: Path,
    out_script_path: Path,
    points: list[tuple[float, float, float]],
    title: str,
    matlab_command: str,
) -> tuple[bool, str | None]:
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    out_script_path.parent.mkdir(parents=True, exist_ok=True)

    if not points:
        return False, "no points"

    def esc(s: str) -> str:
        return s.replace("'", "''")

    x_vals = "; ".join(f"{p[0]:.9f}" for p in points)
    y_vals = "; ".join(f"{p[1]:.9f}" for p in points)
    z_vals = "; ".join(f"{p[2]:.9f}" for p in points)

    script = f"""
X = [{x_vals}];
Y = [{y_vals}];
Z = [{z_vals}];

fig = figure('Visible','off','Color','w','Position',[100,100,900,900]);
scatter3(X, Y, Z, 22, [0.12 0.44 0.93], 'filled');
hold on;

mx = mean(X); my = mean(Y); mz = mean(Z);
sx = max(X)-min(X); sy = max(Y)-min(Y); sz = max(Z)-min(Z);
axis_len = max([sx, sy, sz]) * 0.35;
if axis_len < 1e-4
    axis_len = 0.2;
end

quiver3(mx, my, mz, axis_len, 0, 0, 0, 'LineWidth', 2.0, 'Color', [1.0 0.2 0.2]);
quiver3(mx, my, mz, 0, axis_len, 0, 0, 'LineWidth', 2.0, 'Color', [0.1 0.75 0.2]);
quiver3(mx, my, mz, 0, 0, axis_len, 0, 'LineWidth', 2.0, 'Color', [0.2 0.45 1.0]);
text(mx+axis_len, my, mz, 'X', 'Color', [1.0 0.2 0.2], 'FontSize', 12, 'FontWeight', 'bold');
text(mx, my+axis_len, mz, 'Y', 'Color', [0.1 0.75 0.2], 'FontSize', 12, 'FontWeight', 'bold');
text(mx, my, mz+axis_len, 'Z', 'Color', [0.2 0.45 1.0], 'FontSize', 12, 'FontWeight', 'bold');

xlabel('X'); ylabel('Y'); zlabel('Z');
title('{esc(title)}', 'Interpreter', 'none');
grid on;
axis equal;
view(35, 25);

pad = max([sx, sy, sz]) * 0.15;
if pad < 1e-4
    pad = 0.05;
end
xlim([min(X)-pad, max(X)+pad]);
ylim([min(Y)-pad, max(Y)+pad]);
zlim([min(Z)-pad, max(Z)+pad]);

exportgraphics(fig, '{esc(str(out_png_path.resolve()))}', 'Resolution', 220);
close(fig);
""".strip()

    out_script_path.write_text(script, encoding="utf-8")

    script_for_cmd = str(out_script_path.resolve()).replace("'", "''")
    batch_cmd = [matlab_command, "-batch", f"run('{script_for_cmd}')"]
    legacy_cmd = [matlab_command, "-nodisplay", "-nosplash", "-nodesktop", "-r", f"run('{script_for_cmd}');exit;"]

    try:
        proc = subprocess.run(batch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if proc.returncode == 0 and out_png_path.exists():
            return True, None

        proc2 = subprocess.run(legacy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if proc2.returncode == 0 and out_png_path.exists():
            return True, None
        return False, (proc2.stderr or proc.stderr or "matlab render failed").strip()
    except FileNotFoundError:
        return False, f"MATLAB command not found: {matlab_command}"
    except subprocess.TimeoutExpired:
        return False, "MATLAB render timeout"
    except Exception as ex:
        return False, str(ex)


def save_kpt_overlay_svg(
    out_path: Path,
    image_path: Path,
    width: int,
    height: int,
    points: list[tuple[float, float, float]],
    title: str,
    y_flip: bool = False,
    embed_image: bool = True,
    conf_threshold: float = 0.0,
    main_joint_indices: set[int] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if embed_image:
        png_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        href = f"data:image/png;base64,{png_b64}"
    else:
        href = os.path.relpath(image_path, out_path.parent).replace("\\", "/")
    points = normalize_points_to_pixels(points, width, height)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<image href="{href}" x="0" y="0" width="{width}" height="{height}"/>',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="white" stroke-width="1"/>',
        f'<text x="12" y="22" font-size="16" fill="yellow" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for i, (x, y, c) in enumerate(points):
        if c < conf_threshold:
            continue
        if main_joint_indices is not None and i not in main_joint_indices:
            continue
        if y_flip:
            y = (height - 1) - y
        color = "#00ff66" if c > 0 else "#ff3b30"
        r = 2.8 if c > 0 else 2.0
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" fill-opacity="0.9"/>')
        if i % 8 == 0:
            lines.append(f'<text x="{x + 3:.2f}" y="{y - 3:.2f}" font-size="8" fill="white" font-family="Arial">{i}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_local_image_for_overlay(svg_path: Path, src_image_path: Path) -> Path:
    local_image = svg_path.with_suffix(src_image_path.suffix)
    if local_image.exists():
        return local_image

    local_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_image_path, local_image)
    except Exception:
        try:
            os.symlink(src_image_path.resolve(), local_image)
        except Exception:
            shutil.copy2(src_image_path, local_image)
    return local_image


def validate_action(
    node: ActionNode,
    dataset_root: Path,
    sample_per_frame_shape_check: int,
    overlay_dir: Path,
    overlay_per_action: int,
    overlay_all_frames: bool,
    overlay_all_root: Path,
    viz_conf_threshold: float,
    viz_main_joint_indices: set[int] | None,
    viz_auto_filter_helper_joints: bool,
    kpt3d_viz_root: Path,
    kpt3d_viz_frames: int,
    kpt3d_viz_all_frames: bool,
    kpt3d_3d_backend: str,
    matlab_command: str,
) -> ActionReport:
    action_dir = node.action_path
    report = ActionReport(action_name=node.action_name, action_path=str(action_dir), character_name=node.character_name)

    meta_seq = action_dir / "meta" / "sequence.json"
    if meta_seq.exists():
        seq = read_json(meta_seq)
        if seq is None:
            report.issues.append(Issue("ERROR", "action", "meta/sequence.json 无法解析为合法 JSON"))
        else:
            report.has_meta_sequence = True
            report.sequence = seq
    else:
        report.issues.append(Issue("ERROR", "action", "缺少 meta/sequence.json"))

    kpt3d_npy = action_dir / "kpt3d" / "kpt3d.npy"
    report.has_kpt3d_npy = kpt3d_npy.exists()
    if report.has_kpt3d_npy:
        report.kpt3d_shape = parse_npy_shape(kpt3d_npy)
        if report.kpt3d_shape is None:
            report.issues.append(Issue("ERROR", "action", "kpt3d.npy 形状读取失败"))
    else:
        report.issues.append(Issue("WARN", "action", "缺少 kpt3d/kpt3d.npy"))

    kpt3d_npz = action_dir / "kpt3d" / "kpt3d.npz"
    report.has_kpt3d_npz = kpt3d_npz.exists()
    selected_kpt3d_shape: list[int] | None = None
    selected_kpt3d_data: list[float] | None = None
    if report.has_kpt3d_npz:
        report.kpt3d_npz_arrays = parse_npz_shapes(kpt3d_npz)
        if not report.kpt3d_npz_arrays:
            report.issues.append(Issue("ERROR", "action", "kpt3d.npz 存在但无法解析"))

    if report.has_kpt3d_npy:
        shp, data = parse_npy_float32_data(kpt3d_npy)
        if shp and data is not None and len(shp) == 3 and shp[2] == 3:
            selected_kpt3d_shape, selected_kpt3d_data = shp, data

    if selected_kpt3d_data is None and report.has_kpt3d_npz:
        npz_float_arrays = parse_npz_float32_arrays(kpt3d_npz)
        preferred_names = ["kpt3d.npy", "kpt3d", "arr_0.npy", "arr_0"]
        ordered_names = [n for n in preferred_names if n in npz_float_arrays] + [n for n in npz_float_arrays.keys() if n not in preferred_names]
        for name in ordered_names:
            shape, data = npz_float_arrays[name]
            if len(shape) == 3 and shape[2] == 3:
                selected_kpt3d_shape, selected_kpt3d_data = shape, data
                break

    if selected_kpt3d_shape and selected_kpt3d_data is not None and (kpt3d_viz_all_frames or kpt3d_viz_frames > 0):
        safe_char = re.sub(r"[^A-Za-z0-9_.-]+", "_", node.character_name or "character")
        safe_action = re.sub(r"[^A-Za-z0-9_.-]+", "_", report.action_name)
        matlab_warned = False
        if kpt3d_viz_all_frames:
            frame_ids = list(range(selected_kpt3d_shape[0]))
        else:
            frame_ids = pick_frame_indices(selected_kpt3d_shape[0], kpt3d_viz_frames)
        for fi in frame_ids:
            pts3d = reshape_kpt3d_points(selected_kpt3d_shape, selected_kpt3d_data, fi)
            base = kpt3d_viz_root / safe_char / safe_action / f"kpt3d_frame_{fi:06d}"

            p3d_svg = base.with_name(base.name + "_3d").with_suffix(".svg")
            p3d_png = base.with_name(base.name + "_3d").with_suffix(".png")
            p3d_m = base.with_name(base.name + "_3d_render").with_suffix(".m")
            p3views = base.with_name(base.name + "_3views").with_suffix(".svg")

            title_3d = f"{report.action_name} | kpt3d frame {fi:06d} | 3D"
            if kpt3d_3d_backend == "matlab":
                ok, err = save_kpt3d_perspective_matlab(p3d_png, p3d_m, pts3d, title_3d, matlab_command)
                if ok:
                    report.kpt3d_visuals.append(str(p3d_png))
                else:
                    if not matlab_warned:
                        report.issues.append(Issue("WARN", "action", f"MATLAB 3D render unavailable: {err}; fallback to SVG for remaining frames"))
                        matlab_warned = True
                    save_kpt3d_perspective_svg(p3d_svg, pts3d, title_3d)
                    report.kpt3d_visuals.append(str(p3d_svg))
            else:
                save_kpt3d_perspective_svg(p3d_svg, pts3d, title_3d)
                report.kpt3d_visuals.append(str(p3d_svg))

            save_kpt3d_three_views_svg(p3views, pts3d, f"{report.action_name} | kpt3d frame {fi:06d} | XY/XZ/YZ")
            report.kpt3d_visuals.append(str(p3views))

    camera_ids: set[str] = set()
    frames_root = action_dir / "frames"
    kpt2d_root = action_dir / "kpt2d"
    action_cam_root = action_dir / "cameras"

    if action_cam_root.exists():
        camera_ids.update(d.name for d in action_cam_root.iterdir() if d.is_dir())
    if frames_root.exists():
        for d in frames_root.iterdir():
            if d.is_dir() and d.name.startswith("capture_"):
                camera_ids.add(d.name.replace("capture_", "", 1))
    if kpt2d_root.exists():
        camera_ids.update(d.name for d in kpt2d_root.iterdir() if d.is_dir())

    if not camera_ids:
        report.issues.append(Issue("ERROR", "action", "未发现任何相机目录（cameras/frames/kpt2d）"))
        return report

    expected_sampled: int | None = None
    expected_joints: int | None = None
    if report.sequence:
        sampled_val = report.sequence.get("sampled_frames")
        joints_val = report.sequence.get("joints_count")

        if sampled_val is not None:
            expected_sampled = int(sampled_val)
        if joints_val is not None:
            expected_joints = int(joints_val)

    shared_cam_root = resolve_shared_camera_root(dataset_root, node)

    for cam_id in sorted(camera_ids):
        cam = CameraReport(camera_id=cam_id)

        action_intr = action_cam_root / cam_id / "intrinsics.json"
        action_extr = action_cam_root / cam_id / "extrinsics.json"
        shared_intr = shared_cam_root / cam_id / "intrinsics.json"
        shared_extr = shared_cam_root / cam_id / "extrinsics.json"

        if action_intr.exists() or action_extr.exists():
            intr, extr = action_intr, action_extr
            cam.camera_meta_source = "action"
        else:
            intr, extr = shared_intr, shared_extr
            cam.camera_meta_source = "shared"

        cam.has_intrinsics = intr.exists()
        cam.has_extrinsics = extr.exists()
        if not cam.has_intrinsics:
            cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 intrinsics.json（action/shared cameras）"))
        if not cam.has_extrinsics:
            cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 extrinsics.json（action/shared cameras）"))

        frame_dir = frames_root / f"capture_{cam_id}"
        frame_files = sorted(frame_dir.glob("frame_*.png")) if frame_dir.exists() else []
        frame_indices = extract_indices(frame_files, RE_FRAME)
        cam.frame_count = len(frame_files)
        if frame_indices:
            cam.frame_indices_min = frame_indices[0]
            cam.frame_indices_max = frame_indices[-1]
            cam.frame_index_gaps = count_gaps(frame_indices)
            if cam.frame_index_gaps > 0:
                cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"frame 序号存在 {cam.frame_index_gaps} 处间断"))
        else:
            cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "未找到 frame_*.png"))

        kpt2d_dir = kpt2d_root / cam_id
        kpt2d_main, kpt2d_npz, per_frame_files, kpt2d_prefix = detect_kpt2d_files(kpt2d_dir)

        cam.has_kpt2d_npy = kpt2d_main is not None and kpt2d_main.exists()
        if cam.has_kpt2d_npy and kpt2d_main is not None:
            cam.kpt2d_shape = parse_npy_shape(kpt2d_main)
            if cam.kpt2d_shape is None:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"{kpt2d_main.name} shape 读取失败"))
        else:
            if per_frame_files:
                cam.issues.append(Issue("WARN", f"camera:{cam_id}", "缺少主 kpt npy（但存在 *_NNNNNN.npy）"))
            else:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 kpt2d 主文件与逐帧文件"))

        cam.has_kpt2d_npz = kpt2d_npz is not None and kpt2d_npz.exists()
        npz_kpt_shape: list[int] | None = None
        npz_kpt_data: list[float] | None = None
        if cam.has_kpt2d_npz and kpt2d_npz is not None:
            cam.kpt2d_npz_arrays = parse_npz_shapes(kpt2d_npz)
            if not cam.kpt2d_npz_arrays:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"{kpt2d_npz.name} 存在但无法解析"))
            else:
                npz_float_arrays = parse_npz_float32_arrays(kpt2d_npz)
                preferred_names = ["kpt2d.npy", "kpt2d", "arr_0.npy", "arr_0"]
                ordered_names = [n for n in preferred_names if n in npz_float_arrays] + [n for n in npz_float_arrays.keys() if n not in preferred_names]
                for name in ordered_names:
                    shape, data = npz_float_arrays[name]
                    if (len(shape) == 3 and shape[2] == 3) or (len(shape) == 2 and shape[1] == 3):
                        npz_kpt_shape, npz_kpt_data = shape, data
                        break

        per_idx = extract_indices(per_frame_files, RE_KPT2D_PER_FRAME)
        if not per_idx:
            per_idx = sorted(
                int(m.group(2))
                for p in per_frame_files
                for m in [RE_GENERIC_PER_FRAME_NPY.match(p.name)]
                if m is not None
            )
        cam.kpt2d_per_frame_count = len(per_frame_files)
        cam.kpt2d_per_frame_gaps = count_gaps(per_idx)
        if per_frame_files and cam.kpt2d_per_frame_gaps > 0:
            cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"kpt2d_*.npy 序号存在 {cam.kpt2d_per_frame_gaps} 处间断"))

        if per_frame_files:
            check_files = per_frame_files if sample_per_frame_shape_check <= 0 else per_frame_files[:sample_per_frame_shape_check]
            shape_ok = True
            for p in check_files:
                shp = parse_npy_shape(p)
                if shp is None or len(shp) != 2 or shp[1] != 3:
                    shape_ok = False
                    cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"{p.name} 形状异常，期望 (J,3)"))
                    break
            cam.kpt2d_per_frame_shape_ok = shape_ok

        if expected_sampled is not None:
            cam.frame_vs_expected_delta = cam.frame_count - expected_sampled
            if cam.frame_count != expected_sampled:
                cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"frame 数({cam.frame_count}) != sampled_frames({expected_sampled})"))

        if cam.kpt2d_shape and len(cam.kpt2d_shape) == 3 and expected_joints is not None:
            if cam.kpt2d_shape[1] != expected_joints or cam.kpt2d_shape[2] != 3:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"kpt2d.npy shape={tuple(cam.kpt2d_shape)} 与 joints_count={expected_joints} 不一致"))

        main_kpt_shape: list[int] | None = None
        main_kpt_data: list[float] | None = None
        if kpt2d_main is not None and kpt2d_main.exists():
            main_kpt_shape, main_kpt_data = parse_npy_float32_data(kpt2d_main)

        def get_points_for_frame(frame_index: int) -> list[tuple[float, float, float]]:
            if npz_kpt_shape and npz_kpt_data is not None:
                cam.overlay_point_source = "npz"
                return reshape_kpt2d_points(npz_kpt_shape, npz_kpt_data, frame_index)

            per_frame_kpt = resolve_kpt2d_frame_file(kpt2d_dir, frame_index, kpt2d_prefix)
            if per_frame_kpt is not None:
                shp, data = parse_npy_float32_data(per_frame_kpt)
                if shp and data is not None:
                    cam.overlay_point_source = f"npy-per-frame:{per_frame_kpt.parent.name}"
                    return reshape_kpt2d_points(shp, data)

            if main_kpt_shape and main_kpt_data is not None:
                cam.overlay_point_source = "npy-main"
                return reshape_kpt2d_points(main_kpt_shape, main_kpt_data, frame_index)

            return []

        report.camera_reports.append(cam)

    if report.kpt3d_shape and report.sequence:
        sampled = report.sequence.get("sampled_frames")
        joints = report.sequence.get("joints_count")
        shp = report.kpt3d_shape
        if len(shp) != 3 or shp[2] != 3:
            report.issues.append(Issue("ERROR", "action", f"kpt3d.npy shape={tuple(shp)} 非法，期望 (T,J,3)"))
        else:
            if sampled is not None and shp[0] != int(sampled):
                report.issues.append(Issue("WARN", "action", f"kpt3d T={shp[0]} != sampled_frames={sampled}"))
            if joints is not None and shp[1] != int(joints):
                report.issues.append(Issue("WARN", "action", f"kpt3d J={shp[1]} != joints_count={joints}"))

    return report


def save_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width = max(1000, 80 + 26 * max(1, len(labels)))
    height = 520
    left, right, top, bottom = 70, 30, 60, 150
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    bars = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + i * (plot_w / max(1, len(labels))) + 2
        bw = max(4, (plot_w / max(1, len(labels))) - 4)
        bh = (val / max_v) * plot_h
        y = top + (plot_h - bh)
        bars.append((x, y, bw, bh, lab, val))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{svg_escape(title)}</text>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20,{top+plot_h/2})" text-anchor="middle" font-size="12" font-family="Arial">{svg_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]

    for t in range(6):
        v = max_v * t / 5
        y = top + plot_h - (plot_h * t / 5)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#eee"/>')
        lines.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial">{int(v)}</text>')

    for x, y, bw, bh, lab, val in bars:
        lines.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="#4e79a7"/>')
        lines.append(f'<text x="{x+bw/2}" y="{y-4}" text-anchor="middle" font-size="10" font-family="Arial">{int(val)}</text>')
        lines.append(
            f'<text x="{x+bw/2}" y="{top+plot_h+14}" text-anchor="end" transform="rotate(-65 {x+bw/2},{top+plot_h+14})" '
            f'font-size="10" font-family="Arial">{svg_escape(lab)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_svg_issue_summary(path: Path, errors: int, warns: int, infos: int) -> None:
    labels = ["ERROR", "WARN", "INFO"]
    values = [errors, warns, infos]
    colors = ["#e15759", "#f28e2b", "#59a14f"]

    width, height = 640, 420
    left, top = 80, 70
    plot_w, plot_h = 500, 260
    max_v = max(values) if max(values) > 0 else 1
    bw = 100
    gap = (plot_w - bw * 3) / 4

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="320" y="30" text-anchor="middle" font-size="20" font-family="Arial">Issue Severity Summary</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]

    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x = left + gap + i * (bw + gap)
        bh = 0 if max_v == 0 else (val / max_v) * plot_h
        y = top + (plot_h - bh)
        lines.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{col}"/>')
        lines.append(f'<text x="{x+bw/2}" y="{y-6}" text-anchor="middle" font-size="12" font-family="Arial">{val}</text>')
        lines.append(f'<text x="{x+bw/2}" y="{top+plot_h+24}" text-anchor="middle" font-size="12" font-family="Arial">{lab}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_visuals(report: FullReport, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"
    overlay_dir = fig_dir / "overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    sev_path = fig_dir / "issues_summary.svg"
    save_svg_issue_summary(sev_path, report.summary.error_count, report.summary.warn_count, report.summary.info_count)
    generated.append(str(sev_path))

    for action in report.actions:
        cams = sorted(action.camera_reports, key=lambda c: c.camera_id)
        if cams:
            labels = [c.camera_id for c in cams]
            frame_vals = [float(c.frame_count) for c in cams]
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.action_name)

            p1 = fig_dir / f"{safe_name}_frames_per_camera.svg"
            save_svg_bar_chart(p1, f"{action.action_name} - Frames per Camera", labels, frame_vals, "#frames")
            generated.append(str(p1))

        for cam in cams:
            if cam.overlay_svg:
                generated.append(cam.overlay_svg)

        for p in action.kpt3d_visuals:
            generated.append(p)

    return generated


def build_report(actions: list[ActionReport], dataset_root: Path) -> FullReport:
    summary = Summary(checked_at_utc=datetime.utcnow().isoformat() + "Z", dataset_root=str(dataset_root))
    summary.action_count = len(actions)
    summary.camera_count = sum(len(a.camera_reports) for a in actions)

    all_issues = [i for a in actions for i in (a.issues + [ii for c in a.camera_reports for ii in c.issues])]
    summary.error_count = sum(1 for i in all_issues if i.severity == "ERROR")
    summary.warn_count = sum(1 for i in all_issues if i.severity == "WARN")
    summary.info_count = sum(1 for i in all_issues if i.severity == "INFO")
    summary.passed = summary.error_count == 0

    return FullReport(summary=summary, actions=actions)


def write_report_files(report: FullReport, report_dir: Path, stamp: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    report.visuals = generate_visuals(report, report_dir)

    json_path = report_dir / f"dataset_check_report_{stamp}.json"
    md_path = report_dir / f"dataset_check_report_{stamp}.md"

    json_path.write_text(
        json.dumps({"summary": asdict(report.summary), "visuals": report.visuals, "actions": [asdict(a) for a in report.actions]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path


def to_markdown(report: FullReport) -> str:
    s = report.summary
    lines: list[str] = []
    lines.append("# Dataset Validation Report")
    lines.append("")
    lines.append(f"- Checked at (UTC): {s.checked_at_utc}")
    lines.append(f"- Dataset root: {s.dataset_root}")
    lines.append(f"- Actions: {s.action_count}")
    lines.append(f"- Cameras: {s.camera_count}")
    lines.append(f"- Errors: {s.error_count}")
    lines.append(f"- Warnings: {s.warn_count}")
    lines.append(f"- Infos: {s.info_count}")
    lines.append(f"- Passed: {s.passed}")
    lines.append("")

    if report.visuals:
        lines.append("## Visualizations")
        lines.append("")
        for v in report.visuals:
            lines.append(f"- {v}")
        lines.append("")

    for action in report.actions:
        lines.append(f"## Action: {action.action_name}")
        lines.append("")
        lines.append(f"- Character: {action.character_name or '(none)'}")
        lines.append(f"- Path: {action.action_path}")
        if action.sequence:
            lines.append(
                "- Sequence: "
                f"total_frames={action.sequence.get('total_frames')}, "
                f"sampled_frames={action.sequence.get('sampled_frames')}, "
                f"joints_count={action.sequence.get('joints_count')}, "
                f"pose_every_n_frames={action.sequence.get('pose_every_n_frames')}"
            )
        else:
            lines.append("- Sequence: (missing)")

        lines.append(f"- kpt3d.npy: {'Y' if action.has_kpt3d_npy else 'N'} shape={tuple(action.kpt3d_shape) if action.kpt3d_shape else '-'}")
        lines.append(f"- kpt3d.npz: {'Y' if action.has_kpt3d_npz else 'N'} arrays={list(action.kpt3d_npz_arrays.keys()) if action.kpt3d_npz_arrays else []}")
        lines.append(f"- kpt3d visuals: {len(action.kpt3d_visuals)}")
        for p in action.kpt3d_visuals:
            lines.append(f"  - {p}")

        frame_counts = [c.frame_count for c in action.camera_reports]
        if frame_counts:
            lines.append(f"- Frame stats: min={min(frame_counts)}, max={max(frame_counts)}, mean={statistics.mean(frame_counts):.2f}")

        lines.append("")
        lines.append("### Cameras")
        lines.append("")
        lines.append("| Camera | MetaSrc | Intr | Extr | Frames | FrameRange | Gap# | kpt2d.npy | kpt2d.npz | kpt2d per-frame | Overlay(all) | Overlay src | Overlay(sample) |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|")
        for cam in sorted(action.camera_reports, key=lambda c: c.camera_id):
            fr = "-"
            if cam.frame_indices_min is not None and cam.frame_indices_max is not None:
                fr = f"{cam.frame_indices_min}..{cam.frame_indices_max}"
            ov = cam.overlay_svg if cam.overlay_svg else "-"
            lines.append(
                f"| {cam.camera_id} | {cam.camera_meta_source or '-'} | {'Y' if cam.has_intrinsics else 'N'} | {'Y' if cam.has_extrinsics else 'N'} | "
                f"{cam.frame_count} | {fr} | {cam.frame_index_gaps} | {'Y' if cam.has_kpt2d_npy else 'N'} | {'Y' if cam.has_kpt2d_npz else 'N'} | {cam.kpt2d_per_frame_count} | {cam.overlay_all_count} | {cam.overlay_point_source or '-'} | {ov} |"
            )

        all_issues = action.issues + [i for c in action.camera_reports for i in c.issues]
        lines.append("")
        lines.append("### Issues")
        lines.append("")
        if not all_issues:
            lines.append("- None")
        else:
            for issue in all_issues:
                lines.append(f"- [{issue.severity}] ({issue.scope}) {issue.message}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    viz_main_joint_indices: set[int] | None = None
    if args.viz_main_joints.strip():
        try:
            viz_main_joint_indices = {int(part.strip()) for part in args.viz_main_joints.split(",") if part.strip()}
        except ValueError:
            print("[check.py] ERROR: --viz-main-joints 格式非法，应为逗号分隔整数，如 0,1,2,5")
            return 2

    nodes = discover_action_nodes(dataset_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not nodes:
        report = build_report(
            [ActionReport(action_name="(none)", action_path=str(dataset_root), issues=[Issue("ERROR", "dataset", "未发现动作目录")])],
            dataset_root,
        )
    else:
        overlay_dir = output_dir / "figures" / "overlays"
        overlay_all_root = output_dir / "figures" / "overlays_all"
        kpt3d_viz_root = output_dir / "figures" / "kpt3d"
        actions = [
            validate_action(
                n,
                dataset_root,
                args.sample_per_frame_shape_check,
                overlay_dir,
                args.overlay_per_action,
                args.overlay_all_frames,
                overlay_all_root,
                args.viz_conf_threshold,
                viz_main_joint_indices,
                args.viz_auto_filter_helper_joints,
                kpt3d_viz_root,
                args.kpt3d_viz_frames,
                args.kpt3d_viz_all_frames,
                args.kpt3d_3d_backend,
                args.matlab_command,
            )
            for n in nodes
        ]
        report = build_report(actions, dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path, md_path = write_report_files(report, output_dir, stamp)

    if any(a.character_name for a in report.actions):
        for character_name in sorted({a.character_name for a in report.actions if a.character_name}):
            character_actions = [a for a in report.actions if a.character_name == character_name]
            character_report = build_report(character_actions, dataset_root)
            character_dir = output_dir / character_name
            write_report_files(character_report, character_dir, stamp)

    print(f"[check.py] dataset_root = {dataset_root}")
    print(f"[check.py] actions = {report.summary.action_count}, cameras = {report.summary.camera_count}")
    print(f"[check.py] errors = {report.summary.error_count}, warnings = {report.summary.warn_count}, passed = {report.summary.passed}")
    print(f"[check.py] report(json) = {json_path}")
    print(f"[check.py] report(md)   = {md_path}")
    print(f"[check.py] visuals = {len(report.visuals)} files in {output_dir / 'figures'}")

    return 0 if report.summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
