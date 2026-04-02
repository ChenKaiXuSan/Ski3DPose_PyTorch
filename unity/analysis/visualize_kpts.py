#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import math
import re
import struct
from pathlib import Path


RE_FRAME_PNG = re.compile(r"^frame_(\d+)\.png$")
RE_KPT2D_NPY = re.compile(r"^kpt2d_(\d+)\.npy$")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ski dataset 2D keypoints on frames and 3D keypoints without third-party packages.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=resolve_default_dataset_root(),
        help="Dataset root path (default: auto-detected SkiDataset location)",
    )
    parser.add_argument(
        "--character",
        type=str,
        default="all",
        choices=["all", "male", "female"],
        help="Character split. Default: all",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="",
        help="Action folder name, e.g. Anim_Male_Skier_Braking. Empty means all actions.",
    )
    parser.add_argument("--camera", type=str, default="", help="Camera id, e.g. L0_A000. Empty picks the first available camera")
    parser.add_argument(
        "--frames",
        type=str,
        default="0,16,32",
        help="Comma separated frame ids to render, e.g. 0,10,20. Out-of-range values are ignored.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "reports" / "figures" / "custom_viz"),
        help="Output directory",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Only draw 2D points with conf >= threshold",
    )
    parser.add_argument(
        "--y-flip",
        action="store_true",
        default=False,
        help="Flip Y for keypoints before drawing (for normalized coordinates in some pipelines)",
    )
    return parser.parse_args()


def parse_npy_header_from_bytes(raw: bytes) -> tuple[str | None, list[int] | None, int | None]:
    try:
        if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
            return None, None, None

        major = raw[6]
        if major == 1:
            hlen = int.from_bytes(raw[8:10], "little")
            header_start = 10
        elif major in (2, 3):
            hlen = int.from_bytes(raw[8:12], "little")
            header_start = 12
        else:
            return None, None, None

        header = raw[header_start : header_start + hlen].decode("latin1")
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


def parse_npy_float32(path: Path) -> tuple[list[int] | None, list[float] | None]:
    try:
        raw = path.read_bytes()
        dtype, shape, data_offset = parse_npy_header_from_bytes(raw)
        if dtype not in ("<f4", "|f4") or shape is None or data_offset is None:
            return None, None

        count = 1
        for dim in shape:
            count *= dim

        needed = data_offset + 4 * count
        if needed > len(raw):
            return None, None

        data = struct.unpack("<" + "f" * count, raw[data_offset:needed])
        return shape, list(data)
    except Exception:
        return None, None


def parse_png_size(path: Path) -> tuple[int, int] | None:
    try:
        raw = path.read_bytes()
        if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        if raw[12:16] != b"IHDR":
            return None
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        return width, height
    except Exception:
        return None


def list_camera_ids(action_dir: Path) -> list[str]:
    kpt2d_root = action_dir / "kpt2d"
    if not kpt2d_root.exists():
        return []
    return sorted([d.name for d in kpt2d_root.iterdir() if d.is_dir()])


def resolve_kpt2d_variant_paths(kpt2d_dir: Path, frame_id: int) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for variant in KPT2D_VARIANT_NAMES:
        variant_dir = kpt2d_dir / variant
        if not variant_dir.is_dir():
            continue
        candidate = variant_dir / f"kpt2d_{frame_id:06d}.npy"
        if candidate.exists():
            out[variant] = candidate

    if not out:
        candidate = kpt2d_dir / f"kpt2d_{frame_id:06d}.npy"
        if candidate.exists():
            out["kpt2d"] = candidate

    return out


def resolve_kpt3d_variant_paths(kpt3d_dir: Path, frame_id: int) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for variant in KPT2D_VARIANT_NAMES:
        variant_dir = kpt3d_dir / variant
        if not variant_dir.is_dir():
            continue
        candidate = variant_dir / f"frame_{frame_id:06d}.npy"
        if candidate.exists():
            out[variant] = candidate

    if not out:
        candidate = kpt3d_dir / f"frame_{frame_id:06d}.npy"
        if candidate.exists():
            out["kpt3d"] = candidate

    return out


def discover_actions(character_root: Path) -> list[str]:
    if not character_root.exists():
        return []
    out: list[str] = []
    for p in sorted(character_root.iterdir()):
        if p.is_dir() and (p / "frames").exists() and (p / "kpt2d").exists() and (p / "kpt3d").exists():
            out.append(p.name)
    return out


def list_frame_ids(frame_dir: Path) -> list[int]:
    out: list[int] = []
    if not frame_dir.exists():
        return out
    for p in frame_dir.iterdir():
        m = RE_FRAME_PNG.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def reshape_2d_points(shape: list[int], data: list[float], frame_idx: int = 0) -> list[tuple[float, float, float]]:
    if len(shape) == 2 and shape[1] == 3:
        pts: list[tuple[float, float, float]] = []
        for j in range(shape[0]):
            base = j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    if len(shape) == 3 and shape[2] == 3:
        total_frames = shape[0]
        joints = shape[1]
        idx = max(0, min(frame_idx, total_frames - 1))
        start = idx * joints * 3
        pts = []
        for j in range(joints):
            base = start + j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    return []


def reshape_3d_points(shape: list[int], data: list[float], frame_idx: int = 0) -> list[tuple[float, float, float]]:
    if len(shape) != 3 or shape[2] != 3:
        return []
    total_frames, joints, _ = shape
    if total_frames <= 0 or joints <= 0:
        return []

    idx = max(0, min(frame_idx, total_frames - 1))
    start = idx * joints * 3
    pts: list[tuple[float, float, float]] = []
    for j in range(joints):
        base = start + j * 3
        pts.append((data[base], data[base + 1], data[base + 2]))
    return pts


def normalize_points_to_pixels(points: list[tuple[float, float, float]], width: int, height: int) -> list[tuple[float, float, float]]:
    if not points:
        return points

    max_abs = max(max(abs(p[0]) for p in points), max(abs(p[1]) for p in points))

    # Treat values in [0,1] or [-1,1]-like ranges as normalized coordinates.
    if max_abs <= 2.0:
        out: list[tuple[float, float, float]] = []
        for x, y, c in points:
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                out.append((x * width, y * height, c))
            else:
                out.append(((x + 1.0) * 0.5 * width, (y + 1.0) * 0.5 * height, c))
        return out

    return points


def svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_2d_overlay_svg(
    out_svg: Path,
    frame_png: Path,
    width: int,
    height: int,
    points: list[tuple[float, float, float]],
    title: str,
    conf_threshold: float,
    y_flip: bool,
) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    href = "data:image/png;base64," + base64.b64encode(frame_png.read_bytes()).decode("ascii")

    pts = normalize_points_to_pixels(points, width, height)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<image href="{href}" x="0" y="0" width="{width}" height="{height}"/>',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="white" stroke-width="1"/>',
        f'<text x="12" y="24" font-size="18" fill="#fff34d" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for idx, (x, y, conf) in enumerate(pts):
        if conf < conf_threshold:
            continue
        draw_y = (height - y) if y_flip else y
        color = "#00ff66" if conf > 0 else "#ff4d4f"
        radius = 3.0 if conf > 0 else 2.0
        lines.append(f'<circle cx="{x:.2f}" cy="{draw_y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.92"/>')
        if idx % 8 == 0:
            lines.append(
                f'<text x="{x + 3:.2f}" y="{draw_y - 3:.2f}" font-size="10" fill="{color}" font-family="Arial">{idx}</text>'
            )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def project_to_panel(
    a: float,
    b: float,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> tuple[float, float]:
    ar = a_max - a_min
    br = b_max - b_min
    na = 0.5 if abs(ar) < 1e-9 else (a - a_min) / ar
    nb = 0.5 if abs(br) < 1e-9 else (b - b_min) / br
    px = panel_x + na * panel_w
    py = panel_y + (1.0 - nb) * panel_h
    return px, py


def save_3d_three_views_svg(out_svg: Path, points: list[tuple[float, float, float]], title: str) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    if not points:
        out_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="360"><text x="20" y="40" font-size="16">No 3D points</text></svg>',
            encoding="utf-8",
        )
        return

    width, height = 980, 360
    margin, gap = 24, 18
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
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for i, (label, a_vals, b_vals, a_min, a_max, b_min, b_max) in enumerate(panels):
        panel_x = margin + i * (panel_w + gap)
        lines.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#f9f9fb" stroke="#d8d8de"/>'
        )
        lines.append(
            f'<text x="{panel_x + panel_w / 2}" y="{panel_y - 10}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )

        for j, (av, bv) in enumerate(zip(a_vals, b_vals)):
            px, py = project_to_panel(av, bv, a_min, a_max, b_min, b_max, panel_x, panel_y, panel_w, panel_h)
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.8" fill="#2a72d8" fill-opacity="0.9"/>')
            if j % 10 == 0:
                lines.append(f'<text x="{px + 3:.2f}" y="{py - 3:.2f}" font-size="9" fill="#1f4fa0" font-family="Arial">{j}</text>')

        lines.append(
            f'<text x="{panel_x + 6}" y="{panel_y + panel_h + 16}" font-size="10" fill="#666" font-family="Arial">min/max A: {a_min:.3f} / {a_max:.3f}</text>'
        )
        lines.append(
            f'<text x="{panel_x + 6}" y="{panel_y + panel_h + 30}" font-size="10" fill="#666" font-family="Arial">min/max B: {b_min:.3f} / {b_max:.3f}</text>'
        )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def save_3d_perspective_svg(out_svg: Path, points: list[tuple[float, float, float]], title: str) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    if not points:
        out_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420"><text x="20" y="40" font-size="16">No 3D points</text></svg>',
            encoding="utf-8",
        )
        return

    width, height = 420, 420
    cx, cy = width / 2, height / 2 + 10

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    mz = sum(zs) / len(zs)

    rx = math.radians(25.0)
    ry = math.radians(-35.0)
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

    u_vals = [p[1] for p in transformed]
    v_vals = [p[2] for p in transformed]
    u_min, u_max = min(u_vals), max(u_vals)
    v_min, v_max = min(v_vals), max(v_vals)
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)

    scale = min((width * 0.78) / max(1e-6, u_max - u_min), (height * 0.72) / max(1e-6, v_max - v_min))

    projected: list[tuple[float, float, float, int]] = []
    for depth, ux, uy, idx in transformed:
        px = cx + (ux - u_mid) * scale
        py = cy - (uy - v_mid) * scale
        projected.append((depth, px, py, idx))

    projected.sort(key=lambda x: x[0])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="16" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for depth, px, py, idx in projected:
        tone = max(70, min(220, int(140 + depth * 35)))
        color = f"rgb({tone},{min(255, tone + 20)},255)"
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.1" fill="{color}" fill-opacity="0.9"/>')
        if idx % 10 == 0:
            lines.append(f'<text x="{px + 4:.2f}" y="{py - 4:.2f}" font-size="9" fill="#333" font-family="Arial">{idx}</text>')

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def parse_frame_ids(frames_text: str) -> list[int]:
    out: list[int] = []
    for item in frames_text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return sorted(set(out))


def resolve_targets(dataset_root: Path, character_arg: str, action_arg: str) -> list[tuple[str, str, Path]]:
    char_list = ["male", "female"] if character_arg == "all" else [character_arg]
    targets: list[tuple[str, str, Path]] = []
    for character in char_list:
        char_root = dataset_root / character
        if not char_root.exists():
            continue

        actions = [action_arg] if action_arg.strip() else discover_actions(char_root)
        for action in actions:
            action_dir = char_root / action
            if action_dir.exists() and action_dir.is_dir():
                targets.append((character, action, action_dir))
    return targets


def main() -> int:
    args = parse_args()

    targets = resolve_targets(args.dataset_root, args.character, args.action)
    if not targets:
        print("[ERROR] no valid character/action targets found.")
        return 2

    requested_ids_raw = parse_frame_ids(args.frames)
    if not requested_ids_raw:
        print("[ERROR] --frames is empty or invalid.")
        return 2

    total_2d = 0
    total_3d_frames = 0
    total_targets = 0

    for character, action, action_dir in targets:
        camera_ids = list_camera_ids(action_dir)
        if not camera_ids:
            print(f"[WARN] skip {character}/{action}: no camera found under {action_dir / 'kpt2d'}")
            continue

        camera_id = args.camera.strip() or camera_ids[0]
        if camera_id not in camera_ids:
            print(f"[WARN] skip {character}/{action}: camera '{camera_id}' not found")
            continue

        frame_dir = action_dir / "frames" / f"capture_{camera_id}"
        kpt2d_dir = action_dir / "kpt2d" / camera_id
        kpt3d_dir = action_dir / "kpt3d"

        if not frame_dir.exists() or not kpt2d_dir.exists() or not kpt3d_dir.exists():
            print(f"[WARN] skip {character}/{action}: missing frames/kpt2d/kpt3d path")
            continue

        available_frame_ids = set(list_frame_ids(frame_dir))
        target_ids = [fid for fid in requested_ids_raw if fid in available_frame_ids]
        if not target_ids:
            print(f"[WARN] skip {character}/{action}: no valid frame ids in {requested_ids_raw}")
            continue

        out_root = args.out_dir / character / action / camera_id

        rendered_2d = 0
        rendered_3d = 0
        for frame_id in target_ids:
            frame_png = frame_dir / f"frame_{frame_id:06d}.png"
            variant_paths = resolve_kpt2d_variant_paths(kpt2d_dir, frame_id)
            kpt3d_variant_paths = resolve_kpt3d_variant_paths(kpt3d_dir, frame_id)
            if not frame_png.exists() or not variant_paths or not kpt3d_variant_paths:
                continue

            size = parse_png_size(frame_png)
            if not size:
                continue
            width, height = size

            for variant_name, kpt2d_npy in variant_paths.items():
                shape, data = parse_npy_float32(kpt2d_npy)
                if shape is None or data is None:
                    continue

                points2d = reshape_2d_points(shape, data)
                title = f"{character}/{action}/{camera_id}/{variant_name} frame={frame_id:06d}"
                out_2d = out_root / variant_name / "2d"
                out_2d.mkdir(parents=True, exist_ok=True)
                save_2d_overlay_svg(
                    out_2d / f"overlay_{frame_id:06d}.svg",
                    frame_png,
                    width,
                    height,
                    points2d,
                    title,
                    args.conf_threshold,
                    args.y_flip,
                )
                rendered_2d += 1

            for variant_name, kpt3d_npy in kpt3d_variant_paths.items():
                shape3d, data3d = parse_npy_float32(kpt3d_npy)
                if shape3d is None or data3d is None:
                    continue

                points3d = reshape_3d_points(shape3d, data3d, frame_id)
                if not points3d:
                    continue

                title = f"{character}/{action}/{camera_id}/{variant_name} 3D frame={frame_id:06d}"
                out_3d = out_root / variant_name / "3d"
                out_3d.mkdir(parents=True, exist_ok=True)
                save_3d_three_views_svg(out_3d / f"kpt3d_{frame_id:06d}_3views.svg", points3d, title)
                save_3d_perspective_svg(out_3d / f"kpt3d_{frame_id:06d}_perspective.svg", points3d, title)
                rendered_3d += 1

        total_targets += 1
        total_2d += rendered_2d
        total_3d_frames += rendered_3d

        print(
            f"[visualize_kpts.py] {character}/{action}/{camera_id} -> "
            f"2D={rendered_2d}, 3D files={rendered_3d * 2}"
        )

    print(f"[visualize_kpts.py] dataset_root = {args.dataset_root.resolve()}")
    print(f"[visualize_kpts.py] targets rendered = {total_targets}")
    print(f"[visualize_kpts.py] 2D overlays generated = {total_2d}")
    print(f"[visualize_kpts.py] 3D visuals generated = {total_3d_frames * 2} files")
    print(f"[visualize_kpts.py] output root = {args.out_dir.resolve()}")

    return 0 if total_2d > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
