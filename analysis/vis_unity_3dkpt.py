#!/usr/bin/env python3
"""Visualize 3D keypoints from SkiDataset kpt3d .npy files.

Works without numpy/matplotlib — stdlib only.
Outputs SVG files: three orthographic views (X-Y, X-Z, Y-Z) and one
perspective view for each requested frame / variant / action.

Usage
-----
# All actions, all variants, frames 0,16,32
python analysis/visualize_kpt3d.py

# Specific action & variant
python analysis/visualize_kpt3d.py --action Anim_Male_Skier_Skiing --variant character --frames 0,8,16

# Specific character
python analysis/visualize_kpt3d.py --character male --frames 0,10,20,30
"""
from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANTS = ("character", "pole", "ski")
RE_FRAME_NPY = re.compile(r"^frame_(\d+)\.npy$")

UNITY_MAPPING = {
    51: "Bone_Eye_L",
    52: "Bone_Eye_R",
    4: "Upperarm_L",
    27: "Upperarm_R",
    5: "lowerarm_l",
    28: "lowerarm_r",
    78: "Thigh_L",
    87: "Thigh_R",
    79: "calf_l",
    88: "calf_r",
    81: "Foot_L",
    90: "Foot_R",
    29: "Hand_R",
    6: "Hand_L",
    49: "neck_01",
}

# Simple approximate human skeleton edges (joint indices from Unity auto-scan
# are unknown, so we draw all consecutive-index connections as a fallback).
# If you know the exact bone order you can replace this with a proper skeleton.


# ---------------------------------------------------------------------------
# NPY helpers (stdlib only)
# ---------------------------------------------------------------------------

def _parse_npy_header(raw: bytes):
    """Return (dtype_str, shape, data_offset) or (None, None, None)."""
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        return None, None, None
    major = raw[6]
    if major == 1:
        hlen = int.from_bytes(raw[8:10], "little")
        hdr_start = 10
    elif major in (2, 3):
        hlen = int.from_bytes(raw[8:12], "little")
        hdr_start = 12
    else:
        return None, None, None

    header = raw[hdr_start : hdr_start + hlen].decode("latin1")
    m_dtype = re.search(r"'descr':\s*'([^']+)'", header)
    m_shape = re.search(r"'shape':\s*\((.*?)\)", header)
    if not m_dtype or not m_shape:
        return None, None, None

    dtype = m_dtype.group(1)
    shape_raw = m_shape.group(1).strip()
    shape = [int(x.strip()) for x in shape_raw.split(",") if x.strip()] if shape_raw else []
    return dtype, shape, hdr_start + hlen


def load_kpt3d_npy(path: Path) -> tuple[list[int] | None, list[float] | None]:
    """Load a float32 npy file.  Returns (shape, flat_data) or (None, None)."""
    try:
        raw = path.read_bytes()
        dtype, shape, offset = _parse_npy_header(raw)
        if dtype not in ("<f4", "|f4") or shape is None or offset is None:
            return None, None
        count = 1
        for d in shape:
            count *= d
        needed = offset + 4 * count
        if needed > len(raw):
            return None, None
        data = list(struct.unpack("<" + "f" * count, raw[offset:needed]))
        return shape, data
    except Exception:
        return None, None


def extract_points(shape: list[int], data: list[float]) -> list[tuple[float, float, float]]:
    """Extract list of (x,y,z) from flat data with shape (J,3) or (T,J,3).

    For (T,J,3) files (merged), always uses frame 0.
    """
    if len(shape) == 2 and shape[1] == 3:
        J = shape[0]
        return [(data[j * 3], data[j * 3 + 1], data[j * 3 + 2]) for j in range(J)]
    if len(shape) == 3 and shape[2] == 3:
        J = shape[1]
        return [(data[j * 3], data[j * 3 + 1], data[j * 3 + 2]) for j in range(J)]
    return []


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _project_panel(
    a: float, b: float,
    a_min: float, a_max: float,
    b_min: float, b_max: float,
    px: float, py: float, pw: float, ph: float,
) -> tuple[float, float]:
    ar = a_max - a_min or 1e-9
    br = b_max - b_min or 1e-9
    na = (a - a_min) / ar
    nb = (b - b_min) / br
    return px + na * pw, py + (1.0 - nb) * ph


def infer_ski_pole_edges(points_len: int, joint_names: dict[int, str]) -> list[tuple[int, int, str]]:
    """Infer pole/ski edges. Returns [(idx_a, idx_b, kind), ...]."""
    edges: list[tuple[int, int, str]] = []

    if points_len <= 1:
        return edges

    if joint_names:
        name_to_idx = {name.lower(): idx for idx, name in joint_names.items()}

        def add_named(a: str, b: str, kind: str) -> None:
            ia = name_to_idx.get(a.lower())
            ib = name_to_idx.get(b.lower())
            if ia is None or ib is None:
                return
            if 0 <= ia < points_len and 0 <= ib < points_len and ia != ib:
                edges.append((ia, ib, kind))

        # Pole edges
        add_named("Pole_L_Handle", "Pole_L_Tip", "pole")
        add_named("Pole_R_Handle", "Pole_R_Tip", "pole")

        # Ski edges (triangle per ski)
        add_named("Ski_L_center", "Ski_L_front", "ski")
        add_named("Ski_L_center", "Ski_L_back", "ski")
        add_named("Ski_L_front", "Ski_L_back", "ski")

        add_named("Ski_R_center", "Ski_R_front", "ski")
        add_named("Ski_R_center", "Ski_R_back", "ski")
        add_named("Ski_R_front", "Ski_R_back", "ski")

        if edges:
            return edges

    # Fallback when joint_names.txt is missing:
    # pole variant usually has 4 points -> two handles/tips
    if points_len == 4:
        edges.extend([(0, 1, "pole"), (2, 3, "pole")])
        return edges

    # ski variant usually has 6 points -> two ski triangles
    if points_len == 6:
        edges.extend([
            (0, 1, "ski"), (0, 2, "ski"), (1, 2, "ski"),
            (3, 4, "ski"), (3, 5, "ski"), (4, 5, "ski"),
        ])
        return edges

    return edges


def save_three_views_svg(
    out_path: Path,
    points: list[tuple[float, float, float]],
    title: str,
    joint_names: dict[int, str] | None = None,
) -> None:
    """Save an SVG with three orthographic panels: X-Y, X-Z, Y-Z."""
    if joint_names is None:
        joint_names = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        out_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="360">'
            '<text x="20" y="40" font-size="16">No points</text></svg>',
            encoding="utf-8",
        )
        return

    W, H = 2940, 1200
    margin, gap = 36, 24
    panel_w = (W - margin * 2 - gap * 2) / 3
    panel_h = H - 220
    panel_y = 100

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    def _pad(mn, mx, frac=0.08):
        r = (mx - mn) or 0.1
        return mn - r * frac, mx + r * frac

    xn, xx = _pad(min(xs), max(xs))
    yn, yx = _pad(min(ys), max(ys))
    zn, zx = _pad(min(zs), max(zs))

    panels = [
        ("Front  (X – Y)", xs, ys, xn, xx, yn, yx, "X →", "Y ↑"),
        ("Side   (X – Z)", xs, zs, xn, xx, zn, zx, "X →", "Z ↑"),
        ("Top    (Y – Z)", ys, zs, yn, yx, zn, zx, "Y →", "Z ↑"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="#1e1e2e"/>',
        f'<text x="{W/2}" y="60" text-anchor="middle" font-size="40" '
        f'font-family="Arial" fill="#cdd6f4">{_escape(title)}</text>',
    ]

    for i, (label, a_vals, b_vals, a_min, a_max, b_min, b_max, a_lbl, b_lbl) in enumerate(panels):
        px = margin + i * (panel_w + gap)
        # panel background
        lines.append(
            f'<rect x="{px}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
            f'fill="#181825" stroke="#45475a" stroke-width="1"/>'
        )
        # panel title
        lines.append(
            f'<text x="{px + panel_w/2}" y="{panel_y - 20}" text-anchor="middle" '
            f'font-size="28" font-family="Arial" fill="#a6adc8">{label}</text>'
        )
        # axis labels
        lines.append(
            f'<text x="{px + panel_w - 8}" y="{panel_y + panel_h + 32}" '
            f'text-anchor="end" font-size="20" font-family="Arial" fill="#6c7086">{a_lbl}</text>'
        )
        lines.append(
            f'<text x="{px + 8}" y="{panel_y - 4}" '
            f'font-size="20" font-family="Arial" fill="#6c7086">{b_lbl}</text>'
        )

        # draw consecutive-index skeleton lines (light)
        for j in range(len(points) - 1):
            ax1, ay1 = _project_panel(a_vals[j], b_vals[j], a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            ax2, ay2 = _project_panel(a_vals[j+1], b_vals[j+1], a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            lines.append(
                f'<line x1="{ax1:.2f}" y1="{ay1:.2f}" x2="{ax2:.2f}" y2="{ay2:.2f}" '
                f'stroke="#313244" stroke-width="2.0"/>'
            )

        # Highlight dedicated ski/pole edges.
        special_edges = infer_ski_pole_edges(len(points), joint_names)
        for ia, ib, kind in special_edges:
            ax1, ay1 = _project_panel(a_vals[ia], b_vals[ia], a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            ax2, ay2 = _project_panel(a_vals[ib], b_vals[ib], a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            edge_color = "#f38ba8" if kind == "pole" else "#a6e3a1"
            lines.append(
                f'<line x1="{ax1:.2f}" y1="{ay1:.2f}" x2="{ax2:.2f}" y2="{ay2:.2f}" '
                f'stroke="{edge_color}" stroke-width="4.0" stroke-linecap="round"/>'
            )

        # draw joints
        for j, (av, bv) in enumerate(zip(a_vals, b_vals)):
            cx, cy = _project_panel(av, bv, a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="6.5" fill="#89b4fa" fill-opacity="0.93"/>')
            label_text = f"{j}"
            if j in joint_names:
                label_text = f"{j}:{joint_names[j]}"
            lines.append(
                f'<text x="{cx+8:.2f}" y="{cy-8:.2f}" font-size="16" '
                f'font-family="Arial" fill="#cba6f7">{_escape(label_text)}</text>'
            )

        # stats
        lines.append(
            f'<text x="{px+8}" y="{panel_y + panel_h + 60}" font-size="18" '
            f'font-family="Arial" fill="#585b70">'
            f'{a_lbl.split()[0]}: [{a_min:.3f}, {a_max:.3f}]  '
            f'{b_lbl.split()[0]}: [{b_min:.3f}, {b_max:.3f}]'
            f'</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_selected_mapping_three_views_svg(
    out_path: Path,
    points: list[tuple[float, float, float]],
    title: str,
    joint_names: dict[int, str] | None = None,
) -> None:
    """Save one extra SVG that only draws joints from UNITY_MAPPING."""
    if joint_names is None:
        joint_names = {}

    selected: list[tuple[int, str, float, float, float]] = []
    for idx, name in UNITY_MAPPING.items():
        if 0 <= idx < len(points):
            x, y, z = points[idx]
            selected.append((idx, name, x, y, z))

    name_to_idx = {name.lower(): idx for idx, name in joint_names.items()}

    def find_idx(name: str, fallback: int | None = None) -> int | None:
        idx = name_to_idx.get(name.lower())
        if idx is not None and 0 <= idx < len(points):
            return idx
        if fallback is not None and 0 <= fallback < len(points):
            return fallback
        return None

    equipment_points: list[tuple[int, str, str, float, float, float]] = []
    equipment_index_set: set[int] = set()

    def add_equipment(name: str, kind: str, fallback: int | None = None) -> None:
        idx = find_idx(name, fallback)
        if idx is None or idx in equipment_index_set:
            return
        x, y, z = points[idx]
        equipment_points.append((idx, name, kind, x, y, z))
        equipment_index_set.add(idx)

    # Poles
    add_equipment("Pole_L_Handle", "pole", 7)
    add_equipment("Pole_L_Tip", "pole", 8)
    add_equipment("Pole_R_Handle", "pole", 30)
    add_equipment("Pole_R_Tip", "pole", 31)

    # Skis
    add_equipment("Ski_L_center", "ski", 83)
    add_equipment("Ski_L_front", "ski", 84)
    add_equipment("Ski_L_back", "ski", 85)
    add_equipment("Ski_R_center", "ski", 92)
    add_equipment("Ski_R_front", "ski", 93)
    add_equipment("Ski_R_back", "ski", 94)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not selected:
        out_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="520">'
            '<text x="20" y="40" font-size="20">No selected mapping joints found.</text></svg>',
            encoding="utf-8",
        )
        return

    W, H = 2940, 1200
    margin, gap = 36, 24
    panel_w = (W - margin * 2 - gap * 2) / 3
    panel_h = H - 220
    panel_y = 100

    all_points_xyz = [(p[2], p[3], p[4]) for p in selected] + [(p[3], p[4], p[5]) for p in equipment_points]
    xs = [p[0] for p in all_points_xyz]
    ys = [p[1] for p in all_points_xyz]
    zs = [p[2] for p in all_points_xyz]

    def _pad(mn, mx, frac=0.12):
        r = (mx - mn) or 0.1
        return mn - r * frac, mx + r * frac

    xn, xx = _pad(min(xs), max(xs))
    yn, yx = _pad(min(ys), max(ys))
    zn, zx = _pad(min(zs), max(zs))

    panels = [
        ("Selected Front  (X - Y)", "x", "y", xn, xx, yn, yx, "X ->", "Y ^"),
        ("Selected Side   (X - Z)", "x", "z", xn, xx, zn, zx, "X ->", "Z ^"),
        ("Selected Top    (Y - Z)", "y", "z", yn, yx, zn, zx, "Y ->", "Z ^"),
    ]

    selected_map = {idx: (name, x, y, z) for idx, name, x, y, z in selected}
    equipment_map = {idx: (name, kind, x, y, z) for idx, name, kind, x, y, z in equipment_points}

    def comp(x: float, y: float, z: float, axis: str) -> float:
        if axis == "x":
            return x
        if axis == "y":
            return y
        return z

    body_edges = [
        (49, 4), (4, 5), (5, 6),
        (49, 27), (27, 28), (28, 29),
        (49, 78), (78, 79), (79, 81),
        (49, 87), (87, 88), (88, 90),
        (49, 51), (49, 52), (51, 52),
    ]

    equipment_edges_named = [
        ("Pole_L_Handle", "Pole_L_Tip", "pole"),
        ("Pole_R_Handle", "Pole_R_Tip", "pole"),
        ("Ski_L_center", "Ski_L_front", "ski"),
        ("Ski_L_center", "Ski_L_back", "ski"),
        ("Ski_L_front", "Ski_L_back", "ski"),
        ("Ski_R_center", "Ski_R_front", "ski"),
        ("Ski_R_center", "Ski_R_back", "ski"),
        ("Ski_R_front", "Ski_R_back", "ski"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="#10131a"/>',
        f'<text x="{W/2}" y="60" text-anchor="middle" font-size="40" '
        f'font-family="Arial" fill="#d9e2ff">{_escape(title)}</text>',
    ]

    for i, (label, axa, axb, a_min, a_max, b_min, b_max, a_lbl, b_lbl) in enumerate(panels):
        px = margin + i * (panel_w + gap)
        lines.append(
            f'<rect x="{px}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
            f'fill="#161b22" stroke="#3a4453" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{px + panel_w/2}" y="{panel_y - 20}" text-anchor="middle" '
            f'font-size="28" font-family="Arial" fill="#a5b4d4">{label}</text>'
        )
        lines.append(
            f'<text x="{px + panel_w - 8}" y="{panel_y + panel_h + 32}" '
            f'text-anchor="end" font-size="20" font-family="Arial" fill="#72809c">{a_lbl}</text>'
        )
        lines.append(
            f'<text x="{px + 8}" y="{panel_y - 4}" '
            f'font-size="20" font-family="Arial" fill="#72809c">{b_lbl}</text>'
        )

        for ia, ib in body_edges:
            if ia not in selected_map or ib not in selected_map:
                continue
            _na, xa, ya, za = selected_map[ia]
            _nb, xb, yb, zb = selected_map[ib]
            x1, y1 = _project_panel(
                comp(xa, ya, za, axa),
                comp(xa, ya, za, axb),
                a_min,
                a_max,
                b_min,
                b_max,
                px,
                panel_y,
                panel_w,
                panel_h,
            )
            x2, y2 = _project_panel(
                comp(xb, yb, zb, axa),
                comp(xb, yb, zb, axb),
                a_min,
                a_max,
                b_min,
                b_max,
                px,
                panel_y,
                panel_w,
                panel_h,
            )
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#69db7c" stroke-width="4.0" stroke-linecap="round"/>'
            )

        for name_a, name_b, kind in equipment_edges_named:
            ia = find_idx(name_a)
            ib = find_idx(name_b)
            if ia is None or ib is None:
                continue
            xa, ya, za = points[ia]
            xb, yb, zb = points[ib]
            x1, y1 = _project_panel(
                comp(xa, ya, za, axa),
                comp(xa, ya, za, axb),
                a_min,
                a_max,
                b_min,
                b_max,
                px,
                panel_y,
                panel_w,
                panel_h,
            )
            x2, y2 = _project_panel(
                comp(xb, yb, zb, axa),
                comp(xb, yb, zb, axb),
                a_min,
                a_max,
                b_min,
                b_max,
                px,
                panel_y,
                panel_w,
                panel_h,
            )
            edge_color = "#ff6b6b" if kind == "pole" else "#ffd43b"
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{edge_color}" stroke-width="4.5" stroke-linecap="round"/>'
            )

        for idx_sel, name_sel, x, y, z in selected:
            av = comp(x, y, z, axa)
            bv = comp(x, y, z, axb)
            cx, cy = _project_panel(av, bv, a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="9.0" fill="#4cc9f0" fill-opacity="0.95"/>')
            lines.append(
                f'<text x="{cx+10:.2f}" y="{cy-10:.2f}" font-size="18" '
                f'font-family="Arial" fill="#ffd166">{_escape(f"{idx_sel}:{name_sel}")}</text>'
            )

        for idx_eq, name_eq, kind_eq, x, y, z in equipment_points:
            av = comp(x, y, z, axa)
            bv = comp(x, y, z, axb)
            cx, cy = _project_panel(av, bv, a_min, a_max, b_min, b_max, px, panel_y, panel_w, panel_h)
            fill = "#ff8787" if kind_eq == "pole" else "#ffe066"
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="8.0" fill="{fill}" fill-opacity="0.95"/>')
            lines.append(
                f'<text x="{cx+10:.2f}" y="{cy+18:.2f}" font-size="16" '
                f'font-family="Arial" fill="{fill}">{_escape(f"{idx_eq}:{name_eq}")}</text>'
            )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_perspective_svg(
    out_path: Path,
    points: list[tuple[float, float, float]],
    title: str,
    joint_names: dict[int, str] | None = None,
) -> None:
    """Save a simple perspective-projected SVG."""
    if joint_names is None:
        joint_names = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        out_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="480">'
            '<text x="20" y="40" font-size="16">No points</text></svg>',
            encoding="utf-8",
        )
        return

    W, H = 960, 960
    cx, cy = W / 2, H / 2 + 40

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mx, my, mz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)

    rx = math.radians(20.0)   # tilt down slightly
    ry = math.radians(-40.0)  # rotate left
    cosx, sinx = math.cos(rx), math.sin(rx)
    cosy, siny = math.cos(ry), math.sin(ry)
    DIST = 3.5

    def _to_view(x, y, z):
        x -= mx; y -= my; z -= mz
        xr = x * cosy + z * siny
        zr = -x * siny + z * cosy
        yr = y * cosx - zr * sinx
        zr2 = y * sinx + zr * cosx
        denom = max(0.2, DIST + zr2 * 0.3)
        return zr2, xr / denom, yr / denom

    proj: list[tuple[float, float, float, int]] = []
    for idx, (x, y, z) in enumerate(points):
        depth, ux, uy = _to_view(x, y, z)
        proj.append((depth, ux, uy, idx))

    us = [p[1] for p in proj]
    vs = [p[2] for p in proj]
    u_rng = max(max(us) - min(us), 1e-6)
    v_rng = max(max(vs) - min(vs), 1e-6)
    scale = min(W * 0.78 / u_rng, H * 0.72 / v_rng)
    u_mid = 0.5 * (max(us) + min(us))
    v_mid = 0.5 * (max(vs) + min(vs))

    screen: list[tuple[float, float, float, int]] = [
        (d, cx + (u - u_mid) * scale, cy - (v - v_mid) * scale, idx)
        for d, u, v, idx in proj
    ]
    screen.sort(key=lambda x: x[0])  # paint back-to-front

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="#1e1e2e"/>',
        f'<text x="{W/2}" y="48" text-anchor="middle" font-size="32" '
        f'font-family="Arial" fill="#cdd6f4">{_escape(title)}</text>',
        f'<text x="{W/2}" y="88" text-anchor="middle" font-size="20" '
        f'font-family="Arial" fill="#6c7086">perspective (rx=20° ry=-40°)</text>',
    ]

    # skeleton lines (consecutive joints, back-to-front order not applied to lines for speed)
    for j in range(len(points) - 1):
        _, px1, py1, _ = screen[j]   # Note: screen is depth-sorted, use original order
        pass  # skip — draw by index order instead

    # draw lines by original joint order
    idx_to_screen = {idx: (px, py) for (_, px, py, idx) in screen}
    for j in range(len(points) - 1):
        if j in idx_to_screen and j + 1 in idx_to_screen:
            x1, y1 = idx_to_screen[j]
            x2, y2 = idx_to_screen[j + 1]
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#313244" stroke-width="2.0"/>'
            )

    # Highlight dedicated ski/pole edges.
    special_edges = infer_ski_pole_edges(len(points), joint_names)
    for ia, ib, kind in special_edges:
        if ia in idx_to_screen and ib in idx_to_screen:
            x1, y1 = idx_to_screen[ia]
            x2, y2 = idx_to_screen[ib]
            edge_color = "#f38ba8" if kind == "pole" else "#a6e3a1"
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{edge_color}" stroke-width="4.0" stroke-linecap="round"/>'
            )

    # draw joints (depth-sorted, colored by depth)
    d_min = min(p[0] for p in screen)
    d_max = max(p[0] for p in screen)
    d_rng = max(d_max - d_min, 1e-6)
    for depth, px, py, idx in screen:
        t = (depth - d_min) / d_rng        # 0=back, 1=front
        r_ch = int(80 + t * 90)
        g_ch = int(140 + t * 60)
        b_ch = int(220 + t * 35)
        color = f"rgb({r_ch},{g_ch},{min(255, b_ch)})"
        rad = 5.0 + t * 3.0
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{rad:.2f}" fill="{color}" fill-opacity="0.9"/>')
        label_text = f"{idx}"
        if idx in joint_names:
            label_text = f"{idx}:{joint_names[idx]}"
        lines.append(
            f'<text x="{px+8:.2f}" y="{py-8:.2f}" font-size="16" '
            f'font-family="Arial" fill="#cba6f7">{_escape(label_text)}</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_actions(char_root: Path) -> list[str]:
    return sorted(
        p.name for p in char_root.iterdir()
        if p.is_dir() and (p / "kpt3d").exists()
    )


def list_frame_ids(variant_dir: Path) -> list[int]:
    ids: list[int] = []
    if not variant_dir.exists():
        return ids
    for p in variant_dir.iterdir():
        m = RE_FRAME_NPY.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_joint_names(variant_dir: Path) -> dict[int, str]:
    """Load joint_names.txt if it exists. Returns {idx: name} dict."""
    names_file = variant_dir / "joint_names.txt"
    result: dict[int, str] = {}
    if not names_file.exists():
        return result
    try:
        with open(names_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        result[idx] = name
                    except ValueError:
                        pass
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize 3D keypoints from SkiDataset (stdlib-only, outputs SVG).")
    p.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/skiing_unity_dataset/data_pole_ski"),
                   help="Dataset root, e.g. unity/SkiDataset (default: auto-detect)")
    p.add_argument("--character", default="all", choices=["all", "male", "female"],
                   help="Character split (default: all)")
    p.add_argument("--action", default="",
                   help="Action name, e.g. Anim_Male_Skier_Skiing. Empty = all actions.")
    p.add_argument("--variant", default="all",
                   help="Keypoint variant: character, pole, ski, or all (default: all)")
    p.add_argument("--frames", default="0,8,16,24,32",
                   help="Comma-separated frame indices (default: 0,8,16,24,32)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "reports"  / "kpt3d_viz",
                   help="Output directory for SVG files")
    p.add_argument("--no-perspective", action="store_true",
                   help="Skip perspective SVG (only save three-view)")
    return p.parse_args()


def _parse_frames(s: str) -> list[int]:
    ids: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            try:
                ids.append(int(tok))
            except ValueError:
                pass
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    dataset_root: Path = args.dataset_root
    char_list = ["male", "female"] if args.character == "all" else [args.character]
    wanted_variants = VARIANTS if args.variant == "all" else (args.variant,)
    frame_ids = _parse_frames(args.frames)

    if not frame_ids:
        print("[ERROR] --frames is empty or invalid.")
        return 2

    total_saved = 0

    for character in char_list:
        char_root = dataset_root / character
        if not char_root.exists():
            # Also try dataset_root itself as character root (e.g. SkiDataset/male/)
            char_root = dataset_root
            if not char_root.exists():
                print(f"[WARN] character root not found: {char_root}")
                continue

        actions = [args.action] if args.action.strip() else discover_actions(char_root)
        if not actions:
            print(f"[WARN] no actions found under {char_root}")
            continue

        for action in actions:
            action_dir = char_root / action
            kpt3d_dir = action_dir / "kpt3d"
            if not kpt3d_dir.exists():
                print(f"[WARN] skip {character}/{action}: no kpt3d directory")
                continue

            for variant in wanted_variants:
                variant_dir = kpt3d_dir / variant
                if not variant_dir.exists():
                    continue

                available = set(list_frame_ids(variant_dir))
                targets = [f for f in frame_ids if f in available]
                if not targets:
                    # Fallback: take first min(5, N) available frames
                    all_ids = list_frame_ids(variant_dir)
                    targets = all_ids[: min(5, len(all_ids))]
                    if not targets:
                        print(f"[WARN] {character}/{action}/{variant}: no frames available")
                        continue
                    print(f"[INFO] {character}/{action}/{variant}: requested frames not found, "
                          f"falling back to frames {targets}")

                out_root = args.out_dir / character / action / variant
                joints_printed = False
                joint_names_map = load_joint_names(variant_dir)

                for frame_id in targets:
                    npy_path = variant_dir / f"frame_{frame_id:06d}.npy"
                    if not npy_path.exists():
                        continue

                    shape, data = load_kpt3d_npy(npy_path)
                    if shape is None or data is None:
                        print(f"[WARN] could not load {npy_path}")
                        continue

                    pts = extract_points(shape, data)
                    if not pts:
                        print(f"[WARN] empty points in {npy_path}")
                        continue

                    if not joints_printed:
                        print(f"\n  Joint list  {character}/{action}/{variant}  frame {frame_id:06d}  ({len(pts)} joints)")
                        print(f"  {'idx':>4}  {'name':30}  {'x':>12}  {'y':>12}  {'z':>12}")
                        print(f"  {'-'*4}  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
                        for ji, (jx, jy, jz) in enumerate(pts):
                            name = joint_names_map.get(ji, "UNKNOWN")
                            print(f"  {ji:>4}  {name:30}  {jx:>12.6f}  {jy:>12.6f}  {jz:>12.6f}")
                        joints_printed = True

                    title_base = f"{character}/{action}/{variant}  frame {frame_id:06d}  ({len(pts)} joints)"

                    # Three-view SVG
                    out_3v = out_root / f"frame_{frame_id:06d}_3views.svg"
                    save_three_views_svg(out_3v, pts, title_base, joint_names_map)
                    total_saved += 1

                    # Selected mapping joints only (extra image)
                    out_sel = out_root / f"frame_{frame_id:06d}_selected_mapping_3views.svg"
                    save_selected_mapping_three_views_svg(
                        out_sel,
                        pts,
                        f"{title_base}  [selected mapping joints only]",
                        joint_names_map,
                    )
                    total_saved += 1

                    # Perspective SVG
                    if not args.no_perspective:
                        out_persp = out_root / f"frame_{frame_id:06d}_perspective.svg"
                        save_perspective_svg(out_persp, pts, title_base, joint_names_map)
                        total_saved += 1

                print(f"[OK] {character}/{action}/{variant}  "
                      f"frames={targets}  joints={len(pts)}  "
                      f"-> {out_root.relative_to(args.out_dir)}")

    print(f"\n[Done] {total_saved} SVG files saved to: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
