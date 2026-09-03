"""Deterministic camera grouping for frozen pairwise N-view evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


_CAMERA_PATTERN = re.compile(r"^capture_L(?P<layer>\d+)_A(?P<azimuth>\d{3})$")


@dataclass(frozen=True)
class CameraGroup:
    group_id: str
    person_id: str
    action_id: str
    layer: int
    cameras: tuple[str, str, str, str]


def _parse_camera(camera_id: str) -> tuple[int, int]:
    match = _CAMERA_PATTERN.fullmatch(str(camera_id))
    if match is None:
        raise ValueError(f"Unsupported Unity camera id: {camera_id}")
    layer = int(match.group("layer"))
    azimuth = int(match.group("azimuth"))
    if azimuth < 0 or azimuth >= 360 or azimuth % 10:
        raise ValueError(f"Unity camera azimuth must use the 10-degree grid: {camera_id}")
    return layer, azimuth


def build_nested_camera_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[CameraGroup]:
    """Build nine cyclically unique four-camera rings per action and layer."""

    cameras_by_sequence: dict[tuple[str, str, int], dict[int, str]] = {}
    for row in rows:
        person_id = str(row["person_id"])
        action_id = str(row["action_id"])
        for field in ("cam1_id", "cam2_id"):
            camera_id = str(row[field])
            layer, azimuth = _parse_camera(camera_id)
            key = (person_id, action_id, layer)
            ring = cameras_by_sequence.setdefault(key, {})
            previous = ring.get(azimuth)
            if previous is not None and previous != camera_id:
                raise ValueError(
                    f"Conflicting camera ids at {key} azimuth {azimuth}: "
                    f"{previous} vs {camera_id}"
                )
            ring[azimuth] = camera_id

    expected = set(range(0, 360, 10))
    groups: list[CameraGroup] = []
    for (person_id, action_id, layer), ring in sorted(cameras_by_sequence.items()):
        missing = sorted(expected.difference(ring))
        if missing:
            raise ValueError(
                "Cannot form an incomplete 90-degree camera ring for "
                f"{person_id}/{action_id}/L{layer}; missing azimuths {missing}"
            )
        for start in range(0, 90, 10):
            azimuths = tuple((start + offset) % 360 for offset in (0, 90, 180, 270))
            cameras = tuple(ring[azimuth] for azimuth in azimuths)
            groups.append(
                CameraGroup(
                    group_id=f"{person_id}__{action_id}__L{layer}__A{start:03d}",
                    person_id=person_id,
                    action_id=action_id,
                    layer=layer,
                    cameras=cameras,  # type: ignore[arg-type]
                )
            )
    return groups


def nested_cameras(group: CameraGroup, n_views: int) -> tuple[str, ...]:
    indices = {1: (0,), 2: (0, 2), 3: (0, 1, 2), 4: (0, 1, 2, 3)}
    if int(n_views) not in indices:
        raise ValueError("n_views must be one of 1, 2, 3, 4")
    return tuple(group.cameras[index] for index in indices[int(n_views)])


from dual2pose.eval.nview_loading import (
    InsufficientCommonFrames,
    MultiViewSample,
    load_multiview_sample,
)
