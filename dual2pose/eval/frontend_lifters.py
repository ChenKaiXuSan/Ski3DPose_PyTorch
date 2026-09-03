"""Shared inputs and official-model adapters for 2D-to-3D front ends.

The adapters intentionally load code from user-supplied clones of the official
repositories. Third-party source and weights are not copied into this project.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

H36M17_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
)

_H36M_ALIASES = (
    ("pelvis", "hip"),
    ("thigh_r", "right_hip", "rightupleg"),
    ("calf_r", "right_knee", "rightleg"),
    ("foot_r", "right_foot"),
    ("thigh_l", "left_hip", "leftupleg"),
    ("calf_l", "left_knee", "leftleg"),
    ("foot_l", "left_foot"),
    ("spine_02", "spine2", "spine", "spine1"),
    ("spine_03", "spine3", "thorax"),
    ("neck_01", "neck", "neck_base", "neckbase"),
    ("head", "head_top", "headtop", "dummy_helmet"),
    ("upperarm_l", "left_shoulder", "leftarm"),
    ("lowerarm_l", "left_elbow", "leftforearm"),
    ("hand_l", "left_hand"),
    ("upperarm_r", "right_shoulder", "rightarm"),
    ("lowerarm_r", "right_elbow", "rightforearm"),
    ("hand_r", "right_hand"),
)

_LEFT_JOINTS = (4, 5, 6, 11, 12, 13)
_RIGHT_JOINTS = (1, 2, 3, 14, 15, 16)

# Official Human3.6M-17 semantics are
# [root, legs, spine, thorax, neck, head, arms]. CanonFuse has no thorax
# point, so its final common joint uses H36M neck (index 9), not thorax (8).
H36M17_TO_CANONFUSE_COMMON13_INDICES = (
    11, 14, 12, 15, 4, 1, 5, 2, 6, 3, 16, 13, 9
)
MOTIONBERT_H36M_BBOX_CENTER = (528.0, 427.0)
MOTIONBERT_H36M_BBOX_SCALE = 400.0
MOTIONBERT_INFERENCE_FACTOR = 4.0


def _normalized_joint_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def unity_to_h36m17(
    keypoints: np.ndarray, joint_names: Sequence[str]
) -> np.ndarray:
    """Map Unity character joints into the standard H36M-17 order by name."""

    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != len(joint_names) or array.shape[2] < 2:
        raise ValueError(
            "Unity 2D keypoints must have shape T x len(joint_names) x C, "
            f"got {array.shape} and {len(joint_names)} names"
        )
    lookup: dict[str, int] = {}
    for index, name in enumerate(joint_names):
        lookup.setdefault(_normalized_joint_name(str(name)), index)

    selected: list[int] = []
    missing: list[str] = []
    for target_name, aliases in zip(H36M17_NAMES, _H36M_ALIASES):
        source_index = next(
            (
                lookup[_normalized_joint_name(alias)]
                for alias in aliases
                if _normalized_joint_name(alias) in lookup
            ),
            None,
        )
        if source_index is None:
            missing.append(target_name)
        else:
            selected.append(source_index)
    if missing:
        raise KeyError(f"Unity skeleton is missing H36M joints: {', '.join(missing)}")

    output = array[:, selected, : min(array.shape[2], 3)].copy()
    if output.shape[2] == 2:
        output = np.concatenate(
            [output, np.ones((*output.shape[:2], 1), dtype=np.float32)], axis=-1
        )
    return np.ascontiguousarray(output, dtype=np.float32)


def normalize_screen_coordinates(
    points: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Apply the Human3.6M normalization used by VideoPose3D/PoseFormer."""

    array = np.asarray(points, dtype=np.float32)
    if array.shape[-1] != 2:
        raise ValueError(f"Expected final coordinate dimension 2, got {array.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    return array / float(width) * 2.0 - np.asarray(
        [1.0, float(height) / float(width)], dtype=np.float32
    )


def normalize_motionbert_pose_2d(
    keypoints: np.ndarray,
    image_size: tuple[int, int],
    bbox_center: tuple[float, float] = MOTIONBERT_H36M_BBOX_CENTER,
    bbox_scale: float = MOTIONBERT_H36M_BBOX_SCALE,
) -> np.ndarray:
    """Apply MMPose default H36M bbox normalization for MotionBERT."""

    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(f"Expected normalized T x 17 x 3 input, got {array.shape}")
    width, height = image_size
    if width <= 0 or height <= 0 or bbox_scale <= 0:
        raise ValueError("Image dimensions and MotionBERT bbox scale must be positive")

    output = array.copy()
    pixels = output[..., :2].copy()
    pixels[..., 0] = (pixels[..., 0] + 1.0) * float(width) / 2.0
    pixels[..., 1] = (
        pixels[..., 1] + float(height) / float(width)
    ) * float(width) / 2.0
    lower = pixels.min(axis=1, keepdims=True)
    upper = pixels.max(axis=1, keepdims=True)
    source_center = 0.5 * (lower + upper)
    source_scale = np.max(upper - lower, axis=-1, keepdims=True)
    if np.any(source_scale <= 1e-8):
        raise ValueError("MotionBERT input contains a degenerate 2D pose bbox")
    pixels = (
        (pixels - source_center) / source_scale * float(bbox_scale)
        + np.asarray(bbox_center, dtype=np.float32).reshape(1, 1, 2)
    )
    output[..., :2] = normalize_screen_coordinates(pixels, width, height)
    return np.ascontiguousarray(output, dtype=np.float32)


def motionbert_output_scale_m(
    image_size: tuple[int, int],
    factor: float = MOTIONBERT_INFERENCE_FACTOR,
) -> float:
    """Return the MMPose MotionBERT decoder scale from normalized units to m."""

    width, height = image_size
    if width <= 0 or height <= 0 or factor <= 0:
        raise ValueError("Image dimensions and MotionBERT factor must be positive")
    return float(width) / 2.0 * float(factor) / 1000.0


def _extract_frame_index(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"No frame index in keypoint filename: {path}")
    long_matches = [value for value in matches if len(value) >= 6]
    return int(long_matches[0] if long_matches else matches[-1])


def load_unity_2d_stream(
    keypoint_dir: Path,
    joint_names_path: Path,
    sequence_meta_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load and normalize one complete Unity GT-2D camera stream."""

    keypoint_dir = Path(keypoint_dir)
    files = sorted(keypoint_dir.glob("kpt2d_*.npy"), key=_extract_frame_index)
    if not files:
        raise FileNotFoundError(f"No kpt2d_*.npy files in {keypoint_dir}")
    names_payload = json.loads(
        Path(joint_names_path).read_text(encoding="utf-8-sig")
    )
    joint_names = names_payload.get("joint_names")
    if not isinstance(joint_names, list) or not joint_names:
        raise ValueError(f"joint_names missing from {joint_names_path}")
    sequence_meta = json.loads(
        Path(sequence_meta_path).read_text(encoding="utf-8-sig")
    )
    width = int(sequence_meta["width"])
    height = int(sequence_meta["height"])
    raw = np.stack(
        [np.asarray(np.load(path, allow_pickle=False), dtype=np.float32) for path in files]
    )
    pose = unity_to_h36m17(raw, joint_names)
    pose[..., :2] = normalize_screen_coordinates(pose[..., :2], width, height)
    frame_indices = np.asarray([_extract_frame_index(path) for path in files], dtype=np.int64)
    if len(set(frame_indices.tolist())) != len(frame_indices):
        raise ValueError(f"Duplicate frame indices in {keypoint_dir}")
    return pose, frame_indices, (width, height)


def h36m17_to_canonfuse15(pose: np.ndarray) -> np.ndarray:
    """Map H36M-17 to CanonFuse-15; synthesize only its unavailable eye pair.

    The synthesized eyes provide a deterministic facing sign to CanonFuse's
    canonicalizer. Publication metrics must use joints 2:15 (the common 13).
    """

    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(f"Expected H36M pose T x 17 x 3, got {array.shape}")
    output = np.empty((array.shape[0], 15, 3), dtype=np.float32)
    output[:, 2:, :] = array[:, H36M17_TO_CANONFUSE_COMMON13_INDICES, :]

    left_hip, right_hip = array[:, 4], array[:, 1]
    neck, head = array[:, 9], array[:, 10]
    left_shoulder, right_shoulder = array[:, 11], array[:, 14]

    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(norm, 1e-8)

    body_right = _unit(right_hip - left_hip)
    body_up = _unit(neck - 0.5 * (left_hip + right_hip))
    body_forward = _unit(np.cross(body_right, body_up))
    torso = np.linalg.norm(neck - 0.5 * (left_hip + right_hip), axis=-1, keepdims=True)
    shoulder_width = np.linalg.norm(
        right_shoulder - left_shoulder, axis=-1, keepdims=True
    )
    eye_center = head + 0.05 * torso * body_forward
    eye_offset = 0.08 * shoulder_width * body_right
    output[:, 0] = eye_center - eye_offset
    output[:, 1] = eye_center + eye_offset
    if not np.isfinite(output).all():
        raise ValueError("Converted CanonFuse pose contains non-finite values")
    return output


def centered_temporal_windows(sequence: torch.Tensor, window: int) -> torch.Tensor:
    """Return one odd-sized, repeat-edge temporal window per source frame."""

    if sequence.ndim < 1 or sequence.shape[0] <= 0:
        raise ValueError("Temporal sequence must be non-empty")
    if window <= 0 or window % 2 == 0:
        raise ValueError("Temporal window must be a positive odd integer")
    radius = window // 2
    prefix = sequence[:1].repeat((radius,) + (1,) * (sequence.ndim - 1))
    suffix = sequence[-1:].repeat((radius,) + (1,) * (sequence.ndim - 1))
    padded = torch.cat([prefix, sequence, suffix], dim=0)
    return torch.stack([padded[index : index + window] for index in range(sequence.shape[0])])


def temporal_chunks(length: int, max_length: int) -> list[tuple[int, int]]:
    if length <= 0 or max_length <= 0:
        raise ValueError("Temporal length and maximum length must be positive")
    return [(start, min(start + max_length, length)) for start in range(0, length, max_length)]


def _flip_h36m(tensor: torch.Tensor) -> torch.Tensor:
    output = tensor.clone()
    output[..., 0] *= -1
    output[..., list(_LEFT_JOINTS + _RIGHT_JOINTS), :] = output[
        ..., list(_RIGHT_JOINTS + _LEFT_JOINTS), :
    ]
    return output


def _load_python_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temporary_import_root(root: Path, prefix: str) -> Iterator[None]:
    root_string = str(root.resolve())
    stale = {name: module for name, module in sys.modules.items() if name == prefix or name.startswith(prefix + ".")}
    for name in stale:
        sys.modules.pop(name, None)
    sys.path.insert(0, root_string)
    try:
        yield
    finally:
        if root_string in sys.path:
            sys.path.remove(root_string)
        for name in list(sys.modules):
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") if module else ""
            if (name == prefix or name.startswith(prefix + ".")) and module_file and str(module_file).startswith(root_string):
                sys.modules.pop(name, None)
        sys.modules.update(stale)


def _normalize_checkpoint_key(key: str) -> str:
    """Normalize original/DataParallel/MMPose MotionBERT state names."""

    if key.startswith("module."):
        key = key[7:]
    if key.startswith("head.pre_logits."):
        key = "pre_logits." + key[len("head.pre_logits.") :]
    elif key.startswith("head.fc."):
        key = "head." + key[len("head.fc.") :]
    elif key.startswith("backbone.attn_regress."):
        key = "ts_attn." + key[len("backbone.attn_regress.") :]
    elif key.startswith("backbone."):
        key = key[len("backbone.") :]
    if key == "spat_embed":
        key = "pos_embed"
    key = re.sub(r"\.(mlp_[st])\.0\.", r".\1.fc1.", key)
    key = re.sub(r"\.(mlp_[st])\.2\.", r".\1.fc2.", key)
    return key


def _checkpoint_state(
    checkpoint_path: Path,
    allow_unsafe_checkpoint: bool = False,
    allow_numpy_checkpoint_state: bool = False,
) -> Mapping[str, torch.Tensor]:
    if allow_unsafe_checkpoint and allow_numpy_checkpoint_state:
        raise ValueError("Choose either unsafe pickle loading or the NumPy safe allowlist")
    if allow_numpy_checkpoint_state:
        from numpy._core.multiarray import _reconstruct
        from numpy.random._pickle import __bit_generator_ctor, __randomstate_ctor

        trusted_globals = [
            np.ndarray,
            np.dtype,
            np.random.RandomState,
            np.random.MT19937,
            __randomstate_ctor,
            __bit_generator_ctor,
            _reconstruct,
            (_reconstruct, "numpy.core.multiarray._reconstruct"),
        ]
        trusted_globals.extend(
            {type(np.dtype(name)) for name in ("float32", "float64", "uint32")}
        )
        with torch.serialization.safe_globals(trusted_globals):
            payload = torch.load(
                Path(checkpoint_path), map_location="cpu", weights_only=True
            )
    else:
        payload = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=not allow_unsafe_checkpoint,
        )
    if isinstance(payload, Mapping):
        for key in ("model_pos", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            normalized_key = _normalize_checkpoint_key(str(key))
            if normalized_key in state:
                raise ValueError(
                    f"Checkpoint keys collide after normalization: {normalized_key}"
                )
            state[normalized_key] = value
    if not state:
        raise ValueError(f"Checkpoint contains no tensor state: {checkpoint_path}")
    return state


@dataclass
class OfficialPoseLifter:
    name: str
    model: torch.nn.Module
    device: torch.device
    temporal_length: int
    batch_size: int = 64

    def predict(
        self, keypoints: np.ndarray, image_size: tuple[int, int]
    ) -> np.ndarray:
        if self.name == "motionbert":
            keypoints = normalize_motionbert_pose_2d(keypoints, image_size)
        inputs = torch.from_numpy(np.asarray(keypoints, dtype=np.float32)).to(self.device)
        if inputs.ndim != 3 or inputs.shape[1:] != (17, 3):
            raise ValueError(f"Expected normalized T x 17 x 3 input, got {tuple(inputs.shape)}")
        self.model.eval()
        with torch.inference_mode():
            if self.name == "videopose3d":
                prediction = self._predict_videopose3d(inputs)
            elif self.name == "poseformer":
                prediction = self._predict_poseformer(inputs)
            elif self.name == "motionbert":
                prediction = self._predict_motionbert(inputs, image_size=image_size)
            else:
                raise ValueError(f"Unknown pose lifter: {self.name}")
        prediction = prediction - prediction[:, :1, :]
        return prediction.detach().cpu().numpy().astype(np.float32, copy=False)

    def _tta(self, inputs: torch.Tensor) -> torch.Tensor:
        plain = self.model(inputs)
        flipped = _flip_h36m(self.model(_flip_h36m(inputs)))
        return 0.5 * (plain + flipped)

    def _predict_videopose3d(self, inputs: torch.Tensor) -> torch.Tensor:
        xy = inputs[..., :2]
        radius = self.temporal_length // 2
        padded = torch.cat(
            [xy[:1].repeat(radius, 1, 1), xy, xy[-1:].repeat(radius, 1, 1)]
        ).unsqueeze(0)
        output = self._tta(padded).squeeze(0)
        if output.shape[0] != xy.shape[0]:
            raise RuntimeError(
                f"VideoPose3D returned {output.shape[0]} frames for {xy.shape[0]} inputs"
            )
        return output

    def _predict_poseformer(self, inputs: torch.Tensor) -> torch.Tensor:
        windows = centered_temporal_windows(inputs[..., :2], self.temporal_length)
        outputs: list[torch.Tensor] = []
        for start, end in temporal_chunks(len(windows), self.batch_size):
            outputs.append(self._tta(windows[start:end]).squeeze(1))
        return torch.cat(outputs, dim=0)

    def _predict_motionbert(
        self, inputs: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start, end in temporal_chunks(len(inputs), self.temporal_length):
            outputs.append(self._tta(inputs[start:end].unsqueeze(0)).squeeze(0))
        normalized = torch.cat(outputs, dim=0)
        # Match MMPose MotionBERTLabel.decode: res_w/2, fallback factor=4,
        # then millimetres to metres.
        return normalized * motionbert_output_scale_m(image_size)


def load_official_lifter(
    frontend: str,
    repo_path: Path,
    checkpoint_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
    motionbert_config: Path | None = None,
    poseformer_frames: int = 81,
    allow_unsafe_checkpoint: bool = False,
    allow_numpy_checkpoint_state: bool = False,
) -> OfficialPoseLifter:
    """Instantiate one official repository model with a strict checkpoint load."""

    name = frontend.lower().replace("-", "")
    aliases = {"vp3d": "videopose3d", "videopose3d": "videopose3d", "poseformer": "poseformer", "motionbert": "motionbert"}
    if name not in aliases:
        raise ValueError(f"Unsupported frontend {frontend!r}")
    name = aliases[name]
    repo_path = Path(repo_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Official repository not found: {repo_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if name == "videopose3d":
        module = _load_python_file(
            "canonfuse_official_videopose3d", repo_path / "common/model.py"
        )
        model = module.TemporalModel(17, 2, 17, [3, 3, 3, 3, 3])
        temporal_length = int(model.receptive_field())
    elif name == "poseformer":
        module = _load_python_file(
            "canonfuse_official_poseformer", repo_path / "common/model_poseformer.py"
        )
        model = module.PoseTransformer(
            num_frame=int(poseformer_frames),
            num_joints=17,
            in_chans=2,
            embed_dim_ratio=32,
            depth=4,
            num_heads=8,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_path_rate=0.0,
        )
        temporal_length = int(poseformer_frames)
    else:
        import yaml

        config_path = (
            Path(motionbert_config).resolve()
            if motionbert_config
            else repo_path / "configs/pose3d/MB_ft_h36m.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        with _temporary_import_root(repo_path, "lib"):
            module = _load_python_file(
                "canonfuse_official_motionbert", repo_path / "lib/model/DSTformer.py"
            )
            model = module.DSTformer(
                dim_in=2 if bool(config.get("no_conf", False)) else 3,
                dim_out=3,
                dim_feat=int(config["dim_feat"]),
                dim_rep=int(config["dim_rep"]),
                depth=int(config["depth"]),
                num_heads=int(config["num_heads"]),
                mlp_ratio=float(config["mlp_ratio"]),
                maxlen=int(config["maxlen"]),
                num_joints=int(config.get("num_joints", 17)),
                att_fuse=bool(config.get("att_fuse", True)),
            )
        temporal_length = int(config["maxlen"])

    incompatible = model.load_state_dict(
        _checkpoint_state(
            checkpoint_path,
            allow_unsafe_checkpoint=allow_unsafe_checkpoint,
            allow_numpy_checkpoint_state=allow_numpy_checkpoint_state,
        ),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    return OfficialPoseLifter(
        name=name,
        model=model,
        device=torch_device,
        temporal_length=temporal_length,
        batch_size=int(batch_size),
    )
