#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""HF SAM3 image inference utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig, ListConfig
from PIL import Image
from transformers import Sam3Model, Sam3Processor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from .vis import overlay_masks

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
ImageInput = Union[np.ndarray, Image.Image, str, Path]
BBoxInput = Optional[Union[List[float], Tuple[float, float, float, float]]]
PromptInput = Optional[Union[str, Sequence[str]]]


def list_capture_image_files(capture_dir: Path) -> List[Path]:
    """List image files under one capture directory in deterministic order."""
    return sorted(
        [
            p
            for p in capture_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def load_sam3d_bbox(npz_path: Path) -> Optional[List[float]]:
    """Load a single XYXY bbox from a SAM3D-Body result npz file.

    Supports two layouts:
    - Unity:  data['output'] is a scalar object dict  → data['output'].item()['bbox']
    - Person: data['outputs'] is shape-(1,) object    → data['outputs'][0]['bbox']

    Returns a list of 4 floats [x1, y1, x2, y2], or None if the file is missing /
    the bbox cannot be parsed.
    """
    if not npz_path.is_file():
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "output" in data:
            result_dict = data["output"].item()
        elif "outputs" in data:
            result_dict = data["outputs"][0]
        else:
            logger.warning("Unknown SAM3D-Body npz format: %s", npz_path)
            return None
        bbox = result_dict["bbox"]
        return [float(x) for x in bbox[:4]]
    except Exception as exc:
        logger.warning("Failed to load bbox from %s: %s", npz_path, exc)
        return None


def _chunks(items: List[Path], chunk_size: int):
    size = max(int(chunk_size), 1)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _flatten_path_chunks(path_chunks):
    """Flatten chunked path iterables for linear progress display."""
    for chunk in path_chunks:
        for p in chunk:
            yield p


class HFSam3ImageInferencer:
    """Batch image inference with HuggingFace Sam3Processor/Sam3Model."""

    def __init__(self, cfg: DictConfig) -> None:
        model_id = str(getattr(cfg.model, "hf_model_id", None))
        requested_device = str(getattr(cfg.infer, "gpu", 0)).lower()
        use_cpu = requested_device == "cpu" or not torch.cuda.is_available()

        self.device = torch.device("cpu" if use_cpu else "cuda")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.text_prompt = self._normalize_prompt(
            getattr(cfg.model, "text_prompt", None)
        )
        self.batch_size = max(
            int(getattr(cfg.model, "batch_size", None)),
            1,
        )
        self.threshold = float(getattr(cfg.model, "threshold", None))
        self.mask_threshold = float(getattr(cfg.model, "mask_threshold", None))
        self.save_overlay = bool(getattr(cfg.visualize, "overlay_masks", True))

        self.model.eval()
        logger.info(
            "HF Sam3 inferencer ready: model=%s, device=%s, batch_size=%d, prompt=%s",
            model_id,
            self.device,
            self.batch_size,
            self.text_prompt,
        )

    def _normalize_prompt(self, prompt: ListConfig) -> Union[str, List[str]]:
        prompt = list(prompt)
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            normalized = [str(x) for x in prompt if str(x).strip()]
            if not normalized:
                return "person"
            return normalized
        return "person"

    def _to_pil_rgb(self, img: ImageInput) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, (str, Path)):
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("img ndarray must be HxWx3")
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img).convert("RGB")
        raise TypeError(f"Unsupported img type: {type(img)}")

    def infer_mask(
        self,
        img: ImageInput,
        bbox: BBoxInput = None,
        prompt: PromptInput = None,
    ) -> Dict[str, Any]:
        """Infer SAM3 masks for one image with optional bbox and text prompt.

        Args:
            img: Input image (numpy RGB array / PIL / path).
            bbox: Optional XYXY bbox in absolute pixels.
            prompt: Optional text prompt. Defaults to configured text_prompt.

        Returns:
            dict containing masks/boxes/scores as numpy arrays.
        """
        pil_image = self._to_pil_rgb(img)
        text_prompt = prompt

        processor_kwargs: Dict[str, Any] = {
            "images": pil_image,
            "text": text_prompt,
            "return_tensors": "pt",
        }
        if bbox is not None:
            box_xyxy = [float(x) for x in bbox]
            processor_kwargs["input_boxes"] = [[box_xyxy]]
            processor_kwargs["input_boxes_labels"] = [[1]]

        with torch.no_grad():
            inputs = self.processor(**processor_kwargs).to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

        image_rgb = np.asarray(pil_image, dtype=np.uint8)
        boxes = result.get("boxes")
        scores = result.get("scores")
        masks = result.get("masks")

        boxes_np = (
            boxes.detach().cpu().numpy()
            if boxes is not None
            else np.empty((0, 4), dtype=np.float32)
        )
        scores_np = (
            scores.detach().cpu().numpy()
            if scores is not None
            else np.empty((0,), dtype=np.float32)
        )
        if masks is not None:
            masks_np = masks.detach().cpu().numpy().astype(np.uint8)
        else:
            h, w = image_rgb.shape[:2]
            masks_np = np.empty((0, h, w), dtype=np.uint8)

        return {
            "masks": masks_np,
            "boxes": boxes_np,
            "scores": scores_np,
            "image_rgb": image_rgb,
        }

    def process_capture(
        self,
        image_files: List[Path],
        out_dir: Path,
        infer_dir: Path,
        sam3d_result_dir: Optional[Path] = None,
    ) -> None:
        if not image_files:
            return

        prompts = (
            self.text_prompt
            if isinstance(self.text_prompt, list)
            else [self.text_prompt]
        )

        with torch.no_grad():
            for prompt in prompts:
                prompt_key = prompt.replace(" ", "_")
                prompt_infer_dir = infer_dir / prompt_key
                prompt_out_dir = out_dir / prompt_key
                prompt_infer_dir.mkdir(parents=True, exist_ok=True)
                if self.save_overlay:
                    prompt_out_dir.mkdir(parents=True, exist_ok=True)

                chunked_files = _chunks(image_files, self.batch_size)
                flat_files = _flatten_path_chunks(chunked_files)
                progress_desc = f"SAM3 capture[{prompt_key}]"
                for image_file in tqdm(
                    flat_files,
                    total=len(image_files),
                    desc=progress_desc,
                    unit="img",
                    leave=False,
                ):
                    stem = image_file.stem
                    bbox = None
                    if sam3d_result_dir is not None:
                        bbox = load_sam3d_bbox(
                            sam3d_result_dir / f"{stem}_sam3d_body.npz"
                        )
                    result = self.infer_mask(
                        img=image_file, bbox=bbox, prompt=prompt
                    )
                    image_rgb = result["image_rgb"]
                    masks_np = result["masks"]

                    np.savez_compressed(
                        prompt_infer_dir / f"{stem}.npz",
                        boxes=result["boxes"],
                        scores=result["scores"],
                        masks=masks_np,
                    )

                    if self.save_overlay:
                        overlay = overlay_masks(image_rgb, masks_np)
                        overlay_bgr = overlay[:, :, ::-1]
                        from cv2 import imwrite

                        imwrite(str(prompt_out_dir / f"{stem}.jpg"), overlay_bgr)

    def process_rgb_frames(
        self,
        frames: List[np.ndarray],
        out_dir: Path,
        infer_dir: Path,
        frame_prefix: str = "frame",
        sam3d_result_dir: Optional[Path] = None,
    ) -> None:
        """Process a list of in-memory RGB frames and save masks/visualizations."""
        if not frames:
            return

        prompts = (
            self.text_prompt
            if isinstance(self.text_prompt, list)
            else [self.text_prompt]
        )

        for prompt in prompts:
            prompt_key = prompt.replace(" ", "_")
            prompt_infer_dir = infer_dir / prompt_key
            prompt_out_dir = out_dir / prompt_key
            prompt_infer_dir.mkdir(parents=True, exist_ok=True)
            if self.save_overlay:
                prompt_out_dir.mkdir(parents=True, exist_ok=True)

            progress_desc = f"SAM3 video[{prompt_key}]"
            for idx, frame in enumerate(
                tqdm(
                    frames,
                    total=len(frames),
                    desc=progress_desc,
                    unit="frame",
                    leave=False,
                )
            ):
                stem = f"{frame_prefix}_{idx:06d}"
                bbox = None
                if sam3d_result_dir is not None:
                    # Person SAM3D-Body files: frame_{N:04d}_sam_3d_body_outputs.npz
                    bbox = load_sam3d_bbox(
                        sam3d_result_dir / f"frame_{idx:04d}_sam_3d_body_outputs.npz"
                    )
                result = self.infer_mask(img=frame, bbox=bbox, prompt=prompt)
                image_rgb = result["image_rgb"]
                masks_np = result["masks"]

                np.savez_compressed(
                    prompt_infer_dir / f"{stem}.npz",
                    boxes=result["boxes"],
                    scores=result["scores"],
                    masks=masks_np,
                )

                if self.save_overlay:
                    overlay = overlay_masks(image_rgb, masks_np)
                    overlay_bgr = overlay[:, :, ::-1]
                    from cv2 import imwrite

                    imwrite(str(prompt_out_dir / f"{stem}.jpg"), overlay_bgr)
