#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run SAM3 inference with unified DataConfig dispatch for unity/person data."""

import logging
import multiprocessing as mp
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import HFSam3ImageInferencer, list_capture_image_files
from .load import (
    DataConfig,
    build_person_data_configs,
    build_unity_data_configs,
    load_video_frames,
)

logger = logging.getLogger(__name__)


def resolve_runtime_paths(cfg: DictConfig):
    """Resolve source/inference/visualization roots from sam3.yaml format."""
    infer_type = str(getattr(cfg.infer, "type", "unity")).lower()

    if infer_type == "person":
        source_root = Path(cfg.paths.person.person_video_path).resolve()
        infer_root = Path(cfg.paths.person.person_sam3_result_root).resolve()
    else:
        source_root = Path(cfg.paths.unity.unity_dataset_data_path).resolve()
        infer_root = Path(cfg.paths.unity.unity_sam3_result_path).resolve()

    # When infer_root already points to an inference folder, place visual outputs nearby.
    vis_root = (
        infer_root.parent / "visualization"
        if infer_root.name == "inference"
        else infer_root / "visualization"
    )

    return infer_type, source_root, vis_root, infer_root


def build_data_configs(
    cfg: DictConfig,
    infer_type: str,
    source_root: Path,
) -> List[DataConfig]:
    """Build unified DataConfig list for current inference type."""
    if infer_type == "person":
        sam3d_root = Path(cfg.paths.person.person_sam3d_result_root).resolve()
        sam3_root = Path(cfg.paths.person.person_sam3_result_root).resolve()
        return build_person_data_configs(
            person_root=source_root,
            sam3d_root=sam3d_root,
            sam3_root=sam3_root,
        )

    sam3d_root = Path(cfg.paths.unity.unity_sam3d_result_path).resolve()
    sam3_root = Path(cfg.paths.unity.unity_sam3_result_path).resolve()
    return build_unity_data_configs(
        unity_root=source_root,
        sam3d_root=sam3d_root,
        sam3_root=sam3_root,
    )


def split_evenly(items: List[DataConfig], num_chunks: int) -> List[List[DataConfig]]:
    """Split a list into near-even contiguous chunks."""
    if num_chunks <= 0:
        return []

    n = len(items)
    base = n // num_chunks
    extra = n % num_chunks

    chunks: List[List[DataConfig]] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(items[start:end])
        start = end
    return chunks


def process_single_dataconfig(
    data_cfg: DataConfig,
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    hf_inferencer: HFSam3ImageInferencer,
) -> None:
    """Process one DataConfig item for SAM3 inference."""
    action_id = f"{data_cfg.subject_name}__{data_cfg.camera_name}"

    log_dir = action_log_root / "action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    action_log_file = log_dir / f"{action_id}.log"

    handler = logging.FileHandler(action_log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    action_logger = logging.getLogger(f"action_{action_id}")
    action_logger.handlers.clear()
    action_logger.setLevel(logging.INFO)
    action_logger.addHandler(handler)
    action_logger.propagate = False

    action_logger.info(
        "==== Start Item: subject=%s camera=%s frame_path=%s ====",
        data_cfg.subject_name,
        data_cfg.camera_name,
        data_cfg.frame_path,
    )

    infer_dir = data_cfg.sam3_output_path
    try:
        rel_out = infer_dir.relative_to(infer_root)
        out_dir = vis_root / rel_out
    except ValueError:
        out_dir = vis_root / data_cfg.subject_name / data_cfg.camera_name

    out_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    frame_path = data_cfg.frame_path
    if frame_path.is_dir():
        image_files = list_capture_image_files(frame_path)
        if not image_files:
            action_logger.warning("[Skip] Empty image capture dir: %s", frame_path)
            return

        action_logger.info(
            "Processing image capture: %s, frame_count=%d", frame_path, len(image_files)
        )
        hf_inferencer.process_capture(
            image_files=image_files,
            out_dir=out_dir,
            infer_dir=infer_dir,
            sam3d_result_dir=data_cfg.sam_3d_body_result_path,
        )
    elif frame_path.is_file():
        frame_list = load_video_frames(frame_path)
        if not frame_list:
            action_logger.warning("[Skip] Empty video capture: %s", frame_path)
            return

        action_logger.info(
            "Processing video capture: %s, frame_count=%d", frame_path, len(frame_list)
        )
        hf_inferencer.process_rgb_frames(
            frames=frame_list,
            out_dir=out_dir,
            infer_dir=infer_dir,
            frame_prefix=frame_path.stem,
            sam3d_result_dir=data_cfg.sam_3d_body_result_path,
        )
    else:
        action_logger.warning("[Skip] Unsupported frame_path: %s", frame_path)
        return

    action_logger.info("==== Finished Item: %s ====", action_id)


def gpu_worker(
    gpu_id: Union[int, str],
    data_configs: List[DataConfig],
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    cfg_dict: dict,
    worker_id: int,
) -> None:
    """Worker entrypoint: pin device and process assigned data configs."""
    is_cpu = isinstance(gpu_id, str) and gpu_id.lower() == "cpu"
    if is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    local_cfg_dict = deepcopy(cfg_dict)
    local_cfg_dict.setdefault("infer", {})
    local_cfg_dict["infer"]["gpu"] = "cpu" if is_cpu else 0
    cfg = OmegaConf.create(local_cfg_dict)
    hf_inferencer = HFSam3ImageInferencer(cfg)

    logger.info(
        "[Worker %d] GPU %s started, items=%d",
        worker_id,
        gpu_id,
        len(data_configs),
    )

    for data_cfg in data_configs:
        try:
            process_single_dataconfig(
                data_cfg,
                vis_root,
                infer_root,
                action_log_root,
                hf_inferencer=hf_inferencer,
            )
        except Exception as exc:
            logger.error(
                "[Worker %d] Failed on item %s/%s: %s",
                worker_id,
                data_cfg.subject_name,
                data_cfg.camera_name,
                exc,
            )

    logger.info("[Worker %d] GPU %s finished", worker_id, gpu_id)


def normalize_gpu_ids(raw_gpu_ids) -> List[Union[int, str]]:
    raw_gpu_ids = list(raw_gpu_ids)  # ListConfig > list
    """Normalize gpu config to a list of integer ids."""
    if isinstance(raw_gpu_ids, str) and raw_gpu_ids.lower() == "cpu":
        return ["cpu"]

    if isinstance(raw_gpu_ids, int):
        return [raw_gpu_ids]

    if isinstance(raw_gpu_ids, str):
        if "," in raw_gpu_ids:
            parsed_str_ids: List[Union[int, str]] = []
            for x in raw_gpu_ids.split(","):
                x = x.strip()
                if not x:
                    continue
                parsed_str_ids.append("cpu" if x.lower() == "cpu" else int(x))
            return parsed_str_ids
        return [int(raw_gpu_ids)]

    if isinstance(raw_gpu_ids, (list, tuple)):
        parsed_seq_ids: List[Union[int, str]] = []
        for x in raw_gpu_ids:
            if isinstance(x, str) and x.lower() == "cpu":
                parsed_seq_ids.append("cpu")
            else:
                parsed_seq_ids.append(int(x))
        return parsed_seq_ids

    return [0]


def select_action_shard(
    data_configs: List[DataConfig],
    shard_count: int,
    shard_index: int,
) -> List[DataConfig]:
    """Select one shard of actions for multi-node execution."""
    if shard_count <= 1:
        return data_configs

    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"Invalid shard index {shard_index} for shard_count {shard_count}"
        )

    shard_chunks = split_evenly(data_configs, shard_count)
    return shard_chunks[shard_index]


@hydra.main(config_path="../configs", config_name="sam3", version_base=None)
def main(cfg: DictConfig) -> None:
    infer_type, source_root, vis_root, infer_root = resolve_runtime_paths(cfg)
    action_log_root = Path(cfg.paths.log_path).resolve()
    vis_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = normalize_gpu_ids(cfg.infer.gpu)
    workers_per_gpu = int(cfg.infer.workers_per_gpu)
    workers_per_gpu = max(workers_per_gpu, 1)

    expanded_gpu_ids: List[Union[int, str]] = []
    for gid in gpu_ids:
        expanded_gpu_ids.extend([gid] * workers_per_gpu)

    total_workers = len(expanded_gpu_ids)
    if total_workers < 1:
        logger.error("No worker created. Check infer.gpu / infer.workers_per_gpu")
        return

    data_configs_all = build_data_configs(
        cfg,
        infer_type,
        source_root,
    )
    if not data_configs_all:
        logger.error("No data configs found in: %s", source_root)
        return

    shard_count = max(int(getattr(cfg.infer, "shard_count", 1)), 1)
    shard_index = int(getattr(cfg.infer, "shard_index", 0))

    try:
        data_configs = select_action_shard(data_configs_all, shard_count, shard_index)
    except ValueError as exc:
        logger.error("%s", exc)
        return

    if not data_configs:
        logger.warning(
            "No data items assigned to this shard (index=%d/%d). Exit.",
            shard_index,
            shard_count,
        )
        return

    # Persist shard assignment for post-run overlap checks across nodes.
    shard_log_dir = action_log_root / "shard_actions"
    shard_log_dir.mkdir(parents=True, exist_ok=True)
    shard_log_file = shard_log_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}.txt"
    with shard_log_file.open("w", encoding="utf-8") as f:
        f.write(f"shard_index={shard_index}\n")
        f.write(f"shard_count={shard_count}\n")
        f.write(f"total_items_all={len(data_configs_all)}\n")
        f.write(f"items_in_shard={len(data_configs)}\n")
        f.write("items:\n")
        for data_cfg in data_configs:
            f.write(
                f"{data_cfg.subject_name}\t{data_cfg.camera_name}\t{data_cfg.frame_path.as_posix()}\n"
            )

    logger.info("Shard action list saved: %s", shard_log_file)

    chunks = split_evenly(data_configs, total_workers)

    logger.info("Infer type: %s", infer_type)
    logger.info("Source data root: %s", source_root)
    logger.info("Inference root: %s", infer_root)
    logger.info("Visualization root: %s", vis_root)
    logger.info("GPU ids: %s, workers_per_gpu=%d", gpu_ids, workers_per_gpu)
    logger.info(
        "Shard: index=%d/%d, items_in_shard=%d, total_items=%d",
        shard_index,
        shard_count,
        len(data_configs),
        len(data_configs_all),
    )
    logger.info("Total workers: %d, total items: %d", total_workers, len(data_configs))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.error("Failed to convert config to dict")
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes: List[mp.Process] = []
    for i, gpu_id in enumerate(expanded_gpu_ids):
        item_list = chunks[i]
        if not item_list:
            continue

        logger.info(
            "Assign worker=%d, gpu=%s, item_count=%d",
            i,
            gpu_id,
            len(item_list),
        )

        process = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                item_list,
                vis_root,
                infer_root,
                action_log_root,
                cfg_dict,
                i,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    logger.info("[SUCCESS] All action workers completed")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()  # type: ignore[call-arg]
