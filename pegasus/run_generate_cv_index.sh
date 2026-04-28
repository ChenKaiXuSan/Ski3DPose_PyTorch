#!/usr/bin/env bash
# === 生成交叉验证索引文件 ===

set -euo pipefail

# === 1. 环境准备 ===
PROJECT_ROOT="${PROJECT_ROOT:-/work/SKIING/chenkaixu/code/Ski3DPose_PyTorch}"
cd "${PROJECT_ROOT}" || exit 1

CONDA_SH="${CONDA_SH:-/home/SKIING/chenkaixu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] conda.sh 不存在: ${CONDA_SH}"
    exit 1
fi

source "${CONDA_SH}"
conda deactivate || true
conda activate "${CONDA_ENV_PATH}"

conda env list

# === 2. 参数配置 ===
DATA_ROOT="${DATA_ROOT:-/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset}"
STRATEGIES="${STRATEGIES:-[by_action]}"
N_SPLITS="${N_SPLITS:-5}"
FORCE_RECREATE="${FORCE_RECREATE:-false}"
USE_LAYER_CAMERA_FILTER="${USE_LAYER_CAMERA_FILTER:-true}"
SELECTED_LAYERS="${SELECTED_LAYERS:-[0,1,2,3,4]}"
SELECTED_CAMERAS_PER_LAYER="${SELECTED_CAMERAS_PER_LAYER:-{0:[0,90,180,270],1:[0,90,180,270],2:[0,90,180,270],3:[0,90,180,270],4:[0,90,180,270]}}"

echo "=============================="
echo " 生成交叉验证索引"
echo "=============================="
echo "Project Root : ${PROJECT_ROOT}"
echo "Data Root    : ${DATA_ROOT}"
echo "Strategies   : ${STRATEGIES}"
echo "N Splits     : ${N_SPLITS}"
echo "Force Recreate: ${FORCE_RECREATE}"
echo "Use Layer Filter: ${USE_LAYER_CAMERA_FILTER}"
echo "Selected Layers: ${SELECTED_LAYERS}"
echo "Selected Cameras Per Layer: ${SELECTED_CAMERAS_PER_LAYER}"
echo "Started at   : $(date)"
echo "=============================="

# === 3. 执行生成 ===
python -m cross_validation.generate_cv_index \
    data.root_path="${DATA_ROOT}" \
    data.cross_validation.strategies="${STRATEGIES}" \
    data.cross_validation.n_splits="${N_SPLITS}" \
    data.cross_validation.force_recreate="${FORCE_RECREATE}" \
    data.cross_validation.use_layer_camera_filter="${USE_LAYER_CAMERA_FILTER}" \
    data.cross_validation.selected_layers="${SELECTED_LAYERS}" \
    data.cross_validation.selected_cameras_per_layer="${SELECTED_CAMERAS_PER_LAYER}"

echo "=============================="
echo " 完成 at: $(date)"
echo "=============================="
