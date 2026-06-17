#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N dual2pose_train
#PBS -t 0-2
#PBS -o logs/pegasus/dual2pose/train_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/dual2pose/train_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SKIING/chenkaixu/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/dual2pose

source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

conda env list

# === 2. 训练参数（按需修改） ===
DATA_ROOT="/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset"
INDEX_MAPPING_DIR="${DATA_ROOT}/index_mapping"
INDEX_MAPPING_PATH="${INDEX_MAPPING_DIR}/use_layer_camera_filter_enabled/camera_pairs_by_action_folds"

MODEL_BACKBONE="dual2pose"

NUM_WORKERS=24
BATCH_SIZE=4

# fold assignment:
# - PBS array mode: use PBS_ARRAY_INDEX
# - non-array/manual mode: allow env FOLD_ID override, default 0
FOLD_ID=${PBS_SUBREQNO:-${FOLD_ID:-0}}
printf -v FOLD_FILE "fold_%02d.json" "${FOLD_ID}"

echo "🏁 Train job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Index Mapping: ${INDEX_MAPPING_PATH}/${FOLD_FILE}"
echo "GPU: 0, Epochs: ${MAX_EPOCHS}, Workers: ${NUM_WORKERS}"
echo "Backbone: ${MODEL_BACKBONE}"
echo "Fold: ${FOLD_ID}"

# === 3. 执行训练（每个作业只跑一个 fold） ===
python dual2pose/train_unity.py \
    data.unity.root_path=${DATA_ROOT} \
    data.unity.index_mapping_path=${INDEX_MAPPING_PATH}/${FOLD_FILE} \
    train.gpu=0 \
    data.num_workers=${NUM_WORKERS} \
    data.batch_size=${BATCH_SIZE} \
    model.backbone=${MODEL_BACKBONE} \
    train.fold=${FOLD_ID}

echo "🏁 Train job finished at: $(date)"