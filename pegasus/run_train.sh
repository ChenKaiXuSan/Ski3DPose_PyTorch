#!/bin/bash
#PBS -A SKIIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N pole_ski_train
#PBS -t 0-4
#PBS -o logs/pegasus/train_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SKIING/chenkaixu/code/Ski3DPose_PyTorch"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/

source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

conda env list

# === 2. 训练参数（按需修改） ===
DATA_ROOT="/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset"
INDEX_MAPPING_DIR="${DATA_ROOT}/index_mapping"
INDEX_MAPPING_PATH="${INDEX_MAPPING_DIR}/camera_pairs_by_action_folds"

MODEL_BACKBONE="pose2equip"
FUSE_METHOD="mamba_ssm"

NUM_WORKERS=32
BATCH_SIZE=16

# fold assignment:
# - PBS array mode: use PBS_ARRAY_INDEX
# - non-array/manual mode: allow env FOLD_ID override, default 0
FOLD_ID=${PBS_SUBREQNO:-${FOLD_ID:-0}}

echo "🏁 Train job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Index Mapping: ${INDEX_MAPPING_PATH}"
echo "GPU: 0, Epochs: ${MAX_EPOCHS}, Workers: ${NUM_WORKERS}"
echo "Backbone: ${MODEL_BACKBONE}, Fuse: ${FUSE_METHOD}"
echo "Fold: ${FOLD_ID}"

# === 3. 执行训练（每个作业只跑一个 fold） ===
python -m project.main \
    data.root_path=${DATA_ROOT} \
    data.index_mapping=${INDEX_MAPPING_DIR} \
    data.index_mapping_path=${INDEX_MAPPING_PATH} \
    train.gpu=0 \
    data.num_workers=${NUM_WORKERS} \
    data.batch_size=${BATCH_SIZE} \
    model.backbone=${MODEL_BACKBONE} \
    model.fuse_method=${FUSE_METHOD} \
    train.fold=${FOLD_ID}


echo "🏁 Train job finished at: $(date)"