#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f main.py ]]; then
  echo "Run this script from the repo's root directory."
  exit 1
fi

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_PATH="${DATA_PATH:-data/qm9_data}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/qm9_oct64}"
RESUME="${RESUME:-${OUTPUT_DIR}}"

uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
  --train_mode molecule_octree \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${RESUME}" \
  --model MoleculeOctreeDiT-B \
  --octree_depth 6 \
  --octree_depth_increment 2 \
  --box_extent 15.36 \
  --rope_coord_scale 1.0 \
  --self_condition \
  --epochs 4000 \
  --warmup_epochs 5 \
  --batch_size 256 \
  --blr 5e-5 \
  --lr_schedule constant \
  --sampling_method heun \
  --num_sampling_steps 50 \
  --num_samples 1024 \
  --eval_freq 10 \
  --online_eval \
  --gen_bsz 256 \
  --num_workers 8 \
  --pin_mem \
  --save_last_freq 5 \
  --log_freq 100 \
  --device cuda
