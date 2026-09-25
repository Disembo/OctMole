#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f main.py ]]; then
  echo "Run this script from the repo's root directory."
  exit 1
fi

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_PATH="${DATA_PATH:-data/drug_data}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/drug_feat64}"
RESUME="${RESUME:-${OUTPUT_DIR}}"

uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
  --train_mode molecule_feature \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${RESUME}" \
  --model MoleculeDiT-M \
  --voxel_size 64 \
  --box_extent 30.72 \
  --rope_coord_scale 5.0 \
  --atom_type_loss_weight 0.1 \
  --epochs 11 \
  --warmup_epochs 5 \
  --batch_size 128 \
  --blr 5e-5 \
  --min_lr 0.0 \
  --lr_schedule constant \
  --sampling_method heun \
  --num_sampling_steps 50 \
  --num_samples 1000 \
  --eval_freq 1 \
  --online_eval \
  --gen_bsz 128 \
  --random_rotation \
  --random_cell_shift \
  --num_workers 8 \
  --pin_mem \
  --save_last_freq 1 \
  --save_freq 10 \
  --log_freq 100 \
  --device cuda
