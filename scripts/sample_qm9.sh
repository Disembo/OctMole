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
OUT_DIR="${OUT_DIR:-output/qm9_oct64}"
STAGE1_MODEL="${STAGE1_MODEL:-logs/qm9_oct64/checkpoint-last.pth}"
STAGE2_MODEL="${STAGE2_MODEL:-logs/qm9_feat64/checkpoint-1000.pth}"
EXTRA_ARGS=()
if [[ "${STAGE1_GT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--stage1-gt)
fi

uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" tools/molecule_end2end.py \
  --data-path "${DATA_PATH}" \
  --out-dir "${OUT_DIR}" \
  --octree-depth 6 \
  --box-extent 15.36 \
  --metrics-dataset qm9 \
  --stage1-model "${STAGE1_MODEL}" \
  --stage2-model "${STAGE2_MODEL}" \
  --stage1-bsz 256 \
  --stage2-bsz 256 \
  --num-samples 10000 \
  --seed "${SEED:-0}" \
  --device cuda \
  "${EXTRA_ARGS[@]}"
