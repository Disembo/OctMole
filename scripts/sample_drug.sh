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
OUT_DIR="${OUT_DIR:-output/drug_oct64_2}"
STAGE1_MODEL="${STAGE1_MODEL:-logs/drug_oct64/checkpoint-last.pth}"
STAGE2_MODEL="${STAGE2_MODEL:-logs/drug_feat64/checkpoint-10.pth}"
EXTRA_ARGS=()
if [[ "${STAGE1_GT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--stage1-gt)
fi

uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" tools/molecule_end2end.py \
  --data-path "${DATA_PATH}" \
  --out-dir "${OUT_DIR}" \
  --octree-depth 6 \
  --box-extent 30.72 \
  --metrics-dataset drug \
  --stage1-model "${STAGE1_MODEL}" \
  --stage2-model "${STAGE2_MODEL}" \
  --stage1-bsz 128 \
  --stage2-bsz 128 \
  --num-samples 10000 \
  --seed "${SEED:-0}" \
  --device cuda \
  "${EXTRA_ARGS[@]}"
