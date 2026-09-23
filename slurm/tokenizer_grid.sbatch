#!/bin/bash
#SBATCH --job-name=tok-grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/tok_grid_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CELLS=(
  "bpe drop off" "bpe drop on" "bpe keep off" "bpe keep on"
  "wp drop off" "wp drop on" "wp keep off" "wp keep on"
  "sp drop off" "sp drop on" "sp keep off" "sp keep on"
)
read -r ALGO DIA NORM <<< "${CELLS[$((SLURM_ARRAY_TASK_ID - 1))]}"
NAME="${ALGO}_${DIA}$([ "${NORM}" = on ] && echo _norm)"
OUT="tokenizers/fw2hq_${NAME}"
ARGS=(--vocab-size 80000 --dataset-shard-dir data/fineweb2hq-ar/tok-train)
[ "${DIA}" = drop ] && ARGS+=(--drop-diacritics)
[ "${NORM}" = off ] && ARGS+=(--no-arabic-normalization)
ALGO="${ALGO/wp/wordpiece}"
python -m scripts.tokenizer.train "${ALGO}" "${OUT}" "${ARGS[@]}"
