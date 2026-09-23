#!/bin/bash
#SBATCH --job-name=lep-classification
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/classification_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
MODEL="${1:?usage: sbatch slurm/classification.sbatch <model> <tag>}"
TAG="${2:?}"
for SEED in 1 2 3; do
  for SHOTS in 0 3; do
    python -m scripts.eval.classification --model "${MODEL}" --shots "${SHOTS}" --seed "${SEED}" \
      --out "results/downstream_preds_mcq/${TAG}"
  done
done
