#!/bin/bash
#SBATCH --job-name=lep-grid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/grid/logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-outputs/grid/manifest.tsv}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
IFS=$'\t' read -r ARM NAME MODE TOKENIZER MODEL DATA LR LAYERS SEED MAX_STEPS OVERRIDES <<< "${LINE%$'\r'}"
FREEZE=true
[[ "${MODE}" == "cpt" ]] && FREEZE=false
echo "=== ${NAME} (${ARM}, ${MODE})"
python -m scripts.lep.train \
  seed="${SEED}" model_path="${MODEL}" tokenizer_path="${TOKENIZER}" data_path="${DATA}" \
  freeze_old_embeddings="${FREEZE}" include_lm_head_freeze="${FREEZE}" \
  unfreeze_transformer_layers="${LAYERS}" training.learning_rate="${LR}" \
  training.max_steps="${MAX_STEPS}" hydra.run.dir="outputs/grid/${NAME}" ${OVERRIDES}
