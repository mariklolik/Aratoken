#!/bin/bash
#SBATCH --job-name=lep-500m
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/flagship/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
VARIANT="${1:-preserve}"
case "${VARIANT}" in
  preserve) MODEL=models/qwen3-0.6b-fw2hq-alif-preserve-150k-mean
            TOKENIZER=tokenizers/extended/fw2hq-alif-preserve-150k
            DATA=data/pretokenized/fw2hq-alif-preserve FREEZE=true ;;
  cpt)      MODEL=Qwen/Qwen3-0.6B-Base TOKENIZER=Qwen/Qwen3-0.6B-Base
            DATA=data/pretokenized/fw2hq-qwen FREEZE=false ;;
  *) echo "usage: sbatch slurm/train_500m.sbatch preserve|cpt" >&2; exit 1 ;;
esac
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  --main_process_port $((29500 + SLURM_JOB_ID % 1000)) -m scripts.lep.train \
  seed=1 model_path="${MODEL}" tokenizer_path="${TOKENIZER}" data_path="${DATA}" \
  freeze_old_embeddings="${FREEZE}" include_lm_head_freeze="${FREEZE}" \
  training.max_steps=5086 training.per_device_train_batch_size=8 \
  training.gradient_accumulation_steps=3 training.save_steps=1000 training.eval_steps=500 \
  hydra.run.dir="outputs/flagship/${VARIANT}_s1"
