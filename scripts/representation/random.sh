#!/bin/bash

#SBATCH --job-name=top_head_random
#SBATCH --output=logs/top_head_random_%A_%a.out
#SBATCH --error=logs/top_head_random_%A_%a.out
#SBATCH --array=0-8
#SBATCH --partition=almanach,gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --hint=nomultithread
#SBATCH --account=almanach
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|a100"
#SBATCH --time=04:00:00

models=("google/gemma-3-270m" "google/gemma-3-1b-pt" "google/gemma-3-4b-pt" "google/gemma-3-12b-pt" "Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")

model=${models[$SLURM_ARRAY_TASK_ID]}

# For loop over type lang and trad
echo "Processing model: $model for task type: trad and lang"
uv run src/representation/random_text.py $model trad
uv run src/representation/random_text.py $model lang



