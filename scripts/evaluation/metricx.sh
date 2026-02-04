#!/bin/bash

#SBATCH --job-name=metricx
#SBATCH --output=logs/metricx_%A_%a.out
#SBATCH --error=logs/metricx_%A_%a.out
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-17

models=("google/gemma-3-270m" "google/gemma-3-1b-pt" "google/gemma-3-4b-pt" "google/gemma-3-12b-pt" "Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")

# Slurm array task id / 2
model=${models[$((SLURM_ARRAY_TASK_ID / 2))]}

# string 'true' if even, 'false' if odd
is_qe=$([ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ] && echo "true" || echo "false")

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv -B $XDG_CACHE_HOME mitra_metricx.sif python3.12 src/evaluation/metricx.py $model $1 $is_qe



