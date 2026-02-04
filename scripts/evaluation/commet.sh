#!/bin/bash

#SBATCH --job-name=commet
#SBATCH --output=logs/commet_%A_%a.out
#SBATCH --error=logs/commet_%A_%a.out
#SBATCH --array=0-8
#SBATCH --partition=defq
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

models=("google/gemma-3-270m" "google/gemma-3-1b-pt" "google/gemma-3-4b-pt" "google/gemma-3-12b-pt" "Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")

# Slurm array task id
model=${models[$SLURM_ARRAY_TASK_ID]}

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv -B $XDG_CACHE_HOME mitra_commet.sif python3.12 src/evaluation/commet.py $model $1