#!/bin/bash

#SBATCH --job-name=logprobs_diff_lang
#SBATCH --output=logs/logprobs_diff_%A_%a.out
#SBATCH --error=logs/logprobs_diff_%A_%a.out
#SBATCH --array=0-3
#SBATCH --partition=gpu_p5
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --account=zln@a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=04:00:00

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nshot=$(($SLURM_ARRAY_TASK_ID + 1))

echo "Model: google/gemma-3-4b-pt - Language pair: $source -> $target - N-shot: $nshot"
uv run src/translation_nshot_task/lang.py google/gemma-3-4b-pt eng_Latn fra_Latn $nshot