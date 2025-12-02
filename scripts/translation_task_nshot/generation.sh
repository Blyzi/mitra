#!/bin/bash

#SBATCH --job-name=generation
#SBATCH --output=logs/generation_%A_%a.out
#SBATCH --error=logs/generation_%A_%a.out
#SBATCH --array=0-3
#SBATCH --partition=gpu_p5
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --account=zln@a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=10:00:00

nshot=$(($SLURM_ARRAY_TASK_ID + 1))
source="eng_Latn"
target="fra_Latn"
model="google/gemma-3-4b-pt"

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for lang_head in $(seq 0 4); do
    for trad_head in $(seq 0 4); do
        echo "Processing language pair: $source to $target with lang_head=$lang_head and trad_head=$trad_head"
        uv run src/translation_nshot_task/generate.py $model $source $target $source $target $trad_head $lang_head $nshot
    done
done


