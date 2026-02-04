#!/bin/bash

#SBATCH --job-name=token_pos_generation
#SBATCH --output=logs/token_pos_lang_%A_%a.out
#SBATCH --error=logs/token_pos_lang_%A_%a.out
#SBATCH --array=0-7
#SBATCH --partition=defq
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

# Define arrays
en="eng_Latn"
langs=(fra_Latn swh_Latn)
indexes=(0 2 4 "random")

# Get number of elements
num_langs=${#langs[@]}
num_indexes=${#indexes[@]}
total_tasks=$((num_langs * num_indexes))

task_id=$SLURM_ARRAY_TASK_ID

if [ $task_id -ge $total_tasks ]; then
    echo "Error: task_id $task_id is out of range (max: $((total_tasks-1)))"
    exit 1
fi

# Map task_id to (lang, index) pair
lang_idx=$((task_id / num_indexes))
index_idx=$((task_id % num_indexes))

lang=${langs[$lang_idx]}
index=${indexes[$index_idx]}
source=$en
target=$lang

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Model: $1 - Language pair: $source -> $target - Index: $index"

apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/token_position/generate.py $1 $source $target $source $target $2 $2 $index