#!/bin/bash

#SBATCH --job-name=logprobs_diff_translation_nshot
#SBATCH --output=logs/logprobs_diff_translation_nshot_%A_%a.out
#SBATCH --error=logs/logprobs_diff_translation_nshot_%A_%a.out
#SBATCH --array=0-13
#SBATCH --partition=defq
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

# Define arrays
en="eng_Latn"
langs=(fra_Latn swh_Latn)
nshots=(0 1 2 3 10 20 50)

# Get number of elements
num_langs=${#langs[@]}
num_nshots=${#nshots[@]}
total_tasks=$((num_langs * num_nshots))

task_id=$SLURM_ARRAY_TASK_ID

if [ $task_id -ge $total_tasks ]; then
    echo "Error: task_id $task_id is out of range (max: $((total_tasks-1)))"
    exit 1
fi

# Map task_id to (lang, index) pair
lang_idx=$((task_id / num_nshots))
nshot_idx=$((task_id % num_nshots))

lang=${langs[$lang_idx]}
nshot=${nshots[$nshot_idx]}
source=$en
target=$lang

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Model: $1 - Language pair: $source -> $target - Nshot: $nshot"

apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_nshot_task/translation.py $1 $source $target $nshot