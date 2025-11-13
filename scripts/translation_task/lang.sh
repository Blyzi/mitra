#!/bin/bash

#SBATCH --job-name=logprobs_diff_lang
#SBATCH --output=logs/logprobs_diff_lang_%A_%a.out
#SBATCH --error=logs/logprobs_diff_lang_%A_%a.out
#SBATCH --array=0-13
#SBATCH --partition=almanach,gpu
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --account=almanach
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=02:00:00

en="eng_Latn"
arr=(fra_Latn spa_Latn por_Latn jpn_Jpan zho_Hans swh_Latn wol_Latn)

# Get number of elements
num_langs=${#arr[@]}
total_pairs=$((num_langs * 2))

task_id=$SLURM_ARRAY_TASK_ID


if [ $task_id -ge $total_pairs ]; then
    echo "Error: task_id $task_id is out of range (max: $((total_pairs-1)))"
    exit 1
fi

# Determine direction and language
if [ $task_id -lt $num_langs ]; then
    # en -> lang
    source=$en
    target=${arr[$task_id]}
else
    # lang -> en
    lang_index=$((task_id - num_langs))
    source=${arr[$lang_index]}
    target=$en
fi

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running command with: $source $target"
uv run src/translation_task/lang.py $1 $source $target