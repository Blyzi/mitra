#!/bin/bash

#SBATCH --job-name=top_head
#SBATCH --output=logs/top_head_%A_%a.out
#SBATCH --error=logs/top_head_%A_%a.out
#SBATCH --array=0-19
#SBATCH --partition=almanach,gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --hint=nomultithread
#SBATCH --account=almanach
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|a100"
#SBATCH --time=04:00:00

en="eng_Latn"
arr=(fra_Latn spa_Latn por_Latn jpn_Jpan zho_Hans swh_Latn wol_Latn hin_Deva arb_Arab rus_Cyrl)

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
    lang=${arr[$task_id]}
    source=$en
    target=$lang
else
    # lang -> en
    lang_index=$((task_id - num_langs))
    lang=${arr[$lang_index]}
    source=$lang
    target=$en
fi

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For loop over type lang and trad
for type_task in lang trad; do
    echo "Processing language pair: $source to $target for task type: $type_task"
    uv run src/representation/top_head.py $1 $type_task $source $target
done


