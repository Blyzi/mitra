#!/bin/bash

#SBATCH --job-name=generation
#SBATCH --output=logs/generation_%A_%a.out
#SBATCH --error=logs/generation_%A_%a.out
#SBATCH --array=0-19
#SBATCH --partition=gpu_p6
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --account=zln@h100
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=02:00:00

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

for lang_head in $(seq 0 5); do
    for trad_head in $(seq 0 5); do
        echo "Processing language pair: $source to $target with lang_head=$lang_head and trad_head=$trad_head"
        uv run src/translation_task/generate.py $1 $source $target $source $target $trad_head $lang_head 
    done
done


