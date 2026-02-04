#!/bin/bash

#SBATCH --job-name=transfer
#SBATCH --output=logs/transfer_%A_%a.out
#SBATCH --error=logs/transfer_%A_%a.out
#SBATCH --array=0-19
#SBATCH --partition=defq
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1
set -ex

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
    lang_selected=${arr[$task_id]}
    source_trad=$en
    target_trad=$lang_selected
else
    # lang -> en
    lang_index=$((task_id - num_langs))
    lang_selected=${arr[$lang_index]}
    source_trad=$lang_selected
    target_trad=$en
fi

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# export TORCH_COMPILE_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1

for lang in "${arr[@]}"; do
    echo "Model: $1 - Translation pair: $source_trad -> $target_trad - Language: $lang -> $en"
    apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_task/generate.py $1 $source_trad $target_trad $lang $en 1 1
    echo "Model: $1 - Translation pair: $source_trad -> $target_trad - Language: $en -> $lang"
    apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_task/generate.py $1 $source_trad $target_trad $en $lang 1 1
done


