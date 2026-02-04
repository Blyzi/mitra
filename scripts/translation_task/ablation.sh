#!/bin/bash

#SBATCH --job-name=ablation
#SBATCH --output=logs/ablation%A_%a.out
#SBATCH --error=logs/ablation%A_%a.out
#SBATCH --array=0-19
#SBATCH --partition=defq
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

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
export TOKENIZERS_PARALLELISM=false
# export TORCH_COMPILE_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1

for lang_head in $(seq 0 5); do
    for trad_head in $(seq 0 5); do
        apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_task/ablate.py $1 $source $target $source $target $trad_head $lang_head 
    done
done


