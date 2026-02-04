#!/bin/bash

#SBATCH --job-name=few_shot_baseline
#SBATCH --output=logs/few_shot_baseline_%A_%a.out
#SBATCH --error=logs/few_shot_baseline_%A_%a.out
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-8

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

models=("google/gemma-3-270m" "google/gemma-3-1b-pt" "google/gemma-3-4b-pt" "google/gemma-3-12b-pt" "Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")
# nshots=(0 1 2 3 5 10 20 50)

en="eng_Latn"
langs=(fra_Latn spa_Latn por_Latn jpn_Jpan zho_Hans swh_Latn wol_Latn hin_Deva arb_Arab rus_Cyrl)

model=${models[$SLURM_ARRAY_TASK_ID]}

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

echo "Running model: $1 - Language: $lang"

# Loop over languages

for lang in "${langs[@]}"; do
    apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_nshot_task/generate_few_shot.py $model $en $lang 5
    apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/translation_nshot_task/generate_few_shot.py $model $lang $en 5
done
