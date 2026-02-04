#!/bin/bash

#SBATCH --job-name=generation
#SBATCH --output=logs/generation_amp_%A_%a.out
#SBATCH --error=logs/generation_amp_%A_%a.out
#SBATCH --partition=defq
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

module purge
module load cuda12.8/toolkit/12.8.1 cuda12.8/fft/12.8.1 cuda12.8/blas/12.8.1

model=$1
source=$2
target=$3
perc_heads=$4

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for i in $(seq -3 0.5 3); do
    if [[ "$i" != "0" && "$i" != "0.0" ]]; then
        apptainer exec --nv -B $XDG_CACHE_HOME mitra.sif python3.12 src/amplification_factor/generate.py $model $source $target $source $target $perc_heads $perc_heads $i
    fi
done