#!/bin/bash

#SBATCH --job-name=decode
#SBATCH --output=logs/decode_%A_%a.out
#SBATCH --error=logs/decode_%A_%a.out
#SBATCH --partition=almanach,gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --hint=nomultithread
#SBATCH --account=almanach
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|a100"
#SBATCH --time=04:00:00
#SBATCH --array=0-1

en="eng_Latn"
langs=(fra_Latn por_Latn zho_Hans swh_Latn arb_Arab)
types=(lang trad)

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

type_index=$SLURM_ARRAY_TASK_ID
if [ $type_index -ge ${#types[@]} ]; then
    echo "Error: type_index $type_index is out of range (max: $((${#types[@]}-1)))"
    exit 1
fi

type_task=${types[$type_index]}
echo "Processing task type: $type_task"

# For loop over language pairs
for lang in "${langs[@]}"; do
    # en -> lang
    source=$en
    target=$lang
    echo "Decoding language pair: $source to $target"
    uv run src/representation/decode.py $1 $type_task $source $target
done



