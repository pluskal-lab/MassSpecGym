#!/bin/bash

echo "job_key \"${job_key}\""

# Prepare project environment
. "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate massspecgym

# Move to running dir
cd "${SLURM_SUBMIT_DIR}" || exit 1

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --export=ALL --preserve-env python3 train.py \
    --job_key="${job_key}" \
    --run_name=debug \
    --task=de_novo \
    --model=smiles_transformer

# Obsolete (for FingerprintFFN retrieval)
# srun --export=ALL --preserve-env python3 train.py \
#     --job_key="${job_key}" \
#     --run_name=debug \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --val_check_interval=0.5
