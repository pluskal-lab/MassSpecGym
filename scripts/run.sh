#!/bin/bash

echo "job_key \"${job_key}\""

# Prepare project environment
. "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate massspecgym

# Move to running dir
cd "${SLURM_SUBMIT_DIR}" || exit 1

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Test random baseline on retrieval
srun --export=ALL --preserve-env python3 run.py \
    --job_key="${job_key}" \
    --run_name=rebuttal_random_test_formula \
    --devices=1 \
    --test_only \
    --task=retrieval \
    --model=random \
    --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
    --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json"

# Test MIST on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_MIST_test_formula \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=from_dict \
#     --dct_path="./dct_MIST.pkl" \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json"

# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=debug \
#     --task=de_novo \
#     --model=smiles_transformer

# Obsolete (for FingerprintFFN retrieval)
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=debug \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --val_check_interval=0.5
