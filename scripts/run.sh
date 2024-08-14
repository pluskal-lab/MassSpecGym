#!/bin/bash

# Usage example:
# ./submit.sh run.sh 24:00 8

echo "job_key \"${job_key}\""

# Prepare project environment
. "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate massspecgym

# Move to running dir
cd "${SLURM_SUBMIT_DIR}" || exit 1

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train SmilesTransformer on de novo
srun --export=ALL --preserve-env python3 run.py \
    --devices=1 \
    --job_key="${job_key}" \
    --run_name=debug_rebuttal_smiles_transformer_train \
    --task=de_novo \
    --model=smiles_transformer \
    --log_only_loss_at_stages="train,val" \
    --batch_size=128 \
    --lr=0.0003 \
    --k_predictions=10 \
    --d_model=256 \
    --nhead=4 \
    --num_encoder_layers=6 \
    --smiles_tokenizer="selfies" \
    --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv"

# Train DeepSets on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_deepsets_train_mass \
#     --task=retrieval \
#     --model=deepsets \
#     --log_only_loss_at_stages="train,val" \
#     --val_check_interval=0.5 \
#     --batch_size=16 \
#     --lr=0.001 \
#     --hidden_channels=128 \
#     --num_layers_per_mlp=7 \
#     --dropout=0.1 \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json"

# Train FingerprintFFN on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_sigmoid_train_mass \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --log_only_loss_at_stages="train,val" \
#     --val_check_interval=0.5 \
#     --batch_size=16 \
#     --lr=0.001 \
#     --hidden_channels=128 \
#     --num_layers=7 \
#     --dropout=0.1 \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json"

# Test DeepSets from checkpoint on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_deepsets_test_mass \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=deepsets \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json" \
#     --checkpoint_pth="./MassSpecGymRetrieval/iVJ15YLWQN/step=006067-val_loss=0.589.ckpt"
#     # mass
#     # --checkpoint_pth="./MassSpecGymRetrieval/iVJ15YLWQN/step=006067-val_loss=0.589.ckpt"
#     # formula
#     # TODO

# Test FingerprintFFN from checkpoint on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_test_formula \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json" \
#     --checkpoint_pth="./MassSpecGymRetrieval/r3JTQeI884/step=004550-val_loss=0.589.ckpt"
#     # mass
#     # --checkpoint_pth="./MassSpecGymRetrieval/4pb6nyf8nT/step=000950-val_loss=0.587.ckpt"
#     # formula
#     # --checkpoint_pth="./MassSpecGymRetrieval/r3JTQeI884/step=004550-val_loss=0.589.ckpt"

# Test random baseline on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_random_test_mass \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=random \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json"

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
