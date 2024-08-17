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

# Train SmilesTransformer on de novo formula bonus challenge
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_smiles_transformer_formula_train \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --log_only_loss_at_stages=train,val \
#     --batch_size=64 \
#     --lr=0.0003 \
#     --k_predictions=10 \
#     --d_model=256 \
#     --nhead=4 \
#     --num_encoder_layers=3 \
#     --smiles_tokenizer=smiles_bpe \
#     --use_chemical_formula \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv"

# Train SelfiesTransformer on de novo formula bonus challenge
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_selfies_transformer_formula_train \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --log_only_loss_at_stages=train,val \
#     --batch_size=64 \
#     --lr=0.0003 \
#     --k_predictions=10 \
#     --d_model=256 \
#     --nhead=8 \
#     --num_encoder_layers=6 \
#     --smiles_tokenizer=selfies \
#     --use_chemical_formula \
#     --dataset_pth="../data/MIST_CANOPUS_with_MassSpecGym_test.tsv"

# Train SmilesTransformer on de novo using MIST CANOPUS dataset
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_smiles_transformer_mist_canopus_1550_train \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --log_only_loss_at_stages=train,val \
#     --batch_size=64 \
#     --lr=0.0003 \
#     --k_predictions=10 \
#     --d_model=256 \
#     --nhead=4 \
#     --num_encoder_layers=3 \
#     --smiles_tokenizer=smiles_bpe \
#     --dataset_pth=../data/MIST_CANOPUS_with_MassSpecGym_test.tsv \
#     --max_mz=1550

# Train SelfiesTransformer on de novo using MIST CANOPUS dataset
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_selfies_transformer_mist_canopus_1550_train \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --log_only_loss_at_stages=train,val \
#     --batch_size=64 \
#     --lr=0.0003 \
#     --k_predictions=10 \
#     --d_model=256 \
#     --nhead=8 \
#     --num_encoder_layers=6 \
#     --smiles_tokenizer=selfies \
#     --dataset_pth=../data/MIST_CANOPUS_with_MassSpecGym_test.tsv \
#     --max_mz=1550

# Test SmilesTransformer on de novo from checkpoint using MIST CANOPUS dataset
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_smiles_transformer_mist_canopus_1550_test \
#     --devices=1 \
#     --test_only \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --smiles_tokenizer="smiles_bpe" \
#     --dataset_pth="../data/MIST_CANOPUS_with_MassSpecGym_test.tsv" \
#     --max_mz=1550 \
#     --checkpoint_pth="./MassSpecGymDeNovo/3cQVKpZyEt/step=000600-val_loss=0.231.ckpt"
#     # smiles_bpe
#     # "./MassSpecGymDeNovo/3cQVKpZyEt/step=000600-val_loss=0.231.ckpt"
#     # selfies
#     # "./MassSpecGymDeNovo/Vsz61ZBQIu/step=000588-val_loss=0.350.ckpt"

# Train SmilesTransformer on de novo
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_smiles_transformer_train \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --log_only_loss_at_stages="train,val" \
#     --batch_size=128 \
#     --lr=0.0003 \
#     --k_predictions=10 \
#     --d_model=256 \
#     --nhead=4 \
#     --num_encoder_layers=6 \
#     --smiles_tokenizer="selfies" \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv"

# Test SmilesTransformer on de novo from checkpoint
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_selfies_transformer_test \
#     --devices=1 \
#     --test_only \
#     --task=de_novo \
#     --model=smiles_transformer \
#     --smiles_tokenizer="selfies" \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --checkpoint_pth="./MassSpecGymDeNovo/ni6D2Mxb3K/step=003800-val_loss=0.577.ckpt"
#     # smiles_bpe
#     # --checkpoint_pth="./MassSpecGymDeNovo/d4NkRiSueL/step=006460-val_loss=0.304.ckpt"
#     # selfies
#     # --checkpoint_pth="./MassSpecGymDeNovo/ni6D2Mxb3K/step=003800-val_loss=0.577.ckpt"

# Test SmilesTransformer on de novo from checkpoint (bonus formula challenge)
srun --export=ALL --preserve-env python3 run.py \
    --job_key="${job_key}" \
    --run_name=rebuttal_selfies_transformer_formula_test \
    --devices=1 \
    --test_only \
    --task=de_novo \
    --model=smiles_transformer \
    --smiles_tokenizer="selfies" \
    --use_chemical_formula \
    --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
    --checkpoint_pth="./MassSpecGymDeNovo/6IjntsDfWm/step=000600-val_loss=0.350.ckpt"
    # smiles_bpe
    # ./MassSpecGymDeNovo/NEA3EkbCos/step=006460-val_loss=0.295.ckpt
    # selfies
    # ./MassSpecGymDeNovo/6IjntsDfWm/step=000600-val_loss=0.350.ckpt



# Train DeepSets on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_deepsets_ff_train_mass \
#     --task=retrieval \
#     --model=deepsets \
#     --log_only_loss_at_stages="train,val" \
#     --val_check_interval=0.5 \
#     --batch_size=16 \
#     --lr=0.0003 \
#     --hidden_channels=128 \
#     --num_layers_per_mlp=4 \
#     --dropout=0.1 \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json"

# Train DeepSets on retrieval (bonus formula challenge)
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_deepsets_ff_train_formula \
#     --task=retrieval \
#     --model=deepsets \
#     --log_only_loss_at_stages="train,val" \
#     --val_check_interval=0.5 \
#     --batch_size=128 \
#     --lr=0.001 \
#     --hidden_channels=128 \
#     --num_layers_per_mlp=4 \
#     --dropout=0.1 \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json"

# Test DeepSets from checkpoint on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_deepsets_test_formula \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=deepsets \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json" \
#     --checkpoint_pth="./MassSpecGymRetrieval/eknSFeAdyN/step=000380-val_loss=0.591.ckpt"
#     # mass
#     # --checkpoint_pth="./MassSpecGymRetrieval/iVJ15YLWQN/step=006067-val_loss=0.589.ckpt"
#     # formula
#     # --checkpoint_pth="./MassSpecGymRetrieval/eknSFeAdyN/step=000380-val_loss=0.591.ckpt"

# Test DeepSets with Fourier features from checkpoint on retrieval
srun --export=ALL --preserve-env python3 run.py \
    --job_key="${job_key}" \
    --run_name=rebuttal_deepsets_ff_test_mass \
    --devices=1 \
    --test_only \
    --task=retrieval \
    --model=deepsets \
    --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
    --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json" \
    --checkpoint_pth="./MassSpecGymRetrieval/y3QXN6wDDk/step=003033-val_loss=0.570.ckpt"
    # mass
    # --checkpoint_pth="./MassSpecGymRetrieval/y3QXN6wDDk/step=003033-val_loss=0.570.ckpt"
    # formula
    # --checkpoint_pth="./MassSpecGymRetrieval/gO6htxckUn/step=000380-val_loss=0.566.ckpt"

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

# Train FingerprintFFN on retrieval using MIST CANOPUS dataset
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_sigmoid_mist_canopus_1550_train_mass \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --log_only_loss_at_stages=train,val \
#     --val_check_interval=0.5 \
#     --batch_size=16 \
#     --lr=0.001 \
#     --hidden_channels=128 \
#     --num_layers=7 \
#     --dropout=0.1 \
#     --dataset_pth=../data/MIST_CANOPUS_with_MassSpecGym_test.tsv \
#     --max_mz=1550 \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json"

# Train FingerprintFFN on retrieval using MIST CANOPUS dataset (bonus formula challenge)
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_sigmoid_mist_canopus_1550_train_formula \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --log_only_loss_at_stages=train,val \
#     --val_check_interval=0.5 \
#     --batch_size=64 \
#     --lr=0.001 \
#     --hidden_channels=128 \
#     --num_layers=7 \
#     --dropout=0.1 \
#     --dataset_pth=../data/MIST_CANOPUS_with_MassSpecGym_test.tsv \
#     --max_mz=1550 \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json"

# Test FingerprintFFN from checkpoint on retrieval
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_test_mass \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --dataset_pth="../data/MassSpecGym_with_test/MassSpecGym_with_test.tsv" \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_mass_with_test.json" \
#     --checkpoint_pth="./MassSpecGymRetrieval/r3JTQeI884/step=004550-val_loss=0.589.ckpt"
#     # mass
#     # --checkpoint_pth="./MassSpecGymRetrieval/r3JTQeI884/step=004550-val_loss=0.589.ckpt"
#     # formula
#     # --checkpoint_pth="./MassSpecGymRetrieval/4pb6nyf8nT/step=000950-val_loss=0.587.ckpt"

# Test FingerprintFFN on retrieval trained on MIST CANOPUS dataset
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${job_key}" \
#     --run_name=rebuttal_fingerprint_ffn_sigmoid_mist_canopus_1550_test_formula \
#     --devices=1 \
#     --test_only \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --dataset_pth=../data/MIST_CANOPUS_with_MassSpecGym_test.tsv \
#     --max_mz=1550 \
#     --candidates_pth="../data/MassSpecGym_with_test/MassSpecGym_retrieval_candidates_formula_with_test.json" \
#     --checkpoint_pth="./MassSpecGymRetrieval/uW1HZbiR1j/step=000600-val_loss=0.510.ckpt"
#     # mass
#     # ./MassSpecGymRetrieval/rJ5HW79ImU/step=001152-val_loss=0.502.ckpt
#     # formula
#     # ./MassSpecGymRetrieval/uW1HZbiR1j/step=000600-val_loss=0.510.ckpt

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
