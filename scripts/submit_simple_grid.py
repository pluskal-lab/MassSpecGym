import os
import time
import itertools


def write_train_sh(p):
    with open('./train.sh', 'w') as f:
        f.write(f"""#!/bin/bash

echo "job_key \"${{job_key}}\""

# Prepare project environment
. "${{WORK}}/miniconda3/etc/profile.d/conda.sh"
conda activate massspecgym

# Move to running dir
cd "${{SLURM_SUBMIT_DIR}}" || exit 1

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
                
# SmilesTransformer de novo
srun --export=ALL --preserve-env python3 run.py \
    --job_key="${{job_key}}" \
    --run_name="k_fixed_lr={p['lr']},bs={p['batch_size']},k={p['k_predictions']},d={p['d_model']},nhead={p['nhead']},nel={p['num_encoder_layers']}" \
    --task=de_novo \
    --model=smiles_transformer \
    --validate_only_loss \
    --batch_size={p['batch_size']} \
    --lr={p['lr']} \
    --k_predictions={p['k_predictions']} \
    --d_model={p['d_model']} \
    --nhead={p['nhead']} \
    --num_encoder_layers={p['num_encoder_layers']}

""")

# # FingerPrintFFN retrieval                
# srun --export=ALL --preserve-env python3 run.py \
#     --job_key="${{job_key}}" \
#     --run_name="v2_formula_lr={p['lr']},bs={p['batch_size']},hc={p['hidden_channels']},l={p['num_layers']},dropout={p['dropout']}" \
#     --task=retrieval \
#     --model=fingerprint_ffn \
#     --validate_only_loss \
#     --val_check_interval=0.5 \
#     --batch_size={p['batch_size']} \
#     --lr={p['lr']} \
#     --hidden_channels={p['hidden_channels']} \
#     --num_layers={p['num_layers']} \
#     --dropout={p['dropout']} \
#     --candidates_pth="molecules/MassSpecGym_retrieval_candidates_formula.json"




def submit_job():
    os.system("./submit.sh train.sh 20:00 8")


def main():

    # FingerprintFFN
    # grid = {
    #     'lr': [3e-4, 1e-3],
    #     'batch_size': [16, 64, 128],  # per GPU
    #     'hidden_channels': [128, 1024],
    #     'num_layers': [3, 7],
    #     'dropout': [0.0, 0.1],
    #     # 'bin_width': [0.1, 1, 5],
    # }

    # SmilesTransformer
    # grid = {
    #     'lr': [3e-4, 1e-4, 5e-5],
    #     'batch_size': [64, 128],  # per GPU
    #     'k_predictions': [10],
    #     'd_model': [256, 512],
    #     'nhead': [4, 8],
    #     'num_encoder_layers': [3, 6],
    #     # 'temperature': [0.8, 1.0, 1.2],
    # }
    grid = {
        'lr': [3e-4],
        'batch_size': [128],  # per GPU
        'k_predictions': [10],
        'd_model': [256],
        'nhead': [4],
        'num_encoder_layers': [6],
        'temperature': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
    }

    keys, values = zip(*grid.items())
    grid_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f'Submitting {len(grid_dicts)} jobs.')
    for params in grid_dicts:
        print(params)
        write_train_sh(params)
        time.sleep(0.5)
        submit_job()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
