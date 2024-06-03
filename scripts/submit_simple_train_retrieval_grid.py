import os
import time
import itertools


def write_train_sh(p):
    with open('./train_retrieval.sh', 'w') as f:
        f.write(f"""#!/bin/bash

echo "job_key \"${{job_key}}\""

# Prepare project environment
. "${{WORK}}/miniconda3/etc/profile.d/conda.sh"
conda activate massspecgym

# Move to running dir
cd "${{SLURM_SUBMIT_DIR}}" || exit 1

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --export=ALL --preserve-env python3 train_retrieval.py \
    --job_key="${{job_key}}" \
    --run_name="lr={p['lr']},bs={p['batch_size']},hc={p['hidden_channels']},l={p['num_layers']},dropout={p['dropout']}" \
    --val_check_interval=0.5 \
    --batch_size={p['batch_size']} \
    --lr={p['lr']} \
    --hidden_channels={p['hidden_channels']} \
    --num_layers={p['num_layers']} \
    --dropout={p['dropout']}
""")


def submit_job():
    os.system("./submit.sh train_retrieval.sh 48:00")


def main():

    grid = {
        'lr': [3e-4, 1e-3],
        'batch_size': [16, 64, 128],  # per GPU
        'hidden_channels': [128, 1024],
        'num_layers': [3, 7],
        'dropout': [0.0, 0.1],
        # 'bin_width': [0.1, 1, 5],
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