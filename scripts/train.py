import argparse
from pathlib import Path

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import massspecgym.utils as utils
from massspecgym.data import RetrievalDataset, MassSpecDataset, MassSpecDataModule
from massspecgym.transforms import MolFingerprinter, SpecBinner, SpecTokenizer
from massspecgym.models.retrieval import FingerprintFFNRetrieval
from massspecgym.models.de_novo import SmilesTransformer


# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# TODO Organize configs better (probably with hydra)
parser = argparse.ArgumentParser()

# Submission
parser.add_argument('--job_key', type=str, required=True)

# Experiment setup
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--project_name', type=str, default=None)
parser.add_argument('--wandb_entity_name', type=str, default='mass-spec-ml')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')

# Data transforms setup

# - Binner
parser.add_argument('--max_mz', type=int, default=1005)
parser.add_argument('--bin_width', type=float, default=1)

# - Tokenizer
parser.add_argument('--n_peaks', type=int, default=60)

# - Fingerprinter
parser.add_argument('--fp_size', type=int, default=4096)

# Training setup
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--accelerator', type=str, default='gpu')
parser.add_argument('--devices', type=int, default=8)
parser.add_argument('--log_every_n_steps', type=int, default=50)
parser.add_argument('--val_check_interval', type=float, default=1.0)

# General hyperparameters
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)

# Task and model
parser.add_argument('--task', type=str, choices=['retrieval', 'de_novo', 'simulation'], required=True)
parser.add_argument('--model', type=str, required=True)

# - De novo

parser.add_argument('--validate_only_loss', action='store_true')

# 1. SmilesTransformer
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_encoder_layers', type=int, default=4)
parser.add_argument('--num_decoder_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--smiles_tokenizer', type=str, default=utils.get_smiles_bpe_tokenizer())
parser.add_argument('--k_predictions', type=int, default=1)
parser.add_argument('--pre_norm', type=bool, default=False)
parser.add_argument('--max_smiles_len', type=int, default=100)
parser.add_argument('--temperature', type=float, default=1)

# - Retrieval

# 1. FingerprintFFN
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
# parser.add_argument('--dropout', type=float, default=0.0)


def main(args):

    pl.seed_everything(args.seed)

    # Init paths to data files
    if args.debug:
        mgf_pth = "../data/debug/example_5_spectra.mgf"
        candidates_pth = "../data/debug/example_5_spectra_candidates.json"
        split_pth="../data/debug/example_5_spectra_split.tsv"
    else:
        # Use default benchmark paths
        mgf_pth = None
        candidates_pth = None
        split_pth = None

    # Load dataset
    if args.task == 'retrieval':
        dataset = RetrievalDataset(
            pth=mgf_pth,
            spec_transform=SpecBinner(max_mz=args.max_mz, bin_width=args.bin_width),
            mol_transform=MolFingerprinter(fp_size=args.fp_size),
            candidates_pth=candidates_pth,
        )
    elif args.task == 'de_novo':
        dataset = MassSpecDataset(
            pth=mgf_pth,
            spec_transform=SpecTokenizer(n_peaks=args.n_peaks),
            mol_transform=None
        )
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=split_pth,
        batch_size=args.batch_size
    )

    # Init model
    if args.task == 'retrieval':
        if args.model == 'fingerprint_ffn':
            model = FingerprintFFNRetrieval(
                lr=args.lr,
                weight_decay=args.weight_decay,
                in_channels=int(args.max_mz * (1 / args.bin_width)),
                hidden_channels=args.hidden_channels,
                out_channels=args.fp_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    elif args.task == 'de_novo':
        if args.model == 'smiles_transformer':
            model = SmilesTransformer(
                input_dim=args.input_dim,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dropout=args.dropout,
                smiles_tokenizer=args.smiles_tokenizer,
                k_predictions=args.k_predictions,
                pre_norm=args.pre_norm,
                max_smiles_len=args.max_smiles_len,
                validate_only_loss=args.validate_only_loss
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # Init logger
    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            name=args.run_name,
            project=args.project_name,
            log_model=False,
            config=args
        )

    # Init callbacks for checkpointing and early stopping
    callbacks = []
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        monitor_name = monitor['monitor']
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor_name,
            save_top_k=1,
            mode=monitor['mode'],
            dirpath=Path(args.project_name) / args.job_key,
            filename=f'{{step:06d}}-{{{monitor_name}:03.03f}}',
            auto_insert_metric_name=True,
            save_last=(i == 0)
        )
        callbacks.append(checkpoint)
        if monitor.get('early_stopping', False):
            early_stopping = EarlyStopping(
                monitor=monitor_name,
                mode=monitor['mode'],
                verbose=True
            )
            callbacks.append(early_stopping)

    # Init trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks
    )

    # Validate before training
    data_module.prepare_data()  # Explicit call needed for validate before fit
    data_module.setup()  # Explicit call needed for validate before fit
    trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)

    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.project_name is None:
        task_name = args.task.replace('_', ' ').title().replace(' ', '')
        args.project_name = f"MassSpecGym{task_name}"

    main(args)
