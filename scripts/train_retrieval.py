import argparse 

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.transforms import MolFingerprinter, SpecBinner
from massspecgym.models.retrieval import DeepSetsRetrieval, RandomRetrieval, FingerprintFFNRetrieval


parser = argparse.ArgumentParser()

# Experiment setup
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--project_name', type=str, default='MassSpecGymRetrieval')
parser.add_argument('--wandb_entity_name', type=str, default='mass-spec-ml')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')

# Training setup
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--accelerator', type=str, default='gpu')
parser.add_argument('--devices', type=int, default=8)
parser.add_argument('--log_every_n_steps', type=int, default=50)

# General hyperparameters
parser.add_argument('--batch_size', type=int, default=64)

# Retrieval setup
parser.add_argument('--fp_size', type=int, default=4096)


def main(args):

    pl.seed_everything(args.seed)

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
    dataset = RetrievalDataset(
        pth=mgf_pth,
        spec_transform=SpecBinner(),
        mol_transform=MolFingerprinter(fp_size=args.fp_size),
        candidates_pth=candidates_pth,
    )

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=split_pth,
        batch_size=args.batch_size
    )

    # Init model
    model = FingerprintFFNRetrieval(
        in_channels=1000,
        out_channels=args.fp_size
    )

    # Init logger
    # You may need to run wandb init first to use the wandb logger
    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            name=args.run_name,
            project=args.project_name,
            log_model=False,
        )

    # Init trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps
    )

    # Validate before training
    data_module.prepare_data()  # Explicit call needed for validate before fit
    data_module.setup()  # Explicit call needed for validate before fit
    trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
