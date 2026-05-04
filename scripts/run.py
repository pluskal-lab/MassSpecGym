import argparse
import datetime
import typing as T
from pathlib import Path

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import massspecgym.utils as utils
from massspecgym.data import RetrievalDataset, MassSpecDataset, MassSpecDataModule
from massspecgym.data.transforms import (
    MolFingerprinter, SpecBinner, SpecTokenizer, MolToFormulaVector,
    MISTPeakFormulaTokenizer,
)
from massspecgym.models.base import Stage
from massspecgym.models.retrieval import (
    FingerprintFFNRetrieval, FromDictRetrieval, RandomRetrieval, DeepSetsRetrieval,
    MISTFingerprintRetrieval, GenerativeRetrieval, IcebergRetrieval,
)
from massspecgym.models.de_novo import SmilesTransformer, FRIGIDDecoder, MolForgeDecoder, DiffMSDecoder
from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
from massspecgym.models.tokenizers import SmilesBPETokenizer, SelfiesTokenizer
from massspecgym.data.fp2mol_dataset import FP2MolDataset
from massspecgym.definitions import MASSSPECGYM_TEST_RESULTS_DIR


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
parser.add_argument('--test_only', action='store_true')

# Data paths
parser.add_argument('--candidates_pth', type=str, default=None)
parser.add_argument('--dataset_pth', type=str, default=None,
    help='Path to the dataset file in the .tsv or .mgf format.')
parser.add_argument('--split_pth', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=1)

# Data transforms setup

# - Binner
parser.add_argument('--max_mz', type=int, default=1005)
parser.add_argument('--bin_width', type=float, default=1)

# - Tokenizer
parser.add_argument('--n_peaks', type=int, default=60)

# - Fingerprinter
parser.add_argument('--fp_size', type=int, default=2048)

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
parser.add_argument('--log_only_loss_at_stages', default=(),
    type=lambda stages: [Stage(s) for s in stages.strip().replace(' ', '').split(',')])
parser.add_argument('--df_test_pth', type=Path, default=None)
parser.add_argument('--checkpoint_pth', type=Path, default=None)
parser.add_argument('--challenge', type=str, default='auto', choices=['auto', 'mass', 'formula'],
    help='Retrieval/de novo challenge type. Formula-only models require formula.')

# - De novo

# 1. SmilesTransformer
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_encoder_layers', type=int, default=4)
parser.add_argument('--num_decoder_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--k_predictions', type=int, default=1)
parser.add_argument('--pre_norm', type=bool, default=False)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--smiles_tokenizer', choices=['smiles_bpe', 'selfies'], default='selfies')
parser.add_argument('--use_chemical_formula', action='store_true')

# - Retrieval

# 1. FingerprintFFN
parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
# parser.add_argument('--dropout', type=float, default=0.0)

# 2. DeepSets
# parser.add_argument('--hidden_channels', type=int, default=512)
parser.add_argument('--num_layers_per_mlp', type=int, default=2)
# parser.add_argument('--dropout', type=float, default=0.0)

# 3. FromDict (for evaluating given fingerprints)
parser.add_argument('--dct_path', type=str, default=None)

# - FP2Mol decoders (FRIGID, MolForge, DiffMS)
parser.add_argument('--training_mode', type=str, default='spec2mol',
    choices=['spec2mol', 'fp2mol_pretrain'],
    help='Training mode for FP2Mol models: spec2mol (with encoder) or fp2mol_pretrain (decoder only)')
parser.add_argument('--molecule_library', type=str, default=None,
    help='Path to molecule library (SMILES file) for fp2mol_pretrain mode')
parser.add_argument('--exclude_inchikeys', type=str, default=None,
    help='Path to InChIKey exclusion list for data safety')
parser.add_argument('--encoder_checkpoint', type=str, default=None,
    help='Path to pretrained MIST encoder checkpoint')
parser.add_argument('--decoder_checkpoint', type=str, default=None,
    help='Path to pretrained decoder checkpoint')
parser.add_argument('--gen_checkpoint', type=str, default=None,
    help='Path to ICEBERG fragment-generation checkpoint')
parser.add_argument('--inten_checkpoint', type=str, default=None,
    help='Path to ICEBERG intensity checkpoint')
parser.add_argument('--subformulae_dir', type=str, default=None,
    help='Path to precomputed MIST subformulae JSON directory')
parser.add_argument('--decoder_type', type=str, default='frigid',
    choices=['frigid', 'molforge', 'diffms'],
    help='FP2Mol decoder type for generative retrieval')
parser.add_argument('--num_generation_samples', type=int, default=10,
    help='Number of molecules to generate per spectrum')
parser.add_argument('--mol_repr', type=str, default='smiles',
    choices=['smiles', 'selfies', 'safe'],
    help='Molecular representation for FP2Mol training data')


FORMULA_ONLY_MODELS = {
    ('retrieval', 'mist_fingerprint'),
    ('retrieval', 'generative_retrieval'),
    ('de_novo', 'frigid'),
    ('de_novo', 'molforge'),
    ('de_novo', 'diffms'),
}


def _infer_challenge(args) -> str:
    if args.challenge != 'auto':
        return args.challenge
    if args.candidates_pth == 'bonus':
        return 'formula'
    if args.candidates_pth and 'formula' in str(args.candidates_pth).lower():
        return 'formula'
    return 'mass'


def _validate_challenge(args) -> None:
    if args.task == 'de_novo' and args.training_mode == 'fp2mol_pretrain':
        return
    challenge = _infer_challenge(args)
    if (args.task, args.model) in FORMULA_ONLY_MODELS and challenge != 'formula':
        raise ValueError(
            f"Model {args.model!r} requires the formula-based challenge because it "
            "uses precursor formula/subformula features."
        )


def _build_mist_encoder(output_size: int) -> SpectraEncoderGrowing:
    return SpectraEncoderGrowing(
        form_embedder="pos-cos",
        output_size=output_size,
        hidden_size=256,
        peak_attn_layers=4,
        num_heads=8,
        refine_layers=4,
        set_pooling="cls",
        pairwise_featurization=True,
    )


def main(args):
    # Seed everything
    pl.seed_everything(args.seed)
    _validate_challenge(args)

    # Get current time
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Process args
    if args.df_test_pth is None and args.devices == 1:
        args.df_test_pth = MASSSPECGYM_TEST_RESULTS_DIR / f"{args.task}/{args.run_name}_{now_formatted}.pkl"

    # Init paths to data files
    if args.debug:
        args.dataset_pth = "../data/debug/example_5_spectra.mgf"
        args.candidates_pth = "../data/debug/example_5_spectra_candidates.json"
        args.split_pth="../data/debug/example_5_spectra_split.tsv"

    # Load dataset
    if args.task == 'retrieval':
        if args.model == 'fingerprint_ffn':
            spec_transform = SpecBinner(max_mz=args.max_mz, bin_width=args.bin_width)
        elif args.model in {'mist_fingerprint', 'generative_retrieval'}:
            spec_transform = MISTPeakFormulaTokenizer(
                n_peaks=args.n_peaks,
                subformulae_dir=args.subformulae_dir,
                mz_to=args.max_mz,
            )
        else:
            spec_transform = SpecTokenizer(n_peaks=args.n_peaks, matchms_kwargs=dict(mz_to=args.max_mz))
        dataset = RetrievalDataset(
            pth=args.dataset_pth,
            spec_transform=spec_transform,
            mol_transform=MolFingerprinter(fp_size=args.fp_size),
            candidates_pth=args.candidates_pth,
        )
    elif args.task == 'de_novo':
        if args.training_mode == 'fp2mol_pretrain' and args.molecule_library is not None:
            dataset = FP2MolDataset(
                smiles_source=args.molecule_library,
                mol_repr=args.mol_repr,
                fp_bits=args.fp_size,
                exclude_inchikeys=args.exclude_inchikeys,
            )
        elif args.model in {'frigid', 'molforge', 'diffms'}:
            dataset = MassSpecDataset(
                pth=args.dataset_pth,
                spec_transform=MISTPeakFormulaTokenizer(
                    n_peaks=args.n_peaks,
                    subformulae_dir=args.subformulae_dir,
                    mz_to=args.max_mz,
                ),
                mol_transform=None,
            )
        else:
            dataset = MassSpecDataset(
                pth=args.dataset_pth,
                spec_transform=SpecTokenizer(n_peaks=args.n_peaks, matchms_kwargs=dict(mz_to=args.max_mz)),
                mol_transform={'formula': MolToFormulaVector(), 'mol': None} if args.use_chemical_formula else None
            )
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=args.split_pth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Init model
    common_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_only_loss_at_stages=args.log_only_loss_at_stages,
        df_test_path=args.df_test_pth,
    )
    if args.task == 'retrieval':
        if args.model == 'fingerprint_ffn':
            model = FingerprintFFNRetrieval(
                in_channels=int(args.max_mz * (1 / args.bin_width)),
                hidden_channels=args.hidden_channels,
                out_channels=args.fp_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                **common_kwargs
            )
        elif args.model == 'deepsets':
            model = DeepSetsRetrieval(
                in_channels=2,
                hidden_channels=args.hidden_channels,
                out_channels=args.fp_size,
                num_layers_per_mlp=args.num_layers_per_mlp,
                dropout=args.dropout,
                **common_kwargs
            )
        elif args.model == 'from_dict':
            model = FromDictRetrieval(
                dct_path=args.dct_path,
                **common_kwargs
            )
        elif args.model == 'random':
            model = RandomRetrieval(
                **common_kwargs
            )
        elif args.model == 'mist_fingerprint':
            model = MISTFingerprintRetrieval(
                encoder_checkpoint=args.encoder_checkpoint,
                fp_bits=args.fp_size,
                similarity="tanimoto",
                **common_kwargs
            )
        elif args.model == 'generative_retrieval':
            model = GenerativeRetrieval(
                decoder_type=args.decoder_type,
                decoder_checkpoint=args.decoder_checkpoint,
                encoder_checkpoint=args.encoder_checkpoint,
                fp_bits=args.fp_size,
                **common_kwargs
            )
        elif args.model == 'iceberg_retrieval':
            model = IcebergRetrieval(
                gen_checkpoint=args.gen_checkpoint,
                inten_checkpoint=args.inten_checkpoint,
                **common_kwargs
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    elif args.task == 'de_novo':
        if args.model == 'smiles_transformer':
            if args.smiles_tokenizer == 'smiles_bpe':
                max_smiles_len = 200
                smiles_tokenizer = SmilesBPETokenizer(max_len=max_smiles_len)
            elif args.smiles_tokenizer == 'selfies':
                max_smiles_len = 150
                smiles_tokenizer = SelfiesTokenizer(max_len=max_smiles_len)
            else:
                raise NotImplementedError(f"Tokenizer {args.smiles_tokenizer} not implemented")
            model = SmilesTransformer(
                input_dim=args.input_dim,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dropout=args.dropout,
                smiles_tokenizer=smiles_tokenizer,
                k_predictions=args.k_predictions,
                pre_norm=args.pre_norm,
                max_smiles_len=max_smiles_len,
                chemical_formula=args.use_chemical_formula,
                **common_kwargs
            )
        elif args.model == 'frigid':
            model = FRIGIDDecoder(
                encoder=_build_mist_encoder(args.fp_size) if args.training_mode == 'spec2mol' else None,
                fingerprint_bits=args.fp_size,
                training_mode=args.training_mode,
                encoder_checkpoint=args.encoder_checkpoint,
                num_generation_samples=args.num_generation_samples,
                **common_kwargs
            )
        elif args.model == 'molforge':
            model = MolForgeDecoder(
                encoder=_build_mist_encoder(args.fp_size) if args.training_mode == 'spec2mol' else None,
                fingerprint_bits=args.fp_size,
                training_mode=args.training_mode,
                encoder_checkpoint=args.encoder_checkpoint,
                num_generation_samples=args.num_generation_samples,
                **common_kwargs
            )
        elif args.model == 'diffms':
            model = DiffMSDecoder(
                encoder=_build_mist_encoder(args.fp_size) if args.training_mode == 'spec2mol' else None,
                fingerprint_bits=args.fp_size,
                training_mode=args.training_mode,
                encoder_checkpoint=args.encoder_checkpoint,
                num_generation_samples=args.num_generation_samples,
                **common_kwargs
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # If checkpoint path is provided, load the model from the checkpoint instead
    # and override the parameters not related to the model architecture and training
    # TODO Extend to pass arguments to be overridden as an argument to the script
    # For example: --override_args="df_test_path,lr,hidden_channels"
    if args.checkpoint_pth is not None:
        model = type(model).load_from_checkpoint(
            args.checkpoint_pth,
            log_only_loss_at_stages=args.log_only_loss_at_stages,
            df_test_path=args.df_test_pth
        )

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

    # Prepare data module to validate or test before training
    data_module.prepare_data()
    data_module.setup()

    if not args.test_only:
        # Validate before training
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
