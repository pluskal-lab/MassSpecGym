import pytest


def denovo_baseline_training():
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer

    from massspecgym.data import MassSpecDataset, MassSpecDataModule
    from massspecgym.data.transforms import SpecTokenizer
    from massspecgym.models.de_novo import SmilesTransformer
    from massspecgym.models.tokenizers import SmilesBPETokenizer

    # Load dataset
    dataset = MassSpecDataset(
        spec_transform=SpecTokenizer(n_peaks=60),
        mol_transform=None
    )

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=32
    )

    # Init model
    model = SmilesTransformer(
        input_dim=2,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.0,
        smiles_tokenizer=SmilesBPETokenizer(max_len=200),
        k_predictions=1,
        pre_norm=False,
        max_smiles_len=100,
        validate_only_loss=True
    )

    # Init trainer
    project = "MassSpecGymDeNovo"
    name = "SmilesTransformer_debug_overfitting"
    logger = pl.loggers.WandbLogger(
        project=project,
        name=name,
        tags=[],
        log_model=False,
        mode="disabled"
    )
    trainer = Trainer(
        accelerator="cpu", max_epochs=1, logger=logger, log_every_n_steps=1, check_val_every_n_epoch=50,
        limit_train_batches=3, limit_val_batches=3
    )

    # Train
    trainer.fit(model, datamodule=data_module)


def test_denovo_baseline_training():
    try:
        denovo_baseline_training()
    except Exception as e:
        pytest.fail(f"denovo_baseline_training() raised an exception: {e}")
