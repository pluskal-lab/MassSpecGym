import pytest


def retrieval_baseline_training():
    from pytorch_lightning import Trainer
    from massspecgym.data import RetrievalDataset, MassSpecDataModule
    from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
    from massspecgym.models.retrieval import DeepSetsRetrieval

    # Load dataset
    dataset = RetrievalDataset(
        spec_transform=SpecTokenizer(n_peaks=60),
        mol_transform=MolFingerprinter(),
    )

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=3
    )

    # Init model
    model = DeepSetsRetrieval(
        bootstrap_metrics=True,
        out_channels=2048,
        fourier_features=True
    )

    # Init trainer
    trainer = Trainer(
        accelerator="cpu", max_epochs=1, log_every_n_steps=1,
        limit_train_batches=3, limit_val_batches=3
    )

    trainer.fit(model, datamodule=data_module)


def test_retrieval_baseline_training():
    try:
        retrieval_baseline_training()
    except Exception as e:
        pytest.fail(f"retrieval_baseline_training() raised an exception: {e}")
