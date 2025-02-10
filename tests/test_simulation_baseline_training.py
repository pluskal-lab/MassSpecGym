import pytest


def simulation_baseline_training():
    import os
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    import os
    import wandb

    from massspecgym.data.datasets import SimulationDataset, RetrievalSimulationDataset
    from massspecgym.data.transforms import SpecToMzsInts, MolToPyG, StandardMeta, MolToFingerprints
    from massspecgym.models.simulation.fp import FPSimulationMassSpecGymModel
    from massspecgym.models.simulation.gnn import GNNSimulationMassSpecGymModel
    from massspecgym.models.simulation.prec_only import PrecOnlySimulationMassSpecGymModel
    from massspecgym.simulation_utils.run_utils import load_config, get_split_ss



    # default settings for simulation runs
    template_fp = "../config/simulation/template.yml"
    # special settings for this demo
    custom_fp = "../config/simulation/demo.yml"

    # W&B logging is disabled
    wandb_mode = "disabled"
    # directory for saving model checkpoints
    checkpoint_dp = "../checkpoint/simulation"

    config_d = load_config(template_fp,custom_fp)

    pl.seed_everything(config_d["seed"], workers=True)

    # set torch multiprocessing strategy
    torch.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

    spec_transform = SpecToMzsInts(
        mz_from=config_d["mz_from"],
        mz_to=config_d["mz_to"],
    )
    if config_d["model_type"] in ["fp", "prec_only"]:
        mol_transform = MolToFingerprints(
            fp_types=config_d["fp_types"]
        )
    elif config_d["model_type"] == "gnn":
        mol_transform = MolToPyG()
    else:
        raise ValueError(f"model_type {config_d['model_type']} not supported")
    meta_transform = StandardMeta(
        adducts=config_d["adducts"],
        instrument_types=config_d["instrument_types"],
        max_collision_energy=config_d["max_collision_energy"]
    )

    # wandb
    wandb.init(
        project=config_d["wandb_project"],
        entity=config_d["wandb_entity"],
        name=config_d["wandb_name"],
        mode=wandb_mode,
        dir=checkpoint_dp,
    )
    logger = pl.loggers.WandbLogger(
        entity=config_d["wandb_entity"],
        project=config_d["wandb_project"],
        name=config_d["wandb_name"],
        mode=wandb_mode,
        tags=[],
        log_model=False,
    )

    # set up df_test_path
    save_df_test = config_d.pop("save_df_test")
    if save_df_test:
        df_test_path = os.path.join(wandb.run.dir, "df_test.pkl")
    else:
        df_test_path = None
    config_d["df_test_path"] = df_test_path

    if config_d["model_type"] == "fp":
        pl_model = FPSimulationMassSpecGymModel(**config_d)
    elif config_d["model_type"] == "prec_only":
        pl_model = PrecOnlySimulationMassSpecGymModel(**config_d)
    elif config_d["model_type"] == "gnn":
        pl_model = GNNSimulationMassSpecGymModel(**config_d)
    else:
        raise ValueError(f"model_type {config_d['model_type']} not supported")
    # print(pl_model)

    ds = SimulationDataset(
        pth=config_d["pth"],
        meta_keys=config_d["meta_keys"],
        spec_transform=spec_transform,
        mol_transform=mol_transform,
        meta_transform=meta_transform
    )

    train_ds, val_ds, test_ds = get_split_ss(
        ds,
        config_d["split_type"],
        subsample_frac=config_d["subsample_frac"]
    )

    dl_config = {
        "num_workers": config_d["num_workers"],
        "batch_size": config_d["batch_size"],
        "drop_last": config_d["drop_last"],
        "pin_memory": config_d["pin_memory"] and config_d["accelerator"] != "cpu",
        "persistent_workers": config_d["persistent_workers"] and config_d["accelerator"] != "cpu",
        "collate_fn": ds.collate_fn
    }

    train_dl = DataLoader(train_ds, shuffle=True, **dl_config)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_config)
    test_dl = DataLoader(test_ds, shuffle=False, **dl_config)

    # NOTE: in this demo, we have disabled the retrieval task
    if config_d["do_retrieval"]:
        # we don't need to create separate datasets, can just overwrite...
        ret_ds = RetrievalSimulationDataset(
            pth=config_d["pth"],
            meta_keys=config_d["meta_keys"],
            spec_transform=spec_transform,
            mol_transform=mol_transform,
            meta_transform=meta_transform,
            candidates_pth=config_d["candidates_pth"]
        )
        ret_dl_config = dl_config.copy()
        ret_dl_config["batch_size"] = config_d["retrieval_batch_size"]
        ret_dl_config["collate_fn"] = ret_ds.collate_fn
        _, _, test_ret_ds = get_split_ss(
            ret_ds,
            config_d["split_type"],
            subsample_frac=config_d["subsample_frac"]
        )
        test_dl = DataLoader(test_ret_ds, shuffle=False, **ret_dl_config)


    # Init trainer
    trainer = pl.Trainer(
        accelerator=config_d["accelerator"], 
        max_epochs=config_d["max_epochs"], 
        logger=logger, 
        log_every_n_steps=config_d["log_every_n_steps"],
        gradient_clip_val=config_d["gradient_clip_val"],
        gradient_clip_algorithm=config_d["gradient_clip_algorithm"]
    )

    # Train
    trainer.fit(
        pl_model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl
    )

    # Test
    trainer.test(
        pl_model,
        dataloaders=test_dl,
        ckpt_path="best"
    )


def test_simulation_baseline_training():
    try:
        simulation_baseline_training()
    except Exception as e:
        pytest.fail(f"simulation_baseline_training() raised an exception: {e}")
