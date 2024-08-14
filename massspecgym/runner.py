import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.notebook import tqdm
from pprint import pprint
from torch.utils.data import Subset
import pytorch_lightning as pl
import numpy as np
import os
import yaml

from massspecgym.data.datasets import SimulationDataset, RetrievalSimulationDataset
from massspecgym.data.transforms import SpecToMzsInts, MolToPyG, StandardMeta, MolToFingerprints
from massspecgym.simulation_utils.misc_utils import print_shapes
from massspecgym.models.simulation.fp import FPSimulationMassSpecGymModel
from massspecgym.models.simulation.gnn import GNNSimulationMassSpecGymModel
from massspecgym.models.simulation.prec_only import PrecOnlySimulationMassSpecGymModel
from massspecgym.simulation_utils.misc_utils import deep_update

def load_config(template_fp, custom_fp):

    assert os.path.isfile(template_fp), template_fp
    if custom_fp:
        assert os.path.isfile(custom_fp), custom_fp
    with open(template_fp, "r") as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)
    # overwrite parts of the config
    if custom_fp:
        with open(custom_fp, "r") as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
        assert all([k in config_d for k in custom_d]), set(custom_d.keys()) - set(config_d.keys())
        config_d = deep_update(config_d, custom_d)
    return config_d

def get_split_ss(ds, split_type, subsample_frac=None):

    metadata = ds.metadata
    assert np.all(metadata.index == np.arange(metadata.shape[0]))
    if split_type == "benchmark":
        train_idxs = metadata[metadata["fold"]=="train"].index
        val_idxs = metadata[metadata["fold"]=="val"].index
        test_idxs = metadata[metadata["fold"]=="test"].index
    elif split_type == "all_inchikey":
        mol_ids = metadata["inchikey"].drop_duplicates()
        train_ids = mol_ids.sample(frac=0.8, replace=False, random_state=42)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False, random_state=69)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = metadata[metadata["inchikey"].isin(train_ids)].index
        val_idxs = metadata[metadata["inchikey"].isin(val_ids)].index
        test_idxs = metadata[metadata["inchikey"].isin(test_ids)].index
    elif split_type == "orbitrap_inchikey":
        mol_ids = metadata[metadata["instrument_type"]=="Orbitrap"]["inchikey"].drop_duplicates()
        train_ids = mol_ids.sample(frac=0.8, replace=False, random_state=42)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False, random_state=69)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = metadata[metadata["inchikey"].isin(train_ids)].index
        val_idxs = metadata[metadata["inchikey"].isin(val_ids)].index
        test_idxs = metadata[metadata["inchikey"].isin(test_ids)].index
    else:
        raise ValueError(f"split_type {split_type} not supported")
    if subsample_frac is not None:
        assert isinstance(subsample_frac, float)
        train_idxs = np.random.choice(
            train_idxs, 
            size=int(subsample_frac*len(train_idxs)),
            replace=False
        )
        val_idxs = np.random.choice(
            val_idxs, 
            size=int(subsample_frac*len(val_idxs)),
            replace=False
        )
        test_idxs = np.random.choice(
            test_idxs, 
            size=int(subsample_frac*len(test_idxs)),
            replace=False
        )
    train_mol_ids = metadata.loc[train_idxs]["inchikey"].unique()
    val_mol_ids = metadata.loc[val_idxs]["inchikey"].unique()
    test_mol_ids = metadata.loc[test_idxs]["inchikey"].unique()
    print(">>> Number of Spectra")
    print(len(train_idxs), len(val_idxs), len(test_idxs))
    print(">>> Number of Unique Molecules")
    print(len(train_mol_ids), len(val_mol_ids), len(test_mol_ids))
    # get subsets
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)
    test_ds = Subset(ds, test_idxs)
    return train_ds, val_ds, test_ds

def init_run(template_fp, custom_fp, wandb_mode):    

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

    # # Init data module
    # data_module = MassSpecDataModule(
    #     dataset=ds,
    #     split_pth=split_pth,
    #     batch_size=8
    # )

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
    
    if config_d["do_retrieval"]:
        # TODO: refactor with test_dl later
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

    logger = pl.loggers.WandbLogger(
        entity=config_d["wandb_entity"],
        project=config_d["wandb_project"],
        name=config_d["wandb_name"],
        mode=wandb_mode,
        tags=[],
        log_model=False,
    )
    # logger = None

    # Init trainer
    trainer = pl.Trainer(
        accelerator=config_d["accelerator"], 
        max_epochs=config_d["max_epochs"], 
        logger=logger, 
        log_every_n_steps=config_d["log_every_n_steps"],
        enable_checkpointing=config_d["enable_checkpointing"],
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
        dataloaders=test_dl
    )