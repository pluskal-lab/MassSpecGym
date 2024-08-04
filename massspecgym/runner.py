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

from massspecgym.data.datasets import SimulationDataset
from massspecgym.transforms import SpecToMzsInts, MolToPyG, StandardMeta, MolToFingerprints
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

def get_split_ss(ds, split_type):

    entry_df = ds.entry_df
    assert np.all(entry_df.index == np.arange(entry_df.shape[0]))
    if split_type == "benchmark":
        train_idxs = entry_df[entry_df["fold"]=="train"].index
        val_idxs = entry_df[entry_df["fold"]=="val"].index
        test_idxs = entry_df[entry_df["fold"]=="test"].index
    elif split_type == "all_inchikey":
        mol_ids = entry_df["mol_id"].drop_duplicates()
        train_ids = mol_ids.sample(frac=0.8, replace=False, random_state=42)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False, random_state=69)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = entry_df[entry_df["mol_id"].isin(train_ids)].index
        val_idxs = entry_df[entry_df["mol_id"].isin(val_ids)].index
        test_idxs = entry_df[entry_df["mol_id"].isin(test_ids)].index
    elif split_type == "orbitrap_inchikey":
        mol_ids = entry_df[entry_df["instrument_type"]=="Orbitrap"]["mol_id"].drop_duplicates()
        train_ids = mol_ids.sample(frac=0.8, replace=False, random_state=42)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False, random_state=69)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = entry_df[entry_df["mol_id"].isin(train_ids)].index
        val_idxs = entry_df[entry_df["mol_id"].isin(val_ids)].index
        test_idxs = entry_df[entry_df["mol_id"].isin(test_ids)].index
    else:
        raise ValueError(f"split_type {split_type} not supported")
    train_mol_ids = entry_df.loc[train_idxs]["mol_id"].unique()
    val_mol_ids = entry_df.loc[val_idxs]["mol_id"].unique()
    test_mol_ids = entry_df.loc[test_idxs]["mol_id"].unique()
    print(">>> Number of Spectra")
    print(len(train_idxs), len(val_idxs), len(test_idxs))
    print(">>> Number of Unique Molecules")
    print(len(train_mol_ids), len(val_mol_ids), len(test_mol_ids))
    # get subsets
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)
    test_ds = Subset(ds, test_idxs)
    # compute counts (for weights)
    all_idxs = np.concatenate([train_idxs,val_idxs,test_idxs],axis=0)
    all_idxs = np.sort(all_idxs)
    ds.compute_counts(all_idxs)
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
        tsv_pth=config_d["tsv_pth"],
        meta_keys=config_d["meta_keys"],
        spec_transform=spec_transform,
        mol_transform=mol_transform,
        meta_transform=meta_transform,
        cache_feats=config_d["cache_feats"])

    # # Init data module
    # data_module = MassSpecDataModule(
    #     dataset=ds,
    #     split_pth=split_pth,
    #     batch_size=8
    # )

    train_ds, val_ds, test_ds = get_split_ss(ds,config_d["split_type"])

    dl_config = {
        "num_workers": config_d["num_workers"],
        "batch_size": config_d["batch_size"],
        "drop_last": config_d["drop_last"],
        "collate_fn": ds.collate_fn
    }

    train_dl = DataLoader(train_ds, shuffle=True, **dl_config)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_config)

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
