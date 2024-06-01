import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.notebook import tqdm
from pprint import pprint
from torch.utils.data import Subset
import pytorch_lightning as pl
import numpy as np

from massspecgym.data.datasets import SimulationDataset
from massspecgym.transforms import SpecToMzsInts, MolToPyG, StandardMeta, MolToFingerprints
from massspecgym.simulation_utils.misc_utils import print_shapes
from massspecgym.models.simulation.fp_ffn import FPFFNSimulationMassSpecGymModel

def get_config_d():

    config_d = {}
    # wandb
    config_d["wandb_entity"] = "adamoyoung"
    config_d["wandb_project"] = "MSG"
    config_d["wandb_name"] = "fp_fnn_simulation_debug"
    config_d["wandb_mode"] = "offline" #"online"
    # data
    config_d["tsv_pth"] = "data/MassSpecGym_3.tsv" #"data/MassSpecGym_3.tsv"
    config_d["meta_keys"] = ["adduct","precursor_mz","instrument_type","collision_energy"]
    config_d["fp_types"] = ["morgan","maccs","rdkit"]
    config_d["adducts"] = ["[M+H]+"]
    config_d["instrument_types"] = ["QTOF","QFT","Orbitrap","ITFT"]
    config_d["max_collision_energy"] = 200. # arbitrary
    config_d["mz_from"] = 10.
    config_d["mz_to"] = 1000.
    # input
    config_d["metadata_insert_location"] = "mlp"
    config_d["collision_energy_insert_size"] = 16
    config_d["adduct_insert_size"] = 16
    config_d["instrument_type_insert_size"] = 16
    config_d["ints_transform"] = "log10t3"
    # output
    config_d["mz_max"] = 1000. # same as mz_to
    config_d["mz_bin_res"] = 0.1
    # model
    config_d["model_type"] = "fp_ffn"
    config_d["mlp_hidden_size"] = 1024
    config_d["mlp_dropout"] = 0.1
    config_d["mlp_num_layers"] = 4
    config_d["mlp_use_residuals"] = True
    config_d["ff_prec_mz_offset"] = 5
    config_d["ff_bidirectional"] = True
    config_d["ff_output_map_size"] = 256
    # optimization
    config_d["lr"] = 0.0003
    config_d["weight_decay"] = 1e-7
    config_d["train_sample_weight"] = True
    config_d["eval_sample_weight"] = True
    # other
    config_d["num_workers"] = 8
    config_d["batch_size"] = 128
    config_d["drop_last"] = False
    config_d["accelerator"] = "cpu"
    config_d["log_every_n_steps"] = 1
    config_d["max_epochs"] = 100
    config_d["seed"] = 420
    config_d["enable_checkpointing"] = False
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
        train_ids = mol_ids.sample(frac=0.8, replace=False)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = entry_df[entry_df["mol_id"].isin(train_ids)].index
        val_idxs = entry_df[entry_df["mol_id"].isin(val_ids)].index
        test_idxs = entry_df[entry_df["mol_id"].isin(test_ids)].index
    elif split_type == "orbitrap_inchikey":
        mol_ids = entry_df[entry_df["instrument_type"]=="Orbitrap"]["mol_id"].drop_duplicates()
        train_ids = mol_ids.sample(frac=0.8, replace=False)
        val_ids = mol_ids.drop(train_ids.index).sample(frac=0.5, replace=False)
        test_ids = mol_ids.drop(train_ids.index).drop(val_ids.index)
        train_idxs = entry_df[entry_df["mol_id"].isin(train_ids)].index
        val_idxs = entry_df[entry_df["mol_id"].isin(val_ids)].index
        test_idxs = entry_df[entry_df["mol_id"].isin(test_ids)].index
    else:
        raise ValueError(f"split_type {split_type} not supported")
    print(len(train_idxs), len(val_idxs), len(test_idxs))
    # get subsets
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)
    test_ds = Subset(ds, test_idxs)
    # compute counts (for weights)
    all_idxs = np.concatenate([train_idxs,val_idxs,test_idxs],axis=0)
    all_idxs = np.sort(all_idxs)
    ds.compute_counts(all_idxs)
    return train_ds, val_ds, test_ds

config_d = get_config_d()

spec_transform = SpecToMzsInts(
    mz_from=config_d["mz_from"],
    mz_to=config_d["mz_to"],
)
mol_transform = MolToFingerprints(
    fp_types=config_d["fp_types"]
)
meta_transform = StandardMeta(
    adducts=config_d["adducts"],
    instrument_types=config_d["instrument_types"],
    max_collision_energy=config_d["max_collision_energy"]
)

if config_d["model_type"] == "fp_ffn":
    pl_model = FPFFNSimulationMassSpecGymModel(**config_d)
elif config_d["model_type"] == "prec_only":
    pl_model = PrecOnlySimulationMassSpecGymModel(**config_d)
else:
    raise ValueError(f"model_type {config_d['model_type']} not supported")

ds = SimulationDataset(
    tsv_pth=config_d["tsv_pth"],
    meta_keys=config_d["meta_keys"],
    spec_transform=spec_transform,
    mol_transform=mol_transform,
    meta_transform=meta_transform,
    cache_feats=False)

# # Init data module
# data_module = MassSpecDataModule(
#     dataset=ds,
#     split_pth=split_pth,
#     batch_size=8
# )

train_ds, val_ds, test_ds = get_split_ss(ds,"orbitrap_inchikey")

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
    mode=config_d["wandb_mode"],
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
    enable_checkpointing=config_d["enable_checkpointing"]
)

# Train
trainer.fit(
    pl_model, 
    train_dataloaders=train_dl, 
    val_dataloaders=val_dl
)

