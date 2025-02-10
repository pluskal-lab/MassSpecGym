from torch.utils.data import Subset
import numpy as np
import os
import yaml

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
    # train_mol_ids = metadata.loc[train_idxs]["inchikey"].unique()
    # val_mol_ids = metadata.loc[val_idxs]["inchikey"].unique()
    # test_mol_ids = metadata.loc[test_idxs]["inchikey"].unique()
    # print(">>> Number of Spectra")
    # print(len(train_idxs), len(val_idxs), len(test_idxs))
    # print(">>> Number of Unique Molecules")
    # print(len(train_mol_ids), len(val_mol_ids), len(test_mol_ids))
    # get subsets
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)
    test_ds = Subset(ds, test_idxs)
    return train_ds, val_ds, test_ds