import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from matchms.importing import load_from_mgf
from pathlib import Path
from typing import Iterable
from massspecgym.preprocessors import SpecPreprocessor, MolPreprocessor


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is responsible for loading
    the data from disk and applying preprocessing steps to the spectra and molecules.
    # TODO: "id" is temporary
    """
    def __init__(
            self,
            mgf_pth: Path,
            spec_preproc: SpecPreprocessor,
            mol_preproc: MolPreprocessor
        ):
        self.mgf_pth = mgf_pth
        self.spectra = list(load_from_mgf(mgf_pth))
        self.spectra_idx = np.array([int(s.get('id')) for s in self.spectra])
        self.spec_preproc = spec_preproc
        self.mol_preproc = mol_preproc
    
    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        item = {
            'spec': self.spec_preproc(self.spectra[i]),
            'mol': self.mol_preproc(self.spectra[i].get('smiles'))
        }
        item.update({
            # TODO: collission energy, instrument type
            k: self.spectra[i].metadata[k] for k in ['precursor_mz', 'adduct']
        })
        return item


class MassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    # TODO: "id" is temporary
    """
    def __init__(
            self,
            spec_preproc: SpecPreprocessor,
            mol_preproc: MolPreprocessor,
            batch_size: int,
            mgf_pth: Path,  # TODO: default value
            split_pth: Path,  # TODO: default value
            num_workers: int = 0
        ):
        """
        :param mgf_pth: Path to a .mgf file containing mass spectra.
        :param split_pth: Path to a .csv file with columns "id", corresponding to dataset item IDs, and "fold", containg
                          "train", "val", "test" values.
        """
        self.spec_preproc = spec_preproc
        self.mol_preproc = mol_preproc
        self.batch_size = batch_size
        self.mgf_pth = mgf_pth
        self.split_pth = split_pth
        self.num_workers = num_workers

    def prepare_data(self):

        # Load dataset
        self.dataset = MassSpecDataset(self.mgf_pth, self.spec_preproc, self.mol_preproc)

        # Load split
        self.split = pd.read_csv(self.split_pth)
        if set(self.split.columns) != {'id', 'fold'}:
            raise ValueError('Split file must contain "id" and "fold" columns.')
        self.split = self.split.set_index('id')['fold']
        if set(self.split) != {'train', 'val', 'test'}:
            raise ValueError('"Folds" column must contain only and all of "train", "val", and "test" values.')
        print(self.split)

    def setup(self, stage=None):
        split_mask = np.array([self.split[i] for i in self.dataset.spectra_idx])
        self.train_dataset = Subset(self.dataset, np.where(split_mask == 'train')[0])
        self.val_dataset = Subset(self.dataset, np.where(split_mask == 'val')[0])
        self.test_dataset = Subset(self.dataset, np.where(split_mask == 'test')[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# TODO: Datasets for unlabeled data.
