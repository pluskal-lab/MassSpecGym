import pytorch_lightning as pl
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
    """
    def __init__(
            self,
            mgf_pth: Path,
            spec_preproc: SpecPreprocessor,
            mol_preproc: MolPreprocessor
        ):
        self.mgf_pth = mgf_pth
        self.spectra = list(load_from_mgf(mgf_pth))
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
    """
    def __init__(
            self,
            mgf_pth: Path,
            spec_preproc: SpecPreprocessor,
            mol_preproc: MolPreprocessor,
            split_mask: Iterable[str],
            batch_size: int,
            num_workers: int = 0
        ):
        if set(split_mask) != {'train', 'val', 'test'}:
            raise ValueError('Split mask must contain only and all of "train", "val", and "test" values.')

        self.mgf_pth = mgf_pth
        self.spec_preproc = spec_preproc
        self.mol_preproc = mol_preproc
        self.split_mask = np.array(split_mask)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.dataset = MassSpecDataset(self.mgf_pth, self.spec_preproc, self.mol_preproc)
        if len(self.split_mask) != len(self.dataset):
            raise ValueError('Split mask must have the same length as the dataset.')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.dataset, np.where(self.split_mask == 'train')[0])
            self.val_dataset = Subset(self.dataset, np.where(self.split_mask == 'val')[0])
        elif stage == 'test':
            self.test_dataset = Subset(self.dataset, np.where(self.split_mask == 'test')[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# TODO: Datasets for unlabeled data.
