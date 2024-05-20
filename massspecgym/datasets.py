import json
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from matchms.importing import load_from_mgf
from pathlib import Path
from typing import Iterable
from massspecgym.transforms import SpecTransform, MolTransform, MolToInChIKey


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is responsible for loading
    the data from disk and applying transformation steps to the spectra and molecules.
    # TODO: "id" is temporary
    """
    def __init__(
            self,
            mgf_pth: Path,
            spec_transform: SpecTransform,
            mol_transform: MolTransform
        ):
        self.mgf_pth = mgf_pth
        self.spectra = list(load_from_mgf(mgf_pth))
        self.spectra_idx = np.array([int(s.get('id')) for s in self.spectra])
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
    
    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, i) -> dict:
        item = {
            'spec': self.spec_transform(self.spectra[i]),
            'mol': self.mol_transform(self.spectra[i].get('smiles'))
        }
        item.update({
            # TODO: collission energy, instrument type
            k: self.spectra[i].metadata[k] for k in ['precursor_mz', 'adduct']
        })
        return item


class RetrievalDataset(MassSpecDataset):
        # Constructur:
    #   - path to candidates json
    #   - candidate_mol_transform: MolTransform = MolToInChIKey()
    # __getitem__:
    #   - return item with candidates
    #   - return mask similar to torchmetrics.retrieval.RetrievalRecall
    #   - custom collate_fn to handle candidates        
    """
    TODO
    """
    def __init__(
            self,
            candidates_pth: Path,
            candidate_mol_transform: MolTransform = MolToInChIKey(),
            **kwargs
        ):
        super().__init__(**kwargs)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(candidates_pth, 'r') as file:
            self.candidates = json.load(file)

        self.candidate_mol_transform = candidate_mol_transform

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i)

        # Read and transform candidate molecules
        item['candidates'] = self.candidates[item['mol']]
        item['candidates'] = [self.candidate_mol_transform(c) for c in item['candidates']]

        # Transform the query molecule
        mol = self.mol_transform(self.spectra[i].get('smiles'))

        # Create neg/pos target mask by matching the query molecule with the candidates
        item['targets'] = [True if c == mol else False for c in item['candidates']]

        # TODO How to collate?

        return item


class MassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    # TODO: "id" is temporary
    """
    def __init__(
            self,
            dataset: MassSpecDataset,
            split_pth: Path,  # TODO: default value
            batch_size: int,
            num_workers: int = 0,
            **kwargs
        ):
        """
        :param mgf_pth: Path to a .mgf file containing mass spectra.
        :param split_pth: Path to a .csv file with columns "id", corresponding to dataset item IDs, and "fold", containg
                          "train", "val", "test" values.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
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
