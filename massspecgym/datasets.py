import json
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from matchms.importing import load_from_mgf
from huggingface_hub import hf_hub_download
from massspecgym.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.definitions import HUGGING_FACE_REPO

class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is responsible for loading
    the data from disk and applying transformation steps to the spectra and molecules.
    """
    def __init__(
            self,
            spec_transform: SpecTransform,
            mol_transform: MolTransform,
            mgf_pth: Optional[Path] = None
        ):
        """
        :param mgf_pth: Path to the .mgf file containing the mass spectra. Default is None, in which case the
                        MassSpecGym dataset is used.
        """
        self.mgf_pth = mgf_pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform

        # Download MassSpecGym dataset from HuggigFace Hub
        if self.mgf_pth is None:
            self.mgf_pth = hf_hub_download(
                repo_id=HUGGING_FACE_REPO,
                filename='data/MassSpecGym_labeled_data.mgf',
                repo_type='dataset'
            )
        
        self.spectra = list(load_from_mgf(self.mgf_pth))
        self.spectra_idx = np.array([s.get('id') for s in self.spectra])
    
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
    # Constructor:
    #   - path to candidates json
    #   - candidate_mol_transform: MolTransform = MolToInChIKey()
    # __getitem__:
    #   - return item with candidates
    #   - return mask similar to torchmetrics.retrieval.RetrievalRecall
    # custom collate_fn to handle candidates        
    """
    TODO
    """
    def __init__(
            self,
            candidate_mol_transform: MolTransform = MolToInChIKey(),
            candidates_pth: Optional[Path] = None,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.candidate_mol_transform = candidate_mol_transform

        # Download candidates from HuggigFace Hub
        if self.candidates_pth is None:
            self.candidates_pth = hf_hub_download(
                repo_id=HUGGING_FACE_REPO,
                filename='data/MassSpecGym_labeled_data_candidates.json',
                repo_type='dataset'
            )

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(candidates_pth, 'r') as file:
            self.candidates = json.load(file)

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
    """
    def __init__(
            self,
            dataset: MassSpecDataset,
            batch_size: int,
            num_workers: int = 0,
            split_pth: Optional[Path] = None,
            **kwargs
        ):
        """
        :param split_pth: Path to a .tsv file with columns "id", corresponding to dataset item IDs, and "fold", containg
                          "train", "val", "test" values. Default is None, in which case the MassSpecGym split is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Download MassSpecGym split from HuggigFace Hub
        if self.split_pth is None:
            self.split_pth = hf_hub_download(
                repo_id=HUGGING_FACE_REPO,
                filename='data/MassSpecGym_labeled_data_split.tsv',
                repo_type='dataset'
            )

    def prepare_data(self):
        # Load split
        self.split = pd.read_csv(self.split_pth, sep='\t')
        if set(self.split.columns) != {'id', 'fold'}:
            raise ValueError('Split file must contain "id" and "fold" columns.')
        
        self.split['id'] = self.split['id'].astype(str)
        self.split = self.split.set_index('id')['fold']

        if set(self.split) != {'train', 'val', 'test'}:
            raise ValueError('"Folds" column must contain only and all of "train", "val", and "test" values.')
        if set(self.dataset.spectra_idx) != set(self.split.index):
            raise ValueError('Dataset item IDs must match the IDs in the split file.')

    def setup(self, stage=None):
        split_mask = self.split.loc[self.dataset.spectra_idx].values
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.dataset, np.where(split_mask == 'train')[0])
            self.val_dataset = Subset(self.dataset, np.where(split_mask == 'val')[0])
        if stage == 'test':
            self.test_dataset = Subset(self.dataset, np.where(split_mask == 'test')[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False
        )

# TODO: Datasets for unlabeled data.
