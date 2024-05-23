import json
import typing as T
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader, default_collate
from matchms.importing import load_from_mgf
from massspecgym.transforms import SpecTransform, MolTransform, MolToInChIKey


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is responsible for loading
    the data from disk and applying transformation steps to the spectra and molecules.
    """

    def __init__(
        self,
        spec_transform: SpecTransform,
        mol_transform: MolTransform,
        mgf_pth: Optional[Path] = None,
    ):
        """
        Args:
            mgf_pth (Optional[Path], optional): Path to the .mgf file containing the mass spectra. 
                Default is None, in which case the MassSpecGym dataset is used.
        """
        self.mgf_pth = mgf_pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform

        # Download MassSpecGym dataset from HuggigFace Hub
        if self.mgf_pth is None:
            self.mgf_pth = utils.hugging_face_download("MassSpecGym_labeled_data.mgf")

        self.spectra = list(load_from_mgf(self.mgf_pth))
        self.spectra_idx = np.array([s.get("id") for s in self.spectra])

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, i, transform_mol=True) -> dict:
        spec = self.spec_transform(self.spectra[i])
        mol = self.spectra[i].get("smiles")
        item = {
            "spec": spec,
            "mol": self.mol_transform(mol) if transform_mol else mol,
        }

        item.update(
            {
                # TODO: collision energy, instrument type
                k: self.spectra[i].metadata[k]
                for k in ["precursor_mz", "adduct"]
            }
        )

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function to handle the outputs of __getitem__.
        """
        return default_collate(batch)


class RetrievalDataset(MassSpecDataset):

    def __init__(
        self,
        mol_label_transform: MolTransform = MolToInChIKey(),
        candidates_pth: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform

        # Download candidates from HuggigFace Hub
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download("MassSpecGym_labeled_data_candidates.json")

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i, transform_mol=False)

        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')

        # Create neg/pos label mask by matching the query molecule with the candidates
        item["candidates"] = self.candidates[item["mol"]]
        item_label = self.mol_label_transform(item["mol"])
        item["labels"] = [
            self.mol_label_transform(c) == item_label for c in item["candidates"]
        ]

        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        # Transform the query and candidate molecules
        item["mol"] = self.mol_transform(item["mol"])
        item["candidates"] = [self.mol_transform(c) for c in item["candidates"]]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        # Standard collate for everything except candidates and their labels (which may have different length per sample)
        collated_batch = {}
        for k in batch[0].keys():
            if k not in ["candidates", "labels"]:
                collated_batch[k] = default_collate([item[k] for item in batch])

        # Collate candidates and labels by concatenating and storing pointers to the start of each list
        collated_batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in batch])
        )
        collated_batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in batch], start=[])
        )
        collated_batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in batch]
        )

        return collated_batch


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
        **kwargs,
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "id", 
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test" 
                values. Default is None, in which case the MassSpecGym split is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Download MassSpecGym split from HuggigFace Hub
        if self.split_pth is None:
            self.split_pth = utils.hugging_face_download("MassSpecGym_labeled_data_split.tsv")

    def prepare_data(self):
        # Load split
        self.split = pd.read_csv(self.split_pth, sep="\t")
        if set(self.split.columns) != {"id", "fold"}:
            raise ValueError('Split file must contain "id" and "fold" columns.')

        self.split["id"] = self.split["id"].astype(str)
        self.split = self.split.set_index("id")["fold"]

        if set(self.split) != {"train", "val", "test"}:
            raise ValueError(
                '"Folds" column must contain only and all of "train", "val", and "test" values.'
            )
        if set(self.dataset.spectra_idx) != set(self.split.index):
            raise ValueError("Dataset item IDs must match the IDs in the split file.")

    def setup(self, stage=None):
        split_mask = self.split.loc[self.dataset.spectra_idx].values
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(
                self.dataset, np.where(split_mask == "train")[0]
            )
            self.val_dataset = Subset(self.dataset, np.where(split_mask == "val")[0])
        if stage == "test":
            self.test_dataset = Subset(self.dataset, np.where(split_mask == "test")[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )


# TODO: Datasets for unlabeled data.
