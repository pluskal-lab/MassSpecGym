import typing as T
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from massspecgym.data.datasets import MassSpecDataset


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
