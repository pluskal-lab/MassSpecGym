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
        **kwargs
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if self.split_pth is None:
            self.split = self.dataset.metadata[["identifier", "fold"]]
        else:
            # NOTE: custom split is not tested
            self.split = pd.read_csv(self.split_pth, sep="\t")
            if set(self.split.columns) != {"identifier", "fold"}:
                raise ValueError('Split file must contain "id" and "fold" columns.')
            self.split["identifier"] = self.split["identifier"].astype(str)
            if set(self.dataset.metadata["identifier"]) != set(self.split["identifier"]):
                raise ValueError(
                    "Dataset item IDs must match the IDs in the split file."
                )

        self.split = self.split.set_index("identifier")["fold"]
        if not set(self.split) < {"train", "val", "test"}:
            raise ValueError(
                '"Folds" column must contain only "train", "val", or "test" values.'
            )

    def setup(self, stage=None):
        split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
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
