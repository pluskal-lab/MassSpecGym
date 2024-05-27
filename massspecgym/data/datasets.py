import json
import typing as T
import numpy as np
import torch
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from rdkit import Chem
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from matchms.importing import load_from_mgf
from massspecgym.transforms import SpecTransform, MolTransform, MolToInChIKey


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is 
    responsible for loading the data from disk and applying transformation steps to the spectra and
    molecules.
    """

    def __init__(
        self,
        spec_transform: Optional[SpecTransform] = None,
        mol_transform: Optional[MolTransform] = None,
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
        spec = self.spectra[i]
        spec = self.spec_transform(spec) if self.spec_transform else spec

        mol = self.spectra[i].get("smiles")
        mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol

        item = {"spec": spec, "mol": mol}
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
    """
    Dataset containing mass spectra and their corresponding molecular structures, with additional
    candidates of molecules for retrieval based on spectral similarity.
    """

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


# TODO: Datasets for unlabeled data.
