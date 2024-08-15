import pandas as pd
import json
import typing as T
import numpy as np
import torch
import matchms
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from rdkit import Chem
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from copy import deepcopy
import os
from matchms.importing import load_from_mgf

from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey, MetaTransform
from massspecgym.simulation_utils.misc_utils import flatten_lol


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
        pth: Optional[Path] = None,
        return_mol_freq: bool = True,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ):
        """
        Args:
            pth (Optional[Path], optional): Path to the .tsv or .mgf file containing the mass spectra.
                Default is None, in which case the MassSpecGym dataset is downloaded from HuggingFace Hub.
        """
        self.pth = pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.return_mol_freq = return_mol_freq
        self.return_identifier = return_identifier
        self.dtype = dtype
        self.load_data()
        self.compute_mol_freq()

    def load_data(self):

        if self.pth is None:
            self.pth = utils.hugging_face_download("MassSpecGym.tsv")

        if isinstance(self.pth, str):
            self.pth = Path(self.pth)

        if self.pth.suffix == ".tsv":
            self.metadata = pd.read_csv(self.pth, sep="\t")
            self.spectra = self.metadata.apply(
                lambda row: matchms.Spectrum(
                    mz=np.array([float(m) for m in row["mzs"].split(",")]),
                    intensities=np.array(
                        [float(i) for i in row["intensities"].split(",")]
                    ),
                    metadata={"precursor_mz": row["precursor_mz"]},
                ),
                axis=1,
            )
            self.metadata = self.metadata.drop(columns=["mzs", "intensities"])
        elif self.pth.suffix == ".mgf":
            self.spectra = pd.Series(list(load_from_mgf(str(self.pth))))
            self.metadata = pd.DataFrame([s.metadata for s in self.spectra])
        else:
            raise ValueError(f"{self.pth.suffix} file format not supported.")

    def compute_mol_freq(self):

        if self.return_mol_freq:
            if "inchikey" not in self.metadata.columns:
                self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)
            self.metadata["mol_freq"] = self.metadata.groupby("inchikey")["inchikey"].transform("count")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra.iloc[i]
        spec = (
            self.spec_transform(spec)
            if transform_spec and self.spec_transform
            else spec
        )
        spec = torch.as_tensor(spec, dtype=self.dtype)

        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]
        mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol
        if isinstance(mol, np.ndarray):
            mol = torch.as_tensor(mol, dtype=self.dtype)

        item = {"spec": spec, "mol": mol}

        # TODO: Add other metadata to the item. Should it be just done in subclasses?
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

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
        candidates_pth: T.Optional[T.Union[Path, str]] = None,
        **kwargs,
    ):
        self.mol_label_transform = mol_label_transform
        self.candidates_pth = candidates_pth
        super().__init__(**kwargs)

    def load_data(self):

        # Download candidates from HuggigFace Hub if not a path to exisiting file is passed
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_mass.json"
            )
        elif isinstance(self.candidates_pth, str):
            if Path(self.candidates_pth).is_file():
                self.candidates_pth = Path(self.candidates_pth)
            else:
                self.candidates_pth = utils.hugging_face_download(self.candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]

        # Create neg/pos label mask by matching the query molecule with the candidates
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
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
            # item["candidates"] = [torch.as_tensor(c, dtype=self.dtype) for c in item["candidates"]]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        # Standard collate for everything except candidates and their labels (which may have different length per sample)
        collated_batch = {}
        for k in batch[0].keys():
            if k not in ["candidates", "labels", "candidates_smiles"]:
                collated_batch[k] = default_collate([item[k] for item in batch])

        # Collate candidates and labels by concatenating and storing sizes of each list
        collated_batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in batch])
        )
        collated_batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in batch], start=[])
        )
        collated_batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in batch]
        )
        collated_batch["candidates_smiles"] = \
            sum([item["candidates_smiles"] for item in batch], start=[])

        return collated_batch

class SimulationDataset(MassSpecDataset):

    def __init__(
        self,
        spec_transform: SpecTransform,
        mol_transform: MolTransform,
        meta_transform: MetaTransform,
        meta_keys: T.List[str],
        pth: Optional[Path] = None,
        return_mol_freq: bool = True,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ): 
        
        self.meta_transform = meta_transform
        self.meta_keys = meta_keys
        super().__init__(
            spec_transform=spec_transform,
            mol_transform=mol_transform,
            pth=pth,
            return_mol_freq=return_mol_freq,
            return_identifier=return_identifier,
            dtype=dtype
        )

    def load_data(self):

        super().load_data()

        # remove any spectra not included in the simulation challenge
        sim_mask = self.metadata["simulation_challenge"]
        sim_metadata = self.metadata[sim_mask].copy(deep=True)
        # verify all datapoints are not missing CE information and are [M+H]+
        assert (sim_metadata["adduct"]=="[M+H]+").all()
        assert (~sim_metadata["collision_energy"].isna()).all()
        # mz checks
        assert (sim_metadata["precursor_mz"] <= self.spec_transform.mz_to).all()
        # do the filtering
        self.spectra = self.spectra[sim_mask]
        self.metadata = sim_metadata.reset_index(drop=True) 

    def _get_spec_feats(self, i):

        spectrum = self.spectra.iloc[i]
        spec_feats = self.spec_transform(spectrum)
        return spec_feats

    def _get_mol_feats(self, i):

        metadata = self.metadata.iloc[i]
        mol_feats = self.mol_transform(metadata["smiles"])
        return mol_feats

    def _get_meta_feats(self, i):

        metadata = self.metadata.iloc[i]
        meta_feats = self.meta_transform({k: metadata[k] for k in self.meta_keys})
        return meta_feats

    def _get_other_feats(self, i):

        metadata = self.metadata.iloc[i]
        other_feats = {}
        other_feats["smiles"] = metadata["smiles"]
        if self.return_mol_freq:
            other_feats["mol_freq"] = torch.tensor(metadata["mol_freq"])
        if self.return_identifier:
            other_feats["identifier"] = metadata["identifier"]
        return other_feats

    def __getitem__(self, i) -> dict:
        item = {}
        item.update(self._get_spec_feats(i))
        item.update(self._get_mol_feats(i))
        item.update(self._get_meta_feats(i))
        item.update(self._get_other_feats(i))
        return item
    
    def get_collate_data(self, batch_data: dict) -> dict:

        collate_data = {}
        # handle spectrum
        collate_data.update(self.spec_transform.collate_fn(batch_data))
        # handle molecule
        collate_data.update(self.mol_transform.collate_fn(batch_data))
        # handle metadata
        collate_data.update(self.meta_transform.collate_fn(batch_data))
        # handle other stuff
        if "smiles" in batch_data:
            collate_data["smiles"] = batch_data["smiles"].copy()
        if "mol_freq" in batch_data:
            collate_data["mol_freq"] = torch.stack(batch_data["mol_freq"],dim=0)
        if "identifier" in batch_data:
            collate_data["identifier"] = batch_data["identifier"].copy()
        return collate_data

    def collate_fn(self, data_list: T.List[dict]) -> dict:

        keys = list(data_list[0].keys())
        collate_data = {}
        batch_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                batch_data[key].append(data[key])
        collate_data = self.get_collate_data(batch_data)
        return collate_data

        
class RetrievalSimulationDataset(SimulationDataset):


    def __init__(
        self,
        mol_label_transform: MolTransform = MolToInChIKey(),
        candidates_pth: T.Optional[T.Union[Path, str]] = None,

        **kwargs,
    ):
        self.mol_label_transform = mol_label_transform
        self.candidates_pth = candidates_pth
        super().__init__(**kwargs)

    def load_data(self):

        super().load_data()
        # Download candidates from HuggigFace Hub
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_mass.json"
            )
        else: 
            assert isinstance(self.candidates_pth, str)
            if not os.path.isfile(self.candidates_pth):
                self.candidates_pth = utils.hugging_face_download(self.candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

        # check that everything has candidates
        smileses = self.metadata["smiles"]
        candidates_mask = []
        for smiles in smileses:
            candidates_mask.append(smiles in self.candidates)
        candidates_mask = np.array(candidates_mask)
        assert candidates_mask.all()

    def __getitem__(self, i):

        item = super().__getitem__(i)
        smiles = item["smiles"]
        assert isinstance(smiles, str)

        # Get candidates
        if smiles not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {smiles}.')
        candidates_smiles = self.candidates[smiles]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = candidates_smiles

        # Create neg/pos label mask by matching the query molecule with the candidates
        item_label = self.mol_label_transform(smiles)
        candidates_labels = [
            self.mol_label_transform(c) == item_label for c in candidates_smiles
        ]
        if not any(candidates_labels):
            raise ValueError(
                f'Query molecule {smiles} not found in the candidates list.'
            )
        item["candidates_labels"] = torch.tensor(candidates_labels)

        item["candidates_mol_feats"] = [self.mol_transform(c) for c in candidates_smiles]

        # candidates_meta_feats = {}
        # for key in self.meta_keys:
        #     candidates_meta_feats[key] = deepcopy(item[key])
        # item[f"candidates_meta_feats"] = candidates_meta_feats

        # TODO: could put metadata information here...
        # TODO: could put true spectrum duplication in here...

        return item

    def collate_fn(self, data_list: T.List[dict]) -> dict:

        keys = list(data_list[0].keys())
        collate_data = {}
        batch_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                batch_data[key].append(data[key])
        collate_data = super().get_collate_data(batch_data)
        # transform candidates mols
        c_collate_data = {}
        c_mol_feats = flatten_lol(batch_data["candidates_mol_feats"])
        c_mol_keys = list(c_mol_feats[0].keys())
        c_mol_batch_data = {key: [] for key in c_mol_keys}
        for c_mol_feats in c_mol_feats:
            for key in c_mol_keys:
                c_mol_batch_data[key].append(c_mol_feats[key])
        c_mol_collate_data = self.mol_transform.collate_fn(c_mol_batch_data)
        # c_meta_feats = batch_data["candidates_meta_feats"]
        # c_meta_keys = list(c_meta_feats[0].keys())
        # c_meta_batch_data = 
        # package it
        prefix = "" # "candidates_"
        for key in c_mol_keys:
            c_collate_data[prefix+key] = c_mol_collate_data[key]
        c_collate_data[prefix+"smiles"] = flatten_lol(batch_data["candidates_smiles"])
        c_collate_data[prefix+"batch_ptr"] = torch.tensor([len(item) for item in batch_data["candidates_smiles"]])
        c_collate_data[prefix+"labels"] = torch.cat(batch_data["candidates_labels"],dim=0)
        # copy relevant keys
        collate_data["candidates_data"] = c_collate_data
        return collate_data

# TODO: Datasets for unlabeled data.
