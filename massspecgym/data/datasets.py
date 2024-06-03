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
from matchms.importing import load_from_mgf
from massspecgym.transforms import SpecTransform, MolTransform, MolToInChIKey, MetaTransform


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
    ):
        """
        Args:
            mgf_pth (Optional[Path], optional): Path to the .tsv or .mgf file containing the mass spectra.
                Default is None, in which case the MassSpecGym dataset is downloaded from HuggingFace Hub.
        """
        self.pth = pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform

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
            raise NotImplementedError("Reading .mgf files is not yet supported.")
        else:
            raise ValueError(f"{self.pth.suffix} file format not supported.")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        spec = (
            self.spec_transform(spec)
            if transform_spec and self.spec_transform
            else spec
        )

        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]
        mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol

        item = {"spec": spec, "mol": mol}

        # TODO: Add other metadata to the item. Should it be just done in subclasses?
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

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
            self.candidates_pth = utils.hugging_face_download(
                "MassSpecGym_labeled_data_candidates.json"
            )

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


class SimulationDataset(MassSpecDataset):

    def __init__(
        self,
        tsv_pth: Path,
        meta_keys: T.List[str],
        spec_transform: SpecTransform,
        mol_transform: MolTransform,
        meta_transform: MetaTransform,
        cache_feats: bool): 
        
        self.tsv_pth = tsv_pth
        self.meta_keys = meta_keys
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.meta_transform = meta_transform
        self.cache_feats = cache_feats
        self.spec_feats = {}
        self.mol_feats = {}
        self.meta_feats = {}
        self.process()
        self.spec_per_mol = {}

    def process(self):

        entry_df = pd.read_csv(self.tsv_pth, sep="\t")
        # remove any spectra not included in the simulation challenge
        entry_df = entry_df[entry_df["simulation_challenge"]]
        # remove examples in train split the are missing CE information or are not [M+H]+
        entry_df = entry_df[(entry_df["adduct"]=="[M+H]+") & (~entry_df["collision_energy"].isna())] 
        # mz checks
        mz_max = entry_df["mzs"].apply(lambda l: max(float(x) for x in l.split(",")))
        assert (mz_max <= self.spec_transform.mz_to).all()
        assert (entry_df["precursor_mz"] <= self.spec_transform.mz_to).all()
        # convert spectrum and CE to usable formats
        entry_df["spectrum"] = entry_df.apply(lambda row: utils.peaks_to_matchms(row["mzs"], row["intensities"], row["precursor_mz"]), axis=1)
        entry_df["collision_energy"] = entry_df["collision_energy"].apply(utils.ce_str_to_float)
        entry_df = entry_df.drop(columns=["mzs","intensities"])
        # assign id
        entry_df["spec_id"] = np.arange(entry_df.shape[0])
        inchikey_map = {ik:idx for idx, ik in enumerate(sorted(entry_df["inchikey"].unique()))}
        entry_df["mol_id"] = entry_df["inchikey"].map(inchikey_map)
        entry_df = entry_df.reset_index(drop=True)
    
        self.entry_df = entry_df   

    def __len__(self) -> int:

        return self.entry_df.shape[0]

    def _get_spec_feats(self, i):

        entry = self.entry_df.iloc[i]
        spec_id = entry["spec_id"]
        if i in self.spec_feats:
            spec_feats = self.spec_feats[spec_id]
        else:
            spec_feats = self.spec_transform(entry["spectrum"])
            if self.cache_feats:
                self.spec_feats[i] = spec_feats
        return spec_feats

    def _get_mol_feats(self, i):

        entry = self.entry_df.iloc[i]
        mol_id = entry["mol_id"]
        if mol_id in self.mol_feats:
            mol_feats = self.mol_feats[mol_id]
        else:
            mol_feats = self.mol_transform(entry["smiles"])
            if self.cache_feats:
                self.mol_feats[mol_id] = mol_feats
        return mol_feats

    def _get_meta_feats(self, i):

        entry = self.entry_df.iloc[i]
        spec_id = entry["spec_id"]
        if spec_id in self.mol_feats:
            meta_feats = self.meta_feats[spec_id]
        else:
            meta_feats = self.meta_transform({k: entry[k] for k in self.meta_keys})
            if self.cache_feats:
                self.meta_feats[spec_id] = meta_feats
        return meta_feats

    def _get_frag_feats(self, i):

        raise NotImplementedError

    def _get_other_feats(self, i):

        entry = self.entry_df.iloc[i]
        spec_id = entry["spec_id"]
        other_feats = {}
        other_feats["spec_id"] = torch.tensor(spec_id)
        weight = 1./float(self.spec_per_mol.get(spec_id,1.))
        other_feats["weight"] = torch.tensor(weight)
        return other_feats

    def compute_counts(self, index: Optional[np.ndarray]):

        entry_df = self.entry_df.iloc[index]
        spec_per_mol = entry_df[["mol_id","spec_id"]].drop_duplicates().groupby("mol_id").size().reset_index(name="count")
        spec_per_mol = spec_per_mol.merge(self.entry_df[["spec_id","mol_id"]], on="mol_id", how="inner")[["spec_id","count"]]
        self.spec_per_mol = spec_per_mol.set_index("spec_id")["count"].to_dict()

    def __getitem__(self, i) -> dict:
        item = {}
        item.update(self._get_spec_feats(i))
        item.update(self._get_mol_feats(i))
        item.update(self._get_meta_feats(i))
        item.update(self._get_other_feats(i))
        return item
    
    def collate_fn(self, data_list):

        keys = list(data_list[0].keys())
        collate_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                collate_data[key].append(data[key])
        # handle spectrum
        self.spec_transform.collate_fn(collate_data)
        # handle molecule
        self.mol_transform.collate_fn(collate_data)
        # handle metadata
        self.meta_transform.collate_fn(collate_data)
        # handle other stuff
        for key in ["spec_id","weight"]:
            collate_data[key] = torch.stack(collate_data[key],dim=0)
        return collate_data
        

# TODO: Datasets for unlabeled data.
