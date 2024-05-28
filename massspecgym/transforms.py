import numpy as np
import matchms
import matchms.filtering as ms_filters
import massspecgym.utils as utils
from rdkit.Chem import AllChem as Chem

from typing import Optional
from abc import ABC, abstractmethod
import torch as th
from torch_geometric.data import Batch

from massspecgym.simulation_utils.feat_utils import MolGraphFeaturizer, get_fingerprints
from massspecgym.simulation_utils.misc_utils import scatter_reduce


class SpecTransform(ABC):
    """
    Base class for spectrum transformations. Custom transformatios should inherit from this class.
    The transformation consists of two consecutive steps:
        1. Apply a series of matchms filters to the input spectrum (method `matchms_transforms`).
        2. Convert the matchms spectrum to a numpy array (method `matchms_to_numeric`).
    """

    @abstractmethod
    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        """
        Apply a series of matchms filters to the input spectrum. Abstract method.
        """

    @abstractmethod
    def matchms_to_numeric(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Convert a matchms spectrum to a numpy array. Abstract method.
        """

    def __call__(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Compose the matchms filters and the numpy conversion.
        """
        return self.matchms_to_numeric(self.matchms_transforms(spec))


def default_matchms_transforms(
    spec: matchms.Spectrum,
    n_max_peaks: int = 60,
    mz_from: float = 10,
    mz_to: float = 1000,
) -> matchms.Spectrum:
    spec = ms_filters.select_by_mz(spec, mz_from=mz_from, mz_to=mz_to)
    spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=n_max_peaks)
    spec = ms_filters.normalize_intensities(spec)
    return spec


class SpecTokenizer(SpecTransform):
    def __init__(
        self,
        n_peaks: Optional[int] = 60,
    ) -> None:
        self.n_peaks = n_peaks

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        return default_matchms_transforms(spec, n_max_peaks=self.n_peaks)

    def matchms_to_numeric(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2).
        If the number of peaks is less than `n_peaks`, pad the matrix with zeros.
        """
        spec = np.vstack([spec.peaks.mz, spec.peaks.intensities]).T
        if self.n_peaks is not None:
            spec = utils.pad_spectrum(spec, self.n_peaks)
        return spec


class SpecToMzsInts(SpecTransform):

    def __init__(
        self,
        n_peaks: Optional[int] = None,
        mz_from: Optional[float] = 10.,
        mz_to: Optional[float] = 1000.
    ) -> None:
        self.n_peaks = n_peaks
        self.mz_from = mz_from
        self.mz_to = mz_to

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        spec = ms_filters.select_by_mz(spec, mz_from=self.mz_from, mz_to=self.mz_to)
        if self.n_peaks is not None:
            spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=self.n_peaks)
        spec = ms_filters.normalize_intensities(spec)
        return spec

    def matchms_to_numeric(self, spec: matchms.Spectrum) -> dict:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2).
        If the number of peaks is less than `n_peaks`, pad the matrix with zeros.
        """
        mzs = th.as_tensor(spec.peaks.mz)
        ints = th.as_tensor(spec.peaks.intensities)
        return {"spec_mzs": mzs, "spec_ints": ints}
    
    def collate_fn(self, collate_data_d: dict) -> None:
        # mutates dict!

        collate_data_d["spec_mzs"] = th.cat(collate_data_d["spec_mzs"],dim=0)
        collate_data_d["spec_ints"] = th.cat(collate_data_d["spec_ints"],dim=0)
        

class MolTransform(ABC):
    @abstractmethod
    def from_smiles(self, mol: str):
        """
        Convert a SMILES string to a tensor-like representation. Abstract method.
        """

    def __call__(self, mol: str):
        return self.from_smiles(mol)


class MolFingerprinter(MolTransform):
    def __init__(self, type: str = "morgan", fp_size: int = 2048, radius: int = 2):
        if type != "morgan":
            raise NotImplementedError(
                "Only Morgan fingerprints are implemented at the moment."
            )
        self.type = type
        self.fp_size = fp_size
        self.radius = radius

    def from_smiles(self, mol: str):
        mol = Chem.MolFromSmiles(mol)
        return utils.morgan_fp(
            mol, fp_size=self.fp_size, radius=self.radius, to_np=True
        )


class MolToInChIKey(MolTransform):
    def __init__(self, twod: bool = True) -> None:
        self.twod = twod

    def from_smiles(self, mol: str) -> str:
        mol = Chem.MolFromSmiles(mol)
        return utils.mol_to_inchi_key(mol, twod=self.twod)


class MolToPyG(MolTransform):

    def __init__(
        self, 
        pyg_node_feats: list[str] = [
            "a_onehot",
            "a_degree",
            "a_hybrid",
            "a_formal",
            "a_radical",
            "a_ring",
            "a_mass",
            "a_chiral"
        ],
        pyg_edge_feats: list[str] = [
            "b_degree",
        ],
        pyg_pe_embed_k: int = 0,
        pyg_bigraph: bool = True):

        self.pyg_node_feats = pyg_node_feats
        self.pyg_edge_feats = pyg_edge_feats
        self.pyg_pe_embed_k = pyg_pe_embed_k
        self.pyg_bigraph = pyg_bigraph

    def from_smiles(self, mol: str):
        
        mol = Chem.MolFromSmiles(mol)
        mg_featurizer = MolGraphFeaturizer(
            self.pyg_node_feats,
            self.pyg_edge_feats,
            self.pyg_pe_embed_k
        )
        mol_pyg = mg_featurizer.get_pyg_graph(mol,bigraph=self.pyg_bigraph)
        return {"mol": mol_pyg}
    
    def get_pyg_sizes(self):

        pass

    def collate_fn(self, collate_data_d: dict) -> None:
        # mutates dict!

        collate_data_d["mol"] = Batch.from_data_list(collate_data_d["mol"])


class MolToFingerprints(MolTransform):

    def __init__(
        self,
        fp_types: list[str] = [
            "morgan",
            "maccs",
            "rdkit"]):

        self.fp_types = sorted(fp_types)

    def from_smiles(self, mol: str) -> th.Tensor:
        
        fps = []
        mol = Chem.MolFromSmiles(mol)
        fps = get_fingerprints(mol, self.fp_types)
        fps = th.as_tensor(fps)
        return {"fps": fps}
    
    def get_fp_size(self) -> int:

        mol = Chem.MolFromSmiles("CCO")
        fp = self.from_smiles(mol)
        return fp.shape[0] 
    
    def collate_fn(self, collate_data_d: dict) -> None:
        # mutates dict!

        collate_data_d["fps"] = th.stack(collate_data_d["fps"],dim=0)


class MetaTransform(ABC):

    @abstractmethod
    def from_meta(self, metadata: dict) -> dict:
        """
        Convert metadata to a dict of numerical representations. Abstract method.
        """

    def __call__(self, metadata: dict):
        return self.from_meta(metadata)


class StandardMeta(MetaTransform):
    
    def __init__(self, adducts: list[str], instruments: list[str], max_ce: float):
        self.adduct_to_idx = {adduct: idx for idx, adduct in enumerate(sorted(adducts))}
        self.inst_to_idx = {inst: idx for idx, inst in enumerate(sorted(instruments))}
        self.num_adducts = len(self.adduct_to_idx)
        self.num_insts = len(self.inst_to_idx)
        self.max_ce = max_ce

    def transform_ce(self, ce):

        ce = np.clip(ce, a_min=0, a_max=int(self.max_ce)-1)
        ce_idx = int(np.around(ce, decimals=0))
        return ce_idx

    def from_meta(self, metadata: dict):
        prec_mz = metadata["precursor_mz"]
        adduct_idx = self.adduct_to_idx.get(metadata["adduct"],self.num_adducts)
        inst_idx = self.inst_to_idx.get(metadata["instrument_type"],self.num_insts)
        ce_idx = self.transform_ce(metadata["collision_energy"])
        meta_d = {
            "precursor_mz": th.tensor(prec_mz),
            "adduct": th.tensor(adduct_idx),
            "instrument_type": th.tensor(inst_idx),
            "collision_energy": th.tensor(ce_idx)
        }
        return meta_d

    def get_meta_sizes(self):

        size_d = {
            "adduct": self.num_adducts+1,
            "instrument_type": self.num_insts+1,
            "collision_energy": int(self.max_ce),
            "precursor_mz": None, # not applicable
        }
        return size_d

    def collate_fn(self, collate_data_d: dict) -> None:
        # mutates dict!

        for key in ["adduct","instrument_type","collision_energy","precursor_mz"]:
            collate_data_d[key] = th.stack(collate_data_d[key],dim=0)


class FragTransform:

    pass
