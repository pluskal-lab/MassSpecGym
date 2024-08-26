import numpy as np
import torch
import matchms
import matchms.filtering as ms_filters
from rdkit.Chem import AllChem as Chem
from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Batch

from massspecgym.simulation_utils.feat_utils import MolGraphFeaturizer, get_fingerprints
from massspecgym.simulation_utils.misc_utils import scatter_reduce
import massspecgym.utils as utils
from massspecgym.definitions import CHEM_ELEMS


class SpecTransform(ABC):
    """
    Base class for spectrum transformations. Custom transformatios should inherit from this class.
    The transformation consists of two consecutive steps:
        1. Apply a series of matchms filters to the input spectrum (method `matchms_transforms`).
        2. Convert the matchms spectrum to a torch tensor (method `matchms_to_torch`).
    """

    @abstractmethod
    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        """
        Apply a series of matchms filters to the input spectrum. Abstract method.
        """

    @abstractmethod
    def matchms_to_torch(self, spec: matchms.Spectrum) -> torch.Tensor:
        """
        Convert a matchms spectrum to a torch tensor. Abstract method.
        """

    def __call__(self, spec: matchms.Spectrum) -> torch.Tensor:
        """
        Compose the matchms filters and the torch conversion.
        """
        return self.matchms_to_torch(self.matchms_transforms(spec))


def default_matchms_transforms(
    spec: matchms.Spectrum,
    n_max_peaks: int = 60,
    mz_from: float = 10,
    mz_to: float = 1000,
) -> matchms.Spectrum:
    spec = ms_filters.select_by_mz(spec, mz_from=mz_from, mz_to=mz_to)
    if n_max_peaks is not None:
        spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=n_max_peaks)
    spec = ms_filters.normalize_intensities(spec)
    return spec


class SpecTokenizer(SpecTransform):
    def __init__(
        self,
        n_peaks: Optional[int] = 60,
        prec_mz_intensity: Optional[float] = 1.1,
        matchms_kwargs: Optional[dict] = None
    ) -> None:
        self.n_peaks = n_peaks
        self.prec_mz_intensity = prec_mz_intensity
        self.matchms_kwargs = matchms_kwargs if matchms_kwargs is not None else {}

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        return default_matchms_transforms(spec, n_max_peaks=self.n_peaks, **self.matchms_kwargs)

    def matchms_to_torch(self, spec: matchms.Spectrum) -> torch.Tensor:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2).
        If the number of peaks is less than `n_peaks`, pad the matrix with zeros.
        """
        spec_t = np.vstack([spec.peaks.mz, spec.peaks.intensities]).T
        if self.prec_mz_intensity is not None:
            spec_t = np.vstack([[spec.metadata["precursor_mz"], self.prec_mz_intensity], spec_t])
        if self.n_peaks is not None:
            spec_t = utils.pad_spectrum(
                spec_t,
                self.n_peaks + 1 if self.prec_mz_intensity is not None else self.n_peaks
            )
        return torch.from_numpy(spec_t)


class SpecBinner(SpecTransform):
    def __init__(
        self,
        max_mz: float = 1005,
        bin_width: float = 1,
        to_rel_intensities: bool = True,
    ) -> None:
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.to_rel_intensities = to_rel_intensities
        if not (max_mz / bin_width).is_integer():
            raise ValueError("`max_mz` must be divisible by `bin_width`.")

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        return default_matchms_transforms(spec, mz_to=self.max_mz, n_max_peaks=None)

    def matchms_to_torch(self, spec: matchms.Spectrum) -> torch.Tensor:
        """
        Bin the spectrum into a fixed number of bins.
        """
        binned_spec = self._bin_mass_spectrum(
            mzs=spec.peaks.mz,
            intensities=spec.peaks.intensities,
            max_mz=self.max_mz,
            bin_width=self.bin_width,
            to_rel_intensities=self.to_rel_intensities,
        )
        return torch.from_numpy(binned_spec)

    def _bin_mass_spectrum(
        self, mzs, intensities, max_mz, bin_width, to_rel_intensities=True
    ):
        # Calculate the number of bins
        num_bins = int(np.ceil(max_mz / bin_width))

        # Calculate the bin indices for each mass
        bin_indices = np.floor(mzs / bin_width).astype(int)

        # Filter out mzs that exceed max_mz
        valid_indices = bin_indices[mzs <= max_mz]
        valid_intensities = intensities[mzs <= max_mz]

        # Clip bin indices to ensure they are within the valid range
        valid_indices = np.clip(valid_indices, 0, num_bins - 1)

        # Initialize an array to store the binned intensities
        binned_intensities = np.zeros(num_bins)

        # Use np.add.at to sum intensities in the appropriate bins
        np.add.at(binned_intensities, valid_indices, valid_intensities)

        # Generate the bin edges for reference
        # bin_edges = np.arange(0, max_mz + bin_width, bin_width)

        # Normalize the intensities to relative intensities
        if to_rel_intensities:
            binned_intensities /= np.max(binned_intensities)

        return binned_intensities  # , bin_edges


class SpecToMzsInts(SpecTransform):

    def __init__(
        self,
        n_peaks: Optional[int] = None,
        mz_from: Optional[float] = 10.,
        mz_to: Optional[float] = 1000.,
        mz_bin_res: Optional[float] = 0.01
    ) -> None:
        self.n_peaks = n_peaks
        self.mz_from = mz_from
        self.mz_to = mz_to
        self.mz_bin_res = mz_bin_res

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        # little hack to avoid selecting peaks at mz_to exactly
        spec = ms_filters.select_by_mz(spec, mz_from=self.mz_from, mz_to=self.mz_to)
        if self.n_peaks is not None:
            spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=self.n_peaks)
        spec = ms_filters.normalize_intensities(spec)
        # spec.peaks.intensities = spec.peaks.intensities * 1000.
        return spec

    def matchms_to_torch(self, spec: matchms.Spectrum) -> dict:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2).
        If the number of peaks is less than `n_peaks`, pad the matrix with zeros.
        """
        mzs = torch.as_tensor(spec.peaks.mz, dtype=torch.float32)
        ints = torch.as_tensor(spec.peaks.intensities, dtype=torch.float32)
        return {"spec_mzs": mzs, "spec_ints": ints}
    
    def collate_fn(self, batch_data_d: dict) -> dict:

        device = batch_data_d["spec_mzs"][0].device
        counts = torch.tensor([spec_mzs.shape[0] for spec_mzs in batch_data_d["spec_mzs"]],device=device,dtype=torch.int64)
        batch_idxs = torch.repeat_interleave(torch.arange(counts.shape[0],device=device),counts,dim=0)
        collate_data_d = {}
        collate_data_d["spec_mzs"] = torch.cat(batch_data_d["spec_mzs"],dim=0)
        collate_data_d["spec_ints"] = torch.cat(batch_data_d["spec_ints"],dim=0)
        collate_data_d["spec_batch_idxs"] = batch_idxs
        return collate_data_d

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


class MolToFormulaVector(MolTransform):
    def __init__(self):
        self.element_index = {element: i for i, element in enumerate(CHEM_ELEMS)}

    def from_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Add explicit hydrogens to the molecule
        mol = Chem.AddHs(mol)

        # Initialize a vector of zeros for the 118 elements
        formula_vector = np.zeros(118, dtype=np.int32)

        # Iterate over atoms in the molecule and count occurrences of each element
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in self.element_index:
                index = self.element_index[symbol]
                formula_vector[index] += 1
            else:
                raise ValueError(f"Element '{symbol}' not found in the list of 118 elements.")

        return formula_vector

    @staticmethod
    def num_elements():
        return len(CHEM_ELEMS)


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
        return {"mol_pyg": mol_pyg}
    
    def get_input_sizes(self):

        g = self.from_smiles("CCO")["mol_pyg"]
        size_d = {
            "mol_node_feats_size": g.x.shape[1],
            "mol_edge_feats_size": g.edge_attr.shape[1],
        }
        return size_d

    def collate_fn(self, batch_data_d: dict) -> dict:

        collate_data_d = {}
        collate_data_d["mol_pyg"] = Batch.from_data_list(batch_data_d["mol_pyg"])
        return collate_data_d


class MolToFingerprints(MolTransform):

    def __init__(
        self,
        fp_types: list[str] = [
            "morgan",
            "maccs",
            "rdkit"]):

        self.fp_types = sorted(fp_types)

    def from_smiles(self, mol: str) -> dict:
        
        fps = []
        mol = Chem.MolFromSmiles(mol)
        fps = get_fingerprints(mol, self.fp_types)
        fps = torch.as_tensor(fps, dtype=torch.float32)
        return {"fps": fps}
    
    def get_input_sizes(self) -> dict:

        fps = self.from_smiles("CCO")["fps"]
        return {"fps_input_size": fps.shape[0]} 
    
    def collate_fn(self, batch_data_d: dict, prefix="") -> dict:

        collate_data_d = {}
        collate_data_d["fps"] = torch.stack(batch_data_d["fps"],dim=0)
        return collate_data_d


class MetaTransform(ABC):

    @abstractmethod
    def from_meta(self, metadata: dict) -> dict:
        """
        Convert metadata to a dict of numerical representations. Abstract method.
        """

    def __call__(self, metadata: dict):
        return self.from_meta(metadata)


class StandardMeta(MetaTransform):
    
    def __init__(self, adducts: list[str], instrument_types: list[str], max_collision_energy: float):
        self.adduct_to_idx = {adduct: idx for idx, adduct in enumerate(sorted(adducts))}
        self.inst_to_idx = {inst: idx for idx, inst in enumerate(sorted(instrument_types))}
        self.num_adducts = len(self.adduct_to_idx)
        self.num_instrument_types = len(self.inst_to_idx)
        self.num_collision_energies = int(max_collision_energy)
        self.max_collision_energy = max_collision_energy

    def transform_ce(self, ce):

        ce = np.clip(ce, a_min=0, a_max=int(self.max_collision_energy)-1)
        ce_idx = int(np.around(ce, decimals=0))
        return ce_idx

    def from_meta(self, metadata: dict):
        prec_mz = metadata["precursor_mz"]
        adduct_idx = self.adduct_to_idx.get(metadata["adduct"],self.num_adducts)
        inst_idx = self.inst_to_idx.get(metadata["instrument_type"],self.num_instrument_types)
        ce_idx = self.transform_ce(metadata["collision_energy"])
        meta_d = {
            "precursor_mz": torch.tensor(prec_mz,dtype=torch.float32),
            "adduct": torch.tensor(adduct_idx),
            "instrument_type": torch.tensor(inst_idx),
            "collision_energy": torch.tensor(ce_idx)
        }
        return meta_d

    def get_input_sizes(self):

        size_d = {
            "adduct_input_size": self.num_adducts+1,
            "instrument_type_input_size": self.num_instrument_types+1,
            "collision_energy_input_size": int(self.num_collision_energies)
        }
        return size_d

    @property
    def collate_keys(self):

        return ["adduct","instrument_type","collision_energy","precursor_mz"]

    def collate_fn(self, batch_data_d: dict) -> dict:

        collate_data_d = {}
        for key in self.collate_keys:
            collate_data_d[key] = torch.stack(batch_data_d[key],dim=0)
        return collate_data_d
