import numpy as np
import matchms
import matchms.filtering as ms_filters
import massspecgym.utils as utils
from rdkit.Chem import AllChem as Chem
from typing import Optional
from abc import ABC, abstractmethod


class SpecTransform(ABC):
    """
    Base class for spectrum transformations. Custom transformatios should inherit from this class.
    The transformation consists of two consecutive steps:
        1. Apply a series of matchms filters to the input spectrum (method `matchms_transforms`).
        2. Convert the matchms spectrum to a numpy array (method `matchms_to_numpy`).
    """

    @abstractmethod
    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        """
        Apply a series of matchms filters to the input spectrum. Abstract method.
        """

    @abstractmethod
    def matchms_to_numpy(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Convert a matchms spectrum to a numpy array. Abstract method.
        """

    def __call__(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Compose the matchms filters and the numpy conversion.
        """
        return self.matchms_to_numpy(self.matchms_transforms(spec))


def default_matchms_transforms(
        spec: matchms.Spectrum,
        n_max_peaks: int = 60,
        mz_from: float = 10,
        mz_to: float = 1000
    ) -> matchms.Spectrum:
    spec = ms_filters.normalize_intensities(spec)
    spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=n_max_peaks)
    spec = ms_filters.select_by_mz(spec, mz_from=mz_from, mz_to=mz_to)
    return spec


class SpecTokenizer(SpecTransform):
    def __init__(
        self,
        n_peaks: Optional[int] = 60,
    ) -> None:
        self.n_peaks = n_peaks

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        return default_matchms_transforms(spec, n_max_peaks=self.n_peaks)

    def matchms_to_numpy(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2).
        If the number of peaks is less than `n_peaks`, pad the matrix with zeros.
        """
        spec = np.vstack([spec.peaks.mz, spec.peaks.intensities]).T
        if self.n_peaks is not None:
            spec = utils.pad_spectrum(spec, self.n_peaks)
        return spec


class SpecBinner(SpecTransform):
    # TODO
    pass


class MolTransform(ABC):
    @abstractmethod
    def from_smiles(self, mol: str):
        """
        Convert a SMILES string to a tensor-like representation. Abstract method.
        """

    def __call__(self, mol: str):
        return self.from_smiles(mol)


class MolFingerprinter(MolTransform):
    def __init__(
            self,
            type: str = 'morgan',
            fp_size: int = 2048,
            radius: int = 2
        ):
        if type != 'morgan':
            raise NotImplementedError('Only Morgan fingerprints are implemented at the moment.')
        self.type = type
        self.fp_size = fp_size
        self.radius = radius

    def from_smiles(self, mol: str):
        mol = Chem.MolFromSmiles(mol)
        return utils.morgan_fp(mol, fp_size=self.fp_size, radius=self.radius, to_np=True)
    

class MolToInChIKey(MolTransform):
    def from_smiles(
            self,
            mol: str,
            twod: bool = True
        ) -> str:
        mol = Chem.MolFromSmiles(mol)
        mol = Chem.MolToInchiKey(mol)
        if twod:
            mol = mol.split('-')[0]
        return mol
