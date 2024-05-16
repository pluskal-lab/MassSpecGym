import numpy as np
import matchms
import matchms.filtering as ms_filters
import massspecgym.utils as utils
from rdkit.Chem import AllChem as Chem
from typing import Any, Optional


class SpecPreprocessor:
    """
    Base class for spectrum preprocessors. Custom preprocessors should inherit from this class.
    The preprocessing consists of two consecutive steps:
        1. Apply a series of matchms filters to the input spectrum (method `matchms_transforms`).
        2. Convert the matchms spectrum to a numpy array (method `matchms_to_numpy`).
    """
    # def from_matchms(self, spec: matchms.Spectrum) -> np.ndarray:
    #     """
    #     Apply a series of matchms filters to the input spectrum and convert it to a numpy array.
    #     """
    #     raise NotImplementedError('This method must be implemented in a SpecPreprocessor subclass.')

    # def __call__(self, spec) -> Any:
    #     return self.matchms_to_numpy(self.matchms_transforms(spec))

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        """
        Apply a series of matchms filters to the input spectrum. Abstract method.
        """
        raise NotImplementedError('This method must be implemented in a SpecPreprocessor subclass.')

    def matchms_to_numpy(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Convert a matchms spectrum to a numpy array. Abstract method.
        """
        raise NotImplementedError('This method must be implemented in a SpecPreprocessor subclass.')

    def __call__(self, spec: matchms.Spectrum) -> np.ndarray:
        return self.matchms_to_numpy(self.matchms_transforms(spec))


class SpecFilter(SpecPreprocessor):
    def __init__(
        self,
        max_peaks_n: int = 60,
        mz_from: float = 10,
        mz_to: float = 1000
    ) -> None:
        super().__init__()
        self.max_peaks_n = max_peaks_n
        self.mz_from = mz_from
        self.mz_to = mz_to

    def matchms_transforms(
        self,
        spec: matchms.Spectrum, 
    ) -> matchms.Spectrum:
        spec = ms_filters.normalize_intensities(spec)
        spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=self.max_peaks_n)
        spec = ms_filters.select_by_mz(spec, mz_from=self.mz_from, mz_to=self.mz_to)
        return spec

class SpecTokenizer(SpecFilter):
    def __init__(
        self,
        pad_to_n_peaks: Optional[int] = 60,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pad_to_n_peaks = pad_to_n_peaks

    def matchms_to_numpy(self, spec: matchms.Spectrum) -> np.ndarray:
        """
        Stack arrays of mz and intensities into a matrix of shape (num_peaks, 2)
        """
        spec = np.vstack([spec.peaks.mz, spec.peaks.intensities]).T
        if self.pad_to_n_peaks is not None:
            spec = utils.pad_spectrum(spec, self.pad_to_n_peaks)
        return spec


class SpecBinner(SpecFilter):
    # TODO
    pass


class MolPreprocessor:
    def from_smiles(self, mol: str):
        raise NotImplementedError('This method must be implemented in a MolPreprocessor subclass.')

    def __call__(self, mol: str):
        return self.from_smiles(mol)


class MolFingerprinter(MolPreprocessor):
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
