import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs


def pad_spectrum(spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    :param spec: Spectrum to pad represented as numpy array of shape (n_peaks, 2).
    :param max_n_peaks: Maximum number of peaks in the padded spectrum.
    :param pad_value: Value to use for padding.
    """
    n_peaks = spec.shape[0]
    if n_peaks > max_n_peaks:
        raise ValueError(f'Number of peaks in the spectrum ({n_peaks}) is greater than the maximum number of peaks.')                         
    else:
        return np.pad(spec, ((0, max_n_peaks - n_peaks), (0, 0)), mode='constant', constant_values=pad_value)
    

def morgan_fp(mol: Chem.Mol, fp_size=4096, radius=2, to_np=True):
    """
    Compute Morgan fingerprint for a molecule.
    :param fp_size: Size of the fingerprint.
    :param radius: Radius of the fingerprint.
    :param to_np: Convert the fingerprint to numpy array.
    """

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp
