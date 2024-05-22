import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import typing as T
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from huggingface_hub import hf_hub_download
from standardizeUtils.standardizeUtils import (
    standardize_structure_with_pubchem,
    standardize_structure_list_with_pubchem,
)


def pad_spectrum(
    spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    :param spec: Spectrum to pad represented as numpy array of shape (n_peaks, 2).
    :param max_n_peaks: Maximum number of peaks in the padded spectrum.
    :param pad_value: Value to use for padding.
    """
    n_peaks = spec.shape[0]
    if n_peaks > max_n_peaks:
        raise ValueError(
            f"Number of peaks in the spectrum ({n_peaks}) is greater than the maximum number of peaks."
        )
    else:
        return np.pad(
            spec,
            ((0, max_n_peaks - n_peaks), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )


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


def standardize_smiles(smiles: T.Union[str, T.List[str]]) -> T.Union[str, T.List[str]]:
    """
    Standardize SMILES representation of a molecule using PubChem standardization.
    """
    if isinstance(smiles, str):
        return standardize_structure_with_pubchem(smiles, 'smiles')
    elif isinstance(smiles, list):
        return standardize_structure_list_with_pubchem(smiles, 'smiles')
    else:
        raise ValueError("Input should be a SMILES tring or a list of SMILES strings.")


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    :param file_name: Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym_beta",  # TODO: not beta
        filename="data/" + file_name,
        repo_type="dataset",
    )


def init_plotting(figsize=(6, 2), font_scale=0.95, style="whitegrid"):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set_theme(rc={"figure.figsize": figsize})
    # Set default style and font scale
    sns.set_style(style)
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(["#009473", "#D94F70", "#5A5B9F", "#F0C05A", "#7BC4C4", "#FF6F61"])
