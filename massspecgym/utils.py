import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import typing as T
import pulp
from myopic_mces.myopic_mces import MCES
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from standardizeUtils.standardizeUtils import (
    standardize_structure_with_pubchem,
    standardize_structure_list_with_pubchem,
)


def pad_spectrum(
    spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    
    Args:
        spec (np.ndarray): Spectrum to pad represented as numpy array of shape (n_peaks, 2).
        max_n_peaks (int): Maximum number of peaks in the padded spectrum.
        pad_value (float, optional): Value to use for padding.
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


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
        to_np (bool, optional): Convert the fingerprint to numpy array.
    """

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp

def tanimoto_morgan_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(morgan_fp(mol1, to_np=False), morgan_fp(mol2, to_np=False))


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


def mol_to_inchi_key(mol: Chem.Mol, twod: bool = True) -> str:
    """
    Convert a molecule to InChI Key representation.
    
    Args:
        mol (Chem.Mol): RDKit molecule object.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    inchi_key = Chem.MolToInchiKey(mol)
    if twod:
        inchi_key = inchi_key.split("-")[0]
    return inchi_key


def smiles_to_inchi_key(mol: str, twod: bool = True) -> str:
    """
    Convert a SMILES molecule to InChI Key representation.
    
    Args:
        mol (str): SMILES string.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    mol = Chem.MolFromSmiles(mol)
    return mol_to_inchi_key(mol, twod)


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
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


def get_smiles_bpe_tokenizer() -> ByteLevelBPETokenizer:
    """
    Return a Byte-level BPE tokenizer trained on the SMILES strings from the
    `MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv` dataset.
    TODO: refactor to a well-organized class.
    """
    # Initialize the tokenizer
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    smiles_tokenizer = ByteLevelBPETokenizer()
    smiles = pd.read_csv(hugging_face_download(
        "molecules/MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv"
    ), sep="\t")["smiles"]
    smiles_tokenizer.train_from_iterator(smiles, special_tokens=special_tokens)

    # Enable padding
    smiles_tokenizer.enable_padding(direction='right', pad_token="<pad>")

    # Add template processing to include start and end of sequence tokens
    smiles_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", smiles_tokenizer.token_to_id("<s>")),
            ("</s>", smiles_tokenizer.token_to_id("</s>")),
        ],
    )
    return smiles_tokenizer


def parse_spec_array(arr: str) -> np.ndarray:
    return list(map(float, arr.split(",")))


def plot_spectrum(spec, hue=None, xlim=None, ylim=None, mirror_spec=None, highl_idx=None,
                  figsize=(6, 2), colors=None, save_pth=None):

    if colors is not None:
        assert len(colors) >= 3
    else:
        colors = ['blue', 'green', 'red']

    # Normalize input spectrum
    def norm_spec(spec):
        assert len(spec.shape) == 2
        if spec.shape[0] != 2:
            spec = spec.T
        mzs, ins = spec[0], spec[1]
        return mzs, ins / max(ins) * 100
    mzs, ins = norm_spec(spec)

    # Initialize plotting
    init_plotting(figsize=figsize)
    fig, ax = plt.subplots(1, 1)

    # Setup color palette
    if hue is not None:
        norm = matplotlib.colors.Normalize(vmin=min(hue), vmax=max(hue), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        plt.colorbar(mapper, ax=ax)

    # Plot spectrum
    for i in range(len(mzs)):
        if hue is not None:
            color = mcolors.to_hex(mapper.to_rgba(hue[i]))
        else:
            color = colors[0]
        plt.plot([mzs[i], mzs[i]], [0, ins[i]], color=color, marker='o', markevery=(1, 2), mfc='white', zorder=2)

    # Plot mirror spectrum
    if mirror_spec is not None:
        mzs_m, ins_m = norm_spec(mirror_spec)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = str(round(-x)) if x < 0 else str(round(x))
            return label

        for i in range(len(mzs_m)):
            plt.plot([mzs_m[i], mzs_m[i]], [0, -ins_m[i]], color=colors[2], marker='o', markevery=(1, 2), mfc='white',
                     zorder=1)
        ax.yaxis.set_major_formatter(major_formatter)

    # Setup axes
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, max(mzs) + 10)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('m/z')
    plt.ylabel('Intensity [%]')

    if save_pth is not None:
        raise NotImplementedError()


class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = False, # makes computations a lot faster
        solver_options: dict = None
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options
        )
        dist = retval[1]
        return dist
