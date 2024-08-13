import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import typing as T
import selfies as sf
import pulp
from pathlib import Path
from myopic_mces.myopic_mces import MCES
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, Draw
from rdkit.Chem.Descriptors import ExactMolWt
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
from standardizeUtils.standardizeUtils import (
    standardize_structure_with_pubchem,
    standardize_structure_list_with_pubchem,
)


def load_massspecgym():
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df['mzs'] = df['mzs'].apply(parse_spec_array)
    df['intensities'] = df['intensities'].apply(parse_spec_array)
    return df


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


def tanimoto_morgan_similarity(mol1: T.Union[Chem.Mol, str], mol2: T.Union[Chem.Mol, str]) -> float:
    """
    Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        mol1 (T.Union[Chem.Mol, str]): First molecule as RDKit molecule or SMILES string.
        mol2 (T.Union[Chem.Mol, str]): Second molecule as RDKit molecule or SMILES string.
    """
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
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


def init_plotting(figsize=(6, 2), font_scale=1.0, style="whitegrid"):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set_theme(rc={"figure.figsize": figsize})
    mpl.rcParams['svg.fonttype'] = 'none'
    # Set default style and font scale
    sns.set_style(style)
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(["#009473", "#D94F70", "#5A5B9F", "#F0C05A", "#7BC4C4", "#FF6F61"])


class SpecialSymbolsBaseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        pad_token: str = "<pad>",
        sos_token: str = "<s>",
        eos_token: str = "</s>",
        max_length: T.Optional[int] = None,
    ):
        """Initialize the base tokenizer with special tokens and optional padding."""
        super().__init__(tokenizer)

        # Save essential attributes
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length

        # Add special tokens
        self.add_special_tokens([pad_token, sos_token, eos_token])

        # Get token IDs
        self.pad_token_id = self.token_to_id(pad_token)
        self.sos_token_id = self.token_to_id(sos_token)
        self.eos_token_id = self.token_to_id(eos_token)

        # Enable padding
        self.enable_padding(
            direction="right",
            pad_token=pad_token,
            pad_id=self.pad_token_id,
            length=max_length,
        )

        # Enable truncation
        self.enable_truncation(max_length)

        # Set post-processing to add SOS and EOS tokens
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{sos_token} $A {eos_token}",
            pair=f"{sos_token} $A {eos_token} {sos_token} $B {eos_token}",
            special_tokens=[
                (sos_token, self.sos_token_id),
                (eos_token, self.eos_token_id),
            ],
        )


class SelfiesTokenizer(SpecialSymbolsBaseTokenizer):
    def __init__(self, **kwargs):
        """Initialize the SELFIES tokenizer with a custom vocabulary."""
        alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab))
        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of SELFIES token IDs."""
        selfies_string = sf.encoder(text)
        selfies_tokens = list(sf.split_selfies(selfies_string))
        return super().encode(
            selfies_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of SELFIES token IDs back into a SMILES string."""
        selfies_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        selfies_string = self._decode_wordlevel_str_to_selfies(
            selfies_string, skip_special_tokens=skip_special_tokens
        )
        return sf.decoder(selfies_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of SELFIES token IDs."""
        selfies_strings = [
            list(sf.split_selfies(sf.encoder(text))) for text in texts
        ]
        return super().encode_batch(
            selfies_strings, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of SELFIES token IDs back into SMILES strings."""
        selfies_strings = super().decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        return [
            sf.decoder(
                self._decode_wordlevel_str_to_selfies(
                    selfies_string, skip_special_tokens=skip_special_tokens
                )
            )
            for selfies_string in selfies_strings
        ]

    def _decode_wordlevel_str_to_selfies(
        self, text: str, skip_special_tokens: bool = True
    ) -> str:
        """Converts a WordLevel string back to a SELFIES string."""
        text = text.replace(" ", "")
        return text


class SmilesBPETokenizer(SpecialSymbolsBaseTokenizer):
    def __init__(self, smiles_pth: T.Optional[str] = None, **kwargs):
        """Initialize the BPE tokenizer for SMILES strings, with optional training data."""
        tokenizer = ByteLevelBPETokenizer()
        if smiles_pth:
            tokenizer.train(smiles_pth)
        else:
            smiles = pd.read_csv(
                hugging_face_download(
                    "molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv"
                ),
                sep="\t",
            )["smiles"]
            print(f"Training tokenizer on {len(smiles)} SMILES strings.")
            tokenizer.train_from_iterator(smiles)

        super().__init__(tokenizer, **kwargs)


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


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


def show_mols(mols, legends='new_indices', smiles_in=False, svg=False, sort_by_legend=False, max_mols=500,
              legend_float_decimals=4, mols_per_row=6, save_pth: T.Optional[Path] = None):
    """
    Returns svg image representing a grid of skeletal structures of the given molecules. Copy-pasted
     from https://github.com/pluskal-lab/DreaMS/blob/main/dreams/utils/mols.py

    :param mols: list of rdkit molecules
    :param smiles_in: True - SMILES inputs, False - RDKit mols
    :param legends: list of labels for each molecule, length must be equal to the length of mols
    :param svg: True - return svg image, False - return png image
    :param sort_by_legend: True - sort molecules by legend values
    :param max_mols: maximum number of molecules to show
    :param legend_float_decimals: number of decimal places to show for float legends
    :param mols_per_row: number of molecules per row to show
    :param save_pth: path to save the .svg image to
    """
    if smiles_in:
        mols = [Chem.MolFromSmiles(e) for e in mols]

    if legends == 'new_indices':
        legends = list(range(len(mols)))
    elif legends == 'masses':
        legends = [ExactMolWt(m) for m in mols]
    elif callable(legends):
        legends = [legends(e) for e in mols]

    if sort_by_legend:
        idx = np.argsort(legends).tolist()
        legends = [legends[i] for i in idx]
        mols = [mols[i] for i in idx]

    legends = [f'{l:.{legend_float_decimals}f}' if isinstance(l, float) else str(l) for l in legends]

    img = Draw.MolsToGridImage(mols, maxMols=max_mols, legends=legends, molsPerRow=min(max_mols, mols_per_row),
                         useSVG=svg, returnPNG=False)

    if save_pth:
        with open(save_pth, 'w') as f:
            f.write(img.data)

    return img


class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = True, # "False" makes computations a lot faster, but leads to overall higher MCES values
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
