import numpy as np
import matchms
from rdkit.Chem import AllChem as Chem
from massspecgym.preprocessors import SpecTokenizer
from massspecgym.utils import pad_spectrum, morgan_fp


def test_pad_spectrum():
    spec = np.array([
        [45.134823, 45.13699 , 110.096245, 130.064972, 136.111862, 277.16571 , 289.165924, 307.177856, 406.223083],
        [0.01082, 0.01064, 0.17184, 0.1397, 0.00874, 1., 0.52842, 0.00793, 0.43696],
    ], dtype=float).T
    
    i = 32
    spec_pad = pad_spectrum(spec, i, pad_value=0)
    assert spec_pad.shape == (i, 2)
    assert (spec_pad[0, :spec.shape[0]] == spec[0]).all()
    assert (spec_pad[1, :spec.shape[0]] == spec[1]).all()
    assert (spec_pad == 0).sum() == 2 * (i - spec.shape[0])
    assert (pad_spectrum(spec, spec.shape[0]) == spec).all()

    i = 5
    try:
        pad_spectrum(spec, i)
    except ValueError:
        pass
    else:
        assert False


def test_spec_tokenizer():
    spec = matchms.Spectrum(
        mz=np.array([45.134823, 45.13699 , 110.096245, 130.064972, 136.111862, 277.16571 , 289.165924, 307.177856, 406.223083]),
        intensities=np.array([0.01082, 0.01064, 0.17184, 0.1397, 0.00874, 1., 0.52842, 0.00793, 0.43696]),
        metadata={'precursor_mz': 406.22}
    )
    tokenizer = SpecTokenizer(pad_to_n_peaks=None)
    spec_np = tokenizer(spec)
    assert spec_np.shape == (9, 2)
    assert (spec_np[:, 0] == spec.peaks.mz).all()
    assert (spec_np[:, 1] == spec.peaks.intensities).all()

    tokenizer = SpecTokenizer(pad_to_n_peaks=60)
    spec_np = tokenizer(spec)
    assert spec_np.shape == (60, 2)


def test_morgan_fp():
    mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(C)c(=O)n2C')
    for i in [2048, 4096]:
        fp = morgan_fp(mol, fp_size=i)
        assert fp.shape == (i,)
        assert fp.dtype == np.int32    