import numpy as np
import matchms
import os
from rdkit.Chem import AllChem as Chem
import massspecgym.utils as utils


def test_pad_spectrum():
    spec = np.array(
        [
            [
                45.134823,
                45.13699,
                110.096245,
                130.064972,
                136.111862,
                277.16571,
                289.165924,
                307.177856,
                406.223083,
            ],
            [
                0.01082,
                0.01064,
                0.17184,
                0.1397,
                0.00874,
                1.0,
                0.52842,
                0.00793,
                0.43696,
            ],
        ],
        dtype=float,
    ).T

    i = 32
    spec_pad = utils.pad_spectrum(spec, i, pad_value=0)
    assert spec_pad.shape == (i, 2)
    assert (spec_pad[0, : spec.shape[0]] == spec[0]).all()
    assert (spec_pad[1, : spec.shape[0]] == spec[1]).all()
    assert (spec_pad == 0).sum() == 2 * (i - spec.shape[0])
    assert (utils.pad_spectrum(spec, spec.shape[0]) == spec).all()

    i = 5
    try:
        utils.pad_spectrum(spec, i)
    except ValueError:
        pass
    else:
        assert False


def test_morgan_fp():
    mol = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    for i in [2048, 4096]:
        fp = utils.morgan_fp(mol, fp_size=i)
        assert fp.shape == (i,)
        assert fp.dtype == np.int32


def test_standardize_smiles():
    def asserts():
        assert utils.standardize_smiles("OCO") == "C(O)O"
        assert utils.standardize_smiles(["OCO", "CC"]) == ["C(O)O", "CC"]
    try:
        asserts()
    except ImportError:
        os.system("pip install git+https://github.com/boecker-lab/standardizeUtils@b415f1c51b49f6c5cd0e9c6ab89224c8ad657a35#egg=standardizeUtils")
        asserts()
