import pytest
import numpy as np
import torch
import matchms
from massspecgym.data.transforms import SpecTokenizer, SpecBinner, MolToFormulaVector


def test_spec_tokenizer():
    spec = matchms.Spectrum(
        mz=np.array(
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
            ]
        ),
        intensities=np.array(
            [0.01082, 0.01064, 0.17184, 0.1397, 0.00874, 1.0, 0.52842, 0.00793, 0.43696]
        ),
        metadata={"precursor_mz": 406.22},
    )

    # Prepend precursor token
    tokenizer = SpecTokenizer(n_peaks=60, prec_mz_intensity=1.1)
    spec_t = tokenizer(spec)
    assert spec_t.shape == (61, 2)

    # Do not prepend precursor token
    tokenizer = SpecTokenizer(n_peaks=60, prec_mz_intensity=None)
    spec_t = tokenizer(spec)
    assert spec_t.shape == (60, 2)
    assert (
        spec_t[: spec.peaks.mz.shape[0], 0] == torch.from_numpy(spec.peaks.mz)
    ).all()
    assert (
        spec_t[: spec.peaks.intensities.shape[0], 1]
        == torch.from_numpy(spec.peaks.intensities)
    ).all()


def test_spec_binner():
    spec = matchms.Spectrum(
        mz=np.array(
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
            ]
        ),
        intensities=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.1, 0.2]),
        metadata={"precursor_mz": 406.22},
    )
    binner = SpecBinner(max_mz=1000, bin_width=100, to_rel_intensities=False)
    spec_t = binner(spec)
    assert spec_t.shape == (1000 // 100,)
    assert torch.allclose(
        spec_t,
        torch.tensor([0.2, 0.3, 1.2, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]).double(),
    )


def test_resiniferatoxin_formula_vector():
    transform = MolToFormulaVector()
    # Resiniferatoxin (C37H40O9)
    smiles = "C[C@@H]1C[C@]2([C@H]3[C@H]4[C@]1([C@@H]5C=C(C(=O)[C@]5(CC(=C4)COC(=O)Cc6ccc(c(c6)OC)O)O)C)O[C@](O3)(O2)Cc7ccccc7)C(=C)C"
    vector = transform.from_smiles(smiles)
    
    expected_vector = np.zeros(118, dtype=np.float32)
    expected_vector[transform.element_index["C"]] = 37
    expected_vector[transform.element_index["H"]] = 40
    expected_vector[transform.element_index["O"]] = 9
    
    assert np.array_equal(vector, expected_vector), "Ethanol vector does not match expected output"

def test_water_formula_vector():
    transform = MolToFormulaVector()
    # Water (H2O)
    smiles = "O"
    vector = transform.from_smiles(smiles)
    
    expected_vector = np.zeros(118, dtype=np.float32)
    expected_vector[transform.element_index["H"]] = 2
    expected_vector[transform.element_index["O"]] = 1
    
    assert np.array_equal(vector, expected_vector), "Water vector does not match expected output"

# Test case for molecule with an invalid SMILES
def test_invalid_element():
    transform = MolToFormulaVector()
    smiles = "JkO"  # A hypothetical compound
    with pytest.raises(ValueError, match="Invalid SMILES string: JkO"):
        transform.from_smiles(smiles)
