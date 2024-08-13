import pytest
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from tokenizers import Tokenizer
from typing import List
from massspecgym.utils import SelfiesTokenizer, SmilesBPETokenizer, tanimoto_morgan_similarity


def assert_smiles_equal(smiles1: str, smiles2: str):
    """Assert that two SMILES strings represent the same molecule."""
    assert tanimoto_morgan_similarity(smiles1, smiles2) == 1.0


def assert_smiles_batch_equal(smiles_batch1: List[str], smiles_batch2: List[str]):
    """Assert that two lists of SMILES strings represent the same molecules."""
    assert len(smiles_batch1) == len(smiles_batch2)
    for smi1, smi2 in zip(smiles_batch1, smiles_batch2):
        assert_smiles_equal(smi1, smi2)


@pytest.fixture(scope="module")
def selfies_tokenizer():
    """Fixture for SelfiesTokenizer that ensures it's initialized once per module."""
    return SelfiesTokenizer(max_length=10)


@pytest.fixture(scope="module")
def bpe_tokenizer():
    """Fixture for SmilesBPETokenizer that ensures it's initialized once per module."""
    return SmilesBPETokenizer(max_length=10)


def test_selfies_encode_decode(selfies_tokenizer: Tokenizer):
    """Test encoding and decoding of a single SMILES string with SelfiesTokenizer."""
    smiles = "c1ccccc1"
    encoded = selfies_tokenizer.encode(smiles)
    assert isinstance(encoded.ids, list)
    assert len(encoded.ids) <= 10

    decoded = selfies_tokenizer.decode(encoded.ids)
    assert_smiles_equal(decoded, smiles)


def test_selfies_encode_decode_batch(selfies_tokenizer: Tokenizer):
    """Test batch encoding and decoding with SelfiesTokenizer."""
    smiles_batch = ["c1ccccc1", "O=C=O"]
    encoded_batch = selfies_tokenizer.encode_batch(smiles_batch)

    assert len(encoded_batch) == len(smiles_batch)
    for encoding in encoded_batch:
        assert isinstance(encoding.ids, list)
        assert len(encoding.ids) <= 10

    decoded_batch = selfies_tokenizer.decode_batch([encoding.ids for encoding in encoded_batch])
    assert_smiles_batch_equal(decoded_batch, smiles_batch)


def test_selfies_padding_truncation(selfies_tokenizer: Tokenizer):
    """Test padding and truncation with SelfiesTokenizer."""
    long_smiles = "C1CC(C=C1)C(=O)OCC2C(C(C(C(O2)OC3=C(C=C(C=C3)O)COC(=O)C4(C=CCCC4=O)O)OC(=O)C5CCC=C5)O)O"  # Longer SMILES string
    encoded = selfies_tokenizer.encode(long_smiles)
    assert len(encoded.ids) == 10  # Ensure it is truncated to max_length

    selfies_tokenizer.enable_padding(length=15)  # Increase padding length
    encoded = selfies_tokenizer.encode(long_smiles)
    assert len(encoded.ids) == 15  # Ensure it is padded to 15 tokens


def test_bpe_encode_decode(bpe_tokenizer: Tokenizer):
    """Test encoding and decoding of a single SMILES string with SmilesBPETokenizer."""
    smiles = "c1ccccc1"
    encoded = bpe_tokenizer.encode(smiles)
    assert isinstance(encoded.ids, list)
    assert len(encoded.ids) <= 10

    decoded = bpe_tokenizer.decode(encoded.ids)
    assert_smiles_equal(decoded, smiles)


def test_bpe_encode_decode_batch(bpe_tokenizer: Tokenizer):
    """Test batch encoding and decoding with SmilesBPETokenizer."""
    smiles_batch = ["c1ccccc1", "O=C=O"]
    encoded_batch = bpe_tokenizer.encode_batch(smiles_batch)

    assert len(encoded_batch) == len(smiles_batch)
    for encoding in encoded_batch:
        assert isinstance(encoding.ids, list)
        assert len(encoding.ids) <= 10

    decoded_batch = bpe_tokenizer.decode_batch([encoding.ids for encoding in encoded_batch])
    assert_smiles_batch_equal(decoded_batch, smiles_batch)


def test_bpe_padding_truncation(bpe_tokenizer: Tokenizer):
    """Test padding and truncation with SmilesBPETokenizer."""
    long_smiles = "C1CC(C=C1)C(=O)OCC2C(C(C(C(O2)OC3=C(C=C(C=C3)O)COC(=O)C4(C=CCCC4=O)O)OC(=O)C5CCC=C5)O)O"  # Longer SMILES string
    encoded = bpe_tokenizer.encode(long_smiles)
    assert len(encoded.ids) == 10  # Ensure it is truncated to max_length

    bpe_tokenizer.enable_padding(length=15)  # Increase padding length
    encoded = bpe_tokenizer.encode(long_smiles)
    assert len(encoded.ids) == 15  # Ensure it is padded to 15 tokens


def test_special_tokens(selfies_tokenizer: Tokenizer):
    """Test that special tokens are correctly added and handled in the tokenizer."""
    special_tokens = [selfies_tokenizer.pad_token, selfies_tokenizer.sos_token, selfies_tokenizer.eos_token]
    special_token_ids = [selfies_tokenizer.pad_token_id, selfies_tokenizer.sos_token_id, selfies_tokenizer.eos_token_id]

    for token, token_id in zip(special_tokens, special_token_ids):
        assert token_id == selfies_tokenizer.token_to_id(token)


def test_unpadding(selfies_tokenizer: Tokenizer):
    """Test unpadding functionality in the SelfiesTokenizer."""
    smiles = "c1ccccc1"
    encoded = selfies_tokenizer.encode(smiles)
    unpadded = [id for id in encoded.ids if id != selfies_tokenizer.pad_token_id]
    assert len(unpadded) <= len(encoded.ids)
    assert selfies_tokenizer.pad_token_id not in unpadded