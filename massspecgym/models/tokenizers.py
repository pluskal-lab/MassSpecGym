import pandas as pd
import typing as T
import selfies as sf
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
from massspecgym.utils import hugging_face_download


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