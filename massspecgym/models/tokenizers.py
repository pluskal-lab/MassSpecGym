import pandas as pd
import typing as T
import selfies as sf
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
import massspecgym.utils as utils
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class SpecialTokensBaseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_len: T.Optional[int] = None,
    ):
        """Initialize the base tokenizer with special tokens and optional padding."""
        super().__init__(tokenizer)

        # Save essential attributes
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len

        # Add special tokens
        self.add_special_tokens([self.pad_token, self.sos_token, self.eos_token, self.unk_token])

        # Get token IDs
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.sos_token_id = self.token_to_id(self.sos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)

        # Enable padding
        self.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.pad_token_id,
            length=max_len,
        )

        # Enable truncation
        self.enable_truncation(max_len)

        # Set post-processing to add SOS and EOS tokens
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            pair=f"{self.sos_token} $A {self.eos_token} {self.sos_token} $B {self.eos_token}",
            special_tokens=[
                (self.sos_token, self.sos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


class SelfiesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(
            self,
            selfies_train: T.Optional[T.Union[str, T.List[str]]] = None,
            **kwargs
        ):
        """
        Initialize the SELFIES tokenizer with optional training data to build a vocanulary.

        Args:
            selfies_train (str or list of str): Either a list of SELFIES strings to build the vocabulary from,
                or a `semantic_robust_alphabet` string indicating the usahe of `selfies.get_semantic_robust_alphabet()`
                alphabet. If None, the MassSpecGym training molecules will be used.
        """

        if selfies_train == 'semantic_robust_alphabet':
            alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        else:
            if not selfies_train:
                selfies_train = utils.load_train_mols()
                selfies = [sf.encoder(s, strict=False) for s in selfies_train]
            else:
                selfies = selfies_train
            alphabet = list(sorted(sf.get_alphabet_from_selfies(selfies))) 

        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))

        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of SELFIES token IDs."""
        selfies_string = sf.encoder(text, strict=False)
        selfies_tokens = list(sf.split_selfies(selfies_string))
        return super().encode(
            selfies_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of SELFIES token IDs back into a SMILES string."""
        selfies_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        selfies_string = self._decode_wordlevel_str_to_selfies(selfies_string)
        return sf.decoder(selfies_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of SELFIES token IDs."""
        selfies_strings = [
            list(sf.split_selfies(sf.encoder(text, strict=False))) for text in texts
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
                    selfies_string
                )
            )
            for selfies_string in selfies_strings
        ]

    def _decode_wordlevel_str_to_selfies(self, text: str) -> str:
        """Converts a WordLevel string back to a SELFIES string."""
        return text.replace(" ", "")


class SmilesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, smiles_pth: T.Optional[str] = None, **kwargs):
        """
        Initialize the BPE tokenizer for SMILES strings, with optional training data.

        Args:
            smiles_pth (str): Path to a file containing SMILES strings to train the tokenizer on. If None,
                the MassSpecGym training molecules will be used.
        """
        tokenizer = ByteLevelBPETokenizer()
        if smiles_pth:
            tokenizer.train(smiles_pth)
        else:
            smiles = utils.load_unlabeled_mols("smiles").tolist()
            smiles += utils.load_train_mols().tolist()

            print(f"Training tokenizer on {len(smiles)} SMILES strings.")
            tokenizer.train_from_iterator(smiles)

        super().__init__(tokenizer, **kwargs)
