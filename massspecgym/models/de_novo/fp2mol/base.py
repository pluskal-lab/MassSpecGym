"""
Base class for all fingerprint-to-molecule de novo generation models.

FP2MolDeNovoModel extends DeNovoMassSpecGymModel to support the common
two-stage pipeline: spectrum -> fingerprint -> molecule. It provides:

1. Optional MIST encoder for end-to-end spectrum-to-molecule inference.
2. Two training modes:
   - ``fp2mol_pretrain``: Train only the decoder on (fingerprint, formula, molecule)
     triples from any molecule library.
   - ``spec2mol``: Train or evaluate the full encoder+decoder pipeline on
     MassSpecGym (spectrum, molecule) paired data.
3. A unified ``step()`` that routes to decoder-specific loss and generation.
"""

import typing as T
from abc import abstractmethod
from pathlib import Path

import torch

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel


class FP2MolDeNovoModel(DeNovoMassSpecGymModel):
    """Base class for fingerprint-to-molecule de novo models.

    Subclasses must implement:
    - ``decode_from_fingerprint``: Generate molecules from a fingerprint (+formula).
    - ``compute_decoder_loss``: Compute decoder-specific training loss from a batch.

    Args:
        encoder: Optional MIST encoder module (for end-to-end spec2mol mode).
        encoder_checkpoint: Path to pretrained encoder weights.
        fingerprint_bits: Dimensionality of the fingerprint vector.
        use_formula: Whether the decoder uses formula conditioning.
        training_mode: One of 'fp2mol_pretrain' or 'spec2mol'.
        freeze_encoder: If True, freeze encoder weights during training.
        num_generation_samples: Number of molecules to generate per spectrum at inference.
    """

    def __init__(
        self,
        encoder: T.Optional[torch.nn.Module] = None,
        encoder_checkpoint: T.Optional[str] = None,
        fingerprint_bits: int = 4096,
        use_formula: bool = True,
        training_mode: str = "spec2mol",
        freeze_encoder: bool = True,
        num_generation_samples: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.fingerprint_bits = fingerprint_bits
        self.use_formula = use_formula
        self.training_mode = training_mode
        self.freeze_encoder = freeze_encoder
        self.num_generation_samples = num_generation_samples

        if encoder_checkpoint is not None and self.encoder is not None:
            self._load_encoder_checkpoint(encoder_checkpoint)

        if self.freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_encoder_checkpoint(self, checkpoint_path: str):
        """Load pretrained encoder weights from a checkpoint file."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get("state_dict", ckpt)
        self.encoder.load_state_dict(state_dict, strict=False)

    def encode_spectrum(self, batch: dict) -> T.Tuple[torch.Tensor, dict]:
        """Run the MIST encoder on a spectrum batch.

        Args:
            batch: Batch dict containing spectrum data (in MIST featurized format).

        Returns:
            Tuple of (fingerprint tensor [B, fingerprint_bits], aux_outputs dict).
        """
        if self.encoder is None:
            raise RuntimeError(
                "No encoder available. Either provide an encoder module or use "
                "training_mode='fp2mol_pretrain' with pre-computed fingerprints."
            )
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(batch)
        return self.encoder(batch)

    @abstractmethod
    def decode_from_fingerprint(
        self,
        fingerprint: torch.Tensor,
        formula: T.Optional[T.Any] = None,
        num_samples: int = 1,
    ) -> list[list[T.Optional[str]]]:
        """Generate molecules from fingerprint (and optionally formula).

        Args:
            fingerprint: Fingerprint tensor [B, fingerprint_bits].
            formula: Optional formula conditioning (format depends on decoder).
            num_samples: Number of molecules to generate per input.

        Returns:
            List of lists of SMILES strings (or None for failed generations).
            Shape: (batch_size, num_samples).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_decoder_loss(self, batch: dict) -> torch.Tensor:
        """Compute the decoder-specific training loss.

        Args:
            batch: Batch dict. In fp2mol_pretrain mode, contains 'fingerprint',
                   'formula', and target molecule representations. In spec2mol mode,
                   contains spectrum data and molecule targets.

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        """Unified training/evaluation step.

        In fp2mol_pretrain mode:
            - Loss is computed from pre-computed fingerprints in the batch.
            - Molecule generation is only performed during val/test.

        In spec2mol mode:
            - Fingerprints are obtained from the MIST encoder.
            - Loss and generation use the encoder output.
        """
        if self.training_mode == "fp2mol_pretrain":
            loss = self.compute_decoder_loss(batch)
            if stage in self.log_only_loss_at_stages:
                mols_pred = None
            else:
                fp = batch["fingerprint"]
                formula = batch.get("formula", None)
                mols_pred = self.decode_from_fingerprint(
                    fp, formula, num_samples=self.num_generation_samples
                )
        else:
            fp, _ = self.encode_spectrum(batch)
            decoder_batch = dict(batch)
            decoder_batch["fingerprint"] = fp
            if "formula" not in decoder_batch and "formula_str" in decoder_batch:
                decoder_batch["formula"] = decoder_batch["formula_str"]
            loss = self.compute_decoder_loss(decoder_batch)
            if stage in self.log_only_loss_at_stages:
                mols_pred = None
            else:
                formula = decoder_batch.get("formula", None)
                mols_pred = self.decode_from_fingerprint(
                    fp, formula, num_samples=self.num_generation_samples
                )
        return dict(loss=loss, mols_pred=mols_pred, processable_mask=batch.get("processable_mask", None))
