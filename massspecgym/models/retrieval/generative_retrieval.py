"""
Generative retrieval: generate molecule from spectrum, rank candidates by FP similarity.

Uses any MIST+decoder model to generate a top-1 molecule from the spectrum,
converts it to a Morgan fingerprint, and ranks retrieval candidates by
Tanimoto similarity to the generated molecule's fingerprint.

This is a bonus-task retrieval strategy that can use any FP2Mol decoder
(FRIGID, DiffMS, MolForge).
"""

import typing as T

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class GenerativeRetrieval(RetrievalMassSpecGymModel):
    """Generative retrieval: generate molecule, then rank by fingerprint similarity.

    Pipeline:
    1. MIST encoder: spectrum → fingerprint
    2. FP2Mol decoder: fingerprint → top-1 molecule (SMILES)
    3. Compute Morgan FP of generated molecule
    4. Rank candidates by Tanimoto similarity to generated FP

    Args:
        decoder_type: Type of decoder ('frigid', 'diffms', 'molforge').
        decoder_checkpoint: Path to pretrained decoder checkpoint.
        encoder_checkpoint: Path to pretrained MIST encoder checkpoint.
        fp_bits: Fingerprint dimensionality for comparison.
        fp_radius: Morgan FP radius.
    """

    def __init__(
        self,
        decoder_type: str = "frigid",
        decoder_checkpoint: T.Optional[str] = None,
        encoder_checkpoint: T.Optional[str] = None,
        fp_bits: int = 2048,
        fp_radius: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self._decoder = None
        self._decoder_checkpoint = decoder_checkpoint
        self._encoder_checkpoint = encoder_checkpoint

    def _get_decoder(self):
        """Lazy-load the FP2Mol decoder."""
        if self._decoder is not None:
            return self._decoder

        if self.decoder_type == "frigid":
            from massspecgym.models.de_novo.fp2mol.frigid import FRIGIDDecoder
            from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
            encoder = SpectraEncoderGrowing(
                form_embedder="pos-cos", output_size=4096, hidden_size=256,
                peak_attn_layers=4, num_heads=8, refine_layers=4,
                set_pooling="cls", pairwise_featurization=True,
            )
            self._decoder = FRIGIDDecoder(
                encoder=encoder,
                encoder_checkpoint=self._encoder_checkpoint,
                training_mode="spec2mol",
            )
        elif self.decoder_type == "molforge":
            from massspecgym.models.de_novo.fp2mol.molforge import MolForgeDecoder
            from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
            encoder = SpectraEncoderGrowing(
                form_embedder="pos-cos", output_size=2048, hidden_size=256,
                peak_attn_layers=4, num_heads=8, refine_layers=4,
                set_pooling="cls", pairwise_featurization=True,
            )
            self._decoder = MolForgeDecoder(
                encoder=encoder,
                encoder_checkpoint=self._encoder_checkpoint,
                training_mode="spec2mol",
            )
        elif self.decoder_type == "diffms":
            from massspecgym.models.de_novo.fp2mol.diffms import DiffMSDecoder
            from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
            encoder = SpectraEncoderGrowing(
                form_embedder="pos-cos", output_size=2048, hidden_size=256,
                peak_attn_layers=4, num_heads=8, refine_layers=4,
                set_pooling="cls", pairwise_featurization=True,
            )
            self._decoder = DiffMSDecoder(
                encoder=encoder,
                encoder_checkpoint=self._encoder_checkpoint,
                training_mode="spec2mol",
            )
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        if self._decoder_checkpoint:
            ckpt = torch.load(self._decoder_checkpoint, map_location="cpu")
            self._decoder.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

        self._decoder = self._decoder.to(self.device)
        return self._decoder

    def _smiles_to_fp(self, smiles: str) -> torch.Tensor:
        """Convert SMILES to Morgan fingerprint tensor."""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return torch.zeros(self.fp_bits, dtype=torch.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
        arr = np.zeros(self.fp_bits, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.from_numpy(arr)

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        loss = torch.tensor(0.0, device=self.device)
        batch_size = batch["batch_ptr"].size(0)
        cands = batch.get("candidates_mol", batch.get("candidates"))
        batch_ptr = batch["batch_ptr"]

        generated_fps = []
        decoder = self._get_decoder()
        with torch.no_grad():
            encoded_fp, _ = decoder.encode_spectrum(batch)
        formula = batch.get("formula_str", batch.get("formula", None))
        for i in range(batch_size):
            try:
                mols_pred = decoder.decode_from_fingerprint(
                    encoded_fp[i:i+1],
                    formula=[formula[i]] if formula is not None else None,
                    num_samples=1,
                )
                top1_smiles = mols_pred[0][0] if mols_pred and mols_pred[0] else None
            except Exception:
                top1_smiles = None
            generated_fps.append(self._smiles_to_fp(top1_smiles))

        gen_fps = torch.stack(generated_fps).to(self.device)
        gen_fps_repeated = gen_fps.repeat_interleave(batch_ptr, dim=0)

        intersection = (gen_fps_repeated * cands).sum(dim=-1)
        union = gen_fps_repeated.sum(dim=-1) + cands.sum(dim=-1) - intersection
        scores = intersection / union.clamp(min=1e-8)

        return dict(loss=loss, scores=scores, processable_mask=batch.get("processable_mask", None))
