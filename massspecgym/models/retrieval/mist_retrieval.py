"""
MIST fingerprint retrieval: predict Morgan FP from spectrum, rank by Tanimoto.

Uses the MIST encoder (SpectraEncoderGrowing) to predict a 2048-bit molecular
fingerprint from the MS/MS spectrum, then ranks retrieval candidates by
Tanimoto similarity between predicted and candidate fingerprints.

This is a bonus-task retrieval strategy.
"""

import typing as T

import torch
import torch.nn as nn

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.utils import CosSimLoss


class MISTFingerprintRetrieval(RetrievalMassSpecGymModel):
    """MIST-based retrieval via predicted fingerprint similarity.

    Loads a pretrained MIST SpectraEncoderGrowing checkpoint, predicts
    a fingerprint for each query spectrum, and ranks candidates by
    Tanimoto (or cosine) similarity.

    Note: Requires MIST-featurized input (subformulae assignment).
    Use MISTDataMixin for automatic data preparation.

    Args:
        encoder_checkpoint: Path to pretrained MIST encoder checkpoint.
        fp_bits: Fingerprint dimensionality (4096 for Morgan).
        similarity: Similarity function ('cosine' or 'tanimoto').
    """

    def __init__(
        self,
        encoder_checkpoint: T.Optional[str] = None,
        fp_bits: int = 2048,
        similarity: str = "tanimoto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fp_bits = fp_bits
        self.similarity = similarity
        self.loss_fn = CosSimLoss()

        from massspecgym.models.encoders.mist.encoder import SpectraEncoderGrowing
        self.encoder = SpectraEncoderGrowing(
            form_embedder="pos-cos", output_size=fp_bits, hidden_size=256,
            peak_attn_layers=4, num_heads=8, refine_layers=4,
            set_pooling="cls", pairwise_featurization=True,
        )

        if encoder_checkpoint:
            ckpt = torch.load(encoder_checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, batch: dict) -> torch.Tensor:
        """Predict fingerprint from spectrum."""
        fp_pred, _ = self.encoder(batch)
        return fp_pred

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        fp_pred = self.forward(batch)
        fp_true = batch.get("mol")
        if fp_true is not None and fp_true.shape[-1] == fp_pred.shape[-1]:
            loss = self.loss_fn(fp_true, fp_pred)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=fp_pred.device)

        cands = batch.get("candidates_mol", batch.get("candidates"))
        batch_ptr = batch["batch_ptr"]
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)

        if self.similarity == "tanimoto":
            intersection = (fp_pred_repeated * cands).sum(dim=-1)
            union = fp_pred_repeated.sum(dim=-1) + cands.sum(dim=-1) - intersection
            scores = intersection / union.clamp(min=1e-8)
        else:
            scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        return dict(loss=loss, scores=scores, processable_mask=batch.get("processable_mask", None))
