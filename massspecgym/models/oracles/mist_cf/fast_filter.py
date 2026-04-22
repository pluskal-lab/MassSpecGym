"""
FastFFN: Formula pre-filter model for MIST-CF.

Scores candidate formulas based only on formula composition (no spectrum),
used to reduce large SIRIUS candidate sets before the main neural scorer.

Ported from ~/mist-cf/src/mist_cf/fast_form_score/fast_form_model.py.
Uses the original mist-cf element ordering so that pretrained checkpoints
from that repo load correctly.
"""

import re
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from massspecgym.models.encoders.mist.form_embedders import get_embedder
from massspecgym.models.encoders.mist.modules import MLPBlocks

# Original mist-cf element ordering — must match training data for checkpoint compat.
_VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F",
    "Br", "B", "Se", "Fe", "Co", "As", "K", "Na",
]
_ELEMENT_VECTORS = np.eye(len(_VALID_ELEMENTS))
_element_to_pos = {el: _ELEMENT_VECTORS[i] for i, el in enumerate(_VALID_ELEMENTS)}
_FORMULA_RE = re.compile(r"([A-Z][a-z]*)(\d*)")


def _formula_to_dense(formula: str) -> np.ndarray:
    """Dense element-count vector using original mist-cf element ordering."""
    vec = np.zeros(len(_VALID_ELEMENTS), dtype=np.float32)
    for elem, count in _FORMULA_RE.findall(formula):
        if elem in _element_to_pos:
            vec += _element_to_pos[elem] * (int(count) if count else 1)
    return vec


class _InferenceDataset(Dataset):
    def __init__(self, formulas: List[str]):
        self.formulas = formulas
        self.dense = [_formula_to_dense(f) for f in formulas]

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, i):
        return {"formula": self.formulas[i], "x": self.dense[i]}

    @staticmethod
    def collate_fn(batch):
        return {
            "formulas": [b["formula"] for b in batch],
            "x": torch.from_numpy(np.stack([b["x"] for b in batch])),
        }


class FastFFN(pl.LightningModule):
    """Formula pre-filter FFN.

    Scores formulas by composition alone (no spectrum). Used to trim SIRIUS
    candidate sets before running the full MistCFNet scorer.

    Load pretrained weights with ``FastFFN.load_from_checkpoint(path)``
    (compatible with checkpoints from ~/mist-cf/src/mist_cf/fast_form_score/).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 7e-4,
        lr_decay_frac: float = 1.0,
        weight_decay: float = 0.0,
        form_encoder: str = "abs-sines",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.form_embedder = get_embedder(form_encoder)
        input_dim = self.form_embedder.full_dim
        self.mlp = MLPBlocks(
            input_size=input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=layers,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, formulae: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            formulae (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor, shape [batch] with scores for each candidate.
        """
        inputs = self.form_embedder(formulae)
        output = self.mlp(inputs)
        output = self.output_layer(output)
        output = self.sigmoid(output)
        return output.squeeze(-1)

    def _step(self, batch, stage: str):
        x, y = batch["x"].float(), batch["y"].float()
        loss = F.binary_cross_entropy(self(x), y)
        self.log(f"{stage}_loss", loss)
        return {"loss": loss}

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def fast_filter_candidates(
    formulas: List[str],
    model: FastFFN,
    max_k: int,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> List[int]:
    """Score formulas and return indices of the top-k candidates.

    Args:
        formulas: Candidate formula strings.
        model: Loaded FastFFN instance.
        max_k: Maximum number of candidates to keep.
        device: Torch device (defaults to CPU).
        batch_size: Inference batch size.

    Returns:
        List of int indices into ``formulas``, sorted best-first.
    """
    if not formulas:
        return []
    if device is None:
        device = torch.device("cpu")

    ds = _InferenceDataset(formulas)
    loader = DataLoader(
        ds, batch_size=batch_size,
        collate_fn=_InferenceDataset.collate_fn,
        shuffle=False,
    )
    model = model.to(device).eval()
    scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].float().to(device)
            scores.extend(model(x).cpu().tolist())

    scores_arr = np.array(scores)
    top_k = min(max_k, len(scores_arr))
    return np.argsort(scores_arr)[::-1][:top_k].tolist()
