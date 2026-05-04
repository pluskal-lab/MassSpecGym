"""
Fast formula pre-filter for MIST-CF candidate lists.

The original MIST-CF pipeline can enumerate a large number of formulas with
SIRIUS. FastFFN scores formulas by composition only and keeps a smaller set
before the spectrum-conditioned MIST-CF scorer runs.
"""

import re
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from massspecgym.models.encoders.mist.form_embedders import get_embedder
from massspecgym.models.encoders.mist.modules import MLPBlocks


# Original mist-cf element ordering. This is intentionally separate from
# massspecgym.models.encoders.mist.chem_constants.VALID_ELEMENTS so checkpoints
# trained in the upstream mist-cf code see the expected input order.
_VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F",
    "Br", "B", "Se", "Fe", "Co", "As", "K", "Na",
]
_ELEMENT_VECTORS = np.eye(len(_VALID_ELEMENTS))
_ELEMENT_TO_POS = {el: _ELEMENT_VECTORS[i] for i, el in enumerate(_VALID_ELEMENTS)}
_FORMULA_RE = re.compile(r"([A-Z][a-z]*)(\d*)")


def _formula_to_dense(formula: str) -> np.ndarray:
    vec = np.zeros(len(_VALID_ELEMENTS), dtype=np.float32)
    for elem, count in _FORMULA_RE.findall(formula):
        if elem in _ELEMENT_TO_POS:
            vec += _ELEMENT_TO_POS[elem] * (int(count) if count else 1)
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
    """Formula-only pre-filter model used by MIST-CF."""

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
        self.mlp = MLPBlocks(
            input_size=self.form_embedder.full_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=layers,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, formulae: torch.Tensor) -> torch.Tensor:
        inputs = self.form_embedder(formulae)
        output = self.mlp(inputs)
        output = self.output_layer(output)
        return self.sigmoid(output).squeeze(-1)

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
    """Return top-k formula indices sorted by FastFFN score."""
    if not formulas:
        return []
    if device is None:
        device = torch.device("cpu")

    ds = _InferenceDataset(formulas)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_InferenceDataset.collate_fn,
        shuffle=False,
    )
    model = model.to(device).eval()
    scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].float().to(device)
            scores.extend(model(x).cpu().tolist())

    scores_arr = np.asarray(scores)
    top_k = min(max_k, len(scores_arr))
    return np.argsort(scores_arr)[::-1][:top_k].tolist()
