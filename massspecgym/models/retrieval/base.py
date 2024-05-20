import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import CosineSimilarity

from massspecgym.models.base import MassSpecGymModel


class RetrievalMassSpecGymModel(MassSpecGymModel, ABC):
    
    def evaluate_fingerprint_step(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        batch_idx: T.Optional[torch.Tensor] = None,
        metric_pref: str = ''
    ) -> None:
        # Cosine similarity between predicted and true fingerprints
        self._update_metric(
            metric_pref + 'fingerprint_cos_sim',
            CosineSimilarity,
            (y_pred, y_true),
            batch_size=y_true.size(0),
            metric_kwargs=dict(reduction='mean')
        )
