import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, CosineSimilarity

from massspecgym.models.base import MassSpecGymModel


class RetrievalMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        top_ks: T.Iterable[int] = (1, 5, 10),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.top_ks = top_ks

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        metric_pref: str = ''
    ) -> None:
        assert isinstance(outputs, dict) and 'cos_sim' in outputs, 'No cosine similarity in the model outputs.'
        self.evaluate_retrieval_step(
            outputs['cos_sim'],
            batch['labels'],
            batch['batch_ptr'],
            metric_pref=metric_pref
        )

    def evaluate_retrieval_step(
        self,
        cos_sim: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        metric_pref: str = ''
    ) -> None:
        indexes = torch.repeat_interleave(torch.arange(batch_ptr.size(0)), batch_ptr)
        for top_k in self.top_ks:
            self._update_metric(
                metric_pref + f'hit_rate@{top_k}',
                RetrievalHitRate,
                (cos_sim, labels, indexes),
                batch_size=batch_ptr.size(0),
                metric_kwargs=dict(top_k=top_k)
            )    

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
