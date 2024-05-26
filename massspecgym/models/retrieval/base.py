import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, CosineSimilarity

from massspecgym.models.base import MassSpecGymModel


class RetrievalMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(self, at_ks: T.Iterable[int] = (1, 5, 10), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.at_ks = at_ks

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, metric_pref: str = ""
    ) -> None:
        """
        Compute evaluation metrics for the retrieval model based on the batch and corresponding
        predictions.
        """
        assert (
            isinstance(outputs, dict) and "scores" in outputs
        ), "No predicted candidate scores in the model outputs."
        self.evaluate_retrieval_step(
            outputs["scores"],
            batch["labels"],
            batch["batch_ptr"],
            metric_pref=metric_pref,
        )

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        metric_pref: str = "",
    ) -> None:
        """
        Main evaluation method for the retrieval models. The retrieval step is evaluated by 
        computing the hit rate at different top-k values.

        Args:
            scores (torch.Tensor): Concatenated scores for all candidates for all samples in the 
                batch
            labels (torch.Tensor): Concatenated True/False labels for all candidates for all samples
                 in the batch
            batch_ptr (torch.Tensor): Pointer to the start of each sample's candidates in the 
                concatenated tensors
        """
        indexes = torch.repeat_interleave(torch.arange(batch_ptr.size(0)), batch_ptr)
        for at_k in self.at_ks:
            self._update_metric(
                metric_pref + f"hit_rate@{at_k}",
                RetrievalHitRate,
                (scores, labels, indexes),
                batch_size=batch_ptr.size(0),
                metric_kwargs=dict(top_k=at_k),
            )

    def evaluate_fingerprint_step(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        batch_idx: T.Optional[torch.Tensor] = None,
        metric_pref: str = "",
    ) -> None:
        """
        Utility evaluation method to assess the quality of predicted fingerprints. This method is
        not a part of the necessary evaluation logic (not called in the `on_batch_end` method)
        since retrieval models are not bound to predict fingerprints.

        Args:
            y_true (torch.Tensor): [batch_size, fingerprint_size] tensor of true fingerprints
            y_pred (torch.Tensor): [batch_size, fingerprint_size] tensor of predicted fingerprints
        """
        # Cosine similarity between predicted and true fingerprints
        self._update_metric(
            metric_pref + "fingerprint_cos_sim",
            CosineSimilarity,
            (y_pred, y_true),
            batch_size=y_true.size(0),
            metric_kwargs=dict(reduction="mean"),
        )
