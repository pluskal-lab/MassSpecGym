import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, CosineSimilarity

from massspecgym.models.base import MassSpecGymModel


class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, metric_pref: str = ""
    ) -> None:
        """
        Compute evaluation metrics for the retrieval model based on the batch and corresponding predictions.
        This method will be used in the `on_train_batch_end`, `on_validation_batch_end`, since `on_test_batch_end` is
        overridden below.
        """
        self.evaluate_cos_similarity_step(
            outputs["spec_pred"],
            batch["spec"],
            metric_pref=metric_pref,
        )

    def on_test_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int
    ) -> None:
        metric_pref = "_test"
        self.evaluate_cos_similarity_step(
            outputs["spec_pred"],
            batch["spec"],
            metric_pref=metric_pref
        )
        self.evaluate_hit_rate_step(
            outputs["spec_pred"],
            batch["spec"],
            metric_pref=metric_pref
        )

    def evaluate_cos_similarity_step(
        self,
        specs_pred: torch.Tensor,
        specs: torch.Tensor,
        metric_pref: str = ""
    ) -> None:
        """
        Evaulate cosine similarity.
        """
        raise NotImplementedError

    def evaluate_hit_rate_step(
        self,
        specs_pred: torch.Tensor,
        specs: torch.Tensor,
        metric_pref: str = ""
    ) -> None:
        """
        Evaulate Hit rate @ {1, 5, 20} (typically reported as Accuracy @ {1, 5, 20}).
        """
        raise NotImplementedError
