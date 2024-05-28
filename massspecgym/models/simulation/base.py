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
        Compute evaluation metrics for the retrieval model based on the batch and corresponding
        predictions.
        """
        self.evaluate_simulation_step(
            outputs["spec_pred"],
            batch["spec"],
            metric_pref=metric_pref,
        )

    def evaluate_simulation_step(
        self,
        specs_pred: torch.Tensor,
        specs: torch.Tensor,
        metric_pref: str = ""
    ) -> None:
        """
        TODO: Implement Hit rate @ {1, 5, 20} (typically reported as Accuracy @ {1, 5, 20}) and cosine similarity
              evaluation metrics.
        """
        raise NotImplementedError
