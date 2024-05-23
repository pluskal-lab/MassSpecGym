import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl

from massspecgym.models.base import MassSpecGymModel


class DeNovoMassSpecGymModel(MassSpecGymModel, ABC):

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        metric_pref: str = ''
    ) -> None:
        pass
        # TODO: Implement evaluation for de novo models
        # Get `mols` from the `outputs` dict and compute metrics
    
        # TODO Probably implement evaluate_<XXX>_step for different XXX groups of metrics, or 
        # with different meaning / role that will be called here, or optionally in the step method  
