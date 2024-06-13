import typing as T
from enum import Enum
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from torchmetrics import Metric, SumMetric


class Stage(Enum):

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def to_pref(self) -> str:
        return f"{self.value}_"


class MassSpecGymModel(pl.LightningModule, ABC):

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        validate_only_loss: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validate_only_loss = validate_only_loss

    @abstractmethod
    def step(
        self, batch: dict, stage: Stage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Method `step` must be implemented in the model-specific child class."
        )

    def training_step(
        self, batch: dict, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(batch, stage=Stage.TRAIN)

    def validation_step(
        self, batch: dict, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(batch, stage=Stage.VAL)

    def test_step(
        self, batch: dict, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(batch, stage=Stage.TEST)

    @abstractmethod
    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        """
        Method to be called at the end of each batch. This method should be implemented by a child,
        task-dedicated class and contain the evaluation necessary for the task.
        """
        raise NotImplementedError(
            "Method `on_batch_end` must be implemented in the task-specific child class."
        )

    def on_train_batch_end(self, *args, **kwargs):
        return self.on_batch_end(*args, **kwargs, stage=Stage.TRAIN)

    def on_validation_batch_end(self, *args, **kwargs):
        return self.on_batch_end(*args, **kwargs, stage=Stage.VAL)

    def on_test_batch_end(self, *args, **kwargs):
        return self.on_batch_end(*args, **kwargs, stage=Stage.TEST)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_checkpoint_monitors(self) -> list[dict]:
        monitors = [
            {"monitor": f"{Stage.VAL.to_pref()}loss", "mode": "min", "early_stopping": True}
        ]
        return monitors

    def _update_metric(
        self,
        name: str,
        metric_class: type[Metric],
        update_args: T.Any,
        batch_size: T.Optional[int] = None,
        prog_bar: bool = False,
        metric_kwargs: T.Optional[dict] = None,
        log: bool = True,
        log_n_samples: bool = False,
    ) -> None:
        """
        This method enables updating and logging metrics without instantiating them in advance in
        the __init__ method. The metrics are aggreated over batches and logged at the end of the
        epoch. If the metric does not exist yet, it is instantiated and added as an attribute to the
        model.
        """
        # Log total number of samples for debugging
        if log_n_samples:
            self._update_metric(
                name=name + "_n_samples",
                metric_class=SumMetric,
                update_args=(len(update_args[0]),),
                batch_size=1,
            )

        # Init metric if does not exits yet
        if hasattr(self, name):
            metric = getattr(self, name)
        else:
            if metric_kwargs is None:
                metric_kwargs = dict()
            metric = metric_class(**metric_kwargs).to(self.device)
            setattr(self, name, metric)

        # Update
        metric(*update_args)

        # Log
        if log:
            self.log(
                name,
                metric,
                prog_bar=prog_bar,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                metric_attribute=name,  # Suggested by a torchmetrics error
            )
