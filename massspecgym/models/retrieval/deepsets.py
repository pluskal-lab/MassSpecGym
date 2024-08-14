import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.utils import CosSimLoss


class DeepSetsRetrieval(RetrievalMassSpecGymModel):
    def __init__(
        self,
        in_channels: int = 2,  # m/z and intensity of a peak
        hidden_channels: int = 512,  # hidden layer size
        out_channels: int = 4096,  # fingerprint size
        num_layers_per_mlp: int = 2,
        dropout: float = 0.0,
        norm: T.Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.phi = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_per_mlp,
            dropout=dropout,
            norm=norm
        )

        self.rho = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers_per_mlp,
            dropout=dropout,
            norm=norm
        )

        self.loss_fn = CosSimLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)
        x = x.sum(dim=-2)  # sum over peaks
        x = self.rho(x)
        x = F.sigmoid(x)  # predict proper fingerprint
        return x

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack inputs
        x = batch["spec"]
        fp_true = batch["mol"]
        cands = batch["candidates"]
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        fp_pred = self.forward(x)

        # Calculate loss
        loss = self.loss_fn(fp_true, fp_pred)

        # Evaluation performance on fingerprint prediction (optional)
        self.evaluate_fingerprint_step(fp_true, fp_pred, stage=stage)

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        return dict(loss=loss, scores=scores)
