import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - F.cosine_similarity(inputs, targets).mean()


class FingerprintFFNRetrieval(RetrievalMassSpecGymModel):
    def __init__(
        self,
        in_channels: int = 1000,  # number of bins
        hidden_channels: int = 512,  # hidden layer size
        out_channels: int = 4096,  # fingerprint size
        num_layers: int = 2,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ffn = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout
        )

        self.loss_fn = CosSimLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        return x

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack inputs
        x = batch["spec"].float()  # TODO Remove retyping
        fp_true = batch["mol"]
        cands = batch["candidates"]
        labels = batch["labels"]
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        fp_pred = self.forward(x)

        # Calculate loss
        fp_true = fp_true.type_as(fp_pred)  # convert fingerprint from int to float/double
        loss = self.loss_fn(fp_true, fp_pred)

        # Log loss
        self.log(
            metric_pref + "loss_step",
            loss,
            batch_size=x.size(0),
            sync_dist=True,
            prog_bar=True,
        )

        # Evaluation performance on fingerprint prediction (optional)
        self.evaluate_fingerprint_step(fp_true, fp_pred, metric_pref=metric_pref)

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        return dict(loss=loss, scores=scores)
