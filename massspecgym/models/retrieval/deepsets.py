import torch
import torch.nn as nn

from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class DeepSetsRetrieval(RetrievalMassSpecGymModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.phi = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.phi(x)
        x = x.sum(dim=-2)  # sum over peaks
        x = self.rho(x)
        return x

    def step(
        self, 
        batch: dict, 
        metric_pref: str = ''
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack inputs
        x = batch['spec'].float()  # TODO Remove retyping
        y_true = batch['mol']

        # Predict
        y_pred = self.forward(x)

        # Calculate loss
        y_true = y_true.type_as(y_pred)  # convert fingerprint from int to float/double
        loss = nn.functional.mse_loss(y_true, y_pred)  # TODO Change to cosine similarity?

        # Log loss
        self.log(
            metric_pref + 'loss_step',
            loss,
            batch_size=x.size(0),
            sync_dist=True,
            prog_bar=True,
        )

        # Evaluation performance on fingerprint prediction
        self.evaluate_fingerprint_step(
            y_true,
            y_pred, 
            metric_pref=metric_pref
        )

        return loss, y_pred
