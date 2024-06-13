import torch
import torch.nn as nn

from massspecgym.models.base import Stage
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
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2048), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)
        x = x.sum(dim=-2)  # sum over peaks
        x = self.rho(x)
        return x

    def step(
        self, batch: dict, stage: Stage
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
        fp_true = fp_true.type_as(
            fp_pred
        )  # convert fingerprint from int to float/double
        loss = nn.functional.mse_loss(
            fp_true, fp_pred
        )  # TODO Change to cosine similarity?

        # Evaluation performance on fingerprint prediction (optional)
        self.evaluate_fingerprint_step(fp_true, fp_pred, stage=stage)

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        return dict(loss=loss, scores=scores)
