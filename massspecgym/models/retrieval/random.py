import torch

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class RandomRetrieval(RetrievalMassSpecGymModel):

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate random retrieval scores
        scores = torch.rand(batch["candidates_mol"].shape[0]).to(self.device)

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None
