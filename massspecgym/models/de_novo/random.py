import torch

from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel


class RandomDeNovo(DeNovoMassSpecGymModel):

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO Implement random de novo model to generate rdkit molecules
        mols_pred = None  # (bs, k) list of rdkit molecules

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)

        # TODO Return molecules in the dict
        return dict(loss=loss, mols_pred=mols_pred)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None
