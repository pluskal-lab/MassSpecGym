import torch

from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel


class RandomDeNovo(DeNovoMassSpecGymModel):

    def __init__(self, formula_known: bool = False):
        super(RandomDeNovo, self).__init__()
        self.formula_known = formula_known

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mols = batch["mol"]  # List of SMILES of length batch_size
        
        # TODO: Implement random generation of molecules
        # If formula_known is True, we should generate molecules with the same formula as label (`mols` above)
        # If formula_known is False, we should generate any molecule with the same mass as label (`mols` above)
        mols_pred = None  # (bs, k) list of rdkit molecules

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        return dict(loss=loss, mols_pred=mols_pred)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None
