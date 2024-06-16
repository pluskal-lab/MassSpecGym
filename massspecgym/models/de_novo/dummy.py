import random

import torch

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel


class DummyDeNovo(DeNovoMassSpecGymModel):

    def __init__(self, n_samples: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

        self.dummy_smiles = [
            "O",                          # Water (H₂O)
            "C",                          # Methane (CH₄)
            "CCO",                        # Ethanol (C₂H₆O)
            "C(C1C(C(C(C(O1)O)O)O)O)O",   # Glucose (C₆H₁₂O₆)
            "CC(=O)C",                    # Acetone (C₃H₆O)
            "CC(=O)Oc1ccccc1C(=O)O",      # Aspirin (C₉H₈O₄)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine (C₈H₁₀N₄O₂)
            "c1ccccc1",                   # Benzene (C₆H₆)
            "CC(=O)O",                    # Acetic Acid (C₂H₄O₂)
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen (C₁₃H₁₈O₂)
            None
        ]
        self.mol_pred_kind = "smiles"

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = batch['spec'].shape[0]

        # Sample dummy molecules from the pre-defined list
        mols_pred = [[random.choice(self.dummy_smiles) for _ in range(self.n_samples)] for _ in range(bs)]

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)

        # Return molecules in the dict
        return dict(loss=loss, mols_pred=mols_pred)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None
