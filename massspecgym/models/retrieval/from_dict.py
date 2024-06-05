import typing as T

import torch
import torch.nn as nn

from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class FromDictRetrieval(RetrievalMassSpecGymModel):

    def __init__(self, dct: dict[str, T.Any], *args, **kwargs):
        """
        Read predictions from dictionary with MassSpecGym ids as keys. Currently, the class
        only implements reading fingerprints from the dictionary.
        """
        super().__init__(*args, **kwargs)
        dct = {k: torch.tensor(v) for k, v in dct.items()}
        self.dct = dct

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack inputs
        ids = batch["identifier"]
        fp_true = batch["mol"]
        cands = batch["candidates"]
        batch_ptr = batch["batch_ptr"]

        # Read predicted fingerprints from dictionary
        fp_pred = torch.stack([self.dct[id] for id in ids])

        # Convert fingerprint from int to float/double
        fp_true = fp_true.type_as(fp_pred)

        # Evaluation performance on fingerprint prediction (optional)
        self.evaluate_fingerprint_step(fp_true, fp_pred, metric_pref=metric_pref)

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        # No training, so no optimizers
        return None
