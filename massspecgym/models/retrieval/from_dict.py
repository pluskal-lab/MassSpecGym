import pickle
import typing as T
from pathlib import Path

import torch
import torch.nn as nn

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class FromDictRetrieval(RetrievalMassSpecGymModel):
    """
    Read predictions from dictionary with MassSpecGym ids as keys. Currently, the class
    only implements reading fingerprints from the dictionary.
    """

    def __init__(
        self,
        dct: T.Optional[dict[str, T.Any]] = None,
        dct_path: T.Optional[T.Union[str, Path]] = None,  # pickled dict path
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if dct is None and dct_path is None:
            raise ValueError("Either dct or dct_path must be provided.")
        
        if dct is not None and dct_path is not None:
            raise ValueError("Only one of dct or dct_path must be provided.")
        
        if dct_path is not None:
            with open(dct_path, "rb") as file:
                dct = pickle.load(file)

        dct = {k: torch.tensor(v) for k, v in dct.items()}
        self.dct = dct

    def step(
        self, batch: dict, stage: Stage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack inputs
        ids = batch["identifier"]
        fp_true = batch["mol"]
        cands = batch["candidates"]
        batch_ptr = batch["batch_ptr"]

        # Read predicted fingerprints from dictionary
        fp_pred = torch.stack([self.dct[id] for id in ids]).to(fp_true.device)

        # Evaluation performance on fingerprint prediction (optional)
        self.evaluate_fingerprint_step(fp_true, fp_pred, stage=stage)

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands).to(fp_true.device)

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True, device=fp_true.device)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        # No training, so no optimizers
        return None
