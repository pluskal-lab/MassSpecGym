import typing as T
from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex

from massspecgym.models.base import MassSpecGymModel


class DeNovoMassSpecGymModel(MassSpecGymModel, ABC):

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        metric_pref: str = ''
    ) -> None:
        self.evaluate_retrieval_step(
            outputs["mols_pred"],  # (bs, k) list of generated rdkit molecules
            batch["mol"],  # (bs, ) list of rdkit molecules
            metric_pref=metric_pref
        )

    def evaluate_de_novo_step(
        self,
        mols_pred: torch.Tensor,
        mol_label: torch.Tensor,
        batch_ptr: torch.Tensor,
        metric_pref: str = "",
    ) -> None:
        
        # TODO
        # For k in (1, 10)
            # A
            #   https://github.com/AlBi-HHU/myopic-mces/blob/main/src/myopic_mces/myopic_mces.py#L16
            #   1. mol_label agains all mols_pred
            #   MCES(ind=1, s1, s2, threshold=15, solver='CPLEX_CMD')
            #   2. Choose min dist [:k]
            #   3. Report average mean across epoch

            # B
            # Same as A but with Tanimoto similarity instead of MCES (max instead of min)

            # C
            # utils.mol_to_inchi_key(mol_label) in utils.mol_to_inchi_key(mols_pred[:k])


        pass

        # self._update_metric(
        #     metric_pref + "tanimoto",
        #     BinaryJaccardIndex,
        #     (y_pred, y_true),
        #     batch_size=y_true.size(0),
        #     metric_kwargs=dict(reduction="mean"),
        # )

        # torch.from_numpy(mol_label)
        # from torch import tensor
        
        # target = tensor([1, 1, 0, 0])
        # preds = tensor([0, 1, 0, 0])
        # metric = BinaryJaccardIndex()
        # metric(preds, target)
