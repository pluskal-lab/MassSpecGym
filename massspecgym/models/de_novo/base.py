import typing as T
from abc import ABC

from rdkit import Chem
import pulp
from myopic_mces.myopic_mces import MCES
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.aggregation import MeanMetric

from massspecgym.models.base import MassSpecGymModel


class DeNovoMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        top_ks: T.Iterable[int] = (1, 10),
        myopic_mces_kwargs: T.Optional[T.Mapping] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.top_ks = top_ks

        self.myopic_mces_kwargs = dict(
            ind=0,  # dummy index
            solver=pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
            threshold=15,  # MCES threshold
            solver_options=dict(msg=0)  # make ILP solver silent
        )
        self.myopic_mces_kwargs |= myopic_mces_kwargs or {}

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        metric_pref: str = ''
    ) -> None:
        self.evaluate_de_novo_step(
            outputs["mols_pred"],  # (bs, k) list of generated rdkit molecules or SMILES strings
            batch["mol"],  # (bs) list of ground truth SMILES strings
            metric_pref=metric_pref
        )

    def evaluate_de_novo_step(
        self,
        mols_pred: list[list[Chem.Mol | str]],
        mol_label: list[str],
        metric_pref: str = "",
    ) -> None:
        """
        Main evaluation method for the models for de novo molecule generation from mass spectra.

        Args:
            scores (list[list[Mol | str]]): (bs, k) list of generated rdkit molecules or SMILES
                strings
            labels (list[str]): (bs) list of ground truth SMILES strings
        """
        # Convert SMILEs from Mol objects if needed
        if not isinstance(mols_pred[0][0], str):
            mols_pred = [Chem.MolToSmiles(m) for ms in mols_pred for m in ms]

        for top_k in self.top_ks:
            # 1. Evaluate minimum common edge subgraph:
            # Calculate MCES distance between top-k predicted molecules and ground truth and
            # report the minimum distance. The minimum distances for each sample in the batch are
            # averaged across the epoch.
            min_mces_dists = []
            # Iterate over batch
            for mols_pred_sample, mol_label_sample in zip(mols_pred, mol_label):
                # Iterate over top-k predicted molecule samples
                dists = [
                    MCES(s1=mol_pred, s2=mol_label_sample, **self.myopic_mces_kwargs)[1]
                    for mol_pred in mols_pred_sample[:top_k]
                ]
                min_mces_dists.append(min(dists))
            self._update_metric(
                metric_pref + f"top_{top_k}_min_mces_dist",
                MeanMetric,
                (min_mces_dists,),
                batch_size=len(min_mces_dists),
                
            )

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
