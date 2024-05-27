import typing as T
from abc import ABC

from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import pulp
from myopic_mces.myopic_mces import MCES
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.aggregation import MeanMetric

from massspecgym.models.base import MassSpecGymModel
from massspecgym.utils import morgan_fp, mol_to_inchi_key


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
        mol_true: list[str],
        metric_pref: str = "",
    ) -> None:
        """
        Main evaluation method for the models for de novo molecule generation from mass spectra.

        Args:
            mols_pred (list[list[Mol | str]]): (bs, k) list of generated rdkit molecules or SMILES
                strings
            mol_true (list[str]): (bs) list of ground-truth SMILES strings
        """
        # Get SMILES and RDKit molecule objects for all predictions
        if isinstance(mols_pred[0][0], str):  # SMILES passed
            smiles_pred = mols_pred
            mols_pred = [[Chem.MolFromSmiles(sm) for sm in sms] for sms in mols_pred]
        else:  # RDKit molecules passed
            smiles_pred = [[Chem.MolToSmiles(m) for m in ms] for ms in mols_pred]

        # Get RDKit molecule objects for ground truth
        smile_true = mol_true
        mol_true = [Chem.MolFromSmiles(sm) for sm in mol_true]

        # Evaluate top-k metrics
        for top_k in self.top_ks:
            # Get top-k predicted molecules for each ground-truth sample
            smiles_pred_top_k = [smiles_pred_sample[:top_k] for smiles_pred_sample in smiles_pred]
            mols_pred_top_k = [mols_pred_sample[:top_k] for mols_pred_sample in mols_pred]

            # 1. Evaluate minimum common edge subgraph:
            # Calculate MCES distance between top-k predicted molecules and ground truth and
            # report the minimum distance. The minimum distances for each sample in the batch are
            # averaged across the epoch.
            min_mces_dists = []
            # Iterate over batch
            for preds, true in zip(smiles_pred_top_k, smile_true):
                # Iterate over top-k predicted molecule samples
                dists = [MCES(s1=true, s2=pred, **self.myopic_mces_kwargs)[1] for pred in preds]
                min_mces_dists.append(min(dists))
            self._update_metric(
                metric_pref + f"top_{top_k}_min_mces_dist",
                MeanMetric,
                (min_mces_dists,),
                batch_size=len(min_mces_dists),
            )

            # 2. Evaluate Tanimoto similarity:
            # Calculate Tanimoto similarity between top-k predicted molecules and ground truth and
            # report the maximum similarity. The maximum similarities for each sample in the batch
            # are averaged across the epoch.
            fps_pred_top_k = [[morgan_fp(m, to_np=False) for m in ms] for ms in mols_pred_top_k]
            fp_true = [morgan_fp(m, to_np=False) for m in mol_true]            
            max_tanimoto_sims = []
            # Iterate over batch
            for preds, true in zip(fps_pred_top_k, fp_true):
                # Iterate over top-k predicted molecule samples
                sims = [TanimotoSimilarity(true, pred) for pred in preds]
                max_tanimoto_sims.append(max(sims))
            self._update_metric(
                metric_pref + f"top_{top_k}_max_tanimoto_sim",
                MeanMetric,
                (max_tanimoto_sims,),
                batch_size=len(max_tanimoto_sims),
            )

            # 3. Evaluate exact match (accuracy):
            # Calculate if the ground truth molecule is in the top-k predicted molecules and report
            # the average across the epoch.
            in_top_k = [
                mol_to_inchi_key(true) in [mol_to_inchi_key(pred) for pred in preds]
                for true, preds in zip(mol_true, mols_pred_top_k)
            ]
            self._update_metric(
                metric_pref + f"top_{top_k}_accuracy",
                MeanMetric,
                (in_top_k,),
                batch_size=len(in_top_k)
            )
