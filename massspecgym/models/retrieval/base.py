import typing as T
from abc import ABC

import pandas as pd
import torch
from torchmetrics import CosineSimilarity, MeanMetric
from torchmetrics.functional.retrieval import retrieval_hit_rate
from torch_geometric.utils import unbatch

from massspecgym.models.base import MassSpecGymModel, Stage
import massspecgym.utils as utils


class RetrievalMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        at_ks: T.Iterable[int] = (1, 5, 20), 
        myopic_mces_kwargs: T.Optional[T.Mapping] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.at_ks = at_ks
        self.myopic_mces = utils.MyopicMCES(**(myopic_mces_kwargs or {}))

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        """
        Compute evaluation metrics for the retrieval model based on the batch and corresponding
        predictions.
        """
        self.log(
            f"{stage.to_pref()}loss",
            outputs['loss'],
            batch_size=batch['spec'].size(0),
            sync_dist=True,
            prog_bar=True,
        )
        if stage in self.log_only_loss_at_stages:
            return

        metric_vals = {}
        metric_vals |= self.evaluate_retrieval_step(
            outputs["scores"],
            batch["labels"],
            batch["batch_ptr"],
            stage=stage,
        )
        metric_vals |= self.evaluate_mces_at_1(
            outputs["scores"],
            batch["labels"],
            batch["smiles"],
            batch["candidates_smiles"],
            batch["batch_ptr"],
            stage=stage,
        )
        if stage == Stage.TEST and self.df_test_path is not None:
            self._update_df_test(metric_vals)

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        stage: Stage,
    ) -> dict[str, torch.Tensor]:
        """
        Main evaluation method for the retrieval models. The retrieval step is evaluated by 
        computing the hit rate at different top-k values.

        Args:
            scores (torch.Tensor): Concatenated scores for all candidates for all samples in the 
                batch
            labels (torch.Tensor): Concatenated True/False labels for all candidates for all samples
                 in the batch
            batch_ptr (torch.Tensor): Number of each sample's candidates in the concatenated tensors
        """
        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # Evaluate hitrate at different top-k values
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        for at_k in self.at_ks:
            hit_rates = []
            for scores_sample, labels_sample in zip(scores, labels):
                hit_rates.append(retrieval_hit_rate(scores_sample, labels_sample, top_k=at_k))
            hit_rates = torch.tensor(hit_rates, device=batch_ptr.device)

            metric_name = f"{stage.to_pref()}hit_rate@{at_k}"
            self._update_metric(
                metric_name,
                MeanMetric,
                (hit_rates,),
                batch_size=batch_ptr.size(0),
                bootstrap=stage == Stage.TEST
            )
            metric_vals[metric_name] = hit_rates

        return metric_vals

    def evaluate_mces_at_1(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        smiles: list[str],
        candidates_smiles: list[str],
        batch_ptr: torch.Tensor,
        stage: Stage,
    ) -> dict[str, torch.Tensor]:
        """
        TODO
        """
        if labels.sum() != len(smiles):
            raise ValueError("MCES@1 evaluation currently supports exactly 1 positive candidate per sample.")
        
        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # Get top-1 predicted molecules for each ground-truth sample
        smiles_pred_top_1 = []
        batch_ptr = torch.cumsum(batch_ptr, dim=0)
        for i, j in zip(torch.cat([torch.tensor([0], device=batch_ptr.device), batch_ptr]), batch_ptr):
            scores_sample = scores[i:j]
            top_1_idx = i + torch.argmax(scores_sample)
            smiles_pred_top_1.append(candidates_smiles[top_1_idx])

        # Calculate MCES distance between top-1 predicted molecules and ground truth
        mces_dists = [
            self.myopic_mces(sm, sm_pred)
            for sm, sm_pred in zip(smiles, smiles_pred_top_1)
        ]
        mces_dists = torch.tensor(mces_dists, device=scores.device)

        # Log
        metric_name = f"{stage.to_pref()}mces@1"
        self._update_metric(
            metric_name,
            MeanMetric,
            (mces_dists,),
            batch_size=len(mces_dists),
            bootstrap=stage == Stage.TEST
        )
        metric_vals[metric_name] = mces_dists

        return metric_vals

    def evaluate_fingerprint_step(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        stage: Stage,
    ) -> None:
        """
        Utility evaluation method to assess the quality of predicted fingerprints. This method is
        not a part of the necessary evaluation logic (not called in the `on_batch_end` method)
        since retrieval models are not bound to predict fingerprints.

        Args:
            y_true (torch.Tensor): [batch_size, fingerprint_size] tensor of true fingerprints
            y_pred (torch.Tensor): [batch_size, fingerprint_size] tensor of predicted fingerprints
        """
        # Cosine similarity between predicted and true fingerprints
        self._update_metric(
            f"{stage.to_pref()}fingerprint_cos_sim",
            CosineSimilarity,
            (y_pred, y_true),
            batch_size=y_true.size(0),
            metric_kwargs=dict(reduction="mean")
        )

    def test_step(
        self,
        batch: dict,
        batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = super().test_step(batch, batch_idx)
        
        # Get sorted candidate SMILES based on the predicted scores for each sample
        if self.df_test_path is not None:
            indexes = utils.batch_ptr_to_batch_idx(batch['batch_ptr'])
            scores = unbatch(outputs['scores'], indexes)
            candidates_smiles = utils.unbatch_list(batch['candidates_smiles'], indexes)
            sorted_candidate_smiles = []
            for scores_sample, candidates_smiles_sample in zip(scores, candidates_smiles):
                candidates_smiles_sample = [
                    x for _, x in sorted(zip(scores_sample, candidates_smiles_sample), reverse=True)
                ]
                sorted_candidate_smiles.append(candidates_smiles_sample)
            self._update_df_test({
                'identifier': batch['identifier'],
                'sorted_candidate_smiles': sorted_candidate_smiles
            })

        return outputs

    def on_test_epoch_end(self):
        # Save test data frame to disk
        if self.df_test_path is not None:
            df_test = pd.DataFrame(self.df_test)
            self.df_test_path.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_pickle(self.df_test_path)
