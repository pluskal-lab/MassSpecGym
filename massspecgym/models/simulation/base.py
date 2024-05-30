import typing as T
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import pytorch_lightning as pl

from massspecgym.models.base import MassSpecGymModel
from massspecgym.simulation_utils.misc_utils import scatter_logl2normalize, scatter_logsumexp, safelog
from massspecgym.simulation_utils.spec_utils import batched_bin_func, sparse_cosine_distance


def get_batch_metric_reduce_fn(sample_weight):

    def _batch_metric_reduce(scores, weights, return_weight=False):
        if sample_weight == "none":
            # ignore weights (uniform averaging)
            weights = torch.ones_like(weights)
        w_total = torch.sum(weights, dim=0)
        w_mean_score = torch.sum(scores * weights, dim=0) / w_total
        if return_weight:
            return w_mean_score, w_total
        else:
            return w_mean_score

    return _batch_metric_reduce


class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self._setup_model()
        # self._setup_tolerance()
        self._setup_loss_fn()
        self._setup_spec_fns()
        self._setup_metric_fns()

    @abstractmethod
    def _setup_model(self):

        pass

    # def _setup_tolerance(self):

    #     # set tolerance
    #     if self.hparams.loss_tolerance_rel is not None:
    #         self.tolerance = self.hparams.loss_tolerance_rel
    #         self.relative = True
    #         self.tolerance_min_mz = self.hparams.loss_tolerance_min_mz
    #     else:
    #         assert self.hparams.loss_tolerance_abs is not None
    #         self.tolerance = self.hparams.loss_tolerance_abs
    #         self.relative = False
    #         self.tolerance_min_mz = None

    @abstractmethod
    def _setup_loss_fn(self):

        pass

    @abstractmethod
    def _setup_metric_fns(self):

        train_reduce_fn = get_batch_metric_reduce_fn(self.hparams.train_sample_weight)
        eval_reduce_fn = get_batch_metric_reduce_fn(self.hparams.eval_sample_weight)

        def _train_metric_fn(
            pred_mzs,
            pred_ints,
            pred_batch_idxs,
            true_mzs,
            true_ints,
            true_batch_idxs,
            weights):

            cos_sim = 1.-sparse_cosine_distance(
                pred_mzs,
                pred_ints,
                pred_batch_idxs,
                true_mzs,
                true_ints,
                true_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )
            return train_reduce_fn(cos_sim, weights)

        def _eval_metric_fn(
            pred_mzs,
            pred_ints,
            pred_batch_idxs,
            true_mzs,
            true_ints,
            true_batch_idxs,
            weights):

            cos_sim = 1.-sparse_cosine_distance(
                pred_mzs,
                pred_ints,
                pred_batch_idxs,
                true_mzs,
                true_ints,
                true_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )
            return eval_reduce_fn(cos_sim, weights)

        # need to name them like this for the lightning module to recognize them
        self.train_spec_cos_sim = _train_metric_fn
        self.val_spec_cos_sim = _eval_metric_fn
        self.test_spec_cos_sim = _eval_metric_fn
        self.train_reduce_fn = train_reduce_fn
        self.eval_reduce_fn = eval_reduce_fn

    def step(self, batch: dict, metric_pref: str = "") -> dict:

        pred_mzs, pred_logprobs, pred_batch_idxs = self.model(
            **batch
        )
        loss = self.loss_fn(
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            batch["spec_mzs"],
            safelog(batch["spec_ints"]),
            batch["spec_batch_idxs"]
        )
        reduce_fn = self.train_reduce_fn if metric_pref == "train_" else self.eval_reduce_fn
        mean_loss, total_weight = reduce_fn(loss, batch["weight"], return_weight=True)
        batch_size = torch.max(pred_batch_idxs)+1

        # little trick to work with automatic batch averaging
        scaled_loss = loss * (batch_size / total_weight)

        # Log loss
        # TODO: not sure if this batch_size param messes up running total
        self.log(
            metric_pref + "loss_step",
            scaled_loss,
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
        )

        return {"loss": scaled_loss, "pred_mzs": pred_mzs, "pred_ints": pred_logprobs, "pred_batch_idxs": pred_batch_idxs}

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, metric_pref: str = ""
    ) -> None:
        """
        """

        self.evaluate_simulation_step(
            pred_mzs=outputs["pred_mzs"],
            pred_logprobs=outputs["pred_logprobs"],
            pred_batch_idxs=outputs["pred_batch_idxs"],
            true_mzs=batch["true_mzs"],
            true_logprobs=safelog(batch["true_ints"]),
            true_batch_idxs=batch["true_batch_idxs"],
            weight=batch["weight"],
            metric_pref=metric_pref,
        )

    def evaluate_simulation_step(
        self,
        pred_mzs: torch.Tensor,
        pred_logprobs: torch.Tensor,
        pred_batch_idxs: torch.Tensor,
        true_mzs: torch.Tensor,
        true_logprobs: torch.Tensor,
        true_batch_idxs: torch.Tensor,
        metric_pref: str,
    ) -> None:
        """
        Main evaluation method for the retrieval models. The retrieval step is evaluated by 
        computing the hit rate at different top-k values.

        Args:
            scores (torch.Tensor): Concatenated scores for all candidates for all samples in the 
                batch
            labels (torch.Tensor): Concatenated True/False labels for all candidates for all samples
                 in the batch
            batch_ptr (torch.Tensor): Pointer to the start of each sample's candidates in the 
                concatenated tensors
        """
        
        # Cosine similarity between predicted and true fingerprints
        batch_size = torch.max(pred_batch_idxs)+1
        self._update_metric(
            metric_pref + "spec_cos_sim",
            None, # must be initialized
            (pred_mzs, pred_logprobs, pred_batch_idxs, true_mzs, true_logprobs, true_batch_idxs),
            batch_size=batch_size
        )
