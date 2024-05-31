import typing as T
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import pytorch_lightning as pl

from massspecgym.models.base import MassSpecGymModel
from massspecgym.simulation_utils.misc_utils import scatter_logl2normalize, scatter_logsumexp, safelog
from massspecgym.simulation_utils.spec_utils import batched_bin_func, sparse_cosine_distance, \
    get_ints_transform_func, get_ints_untransform_func, batched_l1_normalize



class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(self, **kwargs):
        super().__init__()
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

    def _setup_spec_fns(self):

        self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
        self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
        self.ints_normalize_func = batched_l1_normalize

    def _preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs):

        # transform
        spec_ints = self.ints_transform_func(spec_ints)
        # renormalize
        spec_ints = self.ints_normalize_func(
            spec_ints,
            spec_batch_idxs
        )
        # log
        spec_ints = safelog(spec_ints)
        return spec_mzs, spec_ints, spec_batch_idxs

    @abstractmethod
    def _setup_loss_fn(self):

        pass

    def get_cos_sim_fn(self, sample_weight: bool, untransform: bool):

        def _cos_sim_fn(
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            weights):

            if untransform:
                # untransform
                true_logprobs = safelog(self.ints_normalize_func(
                    self.ints_untransform_func(torch.exp(true_logprobs)), 
                    true_batch_idxs
                ))
                pred_logprobs = safelog(self.ints_normalize_func(
                    self.ints_untransform_func(torch.exp(pred_logprobs)), 
                    pred_batch_idxs
                ))

            if not sample_weight:
                # ignore weights (uniform averaging)
                weights = torch.ones_like(weights)

            cos_sim = 1.-sparse_cosine_distance(
                pred_mzs=pred_mzs,
                pred_logprobs=pred_logprobs,
                pred_batch_idxs=pred_batch_idxs,
                true_mzs=true_mzs,
                true_logprobs=true_logprobs,
                true_batch_idxs=true_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )

            return cos_sim
        
        return _cos_sim_fn

    def get_batch_metric_reduce_fn(self, sample_weight: bool):

        def _batch_metric_reduce(scores, weights, return_weight=False):
            if not sample_weight:
                # ignore weights (uniform averaging)
                weights = torch.ones_like(weights)
            w_total = torch.sum(weights, dim=0)
            w_mean_score = torch.sum(scores * weights, dim=0) / w_total
            if return_weight:
                return w_mean_score, w_total
            else:
                return w_mean_score

        return _batch_metric_reduce

    def _setup_metric_fns(self):

        self.train_reduce_fn = self.get_batch_metric_reduce_fn(self.hparams.train_sample_weight)
        self.eval_reduce_fn = self.get_batch_metric_reduce_fn(self.hparams.eval_sample_weight)

        train_cos_sim_fn = self.get_cos_sim_fn(
            sample_weight=self.hparams.train_sample_weight,
            untransform=False
        )
        train_cos_sim_obj_fn = self.get_cos_sim_fn(
            sample_weight=self.hparams.eval_sample_weight,
            untransform=True
        )
        eval_cos_sim_fn = self.get_cos_sim_fn(
            sample_weight=self.hparams.train_sample_weight,
            untransform=False
        )
        eval_cos_sim_obj_fn = self.get_cos_sim_fn(
            sample_weight=self.hparams.eval_sample_weight,
            untransform=True
        )
        # aliases
        self.train_spec_cos_sim = train_cos_sim_fn
        self.train_spec_cos_sim_obj = train_cos_sim_obj_fn
        self.val_spec_cos_sim = eval_cos_sim_fn
        self.val_spec_cos_sim_obj = eval_cos_sim_obj_fn
        self.test_spec_cos_sim = eval_cos_sim_fn
        self.test_spec_cos_sim_obj = eval_cos_sim_obj_fn

    def forward(self, **kwargs) -> dict:

        return self.model.forward(**kwargs)

    def step(self, batch: dict, metric_pref: str = "") -> dict:

        true_mzs, true_logprobs, true_batch_idxs = self._preproc_spec(
            batch["spec_mzs"],
            batch["spec_ints"],
            batch["spec_batch_idxs"]
        )
        out_d = self.model.forward(
            **batch
        )
        pred_mzs = out_d["pred_mzs"]
        pred_logprobs = out_d["pred_logprobs"]
        pred_batch_idxs = out_d["pred_batch_idxs"]
        loss = self.loss_fn(
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch_idxs,
            pred_mzs=pred_mzs,
            pred_logprobs=pred_logprobs,
            pred_batch_idxs=pred_batch_idxs
        )
        reduce_fn = self.train_reduce_fn if metric_pref == "train_" else self.eval_reduce_fn
        mean_loss, total_weight = reduce_fn(loss, batch["weight"], return_weight=True)
        batch_size = torch.max(pred_batch_idxs)+1

        # # little trick to work with automatic batch averaging
        # scaled_loss = loss * (batch_size / total_weight)

        # Log loss
        # TODO: not sure if this batch_size param messes up running total
        self.log(
            metric_pref + "loss_step",
            mean_loss,
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
        )

        out_d = {
            "loss": mean_loss, 
            "pred_mzs": pred_mzs, 
            "pred_logprobs": pred_logprobs, 
            "pred_batch_idxs": pred_batch_idxs,
            "true_mzs": true_mzs,
            "true_logprobs": true_logprobs,
            "true_batch_idxs": true_batch_idxs}

        return out_d

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, metric_pref: str = ""
    ) -> None:
        """
        """

        self.evaluate_simulation_step(
            pred_mzs=outputs["pred_mzs"],
            pred_logprobs=outputs["pred_logprobs"],
            pred_batch_idxs=outputs["pred_batch_idxs"],
            true_mzs=outputs["true_mzs"],
            true_logprobs=outputs["true_logprobs"],
            true_batch_idxs=outputs["true_batch_idxs"],
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
        weight: torch.Tensor,
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
        
        pass
        # print(metric_pref + "spec_cos_sim_obj")
        # batch_size = torch.max(pred_batch_idxs)+1
        # self._update_metric(
        #     metric_pref + "spec_cos_sim_obj",
        #     None, # must be initialized
        #     (pred_mzs, pred_logprobs, pred_batch_idxs, true_mzs, true_logprobs, true_batch_idxs, weight),
        #     batch_size=batch_size
        # )
        # self._update_metric(
        #     metric_pref + "spec_cos_sim",
        #     None, # must be initialized
        #     (pred_mzs, pred_logprobs, pred_batch_idxs, true_mzs, true_logprobs, true_batch_idxs, weight),
        #     batch_size=batch_size
        # )
