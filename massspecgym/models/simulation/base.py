import typing as T
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric

from massspecgym.models.base import MassSpecGymModel, Stage
from massspecgym.simulation_utils.misc_utils import scatter_logl2normalize, scatter_logsumexp, safelog, \
    scatter_reduce
from massspecgym.simulation_utils.spec_utils import batched_bin_func, sparse_cosine_distance, \
    get_ints_transform_func, get_ints_untransform_func, batched_l1_normalize
from massspecgym.simulation_utils.nn_utils import build_lr_scheduler

class CosSimMetric(Metric):

    def __init__(self, transform_fn, mz_bin_res, mz_max, **kwargs):

        super().__init__(**kwargs)
        self.transform_fn = transform_fn
        self.mz_max = mz_max
        self.mz_bin_res = mz_bin_res
        self.add_state("cos_sims", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(
            self, 
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            true_mzs,
            true_logprobs,
            true_batch_idxs
        ):

        true_logprobs = self.transform_fn(true_logprobs, true_batch_idxs)
        pred_logprobs = self.transform_fn(pred_logprobs, pred_batch_idxs)
        batch_size = torch.max(true_batch_idxs)+1
        cos_sims = 1.-sparse_cosine_distance(
            pred_mzs=pred_mzs,
            pred_logprobs=pred_logprobs,
            pred_batch_idxs=pred_batch_idxs,
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch_idxs,
            mz_max=self.mz_max,
            mz_bin_res=self.mz_bin_res
        )
        self.cos_sims += torch.sum(cos_sims)
        self.count += batch_size
        
    def compute(self):

        return self.cos_sims.float() / self.count.float()

class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        optimizer_type,
        lr_schedule,
        lr_decay_rate,
        lr_warmup_steps,
        lr_decay_steps,
        ints_transform,
        mz_max,
        mz_bin_res,
        **kwargs
    ):
        super().__init__(
            # include lr and weight_decay
            **kwargs
        )
        self.optimizer_type = optimizer_type
        self.lr_schedule = lr_schedule
        self.lr_decay_rate = lr_decay_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.ints_transform = ints_transform
        self.mz_max = mz_max
        self.mz_bin_res = mz_bin_res

    @abstractmethod
    def _setup_model(self):

        pass

    def configure_optimizers(self):

        if self.optimizer_type == "adam":
            optimizer_cls = torch.optim.Adam
        elif self.optimizer_type == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif self.optimizer_type == "sgd":
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        optimizer = optimizer_cls(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        ret = {
            "optimizer": optimizer,
        }
        if self.lr_schedule:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, 
                decay_rate=self.lr_decay_rate, 
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }
        return ret

    def _setup_spec_fns(self):

        self.ints_transform_func = get_ints_transform_func(self.ints_transform)
        self.ints_untransform_func = get_ints_untransform_func(self.ints_transform)
        self.ints_normalize_func = batched_l1_normalize

    def _preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs):

        # transform
        spec_ints = spec_ints * 1000.
        spec_ints = self.ints_transform_func(spec_ints)
        # renormalize
        spec_ints = self.ints_normalize_func(
            spec_ints,
            spec_batch_idxs
        )
        spec_ints = safelog(spec_ints)
        return spec_mzs, spec_ints, spec_batch_idxs

    def _setup_loss_fn(self):

        def _loss_fn(
            true_mzs: torch.Tensor, 
            true_logprobs: torch.Tensor,
            true_batch_idxs: torch.Tensor,
            pred_mzs: torch.Tensor,
            pred_logprobs: torch.Tensor,
            pred_batch_idxs: torch.Tensor
        ):

            cos_dist = sparse_cosine_distance(
                true_mzs=true_mzs,
                true_logprobs=true_logprobs,
                true_batch_idxs=true_batch_idxs,
                pred_mzs=pred_mzs,
                pred_logprobs=pred_logprobs,
                pred_batch_idxs=pred_batch_idxs,
                mz_max=self.mz_max,
                mz_bin_res=self.mz_bin_res
            )
            return cos_dist

        self.loss_fn = _loss_fn

    # def get_cos_sim_fn(self, untransform: bool):

    #     def _cos_sim_fn(
    #         pred_mzs,
    #         pred_logprobs,
    #         pred_batch_idxs,
    #         true_mzs,
    #         true_logprobs,
    #         true_batch_idxs):

    #         if untransform:
    #             # untransform
    #             true_logprobs = safelog(self.ints_normalize_func(
    #                 self.ints_untransform_func(torch.exp(true_logprobs), true_batch_idxs), 
    #                 true_batch_idxs
    #             ))
    #             pred_logprobs = safelog(self.ints_normalize_func(
    #                 self.ints_untransform_func(torch.exp(pred_logprobs), pred_batch_idxs), 
    #                 pred_batch_idxs
    #             ))

    #         cos_sim = 1.-sparse_cosine_distance(
    #             pred_mzs=pred_mzs,
    #             pred_logprobs=pred_logprobs,
    #             pred_batch_idxs=pred_batch_idxs,
    #             true_mzs=true_mzs,
    #             true_logprobs=true_logprobs,
    #             true_batch_idxs=true_batch_idxs,
    #             mz_max=self.mz_max,
    #             mz_bin_res=self.mz_bin_res
    #         )

    #         return cos_sim
        
    #     return _cos_sim_fn

    def _setup_metric_kwargs(self):

        def transform_fn(logprobs, batch_idxs):
            logprobs = self.ints_untransform_func(torch.exp(logprobs), batch_idxs)
            logprobs = safelog(self.ints_normalize_func(logprobs, batch_idxs))
            return logprobs
        cos_sim_metric_kwargs = {
            "transform_fn": deepcopy(transform_fn),
            "mz_bin_res": self.mz_bin_res,
            "mz_max": self.mz_max
        }
        self.cos_sim_metric_kwargs = cos_sim_metric_kwargs

        transform_fn = lambda x, y: x
        cos_sim_obj_metric_kwargs = {
            "transform_fn": deepcopy(transform_fn),
            "mz_bin_res": self.mz_bin_res,
            "mz_max": self.mz_max
        }
        self.cos_sim_obj_metric_kwargs = cos_sim_obj_metric_kwargs

    def forward(self, **kwargs) -> dict:

        return self.model.forward(**kwargs)

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:

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
        mean_loss = torch.mean(loss)
        batch_size = torch.max(pred_batch_idxs)+1

        # Log loss
        # TODO: not sure if this batch_size param messes up running total
        self.log(
            stage.to_pref() + "loss_step",
            mean_loss,
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False
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
        self, outputs: T.Any, batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        """
        Compute evaluation metrics for the retrieval model based on the batch and corresponding predictions.
        This method will be used in the `on_train_batch_end`, `on_validation_batch_end`, since `on_test_batch_end` is
        overriden below.
        """
        self.evaluate_cos_similarity_step(
            pred_mzs=outputs["pred_mzs"],
            pred_logprobs=outputs["pred_logprobs"],
            pred_batch_idxs=outputs["pred_batch_idxs"],
            true_mzs=outputs["true_mzs"],
            true_logprobs=outputs["true_logprobs"],
            true_batch_idxs=outputs["true_batch_idxs"],
            stage=stage
        )

    def on_test_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int
    ) -> None:
        
        stage = Stage.TEST
        self.evaluate_cos_similarity_step(
            pred_mzs=outputs["pred_mzs"],
            pred_logprobs=outputs["pred_logprobs"],
            pred_batch_idxs=outputs["pred_batch_idxs"],
            true_mzs=outputs["true_mzs"],
            true_logprobs=outputs["true_logprobs"],
            true_batch_idxs=outputs["true_batch_idxs"],
            stage=stage
        )
        # self.evaluate_hit_rate_step(
        #     outputs["spec_pred"],
        #     batch["spec"],
        #     metric_pref=metric_pref
        # )

    def evaluate_cos_similarity_step(
        self,
        pred_mzs: torch.Tensor,
        pred_logprobs: torch.Tensor,
        pred_batch_idxs: torch.Tensor,
        true_mzs: torch.Tensor,
        true_logprobs: torch.Tensor,
        true_batch_idxs: torch.Tensor,
        stage: Stage
    ) -> None:
        
        batch_size = torch.max(true_batch_idxs).item()+1
        update_args = (
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            true_mzs,
            true_logprobs,
            true_batch_idxs
        )
        self._update_metric(
            name=stage.to_pref() + "cos_sim",
            metric_class=CosSimMetric,
            update_args=update_args,
            batch_size=batch_size,
            metric_kwargs=self.cos_sim_metric_kwargs,
            prog_bar=True,
            log=True,
            log_n_samples=False
        )
        self._update_metric(
            name=stage.to_pref() + "cos_sim_obj",
            metric_class=CosSimMetric,
            update_args=update_args,
            batch_size=batch_size,
            metric_kwargs=self.cos_sim_obj_metric_kwargs,
            prog_bar=True,
            log=True,
            log_n_samples=False
        )

    def evaluate_hit_rate_step(
        self,
        pred_mzs: torch.Tensor,
        pred_logprobs: torch.Tensor,
        pred_batch_idxs: torch.Tensor,
        true_mzs: torch.Tensor,
        true_logprobs: torch.Tensor,
        true_batch_idxs: torch.Tensor,
        stage: Stage
    ) -> None:
        """
        Evaulate Hit rate @ {1, 5, 20} (typically reported as Accuracy @ {1, 5, 20}).
        """
        
        raise NotImplementedError

