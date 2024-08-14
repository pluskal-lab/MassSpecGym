import typing as T
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics.retrieval import RetrievalHitRate

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

    def calculate(
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
        return cos_sims

    def update(
        self, 
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        true_mzs,
        true_logprobs,
        true_batch_idxs
    ):

        cos_sims = self.calculate(
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            true_mzs,
            true_logprobs,
            true_batch_idxs
        )
        self.cos_sims += torch.sum(cos_sims)
        self.count += torch.max(true_batch_idxs)+1
        
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
        do_retrieval,
        at_ks,
        **kwargs
    ):
        self.optimizer_type = optimizer_type
        self.lr_schedule = lr_schedule
        self.lr_decay_rate = lr_decay_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.ints_transform = ints_transform
        self.mz_max = mz_max
        self.mz_bin_res = mz_bin_res
        # for retrieval
        self.do_retrieval = do_retrieval
        self.at_ks = at_ks
        # call super class
        super().__init__(
            # include lr and weight_decay
            **kwargs
        )
        # call setup functions
        self._setup_model()
        self._setup_loss_fn()
        self._setup_spec_fns()
        self._setup_metric_kwargs()

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

    def get_retrieval_outputs(
        self, 
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        candidates_data,
        **kwargs
    ) -> dict:

        input_data = {}
        for k in candidates_data:
            input_data[k] = candidates_data[k]
        c_ptr = input_data["batch_ptr"]
        for k in ["precursor_mz", "adduct", "instrument_type", "collision_energy"]:
            if k in kwargs:
                assert not k in input_data, k
                input_data[k] = torch.repeat_interleave(kwargs[k],c_ptr)
        output_data = self.forward(**input_data)
        c_offset = torch.cumsum(c_ptr, dim=0)
        c_offset = torch.cat([
            torch.zeros((1,),dtype=c_offset.dtype,device=c_offset.device),
            c_offset
        ], dim=0)
        c_total = torch.sum(c_ptr)
        c_true_mzs, c_true_logprobs, c_true_batch_idxs = [], [], []
        for idx in range(c_ptr.shape[0]):
            mask = true_batch_idxs==idx
            ptr = c_ptr[idx]
            c_true_mzs.append(true_mzs[mask].repeat(ptr))
            c_true_logprobs.append(true_logprobs[mask].repeat(ptr))
            c_true_batch_idxs.append(torch.repeat_interleave(torch.arange(ptr,device=c_ptr.device),mask.sum()) + c_offset[idx])
        c_true_mzs = torch.cat(c_true_mzs, dim=0)
        c_true_logprobs = torch.cat(c_true_logprobs, dim=0)
        c_true_batch_idxs = torch.cat(c_true_batch_idxs, dim=0)
        assert output_data["pred_batch_idxs"].max() == c_true_batch_idxs.max()
        retrieval_score_metric = CosSimMetric(**self.cos_sim_metric_kwargs)
        c_scores = retrieval_score_metric.calculate(
            true_mzs=c_true_mzs,
            true_logprobs=c_true_logprobs,
            true_batch_idxs=c_true_batch_idxs,
            pred_mzs=output_data["pred_mzs"],
            pred_logprobs=output_data["pred_logprobs"],
            pred_batch_idxs=output_data["pred_batch_idxs"]
        )
        c_labels = input_data["labels"]
        assert c_scores.shape[0] == c_labels.shape[0] == c_total, (c_scores.shape, c_labels.shape, c_total)
        # prepare dictionary
        ret_out_d = {}
        ret_out_d["retrieval_scores"] = c_scores
        ret_out_d["retrieval_labels"] = c_labels
        ret_out_d["retrieval_batch_ptr"] = c_ptr
        return ret_out_d

    def step(
        self, 
        batch: dict, 
        stage: Stage = Stage.NONE
    ) -> dict:

        true_mzs, true_logprobs, true_batch_idxs = self._preproc_spec(
            batch["spec_mzs"],
            batch["spec_ints"],
            batch["spec_batch_idxs"]
        )

        out_d = self.forward(
            **batch
        )
        pred_mzs = out_d["pred_mzs"]
        pred_logprobs = out_d["pred_logprobs"]
        pred_batch_idxs = out_d["pred_batch_idxs"]
        try:
            loss = self.loss_fn(
                true_mzs=true_mzs,
                true_logprobs=true_logprobs,
                true_batch_idxs=true_batch_idxs,
                pred_mzs=pred_mzs,
                pred_logprobs=pred_logprobs,
                pred_batch_idxs=pred_batch_idxs
            )
        except RuntimeError as e:
            import pdb; pdb.set_trace()
        mean_loss = torch.mean(loss)
        batch_size = torch.max(pred_batch_idxs)+1

        # Log loss
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
            "true_batch_idxs": true_batch_idxs
        }

        if stage == Stage.TEST and self.do_retrieval:
            # retrieval
            assert "candidates_data" in batch, batch.keys()
            ret_out_d = self.get_retrieval_outputs(
                true_mzs=true_mzs,
                true_logprobs=true_logprobs,
                true_batch_idxs=true_batch_idxs,
                **batch
            )
            out_d.update(ret_out_d)

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
        if self.do_retrieval:
            self.evaluate_retrieval_step(
                scores=outputs["retrieval_scores"],
                labels=outputs["retrieval_labels"],
                batch_ptr=outputs["retrieval_batch_ptr"],
                stage=stage
            )

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

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        stage: Stage,
    ) -> None:
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
        assert stage == Stage.TEST, stage
        # Evaluate hitrate at different top-k values
        indexes = torch.arange(batch_ptr.size(0), device=batch_ptr.device)
        indexes = torch.repeat_interleave(indexes, batch_ptr)
        for at_k in self.at_ks:
            self._update_metric(
                stage.to_pref() + f"hit_rate@{at_k}",
                RetrievalHitRate,
                (scores, labels, indexes),
                batch_size=batch_ptr.size(0),
                metric_kwargs=dict(top_k=at_k),
            )

