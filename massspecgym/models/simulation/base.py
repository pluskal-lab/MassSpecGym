import typing as T
from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from torchmetrics import Metric, MeanMetric
from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate

from massspecgym.models.base import MassSpecGymModel, Stage
from massspecgym.simulation_utils.misc_utils import safelog
from massspecgym.simulation_utils.spec_utils import sparse_cosine_distance, \
    get_ints_transform_func, get_ints_untransform_func, batched_l1_normalize, \
    sparse_jensen_shannon_similarity
from massspecgym.simulation_utils.nn_utils import build_lr_scheduler
from massspecgym.utils import batch_ptr_to_batch_idx
from torch_geometric.utils import unbatch

def get_cos_sim_fn(transform_fn, mz_max, mz_bin_res):

    def cos_sim_fn(
        true_mzs: torch.Tensor, 
        true_logprobs: torch.Tensor,
        true_batch_idxs: torch.Tensor,
        pred_mzs: torch.Tensor,
        pred_logprobs: torch.Tensor,
        pred_batch_idxs: torch.Tensor
    ) -> torch.Tensor:

        true_logprobs = transform_fn(true_logprobs, true_batch_idxs)
        pred_logprobs = transform_fn(pred_logprobs, pred_batch_idxs)
        cos_sims = 1.-sparse_cosine_distance(
            pred_mzs=pred_mzs,
            pred_logprobs=pred_logprobs,
            pred_batch_idxs=pred_batch_idxs,
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch_idxs,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res
        )
        return cos_sims

    return deepcopy(cos_sim_fn)

def get_js_sim_fn(transform_fn, mz_max, mz_bin_res):

    def js_sim_fn(
        true_mzs: torch.Tensor, 
        true_logprobs: torch.Tensor,
        true_batch_idxs: torch.Tensor,
        pred_mzs: torch.Tensor,
        pred_logprobs: torch.Tensor,
        pred_batch_idxs: torch.Tensor
    ) -> torch.Tensor:

        true_logprobs = transform_fn(true_logprobs, true_batch_idxs)
        pred_logprobs = transform_fn(pred_logprobs, pred_batch_idxs)
        js_sims = sparse_jensen_shannon_similarity(
            pred_mzs=pred_mzs,
            pred_logprobs=pred_logprobs,
            pred_batch_idxs=pred_batch_idxs,
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch_idxs,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res
        )
        return js_sims

    return deepcopy(js_sim_fn)

class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # call setup functions
        self._setup_model()
        self._setup_loss_fn()
        self._setup_spec_fns()
        self._setup_metric_fns()

    @abstractmethod
    def _setup_model(self):

        pass

    def configure_optimizers(self):

        if self.hparams.optimizer_type == "adam":
            optimizer_cls = torch.optim.Adam
        elif self.hparams.optimizer_type == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif self.hparams.optimizer_type == "sgd":
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        optimizer = optimizer_cls(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        ret = {
            "optimizer": optimizer,
        }
        if self.hparams.lr_schedule:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, 
                decay_rate=self.hparams.lr_decay_rate, 
                warmup_steps=self.hparams.lr_warmup_steps,
                decay_steps=self.hparams.lr_decay_steps,
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }
        return ret

    def _setup_spec_fns(self):

        self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
        self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
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
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )
            return cos_dist

        self.loss_fn = _loss_fn

    def _setup_metric_fns(self):

        # untransformed
        def no_transform_fn(logprobs, batch_idxs):
            probs = self.ints_untransform_func(torch.exp(logprobs), batch_idxs)
            logprobs = safelog(self.ints_normalize_func(probs, batch_idxs))
            return logprobs
        self.cos_sim_fn = get_cos_sim_fn(
            deepcopy(no_transform_fn), 
            self.hparams.mz_max, 
            self.hparams.mz_bin_res
        )
        self.js_sim_fn = get_js_sim_fn(
            deepcopy(no_transform_fn), 
            self.hparams.mz_max, 
            self.hparams.mz_bin_res
        )
        # sqrt transform
        def sqrt_transform_fn(logprobs, batch_idxs):
            probs = self.ints_untransform_func(torch.exp(logprobs), batch_idxs)
            probs = probs**0.5
            logprobs = safelog(self.ints_normalize_func(probs, batch_idxs))
            return logprobs
        self.cos_sim_sqrt_fn = get_cos_sim_fn(
            deepcopy(sqrt_transform_fn), 
            self.hparams.mz_max,
            self.hparams.mz_bin_res
        )
        # obj transform
        obj_transform_fn = lambda x, y: x
        self.cos_sim_obj_fn = get_cos_sim_fn(
            deepcopy(obj_transform_fn), 
            self.hparams.mz_max,
            self.hparams.mz_bin_res
        )

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
        # TODO: replace with unbatch?
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
        c_scores = self.cos_sim_fn(
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
        loss = self.loss_fn(
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch_idxs,
            pred_mzs=pred_mzs,
            pred_logprobs=pred_logprobs,
            pred_batch_idxs=pred_batch_idxs
        )
        mean_loss = torch.mean(loss)

        out_d = {
            "loss": mean_loss, 
            "pred_mzs": pred_mzs, 
            "pred_logprobs": pred_logprobs, 
            "pred_batch_idxs": pred_batch_idxs,
            "true_mzs": true_mzs,
            "true_logprobs": true_logprobs,
            "true_batch_idxs": true_batch_idxs
        }

        if stage == Stage.TEST and self.hparams.do_retrieval:
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

        batch_size = torch.max(outputs["true_batch_idxs"])+1
        
        # Log loss
        self.log(
            stage.to_pref() + "loss_step",
            outputs["loss"],
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False
        )

        if stage not in self.log_only_loss_at_stages:

            self.evaluate_similarity_step(
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
        
        self.on_batch_end(
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            stage=stage
        )

        if stage not in self.log_only_loss_at_stages and self.hparams.do_retrieval:
            
            metric_vals = {}
            metric_vals |= self.evaluate_retrieval_step(
                scores=outputs["retrieval_scores"],
                labels=outputs["retrieval_labels"],
                batch_ptr=outputs["retrieval_batch_ptr"],
                stage=stage
            )
            if self.df_test_path is not None:
                self._update_df_test(metric_vals)

    def evaluate_similarity_step(
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
        assert "cos_sim" in self.hparams.sim_metrics, self.hparams.sim_metrics
        for metric_name in self.hparams.sim_metrics:
            metric_fn = getattr(self, metric_name+"_fn")
            metric = metric_fn(
                true_mzs=true_mzs,
                true_logprobs=true_logprobs,
                true_batch_idxs=true_batch_idxs,
                pred_mzs=pred_mzs,
                pred_logprobs=pred_logprobs,
                pred_batch_idxs=pred_batch_idxs
            )
            self._update_metric(
                name=stage.to_pref() + metric_name,
                metric_class=MeanMetric,
                update_args=(metric,),
                batch_size=batch_size,
                prog_bar=True,
                log=True,
                log_n_samples=False,
                bootstrap=(stage == Stage.TEST)
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

        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # Evaluate hitrate at different top-k values
        indexes = batch_ptr_to_batch_idx(batch_ptr)
        scores = scores + 1e-5*torch.randn_like(scores)
        scores = unbatch(scores, indexes)
        labels = unbatch(labels, indexes)

        for at_k in self.hparams.at_ks:
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
                bootstrap=(stage == Stage.TEST)
            )
            metric_vals[metric_name] = hit_rates

        return metric_vals