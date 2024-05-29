import typing as T
from abc import ABC

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from massspecgym.models.base import MassSpecGymModel
from massspecgym.simulation_utils.misc_utils import scatter_logl2normalize, scatter_logsumexp, batched_bin_func, safelog


def get_batch_metric_reduce_fn(sample_weight):

    def _batch_metric_reduce(scores, weights):
        if sample_weight == "none":
            # ignore weights (uniform averaging)
            weights = th.ones_like(weights)
        return th.sum(scores * weights, dim=0) / th.sum(weights, dim=0)

    return _batch_metric_reduce


def sparse_cosine_distance(
        true_mzs: th.Tensor, 
        true_logprobs: th.Tensor,
        true_batch_idxs: th.Tensor,
        pred_mzs: th.Tensor,
        pred_logprobs: th.Tensor,
        pred_batch_idxs: th.Tensor,
        mz_max: float=1500.,
        mz_bin_res: float=0.01,
        log_distance: bool=False) -> th.Tensor:

    # sparse bin
    true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True
    )
    pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True
    )
    # l2 normalize
    true_bin_logprobs = scatter_logl2normalize(
        true_bin_logprobs,
        true_bin_batch_idxs
    )
    pred_bin_logprobs = scatter_logl2normalize(
        pred_bin_logprobs,
        pred_bin_batch_idxs
    )
    # dot product
    pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
    true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
    both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
    assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
    log_cos_sim = scatter_logsumexp(
        both_bin_logprobs,
        pred_bin_batch_idxs[pred_mask],
        dim_size=th.max(true_bin_batch_idxs)+1
    )
    if log_distance:
        cos_dist = th.log1p(-th.exp(log_cos_sim))
    else:
        cos_dist = 1.-th.exp(log_cos_sim)
    return cos_dist


class SimulationMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self._setup_model()
        self._setup_tolerance()
        self._setup_loss_fn()
        self._setup_spec_fns()
        self._setup_metric_fns()

    @abstractmethod
    def _setup_model(self):

        pass

    def _setup_tolerance(self):

        # set tolerance
        if self.hparams.loss_tolerance_rel is not None:
            self.tolerance = self.hparams.loss_tolerance_rel
            self.relative = True
            self.tolerance_min_mz = self.hparams.loss_tolerance_min_mz
        else:
            assert self.hparams.loss_tolerance_abs is not None
            self.tolerance = self.hparams.loss_tolerance_abs
            self.relative = False
            self.tolerance_min_mz = None

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

            cos_dist = sparse_cosine_distance(
                pred_mzs,
                pred_ints,
                pred_batch_idxs,
                true_mzs,
                true_ints,
                true_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )
            return train_reduce_fn(cos_dist, weights)

        def _eval_metric_fn(
            pred_mzs,
            pred_ints,
            pred_batch_idxs,
            true_mzs,
            true_ints,
            true_batch_idxs,
            weights):

            cos_dist = sparse_cosine_distance(
                pred_mzs,
                pred_ints,
                pred_batch_idxs,
                true_mzs,
                true_ints,
                true_batch_idxs,
                mz_max=self.hparams.mz_max,
                mz_bin_res=self.hparams.mz_bin_res
            )
            return eval_reduce_fn(cos_dist, weights)

        # need to name them like this for the lightning module to recognize them
        self.train_spec_cos_sim = _train_metric_fn
        self.val_spec_cos_sim = _eval_metric_fn
        self.test_spec_cos_sim = _eval_metric_fn

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
        pred_mzs: th.Tensor,
        pred_logprobs: th.Tensor,
        pred_batch_idxs: th.Tensor,
        true_mzs: th.Tensor,
        true_logprobs: th.Tensor,
        true_batch_idxs: th.Tensor,
        metric_pref: str,
    ) -> None:
        """
        Main evaluation method for the retrieval models. The retrieval step is evaluated by 
        computing the hit rate at different top-k values.

        Args:
            scores (th.Tensor): Concatenated scores for all candidates for all samples in the 
                batch
            labels (th.Tensor): Concatenated True/False labels for all candidates for all samples
                 in the batch
            batch_ptr (th.Tensor): Pointer to the start of each sample's candidates in the 
                concatenated tensors
        """
        
        # Cosine similarity between predicted and true fingerprints
        batch_size = th.max(pred_batch_idxs)+1
        self._update_metric(
            metric_pref + "spec_cos_sim",
            None, # must be initialized
            (pred_mzs, pred_logprobs, pred_batch_idxs, true_mzs, true_logprobs, true_batch_idxs),
            batch_size=batch_size
        )
