import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.optimize
from typing import Tuple, Union

from massspecgym.simulation_utils.misc_utils import (
    scatter_logsoftmax, 
    scatter_l1normalize, 
    scatter_l2normalize, 
    scatter_reduce, 
    EPS, 
    th_setdiff1d, 
    safelog, 
    scatter_logsumexp, 
    TOLERANCE_MIN_MZ,
    scatter_logl2normalize,
    LOG_ZERO
)
# rough approximations from NIST20
NCE_MEAN = 60.
NCE_STD = 40.
NCE_MAX = 200.
MZ_MAX = 1500.0
MZ_BIN_RES = 0.01
INTS_THRESH = 0.0
LN_2 = float(np.log(2.))

def get_ints_transform_func(ints_transform):

    _func = None
    if ints_transform == "log10":
        _func = lambda ints: torch.log10(ints+1.)
    elif ints_transform == "log10t3":
        _func = lambda ints: torch.log10(ints/3.+1.)
    elif ints_transform == "loge":
        _func = lambda ints: torch.log(ints+1.)
    elif ints_transform == "sqrt":
        _func = lambda ints: torch.sqrt(ints)
    elif ints_transform == "none":
        _func = lambda ints: ints
    else:
        raise ValueError("Invalid ints_transform: {}".format(ints_transform))
    return _func


def get_ints_untransform_func(ints_transform):

    if ints_transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        def _untransform_fn(x): return 10**x - 1.
    elif ints_transform == "log10t3":
        max_ints = float(np.log10(1000. / 3. + 1.))
        def _untransform_fn(x): return 3. * (10**x - 1.)
    elif ints_transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        def _untransform_fn(x): return torch.exp(x) - 1.
    elif ints_transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        def _untransform_fn(x): return x**2
    elif ints_transform == "none":
        max_ints = 1000.
        def _untransform_fn(x): return x
    else:
        raise ValueError("invalid transform")
    def _func(ints, batch_idxs):
        old_max_ints = scatter_reduce(
            ints,
            batch_idxs,
            "amax",
            default=0.,
            include_self=False
        )
        ints = ints / (old_max_ints[batch_idxs] + EPS) * max_ints
        ints = _untransform_fn(ints)
        ints = torch.clamp(ints, min=0.)
        assert not torch.isnan(ints).any()
        return ints
    return _func


def batched_bin_func(
    mzs:torch.Tensor,
    ints:torch.Tensor,
    batch_idxs:torch.Tensor,
    mz_max:float,
    mz_bin_res:float,
    agg:str,
    sparse:bool=False,
    remove_prec_peaks:bool=False,
    prec_mzs:torch.Tensor=None
) -> torch.Tensor:
    """method to get binned spectra for batch

    Args:
        mzs (torch.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.  
        ints (torch.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.  
        batch_idxs (torch.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor. 
        mz_max (float): max mz value allowed
        mz_bin_res (float): bin size
        sum_ints (bool): flag for sum intensities within the bin, else take max
        sparse (bool, optional): flag to use sparse  method. Defaults to False.

    Returns:
        _type_: binned spectra
    """

    if mzs.shape[0] == 0:
        import pdb; pdb.set_trace()
    assert torch.max(mzs) < mz_max, (torch.max(mzs),mz_max)
    batch_size = torch.max(batch_idxs)+1
    bins = torch.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype)
    num_bins = bins.shape[0]
    bin_idxs = torch.searchsorted(bins,mzs,side="right")
    bin_offsets = (torch.arange(batch_size,device=mzs.device)*num_bins)[batch_idxs]
    bin_idxs = bin_idxs + bin_offsets
    if remove_prec_peaks:
        assert prec_mzs is not None
        assert torch.max(prec_mzs) <  mz_max, (torch.max(prec_mzs),mz_max)
        prec_mz_bin_idxs = torch.searchsorted(bins,prec_mzs,side="right")
        prec_mz_bin_offsets = torch.arange(batch_size,device=mzs.device)*num_bins
        prec_mz_bin_idxs = prec_mz_bin_idxs + prec_mz_bin_offsets
        prec_ints_mask = torch.isin(bin_idxs,prec_mz_bin_idxs)
        ints = ints*(1-prec_ints_mask.float())
    if sparse:
        un_bin_idxs, un_bin_idxs_rev = torch.unique(bin_idxs,return_inverse=True)
        new_bin_idxs = torch.arange(un_bin_idxs.shape[0],device=un_bin_idxs.device)
        if agg in ["sum","amax"]:
            un_bin_ints = scatter_reduce(
                src=ints,
                index=new_bin_idxs[un_bin_idxs_rev],
                reduce=agg,
                dim_size=new_bin_idxs.shape[0]
            )
        else:
            assert agg == "lse", agg
            un_bin_ints = scatter_logsumexp(
                ints,
                new_bin_idxs[un_bin_idxs_rev],
                dim_size=new_bin_idxs.shape[0]
            )
        un_bin_batch_idxs = un_bin_idxs // num_bins
        return un_bin_idxs, un_bin_ints, un_bin_batch_idxs
    else:
        if agg in ["sum","amax"]:
            bin_spec = scatter_reduce(
                src=ints,
                index=bin_idxs,
                reduce=agg,
                dim_size=num_bins*batch_size
            )
        else:
            assert agg == "lse", agg
            bin_spec = scatter_logsumexp(
                ints,
                bin_idxs,
                dim_size=num_bins*batch_size
            )
        bin_spec = bin_spec.reshape(batch_size,num_bins)
        if agg in ["sum","amax"] and torch.any(torch.all(bin_spec==0.,dim=1)):
            print("> warning: bin_spec is all zeros!")
            mask = torch.zeros_like(bin_spec,dtype=torch.bool)
            mask[:,0] = 1.
            mask = mask*torch.all(bin_spec==0.,dim=1,keepdim=True)
            bin_spec = bin_spec + mask.float()
        return bin_spec

def sparse_cosine_distance(
    true_mzs: torch.Tensor, 
    true_logprobs: torch.Tensor,
    true_batch_idxs: torch.Tensor,
    pred_mzs: torch.Tensor,
    pred_logprobs: torch.Tensor,
    pred_batch_idxs: torch.Tensor,
    mz_max: float,
    mz_bin_res: float,
    log_distance: bool=False
) -> torch.Tensor:

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
    pred_mask = torch.isin(pred_bin_idxs, true_bin_idxs)
    true_mask = torch.isin(true_bin_idxs, pred_bin_idxs)
    batch_size = torch.max(true_bin_batch_idxs)+1
    if torch.any(pred_mask):
        both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
        assert torch.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
        log_cos_sim = scatter_logsumexp(
            both_bin_logprobs,
            pred_bin_batch_idxs[pred_mask],
            dim_size=batch_size
        )
    else:
        # cosine similarities are all zero
        log_cos_sim = torch.full(
            size=(batch_size,),
            fill_value=LOG_ZERO(pred_bin_logprobs.dtype),
            dtype=pred_bin_logprobs.dtype,
            device=pred_bin_logprobs.device
        )
        # involve pred_logprobs to keep gradient
        log_cos_sim = log_cos_sim + 0.*torch.mean(pred_bin_logprobs,dim=0)
    if log_distance:
        cos_dist = torch.log1p(-torch.exp(log_cos_sim))
    else:
        cos_dist = 1.-torch.exp(log_cos_sim)
    return cos_dist


def batched_l1_normalize(ints, batch_idxs):

    ints = scatter_l1normalize(
        ints,
        batch_idxs
    )
    return ints


def js_sim_helper(
    true_bin_idxs,
    true_bin_ints,
    true_bin_batch_idxs,
    pred_bin_idxs,
    pred_bin_ints,
    pred_bin_batch_idxs
):
    
    batch_size = torch.max(true_bin_batch_idxs)+1
    # l1 normalize
    true_bin_ints = scatter_l1normalize(
        true_bin_ints,
        true_bin_batch_idxs
    )
    pred_bin_ints = scatter_l1normalize(
        pred_bin_ints,
        pred_bin_batch_idxs
    )
    # union distribution
    union_bin_idxs, union_bin_idxs_rev = torch.unique(torch.cat([true_bin_idxs,pred_bin_idxs],dim=0),return_inverse=True)
    union_bin_ints = scatter_reduce(
        src=0.5*torch.cat([true_bin_ints,pred_bin_ints],dim=0),
        index=union_bin_idxs_rev,
        reduce="sum",
        dim_size=union_bin_idxs.shape[0]
    )
    # kl1
    kl1_union_bin_ints = union_bin_ints[union_bin_idxs_rev[:true_bin_idxs.shape[0]]]
    kl1 = scatter_reduce(
        true_bin_ints * (safelog(true_bin_ints) - safelog(kl1_union_bin_ints)),
        true_bin_batch_idxs,
        reduce="sum",
        dim_size=batch_size
    )
    # kl2
    kl2_union_bin_ints = union_bin_ints[union_bin_idxs_rev[true_bin_idxs.shape[0]:]]
    kl2 = scatter_reduce(
        pred_bin_ints * (safelog(pred_bin_ints) - safelog(kl2_union_bin_ints)),
        pred_bin_batch_idxs,
        reduce="sum",
        dim_size=batch_size
    )
    # jss
    jss = LN_2-0.5*(kl1+kl2)
    return jss

def sparse_jensen_shannon_similarity(
    true_mzs: torch.Tensor, 
    true_logprobs: torch.Tensor,
    true_batch_idxs: torch.Tensor,
    pred_mzs: torch.Tensor,
    pred_logprobs: torch.Tensor,
    pred_batch_idxs: torch.Tensor,
    mz_max: float,
    mz_bin_res: float
) -> torch.Tensor:
    
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

    jss = js_sim_helper(
        true_bin_idxs,
        true_bin_logprobs.exp(),
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_bin_logprobs.exp(),
        pred_bin_batch_idxs
    )

    return jss