import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.optimize
from typing import Tuple, Union

from massspecgym.simulation_utils.misc_utils import scatter_logsoftmax, scatter_l1normalize, scatter_l2normalize, scatter_reduce, EPS, th_setdiff1d, safelog, scatter_logsumexp, TOLERANCE_MIN_MZ

# rough approximations from NIST20
NCE_MEAN = 60.
NCE_STD = 40.
NCE_MAX = 200.
MZ_MAX = 1500.0
MZ_BIN_RES = 0.01
INTS_THRESH = 0.0

def get_ints_transform_func(ints_transform):
	""" method to get int transferom func

	Args:
		ints_transform (_type_): _description_

	Raises:
		ValueError: _description_

	Returns:
		_type_: _description_
	"""

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
	"""

	Args:
		ints_transform (_type_): _description_

	Raises:
		ValueError: _description_

	Returns:
		_type_: _description_
	"""
	if ints_transform == "log10":
		max_ints = float(np.log10(1000. + 1.))
		def _untransform_fn(x): return 10**x - 1.
	elif ints_transform == "log10t3":
		max_ints = float(np.log10(1000. + 1.) / 3.)
		def _untransform_fn(x): return 10**(3 * x) - 1.
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
	def _func(ints):
		ints = ints / (torch.max(ints) + EPS) * max_ints
		ints = _untransform_fn(ints)
		ints = torch.clamp(ints, min=0.)
		assert not torch.isnan(ints).any()
		return ints
	return _func

def transform_nce_to_ace(nce:float, mw:float, charge_factor:int = 1) -> float:
	"""get ace from given nce and mw
	   Absolute energy (eV) = (settling NCE) x (Isolation center) / (500 m/z) x (charge factor)
	Args:
		nce (float): normalized collision energy
		mw (float): Isolation center mw, most time it is the precursor ion mass

	Returns:
		float: absolute energy
	"""
	return nce  * mw  / 500 *  charge_factor 	


def batched_bin_func(
		mzs:torch.Tensor,
		ints:torch.Tensor,
		batch_idxs:torch.Tensor,
		mz_max:float,
		mz_bin_res:float,
		agg:str,
		sparse:bool=False,
		remove_prec_peaks:bool=False,
		prec_mzs:torch.Tensor=None) -> torch.Tensor:
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
        log_distance: bool=False) -> torch.Tensor:

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
    both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
    assert torch.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
    log_cos_sim = scatter_logsumexp(
        both_bin_logprobs,
        pred_bin_batch_idxs[pred_mask],
        dim_size=torch.max(true_bin_batch_idxs)+1
    )
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