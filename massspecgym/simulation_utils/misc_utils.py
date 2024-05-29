import time
from functools import wraps
from distutils.util import strtobool
from contextlib import contextmanager
import numpy as np
import torch
import os
import joblib
import tqdm
import pandas as pd
import threading
import torch_geometric as pyg
# import dgl
from collections import defaultdict
import sys

TQDM_DISABLE = False
PPM = 1/1000000
EPS = 1e-9
LOG_HALF = float(np.log(0.5))
LOG_TWO = float(np.log(2.0))
LOG_ZERO_FP32 = float(torch.finfo(torch.float32).min)
LOG_ZERO_FP16 = float(torch.finfo(torch.float16).min)
MAX_CROSS_ENTROPY = 1e19
TOLERANCE_MIN_MZ = 200.0

def LOG_ZERO(dtype):
    if dtype == torch.float32:
        return LOG_ZERO_FP32
    elif dtype == torch.float16:
        return LOG_ZERO_FP16
    else:
        raise ValueError(dtype)

def timeit(func):
    # adapted from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def booltype(x):
    return bool(strtobool(x))

def none_or_nan(thing):
    if thing is None:
        return True
    elif isinstance(thing,float) and np.isnan(thing):
        return True
    elif pd.isnull(thing):
        return True
    elif isinstance(thing,str) and thing == "":
        return True
    else:
        return False

@contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextmanager
def th_temp_seed(seed):
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)

def flatten_lol(lol):
    return [item for sublist in lol for item in sublist]

def wandb_symlink(run_dir,wandb_symlink_dp,job_id):
    symlink_dst = os.path.join(wandb_symlink_dp,str(job_id))
    symlink_src = os.path.split(os.path.abspath(run_dir))[0]
    if os.path.islink(symlink_dst):
        os.unlink(symlink_dst)
    os.symlink(symlink_src,symlink_dst)

def list_str2float(str_list):
    return [float(str_item) for str_item in str_list]

# https://stackoverflow.com/a/58936697/6937913
@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call

class TimeoutException(Exception): pass

# @contextmanager
# def time_limit(seconds):
# 	if seconds is None:
# 		yield
# 	def signal_handler(signum, frame):
# 		raise TimeoutException(f"Timed out! ({seconds} seconds)")
# 	signal.signal(signal.SIGALRM, signal_handler)
# 	signal.alarm(seconds)
# 	try:
# 		yield
# 	finally:
# 		signal.alarm(0)

def timeout_func(func, args=None, kwargs=None, timeout=30, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.
    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    """
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
            self.exc_info = (None, None, None)
        def run(self):
            try:
                self.result = func(*(args or ()), **(kwargs or {}))
            except Exception as err:
                self.exc_info = sys.exc_info()
        def suicide(self):
            raise TimeoutException(
                "{0} timeout (taking more than {1} sec)".format(func.__name__, timeout)
            )
    it = InterruptableThread()
    it.start()
    it.join(timeout)
    if it.exc_info[0] is not None:
        a, b, c = it.exc_info
        raise Exception(a, b, c)  # communicate that to caller
    if it.is_alive():
        it.suicide()
        raise RuntimeError
    else:
        return it.result

def my_tqdm(*args,**kwargs):
    return tqdm.tqdm(*args,**kwargs,disable=TQDM_DISABLE)

def get_tensor_memory_usage(tensor):
    return tensor.nelement()*tensor.element_size()

def get_tensor_dict_memory_usage(**tensor_dict):
    total_memory = 0
    for k,v in tensor_dict.items():
        if isinstance(v,torch.Tensor):
            total_memory += get_tensor_memory_usage(v)
    return total_memory

def get_pyg_memory_usage(pyg_graph):
    return pyg.profile.get_data_size(pyg_graph)

def scatter_masked_softmax(logits,mask,subset_idxs,mask_logprob=None,log=True):
    
    if mask_logprob is None:
        mask_logprob = LOG_ZERO(logits.dtype)	
    # calculate appropriate mask value
    with torch.no_grad():
        c = scatter_masked_logsumexp(logits,mask,subset_idxs)
        lm = torch.gather(
            input=c,
            index=subset_idxs,
            dim=0
        )
        mask_value = mask_logprob + lm
    # apply mask
    masked_logits = mask*logits + (1-mask)*mask_value
    # normalize
    masked_logits = scatter_logsoftmax(masked_logits,subset_idxs)
    if not log:
        # exponentiate
        return torch.exp(masked_logits)
    else:
        return masked_logits

def scatter_masked_logsumexp(logits,mask,subset_idxs,mask_value=None):

    if mask_value is None:
        mask_value = LOG_ZERO(logits.dtype)
    # apply mask
    masked_logits = mask*logits + (1-mask)*mask_value
    # normalize
    masked_logsumexp = scatter_logsumexp(masked_logits,subset_idxs)
    return masked_logsumexp

def scatter_logsumexp(logits,subset_idxs,eps=EPS,dim_size=None):

    if dim_size is None:
        k = torch.max(subset_idxs)+1
    else:
        assert dim_size >= torch.max(subset_idxs)+1
        k = dim_size
    sm = scatter_reduce(
        src=logits,
        index=subset_idxs,
        reduce="amax",
        dim_size=k,
        default=LOG_ZERO(logits.dtype)
    )
    lm = torch.gather(
        input=sm,
        index=subset_idxs,
        dim=0
    )
    logits = logits - lm
    se = scatter_reduce(
        src=torch.exp(logits),
        index=subset_idxs,
        reduce="sum",
        dim_size=k,
        default=0.
    )
    return sm + torch.log(se + eps)

def scatter_logsoftmax(logits,subset_idxs):
    # calculate normalizing constant
    c = scatter_logsumexp(logits,subset_idxs)
    # apply normalizing constant
    logits = logits - c[subset_idxs]
    return logits

def scatter_softmax(logits,subset_idxs):

    return torch.exp(scatter_logsoftmax(logits,subset_idxs))

def scatter_l1normalize(vals,subset_idxs):
    # calculate normalizing constant
    c = scatter_reduce(
        src=vals,
        index=subset_idxs,
        reduce="sum",
        dim_size=torch.max(subset_idxs)+1
    )
    c = torch.clamp(c,min=EPS)
    # apply normalizing constant
    vals = vals/c[subset_idxs]
    return vals

def scatter_l2normalize(vals,subset_idxs):
    # calculate normalizing constant
    c = scatter_reduce(
        src=vals**2,
        index=subset_idxs,
        reduce="sum",
        dim_size=torch.max(subset_idxs)+1
    )
    c = torch.clamp(torch.sqrt(c),min=EPS)
    # apply normalizing constant
    vals = vals/c[subset_idxs]
    return vals

def scatter_logl2normalize(logits,subset_idxs):
    # calculate normalizing constant
    c = scatter_logsumexp(2*logits,subset_idxs)
    # apply normalizing constant
    logits = logits - 0.5*c[subset_idxs]
    return logits

def scatter_var(src,index,dim_size=None,correction=1,sqrt=False):

    # calculate dim_size
    if dim_size is None:
        dim_size = torch.max(index)+1
    else:
        assert dim_size >= torch.max(index)+1
    # calculate mean
    m = scatter_reduce(
        src=src,
        index=index,
        reduce="mean",
        dim_size=dim_size,
        include_self=False
    )
    # calculate variance
    v_num = scatter_reduce(
        src=(src-m[index])**2,
        index=index,
        reduce="sum",
        dim_size=dim_size
    )
    v_den = scatter_reduce(
        src=torch.ones_like(src),
        index=index,
        reduce="sum",
        dim_size=dim_size
    )
    v = v_num/torch.clamp(v_den-correction,min=EPS)
    if sqrt:
        v = torch.sqrt(v)
    return v

def scatter_argmax(src,index,other_index,dim_size=None,return_max=False):

    # calculate dim_size
    if dim_size is None:
        dim_size = torch.max(index)+1
    else:
        assert dim_size >= torch.max(index)+1
    # calculate max
    mx = scatter_reduce(
        src=src,
        index=index,
        reduce="amax",
        dim_size=dim_size,
        include_self=False
    )
    # calculate mask
    ma = src==mx[index]
    # calculate argmax
    ma_idx = other_index*ma + (-1)*(~ma)
    amx = scatter_reduce(
        src=ma_idx,
        index=index,
        reduce="amax",
        dim_size=dim_size,
        include_self=True,
        default=-1
    )
    if return_max:
        return amx, mx
    else:
        return amx

def scatter_reduce(src,index,reduce,dim=0,dim_size=None,default=0.,include_self=True):

    if reduce == "mean" and include_self:
        print("scatter_reduce: mean reduce with include_self=True is not recommended")
    if dim_size is None:
        dim_size = torch.max(index)+1
    else:
        assert dim_size >= torch.max(index)+1
    result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim+1:]
    results = torch.full(result_shape,default,dtype=src.dtype,device=src.device)
    results.scatter_reduce_(
        dim=dim,
        index=index,
        src=src,
        reduce=reduce,
        include_self=include_self
    )
    return results

def safelog(x,eps=EPS):
    return torch.log(torch.clamp(x,min=eps))

def batchwise_max(xs, batch_idxs):
    """ for debugging """

    batch_size = torch.max(batch_idxs)+1
    maxs = torch.zeros([batch_size],device=xs.device,dtype=xs.dtype)
    for b in range(batch_size):
        maxs[b] = torch.max(xs[batch_idxs==b])
    return maxs

def batchwise_lse(xs, batch_idxs):
    """ for debugging """

    batch_size = torch.max(batch_idxs)+1
    lses = torch.zeros([batch_size],device=xs.device,dtype=xs.dtype)
    for b in range(batch_size):
        lses[b] = torch.logsumexp(xs[batch_idxs==b],0)
    return lses

def to_cpu(data_d, non_blocking=True, detach=False):

    for k in data_d.keys():
        if isinstance(data_d[k],torch.Tensor):
            data = data_d[k]
            if detach:
                data = data.detach()
            data = data.to("cpu",non_blocking=non_blocking)
            data_d[k] = data
    return data_d

def to_device(data_d, device, non_blocking=True):

    for k in data_d.keys():
        v = data_d[k]
        # if isinstance(v, torch.Tensor) or isinstance(v, dgl.DGLGraph) or isinstance(v, pyg.data.Data):
        if isinstance(v, torch.Tensor) or isinstance(v, pyg.data.Data):
            v = v.to(device,non_blocking=non_blocking)
            data_d[k] = v
    return data_d

def deep_update(mapping, *updating_mappings):
    """ 
    adapted from pydantic 
    https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200-L208 
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping

def print_shapes(input_dict):

    for k,v in input_dict.items():
        if isinstance(v,torch.Tensor) or isinstance(v,np.ndarray):
            print(k,"-",tuple(v.shape),"-",type(v))
        elif isinstance(v,list) or isinstance(v,tuple):
            print(k,"-",len(v),"-",type(v))
        elif isinstance(v,pyg.data.Data):
            print(k,"-",(v.num_nodes,v.num_edges),"-",type(v))
        # elif isinstance(v,dgl.DGLGraph):
        # 	print(k,"-",(v.number_of_nodes(),v.number_of_edges()),"-",type(v))
        else:
            print(k,"-",None,"-",type(v))

def th_setdiff1d(t1, t2):

    t1 = torch.unique(t1)
    t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]

def get_package_version(package):

    version = package.__version__.split("+")[0]
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))

def check_pyg_compile():

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    assert pyg_major_version >= 2, pyg_major_version
    return th_minor_version >= 1 and pyg_minor_version >= 4

def check_pyg_full_compile():

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    return pyg_major_version >= 2 and pyg_minor_version >= 5

class NestedDefaultDict(defaultdict):
    """ https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict """
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))
