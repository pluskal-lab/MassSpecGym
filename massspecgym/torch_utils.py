import torch as th


def scatter_reduce(src,index,reduce,dim=0,dim_size=None,default=0.,include_self=True):

	if reduce == "mean" and include_self:
		print("scatter_reduce: mean reduce with include_self=True is not recommended")
	if dim_size is None:
		dim_size = th.max(index)+1
	else:
		assert dim_size >= th.max(index)+1
	result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim+1:]
	results = th.full(result_shape,default,dtype=src.dtype,device=src.device)
	results.scatter_reduce_(
		dim=dim,
		index=index,
		src=src,
		reduce=reduce,
		include_self=include_self
	)
	return results
