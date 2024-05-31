import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from massspecgym.simulation_utils.misc_utils import safelog


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_lr_scheduler(
	optimizer, decay_rate: float, decay_steps: int = 5000, warmup_steps: int = 1000
):
	"""build_lr_scheduler.

	Args:
		optimizer:
		decay_rate (float): decay_rate
		decay_steps (int): decay_steps
		warmup_steps (int): warmup_steps
	"""

	def lr_lambda(step):
		if step >= warmup_steps:
			# Adjust
			step = step - warmup_steps
			rate = decay_rate ** (step // decay_steps)
		else:
			rate = 1 - math.exp(-step / warmup_steps)
		return rate

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	return scheduler


class ZerosAggregation(nn.Module):
	""" Zero Aggreation module
		Just sums up features acroess all elements
	Args:
		nn (_type_): _description_
	"""
	def __init__(self):
		super().__init__()
		self.dummy_agg = pyg.nn.SumAggregation()

	def forward(self, x, batch):
		# sum x for the indices of elements for applying the aggregation. 
		# indices here work the same as reduce/sum
		agg = self.dummy_agg(x, batch)
		return 0.*agg


def build_pool_module(pool_type: str, node_dim: int, num_step_set2set:int=3):
	"""medthos build pooling layer for given pool_type and dim
		node dim is required for attention layer. 
		if pool type is not one of 'sum','mean','max','attention', return ZerosAggregation
	Args:
		pool_type (str): pooling type in ['sum','mean','max','attention']
		node_dim (int): node_dim, used for attention

	Returns:
		_type_: _description_
	"""

	if pool_type == "sum":
		pool_module = pyg.nn.SumAggregation()
	elif pool_type == "mean":
		pool_module = pyg.nn.MeanAggregation()
	elif pool_type == "max":
		pool_module = pyg.nn.MaxAggregation()
	elif pool_type == "attention":
		pool_module = pyg.nn.AttentionalAggregation(
			gate_nn = nn.Sequential(
				nn.Linear(node_dim, node_dim),
				nn.ReLU(),
				nn.Linear(node_dim, 1)
			)
		)
	elif pool_type == "set2set":
		"With Set2Set for graph level prediction  (Order Matters: Sequence to sequence for sets) https://arxiv.org/abs/1511.06391"
		pool_module = nn.Sequential(
			pyg.nn.Set2Set(node_dim,processing_steps=num_step_set2set),
			nn.Linear(2 * node_dim, node_dim),
		)
	elif pool_type == "softmax":
		pool_module = pyg.nn.SoftmaxAggregation(learn=True)
	elif pool_type == "powermean":
		pool_module = pyg.nn.PowerMeanAggregation(learn=True)
	elif pool_type == "mean_std_softmax":
		raise NotImplementedError
		# TODO: this need hook up in the way that size of input need change respect this
		#pool_module = pyg.nn.MultiAggregation(['mean', 'std',  pyg.nn.SoftmaxAggregation(learn=True)])
	else:
		assert pool_type == "none", pool_type
		# just return 0s
		pool_module = ZerosAggregation()
	return pool_module

class GNN(nn.Module):
	""" Generic GNN Class
	"""

	def __init__(
		self,
		hidden_size: int,
		num_layers: int,
		node_feats_size: int,
		edge_feats_size: int,
		gnn_type: str,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""_summary_

		Args:
			hidden_size (int): _description_
			num_layers (int): _description_
			node_feats_size (int): _description_
			edge_feats_size (int): _description_
			gnn_type (str): _description_
			dropout (float): _description_

		Raises:
			NotImplementedError: _description_
		"""
		super().__init__()
		self.hidden_size = hidden_size
		# if include ce here, add ce value to each node
		self.edge_feats_size = edge_feats_size
		self.node_feats_size = node_feats_size
		self.dropout = dropout
		self.normalization = normalization

		self.gnn_type = gnn_type
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.input_project = nn.Linear(self.node_feats_size, self.hidden_size)

		if self.gnn_type == "GINE":
			self.gnn = GINE(
				hidden_size=self.hidden_size,
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "NodeMLP":
			self.gnn = NodeMLP(
				hidden_size=self.hidden_size,
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "MPNN":
			self.gnn = MPNN(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				hidden_size=self.hidden_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "GAT" or self.gnn_type == "GATv2":
			self.gnn = GAT(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				hidden_size=self.hidden_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization,
				is_v2=(self.gnn_type == "GATv2")
			)
		else:
			raise NotImplementedError(self.gnn_type)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""encode batch of molecule graph"""
		z = self.input_project(x)
		z = self.gnn(z, batch_index, edge_index, edge_attr)
		return z

def get_norm(normalization,hidden_size):

	# norm_fn may or may not depend on the batch index
	if normalization == "batch":
		norm_mod = nn.BatchNorm1d(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "layer":
		norm_mod = nn.LayerNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "graph":
		norm_mod = pyg.nn.GraphNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x,b)
	else:
		assert normalization == "none", normalization
		norm_mod = nn.Identity()
		norm_fn = lambda x, b, n: n(x)	
	return norm_mod, norm_fn

class GINE(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		node_feats_size: int,
		edge_feats_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""GINE.
			Strategies for Pre-training Graph Neural Networks 
			https://arxiv.org/abs/1905.12265
		Args:
			hidden_size (int): Hidden layer size
			edge_feats (int): Number of edge feats. Must be onehot!
			num_layers (int): Number of message passing steps
			normalization (str): normalization type (batch, layer, none)
			dropout
		"""
		super().__init__()

		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

		self.layers = []
		for i in range(num_layers):
			apply_fn = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
			)
			temp_layer = pyg.nn.conv.GINEConv(
				nn=apply_fn,
				eps=0.,
				edge_dim=None
			)
			self.layers.append(temp_layer)

		self.layers = nn.ModuleList(self.layers)
		# setup norms and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x

class NodeMLP(nn.Module):

	def __init__(
		self,
		hidden_size: int,
		node_feats_size: int,
		edge_feats_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""NodeMLP
		"""
		super().__init__()

		assert edge_feats_size == 0, edge_feats_size

		self.layers = []
		for i in range(num_layers):
			apply_fn = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
			)
			temp_layer = apply_fn
			self.layers.append(temp_layer)

		self.layers = nn.ModuleList(self.layers)
		# setup norm and norm_fn
		# norm_fn may or may not depend on the batch index
		if normalization == "batch":
			norm_mod = nn.BatchNorm1d(hidden_size)
			self.norm_fn = lambda x, b, n: n(x)
		elif normalization == "layer":
			norm_mod = nn.LayerNorm(hidden_size)
			self.norm_fn = lambda x, b, n: n(x)
		elif normalization == "graph":
			norm_mod = pyg.nn.GraphNorm(hidden_size)
			self.norm_fn = lambda x, b, n: n(x,b)
		else:
			assert normalization == "none", normalization
			norm_mod = nn.Identity()
			self.norm_fn = lambda x, b, n: n(x)
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x

class MPNN(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		node_feats_size: int,
		edge_feats_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""Neural Message Passing for Quantum Chemistry
			https://arxiv.org/abs/1704.01212
		"""
		super().__init__()
		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
		# MPNN 
		self.layers = []
		for i in range(num_layers):
			edge_network = edge_network = nn.Sequential(
				nn.Linear(hidden_size, hidden_size), nn.ReLU(),
				nn.Linear(hidden_size, hidden_size * hidden_size))
			layer = pyg.nn.NNConv(
				hidden_size,
				hidden_size,
				edge_network,
				aggr='mean',
				root_weight=False)
			self.layers.append(layer)
		self.layers = nn.ModuleList(self.layers)
		# setup norms and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x

class GAT(nn.Module):
	""" Graph attention networks. https://arxiv.org/abs/1710.10903 
		Graph attention networks v2. (How Attentive are Graph Attention Networks?  https://arxiv.org/abs/2105.14491)
	"""

	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		hidden_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		num_gat_heads: int = 8,
		gat_dropout: float = 0.0,
		is_v2: bool = True,
		is_concat: bool = True,
		**kwargs
	):
		"""_summary_
		Args:
			node_feats_size (int): _description_
			edge_feats_size (int): _description_
			node_hidden_dim (int): _description_
			num_step_message_passing (int): _description_
			num_step_set2set (int): _description_
			output_dim (int): _description_
			gat_heads (int, optional): _description_. Defaults to 1.
			gat_dropout (float, optional): _description_. Defaults to 0.0.
			is_v2 (bool, optional). use v2 GATConv, Defaults to True
			is_concat (bool, optional). concatenated multihead attetion, Defaults to True
		"""
		super(GAT, self).__init__()
		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
		gat_output_size = hidden_size // num_gat_heads if is_concat else hidden_size
		if is_concat and hidden_size % num_gat_heads != 0:
			raise ValueError(f"Ensure that the number of output channels of "
                             f"'GAT' (got '{hidden_size}') is divisible "
                             f"by the number of heads (got '{num_gat_heads}')")	
		GATConv_fn = pyg.nn.conv.GATConv if not is_v2 else pyg.nn.conv.GATv2Conv
		self.layers = []
		for i in range(num_layers):
			layer = GATConv_fn(
				in_channels=hidden_size,
				out_channels=gat_output_size,
				heads=num_gat_heads,
				edge_dim=hidden_size,
				dropout=gat_dropout,
				concat=is_concat)
			self.layers.append(layer)
		self.layers = nn.ModuleList(self.layers)
		# setup norms and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x
	
class MLPBlocks(nn.Module):
	"""Just Good Old Multilayer perceptron with residuals
		layer->dropout->activation(Relu)
		if residuals is True, add a skip connection between block
	Args:
		nn (_type_): _description_
	"""
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		normalization: str,
		dropout: float,
		num_layers: int,
		output_size: int = None,
		use_residuals: bool = False,
	):
		"""_summary_

		Args:
			input_size (int): input dimensions
			hidden_size (int): hidden_size layer dimensions
			dropout (float): dropout ratio
			num_layers (int): num of hidden layers + 1
			output_size (int, optional): output dimensions. Defaults to None.
			use_residuals (bool, optional): if residuals is True, add a skip connection between locks. Defaults to False.
		"""
		super().__init__()
		self.activation = nn.ReLU()
		self.dropout_layer = nn.Dropout(p=dropout)
		self.input_layer = nn.Linear(input_size, hidden_size)
		middle_layer = nn.Linear(hidden_size, hidden_size)
		self.layers = get_clones(middle_layer, num_layers - 1)

		self.output_layer = None
		self.output_size = output_size
		if self.output_size is not None:
			self.output_layer = nn.Linear(hidden_size, self.output_size)

		self.use_residuals = use_residuals

		# setup norm
		if normalization == "batch":
			norm_mod = nn.BatchNorm1d(hidden_size)
		elif normalization == "layer":
			norm_mod = nn.LayerNorm(hidden_size)
		else:
			assert normalization == "none", normalization
			norm_mod = nn.Identity()
		self.norms = get_clones(norm_mod, num_layers)

	def forward(self, x):
		output = x
		output = self.input_layer(x)
		output = self.dropout_layer(output)
		output = self.activation(output)
		old_op = output
		for layer_index, layer in enumerate(self.layers):
			output = layer(output)
			output = self.dropout_layer(output)
			output = self.activation(output)
			output = self.norms[layer_index](output)
			if self.use_residuals:
				output = output + old_op
				old_op = output

		if self.output_layer is not None:
			output = self.output_layer(output)
		return output


class NeimsBlock(nn.Module):
	""" from the NEIMS paper (uses LeakyReLU instead of ReLU) """

	def __init__(self, in_dim, out_dim, dropout):

		super().__init__()
		bottleneck_factor = 0.5
		bottleneck_size = int(round(bottleneck_factor * out_dim))
		self.in_batch_norm = nn.BatchNorm1d(in_dim)
		self.in_activation = nn.LeakyReLU()
		self.in_linear = nn.Linear(in_dim, bottleneck_size)
		self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
		self.out_linear = nn.Linear(bottleneck_size, out_dim)
		self.out_activation = nn.LeakyReLU()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):

		h = x
		h = self.in_batch_norm(h)
		h = self.in_activation(h)
		h = self.dropout(h)
		h = self.in_linear(h)
		h = self.out_batch_norm(h)
		h = self.out_activation(h)
		h = self.out_linear(h)
		return h


class SpecFFN(nn.Module):

	def __init__(self, 
		input_size, 
		hidden_size,
		mz_max,
		mz_bin_res,
		num_layers,
		dropout, 
		prec_mz_offset,
		bidirectional,
		use_residuals,
		output_map_size):

		super().__init__()

		self.input_size = input_size
		self.mz_max = mz_max
		self.mz_bin_res = mz_bin_res
		self.prec_mz_offset = prec_mz_offset
		self.bidirectional = bidirectional
		self.use_residuals = use_residuals
		self._compute_output_size()

		self.in_layer = nn.Linear(input_size, hidden_size)
		self.ff_layers = nn.ModuleList([])
		# self.ff_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
		for i in range(num_layers):
			self.ff_layers.append(
				NeimsBlock(
					hidden_size,
					hidden_size,
					dropout))
		if output_map_size == -1:
			if self.bidirectional:
				# assumes gating, mass masking
				self.forw_out_layer = nn.Linear(hidden_size, self.output_size)
				self.rev_out_layer = nn.Linear(hidden_size, self.output_size)
				self.out_gate = nn.Sequential(
					*[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()])
			else:
				self.out_layer = nn.Linear(hidden_size, self.output_size)
				self.out_gate = nn.Sequential(
					*[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()])
		else:
			if self.bidirectional:
				# assumes gating, mass masking
				self.forw_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.rev_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.out_gate = nn.Sequential(
					*[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()])
			else:
				self.out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.out_gate = nn.Sequential(
					*[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()])
		self.out_activation = nn.ReLU()
		self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)

	def _compute_output_size(self):

		self.mz_bins = nn.Parameter(torch.arange(self.mz_bin_res,self.mz_max+self.mz_bin_res,self.mz_bin_res))
		self.mzs = nn.Parameter(self.mz_bins-0.5*self.mz_bin_res)
		self.output_size = self.mzs.shape[0]

	def _prec_mz_to_idx(self,prec_mz):

		prec_mz_idx = torch.bucketize(torch.clamp(prec_mz,min=0.,max=self.mz_max-self.mz_bin_res),self.mz_bins.to(prec_mz.device),right=True)
		assert torch.max(prec_mz_idx) < self.output_size, (prec_mz_idx,self.output_size)
		return prec_mz_idx

	def forward(self, input_h, prec_mz):

		# get prec_mz_idxs
		prec_mz_idxs = self._prec_mz_to_idx(prec_mz)
		# process inputs
		fh = self.in_layer(input_h)
		# big MLP
		for layer in self.ff_layers:
			if self.use_residuals:
				fh = fh + layer(fh)
			else:
				fh = layer(fh)
		# bidirectional prediction
		if self.bidirectional:
			ff = self.forw_out_layer(fh)
			fr = reverse_prediction(
				self.rev_out_layer(fh),
				prec_mz_idxs,
				self.prec_mz_offset)
			fg = self.out_gate(fh)
			fo = ff * fg + fr * (1. - fg)
			fo = mask_prediction_by_mass(
				fo, prec_mz_idxs, self.prec_mz_offset)
		else:
			# apply output layer
			fo = self.out_layer(fh)
			# apply gating
			fg = self.out_gate(fh)
			fo = fg * fo
		fo = self.out_activation(fo)
		spec = self.out_normalization(fo)
		# handle all zeroes (set to first bin by default)
		all_zero_mask = torch.max(spec,dim=1)[0]<=0.
		all_zero_bonus = torch.zeros_like(spec)
		all_zero_bonus[all_zero_mask,0] = 1.
		spec = spec + all_zero_bonus
		# convert dense spectrum to sparse
		mask = spec>0.
		pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0],-1))[mask]
		pred_logprobs = safelog(spec)[mask]
		pred_batch_idxs = torch.arange(spec.shape[0],device=spec.device).unsqueeze(1).expand(-1,spec.shape[1])[mask]
		# pred_specs = spec
		return pred_mzs, pred_logprobs, pred_batch_idxs #, pred_specs


def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	max_idx = raw_prediction.shape[1]
	assert torch.all(prec_mass_idx < max_idx)
	idx = torch.arange(max_idx, device=prec_mass_idx.device)
	mask = (
		idx.unsqueeze(0) <= (
			prec_mass_idx.unsqueeze(1) +
			prec_mass_offset)).float()
	return mask * raw_prediction


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	batch_size = raw_prediction.shape[0]
	max_idx = raw_prediction.shape[1]
	assert torch.all(prec_mass_idx < max_idx)
	rev_prediction = torch.flip(raw_prediction, dims=(1,))
	# convention is to shift right, so we express as negative to go left
	offset_idx = torch.minimum(
		max_idx * torch.ones_like(prec_mass_idx),
		prec_mass_idx + prec_mass_offset + 1)
	shifts = - (max_idx - offset_idx)
	gather_idx = torch.arange(
		max_idx,
		device=raw_prediction.device).unsqueeze(0).expand(
		batch_size,
		max_idx)
	gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
	offset_rev_prediction = torch.gather(rev_prediction, 1, gather_idx)
	# you could mask_prediction_by_mass here but it's unnecessary
	return offset_rev_prediction


def nan_forward_hook(self, input, output):
	if isinstance(output, tuple):
		outputs = list(output)
	elif isinstance(output, dict):
		outputs = list(output.values())
	else:
		outputs = [output]
	for i, val in enumerate(outputs):
		nan_mask = torch.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def nan_backward_hook(self, grad_input, grad_output):
	for i, val in enumerate(grad_input):
		if val is None:
			continue
		nan_mask = torch.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in grad_input {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])
	for i, val in enumerate(grad_output):
		if val is None:
			continue
		nan_mask = torch.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in grad_output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


class LowRankDense(nn.Module):

    def __init__(self, input_dim, output_dim, rank):
        super(LowRankDense, self).__init__()
        self.layer1 = nn.Linear(input_dim, rank, bias=False)
        self.layer2 = nn.Linear(rank, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
