import torch
import torch.nn as nn
import numpy as np

DEFAULT_MAX_COUNT_INT = 255
DEFAULT_NUM_EXTRA_EMBEDDINGS = 1

class IntFeaturizer(nn.Module):
	"""
	Base class for mapping integers to a vector representation (primarily to be used as a "richer" embedding for NNs
	processing integers).

	Subclasses should define `self.int_to_feat_matrix`, a matrix where each row is the vector representation for that
	integer, i.e. to get a vector representation for `5`, one could call `self.int_to_feat_matrix[5]`.

	Note that this class takes care of creating a fixed number (`self.NUM_EXTRA_EMBEDDINGS` to be precise) of extra
	"learned" embeddings these will be concatenated after the integer embeddings in the forward pass,
	be learned, and be used for extra  non-integer tokens such as the "to be confirmed token" (i.e., pad) token.
	They are indexed starting from `self.max_count_int`.
	"""

	def __init__(self, embedding_dim, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__()
		self.max_count_int = max_count_int
		self.num_extra_embeddings = num_extra_embeddings
		weights = torch.zeros(self.num_extra_embeddings, embedding_dim)
		self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
		nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
		self.embedding_dim = embedding_dim

	def forward(self, tensor):
		"""
		Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
		"""
		orig_shape = tensor.shape
		# out_tensor = torch.empty(
		# 	(*orig_shape, self.embedding_dim), device=tensor.device
		# )
		extra_embed_mask = (tensor >= self.max_count_int).float()

		tensor = tensor.long()
		norm_embeds = self.int_to_feat_matrix[tensor]
		extra_embeds = self._extra_embeddings[torch.maximum(tensor,self.max_count_int*torch.ones_like(tensor))-self.max_count_int]
		out_tensor = (1.-extra_embed_mask).unsqueeze(-1)*norm_embeds + extra_embed_mask.unsqueeze(-1)*extra_embeds

		temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
		return temp_out

	@property
	def num_dim(self):
		return self.int_to_feat_matrix.shape[1]

	# @property
	# def full_dim(self):
	#	 return self.num_dim * common.NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
	"""
	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
		# ^ need at least this many to ensure that the whole input range can be represented on the half circle.

		freqs = 0.5 ** torch.arange(num_freqs, dtype=torch.float32)
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=2 * freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings,
		)  # 2 for cosine and sine

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			torch.arange(self.max_count_int, dtype=torch.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		all_features = torch.cat(
			[torch.cos(combo_of_sinusoid_args), torch.sin(combo_of_sinusoid_args)],
			dim=1,
		)

		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(all_features.float())
		self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
	"""
	Like other fourier feats but sines only

	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
		# ^ need at least this many to ensure that the whole input range can be represented on the half circle.

		freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings	
		)

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			torch.arange(self.max_count_int, dtype=torch.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(
			torch.sin(combo_of_sinusoid_args).float()
		)
		self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
	"""
	Like other fourier feats but sines only and absoluted.

	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2

		freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			torch.arange(self.max_count_int, dtype=torch.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(
			torch.abs(torch.sin(combo_of_sinusoid_args)).float()
		)
		self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
	"""
	A featurizer that puts radial basis functions evenly between 0 and max_count-1. These will have a width of
	(max_count-1) / (num_funcs) to decay to about 0.6 of its original height at reaching the next func.

	"""

	def __init__(self, num_funcs=32, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		"""
		:param num_funcs: number of radial basis functions to use: their width will automatically be chosen -- see class
							docstring.
		"""
		super().__init__(
			embedding_dim=num_funcs,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		width = (self.max_count_int - 1) / num_funcs
		centers = torch.linspace(0, self.max_count_int- 1, num_funcs)

		pre_exponential_terms = (
			-0.5
			* ((torch.arange(self.max_count_int)[:, None] - centers[None, :]) / width)
			** 2
		)
		# ^ shape: MAX_COUNT_INT x num_funcs
		feats = torch.exp(pre_exponential_terms)

		self.int_to_feat_matrix = nn.Parameter(feats.float())
		self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
	"""
	A featurizer that turns integers into their one hot encoding.

	Represents:
	 - 0 as 1000000000...
	 - 1 as 0100000000...
	 - 2 as 0010000000...
	 and so on.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__(
			embedding_dim=max_count_int,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		feats = torch.eye(self.max_count_int)
		self.int_to_feat_matrix = nn.Parameter(feats.float())
		self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
	"""
	Learns the features for the different integers.

	Pretty much `nn.Embedding` but we get to use the forward of the superclass which behaves a bit differently.
	"""

	def __init__(self, feature_dim=32, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__(
			embedding_dim=feature_dim,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		weights = torch.zeros(self.max_count_int, feature_dim)
		self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
		nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


# class FloatFeaturizer(IntFeaturizer):
#	 """
#	 Norms the features
#	 """

#	 def __init__(self):
#		 # Norm vec
#		 # Placeholder..
#		 super().__init__(embedding_dim=1)
#		 self.norm_vec = torch.from_numpy(common.NORM_VEC).float()
#		 self.norm_vec = nn.Parameter(self.norm_vec)
#		 self.norm_vec.requires_grad = False

#	 def forward(self, tensor):
#		 """
#		 Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
#		 """
#		 tens_shape = tensor.shape
#		 out_shape = [1] * (len(tens_shape) - 1) + [-1]
#		 return tensor / self.norm_vec.reshape(*out_shape)

#	 @property
#	 def num_dim(self):
#		 return 1


def get_embedder(embedder,**kwargs):
	if embedder == "fourier":
		embedder = FourierFeaturizer(**kwargs)
	elif embedder == "rbf":
		embedder = RBFFeaturizer(**kwargs)
	elif embedder == "one-hot":
		embedder = OneHotFeaturizer(**kwargs)
	elif embedder == "learnt":
		embedder = LearnedFeaturizer(**kwargs)
	elif embedder == "float":
		raise NotImplementedError
		# embedder = FloatFeaturizer()
	elif embedder == "fourier-sines":
		embedder = FourierFeaturizerSines(**kwargs)
	elif embedder == "abs-sines":
		embedder = FourierFeaturizerAbsoluteSines(**kwargs)
	else:
		raise NotImplementedError
	return embedder
