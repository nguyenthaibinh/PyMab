import copy

from pymab.algorithms.bandit_base import BanditBase
from pymab.algorithms.constant import DEFAULT_INITIAL_ALPHA, DEFAULT_INITIAL_BETA, DEFAULT_NUMBER_OF_SAMPLING

class ThompsonBase(BanditBase):
	def __init__(
		self,
		observed_data=None,
		subtype=None,
		initial_alpha=DEFAULT_INITIAL_ALPHA,
		initial_beta=DEFAULT_INITIAL_BETA,
		num_sampling=DEFAULT_NUMBER_OF_SAMPLING
	):
		self._observed_data = copy.deepcopy(observed_data)
		self._subtype = subtype
		self._initial_alpha = initial_alpha
		self._initial_beta = initial_beta
		self._num_sampling = num_sampling