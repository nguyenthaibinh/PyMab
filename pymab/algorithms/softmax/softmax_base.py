import copy

from pymab.algorithms.bandit_base import BanditBase
from pymab.algorithms.constant import DEFAULT_TAU

class SoftmaxBase(BanditBase):
	def __init__(
		self,
		observed_data=None,
		subtype=None,
		tau=DEFAULT_TAU,
	):
		self._observed_data = copy.deepcopy(observed_data)
		self._subtype = subtype
		self._tau = tau