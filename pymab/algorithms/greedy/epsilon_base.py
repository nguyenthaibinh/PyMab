import copy
import numpy as np
from pandas import Series

from pymab.algorithms.bandit_base import BanditBase
from pymab.algorithms.constant import DEFAULT_EPSILON

class EpsilonBase(BanditBase):
	def __init__(
		self,
		observed_data=None,
		subtype=None,
		epsilon=DEFAULT_EPSILON,
	):
		self._observed_data = copy.deepcopy(observed_data)
		self._subtype = subtype
		self._epsilon = epsilon

	@staticmethod
	def get_best_reward_arms(self):
		"""
		Get the arms that produce best reward so far
		"""
		if self._observed_data is None:
			raise ValueError('arms_history is empty!')

		# list of arm id
		arm_ids = self._observed_data.columns
		
		arm_rewards = Series(np.float64(0), index=arm_ids)
		for arm in arm_ids:
			total = self._observed_data[arm]['total']
			win = self._observed_data[arm]['win']
			lost = self._observed_data[arm]['lost']

			reward = np.float64(win - lost) / total if total > 0 else 0
			arm_rewards[arm] = reward

		best_reward = arm_rewards.max()

		# get all index of arms having max payoff
		best_reward_arms = np.where(arm_rewards.values == best_reward)[0]
		
		# conver index to arm id
		best_reward_arms = map(lambda x: arm_rewards.index[x], best_reward_arms)

		return best_reward_arms