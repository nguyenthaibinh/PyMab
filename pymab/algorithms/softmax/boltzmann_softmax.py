from pandas import Series
import numpy as np
from pymab.algorithms.softmax.softmax_base import SoftmaxBase
from pymab.algorithms.constant import DEFAULT_TAU, SOFTMAX_SUBTYPE_BOLTZMANN


class BoltzmannSoftmax(SoftmaxBase):
	def __init__(
		self,
		observed_data,
		tau=DEFAULT_TAU
		):
		super(BoltzmannSoftmax, self).__init__(
			observed_data=observed_data,
			subtype=SOFTMAX_SUBTYPE_BOLTZMANN,
			tau=tau
			)

	def allocate_arms(self):
		"""
		Compute the allocation probabilities of arms based on their history info

		Input: None
		Output: arm_allocations, a Series that stores arms' allocation probabilities
				* arm_allocations.values: arms' allocation probabilities
				* arm_allocations.index: arm ids
		"""

		# list of arm id
		arm_ids = self._observed_data.columns

		# rewards of arms
		arm_rewards = Series(np.float64(0), index=arm_ids)

		# set epsilon allocation for all arms
		allocation_prob = Series(np.float64(0), index=arm_ids)

		for arm in allocation_prob.index:
			total = self._observed_data[arm]['total']
			win = self._observed_data[arm]['win']
			lost = self._observed_data[arm]['lost']
			reward = np.float64(win - lost) / total if total > 0 else 0

			arm_rewards[arm] = reward

		# calculate avg_payoff / tau
		arm_rewards = arm_rewards / DEFAULT_TAU

		# calculate exp(avg_payoff / tau)
		allocation_prob = np.exp(arm_rewards)

		# calculate exp(avg_payoff / tau) / SUM(exp(avg_payoff / tau))
		allocation_prob = allocation_prob / allocation_prob.sum()

		return allocation_prob
