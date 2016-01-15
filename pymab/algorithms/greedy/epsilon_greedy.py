from pandas import Series
from pymab.algorithms.greedy.epsilon_base import EpsilonBase
from pymab.algorithms.constant import DEFAULT_EPSILON, EPSILON_SUBTYPE_GREEDY

class EpsilonGreedy(EpsilonBase):
	"""
	Class for Epsilon-Greedy algorithm
	"""

	""" The constructor of the class EpsilonGreedy(object):
	Input
	---------------------------------------------------
	observed_data: a pandas.DataFrame structure that stores history infomation of arms.
				* observed_data.columns = list of arm ids
				* observed_data.index = [total, win, lost]
	epsilon: value of epsilon in the method (0 < epsilon < 1)
	"""
	def __init__(
		self,
		observed_data,
		epsilon=DEFAULT_EPSILON
	):

		super(EpsilonGreedy, self).__init__(
			observed_data=observed_data,
			subtype=EPSILON_SUBTYPE_GREEDY,
			epsilon=epsilon,
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

		# number of arms
		num_arms = self._observed_data.columns.size

		# epsilon allocation
		# probability each arm is chosen randomly
		epsilon_allocation = self._epsilon / num_arms

		best_reward_arms = self.get_best_reward_arms(self)
		# number of arms with max payoff
		num_best_arms = len(best_reward_arms)

		# allocation probability of max payoff arms
		best_arm_allocation = (1 - self._epsilon) / num_best_arms

		# set epsilon allocation for all arms
		allocation_prob = Series([epsilon_allocation for i in range(num_arms)], index=arm_ids)

		# set max payoff allocation for max arms
		for arm in best_reward_arms:
			allocation_prob[arm] += best_arm_allocation

		return allocation_prob
