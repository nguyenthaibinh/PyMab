""" Abstract Base Class for multi-armed bandits """

import numpy as np
from pandas import DataFrame
from abc import ABCMeta, abstractmethod

class BanditBase(object):
	""" Interface for bandit algorithms

	"""

	__metaclass__ = ABCMeta

	@abstractmethod
	def allocate_arms(self):
		"""
		Compute the probability that each arm is selected, based on their
		historical information: number of selections, win, lost

		Return a Series structure with data is the allocation probabilities
		and index are arm id.
		"""
		pass

	@staticmethod
	def choose_arm(arms_to_allocations, method="dirichlet", num_random=1):
		"""
		Choose an arm based on the information given in arms_to_allocations

		input: a Series structure with values are arms' alocation probabilities
				and index are arm id
		output: the choosen arm's id
		"""

		if arms_to_allocations is None:
			raise ValueError('Error: arms_to_allocations is empty!')

		if method == "dirichlet":
			# Get winner randomly based on arms' allocation probabilities
			df = DataFrame(np.random.dirichlet(alpha=arms_to_allocations.values, size=num_random))
			winner = np.argmax(df.sum(axis=0))
			# winner = np.argmax(np.random.dirichlet(arms_to_allocations.values))
		elif method == "multinomial":
			winner = np.random.multinomial(1, arms_to_allocations.values)

		# Return won arm
		return arms_to_allocations.index[winner]
