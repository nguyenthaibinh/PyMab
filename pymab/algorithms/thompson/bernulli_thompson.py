from pandas import DataFrame
import numpy as np
from pymab.algorithms.thompson.thompson_base import ThompsonBase
from pymab.algorithms.constant import DEFAULT_INITIAL_ALPHA, DEFAULT_INITIAL_BETA, THOMPSON_SUBTYPE_BERNOULLI, DEFAULT_NUMBER_OF_SAMPLING

class BernoulliThompson(ThompsonBase):
	def __init__(
		self,
		observed_data,
		initial_alpha=DEFAULT_INITIAL_ALPHA,
		initial_beta=DEFAULT_INITIAL_BETA,
		num_sampling=DEFAULT_NUMBER_OF_SAMPLING
		):
		super(BernoulliThompson,self).__init__(
			observed_data=observed_data,
			subtype=THOMPSON_SUBTYPE_BERNOULLI,
			initial_alpha=initial_alpha,
			initial_beta=initial_beta,
			num_sampling=num_sampling
			)

	def allocate_arms(self):
		"""
		Compute the allocation probabilities of arms based on their history info

		Input: None
		Output: arm_allocations, a Series that stores arms' allocation probabilities
				* arm_allocations.values: arms' allocation probabilities
				* arm_allocations.index: arm ids
		"""
		
		# sampled expected success rates of arms
		# data type: DataFrame
		sampled_means = self.sample_success_probabilities()

		if sampled_means is None:
			return None

		# create a matrix that each element has value 1 or 0
		# 1 if it is the largest value in its row, 0 otherwise
		is_max_matrix = sampled_means.apply(lambda row:row.values==row.max(), axis=1)
		is_max_matrix = is_max_matrix.astype(int)

		# calculate allocation probabilities
		# prob_j = (1/G)*SUM(sampled_mu_j(g))
		allocation_prob = None
		allocation_prob = is_max_matrix.mean(axis=0)

		return allocation_prob

	def sample_success_probabilities(self):
		if self._observed_data is None:
			raise ValueError('Error: observed_data is empty!')

		arm_id_list = self._observed_data.columns

		# DataFrame for updated alpha, beta of arms
		beta_params = DataFrame(0, index=['alpha', 'beta'], columns=arm_id_list)
		for arm in arm_id_list:
			total = self._observed_data[arm]['total']
			win = self._observed_data[arm]['win']
			beta_params[arm]['alpha'] = self._initial_alpha + win
			beta_params[arm]['beta'] = self._initial_beta + total - win

		# DataFrame to store sampled estimated expectation
		sampled_means = DataFrame(np.float64(0), index=range(self._num_sampling), columns=arm_id_list)

		# sampling the expections for self._num_sampling times
		# Get list of alpha values
		alpha = beta_params.ix['alpha']
		# Get list of beta values
		beta = beta_params.ix['beta']
		for g in range(self._num_sampling):
			try:
				sampled_means.ix[g] = np.random.beta(alpha, beta)
			except ValueError:
				return None
			finally:
				pass

		return sampled_means
