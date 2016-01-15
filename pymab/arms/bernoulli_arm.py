from numpy.random import binomial

class BernoulliArm:
	def __init__(self, prob=0.5):
		if not 0.0 <= prob <= 1:
			raise ValueError("prob is not in [0.0, 1.0]")
		self._p = prob

	def set_prob(self, prob=0.5):
		if not 0.0 <= prob <= 1:
			raise ValueError("prob is not in [0.0, 1.0]")
		self._p = prob

	def get_prob(self):
		return self._p

	def play(self):
		reward = binomial(1, self._p)
		return reward