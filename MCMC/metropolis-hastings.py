import math
import random
import matplotlib.pyplot as plt

random.seed(42)

### goal: sample from a distribution f,

def normal_cond(x):
	# proportional to the distribution we want to sample
	# in this case, normal distribution with condition x > 5
	mean = 0
	sigma = 1

	if x < 5:
		return 0
	return 1 / math.sqrt(2 * math.pi * sigma ** 2) * math.exp(-(x - mean) ** 2 / 2 / sigma ** 2)


def rand():
	return random.uniform(0, 1)


def symmetric_distr(x):
	# returns a sample of h(x', x)
	u_1 = rand()
	u_2 = rand()
	z = math.sqrt(-2 * math.log(u_1)) * math.cos(2 * math.pi * u_2)
	return x + z

def metropolis_hastings(g, h, x0, iters):
	x = x0
	samples = []
	for i in range(iters):
		x_prime = h(x)
		alpha = g(x_prime) / g(x) if g(x) > 0 else 1
		if rand() <= alpha:
			samples.append(x_prime)
			x = x_prime
	return samples


if __name__ == '__main__':
	samples = metropolis_hastings(normal_cond, symmetric_distr, 5, 100000)
	plt.hist(samples)
	plt.show()
