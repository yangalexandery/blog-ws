# MNIST parsing code taken from here: https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
import struct as st
import numpy as np
import math
import matplotlib.pyplot as plt

filename = {'images': 'data/t10k-images-idx3-ubyte', 'labels': 'data/t10k-labels-idx1-ubyte'}
image_file = open(filename['images'], 'rb')
label_file = open(filename['labels'], 'rb')

image_file.seek(0)
magic = st.unpack('>4B', image_file.read(4))

n = st.unpack('>I', image_file.read(4))[0]
rows = st.unpack('>I', image_file.read(4))[0]
cols = st.unpack('>I', image_file.read(4))[0]

images = np.zeros((n, rows, cols))

total_bytes = n * rows * cols
images =  np.asarray(st.unpack('>' + 'B'*total_bytes, image_file.read(total_bytes))).reshape((n, rows, cols))

magic = st.unpack('>4B', label_file.read(4))

n = st.unpack('>I', label_file.read(4))[0]
total_bytes = n
labels = np.asarray(st.unpack('>' + 'B'*total_bytes, label_file.read(total_bytes))).reshape((n, 1))

n = 100
images = images[:n] / 255.0
labels = labels[:n]

counts = [0] * 10
for i in range(n):
	counts[labels[i][0]] += 1
# print(counts)

def entropy(p):
	"""Calculates the Shannon Entropy of an array p of probabilities."""
	return np.sum(-np.multiply(p, np.log(p + 1e-50))) / np.log(2) 

def SNE(data, perplexity=10, eta=0.01, iters=100):
	"""Unsupervised method for 2D-visualizaton.

	Given data of size (n, d), returns an array of size (n, 2)."""

	print("Calculating sigmas")
	sigmas = []
	n = data.shape[0]
	p = np.zeros((n, n)) # p[i][j] equals p_{j|i}
	for i in range(n):
		low = 0.02
		high = 100

		data_minus_xi = data - data[i]
		norms = np.linalg.norm(data_minus_xi, axis=1)
		norms = -np.square(norms)
		while (high - low > 1e-4):
			mid = (low + high) / 2.0
			# test mid as sigma_i
			p_cond = np.exp(norms / (2 * mid * mid))
			p_cond[i] = 0.0
			p_cond = p_cond / np.sum(p_cond)
			p[i] = p_cond
			h = entropy(p_cond)
			print(h, mid)
			if math.pow(2, h) < perplexity:
				low = mid
			else:
				high = mid
		sigmas.append(low)
	# now we want to come up with y_i's that mimic the p[i][j] distribution
	print("Determining y_i's")
	ys = np.random.randn(n, 2)
	y_tm1 = np.array(ys)
	y_tm2 = np.array(ys)
	for it in range(iters):
		print(it)
		q = np.zeros((n, n))
		for i in range(n):
			# calculate q's
			data_minus_yi = ys - ys[i]
			norms = np.linalg.norm(data_minus_yi, axis=1)
			norms = -np.square(norms)
			# print("0: ", norms, i)
			norms[i] = -1e9
			norms -= np.max(norms)
			# print("1: ", norms)
			q_cond = np.exp(norms)
			q_cond[i] = 0.0
			q_cond = q_cond / np.sum(q_cond)
			# print("2: ", q_cond)
			q[i] = q_cond

		gradient = np.zeros((n, 2))
		# bottleneck of this algorithm
		for i in range(n):
			# print("3a: ", p[i:i+1,:])
			# print("3b: ", p[:,i:i+1].T)
			# print("3c: ", q[i:i+1,:])
			# print("3d: ", q[:,i:i+1].T)
			tmp = p[i:i+1,:] + p[:,i:i+1].T + q[i:i+1,:] + q[:,i:i+1].T
			ydiff = -(ys - ys[i])
			# print("3: ", tmp)
			gradient[i] = 2 * np.dot(tmp, ydiff)
			# for j in range(n):
			# 	if i != j:
			# 		gradient[i] += 2 * (ys[i] - ys[j]) * (p[i][j] + p[j][i] - q[i][j] - q[j][i])
		# print("4: ", gradient)
		ys_tp1 = ys + eta * gradient + 0.5 * (y_tm1 - y_tm2)
		y_tm2 = y_tm1
		y_tm1 = ys
		ys = ys_tp1
	return ys



if __name__ == '__main__':
	images = images.reshape((n, 28 * 28))
	ys = SNE(images, perplexity=5, iters=75)
	plt.scatter(ys[:,0], ys[:,1])
	color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'black', 'grey', 'magenta', 'brown']
	for i in range(10):
		x = []
		y = []
		for j in range(n):
			if labels[j] == i:
				x.append(ys[j][0])
				y.append(ys[j][1])
		plt.scatter(x, y, color=color[i])

	plt.show()
	# TODO: debug. things shouldn't be growing to infinity