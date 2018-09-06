import numpy as np

class KnnNetwork:

	def __init__(self):
		print("Constructor")

	# Works with numpy vectors
	def euclidianDistance(self, data1, data2):
		differenceSquared = np.square(data1 - data2)
		sumOfSquares = np.sum(differenceSquared)

		return np.sqrt(sumOfSquares)

