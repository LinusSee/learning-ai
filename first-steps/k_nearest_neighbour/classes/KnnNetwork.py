import numpy as np

class KnnNetwork:

	def __init__(self):
		print("Constructor")

	# caseToPredict = numpy array of position arguments
	# k = count of nearest neighbours
	def guess(caseToPredict, k):
		distances = [ (el[1], euclidianDistance(el[0], caseToPredict)) for el in self.trainingData ]
		distances.sort(key=lambda x: x[1])
		neighbours = ascDistances[-k:]

		classCount = self.classCount(neighbours)

		currentGuess
		for key, value in classCount:
			maxVal = 0
			if value >= maxVal:
				maxVal = value
				currentGuess = key

		return key

	# Expects a list of tuples. The first tuple element should contain the positional
	# arguments and the other the correct classification
	def train(self, trainingData):
		self.trainingData = trainingData

	# Works with numpy vectors
	def euclidianDistance(self, data1, data2):
		differenceSquared = np.square(np.array(data1) - np.array(data2))
		sumOfSquares = np.sum(differenceSquared)

		return np.sqrt(sumOfSquares)

	def classCount(self, tuples):
		result = {}
		for key, value in tuples:
			result.setdefault(key, 0)
			result[key] = result[key] + 1
		return result
