import numpy as np

class NeuralNetwork:
	def __init__(self, neuronsPerLayer, learningRate=0.5):
		self.neuronsPerLayer = neuronsPerLayer
		# First zip argument excludes last element and second argument excludes first element
		self.weights = [np.random.rand(y, x + 1) for x, y in zip(neuronsPerLayer[:-1], neuronsPerLayer[1:])]
		print(self.weights)

	def feedforward(self, input):
		currentOutput = np.array(input)
		for weightMatrix in self.weights:
			temp = self.biasedVector(currentOutput)
			temp = np.dot(weightMatrix, temp)
			currentOutput = self.sigmoid(temp)

		return currentOutput


	def trainBatch(self):
		print("Currently empty method")

	def sigmoid(self, value):
		# Addition and division is automatically done elementwise for
		# numpy arrays
		return 1 / (1 + np.exp(-value))

	# Expects a numpy array
	def biasedVector(self, vector):
		return np.append(vector, 1)

