import numpy as np

class NeuralNetwork:
	def __init__(self, neuronsPerLayer, learningRate=0.5):
		self.neuronsPerLayer = neuronsPerLayer
		# First zip argument excludes last element and second argument excludes first element
		self.weights = [np.random.rand(y, x + 1) for x, y in zip(neuronsPerLayer[:-1], neuronsPerLayer[1:])]
		self.learningRate = learningRate

	def feedforward(self, input):
		currentOutput = np.array(input)
		for weightMatrix in self.weights:
			temp = self.biasedVector(currentOutput)
			temp = np.dot(weightMatrix, temp)
			currentOutput = self.sigmoid(temp)

		return currentOutput


	def trainBatch(self, inputs, outputs):
		gradients = []
		for input, expectedOutput in zip(inputs, outputs):
			netValues = []
			outValues = [input]
			for weightMatrix in self.weights:
				temp = self.biasedVector(outValues[-1]) # Last element = latestOutput
				netValues.append(np.dot(weightMatrix, temp))
				outValues.append(self.sigmoid(netValues[-1]))

			currentDerivative = self.sigmoid_prime(outValues[-1], useSigmoid=False)
			for x in range(len(currentDerivative)):
				currentDerivative[x] *= (outValues[-1][x] - expectedOutput[x])
			gradients.append(np.dot(np.transpose([currentDerivative]), [self.biasedVector(outValues[-2])]))
			for x in reversed(range(1, len(self.weights))):
				temp = np.dot(np.transpose(self.weightsWithoutBias(self.weights[x])), currentDerivative)
				currentDerivative = np.multiply(temp, self.sigmoid_prime(outValues[x], useSigmoid=False))
				gradients.append(np.dot(np.transpose([currentDerivative]), [self.biasedVector(outValues[x - 1])]))

		for index, gradient in enumerate(reversed(gradients)):
			self.weights[index % len(self.weights)] -= gradient * self.learningRate



	def sigmoid(self, value):
		# Addition and division is automatically done elementwise for
		# numpy arrays
		return 1 / (1 + np.exp(-value))

	def sigmoid_prime(self, value, useSigmoid=True):
		if useSigmoid:
			value = self.sigmoid(value)
		return value * (1 - value)

	# Expects a numpy array
	def biasedVector(self, vector):
		return np.append(vector, 1)

	def weightsWithoutBias(self, weights):
		return np.delete(weights, weights.shape[1] - 1, 1)

