import numpy as np
import random

class NeuralNetwork:
	def __init__(self, neuronsPerLayer, learningRate=0.5):
		self.neuronsPerLayer = neuronsPerLayer
		self.weights = [np.random.randn(y, x) for x, y in zip(neuronsPerLayer[:-1], neuronsPerLayer[1:])]
		#self.biases = [np.random.randn(y, 1) for y in neuronsPerLayer[1:]]
		self.biases = [ np.random.randn(y) for y in neuronsPerLayer[1:] ]
		self.learningRate = learningRate
		print("Weights", self.weights)
		print("Biases", self.biases)

	def feedforward(self, input):
		currentOutput = np.array(input)
		for weightMatrix, biases in zip(self.weights, self.biases):
			temp = np.dot(weightMatrix, currentOutput) + biases
			#print("Temp", temp)
			currentOutput = self.sigmoid(temp)
		return currentOutput

	def trainBatch(self, trainingData, trainingIterations, stochasticTraining=False, batchSize=500):
		if stochasticTraining:
			batchSize = len(trainingData)
		for a in range(trainingIterations):
			random.shuffle(trainingData)
			matrices = []
			for weights, biases in zip(self.weights, self.biases):
				#print(weights)
				#print(biases)
				matrices.append(np.append(weights.copy(), np.transpose([biases.copy()]), 1))

			for input, target in trainingData[:batchSize]:
				netValues = []
				outValues = [input]

				for weightMatrix, biases in zip(self.weights, self.biases):
					temp = np.dot(weightMatrix, outValues[-1]) + biases
					netValues.append(temp)
					outValues.append(self.sigmoid(netValues[-1]))

				error = self.error_prime(outValues[-1], np.array(target))
				currentDerivative = self.sigmoid_prime(outValues[-1], useSigmoid=False) * error
				gradients = [ np.dot(np.transpose([currentDerivative]), [np.append(outValues[-2], 1)]) ]

				for x in reversed(range(1, len(self.weights))):
					temp = np.dot(np.transpose(self.weights[x]), currentDerivative)
					currentDerivative = np.multiply(currentDerivative, self.sigmoid_prime(outValues[x], useSigmoid=False))
					gradients.append(np.dot(np.transpose([currentDerivative]), [np.append(outValues[x - 1], 1)]))

				for index, gradient in enumerate(reversed(gradients)):
					matrices[index] -= gradient * self.learningRate

			weights = [ np.delete(matrix, -1, 1) for matrix in matrices ]
			biases = [ matrix[:,-1] for matrix in matrices ]
			self.weights = weights
			self.biases = biases

	# Expects either a single value or a numpy array
	def sigmoid(self, value):
		return 1 / (1 + np.exp(-value))

	def sigmoid_prime(self, value, useSigmoid=True):
		if useSigmoid:
			value = self.sigmoid(value)
		return value * (1 - value)

	def error_prime(self, actual, target):
		return actual - target;
