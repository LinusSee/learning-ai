import numpy as np
import random

class NeuralNetwork:
	def __init__(self, neuronsPerLayer, learningRate=0.5):
		self.neuronsPerLayer = neuronsPerLayer
		self.weights = [np.random.randn(y, x) for x, y in zip(neuronsPerLayer[:-1], neuronsPerLayer[1:])]
		self.biases = [np.random.randn(y, 1) for y in neuronsPerLayer[1:]]
		#self.biases = [ np.random.randn(y) for y in neuronsPerLayer[1:] ]
		self.learningRate = learningRate
		#print("WeightsConstr", self.weights)
		#print("BiasesConstr", self.biases)

	def feedforward(self, input):
		currentOutput = input
		for weightMatrix, biases in zip(self.weights, self.biases):
			temp = np.dot(weightMatrix, currentOutput) + biases
			currentOutput = self.sigmoid(temp)
		return currentOutput

	def trainBatch(self, trainingData, trainingIterations, stochasticTraining=False, batchSize=500):
		print("Before copy")
		trainingData = list(trainingData)
		print("After copy")
		#print(trainingData)
		if stochasticTraining:
			batchSize = len(trainingData)
		for a in range(trainingIterations):
			random.shuffle(trainingData)
			batches = [ trainingData[x: x+batchSize] for x in range(0, len(trainingData), batchSize) ]
			matrices = []
			for weights, biases in zip(self.weights, self.biases):
				#print(weights)
				#print(biases)
				matrices.append(np.append(weights.copy(), biases.copy(), 1))
			print("Iteration:", a)
			for batch in batches:
				for input, target in batch:
					netValues = []
					outValues = [input]

					for weightMatrix, biases in zip(self.weights, self.biases):
						temp = np.dot(weightMatrix, outValues[-1]) + biases
						netValues.append(temp)
						outValues.append(self.sigmoid(netValues[-1]))
#					print("OutVal", outValues[-1])
#					print("Target", target)
					error = self.error_prime(outValues[-1], target)
#					print("Error", error)
					currentDerivative = self.sigmoid_prime(outValues[-1], useSigmoid=False) * error
#					print("CurrentDeriv", currentDerivative)
					gradients = [ np.dot(currentDerivative, np.transpose(np.append(outValues[-2], [[1]], 0))) ]
#					print("Gradients", gradients)

					for x in reversed(range(1, len(self.weights))):
						temp = np.dot(np.transpose(self.weights[x]), currentDerivative)
#						print("Temp", temp)
						currentLayerDerivative = self.sigmoid_prime(outValues[x], useSigmoid=False)
#						print("CurrentLayerDerivative", currentLayerDerivative)
						currentDerivative = np.multiply(temp, currentLayerDerivative)
#						print("CurrentDerivLoop", currentDerivative)
#						print("Before", np.transpose(np.append(outValues[x - 1], [[1]], 0)))
						gradients.append(np.dot(currentDerivative, np.transpose(np.append(outValues[x - 1], [[1]], 0))))
#						print("Gradient2", gradients[1])

					for index, gradient in enumerate(reversed(gradients)):
						matrices[index] -= gradient * self.learningRate
#				print("Gradients", gradients)
#				print("Matrices", matrices)
				# Can be done easier, no delete necessary
				weights = [ np.delete(matrix, -1, 1) for matrix in matrices ]
				biases = [ matrix[:,[-1]] for matrix in matrices ]
#				print("Weights", weights)
#				print("Biases", biases)
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
