import classes.NeuralNetwork as NeuralNetwork
import classes.MnistLoader as MnistLoader

import numpy as np


loader = MnistLoader.MnistLoader()
training_data, validation_data, test_data = loader.load_data_wrapper()

nn = NeuralNetwork.NeuralNetwork([784, 30, 10], 3.0)
nn.trainBatch(training_data, 10000, batchSize=100)


#nn = NeuralNetwork.NeuralNetwork([2, 2, 1])
countRight = 0
for input, target in test_data:
	#print("Guess", nn.feedforward(input))
	#print("MaxStuff", np.argmax(nn.feedforward(input)))
	#print("Actual", target)
	if np.argmax(nn.feedforward(input)) == target:
		countRight += 1
print("Correct: ", countRight, "/", 10000)

trainingDataXor = [
	(np.array([[1], [1]]), np.array([[0]])),
	(np.array([[1], [0]]), np.array([[1]])),
	(np.array([[0], [1]]), np.array([[1]])),
	(np.array([[0], [0]]), np.array([[0]]))
]

#print("Output", nn.feedforward(trainingDataXor[0][0]))
#print("InputData", trainingDataXor)
#nn.trainBatch(trainingDataXor, 15000)

#print(trainingDataXor[0][0])
#print(trainingDataXor[1][0])
#print(trainingDataXor[2][0])
#print(trainingDataXor[3][0])
#print("#1", nn.feedforward(trainingDataXor[0][0]))
#print("#2", nn.feedforward(trainingDataXor[1][0]))
#print("#3", nn.feedforward(trainingDataXor[2][0]))
#print("#4", nn.feedforward(trainingDataXor[3][0]))
