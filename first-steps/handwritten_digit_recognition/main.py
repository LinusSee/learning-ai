import classes.NeuralNetwork as NeuralNetwork

import numpy as np


nn = NeuralNetwork.NeuralNetwork([2, 2, 1])

trainingDataXor = [
	(np.array([[1], [1]]), np.array([[0]])),
	(np.array([[1], [0]]), np.array([[1]])),
	(np.array([[0], [1]]), np.array([[1]])),
	(np.array([[0], [0]]), np.array([[0]]))
]

#print("Output", nn.feedforward(trainingDataXor[0][0]))
#print("InputData", trainingDataXor)
nn.trainBatch(trainingDataXor, 15000)

#print(trainingDataXor[0][0])
#print(trainingDataXor[1][0])
#print(trainingDataXor[2][0])
#print(trainingDataXor[3][0])
print("#1", nn.feedforward(trainingDataXor[0][0]))
print("#2", nn.feedforward(trainingDataXor[1][0]))
print("#3", nn.feedforward(trainingDataXor[2][0]))
print("#4", nn.feedforward(trainingDataXor[3][0]))
