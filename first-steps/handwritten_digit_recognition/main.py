import classes.NeuralNetwork as NeuralNetwork

nn = NeuralNetwork.NeuralNetwork([2, 2, 1])

trainingDataXor = [
	([1, 1], [0]),
	([1, 0], [1]),
	([0, 1], [1]),
	([0, 0], [0])
]

inputs = [
	[1, 1],
	[1, 0],
	[0, 1],
	[0, 0]
]

xorOutputs = [ [0], [1], [1], [0] ]

print("Output", nn.feedforward(inputs[0]))

nn.trainBatch(trainingDataXor, 15000)

print(nn.feedforward(inputs[0]))
print(nn.feedforward(inputs[1]))
print(nn.feedforward(inputs[2]))
print(nn.feedforward(inputs[3]))
