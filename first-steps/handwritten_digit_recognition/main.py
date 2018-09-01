import classes.NeuralNetwork as NeuralNetwork

nn = NeuralNetwork.NeuralNetwork([2, 2, 1])

inputs = [
	[1, 1],
	[1, 0],
	[0, 1],
	[0, 0]
]

xorOutputs = [ [0], [1], [1], [0] ]

print("Output", nn.feedforward(inputs[0]))

for x in range(10000):
	nn.trainBatch(inputs, xorOutputs)

print(nn.feedforward(inputs[0]))
print(nn.feedforward(inputs[1]))
print(nn.feedforward(inputs[2]))
print(nn.feedforward(inputs[3]))
