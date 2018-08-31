from classes.NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2, 2, 1])
#print(nn.feedforward([1, 0]))

xorInputs = [
	[1, 1],
	[1, 0],
	[0, 1],
	[0, 0]
]
xorOutputs = [
	[0],
	[1],
	[1],
	[0]
]
for x in range(0, 10000):
	nn.trainBatch(xorInputs, xorOutputs)

print(nn.feedforward([1, 1]))
print(nn.feedforward([1, 0]))
print(nn.feedforward([0, 1]))
print(nn.feedforward([0, 0]))
