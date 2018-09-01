from classes.NeuralNetwork import NeuralNetwork

xorNet = NeuralNetwork([2, 2, 1])
andNet = NeuralNetwork([2, 1])
orNet = NeuralNetwork([2, 1])
bigXorNet = NeuralNetwork([2, 2, 2, 1])
rlyBigXorNet = NeuralNetwork([2, 6, 8, 6, 2, 1])
switchNet = NeuralNetwork([2, 4, 3, 2])

inputs = [
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
andOutputs = [
	[1],
	[0],
	[0],
	[0]
]
orOutputs = [
	[1],
	[1],
	[1],
	[0]
]
switchOutputs = [
	[1, 1],
	[0, 1],
	[1, 0],
	[0, 0]
]

for x in range(0, 10000):
	xorNet.trainBatch(inputs, xorOutputs)
	andNet.trainBatch(inputs, andOutputs)
	orNet.trainBatch(inputs, orOutputs)
	bigXorNet.trainBatch(inputs, xorOutputs)
	# Commented out because training didn't even work with 100.000 iterations
	# Tried 1million, which worked but took several minutes (about 10)
	#rlyBigXorNet.trainBatch(inputs, xorOutputs)
	switchNet.trainBatch(inputs, switchOutputs)

def printOutput(network):
	print(network.feedforward(inputs[0]))
	print(network.feedforward(inputs[1]))
	print(network.feedforward(inputs[2]))
	print(network.feedforward(inputs[3]))
	print("\n")

printOutput(xorNet)
printOutput(andNet)
printOutput(orNet)
printOutput(bigXorNet)
printOutput(rlyBigXorNet)
printOutput(switchNet)
