class NeuralNetwork {
	constructor(inputCount, hiddenCount, outputCount, learningRate) {
		this.hiddenWeights = new Array(hiddenCount).fill().map(e => new Array(inputCount).fill().map(v => Math.random() * 2 - 1));
		this.outputWeights = new Array(outputCount).fill().map(e => new Array(hiddenCount).fill().map(v => Math.random() * 2 - 1));
		this.learningRate = learningRate;
	}
}
