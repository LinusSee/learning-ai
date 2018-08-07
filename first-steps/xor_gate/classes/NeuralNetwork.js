class NeuralNetwork {
	constructor(inputCount, hiddenCount, outputCount, learningRate) {
		this.hiddenWeights = new Array(hiddenCount).fill().map(e => new Array(inputCount).fill().map(v => Math.random() * 2 - 1));
		this.outputWeights = new Array(outputCount).fill().map(e => new Array(hiddenCount).fill().map(v => Math.random() * 2 - 1));
		this.learningRate = learningRate;
	}

	feedForward(inputVector) {
		const hiddenOutput = NeuralNetwork.multiplyMatrixWithVector(this.hiddenWeights, inputVector);
		const finalOutput = NeuralNetwork.multiplyMatrixWithVector(this.outputWeights, hiddenOutput);

		return this.activation(finalOutput);
	}


	activation(inputVector) {
		return Array.from(inputVector).map(value => this.sigmoid(value));
	}

	sigmoid(value) {
		return (1 / (1 + Math.pow(Math.E, - value)));
	}

	static multiplyMatrixWithVector(matrix, vector) {
		const result = new Array(matrix.length).fill(0);	// The result of a matrix - vector product must be a vector
		for(let i = 0; i < matrix.length; i++) {
			for(let j = 0; j < matrix[0].length; j++) {
				result[i] = result[i] + matrix[i][j] * vector[j];
			}
		}
		return result;
	}
}
