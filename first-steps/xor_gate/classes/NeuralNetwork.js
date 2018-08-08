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

	train(inputVector, expectedOutput) {
		const output = feedForward(inputVector);
	}

	activation(inputVector) {
		return Array.from(inputVector).map(value => this.sigmoid(value));
	}

	sigmoid(value) {
		return (1 / (1 + Math.pow(Math.E, - value)));
	}

	sigmoidDerivative(value) {
		const sigmoidVal = this.sigmoid(value);
		return (sigmoidVal * (1 - sigmoidVal))
	}

	meanSquaredError(actualOutput, expectedOutput) {
		let result = 0;
		for(let i = 0; i < actualOutput.length; i++) {
			const error = actualOutput[i] - expectedOutput[i];
			result += Math.pow(error, 2);
		}
		return result;
	}

	static multiplyMatrixWithVector(matrix, vector) {
		const result = new Array(matrix.length).fill(0);	// The result of a matrix - vector product must be a vector
		for(let i = 0; i < matrix.length; i++) {
			for(let j = 0; j < matrix[0].length; j++) {
				result[i] = result[i] + matrix[i][j] * vector[j];
			}
		}
		// console.table(matrix);
		// console.table(vector);
		// console.table(result);
		return result;
	}
}
