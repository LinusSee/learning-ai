class NeuralNetwork {
	constructor(inputCount, hiddenCount, outputCount, learningRate) {
		this.hiddenWeights = new Array(hiddenCount).fill().map(e => new Array(inputCount).fill().map(v => Math.random() * 2 - 1));
		this.outputWeights = new Array(outputCount).fill().map(e => new Array(hiddenCount).fill().map(v => Math.random() * 2 - 1));
		this.hiddenBias = Math.random() * 2 - 1;
		this.outputBias = Math.random() * 2 - 1;
		this.learningRate = learningRate;
	}

	feedForward(inputVector) {
		const hiddenNetInput = NeuralNetwork.multiplyMatrixWithVector(this.hiddenWeights, inputVector).map(val => val + this.hiddenBias);
		const hiddenOutput = this.activation(hiddenNetInput);
		const finalNetInput = NeuralNetwork.multiplyMatrixWithVector(this.outputWeights, hiddenOutput).map(val => val + this.outputBias);

		return this.activation(finalNetInput);
	}

	train(inputVector, expectedOutput) {
		const hiddenNetInput = NeuralNetwork.multiplyMatrixWithVector(this.hiddenWeights, inputVector).map(val => val + this.hiddenBias);
		const hiddenOutput = this.activation(hiddenNetInput);
		const finalNetInput = NeuralNetwork.multiplyMatrixWithVector(this.outputWeights, hiddenOutput).map(val => val + this.outputBias);
		const finalOutput = this.activation(finalNetInput);

		// From here on out its only made for a 2-2-1 network (Only temporary)
		const errorGradient = finalOutput[0] - expectedOutput[0];
		const outputGradient = finalNetInput.map(val => this.sigmoidDerivative(val))[0];
		//console.log("ErrGrad:", errorGradient);
		//console.log("OutGrad:", outputGradient);

		const weight5Gradient = errorGradient * outputGradient * hiddenOutput[0];
		const weight6Gradient = errorGradient * outputGradient * hiddenOutput[1];
		//console.log("W5Grad:", weight5Gradient);
		//console.log("W6Grad:", weight6Gradient);

		const weight1Gradient = errorGradient * outputGradient * this.outputWeights[0][0] * hiddenNetInput.map(val => this.sigmoidDerivative(val))[0] * inputVector[0];
		const weight2Gradient = errorGradient * outputGradient * this.outputWeights[0][0] * hiddenNetInput.map(val => this.sigmoidDerivative(val))[0] * inputVector[1];
		const weight3Gradient = errorGradient * outputGradient * this.outputWeights[0][1] * hiddenNetInput.map(val => this.sigmoidDerivative(val))[1] * inputVector[0];
		const weight4Gradient = errorGradient * outputGradient * this.outputWeights[0][1] * hiddenNetInput.map(val => this.sigmoidDerivative(val))[1] * inputVector[1];
		//console.log("W1Grad", weight1Gradient);
		//console.log("W2Grad", weight2Gradient);
		//console.log("W3Grad", weight3Gradient);
		//console.log("W4Grad", weight4Gradient);

		this.hiddenWeights[0][0] -= this.learningRate * weight1Gradient;
		this.hiddenWeights[0][1] -=	this.learningRate * weight2Gradient;
		this.hiddenWeights[1][0] -= this.learningRate * weight3Gradient;
		this.hiddenWeights[1][1] -= this.learningRate * weight4Gradient;

		this.outputWeights[0][0] -= this.learningRate * weight5Gradient;
		this.outputWeights[0][1] -= this.learningRate * weight6Gradient;
	}

	activation(inputVector) {
		return Array.from(inputVector).map(value => this.sigmoid(value));
	}

	sigmoid(value) {
		return (1 / (1 + Math.pow(Math.E, - value)));
	}

	sigmoidDerivative(value) {
		const sigmoidVal = this.sigmoid(value);
		return sigmoidVal * (1 - sigmoidVal);
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
