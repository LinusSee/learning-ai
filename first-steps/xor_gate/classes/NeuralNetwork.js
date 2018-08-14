class NeuralNetwork {
	constructor(inputCount, hiddenCount, outputCount, learningRate) {
		this.hiddenWeights = new Array(hiddenCount).fill().map(e => new Array(inputCount + 1).fill().map(v => Math.random() * 2 - 1));
		this.outputWeights = new Array(outputCount).fill().map(e => new Array(hiddenCount + 1).fill().map(v => Math.random() * 2 - 1));
		//this.hiddenBias = Math.random() * 2 - 1;
		//this.outputBias = Math.random() * 2 - 1;
		this.learningRate = learningRate;
	}

	feedForward(inputVector) {
		const hiddenNetInput = NeuralNetwork.multiplyMatrixWithVector(this.hiddenWeights, this.biasedVector(inputVector));
		const hiddenOutput = this.activation(hiddenNetInput);
		const finalNetInput = NeuralNetwork.multiplyMatrixWithVector(this.outputWeights, this.biasedVector(hiddenOutput));

		return this.activation(finalNetInput);
	}

	biasedVector(inputVector) {
		const newVector = Array.from(inputVector);
		newVector.push(1.0);
		return newVector;
	}

	train(inputVector, expectedOutput) {
		this.trainBatch([inputVector], [expectedOutput]);
	}

	trainBatch(inputs, expectedOutputs) {
		let weight1Gradient = 0;
		let weight2Gradient = 0;
		let weight3Gradient = 0;
		let weight4Gradient = 0;
		let weight5Gradient = 0;
		let weight6Gradient = 0;
		let bias1Gradient = 0;
		let bias2Gradient = 0;
		let bias3Gradient = 0;

		for(let i = 0; i < inputs.length; i++) {
			const hiddenNetInput = NeuralNetwork.multiplyMatrixWithVector(this.hiddenWeights, this.biasedVector(inputs[i]));
			const hiddenOutput = this.activation(hiddenNetInput);
			const finalNetInput = NeuralNetwork.multiplyMatrixWithVector(this.outputWeights, this.biasedVector(hiddenOutput));
			const finalOutput = this.activation(finalNetInput);

			// From here on out its only made for a 2-2-1 network (Only temporary)
			const errorGradient = finalOutput[0] - expectedOutputs[i][0];
			const outputGradient = finalNetInput.map(val => this.sigmoidDerivative(val))[0];

			weight5Gradient += errorGradient * outputGradient * hiddenOutput[0];
			weight6Gradient += errorGradient * outputGradient * hiddenOutput[1];

			const derivedHiddenNetInput = Array.from(hiddenNetInput).map(val => this.sigmoidDerivative(val));
			weight1Gradient += errorGradient * outputGradient * this.outputWeights[0][0] * derivedHiddenNetInput[0] * inputs[i][0];
			weight2Gradient += errorGradient * outputGradient * this.outputWeights[0][0] * derivedHiddenNetInput[0] * inputs[i][1];
			weight3Gradient += errorGradient * outputGradient * this.outputWeights[0][1] * derivedHiddenNetInput[1] * inputs[i][0];
			weight4Gradient += errorGradient * outputGradient * this.outputWeights[0][1] * derivedHiddenNetInput[1] * inputs[i][1];

			bias1Gradient += errorGradient * outputGradient * this.outputWeights[0][0] * derivedHiddenNetInput[0];
			bias2Gradient += errorGradient * outputGradient * this.outputWeights[0][1] * derivedHiddenNetInput[1];
			bias3Gradient += errorGradient * outputGradient;
		}
		this.hiddenWeights[0][0] -= this.learningRate * weight1Gradient;
		this.hiddenWeights[0][1] -=	this.learningRate * weight2Gradient;
		this.hiddenWeights[1][0] -= this.learningRate * weight3Gradient;
		this.hiddenWeights[1][1] -= this.learningRate * weight4Gradient;

		this.outputWeights[0][0] -= this.learningRate * weight5Gradient;
		this.outputWeights[0][1] -= this.learningRate * weight6Gradient;

		this.hiddenWeights[0][2] -= this.learningRate * bias1Gradient;
		this.hiddenWeights[1][2] -= this.learningRate * bias2Gradient;
		this.outputWeights[0][2] -= this.learningRate * bias3Gradient;
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
