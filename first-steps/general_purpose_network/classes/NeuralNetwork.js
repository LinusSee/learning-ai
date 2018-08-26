class NeuralNetwork {

	// neuronsPerLayer = list of neuronCounts for each layer beginning with the input layer
	constructor(neuronsPerLayer, learningRate) {
		this.learningRate = learningRate;
		this.neuronsPerLayer = neuronsPerLayer;
		this.weights = new Array(neuronsPerLayer.length - 1).fill();
		for(let i = 0; i < this.weights.length; i++) {
			this.weights[i] = new Array(neuronsPerLayer[i + 1]).fill().map(val => new Array(neuronsPerLayer[i] + 1).fill().map(val => Math.random() * 2 - 1));
		}
		//console.table(this.weights);
		//console.table(this.weights[0]);
		//console.table(this.weights[1]);
	}

	feedForward(inputVector) {
		let currentOutput = inputVector;
		for(let i = 0; i < this.weights.length; i++) {
			currentOutput = this.activation(math.multiply(this.weights[i], this.biasedVector(currentOutput)));
		}
		return currentOutput;
	}

	temp(target, actual) {
		return (actual - target) * actual * (1 - actual);
	}

	trainBatch(inputs, expectedOutputs) {
		const gradients = [];

		for(let x = 0; x < inputs.length; x++) {
			const input = inputs[x];
			const expectedOutput = expectedOutputs[x];

			const netValues = [];
			const outValues = [];

			outValues.push(input);
			for(let i = 0; i < this.weights.length; i++) {
				const net = math.multiply(this.weights[i], this.biasedVector(outValues[i]));
				const out = this.activation(net);
				netValues.push(net);
				outValues.push(out);
			}

			const derivatives = outValues.slice(1).map((outArr, index, arr) => outArr.map((val, curIndex) => {
				const temp = val * (1 - val);
				if(index == (arr.length - 1)) {
					return temp * (val - expectedOutput[curIndex]);
				} else {
					return temp;
				}
			}));

			// Backprop
			gradients.push(math.multiply(math.transpose([derivatives[derivatives.length - 1]]), [this.biasedVector(outValues[outValues.length - 2])]));
			for(let i = outValues.length - 1; i > 1; i--) {
				//console.log("Weight:", this.weights[i - 1]);
				const step1 = math.multiply(math.transpose(this.weightsWithoutBias(this.weights[i - 1])), derivatives[i - 1]);
				//console.log("Step1:", step1);
				//console.table(step1);
				//console.log("Deriv2:", derivatives[i - 2]);
				const step2  = math.dotMultiply(step1, derivatives[i - 2]);
				derivatives[i - 2] = step2;
				//console.log("Step2:", step2);
				//console.log("OutVal", outValues[i - 2]);
				const gradient = math.multiply(math.transpose([step2]), [this.biasedVector(outValues[i - 2])]);
				gradients.push(gradient);
			}
		}
		const reducedGradients = gradients.map(val => math.multiply(val, this.learningRate));
		//console.log("Gradients:", reducedGradients);
		for(let i = 0, j = reducedGradients.length - 1; j >= 0; i++, j--) {
			if(i % this.weights.length == 0) {
				i = 0;
				//console.log("Reset");
			}
			this.weights[i] = math.subtract(this.weights[i], reducedGradients[j]);
		}
	}

	weightsWithoutBias(matrix) {
		const newMatrix = [];
		for(let x = 0; x < matrix.length; x++) {
			const row = [];
			for(let y = 0; y < matrix[x].length - 1; y++) {
				row.push(matrix[x][y]);
			}
			newMatrix.push(row);
		}
		return newMatrix;
	}

	activation(vector) {
		return math.map(vector, (val) => ( this.sigmoid(val) ));
	}

	sigmoid(value) {
		return 1 / (1 + Math.pow(Math.E, - value));
	}

	biasedVector(vector) {
		//const newSize = vector._size[0] + 1;
		//return vector.resize([newSize], 1.0);
		const newVector = Array.from(vector);
		newVector.push(1.0);
		return newVector;
	}
}
