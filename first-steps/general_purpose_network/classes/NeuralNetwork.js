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
		console.table(this.weights[0]);
		console.table(this.weights[1]);
	}

	feedForward(inputVector) {
		let currentOutput = inputVector;
		for(let i = 0; i < this.weights.length; i++) {
			currentOutput = this.activation(math.multiply(this.weights[i], this.biasedVector(currentOutput)));
		}
		return currentOutput;
	}

	train(input, expectedOutput) {
		// Currently hardcoded for 3Layer network (IHO)
		const net1 = math.multiply(this.weights[0], this.biasedVector(input));
		const out1 = this.activation(net1);
		const net2 = math.multiply(this.weights[1], this.biasedVector(out1));
		const out2 = this.activation(net2);

		const deriv1 = out1.map(val => val * (1 - val));	// Derivatives for layer 1 (hidden)
		const deriv2 = out2.map((val, index, arr) => (expectedOutput[index] - arr[index]) * (val * (1 - val)));	// Derivatives for layer 2 (output)

		// Backpropagation starts here
		const gradientsOutput = math.multiply(math.transpose([out2]), [this.biasedVector(deriv1)]);	// Ugly conversions because of mathjs
		const continueGrad = math.transpose(math.multiply(math.transpose(this.weights[1]), math.diag(out2)));
		const gradientsHidden = math.multiply(math.transpose([out1]), [this.biasedVector(input)]);

		const reducedGradientHidden = math.multiply(gradientsHidden, this.learningRate);
		const reducedGradientOutput = math.multiply(gradientsOutput, this.learningRate);
		this.weights[0] = math.subtract(this.weights[0], reducedGradientHidden);
		this.weights[1] = math.subtract(this.weights[1], reducedGradientOutput);
		console.log("Seperator");
		console.table(reducedGradientHidden);
		console.table(reducedGradientOutput);
	}

	trainBatch(inputs, expectedOutputs) {
		let hiddenGradient = new Array(this.neuronsPerLayer[1]).fill().map(val => new Array(this.neuronsPerLayer[0] + 1).fill(0));
		let outputGradient = new Array(this.neuronsPerLayer[2]).fill().map(val => new Array(this.neuronsPerLayer[1] + 1).fill(0));

		for(let i = 0; i < inputs.length; i++) {
			const input = inputs[i];
			const expectedOutput = expectedOutputs[i];
			const net1 = math.multiply(this.weights[0], this.biasedVector(input));
			const out1 = this.activation(net1);
			const net2 = math.multiply(this.weights[1], this.biasedVector(out1));
			const out2 = this.activation(net2);

			const deriv1 = out1.map(val => val * (1 - val));	// Derivatives for layer 1 (hidden)
			const deriv2 = out2.map((val, index, arr) => (arr[index] - expectedOutput[index]) * (val * (1 - val)));	// Derivatives for layer 2 (output)

			// Backpropagation starts here
			const gradientsOutput = math.multiply(math.transpose([out2]), [this.biasedVector(deriv1)]);	// Ugly conversions because of mathjs
			const continueGrad = math.transpose(math.multiply(math.transpose(this.weights[1]), math.diag(out2)));
			const gradientsHidden = math.multiply(math.transpose([out1]), [this.biasedVector(input)]);

			hiddenGradient = math.add(hiddenGradient, gradientsHidden);
			outputGradient = math.add(outputGradient, gradientsOutput);
		}
		const reducedGradientHidden = math.multiply(hiddenGradient, this.learningRate);
		const reducedGradientOutput = math.multiply(outputGradient, this.learningRate);
		this.weights[0] = math.subtract(this.weights[0], reducedGradientHidden);
		this.weights[1] = math.subtract(this.weights[1], reducedGradientOutput);
		console.log("Separator");
		console.table(reducedGradientHidden);
		console.table(reducedGradientOutput);
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
