class NeuralNetwork {

	// neuronsPerLayer = list of neuronCounts for each layer beginning with the input layer
	constructor(neuronsPerLayer) {
		this.weights = new Array(neuronsPerLayer.length - 1).fill();
		for(let i = 0; i < this.weights.length; i++) {
			this.weights[i] = math.matrix(new Array(neuronsPerLayer[i + 1]).fill().map(val => new Array(neuronsPerLayer[i]).fill().map(val => Math.random() * 2 - 1)));
		}
		console.table(this.weights);
	}

	feedForward(inputVector) {
		let currentOutput = inputVector;
		for(let i = 0; i < this.weights.length; i++) {
			currentOutput = this.activation(math.multiply(this.weights[i], currentOutput));
		}
		return currentOutput;
	}

	activation(vector) {
		return math.map(vector, (val) => ( this.sigmoid(val) ));
	}

	sigmoid(value) {
		return 1 / (1 + Math.pow(Math.E, - value));
	}
}
