class Main {
	constructor() {
		this.nn = new NeuralNetwork(2, 2, 1, 0.5);
		// Dataset (in this case all possible inputs)
		this.xorInputs = [
			[ 1, 1],
			[ 1, 0],
			[ 0, 1],
			[ 0, 0]
		];
		// Solution to the dataset (in this case all possible outputs)
		this.xorOutputs = [
			[0],
			[1],
			[1],
			[0]
		];
	}

	// Evaluates the current solution of the network for each possible input and displays them in the webpage
	evaluate() {
		document.getElementById("outputOne").innerHTML = this.nn.feedForward([this.xorInputs[0][0], this.xorInputs[0][1]]);
		document.getElementById("outputTwo").innerHTML = this.nn.feedForward([this.xorInputs[1][0], this.xorInputs[1][1]]);
		document.getElementById("outputThree").innerHTML = this.nn.feedForward([this.xorInputs[2][0], this.xorInputs[2][1]]);
		document.getElementById("outputFour").innerHTML = this.nn.feedForward([this.xorInputs[3][0], this.xorInputs[3][1]]);
	}

	// Trains the neural network with all input and output for iterationCount times
	train(iterationCount) {
		for(let i = 0; i < iterationCount; i++) {
			this.nn.trainBatch(this.xorInputs, this.xorOutputs);
		}
		this.evaluate();
	}
}



main = new Main();
