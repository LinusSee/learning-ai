class Main {
	constructor() {
		this.nn = new NeuralNetwork(2, 2, 1, 0.5);
		this.xorValues = [
			[ 1, 1, 0],
			[ 1, 0, 1],
			[ 0, 1, 1],
			[ 0, 0, 0]
		];
	}

	evaluate() {
		document.getElementById("outputOne").innerHTML = this.nn.feedForward([this.xorValues[0][0], this.xorValues[0][1]]);
		document.getElementById("outputTwo").innerHTML = this.nn.feedForward([this.xorValues[1][0], this.xorValues[1][1]]);
		document.getElementById("outputThree").innerHTML = this.nn.feedForward([this.xorValues[2][0], this.xorValues[2][1]]);
		document.getElementById("outputFour").innerHTML = this.nn.feedForward([this.xorValues[3][0], this.xorValues[3][1]]);
	}

	train(iterationCount) {
		// TODO: Somehow train all training data at once
		for(let i = 0; i < iterationCount; i++) {
			for(let j = 0; j < this.xorValues.length; j++) {
				// This is not a correct usage of gradient descent (needs a fix)
				const val = this.xorValues[j];
				this.nn.train([val[0], val[1]], [val[2]]);
			}
		}
		this.evaluate();
	}
}



main = new Main();
