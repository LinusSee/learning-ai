class Main {
	constructor() {
		this.nn = new NeuralNetwork(2, 2, 1, 0.5);
		this.xorInputs = [
			[ 1, 1],
			[ 1, 0],
			[ 0, 1],
			[ 0, 0]
		];
		this.xorOutputs = [
			[0],
			[1],
			[1],
			[0]
		];
	}

	evaluate() {
		document.getElementById("outputOne").innerHTML = this.nn.feedForward([this.xorInputs[0][0], this.xorInputs[0][1]]);
		document.getElementById("outputTwo").innerHTML = this.nn.feedForward([this.xorInputs[1][0], this.xorInputs[1][1]]);
		document.getElementById("outputThree").innerHTML = this.nn.feedForward([this.xorInputs[2][0], this.xorInputs[2][1]]);
		document.getElementById("outputFour").innerHTML = this.nn.feedForward([this.xorInputs[3][0], this.xorInputs[3][1]]);
	}

	train(iterationCount) {
		for(let i = 0; i < iterationCount; i++) {
			this.nn.trainBatch(this.xorInputs, this.xorOutputs);
		}
		this.evaluate();
		/*// TODO: Somehow train all training data at once
		for(let i = 0; i < iterationCount; i++) {
			for(let j = 0; j < this.xorValues.length; j++) {
				// This is not a correct usage of gradient descent (needs a fix)
				const val = this.xorValues[j];
				this.nn.train([val[0], val[1]], this.xorOutputs[j]);
			}
		}
		this.evaluate();*/
	}
}



main = new Main();