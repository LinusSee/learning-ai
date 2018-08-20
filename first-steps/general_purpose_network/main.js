class Main {
	constructor() {
		this.nn = new NeuralNetwork([2, 2, 1], 0.5);

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
		]
	}

	evaluate() {
		document.getElementById("outputOne").innerHTML = this.nn.feedForward(this.xorInputs[0]);
		document.getElementById("outputTwo").innerHTML = this.nn.feedForward(this.xorInputs[1]);
		document.getElementById("outputThree").innerHTML = this.nn.feedForward(this.xorInputs[2]);
		document.getElementById("outputFour").innerHTML = this.nn.feedForward(this.xorInputs[3]);
	}

	train(iterationCount) {
		for(let i = 0; i < iterationCount; i++) {
			this.nn.trainBatch(this.xorInputs, this.xorOutputs);
		}
		this.evaluate();
	}
}

main = new Main();
