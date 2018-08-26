class Main {
	constructor() {
		this.xorNet = new NeuralNetwork([2, 2, 1], 0.5);
		this.andNet = new NeuralNetwork([2, 1], 0.5);
		this.orNet = new NeuralNetwork([2, 1], 0.5);
		this.bigXorNet = new NeuralNetwork([2, 2, 2, 1], 0.5);
		this.rlyBigXorNet = new NeuralNetwork([2, 6, 8, 6, 2, 1], 0.5);
		this.switchNet = new NeuralNetwork([2, 4, 3, 2], 0.5);

		this.gateInputs = [
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
		this.andOutputs = [
			[1],
			[0],
			[0],
			[0]
		];
		this.orOutputs = [
			[1],
			[1],
			[1],
			[0]
		];
		this.switchedOutputs = [
			[1, 1],
			[0, 1],
			[1, 0],
			[0, 0]
		];
	}

	evaluate() {
		document.getElementById("xorOutputOne").innerHTML = this.xorNet.feedForward(this.gateInputs[0]);
		document.getElementById("xorOutputTwo").innerHTML = this.xorNet.feedForward(this.gateInputs[1]);
		document.getElementById("xorOutputThree").innerHTML = this.xorNet.feedForward(this.gateInputs[2]);
		document.getElementById("xorOutputFour").innerHTML = this.xorNet.feedForward(this.gateInputs[3]);

		document.getElementById("andOutputOne").innerHTML = this.andNet.feedForward(this.gateInputs[0]);
		document.getElementById("andOutputTwo").innerHTML = this.andNet.feedForward(this.gateInputs[1]);
		document.getElementById("andOutputThree").innerHTML = this.andNet.feedForward(this.gateInputs[2]);
		document.getElementById("andOutputFour").innerHTML = this.andNet.feedForward(this.gateInputs[3]);

		document.getElementById("orOutputOne").innerHTML = this.orNet.feedForward(this.gateInputs[0]);
		document.getElementById("orOutputTwo").innerHTML = this.orNet.feedForward(this.gateInputs[1]);
		document.getElementById("orOutputThree").innerHTML = this.orNet.feedForward(this.gateInputs[2]);
		document.getElementById("orOutputFour").innerHTML = this.orNet.feedForward(this.gateInputs[3]);

		document.getElementById("bigXorOutputOne").innerHTML = this.bigXorNet.feedForward(this.gateInputs[0]);
		document.getElementById("bigXorOutputTwo").innerHTML = this.bigXorNet.feedForward(this.gateInputs[1]);
		document.getElementById("bigXorOutputThree").innerHTML = this.bigXorNet.feedForward(this.gateInputs[2]);
		document.getElementById("bigXorOutputFour").innerHTML = this.bigXorNet.feedForward(this.gateInputs[3]);

		document.getElementById("rlyBigXorOutputOne").innerHTML = this.rlyBigXorNet.feedForward(this.gateInputs[0]);
		document.getElementById("rlyBigXorOutputTwo").innerHTML = this.rlyBigXorNet.feedForward(this.gateInputs[1]);
		document.getElementById("rlyBigXorOutputThree").innerHTML = this.rlyBigXorNet.feedForward(this.gateInputs[2]);
		document.getElementById("rlyBigXorOutputFour").innerHTML = this.rlyBigXorNet.feedForward(this.gateInputs[3]);

		document.getElementById("switchOutputOneOne").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[0])[0] * 10000) / 10000;
		document.getElementById("switchOutputOneTwo").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[0])[1] * 10000) / 10000;
		document.getElementById("switchOutputTwoOne").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[1])[0] * 10000) / 10000;
		document.getElementById("switchOutputTwoTwo").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[0])[1] * 10000) / 10000;
		document.getElementById("switchOutputThreeOne").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[2])[0] * 10000) / 10000;
		document.getElementById("switchOutputThreeTwo").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[0])[1] * 10000) / 10000;
		document.getElementById("switchOutputFourOne").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[3])[0] * 10000) / 10000;
		document.getElementById("switchOutputFourTwo").innerHTML = Math.round(this.switchNet.feedForward(this.gateInputs[0])[1] * 10000) / 10000;
	}

	train(iterationCount) {
		for(let i = 0; i < iterationCount; i++) {
			/*this.xorNet.trainBatch(this.gateInputs, this.xorOutputs);
			this.andNet.trainBatch(this.gateInputs, this.andOutputs);
			this.orNet.trainBatch(this.gateInputs, this.orOutputs);
			//this.bigXorNet.trainBatch(this.gateInputs, this.xorOutputs);
			//this.rlyBigXorNet.trainBatch(this.gateInputs, this.xorOutputs);*/
			this.switchNet.trainBatch(this.gateInputs, this.switchedOutputs);
		}
		this.evaluate();
	}
}

main = new Main();
