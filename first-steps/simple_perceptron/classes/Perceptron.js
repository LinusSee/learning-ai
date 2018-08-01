class Perceptron {
	constructor(learningRate, weightCount) {
		this.weights = new Array(weightCount).fill().map(e => (Math.random()*2) - 1);
		this.learningRate = learningRate;
	}

	evaluate(input) {
		var sum = 0;
		for(var i = 0; i < this.weights.length - 1; i++) {
			sum += this.weights[i] * input[i];
		}
		return this.activationFunction(sum);
	}

	train(input, expectedResult) {
		var result = this.evaluate(input);
		var error = expectedResult - result;

		for(var i = 0; i < this.weights.length; i++) {
			this.weights[i] += error * input[i] * this.learningRate;
		}
	}


	activationFunction(evaluationResult) {
		if(evaluationResult >= 0) {
			return 1;
		} else {
			return -1;
		}
	}
}
