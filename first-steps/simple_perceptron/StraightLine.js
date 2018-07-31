class StraightLine {
	constructor(slope, displacement) {
		this.slope = slope;
		this.displacement = displacement;
	}

	evaluate(x) {
		return (this.slope * x) + this.displacement;
	}

	pointIsAbove(x, y) {
		return y >= this.evaluate(x);
	}
}
