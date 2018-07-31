var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

// Not the best learning rate, but like that more training iterations are needed (better for display)
var perceptron = new Perceptron(0.01);

// Draw a straight line
var line = new StraightLine(1, 0);
ctx.moveTo(0, line.evaluate(0));
ctx.lineTo(canvas.width, line.evaluate(canvas.height));
ctx.stroke();


// Draw a hundred points onto the canvas
var points = new Array(100).fill().map(val => (new Point(canvas.width, canvas.height)));
drawPoints();

// Draws all points on the canvas
// Color is red for incorrect points and green for valid points
function drawPoints() {
	for(point of points) {
		ctx.beginPath();
		ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
		if(perceptron.evaluate([point.x, point.y]) === lineActivationFunction(line.pointIsAbove(point.x, point.y))) {
			ctx.fillStyle = "#00FF00";
		} else {
			ctx.fillStyle = "#FF0000";
		}
		ctx.fill();
		ctx.stroke();
	}
}

function lineActivationFunction(pointIsAbove) {
	if(pointIsAbove) {
		return 1;
	} else {
		return -1;
	}
}

function trainCanvas() {
	for(point of points) {
		perceptron.train([point.x, point.y], lineActivationFunction(line.pointIsAbove(point.x, point.y)));
	}
	drawPoints();
}

