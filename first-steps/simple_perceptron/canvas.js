var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

// Not the best learning rate, but like that more training iterations are needed (better for display)
var perceptron = new Perceptron(0.005, 3);
var bias = 1;

// Draw a straight line
var mainLine = new StraightLine(1, 0);


drawCartesianCoordinateSystem(ctx, canvas.width, canvas.height);
// Draw a hundred points onto the canvas
var points = new Array(100).fill().map(val => (new Point(canvas.width, canvas.height)));
clearCanvasAndRedraw(canvas, [mainLine, approximatedStraightLine(perceptron.weights)], points);


function clearCanvasAndRedraw(canvas, lines, points) {
	var context = ctx;
	context.clearRect(0, 0, canvas.width, canvas.height);

	for(line of lines) {
		drawStraightLine(context, canvas.width, canvas.height, line);
	}
	drawCartesianCoordinateSystem(context, canvas.width, canvas.height);
	drawPoints(context, canvas.width, canvas.height, points);
}

function drawCartesianCoordinateSystem(context, width, height) {
	// y-axis
	context.moveTo(width / 2, 0);
	context.lineTo(width / 2, height);
	context.stroke();

	// x- axis
	context.moveTo(0, height / 2);
	context.lineTo(width, height / 2);
	context.stroke();

	// Origin
	context.beginPath();
	context.arc(width / 2, height / 2, 5, 0, 2 * Math.PI);
	context.stroke();
}


function drawStraightLine(context, height, width, line) {
	context.moveTo(0, height / 2 - line.evaluate(- width / 2));
	context.lineTo(width, height / 2 - line.evaluate(width / 2));
	context.stroke();
}

// Draws all points on the canvas
// Color is red for incorrect points and green for valid points
function drawPoints(context, width, height, points) {
	for(point of points) {
		var canvasPoint = mapPointToCanvas(point, width, height);
		context.beginPath();
		context.arc(canvasPoint.x, canvasPoint.y, 3, 0, 2 * Math.PI);
		if(perceptron.evaluate([point.x, point.y, bias]) === lineActivationFunction(mainLine.pointIsAbove(point.x, point.y))) {
			context.fillStyle = "#00FF00";
		} else {
			context.fillStyle = "#FF0000";
		}
		context.fill();
		context.stroke();
	}
}

function approximatedStraightLine(weights) {
	return new StraightLine(-weights[0] / weights[1], -weights[2] / weights[1]);
}

function mapPointToCanvas(point, canvasWidth, canvasHeight) {
	var newPoint = new Point(canvasWidth, canvasHeight);
	newPoint.x = point.x + canvasWidth / 2;
	newPoint.y = canvasHeight / 2 - point.y;
	return newPoint;
}

function lineActivationFunction(pointIsAbove) {
	if(pointIsAbove) {
		return 1;
	} else {
		return -1;
	}
}

function trainCanvas() {
	console.log("Weights before:", perceptron.weights);
	for(point of points) {
		perceptron.train([point.x, point.y, bias], lineActivationFunction(mainLine.pointIsAbove(point.x, point.y)));
	}
	console.log("Weights after:", perceptron.weights);
	clearCanvasAndRedraw(canvas, [mainLine, approximatedStraightLine(perceptron.weights)], points);
}

