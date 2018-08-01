var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

// Not the best learning rate, but like that more training iterations are needed (better for display)
var perceptron = new Perceptron(0.005, 3);
var bias = 1;

// Draw a straight line
var line = new StraightLine(1, 0);
ctx.moveTo(0, canvas.height / 2 - line.evaluate(- canvas.width / 2));
ctx.lineTo(canvas.width, canvas.height / 2 - line.evaluate(canvas.width / 2));
ctx.stroke();


drawCartesianCoordinateSystem(ctx, canvas.width, canvas.height);
// Draw a hundred points onto the canvas
var points = new Array(100).fill().map(val => (new Point(canvas.width, canvas.height)));
drawPoints();


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

// Draws all points on the canvas
// Color is red for incorrect points and green for valid points
function drawPoints() {
	for(point of points) {
		var canvasPoint = mapPointToCanvas(point, canvas.width, canvas.height);
		ctx.beginPath();
		ctx.arc(canvasPoint.x, canvasPoint.y, 3, 0, 2 * Math.PI);
		if(perceptron.evaluate([point.x, point.y, bias]) === lineActivationFunction(line.pointIsAbove(point.x, point.y))) {
			ctx.fillStyle = "#00FF00";
		} else {
			ctx.fillStyle = "#FF0000";
		}
		ctx.fill();
		ctx.stroke();
	}
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
	for(point of points) {
		perceptron.train([point.x, point.y, bias], lineActivationFunction(line.pointIsAbove(point.x, point.y)));
	}
	drawPoints();
}

