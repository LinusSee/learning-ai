class Point {
	constructor(maxWidth, maxHeight) {
		this.x = Math.floor(Math.random() * maxWidth - maxWidth / 2);
		this.y = Math.floor(Math.random() * maxHeight - maxHeight / 2);
	}
}
