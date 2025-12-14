<script lang="ts">
	let { learnerFeatures, referenceFeatures } = $props();

	let canvasEl: HTMLCanvasElement | undefined = $state();

	// --- Constants ---
	const LEARNER_COLOR = '#E69F00'; // Orange
	const TEMPLATE_COLOR = '#009E73'; // Green
	const POINT_RADIUS = 6;
	const MARKER_SIZE = 7;
	const LINE_WIDTH = 2.5;
	const DATA_MARGIN = 0.05;

	// --- Drawing Helpers (using const arrow functions) ---

	/**
	 * Reshapes a 12-element array into 6 [x, y] pairs.
	 */
	const reshape = (numbers: number[]) => {
		if (!numbers || numbers.length !== 12) return [];
		const points = [];
		for (let i = 0; i < 12; i += 2) {
			points.push([numbers[i], numbers[i + 1]]);
		}
		return points;
	};

	/**
	 * Draws an 'x' marker on the canvas.
	 */
	const drawXMarker = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
		ctx.beginPath();
		ctx.moveTo(x - size, y - size);
		ctx.lineTo(x + size, y + size);
		ctx.moveTo(x + size, y - size);
		ctx.lineTo(x - size, y + size);
		ctx.stroke();
	};

	/**
	 * Draws a filled circle on the canvas.
	 */
	const drawCircleMarker = (
		ctx: CanvasRenderingContext2D,
		x: number,
		y: number,
		radius: number
	) => {
		ctx.beginPath();
		ctx.arc(x, y, radius, 0, 2 * Math.PI);
		ctx.fill();
	};

	// --- Reactive Drawing Effect ---
	// This runs whenever props change or the canvas resizes
	$effect(() => {
		if (!canvasEl) return;

		const ctx = canvasEl.getContext('2d');
		if (!ctx) return;

		// 1. Reshape data
		const P1 = reshape(learnerFeatures);
		const P2 = reshape(referenceFeatures);

		if (P1.length === 0 && P2.length === 0) {
			// Minimal error handling: just don't draw if data is bad
			return;
		}

		// 2. Set canvas resolution (for crisp rendering)
		const dpr = window.devicePixelRatio || 1;
		const rect = canvasEl.getBoundingClientRect();

		if (rect.width === 0 || rect.height === 0) {
			return; // Avoid drawing if canvas is not visible
		}

		canvasEl.width = rect.width * dpr;
		canvasEl.height = rect.height * dpr;
		ctx.scale(dpr, dpr);

		const canvasWidth = rect.width;
		const canvasHeight = rect.height;

		// 3. Find data bounds (Axis limits)
		const allPoints = [...P1, ...P2];
		const allX = allPoints.map((p) => p[0]);
		const allY = allPoints.map((p) => p[1]);

		let xmin = Math.min(...allX);
		let xmax = Math.max(...allX);
		let ymin = Math.min(...allY);
		let ymax = Math.max(...allY);

		let xr = xmax - xmin || 1.0;
		let yr = ymax - ymin || 1.0;

		xmin -= xr * DATA_MARGIN;
		xmax += xr * DATA_MARGIN;
		ymin -= yr * DATA_MARGIN;
		ymax += yr * DATA_MARGIN;

		xr = xmax - xmin;
		yr = ymax - ymin;

		// 4. Calculate scaling (to maintain aspect ratio)
		const dataRange = Math.max(xr, yr);
		const plotSize = Math.min(canvasWidth, canvasHeight);
		const scale = plotSize / dataRange;

		const dataMidX = xmin + xr / 2;
		const dataMidY = ymin + yr / 2;
		const canvasMidX = canvasWidth / 2;
		const canvasMidY = canvasHeight / 2;

		// 5. Define transform functions
		const tx = (dataX: number) => (dataX - dataMidX) * scale + canvasMidX;
		const ty = (dataY: number) => (dataY - dataMidY) * -scale + canvasMidY;

		// 6. Clear and Draw
		// Clear with white background
		ctx.fillStyle = '#ffffff';
		ctx.fillRect(0, 0, canvasWidth, canvasHeight);

		// Draw P1 (Learner, 'x')
		ctx.strokeStyle = LEARNER_COLOR;
		ctx.lineWidth = LINE_WIDTH;
		P1.forEach((point) => {
			drawXMarker(ctx, tx(point[0]), ty(point[1]), MARKER_SIZE);
		});

		// Draw P2 (Template, 'o')
		ctx.fillStyle = TEMPLATE_COLOR;
		P2.forEach((point) => {
			drawCircleMarker(ctx, tx(point[0]), ty(point[1]), POINT_RADIUS);
		});
	});
</script>

<!-- Bind the canvas element to our script variable -->
<canvas bind:this={canvasEl}></canvas>

<style>
	/* Scoped styles for the component */
	canvas {
		width: 100%;
		max-width: 500px;
		height: auto;
		aspect-ratio: 1 / 1;
		display: block;
		margin: 0 auto;
	}
</style>
