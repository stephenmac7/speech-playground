<script lang="ts">
	let {
		learnerFeatures,
		learnerFrame,
		referenceFeatures,
		referenceFrame
	}: {
		learnerFeatures: number[][];
		learnerFrame: number;
		referenceFeatures?: number[][];
		referenceFrame?: number;
	} = $props();

	let canvasEl: HTMLCanvasElement | undefined = $state();
	let showLabels = $state(false);

	// --- Constants ---
	const POINT_LABELS = ['Tongue Body', 'Blade', 'Tip', 'Upper Lip', 'Lower Lip', 'Incisor'];
	const AUDIO_COLOR = '#b2df8a';
	const QUERY_COLOR = '#fdbf6f';
	const MODEL_COLOR = '#a6cee3';
	const POINT_RADIUS = 6;
	const MARKER_SIZE = 7;
	const LINE_WIDTH = 2.5;
	const DATA_MARGIN = 0.1;

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

		// 1. Reshape current frame data
		const P1 = reshape(learnerFeatures[learnerFrame]);
		const P2 = referenceFeatures && referenceFrame != null ? reshape(referenceFeatures[referenceFrame]) : [];

		if (P1.length === 0 && P2.length === 0) return;

		// 2. Compute global bounds across all frames
		const allFrames = [...learnerFeatures, ...(referenceFeatures ?? [])];
		let gxmin = Infinity, gxmax = -Infinity, gymin = Infinity, gymax = -Infinity;
		for (const frame of allFrames) {
			if (!frame || frame.length !== 12) continue;
			for (let i = 0; i < 12; i += 2) {
				const x = frame[i], y = frame[i + 1];
				if (x < gxmin) gxmin = x;
				if (x > gxmax) gxmax = x;
				if (y < gymin) gymin = y;
				if (y > gymax) gymax = y;
			}
		}

		let xr = (gxmax - gxmin) || 1.0;
		let yr = (gymax - gymin) || 1.0;

		const xmin = gxmin - xr * DATA_MARGIN;
		const xmax = gxmax + xr * DATA_MARGIN;
		const ymin = gymin - yr * DATA_MARGIN;
		const ymax = gymax + yr * DATA_MARGIN;

		xr = xmax - xmin;
		yr = ymax - ymin;

		// 3. Set canvas resolution
		const dpr = window.devicePixelRatio || 1;
		const rect = canvasEl.getBoundingClientRect();

		if (rect.width === 0 || rect.height === 0) return;

		// 4. Set canvas aspect ratio to match data, then compute uniform scale
		const dataAspect = xr / yr;
		canvasEl.style.aspectRatio = `${dataAspect}`;
		const rect2 = canvasEl.getBoundingClientRect();
		canvasEl.width = rect2.width * dpr;
		canvasEl.height = rect2.height * dpr;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

		const canvasWidth2 = rect2.width;
		const canvasHeight2 = rect2.height;

		const scale = Math.min(canvasWidth2 / xr, canvasHeight2 / yr);

		const dataMidX = xmin + xr / 2;
		const dataMidY = ymin + yr / 2;
		const canvasMidX = canvasWidth2 / 2;
		const canvasMidY = canvasHeight2 / 2;

		// 5. Define transform functions
		const tx = (dataX: number) => (dataX - dataMidX) * scale + canvasMidX;
		const ty = (dataY: number) => (dataY - dataMidY) * -scale + canvasMidY;

		// 6. Clear and Draw
		ctx.fillStyle = '#ffffff';
		ctx.fillRect(0, 0, canvasWidth2, canvasHeight2);

		if (P2.length > 0) {
			ctx.fillStyle = MODEL_COLOR;
			P2.forEach((point) => {
				drawCircleMarker(ctx, tx(point[0]), ty(point[1]), POINT_RADIUS);
			});

			ctx.strokeStyle = QUERY_COLOR;
			ctx.lineWidth = LINE_WIDTH;
			P1.forEach((point) => {
				drawXMarker(ctx, tx(point[0]), ty(point[1]), MARKER_SIZE);
			});

			const legendX = 8;
			const legendSpacing = 18;
			const legendY = canvasHeight2 - legendSpacing - 12;
			ctx.font = '13px sans-serif';
			ctx.textBaseline = 'middle';

			ctx.fillStyle = MODEL_COLOR;
			drawCircleMarker(ctx, legendX + POINT_RADIUS, legendY, POINT_RADIUS);
			ctx.fillStyle = '#333';
			ctx.fillText('Model', legendX + POINT_RADIUS * 2 + 6, legendY);

			ctx.strokeStyle = QUERY_COLOR;
			ctx.lineWidth = LINE_WIDTH;
			drawXMarker(ctx, legendX + MARKER_SIZE, legendY + legendSpacing, MARKER_SIZE);
			ctx.fillStyle = '#333';
			ctx.fillText('Query', legendX + MARKER_SIZE * 2 + 6, legendY + legendSpacing);
		} else {
			ctx.fillStyle = AUDIO_COLOR;
			P1.forEach((point) => {
				drawCircleMarker(ctx, tx(point[0]), ty(point[1]), POINT_RADIUS);
			});
		}

		if (showLabels) {
			ctx.font = '11px sans-serif';
			ctx.fillStyle = '#333';
			ctx.textBaseline = 'bottom';
			ctx.textAlign = 'center';
			P1.forEach((point, i) => {
				ctx.fillText(POINT_LABELS[i], tx(point[0]), ty(point[1]) - POINT_RADIUS - 2);
			});
		}
	});
</script>

<canvas bind:this={canvasEl}></canvas>
<label><input type="checkbox" bind:checked={showLabels} /> Labels</label>

<style>
	canvas {
		width: 100%;
		max-width: 500px;
		height: auto;
		display: block;
		margin: 0 auto;
	}
	label {
		display: block;
		text-align: center;
		font-size: 0.85rem;
		margin-top: 0.25rem;
	}
</style>
