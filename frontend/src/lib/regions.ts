// Region utilities shared by routes
// Note: DOM operations (document.createElement) occur only when these functions are called in the browser.

export type Region = { id: string; start: number; end: number; content: string; color: string };

export function buildContinuousRegions(
	allScores: number[],
	segments: number[][],
	trigger: number,
	min: number
): Region[] {
	if (allScores.length === 0) return [];

	const regions: Region[] = [];
	let current: { start: number; scores: number[] } | undefined;

	const createRegion = (startFrame: number, endFrame: number, scores: number[]) => {
		const startTime = segments[startFrame][0];
		const endTime = segments[endFrame - 1][1];
		const duration = endTime - startTime;
		if (duration < 0.1) return; // Ignore very short regions

		const avg = scores.reduce((a, b) => a + b, 0) / scores.length;

		// Calculate shade:
		// (min - avg) is the "badness" amount.
		// Divide by min to normalize it (0 -> 1 range, where 1 is worst (avg=0))
		// Multiply by 2.0 to amplify the effect (so even "half-bad" is fully opaque)
		// Clamp between 0 and 1, then apply 80% max opacity.
		const shade = ((min - avg) / min) * 2.0;
		const opacity = Math.max(0, Math.min(1, shade)) * 0.8;

		regions.push({
			id: `region-${startFrame}-${endFrame}`,
			start: startTime,
			end: endTime,
			color: `rgba(255, 0, 0, ${opacity})`,
			content: String(avg.toFixed(2))
		});
	};

	allScores.forEach((score, i) => {
		if (score < trigger) {
			if (!current) {
				// Start of a new region, backtrack to find the real start
				let j = i - 1;
				while (j >= 0 && allScores[j] < trigger) {
					j--;
				}
				// The last "good" frame was j, so the "bad" region started at j + 1
				current = { start: j + 1, scores: allScores.slice(j + 1, i) };
			}
			current.scores.push(score);
		} else if (score < min) {
			// Continue the region if it's below min but not below trigger
			current?.scores.push(score);
		} else if (current) {
			// End of the current region (score is >= min)
			createRegion(current.start, i, current.scores);
			current = undefined;
		}
	});

	// Handle a region that might be open at the end
	if (current) {
		createRegion(current.start, allScores.length, current.scores);
	}

	return regions;
}

export function buildCombinedSegmentRegions(
	scores: number[], // size N
	segments: number[][] // size N
): Region[] {
	const regions: Region[] = [];
	let currentRegion: { startFrame: number; scores: number[] } | undefined;
	let seenGoodFrames = 0;

	const createRegion = (region: { startFrame: number; endFrame: number; scores: number[] }) => {
		const avg = region.scores.reduce((a, b) => a + b, 0) / region.scores.length;
		const opacity = 0.8 * (1 - avg);
		regions.push({
			id: `region-${region.startFrame}-${region.endFrame}`,
			start: segments[region.startFrame][0],
			end: segments[region.endFrame - 1][1],
			color: `rgba(255, 0, 0, ${opacity})`,
			content: String(avg.toFixed(2))
		});
	};

	scores.forEach((score, i) => {
		if (score < 0.5) {
			if (!currentRegion) {
				currentRegion = { startFrame: i, scores: [] };
			}
			currentRegion.scores.push(score);
			seenGoodFrames = 0;
		} else if (currentRegion && seenGoodFrames < 2) {
			// Allow some patience to continue the region
			currentRegion.scores.push(score);
			seenGoodFrames++;
		} else if (currentRegion) {
			// End the current region
			const lastBadFrame = i - seenGoodFrames;
			createRegion({
				startFrame: currentRegion.startFrame,
				endFrame: lastBadFrame,
				scores: currentRegion.scores.slice(0, lastBadFrame - (i - currentRegion.scores.length))
			});
			currentRegion = undefined;
		}
	});

	// Handle a region that might be open at the end -- make sure to deal with last good frames
	if (currentRegion) {
		const lastBadFrame = scores.length - seenGoodFrames;
		createRegion({
			startFrame: currentRegion.startFrame,
			endFrame: lastBadFrame,
			scores: currentRegion.scores.slice(
				0,
				lastBadFrame - (scores.length - currentRegion.scores.length)
			)
		});
	}

	return regions;
}

export function buildSegmentRegions(
	scores: number[], // size N
	segments: number[][], // size N
	combineRegions: boolean = false
): Region[] {
	if (combineRegions) {
		return buildCombinedSegmentRegions(scores, segments);
	}
	const regions: Region[] = [];
	scores.forEach((score, i) => {
		if (score < 0.5) {
			const opacity = 0.8 * (1 - score);
			const startFrame = i;
			const endFrame = i + 1;
			regions.push({
				id: `region-${startFrame}-${endFrame}`,
				start: segments[startFrame][0],
				end: segments[endFrame - 1][1],
				color: `rgba(255, 0, 0, ${opacity})`,
				content: ''
			});
		}
	});
	return regions;
}
