// Region utilities shared by routes

export type Region = {
	id?: string;
	start: number;
	end: number;
	content?: string;
	color?: string;
};

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
	segments: number[][], // size N
	alignmentMap?: number[],
	modelIndexMap?: number[]
): Region[] {
	const regions: Region[] = [];
	let currentRegion: { startFrame: number; scores: number[] } | undefined;
	let seenGoodFrames = 0;

	const createRegion = (region: { startFrame: number; endFrame: number; scores: number[] }) => {
		const avg = region.scores.reduce((a, b) => a + b, 0) / region.scores.length;

		let isPureInsertion = true;
		const validIndices: number[] = [];

		if (alignmentMap) {
			for (let k = region.startFrame; k < region.endFrame; k++) {
				const idx = alignmentMap[k];
				if (idx !== -1) {
					isPureInsertion = false;
					validIndices.push(idx);
				}
			}
		} else {
			isPureInsertion = false;
		}

		let color: string;
		if (isPureInsertion) {
			color = `rgba(255, 255, 0, 0.5)`;
		} else {
			const opacity = 0.8 * (1 - avg);
			color = `rgba(255, 0, 0, ${opacity})`;
		}

		let content = '';
		if (!isPureInsertion && validIndices.length > 0) {
			let mappedIndices = validIndices;
			if (modelIndexMap) {
				mappedIndices = validIndices.map((idx) => modelIndexMap[idx]);
			}

			let min = mappedIndices[0];
			let max = mappedIndices[0];
			for (const idx of mappedIndices) {
				if (idx < min) min = idx;
				if (idx > max) max = idx;
			}

			if (min === max) {
				content = String(min);
			} else {
				content = `${min}-${max}`;
			}
		}

		regions.push({
			id: `region-${region.startFrame}-${region.endFrame}`,
			start: segments[region.startFrame][0],
			end: segments[region.endFrame - 1][1],
			color,
			content
		});
	};

	scores.forEach((score, i) => {
		let isBad = false;
		if (alignmentMap && alignmentMap[i] === -1) {
			isBad = true;
		} else if (score < 0.5) {
			isBad = true;
		}

		if (isBad) {
			if (!currentRegion) {
				currentRegion = { startFrame: i, scores: [score] };
				seenGoodFrames = 0;
			} else {
				currentRegion.scores.push(score);
				seenGoodFrames = 0;
			}
		} else {
			// OK frame
			if (currentRegion && seenGoodFrames < 2) {
				currentRegion.scores.push(score);
				seenGoodFrames++;
			} else if (currentRegion) {
				// End region
				const lastBadFrame = i - seenGoodFrames;
				createRegion({
					startFrame: currentRegion.startFrame,
					endFrame: lastBadFrame,
					scores: currentRegion.scores.slice(
						0,
						lastBadFrame - (i - currentRegion.scores.length)
					)
				});
				currentRegion = undefined;
			}
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
	combineRegions: boolean = false,
	alignmentMap?: number[],
	modelIndexMap?: number[]
): Region[] {
	if (combineRegions) {
		return buildCombinedSegmentRegions(scores, segments, alignmentMap, modelIndexMap);
	}
	const regions: Region[] = [];
	scores.forEach((score, i) => {
		const isInsertion = alignmentMap && alignmentMap[i] === -1;
		if (score < 0.5 || isInsertion) {
			const opacity = 0.8 * (1 - score);
			const startFrame = i;
			const endFrame = i + 1;

			let content = '';
			if (!isInsertion && alignmentMap) {
				const originalIndex = alignmentMap[i];
				if (modelIndexMap && originalIndex !== -1) {
					content = String(modelIndexMap[originalIndex]);
				} else {
					content = String(originalIndex);
				}
			}

			regions.push({
				id: `region-${startFrame}-${endFrame}`,
				start: segments[startFrame][0],
				end: segments[endFrame - 1][1],
				color: isInsertion ? `rgba(255, 255, 0, 0.5)` : `rgba(255, 0, 0, ${opacity})`,
				content
			});
		}
	});
	return regions;
}

export function buildCombinedModelRegions(
	segments: number[][],
	coveredIndices: Set<number>
): { regions: Region[]; indexMap: number[] } {
	const regions: Region[] = [];
	const indexMap: number[] = new Array(segments.length).fill(-1);
	let currentStart = -1;
	let currentEnd = -1;
	let currentIsCovered = false;
	let regionIndex = 0;

	for (let i = 0; i < segments.length; i++) {
		const isCovered = coveredIndices.has(i);

		if (currentStart === -1) {
			// Start new region
			currentStart = i;
			currentEnd = i;
			currentIsCovered = isCovered;
		} else {
			if (isCovered === currentIsCovered) {
				// Continue region
				currentEnd = i;
			} else {
				// End previous region
				regions.push({
					id: `model-combined-${regionIndex}`,
					start: segments[currentStart][0],
					end: segments[currentEnd][1],
					content: regionIndex.toString(),
					color: currentIsCovered ? 'rgba(0, 0, 255, 0.2)' : 'rgba(255, 0, 0, 0.5)'
				});
				// Fill index map
				for (let j = currentStart; j <= currentEnd; j++) {
					indexMap[j] = regionIndex;
				}
				regionIndex++;

				// Start new region
				currentStart = i;
				currentEnd = i;
				currentIsCovered = isCovered;
			}
		}
	}

	// Push last region
	if (currentStart !== -1) {
		regions.push({
			id: `model-combined-${regionIndex}`,
			start: segments[currentStart][0],
			end: segments[currentEnd][1],
			content: regionIndex.toString(),
			color: currentIsCovered ? 'rgba(0, 0, 255, 0.2)' : 'rgba(255, 0, 0, 0.5)'
		});
		// Fill index map
		for (let j = currentStart; j <= currentEnd; j++) {
			indexMap[j] = regionIndex;
		}
	}

	return { regions, indexMap };
}
