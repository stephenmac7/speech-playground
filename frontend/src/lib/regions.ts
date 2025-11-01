// Region utilities shared by routes
// Note: DOM operations (document.createElement) occur only when these functions are called in the browser.

export type Region = { start: number; end: number; content: HTMLElement; color: string };

export type SylberResult = {
	scores: number[];
	xsegments: number[][];
	ysegments: number[][];
	y_to_x_mappings: number[][];
};

export function labelSpan(text: string): HTMLElement {
	const el = document.createElement('span');
	el.textContent = text;
	return el;
}

export function buildContinuousRegions(
    allScores: number[],
    frameDur: number,
    trigger: number,
    min: number
): Region[] {
    if (allScores.length === 0) return [];

    const regions: Region[] = [];
    let current: { start: number; scores: number[] } | undefined;

    const createRegion = (startFrame: number, endFrame: number, scores: number[]) => {
        const duration = scores.length * frameDur;
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
            start: startFrame * frameDur,
            end: endFrame * frameDur,
            color: `rgba(255, 0, 0, ${opacity})`,
            content: labelSpan(avg.toFixed(2))
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

export function buildSyllableRegions(result: SylberResult): Region[] {
	const regions: Region[] = [];
	result.scores.forEach((score, i) => {
		const opacity = 0.8 * (1 - score);
		regions.push({
			start: result.ysegments[i][0],
			end: result.ysegments[i][1],
			color: `rgba(255, 0, 0, ${opacity})`,
			content: labelSpan(result.y_to_x_mappings[i].map((idx) => idx.toString()).join(', '))
		});
	});
	return regions;
}
