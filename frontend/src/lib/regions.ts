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
	min: number,
	discretize: boolean
): Region[] {
	if (allScores.length === 0) return [];

	const regions: Region[] = [];
	let current: { start: number; scores: number[] } | undefined;

	allScores.forEach((score, i) => {
		if (score < trigger) {
			if (!current) {
				for (let j = i - 1; j >= 0; j--) {
					if (allScores[j] >= trigger) {
						current = { start: j + 1, scores: allScores.slice(j + 1, i) };
						break;
					}
				}
				current ??= { start: 0, scores: [] };
			}
			current.scores.push(score);
		} else if (score < min) {
			current?.scores.push(score);
		} else if (current) {
			if (current.scores.length * frameDur >= 0.1) {
				const avg = current.scores.reduce((a, b) => a + b, 0) / current.scores.length;
				const shade = ((min - avg) / Math.abs(min)) * (discretize ? 1 : 2.0);
				const opacity = Math.max(0, Math.min(1, shade)) * 0.8;
				regions.push({
					start: current.start * frameDur,
					end: i * frameDur,
					color: `rgba(255, 0, 0, ${opacity})`,
					content: labelSpan(avg.toFixed(2))
				});
			}
			current = undefined;
		}
	});

	if (current) {
		const avg = current.scores.reduce((a, b) => a + b, 0) / current.scores.length;
		const shade = ((min - avg) / Math.abs(min)) * (discretize ? 1 : 2.0);
		const opacity = Math.max(0, Math.min(1, shade)) * 0.8;
		regions.push({
			start: current.start * frameDur,
			end: allScores.length * frameDur,
			color: `rgba(255, 0, 0, ${opacity})`,
			content: labelSpan(avg.toFixed(2))
		});
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
