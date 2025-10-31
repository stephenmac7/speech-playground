<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';

	let { tracks, active } = $props();

	let audio = $state(tracks['Audio']);
	let modelAudio = $state(tracks['Model']);
	$effect(() => {
		if (active) {
			audio = tracks['Audio'];
			modelAudio = tracks['Model'];
		}
	});

	let convertVoice = $state(false);
	let voiceConversionModel = $state('kanade-25hz');
	let convertedAudio = $state<Blob | undefined>();
	$effect(() => {
		if (!convertVoice) convertedAudio = undefined;
	});

	let comparisonMode = $state<'continuous' | 'syllable'>('continuous');

	// Fixed rate diff state
	let encoder = $state('hubert');
	let discretize = $state(true);
	let dpdp = $state(true);
	let gamma = $state('0.2');
	let frameDuration = $state(0.02);
	let scores = $state<number[]>([]);

	// Syllable diff state
	let sylberResult:
		| { scores: number[]; xsegments: number[][]; ysegments: number[][]; y_to_x_mappings: number[][] }
		| undefined = $state();

	let loading = $state(false);

	const modelRegions = $derived.by(() => {
		if (comparisonMode === 'syllable') {
			const regions: { start: number; end: number; content: HTMLElement; color: string }[] = [];
			const currentResult = sylberResult;
			if (!currentResult) {
				return regions;
			}

			currentResult.xsegments.forEach((segment, i) => {
				const content = document.createElement('span');
				content.textContent = i.toString();
				regions.push({
					start: segment[0],
					end: segment[1],
					content,
					color: 'rgba(0, 0, 255, 0.2)'
				});
			});

			return regions;
		} else {
			return [];
		}
	});

	const userRegions = $derived.by(() => {
		if (comparisonMode === 'continuous') {
			let triggerScore: number;
			let minScore: number;
			if (encoder.startsWith('kanade-')) {
				triggerScore = discretize ? -0.4 : 0.7;
				minScore = discretize ? -0.3 : 0.8;
			} else {
				triggerScore = discretize ? -0.4 : 0.6;
				minScore = discretize ? -0.3 : 0.75;
			}

			if (scores.length === 0) {
				return [];
			}
			const smoothedScores = scores;

			const newRegions: { start: number; end: number; color: string; content: HTMLElement }[] = [];
			let currentRegion: { start: number; scores: number[] } | undefined;

			smoothedScores.forEach((score, i) => {
				if (score < triggerScore) {
					if (!currentRegion) {
						for (let j = i - 1; j >= 0; j--) {
							if (smoothedScores[j] >= triggerScore) {
								currentRegion = { start: j + 1, scores: smoothedScores.slice(j + 1, i) };
								break;
							}
						}
						currentRegion ??= { start: 0, scores: [] };
					}
					currentRegion.scores.push(score);
				} else if (score < minScore) {
					currentRegion?.scores.push(score);
				} else {
					if (currentRegion) {
						if (currentRegion.scores.length * frameDuration < 0.1) {
							currentRegion = undefined;
							return;
						}
						const avgScore =
							currentRegion.scores.reduce((a, b) => a + b, 0) / currentRegion.scores.length;
						const shade = ((minScore - avgScore) / Math.abs(minScore)) * (discretize ? 1 : 2.0);
						const opacity = Math.max(0, Math.min(1, shade)) * 0.8;
						const content = document.createElement('span');
						content.textContent = avgScore.toFixed(2);
						newRegions.push({
							start: currentRegion.start * frameDuration,
							end: i * frameDuration,
							color: `rgba(255, 0, 0, ${opacity})`,
							content
						});
						currentRegion = undefined;
					}
				}
			});

			if (currentRegion) {
				const avgScore =
					currentRegion.scores.reduce((a, b) => a + b, 0) / currentRegion.scores.length;
				const shade = (minScore - avgScore) / minScore;
				const opacity = Math.max(0, Math.min(1, shade)) * 0.8;
				const content = document.createElement('span');
				content.textContent = avgScore.toFixed(2);
				newRegions.push({
					start: currentRegion.start * frameDuration,
					end: scores.length * frameDuration,
					color: `rgba(255, 0, 0, ${opacity})`,
					content
				});
			}

			return newRegions;
		} else {
			// Syllable mode
			const newRegions: { start: number; end: number; content: HTMLElement; color: string }[] = [];

			const currentResult = sylberResult; // for type narrowing
			if (!currentResult) {
				return newRegions;
			}

			currentResult.scores.forEach((score, i) => {
				const shade = 1 - score;
				const opacity = 0.8 * shade;
				const content = document.createElement('span');
				content.textContent = currentResult.y_to_x_mappings[i].map((idx) => idx.toString()).join(', ');
				newRegions.push({
					start: currentResult.ysegments[i][0],
					end: currentResult.ysegments[i][1],
					content,
					color: `rgba(255, 0, 0, ${opacity})`
				});
			});

			return newRegions;
		}
	});

	let audioForComparison = $derived(
		convertVoice && encoder !== voiceConversionModel ? convertedAudio : audio
	);

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audioForComparison || !modelAudio) {
				return;
			}

			loading = true;
			scores = [];
			sylberResult = undefined;

			const formData = new FormData();
			formData.append('file', audioForComparison, 'recording.wav');
			formData.append('model_file', modelAudio, 'model.wav');

			if (comparisonMode === 'continuous') {
				formData.append('encoder', encoder);
				formData.append('discretize', discretize ? '1' : '0');
				formData.append('dpdp', dpdp ? '1' : '0');
				formData.append('gamma', gamma);
				try {
					const intervals_resp = await fetch(`/api/compare`, {
						method: 'POST',
						body: formData,
						signal: controller.signal
					});
					const result = await intervals_resp.json();
					if (intervals_resp.ok) {
						frameDuration = result['frameDuration'];
						scores = result['scores'];
					} else {
						console.error(`Error fetching scores: ${result['detail']}`);
					}
				} catch (e: any) {
					if (e.name !== 'AbortError') {
						console.error('Error fetching scores:', e);
					}
				} finally {
					loading = false;
				}
			} else {
				// Syllable mode
				try {
					const intervals_resp = await fetch(`/api/compare_sylber`, {
						method: 'POST',
						body: formData,
						signal: controller.signal
					});
					const fetchResult = await intervals_resp.json();
					if (intervals_resp.ok) {
						sylberResult = fetchResult;
					} else {
						console.error(`Error fetching diff: ${fetchResult['detail']}`);
					}
				} catch (e: any) {
					if (e.name !== 'AbortError') {
						console.error('Error fetching diff:', e);
					}
				} finally {
					loading = false;
				}
			}
		})();

		return () => controller.abort();
	});

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audio || !modelAudio || !convertVoice) {
				return;
			}

			const formData = new FormData();
			formData.append('source', audio, 'source.wav');
			formData.append('reference', modelAudio, 'reference.wav');
			formData.append('model', voiceConversionModel);
			try {
				const response = await fetch(`/api/convert_voice`, {
					method: 'POST',
					body: formData,
					signal: controller.signal
				});
				if (response.ok) {
					convertedAudio = await response.blob();
				} else {
					console.error(`Error fetching converted voice: ${response.statusText}`);
				}
			} catch (e: any) {
				if (e.name !== 'AbortError') {
					console.error('Error fetching scores:', e);
				}
			}
		})();

		return () => controller.abort();
	});
</script>

<div class={loading ? 'waiting' : ''}>
	<div class="viewer-card">
		<SampleViewer audio={modelAudio} regions={modelRegions} />
	</div>
	<div class="viewer-card">
		<SampleViewer audio={convertVoice ? convertedAudio : audio} regions={userRegions} />
	</div>

	<div class="controls">
		<fieldset>
			<legend>Comparison Mode</legend>
			<div class="radio-group">
				<label>
					<input type="radio" bind:group={comparisonMode} value={'continuous'} />
					Fixed Rate
				</label>
				<label>
					<input type="radio" bind:group={comparisonMode} value={'syllable'} />
					Syllable
				</label>
			</div>
			{#if comparisonMode === 'continuous'}
				<fieldset class="sub-fieldset">
					<legend>Comparison</legend>
					<label>
						Encoder:
						<select bind:value={encoder}>
							<option value="hubert">HuBERT</option>
							<option value="kanade-12hz">Kanade 12.5 Hz</option>
							<option value="kanade-25hz">Kanade 25 Hz</option>
						</select>
					</label>
					<div class="radio-group">
						<label>
							<input type="radio" bind:group={discretize} value={false} />
							Continuous
						</label>
						<label>
							<input type="radio" bind:group={discretize} value={true} />
							Discrete
						</label>
					</div>
					{#if discretize}
						<fieldset class="sub-fieldset">
							<label>
								DPDP:
								<input type="checkbox" bind:checked={dpdp} />
							</label>
							<label class:disabled={!dpdp}>
								Gamma: {gamma}
								<input type="range" min="0.0" max="1.0" step="0.1" bind:value={gamma} disabled={!dpdp} />
							</label>
						</fieldset>
					{/if}
				</fieldset>
			{/if}
		</fieldset>

		<fieldset>
			<legend>Voice Conversion</legend>
			<label> Enabled: <input type="checkbox" bind:checked={convertVoice} /> </label>
			<label class:disabled={!convertVoice}>
				Model:
				<select bind:value={voiceConversionModel}>
					<option value="kanade-12hz">Kanade 12.5 Hz</option>
					<option value="kanade-25hz">Kanade 25 Hz</option>
				</select>
			</label>
		</fieldset>
	</div>
</div>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: flex-start;
		margin-top: 1rem;
	}
	fieldset {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		padding: 1rem;
		background-color: var(--surface-color);
	}
	.sub-fieldset {
		padding: 0.5rem;
		border-style: dashed;
	}
	label {
		display: flex;
		align-items: center;
		gap: 0.5em;
	}
	.radio-group {
		display: flex;
		gap: 1rem;
	}
	select {
		padding: 0.5rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		background-color: var(--surface-color);
		color: var(--foreground-color);
	}
	.disabled {
		opacity: 0.5;
	}
	.viewer-card {
		background-color: var(--surface-color);
		border-radius: 8px;
		border: 1px solid var(--border-color);
		padding: 1rem;
		margin-bottom: 1rem;
	}
</style>
