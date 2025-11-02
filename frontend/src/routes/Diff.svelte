<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postBlob, postJson } from '$lib/api';
	import {
		buildContinuousRegions,
		buildSegmentRegions,
		buildSyllableRegions,
		type Region,
		type SylberResult
	} from '$lib/regions';

	// ---------- Types & helpers ----------
	type ComparisonMode = 'fixedRate' | 'syllable';

	// ---------- Props ----------
	let { tracks, active } = $props();

	// ---------- Base audio state ----------
	let audio = $state(tracks['Audio']);
	let modelAudio = $state(tracks['Model']);
	$effect(() => {
		if (active) {
			audio = tracks['Audio'];
			modelAudio = tracks['Model'];
		}
	});

	// ---------- Voice conversion & reconstruction ----------
	let convertVoice = $state(false);
	let reconstructModel = $state(false);
	let voiceConversionModel = $state('kanade-25hz');
	let convertedAudio = $state<Blob | undefined>();
	let reconstructedAudio = $state<Blob | undefined>();
	$effect(() => {
		if (!convertVoice) convertedAudio = undefined;
		if (!reconstructModel) reconstructedAudio = undefined;
	});

	// ---------- Comparison controls ----------
	let comparisonMode = $state<ComparisonMode>('fixedRate');

	// Fixed-rate diff
	let encoder = $state('hubert');
	let discretize = $state(true);
	let combineRegions = $state(true);
	let dpdp = $state(true);
	let gamma = $state('0.2');
	let frameDuration = $state(0.02);
	let scores = $state<number[]>([]);
	let boundaries = $state<number[] | undefined>();

	// Threshold controls
	let trigger = $state(0.6);

	// Syllable diff
	let sylberResult: SylberResult | undefined = $state();

	let loading = $state(false);

	// ---------- Derived regions ----------
	const modelRegions = $derived.by(() => {
		if (comparisonMode !== 'syllable' || !sylberResult) return [] as Region[];
		// Visualize model syllables by index
		return sylberResult.xsegments.map((segment, i) => ({
			start: segment[0],
			end: segment[1],
			content: (() => {
				const el = document.createElement('span');
				el.textContent = i.toString();
				return el;
			})(),
			color: 'rgba(0, 0, 255, 0.2)'
		}));
	});

	const userRegions = $derived.by(() => {
		if (comparisonMode === 'fixedRate') {
			if (discretize) {
				return buildSegmentRegions(scores, frameDuration, boundaries, combineRegions);
			} else {
				return buildContinuousRegions(scores, frameDuration, trigger, trigger - 0.05);
			}
		}
		return sylberResult ? buildSyllableRegions(sylberResult) : ([] as Region[]);
	});

	// Audio to compare (may be converted/reconstructed)
	let audioForComparison = $derived(
		convertVoice && encoder !== voiceConversionModel ? convertedAudio : audio
	);
	let modelForComparison = $derived(reconstructModel ? reconstructedAudio : modelAudio);

	// ---------- Effects: comparison fetch ----------
	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audioForComparison || !modelForComparison) return;

			loading = true;
			scores = [];
			boundaries = undefined;
			sylberResult = undefined;

			const formData = new FormData();
			formData.append('file', audioForComparison, 'recording.wav');
			formData.append('model_file', modelForComparison, 'model.wav');

			try {
				if (comparisonMode === 'fixedRate') {
					formData.append('encoder', encoder);
					let data: { scores: number[]; boundaries: number[] | undefined; frameDuration: number };
					if (discretize && dpdp) {
						formData.append('gamma', gamma);
						data = await postJson(`/api/compare_dpdp`, formData, controller.signal);
					} else {
						formData.append('discretize', discretize ? '1' : '0');
						data = await postJson(`/api/compare`, formData, controller.signal);
					}
					frameDuration = data.frameDuration;
					scores = data.scores ?? [];
					boundaries = data.boundaries;
				} else {
					const data = await postJson<SylberResult>(
						`/api/compare_sylber`,
						formData,
						controller.signal
					);
					sylberResult = data;
				}
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					console.error('Error fetching diff:', e);
			} finally {
				loading = false;
			}
		})();

		return () => controller.abort();
	});

	// ---------- Effects: voice conversion ----------
	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audio || !modelAudio || !convertVoice) return;

			const formData = new FormData();
			formData.append('source', audio, 'source.wav');
			formData.append('reference', modelAudio, 'reference.wav');
			formData.append('model', voiceConversionModel);
			try {
				const blob = await postBlob(`/api/convert_voice`, formData, controller.signal);
				convertedAudio = blob;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					console.error('Error fetching converted voice:', e);
			}
		})();

		return () => controller.abort();
	});

	// ---------- Effects: reconstruct model voice ----------
	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!modelAudio || !reconstructModel) return;

			const formData = new FormData();
			formData.append('file', modelAudio, 'recording.wav');
			formData.append('model', voiceConversionModel);
			try {
				const blob = await postBlob(`/api/reconstruct`, formData, controller.signal);
				reconstructedAudio = blob;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					console.error('Error fetching reconstructed voice:', e);
			}
		})();

		return () => controller.abort();
	});
</script>

<div class={loading ? 'waiting' : ''}>
	<div class="viewer-card">
		<SampleViewer
			audio={reconstructModel ? reconstructedAudio : modelAudio}
			regions={modelRegions}
		/>
	</div>
	<div class="viewer-card">
		<SampleViewer audio={convertVoice ? convertedAudio : audio} regions={userRegions} />
		<details>
			<summary>Debug Info</summary>
			<p>Scores: {scores.join(', ')}</p>
		</details>
	</div>

	<div class="controls">
		<fieldset>
			<legend>Comparison Mode</legend>
			<div class="radio-group">
				<label>
					<input type="radio" bind:group={comparisonMode} value="fixedRate" />
					Fixed Rate
				</label>
				<label>
					<input type="radio" bind:group={comparisonMode} value="syllable" />
					Syllable
				</label>
			</div>
			{#if comparisonMode === 'fixedRate'}
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
					<fieldset class="sub-fieldset">
						{#if discretize}
							<label>
								Combine Regions:
								<input type="checkbox" bind:checked={combineRegions} />
							</label>
							<label>
								DPDP:
								<input type="checkbox" bind:checked={dpdp} />
							</label>
							<label class:disabled={!dpdp}>
								Gamma: {gamma}
								<input
									type="range"
									min="0.0"
									max="1.0"
									step="0.1"
									bind:value={gamma}
									disabled={!dpdp}
								/>
							</label>
						{:else}
							<label>
								Trigger:
								<input type="range" min="0.0" max="1.0" step="0.05" bind:value={trigger} />
							</label>
						{/if}
					</fieldset>
				</fieldset>
			{/if}
		</fieldset>

		<fieldset>
			<legend>Voice Conversion</legend>
			<label> Enabled: <input type="checkbox" bind:checked={convertVoice} /> </label>
			<label> Reconstruct Model: <input type="checkbox" bind:checked={reconstructModel} /> </label>
			<label>
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
