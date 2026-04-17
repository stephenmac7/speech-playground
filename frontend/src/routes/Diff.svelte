<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postBlob, postJson } from '$lib/api';
	import { reportError } from '$lib/errors';
	import {
		buildContinuousRegions,
		buildSegmentRegions,
		buildCombinedModelRegions,
		buildPhonologicalTier,
		buildPhonologicalTiers,
		type Region,
		type Tier
	} from '$lib/regions';
	import ArticulatoryFeatures from './ArticulatoryFeatures.svelte';
	import Tooltip from '$lib/Tooltip.svelte';
	import EncoderFieldset from '$lib/EncoderFieldset.svelte';
	import type { ModelsResponse, EncoderConfig } from '$lib/types';

	// ---------- Props ----------
	let {
		tracks,
		modelsConfig,
		encoderConfig = $bindable()
	}: {
		tracks: Record<string, import('./AudioLibrary.svelte').TrackData>;
		modelsConfig: ModelsResponse;
		encoderConfig: EncoderConfig;
	} = $props();

	const audio = $derived(tracks['Audio']?.data ?? undefined);
	const modelAudio = $derived(tracks['Model']?.data ?? undefined);

	const vcModelOptions = $derived(modelsConfig.vc_models);
	const selectedEncoderOption = $derived(
		modelsConfig.encoders.find((o) => o.value === encoderConfig.encoder)
	);

	// ---------- TextGrid tiers ----------
	function textgridTiersForKey(key: string): Tier[] {
		const tg = tracks[key]?.textgrid;
		if (!tg) return [];
		return Object.entries(tg).map(([name, intervals]) => ({
			name,
			regions: intervals.map((iv) => ({
				start: iv.start,
				end: iv.end,
				content: iv.content,
				color: 'rgba(160, 200, 255, 0.6)'
			}))
		}));
	}

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

	// ---------- Model viewer (for control from sample viewer) ----------
	let modelViewer: SampleViewer | undefined = $state();
	let learnerViewer: SampleViewer | undefined = $state();

	let combineRegions = $state(false);
	let alignment_mode = $state('global');
	let scores = $state<number[]>([]);
	let alignmentMap = $state<number[] | undefined>();
	let alignedTimes = $state<number[][] | undefined>();
	const modelAlignedTimes = $derived.by(() => {
		if (!alignedTimes) return undefined;
		return alignedTimes.map(([t1, t2]) => [t2, t1]).sort((a, b) => a[0] - b[0]);
	});
	let articulatoryFeatures = $state<number[][] | undefined>();
	let phonologicalActivations = $state<number[][][] | undefined>();
	let phonologicalFeatureNames = $state<string[][] | undefined>();
	let learnerSegments = $state<number[][] | undefined>();
	let modelSegments = $state<number[][] | undefined>();
	let currentTime = $state(0);
	const currentFrame = $derived(Math.floor(currentTime * 50));

	// Continuous diff controls
	let trigger = $state(0.6);
	let dist_method = $state('default');
	let alignment_method = $state('default');
	let customAlpha = $state(false);
	let alpha = $state(1.0);
	let showScore = $state(false);

	let loading = $state(false);

	const isFixedRateEncoder = $derived(selectedEncoderOption?.has_fixed_frame_rate ?? true);
	const isSegmentalAlignment = $derived(
		!encoderConfig.discretize &&
			(alignment_method === 'segmental' || (alignment_method === 'default' && !isFixedRateEncoder))
	);

	// ---------- Derived regions ----------
	const combinedModelData = $derived.by(() => {
		if (!modelSegments) return { regions: [] as Region[], indexMap: undefined };

		const coveredIndices = new Set(alignmentMap?.filter((idx) => idx !== -1) ?? []);
		const isStandardDiscrete = isFixedRateEncoder && encoderConfig.discretize && !encoderConfig.dpdp;

		if (isStandardDiscrete && combineRegions) {
			return buildCombinedModelRegions(modelSegments, coveredIndices);
		}

		return {
			regions: modelSegments.map((segment, i) => ({
				id: `model-segment-${i}`,
				start: segment[0],
				end: segment[1],
				content: i.toString(),
				color: coveredIndices.has(i) ? 'rgba(0, 0, 255, 0.2)' : 'rgba(255, 0, 0, 0.5)'
			})),
			indexMap: undefined
		};
	});

	const modelRegions = $derived(combinedModelData.regions);
	const modelIndexMap = $derived(combinedModelData.indexMap);

	const modelPhonologicalTiers = $derived.by<Tier[]>(() => {
		if (!phonologicalActivations || !phonologicalFeatureNames) return [];
		return buildPhonologicalTiers(phonologicalActivations[0], phonologicalFeatureNames[0]);
	});

	const learnerPhonologicalTiers = $derived.by<Tier[]>(() => {
		if (!phonologicalActivations || !phonologicalFeatureNames) return [];
		const modelActs = phonologicalActivations[0];
		const learnerActs = phonologicalActivations[1];
		const names = phonologicalFeatureNames[1];
		const map = alignmentMap;
		const alignedModelActs: number[][] = new Array(learnerActs.length);
		for (let t = 0; t < learnerActs.length; t++) {
			const width = learnerActs[t].length;
			const modelIdx = map?.[t];
			if (modelIdx === undefined || modelIdx < 0 || modelIdx >= modelActs.length) {
				alignedModelActs[t] = new Array(width).fill(0);
			} else {
				alignedModelActs[t] = modelActs[modelIdx];
			}
		}

		const speechIdx = names.indexOf('speech+');
		const speechMask: boolean[] = new Array(learnerActs.length);
		for (let t = 0; t < learnerActs.length; t++) {
			speechMask[t] =
				speechIdx >= 0
					? learnerActs[t][speechIdx] > 0 || alignedModelActs[t][speechIdx] > 0
					: true;
		}

		const featureOrder: number[] = [];
		const totalDiffs: number[] = new Array(names.length).fill(0);
		for (let f = 0; f < names.length; f++) {
			const n = names[f];
			if (n.endsWith('-') || n === 'speech+') continue;
			featureOrder.push(f);
			let sum = 0;
			for (let t = 0; t < learnerActs.length; t++) {
				if (!speechMask[t]) continue;
				sum += Math.abs(learnerActs[t][f] - alignedModelActs[t][f]);
			}
			totalDiffs[f] = sum;
		}
		featureOrder.sort((a, b) => totalDiffs[b] - totalDiffs[a]);

		let vrange = 0;
		for (let t = 0; t < learnerActs.length; t++) {
			for (let f = 0; f < learnerActs[t].length; f++) {
				const a = Math.max(Math.abs(learnerActs[t][f]), Math.abs(alignedModelActs[t][f]));
				if (a > vrange) vrange = a;
			}
		}
		if (vrange === 0) vrange = 1;

		const tiers: Tier[] = [];
		for (const f of featureOrder) {
			const name = names[f];
			const refTier = buildPhonologicalTier(
				alignedModelActs,
				f,
				`phonological:${name}`,
				0.02,
				vrange
			);
			const learnerTier = buildPhonologicalTier(
				learnerActs,
				f,
				`phonological:${name}`,
				0.02,
				vrange
			);
			const combined: Region[] = [];
			for (const r of refTier.regions) combined.push({ ...r, lane: 'top' });
			for (const r of learnerTier.regions)
				combined.push({ ...r, id: `${r.id}-learner`, lane: 'bottom' });
			tiers.push({ name: refTier.name, regions: combined });
		}
		return tiers;
	});

	const userRegions = $derived.by(() => {
		if (learnerSegments) {
			if (encoderConfig.discretize) {
				return buildSegmentRegions(
					scores,
					learnerSegments,
					combineRegions,
					alignmentMap,
					modelIndexMap,
					showScore
				);
			}

			return buildContinuousRegions(
				scores,
				learnerSegments,
				trigger,
				trigger - 0.05,
				combineRegions,
				alignmentMap,
				modelIndexMap,
				showScore
			);
		}
		return [] as Region[];
	});

	// Audio to compare (may be converted/reconstructed)
	let audioForComparison = $derived(
		convertVoice && encoderConfig.encoder !== voiceConversionModel ? convertedAudio : audio
	);
	let modelForComparison = $derived(reconstructModel ? reconstructedAudio : modelAudio);

	// ---------- Effects: comparison fetch ----------
	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audioForComparison || !modelForComparison) return;

			loading = true;
			scores = [];
			alignmentMap = undefined;
			alignedTimes = undefined;
			learnerSegments = undefined;
			modelSegments = undefined;
			articulatoryFeatures = undefined;
			phonologicalActivations = undefined;
			phonologicalFeatureNames = undefined;

			const formData = new FormData();
			formData.append('file', audioForComparison, 'recording.wav');
			formData.append('model_file', modelForComparison, 'model.wav');

			try {
				formData.append('encoder', encoderConfig.encoder);
				let data: {
					scores: number[];
					alignmentMap?: number[];
					articulatoryFeatures?: number[][];
					phonologicalActivations?: number[][][];
					phonologicalFeatureNames?: string[][];
					alignedTimes?: number[][];
					learnerSegments?: number[][];
					modelSegments?: number[][];
				};
				if (encoderConfig.discretize) {
					formData.append('discretizer', encoderConfig.discretizer);
				} else {
					if (dist_method !== 'default') {
						formData.append('dist_method', dist_method);
					}
					if (alignment_method !== 'default') {
						formData.append('alignment_method', alignment_method);
					}
					if (customAlpha) {
						formData.append('alpha', alpha.toString());
					}
				}
				if (encoderConfig.discretize && encoderConfig.dpdp) {
					formData.append('gamma', encoderConfig.gamma);
					formData.append('alignment_mode', alignment_mode);
					data = await postJson(`/api/compare_dpdp`, formData, controller.signal);
				} else {
					formData.append('alignment_mode', alignment_mode);
					data = await postJson(`/api/compare`, formData, controller.signal);
				}
				scores = data.scores ?? [];
				alignmentMap = data.alignmentMap;
				alignedTimes = data.alignedTimes;
				articulatoryFeatures = data.articulatoryFeatures;
				phonologicalActivations = data.phonologicalActivations;
				phonologicalFeatureNames = data.phonologicalFeatureNames;
				learnerSegments = data.learnerSegments;
				modelSegments = data.modelSegments;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError') reportError('Error fetching diff.', e);
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
					reportError('Error fetching converted voice.', e);
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
					reportError('Error fetching reconstructed voice.', e);
			}
		})();

		return () => controller.abort();
	});
</script>

<div class={loading ? 'waiting' : ''}>
	<div class="viewer-card">
		<div class="viewer-header">
			<h3>Model</h3>
			<Tooltip position="bottom">
				<b>Playback controls</b>
				<ul class="tooltip-list">
					<li>Click on a waveform to seek.</li>
					<li>Drag on a waveform to play a selection.</li>
					<li>Click on a region to play it.</li>
				</ul>
				<strong>Tip:</strong> Hold <kbd>Shift</kbd> while doing any of the above to perform these actions
				on the other track
			</Tooltip>
		</div>
		<SampleViewer
			audio={reconstructModel ? reconstructedAudio : modelAudio}
			tiers={[
				{ name: 'Distance', regions: modelRegions },
				...textgridTiersForKey('Model'),
				...modelPhonologicalTiers
			]}
			transcript={tracks['Model']?.transcript ?? undefined}
			bind:this={modelViewer}
			compareWith={learnerViewer
				? {
						other: learnerViewer,
						alignedTimes: modelAlignedTimes
					}
				: null}
		/>
		<div class="viewer-header">
			<h3>Audio</h3>
		</div>
		<SampleViewer
			audio={convertVoice ? convertedAudio : audio}
			tiers={[
				{ name: 'Distance', regions: userRegions },
				...textgridTiersForKey('Audio'),
				...learnerPhonologicalTiers
			]}
			transcript={tracks['Audio']?.transcript ?? undefined}
			compareWith={modelViewer
				? {
						other: modelViewer,
						alignedTimes: alignedTimes
					}
				: null}
			bind:currentTime
			bind:this={learnerViewer}
		/>
	</div>
	{#if articulatoryFeatures}
		<div class="viewer-card">
			<h3>Articulatory Features</h3>
			<ArticulatoryFeatures
				learnerFeatures={articulatoryFeatures[1][currentFrame]}
				referenceFeatures={alignmentMap
					? articulatoryFeatures[0][alignmentMap[currentFrame]]
					: undefined}
			/>
		</div>
	{/if}
	<!-- <details>
		<summary>Details</summary>
		<p>
			{scores.map((s) => s.toFixed(2)).join(', ')}
		</p>
	</details> -->

	<div class="controls">
		<EncoderFieldset {modelsConfig} bind:config={encoderConfig} />

		<fieldset>
			<legend>Comparison</legend>
			{#if !encoderConfig.discretize}
				<label>
					Distance Method:
					<select bind:value={dist_method}>
						<option value="default"
							>Default{selectedEncoderOption
								? ` (${selectedEncoderOption.default_dist_method})`
								: ''}</option
						>
						<option value="euclidean">Euclidean</option>
						<option value="cosine">Cosine</option>
					</select>
				</label>
				<label>
					Alignment Method:
					<select bind:value={alignment_method}>
						<option value="default"
							>Default{selectedEncoderOption && !isFixedRateEncoder ? ' (Segmental)' : ' (DTW)'}
						</option>
						<option value="dtw">DTW</option>
						<option value="segmental">Segmental</option>
					</select>
				</label>
			{/if}
			{#if encoderConfig.discretize || isSegmentalAlignment}
				<label>
					Alignment Mode:
					<select bind:value={alignment_mode}>
						<option value="global">Global</option>
						<option value="semiglobal">Semi-Global</option>
					</select>
				</label>
			{/if}
			<label>
				Combine Regions:
				<input type="checkbox" bind:checked={combineRegions} />
			</label>
			<label>
				Show Score:
				<input type="checkbox" bind:checked={showScore} />
			</label>
			{#if !encoderConfig.discretize}
				<label>
					Trigger:
					<input type="range" min="0.0" max="1.0" step="0.05" bind:value={trigger} />
				</label>
				<label>
					Custom Alpha:
					<input type="checkbox" bind:checked={customAlpha} />
				</label>
				<label class:disabled={!customAlpha}>
					Alpha:
					<input type="number" step="any" bind:value={alpha} disabled={!customAlpha} />
				</label>
			{/if}
		</fieldset>

		<fieldset>
			<legend class="legend-with-tooltip">
				<span>Voice Conversion</span>
				<Tooltip>
					When enabled, the learner track is converted to the model's voice.
					<br /><br />
					"Reconstruct Model" will process the model's audio through the voice conversion model.
				</Tooltip>
			</legend>
			<label> Enabled: <input type="checkbox" bind:checked={convertVoice} /> </label>
			<label> Reconstruct Model: <input type="checkbox" bind:checked={reconstructModel} /> </label>
			<label>
				Model:
				<select bind:value={voiceConversionModel}>
					{#each vcModelOptions as opt (opt.value)}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
		</fieldset>
	</div>
</div>

<style>
	.tooltip-list {
		margin: 4px 0 0 0;
		padding-left: 20px;
		list-style-type: '- ';
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: flex-start;
		justify-content: stretch;
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
	label {
		display: flex;
		align-items: center;
		gap: 0.5em;
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

	.viewer-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	.viewer-header:not(:first-child) {
		margin-top: 1rem;
	}

	.viewer-header h3 {
		margin: 0;
	}

	.legend-with-tooltip {
		display: flex;
		align-items: center;
		gap: 0.5em;
	}
</style>
