<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postBlob, postJson, getJson } from '$lib/api';
	import { reportError } from '$lib/errors';
	import {
		buildContinuousRegions,
		buildSegmentRegions,
		buildCombinedModelRegions,
		type Region
	} from '$lib/regions';
	import ArticulatoryFeatures from './ArticulatoryFeatures.svelte';
	import Tooltip from '$lib/Tooltip.svelte';

	// ---------- Types & helpers ----------
	// Encoder config from backend (simplified flat lists)
	type EncoderOption = {
		value: string;
		label: string;
		discretizers: Array<string>;
		default_dist_method: string;
		has_fixed_frame_rate: boolean;
		supports_dpdp: boolean;
	};
	type VoiceModelOption = { value: string; label: string };

	type ModelsResponse = {
		encoders: Array<EncoderOption>;
		vc_models: Array<VoiceModelOption>;
	};

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

	// ---------- Model viewer (for control from sample viewer) ----------
	let modelViewer: SampleViewer | undefined = $state();
	let learnerViewer: SampleViewer | undefined = $state();

	// Dynamic encoder options (fetched from backend)
	let encoderOptions = $state<EncoderOption[]>([]);
	let vcModelOptions = $state<VoiceModelOption[]>([]);

	// Fixed-rate diff
	let encoder = $state('hubert_l7');
	let discretize = $state(false);
	let discretizer = $state('bshall');
	let combineRegions = $state(true);
	let dpdp = $state(true);
	let gamma = $state('0.2');
	let scores = $state<number[]>([]);
	let alignmentMap = $state<number[] | undefined>();
	let alignedTimes = $state<number[][] | undefined>();
	const modelAlignedTimes = $derived.by(() => {
		if (!alignedTimes) return undefined;
		return alignedTimes.map(([t1, t2]) => [t2, t1]).sort((a, b) => a[0] - b[0]);
	});
	let articulatoryFeatures = $state<number[][] | undefined>();
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

	let loading = $state(false);

	// Selected encoder capabilities (for UI enable/disable)
	const selectedEncoderOption = $derived.by(() => encoderOptions.find((o) => o.value === encoder));
	const supportsDiscretize = $derived.by(() => {
		// Until list loads, default to true
		if (!encoderOptions.length || !selectedEncoderOption) return true;
		return selectedEncoderOption.discretizers.length > 0;
	});
	const supportsDpdp = $derived.by(() => {
		// Until list loads, default to true
		if (!encoderOptions.length || !selectedEncoderOption) return true;
		return selectedEncoderOption.supports_dpdp;
	});
	const isFixedRateEncoder = $derived(selectedEncoderOption?.has_fixed_frame_rate ?? true);
	const isSegmentalAlignment = $derived(
		!discretize &&
			(alignment_method === 'segmental' || (alignment_method === 'default' && !isFixedRateEncoder))
	);

	// ---------- Derived regions ----------
	const combinedModelData = $derived.by(() => {
		const showIndividualSegments =
			!isFixedRateEncoder || (discretize && dpdp) || isSegmentalAlignment;

		if (modelSegments) {
			const coveredIndices = new Set(alignmentMap?.filter((idx) => idx !== -1) ?? []);
			const isStandardDiscrete = isFixedRateEncoder && discretize && !dpdp;

			if (isStandardDiscrete && combineRegions) {
				return buildCombinedModelRegions(modelSegments, coveredIndices);
			}

			if (showIndividualSegments || (isStandardDiscrete && !combineRegions)) {
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
			}
		}
		return { regions: [] as Region[], indexMap: undefined };
	});

	const modelRegions = $derived(combinedModelData.regions);
	const modelIndexMap = $derived(combinedModelData.indexMap);

	const userRegions = $derived.by(() => {
		if (learnerSegments) {
			if (discretize) {
				return buildSegmentRegions(
					scores,
					learnerSegments,
					combineRegions,
					alignmentMap,
					modelIndexMap
				);
			}

			if (alignmentMap && isSegmentalAlignment) {
				// Variable rate / segments mode (e.g. Sylber)
				return learnerSegments.map((segment, i) => {
					const score = scores[i];
					let opacity = 0;
					if (score < trigger) {
						opacity = 0.8 * ((trigger - score) / trigger);
					}
					const modelIndex = alignmentMap![i];
					if (modelIndex === -1) {
						return {
							id: 'segment-' + i,
							start: segment[0],
							end: segment[1],
							color: `rgba(255, 255, 0, 0.5)`,
							content: ''
						};
					}
					return {
						id: 'segment-' + i,
						start: segment[0],
						end: segment[1],
						color: `rgba(255, 0, 0, ${opacity})`,
						content: modelIndex.toString()
					};
				});
			} else {
				return buildContinuousRegions(scores, learnerSegments, trigger, trigger - 0.05);
			}
		}
		return [] as Region[];
	});

	// Audio to compare (may be converted/reconstructed)
	let audioForComparison = $derived(
		convertVoice && encoder !== voiceConversionModel ? convertedAudio : audio
	);
	let modelForComparison = $derived(reconstructModel ? reconstructedAudio : modelAudio);

	// ---------- Effects: check discretizer ----------
	$effect(() => {
		if (!selectedEncoderOption) return;

		if (!selectedEncoderOption.discretizers.includes(discretizer)) {
			discretizer = selectedEncoderOption.discretizers[0];
		}
		if (!supportsDiscretize) {
			discretize = false;
		}
		if (!supportsDpdp) {
			dpdp = false;
		}
	});

	// ---------- Effects: comparison fetch ----------

	// Load encoders from backend once when active and options not loaded
	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!active) return;
			if (encoderOptions.length) return;
			try {
				const config = await getJson<ModelsResponse>(`/api/models`, controller.signal);
				encoderOptions = config.encoders;
				vcModelOptions = config.vc_models;

				// Ensure current selections are valid
				if (!encoderOptions.some((o) => o.value === encoder) && encoderOptions.length > 0) {
					encoder = encoderOptions[0].value;
				}
				if (
					!vcModelOptions.some((o) => o.value === voiceConversionModel) &&
					vcModelOptions.length > 0
				) {
					voiceConversionModel = vcModelOptions[0].value;
				}
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					reportError('Error fetching encoders.', e);
			}
		})();

		return () => controller.abort();
	});
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

			const formData = new FormData();
			formData.append('file', audioForComparison, 'recording.wav');
			formData.append('model_file', modelForComparison, 'model.wav');

			try {
				formData.append('encoder', encoder);
				let data: {
					scores: number[];
					alignmentMap?: number[];
					articulatoryFeatures?: number[][];
					alignedTimes?: number[][];
					learnerSegments?: number[][];
					modelSegments?: number[][];
				};
				if (discretize) {
					formData.append('discretizer', discretizer);
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
				if (discretize && dpdp) {
					formData.append('gamma', gamma);
					data = await postJson(`/api/compare_dpdp`, formData, controller.signal);
				} else {
					data = await postJson(`/api/compare`, formData, controller.signal);
				}
				scores = data.scores ?? [];
				alignmentMap = data.alignmentMap;
				alignedTimes = data.alignedTimes;
				articulatoryFeatures = data.articulatoryFeatures;
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
			<h3>Model Audio</h3>
			<Tooltip position="bottom">
				<b>Playback controls</b>
				<ul class="tooltip-list">
					<li>Click on a waveform to seek.</li>
					<li>Drag on a waveform to play a selection.</li>
					<li>Click on a region to play it.</li>
				</ul>
				<strong>Tip:</strong> Hold <kbd>Shift</kbd> while doing any of the above to perform these actions on the other track
			</Tooltip>
		</div>
		<SampleViewer
			audio={reconstructModel ? reconstructedAudio : modelAudio}
			regions={modelRegions}
			bind:this={modelViewer}
			compareWith={learnerViewer
				? {
						other: learnerViewer,
						alignedTimes: modelAlignedTimes
					}
				: null}
		/>
		<div class="viewer-header">
			<h3>Learner Audio</h3>
		</div>
		<SampleViewer
			audio={convertVoice ? convertedAudio : audio}
			regions={userRegions}
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
	<details>
		<summary>Details</summary>
		<p>
			{scores.map((s) => s.toFixed(2)).join(', ')}
		</p>
	</details>

	<div class="controls">
		<fieldset>
			<legend>Comparison</legend>
			<label>
				Encoder:
				<select
					bind:value={encoder}
					onchange={() => {
						if (!supportsDiscretize) discretize = false;
						if (!supportsDpdp) dpdp = false;
					}}
				>
					{#if encoderOptions.length}
						{#each encoderOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					{:else}
						<!-- Fallback: HuBERT only until list loads -->
						<option value="hubert_l7">HuBERT L7</option>
					{/if}
				</select>
			</label>
			<div class="radio-group">
				<label>
					<input type="radio" bind:group={discretize} value={false} />
					Continuous
				</label>
				<label class:disabled={!supportsDiscretize}>
					<input type="radio" bind:group={discretize} value={true} disabled={!supportsDiscretize} />
					Discrete
				</label>
			</div>
			<fieldset class="sub-fieldset">
				{#if discretize}
					<label>
						Discretizer:
						<select bind:value={discretizer}>
							{#if supportsDiscretize && selectedEncoderOption}
								{#each selectedEncoderOption.discretizers as opt}
									<option value={opt}>{opt}</option>
								{/each}
							{:else}
								<!-- Fallback: bshall only until list loads -->
								<option value="bshall">bshall</option>
							{/if}
						</select>
					</label>
					<label>
						Combine Regions:
						<input type="checkbox" bind:checked={combineRegions} />
					</label>
					<label class="label-with-tooltip" class:disabled={!supportsDpdp}>
						<span>DPDP:</span>
						<Tooltip align="left">
							Dynamic programming method that produces coarser speech units.
							<br /><br />
							<i
								>Word Segmentation on Discovered Phone Units With Dynamic Programming and
								Self-Supervised Scoring</i
							>
							(Herman Kamper).
							<a
								href="https://doi.org/10.1109/TASLP.2022.3229264"
								target="_blank"
								rel="noopener noreferrer"
							>
								doi:10.1109/TASLP.2022.3229264
							</a>
						</Tooltip>
						<input type="checkbox" bind:checked={dpdp} disabled={!supportsDpdp} />
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
								>Default{selectedEncoderOption && !isFixedRateEncoder
									? ' (Segmental)'
									: ' (DTW)'}
							</option>
							<option value="dtw">DTW</option>
							<option value="segmental">Segmental</option>
						</select>
					</label>
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
						<input
							type="number"
							step="any"
							bind:value={alpha}
							disabled={!customAlpha}
						/>
					</label>
				{/if}
			</fieldset>
		</fieldset>

		<fieldset>
			<legend class="legend-with-tooltip">
				<span>Voice Conversion</span>
				<Tooltip>
					When enabled, the learner track is converted to the model's voice.
					<br /><br />
					"Reconstruct Model" will process the model's audio through the voice conversion model. This
					can be useful to create a more fair comparison, by accounting for any artifacts introduced
					by the voice conversion model itself.
				</Tooltip>
			</legend>
			<label> Enabled: <input type="checkbox" bind:checked={convertVoice} /> </label>
			<label> Reconstruct Model: <input type="checkbox" bind:checked={reconstructModel} /> </label>
			<label>
				Model:
				<select bind:value={voiceConversionModel}>
					{#if vcModelOptions.length}
						{#each vcModelOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					{:else}
						<!-- Fallback: Kanade 25 Hz only until list loads -->
						<option value="kanade-25hz">Kanade 25 Hz</option>
					{/if}
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

	.legend-with-tooltip,
	.label-with-tooltip {
		display: flex;
		align-items: center;
		gap: 0.5em;
	}
</style>
