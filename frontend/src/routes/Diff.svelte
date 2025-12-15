<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postBlob, postJson, getJson } from '$lib/api';
	import { reportError } from '$lib/errors';
	import { buildContinuousRegions, buildSegmentRegions, type Region } from '$lib/regions';
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

	let loading = $state(false);

	// Selected encoder capabilities (for UI enable/disable)
	const selectedEncoderOption = $derived.by(() => encoderOptions.find((o) => o.value === encoder));
	const supportsDiscretize = $derived.by(() => {
		// Until list loads, default to true
		if (!encoderOptions.length || !selectedEncoderOption) return true;
		return selectedEncoderOption.discretizers.length > 0;
	});

	// ---------- Derived regions ----------
	const modelRegions = $derived.by(() => {
		const isFixedRate = selectedEncoderOption?.has_fixed_frame_rate ?? true;

		if (isFixedRate) {
			return [] as Region[];
		}

		if (modelSegments) {
			return modelSegments.map((segment, i) => ({
				id: `model-segment-${i}`,
				start: segment[0],
				end: segment[1],
				content: i.toString(),
				color: 'rgba(0, 0, 255, 0.2)'
			}));
		}
		return [] as Region[];
	});

	const userRegions = $derived.by(() => {
		const isFixedRate = selectedEncoderOption?.has_fixed_frame_rate ?? true;

		if (learnerSegments) {
			if (alignmentMap && !isFixedRate) {
				// Variable rate / segments mode (e.g. Sylber)
				return learnerSegments.map((segment, i) => {
					const score = scores[i];
					let opacity = 0;
					if (score < trigger) {
						opacity = 0.8 * ((trigger - score) / trigger);
					}
					const modelIndex = alignmentMap![i];
					return {
						id: 'segment-' + i,
						start: segment[0],
						end: segment[1],
						color: `rgba(255, 0, 0, ${opacity})`,
						content: modelIndex === -1 ? '' : modelIndex.toString()
					};
				});
			} else {
				if (discretize) {
					return buildSegmentRegions(scores, learnerSegments, combineRegions);
				} else {
					return buildContinuousRegions(scores, learnerSegments, trigger, trigger - 0.05);
				}
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
				} else if (dist_method !== 'default') {
					formData.append('dist_method', dist_method);
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
					<li>Drag on waveform to play a selection.</li>
					<li>
						Hold Shift while dragging to play the corresponding audio in the other track.
					</li>
					<li>
						Click on a region to play it. (Hold Shift to play corresponding audio in the other track.)
					</li>
				</ul>
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
			clickToPlay={!articulatoryFeatures}
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
		<p>{JSON.stringify(scores)}</p>
	</details> -->

	<div class="controls">
		<fieldset>
			<legend>Comparison</legend>
			<label>
				Encoder:
				<select
					bind:value={encoder}
					onchange={() => {
						if (!supportsDiscretize) discretize = false;
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
					<label class="label-with-tooltip">
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
						Trigger:
						<input type="range" min="0.0" max="1.0" step="0.05" bind:value={trigger} />
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
