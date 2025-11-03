<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postBlob, postJson, getJson } from '$lib/api';
	import {
		buildContinuousRegions,
		buildSegmentRegions,
		buildSyllableRegions,
		type Region,
		type SylberResult
	} from '$lib/regions';
	import ArticulatoryFeatures from './ArticulatoryFeatures.svelte';

	// ---------- Types & helpers ----------
	type ComparisonMode = 'fixedRate' | 'syllable';

	// Encoder config from backend (simplified flat lists)
	type ModelsResponse = {
		encoders: Array<{ value: string; label: string; supports_discretization: boolean }>;
		vc_models: Array<{ value: string; label: string }>;
	};

	type EncoderOption = { value: string; label: string; supports_discretization: boolean; disabled?: boolean };
	type VoiceModelOption = { value: string; label: string };

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

	// ---------- Comparison controls ----------
	let comparisonMode = $state<ComparisonMode>('fixedRate');

	// Dynamic encoder options (fetched from backend)
	let encoderOptions = $state<EncoderOption[]>([]);
	let vcModelOptions = $state<VoiceModelOption[]>([]);

	// Fixed-rate diff
	let encoder = $state('hubert');
	let discretize = $state(true);
	let combineRegions = $state(true);
	let dpdp = $state(true);
	let gamma = $state('0.2');
	let frameDuration = $state(0.02);
	let scores = $state<number[]>([]);
	let boundaries = $state<number[] | undefined>();
	let modelBoundaries = $state<number[] | undefined>();
	let alignmentMap = $state<number[] | undefined>();
	let articulatoryFeatures = $state<number[][] | undefined>();
	let currentFrame = $state(0);

	// Threshold controls
	let trigger = $state(0.6);

	// Syllable diff
	let sylberResult: SylberResult | undefined = $state();

	let loading = $state(false);

	// Selected encoder capabilities (for UI enable/disable)
	const selectedEncoderOption = $derived.by(() => encoderOptions.find((o) => o.value === encoder));
	const supportsDiscretize = $derived.by(() => {
		// Until list loads, default to true
		if (!encoderOptions.length) return true;
		return selectedEncoderOption?.supports_discretization ?? true;
	});

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
					console.error('Error fetching encoders:', e);
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
			boundaries = undefined;
			modelBoundaries = undefined;
			alignmentMap = undefined;
			sylberResult = undefined;
			articulatoryFeatures = undefined;

			const formData = new FormData();
			formData.append('file', audioForComparison, 'recording.wav');
			formData.append('model_file', modelForComparison, 'model.wav');

			try {
				if (comparisonMode === 'fixedRate') {
					formData.append('encoder', encoder);
					let data: {
						scores: number[];
						boundaries: number[] | undefined;
						modelBoundaries: number[] | undefined;
						frameDuration: number;
						alignmentMap?: number[];
						articulatoryFeatures?: number[][];
					};
					if (discretize && dpdp) {
						formData.append('gamma', gamma);
						data = await postJson(`/api/compare_dpdp`, formData, controller.signal);
					} else {
						formData.append('discretize', discretize ? '1' : '0');
						data = await postJson(`/api/compare`, formData, controller.signal);
					}
					frameDuration = data.frameDuration;
					scores = data.scores ?? [];
					alignmentMap = data.alignmentMap ?? [];
					boundaries = data.boundaries;
					modelBoundaries = data.modelBoundaries;
					articulatoryFeatures = data.articulatoryFeatures;
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
			bind:this={modelViewer}
		/>
	</div>
	<div class="viewer-card">
		<SampleViewer
			audio={convertVoice ? convertedAudio : audio}
			regions={userRegions}
			compareWith={{ other: modelViewer, boundaries, modelBoundaries, alignmentMap: alignmentMap, frameDuration: frameDuration }}
			bind:currentFrame={currentFrame}
			clickToPlay={!articulatoryFeatures}
		/>
	</div>
	{#if articulatoryFeatures}
		<div class="viewer-card">
			<h3>Articulatory Features</h3>
			<ArticulatoryFeatures
				learnerFeatures={articulatoryFeatures[1][currentFrame]}
				referenceFeatures={alignmentMap ? articulatoryFeatures[0][alignmentMap[currentFrame]] : undefined}
			/>
		</div>
	{/if}

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
						<select
							bind:value={encoder}
							onchange={() => {
								if (!supportsDiscretize) discretize = false;
							}}
						>
							{#if encoderOptions.length}
								{#each encoderOptions as opt}
									<option value={opt.value} disabled={opt.disabled}>{opt.label}{opt.disabled ? ' (missing files)' : ''}</option>
								{/each}
							{:else}
								<!-- Fallback: HuBERT only until list loads -->
								<option value="hubert">HuBERT</option>
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
