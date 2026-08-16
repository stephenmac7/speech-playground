<script lang="ts">
	/* Learn mode: a pared-down pronunciation practice page.

	   Only WavLM + DTW comparison, one model track loaded from one of the
	   backend's data roots and one recorded track. The model file is named by
	   the `input` query parameter, so an exercise can be linked directly:
	     /learn?input=mine/STEAC/contrastive_training/x-y-01.wav
	   Nothing here touches the audio library (Dexie); recordings are ephemeral. */
	import { page } from '$app/state';
	import SampleViewer from '../SampleViewer.svelte';
	import WavesurferRecorder from '../WavesurferRecorder.svelte';
	import { getBlob, getJson, postBlob, postJson } from '$lib/api';
	import { buildContinuousRegions, type Region, type Tier } from '$lib/regions';
	import type { TextGridData } from '$lib/db';
	import type { Segment } from '$lib/types';

	const ENCODER = 'wavlm-base-plus';
	const TEXTGRID_COLOR = 'rgba(160, 200, 255, 0.6)';

	// ---------- Model (reference) track ----------
	/* The practice file comes only from the link. Paths are resolved against the
	   backend's data roots, as in the main app's "Server..." box. */
	const modelPath = $derived((page.url.searchParams.get('input') ?? '').trim().replace(/^\/+/, ''));
	const textParam = $derived(page.url.searchParams.get('text') ?? '');

	let modelAudio = $state<Blob | undefined>();
	let modelTextgrid = $state<TextGridData | undefined>();
	let modelError = $state('');
	let modelLoading = $state(false);

	$effect(() => {
		const path = modelPath;
		if (!path) {
			modelAudio = undefined;
			modelTextgrid = undefined;
			modelError = '';
			return;
		}

		const controller = new AbortController();
		let aborted = false;
		(async () => {
			modelLoading = true;
			modelError = '';
			modelAudio = undefined;
			modelTextgrid = undefined;
			try {
				const blob = await getBlob(`/api/data/${path}`, controller.signal);
				if (aborted) return;
				modelAudio = blob;
				// A TextGrid beside the wav is optional; absence is not an error.
				try {
					const tg = await getJson<TextGridData>(`/api/data_tg/${path}`, controller.signal);
					if (!aborted) modelTextgrid = tg;
				} catch {
					if (!aborted) modelTextgrid = undefined;
				}
			} catch (e: unknown) {
				if ((e as { name?: string })?.name === 'AbortError') return;
				console.error('Error loading model audio:', e);
				if (!aborted)
					modelError = `Could not load "${path}". ${(e as Error)?.message ?? ''}`.trim();
			} finally {
				if (!aborted) modelLoading = false;
			}
		})();

		return () => {
			aborted = true;
			controller.abort();
		};
	});

	/* A whole-utterance "text" tier reads as the sentence to practise rather than
	   as an annotation, so it becomes the caption instead of a tier row. */
	const textgridText = $derived.by(() => {
		if (!modelTextgrid) return '';
		const key = Object.keys(modelTextgrid).find((k) => k.toLowerCase() === 'text');
		if (!key) return '';
		return modelTextgrid[key]
			.map((iv) => iv.content.trim())
			.filter((s) => s.length > 0)
			.join(' ')
			.trim();
	});
	const caption = $derived(textParam || textgridText);

	const modelTiers = $derived.by<Tier[]>(() => {
		if (!modelTextgrid) return [];
		return Object.entries(modelTextgrid)
			.filter(([name]) => name.toLowerCase() !== 'text')
			.map(([name, intervals]) => ({
				name,
				regions: intervals.map((iv, i) => ({
					id: `model-${name}-${i}`,
					start: iv.start,
					end: iv.end,
					content: iv.content,
					color: TEXTGRID_COLOR
				}))
			}));
	});

	// ---------- Learner track ----------
	let recorder: WavesurferRecorder | undefined = $state();
	let learnerAudio = $state<Blob | undefined>();
	let isRecording = $state(false);
	let startingRecording = $state(false);
	let processing = $state(false);
	let recordError = $state('');

	async function processRecording(blob: Blob) {
		processing = true;
		recordError = '';
		const formData = new FormData();
		formData.append('file', blob, 'recording.wav');
		formData.append('apply_vad', 'true');
		try {
			learnerAudio = await postBlob('/api/process_audio', formData);
		} catch (e: unknown) {
			console.error('Error processing recording:', e);
			recordError = (e as Error)?.message ?? 'Could not process the recording.';
		} finally {
			processing = false;
		}
	}

	function toggleRecording() {
		if (!recorder) return;
		if (isRecording) {
			recorder.stopRecording().then((result) => {
				isRecording = false;
				if (result.duration < 500) {
					recordError = 'That recording was too short. Please try again.';
					return;
				}
				processRecording(result.blob);
			});
		} else {
			startingRecording = true;
			recordError = '';
			recorder
				.startRecording()
				.then(() => {
					isRecording = true;
				})
				.catch((e: unknown) => {
					console.error('Error starting recording:', e);
					recordError = 'Could not start recording. Please allow microphone access.';
				})
				.finally(() => {
					startingRecording = false;
				});
		}
	}

	// ---------- Comparison ----------
	let threshold = $state(0.6);
	let scores = $state<number[]>([]);
	let alignmentMap = $state<number[] | undefined>();
	let alignedTimes = $state<number[][] | undefined>();
	let learnerSegments = $state<Segment[] | undefined>();
	let comparing = $state(false);
	let compareError = $state('');

	// [modelTime, learnerTime] pairs, i.e. the inverse of alignedTimes.
	const modelAlignedTimes = $derived(
		alignedTimes ? alignedTimes.map(([t1, t2]) => [t2, t1]).sort((a, b) => a[0] - b[0]) : undefined
	);

	let modelViewer: SampleViewer | undefined = $state();
	let learnerViewer: SampleViewer | undefined = $state();

	$effect(() => {
		const learner = learnerAudio;
		const model = modelAudio;
		if (!learner || !model) {
			scores = [];
			alignmentMap = undefined;
			alignedTimes = undefined;
			learnerSegments = undefined;
			return;
		}

		const controller = new AbortController();
		let aborted = false;
		(async () => {
			comparing = true;
			compareError = '';
			scores = [];
			alignmentMap = undefined;
			alignedTimes = undefined;
			learnerSegments = undefined;

			const formData = new FormData();
			formData.append('file', learner, 'recording.wav');
			formData.append('model_file', model, 'model.wav');
			formData.append('encoder', ENCODER);

			try {
				const data = await postJson<{
					scores: number[];
					alignmentMap?: number[];
					alignedTimes?: number[][];
					learnerSegments?: Segment[];
				}>('/api/compare', formData, controller.signal);
				if (aborted) return;
				scores = data.scores ?? [];
				alignmentMap = data.alignmentMap;
				alignedTimes = data.alignedTimes;
				learnerSegments = data.learnerSegments;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name === 'AbortError') return;
				console.error('Error comparing audio:', e);
				if (!aborted) compareError = (e as Error)?.message ?? 'Could not compare the recordings.';
			} finally {
				if (!aborted) comparing = false;
			}
		})();

		return () => {
			aborted = true;
			controller.abort();
		};
	});

	/* Contiguous stretches of the learner's speech that are far from the model.
	   Region labels (model frame indices) are dropped: they mean nothing to a
	   learner and the colour already carries the message. */
	const learnerRegions = $derived.by<Region[]>(() => {
		if (!learnerSegments || scores.length === 0) return [];
		return buildContinuousRegions(
			scores,
			learnerSegments,
			threshold,
			threshold - 0.05,
			true,
			alignmentMap
		).map((r) => ({ ...r, content: '' }));
	});

	/* Piecewise-linear lookup over sorted [from, to] time pairs. */
	function mapTime(pairs: number[][], t: number): number {
		let low = 0;
		let high = pairs.length - 1;
		while (low <= high) {
			const mid = (low + high) >> 1;
			if (pairs[mid][0] <= t) low = mid + 1;
			else high = mid - 1;
		}
		const i = low - 1;
		if (i < 0) return pairs[0][1];
		if (i >= pairs.length - 1) return pairs[pairs.length - 1][1];
		const [t0, u0] = pairs[i];
		const [t1, u1] = pairs[i + 1];
		if (t1 === t0) return u1;
		return u0 + ((t - t0) / (t1 - t0)) * (u1 - u0);
	}

	/* The model's phone/word intervals warped onto the learner's timeline, so a
	   marked stretch can be read off against the sound it was supposed to be. */
	const learnerTextgridTiers = $derived.by<Tier[]>(() => {
		const pairs = modelAlignedTimes;
		if (!pairs || pairs.length === 0 || modelTiers.length === 0) return [];
		return modelTiers.map((tier) => ({
			name: tier.name,
			regions: tier.regions.flatMap((r) => {
				const start = mapTime(pairs, r.start);
				const end = mapTime(pairs, r.end);
				if (!(end > start)) return [];
				return [{ ...r, id: `learner-${r.id}`, start, end }];
			})
		}));
	});

	/* The waveform slot belongs to the recorder from the moment recording starts
	   until the new attempt is decoded, so the previous one never flashes back. */
	const slotHeld = $derived(isRecording || processing);

	const busy = $derived(modelLoading || processing || comparing);
	const recordDisabled = $derived(!modelAudio || startingRecording || processing);
</script>

<svelte:head>
	<title>Pronunciation Practice</title>
</svelte:head>

<div class="page" class:waiting={busy}>
	<header>
		<h1>Pronunciation Practice</h1>
	</header>

	{#if caption}
		<p class="caption">{caption}</p>
	{/if}

	<section class="card">
		<h2>Model</h2>
		{#if modelError}
			<p class="error">{modelError}</p>
		{/if}
		{#if modelAudio}
			<SampleViewer
				audio={modelAudio}
				tiers={modelTiers}
				bind:this={modelViewer}
				tierMenu={false}
				compareWith={learnerViewer
					? { other: learnerViewer, alignedTimes: modelAlignedTimes }
					: null}
			/>
		{:else if !modelError}
			<p class="placeholder">
				{modelLoading
					? 'Loading the model audio…'
					: 'No practice file was given. Please open the link for the exercise you want to practise.'}
			</p>
		{/if}
	</section>

	<section class="card">
		<h2>You</h2>

		<!-- One waveform slot: the live recorder takes the place of the last
		     attempt while recording, then hands it back. The recorder stays
		     mounted so it keeps its width and can be started at any time; when
		     idle it is clipped to nothing rather than left as an empty box. It
		     stays held through processing: the old attempt must not flash back
		     while its replacement is still on the wire. -->
		<div class="recorder-slot" class:collapsed={!slotHeld}>
			<WavesurferRecorder bind:this={recorder} />
		</div>

		{#if learnerAudio && !slotHeld}
			<SampleViewer
				audio={learnerAudio}
				tiers={[{ name: 'Distance', regions: learnerRegions }, ...learnerTextgridTiers]}
				bind:this={learnerViewer}
				tierMenu={false}
				compareWith={modelViewer ? { other: modelViewer, alignedTimes } : null}
			/>
			<p class="hint">Red marks the parts that sound least like the model.</p>
		{/if}

		<div class="record-row">
			<button class="record-button" onclick={toggleRecording} disabled={recordDisabled}>
				{#if isRecording}
					<svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor">
						<path d="M6 6h12v12H6z" />
					</svg>
					Stop
				{:else}
					<svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor">
						<circle cx="12" cy="12" r="8" />
					</svg>
					{learnerAudio ? 'Record again' : 'Record'}
				{/if}
			</button>
			<span class="status">
				{#if isRecording}
					Recording… press Stop when you finish.
				{:else if processing}
					Processing your recording…
				{:else if comparing}
					Comparing…
				{:else if !modelAudio}
					Waiting for the practice file.
				{:else if !learnerAudio}
					Listen to the model, then record yourself saying the same thing.
				{/if}
			</span>
		</div>

		{#if recordError}
			<p class="error">{recordError}</p>
		{/if}
		{#if compareError}
			<p class="error">{compareError}</p>
		{/if}
	</section>

	<section class="instructions">
		<h2>Controls</h2>
		<ul>
			<li>Click on a waveform to move to that point.</li>
			<li>Drag across a waveform to play just that stretch.</li>
			<li>Click on a coloured block to play it.</li>
			<li>
				Hold <kbd>Shift</kbd> while doing any of the above to do it on the other track instead, at the
				matching moment. Use it to hear a red stretch of your own speech against the model.
			</li>
		</ul>
	</section>

	<section class="settings">
		<h2>Settings</h2>
		<label>
			Sensitivity:
			<input type="range" min="0.0" max="1.0" step="0.05" bind:value={threshold} />
			<span class="threshold-value">{threshold.toFixed(2)}</span>
		</label>
		<p class="hint">Move right to mark more of your speech, left to mark only the worst parts.</p>
	</section>
</div>

<style>
	.page {
		max-width: 1400px;
		margin: 0 auto;
		padding: 0.75rem 1rem 2rem;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	header h1 {
		margin: 0.5rem 0 0;
		font-size: 1.5rem;
	}

	.caption {
		margin: 0;
		font-size: 1.25rem;
		padding: 0.6rem 0.8rem;
		background-color: var(--surface-color);
		border: 1px solid var(--border-color);
		border-left: 4px solid var(--primary-color);
		border-radius: 4px;
	}

	.card {
		background-color: var(--surface-color);
		border: 1px solid var(--border-color);
		border-radius: 8px;
		padding: 0.75rem;
	}

	.card h2 {
		margin: 0 0 0.5rem;
		font-size: 1.05rem;
	}

	.placeholder {
		margin: 0.5rem 0;
		opacity: 0.7;
		font-style: italic;
	}

	/* Clipped rather than unmounted: WavesurferRecorder must stay alive to be
	   started, and a zero-height wrapper keeps its full width, so the live
	   waveform is measured correctly the moment it is revealed. */
	.recorder-slot.collapsed {
		height: 0;
		overflow: hidden;
	}

	.record-row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75em;
	}

	.record-button {
		display: flex;
		align-items: center;
		gap: 0.4em;
		padding: 0.5em 1em;
		font-size: 1.05em;
	}

	.status {
		opacity: 0.8;
	}

	.error {
		margin: 0.5rem 0;
		color: #dc3545;
	}

	.hint {
		margin: 0.5rem 0 0;
		font-size: 0.9em;
		opacity: 0.75;
	}

	.instructions,
	.settings {
		background-color: var(--surface-color);
		border: 1px solid var(--border-color);
		border-radius: 8px;
		padding: 0.75rem;
	}

	.instructions h2,
	.settings h2 {
		margin: 0 0 0.5rem;
		font-size: 1.05rem;
	}

	.instructions ul {
		margin: 0;
		padding-left: 1.4em;
		display: flex;
		flex-direction: column;
		gap: 0.25em;
	}

	kbd {
		font-family: var(--font-family-monospace);
		font-size: 0.9em;
		padding: 0.05em 0.35em;
		border: 1px solid var(--border-color);
		border-bottom-width: 2px;
		border-radius: 3px;
		background-color: var(--background-color);
	}

	.settings label {
		display: flex;
		align-items: center;
		gap: 0.5em;
	}

	.threshold-value {
		font-family: var(--font-family-monospace);
	}
</style>
