<script lang="ts">
	import { getBlob, postBlob } from '$lib/api';
	import { reportError } from '$lib/errors';
	import WavesurferRecorder from './WavesurferRecorder.svelte';

	let { recorder, value = $bindable() }: { recorder: WavesurferRecorder; value: Blob | undefined } =
		$props();

	let startingRecording = $state(false);
	let isRecording = $state(false);

	let processingAudio = $state(false);

	let busy = $derived(processingAudio || startingRecording);

	let apiServerFilePath = $state<string | Error>('');
	let serverFilePath = $state('');
	let serverAudio = $state<Blob | undefined>();

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!apiServerFilePath || apiServerFilePath instanceof Error) {
				return;
			}

			try {
				serverAudio = await getBlob(`/api/data/${apiServerFilePath}`, controller.signal);
				value = serverAudio;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError') {
					reportError(`Error fetching server audio.`, e);
				}
			}
		})();

		return () => controller.abort();
	});

	async function handleProcessAudio(blob: Blob, apply_vad: boolean) {
		processingAudio = true;

		const formData = new FormData();
		formData.append('file', blob, 'recording.wav');
		formData.append('apply_vad', String(apply_vad));

		try {
			const blob = await postBlob('/api/process_audio', formData);
			value = blob;
		} catch (e: unknown) {
			reportError('Error processing audio.', e);
		}
		processingAudio = false;
	}

	function toggleRecording() {
		if (isRecording) {
			recorder.stopRecording().then((result) => {
				const duration = result.duration;
				if (duration < 500) {
					alert(`Recording too short. Please record at least 500ms. Got ${duration}ms`);
					isRecording = false;
					return;
				}
				handleProcessAudio(result.blob, true);
				isRecording = false;
			});
		} else {
			startingRecording = true;
			recorder
				.startRecording()
				.then(() => {
					isRecording = true;
				})
				.finally(() => {
					startingRecording = false;
				});
		}
	}

	function handleFileSelect(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (file) {
			if (isRecording) {
				recorder.cancelRecording();
				isRecording = false;
			}
			handleProcessAudio(file, false);
		}
		target.value = '';
	}

	function setServerFilePath() {
		if (serverFilePath == apiServerFilePath && serverAudio) {
			value = serverAudio; // just set the audio to the existing server audio
		} else {
			apiServerFilePath = serverFilePath; // will trigger fetch
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			setServerFilePath();
		}
	}
</script>

<div class="audio-selector">
	<div class="button-row">
		<!-- Recording -->
		<button onclick={toggleRecording} disabled={busy}>
			{#if isRecording}
				<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
					<path d="M6 6h12v12H6z" />
				</svg>
				Stop
			{:else}
				<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
					<circle cx="12" cy="12" r="8" />
				</svg>
				Record
			{/if}
		</button>

		<!-- File upload -->
		<label class="file-input">
			Browse...
			<input type="file" onchange={handleFileSelect} />
		</label>
	</div>

	<!-- Server file -->
	<div class="server-file">
		<label for="server-file-input">Server file:</label>
		<div class="server-file-controls">
			<input
				id="server-file-input"
				type="text"
				bind:value={serverFilePath}
				onkeydown={handleKeydown}
			/>
			<button onclick={setServerFilePath}>Set</button>
		</div>
		{#if apiServerFilePath instanceof Error}
			<span class="error">{apiServerFilePath.message}</span>
		{/if}
	</div>
</div>

<style>
	.audio-selector {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
	}

	.button-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5em;
	}

	button,
	.file-input {
		border: 1px solid var(--border-color);
		padding: 0.5em 0.75em;
		cursor: pointer;
		background-color: var(--surface-color);
		color: var(--foreground-color);
		border-radius: 4px;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5em;
		font-family: var(--font-family-sans-serif);
		font-size: 1rem;
		text-align: center;
	}

	button:disabled {
		cursor: wait;
		opacity: 0.6;
	}

	button:hover,
	.file-input:hover {
		background-color: var(--background-color);
	}
	button:disabled:hover {
		/* prevent hover color change when disabled */
		background-color: var(--surface-color);
	}

	.file-input input[type='file'] {
		display: none;
	}

	.server-file {
		display: flex;
		flex-direction: column;
		gap: 0.25em;
	}

	.server-file-controls {
		display: flex;
		gap: 0.5em;
	}

	.server-file-controls input {
		flex-grow: 1;
		padding: 0.5rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		background-color: var(--surface-color);
		color: var(--foreground-color);
	}

	.error {
		color: #dc3545;
		font-size: 0.8em;
	}
</style>
