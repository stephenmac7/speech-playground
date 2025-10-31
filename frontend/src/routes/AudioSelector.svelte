<script lang="ts">
	import { PUBLIC_DATA_PREFIX, PUBLIC_EXAMPLE_PATH } from '$env/static/public';

    let { recorder, value = $bindable() } = $props();

	let isRecording = $state(false);

	let processingAudio = $state(false);

	let apiServerFilePath = $state<string | Error>('');
	let serverFilePath = $state('');
    let serverAudio = $state<Blob | undefined>();

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!apiServerFilePath || apiServerFilePath instanceof Error) {
				return;
			}

			const response = await fetch(`/api/data/${apiServerFilePath}`, {
				signal: controller.signal
			});
            if (response.ok) {
                serverAudio = await response.blob();
                value = serverAudio;
            } else {
                console.error(`Error fetching server audio: ${response.statusText}`);
            }
		})();

		return () => controller.abort();
	});

	async function handleProcessAudio(blob : Blob, apply_vad : boolean) {
		processingAudio = true;

		const formData = new FormData();
		formData.append('file', blob, 'recording.wav');
		formData.append('apply_vad', String(apply_vad));

		const response = await fetch('/api/process_audio', {
			method: 'POST',
			body: formData,
		});

		if (response.ok) {
			value = await response.blob();
		} else {
			try {
				const errorResult = await response.json();
				console.error('Error processing audio:', errorResult.detail);
			} catch (e) {
				console.error('Error processing audio:', e);
			}
		}
		processingAudio = false;
	}

	function toggleRecording() {
		if (isRecording) {
			recorder.stopRecording().then((blob: Blob) => {
                handleProcessAudio(blob, true);
                isRecording = false;
            });
		} else {
			recorder.startRecording();
            isRecording = true;
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
		};
		target.value = '';
	}

	function setServerFilePath() {
		if (serverFilePath.startsWith(PUBLIC_DATA_PREFIX)) {
			const nextValue = serverFilePath.substring(PUBLIC_DATA_PREFIX.length);
            if (nextValue == apiServerFilePath) {
                value = serverAudio; // just set the audio to the existing server audio
            } else {
                apiServerFilePath = nextValue; // will trigger fetch
            }
		} else {
			apiServerFilePath = Error(`File must be in ${PUBLIC_DATA_PREFIX}`);
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			setServerFilePath();
		}
	}
</script>

<div class="audio-selector" class:busy={processingAudio}>
	<div class="button-row">
		<!-- Recording -->
		<button onclick={toggleRecording}>
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
				placeholder="{PUBLIC_DATA_PREFIX}..."
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

	.audio-selector.busy {
		cursor: wait;
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

	button:hover,
	.file-input:hover {
		background-color: var(--background-color);
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
