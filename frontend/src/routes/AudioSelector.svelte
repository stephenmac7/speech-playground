<script lang="ts">
	import { getBlob, getJson, postBlob } from '$lib/api';
	import { reportError } from '$lib/errors';
	import type { TextGridData } from '$lib/db';
	import WavesurferRecorder from './WavesurferRecorder.svelte';

	// eslint-disable-next-line no-useless-assignment
	let {
		recorder,
		value = $bindable(),
		textgrid = $bindable(),
		transcript = $bindable()
	}: {
		recorder: WavesurferRecorder;
		value: Blob | undefined;
		textgrid: TextGridData | undefined;
		transcript: string | undefined;
	} = $props();

	let serverPopoverOpen = $state(false);
	let transcriptEditing = $state(false);
	let transcriptDraft = $state('');

	function startEditTranscript() {
		transcriptDraft = transcript ?? '';
		transcriptEditing = true;
	}
	function commitTranscript() {
		if (!transcriptEditing) return;
		transcriptEditing = false;
		if (transcriptDraft !== (transcript ?? '')) {
			transcript = transcriptDraft;
		}
	}
	function cancelTranscript() {
		transcriptEditing = false;
	}

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
				// Try to auto-fetch corresponding TextGrid
				try {
					textgrid = await getJson<TextGridData>(
						`/api/data_tg/${apiServerFilePath}`,
						controller.signal
					);
				} catch {
					textgrid = undefined;
				}
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
		textgrid = undefined;

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
		serverPopoverOpen = false;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			setServerFilePath();
		}
	}

	function handleWindowKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape' && serverPopoverOpen) serverPopoverOpen = false;
	}
</script>

<svelte:window onkeydown={handleWindowKeydown} />

<div class="audio-selector">
	<div class="button-row">
		<!-- Recording -->
		<button onclick={toggleRecording} disabled={busy}>
			{#if isRecording}
				<svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor">
					<path d="M6 6h12v12H6z" />
				</svg>
				Stop
			{:else}
				<svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor">
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

		<!-- Server file (popover) -->
		<div class="server-file-anchor">
			<button onclick={() => (serverPopoverOpen = !serverPopoverOpen)}>Server...</button>
			{#if serverPopoverOpen}
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="server-popover-backdrop"
					onclick={(e) => {
						e.stopPropagation();
						serverPopoverOpen = false;
					}}
				></div>
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<div class="server-popover" onclick={(e) => e.stopPropagation()}>
					<label for="server-file-input">Server file:</label>
					<div class="server-file-controls">
						<input
							id="server-file-input"
							type="text"
							bind:value={serverFilePath}
							onkeydown={handleKeydown}
							{@attach (el: HTMLInputElement) => {
								el.focus();
								el.select();
							}}
						/>
						<button onclick={setServerFilePath}>Set</button>
					</div>
					{#if apiServerFilePath instanceof Error}
						<span class="error">{apiServerFilePath.message}</span>
					{/if}
				</div>
			{/if}
		</div>
	</div>

	<!-- Transcript -->
	<div class="transcript-row">
		{#if transcriptEditing}
			<input
				class="transcript-input"
				type="text"
				placeholder="Transcript"
				bind:value={transcriptDraft}
				onblur={commitTranscript}
				onkeydown={(e) => {
					if (e.key === 'Enter') commitTranscript();
					else if (e.key === 'Escape') cancelTranscript();
				}}
				{@attach (el: HTMLInputElement) => {
					el.focus();
					el.select();
				}}
			/>
		{:else}
			<span class="transcript-text" class:empty={!transcript}>
				{transcript || 'No transcript'}
			</span>
		{/if}
		<button
			class="transcript-edit-button"
			title="Edit transcript"
			onclick={() => (transcriptEditing ? commitTranscript() : startEditTranscript())}
		>
			<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d="M12 20h9" />
				<path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
			</svg>
		</button>
	</div>
</div>

<style>
	.audio-selector {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
	}

	.button-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5em;
	}
	.button-row > :global(*) {
		flex: 1 1 auto;
	}

	.server-file-anchor {
		position: relative;
		display: flex;
	}
	.server-file-anchor > button {
		flex: 1;
	}

	.server-popover-backdrop {
		position: fixed;
		inset: 0;
		z-index: 99;
	}

	.server-popover {
		position: absolute;
		top: calc(100% + 4px);
		right: 0;
		z-index: 100;
		display: flex;
		flex-direction: column;
		gap: 0.25em;
		min-width: 260px;
		padding: 0.6em;
		background: var(--surface-color, #fff);
		border: 1px solid var(--border-color, #e5e7eb);
		border-radius: 6px;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
	}

	.transcript-row {
		display: flex;
		align-items: center;
		gap: 0.5em;
		min-height: 1.8em;
	}
	.transcript-text {
		flex: 1;
		min-width: 0;
		color: var(--foreground-color);
		white-space: normal;
		overflow-wrap: anywhere;
	}
	.transcript-text.empty {
		opacity: 0.5;
		font-style: italic;
	}
	.transcript-input {
		flex: 1;
		padding: 0.25rem 0.4rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		background-color: var(--surface-color);
		color: var(--foreground-color);
		font-family: var(--font-family-sans-serif);
	}
	.transcript-edit-button {
		padding: 0.2em 0.35em;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--surface-color);
		border: 1px solid var(--border-color);
		border-radius: 4px;
		cursor: pointer;
		opacity: 0.6;
	}
	.transcript-edit-button:hover {
		opacity: 1;
		background-color: var(--background-color);
	}

	button,
	.file-input {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.4em;
		text-align: center;
	}

	.file-input {
		border: 1px solid var(--border-color);
		padding: 0.35em 0.5em;
		cursor: pointer;
		background-color: var(--surface-color);
		color: var(--foreground-color);
		border-radius: 4px;
		font-family: var(--font-family-sans-serif);
	}

	button:disabled {
		cursor: wait;
	}

	.file-input:hover {
		background-color: var(--background-color);
	}

	.file-input input[type='file'] {
		display: none;
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
