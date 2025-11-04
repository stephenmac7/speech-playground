<script lang="ts">
	import AudioSelector from './AudioSelector.svelte';
	import SampleViewer from './SampleViewer.svelte';
	import WavesurferRecorder from './WavesurferRecorder.svelte';
	import { untrack } from 'svelte';
	import { reportError } from '$lib/errors';

	import { db } from '$lib/db';
	import { liveQuery } from 'dexie';

	let {
		requestedTracks,
		tracksByKey = $bindable()
	}: { requestedTracks: string[]; tracksByKey: Record<string, Blob | null> } = $props();

	let recorder: WavesurferRecorder | undefined = $state();
	let selectedTrackRequestKey = $state<string | null>(null);

	const colors = ['#ffff99', '#fb9a99', '#a6cee3', '#fdbf6f', '#cab2d6', '#b2df8a'];

	function colorOfString(input: string): string {
		let hash = 5381; // Initial seed
		for (let i = 0; i < input.length; i++) {
			hash = (hash << 5) + hash + input.charCodeAt(i);
		}
		return colors[(hash & 0x7fffffff) % 6];
	}

	let tracks = liveQuery(() => db.audio_tracks.toArray());

	$effect(() => {
		if (!$tracks) {
			return;
		}

		let newTracksByKey: Record<string, Blob | null> = { ...untrack(() => tracksByKey) };
		requestedTracks.forEach((key: string) => {
			const track = $tracks.find((t) => t.keys?.includes(key));
			if (track && track.data) {
				newTracksByKey[key] = track.data;
			} else {
				newTracksByKey[key] = null;
			}
		});
		tracksByKey = newTracksByKey;
	});

	async function addTrack() {
		try {
			await db.audio_tracks.add({ keys: [] });
		} catch (error) {
			reportError('Error adding track to database:', error);
		}
	}

	function deleteTrack(id: number) {
		db.audio_tracks.delete(id);
	}

	function updateTrack(id: number, blob: Blob | undefined) {
		db.audio_tracks.update(id, { data: blob });
	}

	function selectTrackRequest(key: string) {
		if (selectedTrackRequestKey === key) {
			selectedTrackRequestKey = null;
		} else {
			selectedTrackRequestKey = key;
		}
	}

	async function assignTrack(e: MouseEvent, trackId: number) {
		const selectedKey = selectedTrackRequestKey; // narrowing
		if (selectedKey) {
			// Clear previous assignment for this key
			const oldTrack = $tracks?.find((t) => t.keys?.includes(selectedKey));
			if (oldTrack && oldTrack.id === trackId) {
				// The user clicked the track that is already assigned to this key.
				// Do nothing to avoid a race condition.
				selectedTrackRequestKey = null;
				e.stopPropagation();
				return;
			}

			if (oldTrack) {
				const newKeys = oldTrack.keys.filter((k) => k !== selectedKey);
				await db.audio_tracks.update(oldTrack.id, { keys: newKeys });
			}
			// Assign new track
			const newTrack = $tracks?.find((t) => t.id === trackId);
			if (newTrack) {
				const newKeys = [...(newTrack.keys || []), selectedKey];
				await db.audio_tracks.update(trackId, { keys: newKeys });
			}
			selectedTrackRequestKey = null;
			e.stopPropagation();
		}
	}
</script>

<h3>Library</h3>

<WavesurferRecorder bind:this={recorder} />

<div class="requested-tracks">
	{#each requestedTracks as key (key)}
		<button
			class="requested-track-button"
			class:selected={selectedTrackRequestKey === key}
			style:--selection-color={colorOfString(key)}
			onclick={() => selectTrackRequest(key)}
		>
			Set {key}
		</button>
	{/each}
</div>

<div class="audio-library">
	{#each $tracks as track (track.id)}
		<fieldset
			class="track"
			class:selecting={selectedTrackRequestKey !== null}
			onclickcapture={(e) => assignTrack(e, track.id)}
			style={track.keys?.filter((k) => requestedTracks.includes(k)).length
				? `--selection-color: ${colorOfString(
						track.keys?.filter((k) => requestedTracks.includes(k))[0]
					)}`
				: ''}
		>
			{#if track.keys?.length}
				<div class="track-assignments">
					{#each track.keys.filter((k) => requestedTracks.includes(k)) as key (key)}
						<div class="track-assignment" style:--selection-color={colorOfString(key)}></div>
					{/each}
				</div>
			{/if}
			<button class="delete-track" onclick={() => deleteTrack(track.id)} title="Delete track">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="24"
					height="24"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
					<line x1="18" y1="6" x2="6" y2="18"></line>
					<line x1="6" y1="6" x2="18" y2="18"></line>
				</svg>
			</button>
			<SampleViewer audio={track.data} zoom={false} layout="compact" />
			<AudioSelector
				{recorder}
				bind:value={() => track.data, (blob) => updateTrack(track.id, blob)}
			/>
		</fieldset>
	{/each}

	<button class="add-track" onclick={addTrack}>Add Track</button>
</div>

<style>
	:global(.audio-library-wrapper) {
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
	.requested-tracks {
		display: flex;
		gap: 0.5em;
		margin-bottom: 1em;
	}
	.requested-track-button {
		border: 1px solid var(--border-color);
		padding: 0.5em 0.75em;
		cursor: pointer;
		background-color: color-mix(in srgb, var(--selection-color) 25%, var(--surface-color));
		border-radius: 4px;
		font-family: var(--font-family-sans-serif);
		font-size: 1rem;
	}
	.requested-track-button.selected {
		outline: 2px solid var(--selection-color);
		outline-offset: 2px;
	}
	.audio-library {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 1em;
		padding: 0.6em;
		box-sizing: border-box;
		overflow-y: auto;
		scrollbar-width: none;
		max-height: 80em;
	}
	.audio-library::-webkit-scrollbar {
		display: none;
	}
	.track {
		position: relative;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1em;
		border: 1px solid var(--border-color);
		background-color: var(--surface-color);
		border-radius: 4px;
		padding: 1em;
		align-items: center;
	}
	.track.selecting {
		cursor: pointer;
	}
	.track.selecting > :global(*) {
		pointer-events: none;
	}
	.track:has(button:hover) {
		cursor: initial;
	}
	.track[style*='--selection-color'] {
		outline: 2px solid var(--selection-color);
		outline-offset: 2px;
	}
	.track-assignments {
		position: absolute;
		top: -0.5em;
		left: -0.5em;
		display: flex;
		gap: 0.3em;
	}
	.track-assignment {
		width: 1em;
		height: 1em;
		border-radius: 50%;
		background-color: var(--selection-color);
		border: 1px solid var(--border-color);
	}
	.add-track {
		border: 1px solid var(--border-color);
		padding: 0.5em 0.75em;
		cursor: pointer;
		background-color: var(--surface-color);
		border-radius: 4px;
		font-family: var(--font-family-sans-serif);
		font-size: 1rem;
		text-align: center;
	}
	.add-track:hover {
		background-color: var(--background-color);
	}
	.delete-track {
		position: absolute;
		top: -0.6em;
		right: -0.6em;
		background: var(--surface-color);
		border: 1px solid var(--border-color);
		border-radius: 50%;
		cursor: pointer;
		padding: 0.2em;
		width: 28px;
		height: 28px;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.delete-track:hover {
		opacity: 1;
	}
	.delete-track:disabled {
		display: none;
	}
</style>
