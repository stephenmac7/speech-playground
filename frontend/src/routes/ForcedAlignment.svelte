<script lang="ts">
	import { PUBLIC_EXAMPLE_TRANSCRIPT } from '$env/static/public';
	import SampleViewer from './SampleViewer.svelte';
	import { postJson } from '$lib/api';

	let { tracks, active } = $props();

	let audio = $state(tracks['Audio']);
	$effect(() => {
		if (active) audio = tracks['Audio'];
	});

	let transcript = $state(PUBLIC_EXAMPLE_TRANSCRIPT);
	type Alignment = Record<string, unknown>;
	let alignment: Alignment = $state({});
	let mode = $state('phones');
	let regions = $derived(() => {
		if (mode === 'none') return [];
		const val = alignment[mode];
		return Array.isArray(val) ? val : [];
	});

	let gettingIntervals = $state(false);

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audio) {
				return;
			}

			gettingIntervals = true;
			const formData = new FormData();
			formData.append('audio', audio, 'recording.wav');
			formData.append('transcript', transcript);
			try {
				const result = await postJson<Alignment>('/api/align', formData, controller.signal);
				alignment = result;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError') {
					console.error('Error fetching intervals:', e);
					alignment = {};
				}
			} finally {
				gettingIntervals = false;
			}
		})();

		return () => controller.abort();
	});
</script>

<div class="viewer-card">
	<SampleViewer {audio} {regions} />
</div>

<div class="controls">
	{#each ['phones', 'words', 'none'] as m (m)}
		<label>
			<input type="radio" name="mode" value={m} bind:group={mode} />
			{m.charAt(0).toUpperCase() + m.slice(1)}
			<!-- Capitalize properly -->
		</label>
	{/each}

	<input type="text" bind:value={transcript} placeholder="Transcript" />
</div>

{#if gettingIntervals}
	<p>Loading forced alignment...</p>
{/if}

<style>
	.controls {
		display: flex;
		gap: 1rem;
		align-items: center;
		margin-top: 1rem;
	}

	input[type='text'] {
		flex-grow: 1;
		padding: 0.5rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		background-color: var(--surface-color);
		color: var(--foreground-color);
	}

	.viewer-card {
		background-color: var(--surface-color);
		border-radius: 8px;
		border: 1px solid var(--border-color);
		padding: 1rem;
	}
</style>
