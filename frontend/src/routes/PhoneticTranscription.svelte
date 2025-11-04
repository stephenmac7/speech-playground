<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import { postJson } from '$lib/api';
	import { reportError } from '$lib/errors';

	let { tracks, active } = $props();

	let audio = $state(tracks['Audio']);
	$effect(() => {
		if (active) audio = tracks['Audio'];
	});

	let regions: Array<{ start: number; end: number; content: string }> = $state([]);
	let loading = $state(false);

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audio) {
				return;
			}

			loading = true;
			const formData = new FormData();
			formData.append('file', audio, 'recording.wav');
			try {
				const result = await postJson<{
					intervals: Array<{ start: number; end: number; content: string }>;
				}>('/api/ifmdd', formData, controller.signal);
				regions = result.intervals;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError') {
					reportError('Error fetching intervals.', e);
				}
			} finally {
				loading = false;
			}
		})();
	});
</script>

<div class={loading ? 'waiting' : ''}>
	<div class="viewer-card">
		<SampleViewer {audio} {regions} />
	</div>
</div>

<style>
	.viewer-card {
		background-color: var(--surface-color);
		border-radius: 8px;
		border: 1px solid var(--border-color);
		padding: 1rem;
	}
</style>
