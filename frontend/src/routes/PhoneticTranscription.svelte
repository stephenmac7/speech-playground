<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';

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
				const intervals_resp = await fetch(`/api/ifmdd`, {
					method: 'POST',
					body: formData,
					signal: controller.signal,
				});
				const result = await intervals_resp.json();
				if (intervals_resp.ok) {
					regions = result['intervals'];
				} else {
					console.error(`Error fetching intervals: ${result['detail']}`);
				}
			} catch (e : any) {
				if (e.name !== 'AbortError') {
					console.error('Error fetching intervals:', e);
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