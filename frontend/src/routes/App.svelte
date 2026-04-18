<script lang="ts">
	import Diff from './Diff.svelte';
	import Analysis from './Analysis.svelte';
	import AudioLibrary from './AudioLibrary.svelte';
	import { getJson } from '$lib/api';
	import { reportError } from '$lib/errors';
	import type { ModelsResponse, EncoderConfig } from '$lib/types';

	let mode = $state<'analysis' | 'diff'>('diff');
	let tracks: Record<string, import('./AudioLibrary.svelte').TrackData> = $state({});
	let modelsConfig = $state<ModelsResponse | undefined>();
	let encoderConfig = $state<EncoderConfig>({
		encoder: 'wavlm-base-plus',
		discretize: false,
		discretizer: 'bshall',
		dpdp: true,
		gamma: '0.2'
	});

	let requestedTracks = $derived(mode === 'diff' ? ['Model', 'Query'] : ['Audio']);

	$effect(() => {
		const controller = new AbortController();
		(async () => {
			try {
				const config = await getJson<ModelsResponse>(`/api/models`, controller.signal);
				if (!config.encoders.some((o) => o.value === encoderConfig.encoder) && config.encoders.length) {
					encoderConfig.encoder = config.encoders[0].value;
				}
				modelsConfig = config;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					reportError('Error fetching encoders.', e);
			}
		})();
		return () => controller.abort();
	});
</script>

<div class="app-container">
	<header>
		<div class="header-content">
			<h1>Speech Playground</h1>
			<nav class="tool-selection">
				<label>
					Mode:
					<select bind:value={mode}>
						<option value="analysis">Analysis</option>
						<option value="diff">Diff</option>
					</select>
				</label>
			</nav>
		</div>
	</header>

	<main>
		<div class="tool-display">
			{#if modelsConfig}
				{#if mode === 'analysis'}
					<Analysis {tracks} {modelsConfig} bind:encoderConfig />
				{:else}
					<Diff {tracks} {modelsConfig} bind:encoderConfig />
				{/if}
			{/if}
		</div>

		<aside>
			<AudioLibrary {requestedTracks} bind:tracksByKey={tracks} />
		</aside>
	</main>
</div>

<style>
	.app-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
	}

	header {
		border-bottom: 1px solid var(--border-color);
		background-color: var(--surface-color);
	}

	.header-content {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 0.75rem;
		max-width: 2100px;
		margin: 0 auto;
	}

	main {
		display: flex;
		flex-direction: row;
		gap: 1rem;
		padding: 0 0.75rem;
		max-width: 2100px;
		margin: 0 auto;
		width: 100%;
		flex-grow: 1;
		overflow-y: hidden;
		box-sizing: border-box;
	}

	.tool-selection {
		display: flex;
		gap: 1rem;
		align-items: center;
	}

	.tool-selection label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.tool-display {
		flex: 2 1 0;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		overflow-y: auto;
		scrollbar-width: none;
		padding: 0.5rem 0;
	}
	.tool-display::-webkit-scrollbar {
		display: none;
	}

	aside {
		flex: 1 1 0;
		border-left: 1px solid var(--border-color);
		padding-left: 1rem;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
</style>
