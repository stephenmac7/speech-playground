<script lang="ts">
	import ForcedAlignment from './ForcedAlignment.svelte';
	import PhoneticTranscription from './PhoneticTranscription.svelte';
	import Diff from './Diff.svelte';
	import KeepAlive from './KeepAlive.svelte';
	import AudioLibrary from './AudioLibrary.svelte';

	let tools = $state(['forced_alignment']);

	const toolOptions = [
		{ id: 'forced_alignment', label: 'Forced Alignment' },
		{ id: 'transcribe', label: 'Phonetic Transcription' },
		{ id: 'diff', label: 'Diff' }
	];

	let requestedTracks = $derived(tools.includes('diff') ? ['Model', 'Audio'] : ['Audio']);

	let tracks = $state({});
</script>

<div class="app-container">
	<header>
		<div class="header-content">
			<h1>Speech Playground</h1>
			<nav class="tool-selection">
				<span>Tools:</span>
				{#each toolOptions as option (option.id)}
					<label>
						<input type="checkbox" bind:group={tools} value={option.id} />
						{option.label}
					</label>
				{/each}
			</nav>
		</div>
	</header>

	<main>
		<div class="tool-display">
			<KeepAlive active={tools.includes('forced_alignment')}>
				{#if tools.length > 1}<h3>Forced Alignment</h3>{/if}
				<ForcedAlignment {tracks} active={tools.includes('forced_alignment')} />
			</KeepAlive>

			<KeepAlive active={tools.includes('transcribe')}>
				{#if tools.length > 1}<h3>Phonetic Transcription</h3>{/if}
				<PhoneticTranscription {tracks} active={tools.includes('transcribe')} />
			</KeepAlive>

			<KeepAlive active={tools.includes('diff')}>
				{#if tools.length > 1}<h3>Diff</h3>{/if}
				<Diff {tracks} active={tools.includes('diff')} />
			</KeepAlive>
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
		padding: 1rem;
		max-width: 1800px;
		margin: 0 auto;
	}

	main {
		display: flex;
		flex-direction: row;
		gap: 1.5rem;
		padding: 1rem;
		max-width: 1800px;
		margin: 0 auto;
		width: 100%;
		flex-grow: 1;
		overflow-y: hidden;
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
		gap: 1rem;
		overflow-y: auto;
		scrollbar-width: none;
	}
	.tool-display::-webkit-scrollbar {
		display: none;
	}

	aside {
		flex: 1 1 0;
		border-left: 1px solid var(--border-color);
		padding-left: 1.5rem;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
</style>
