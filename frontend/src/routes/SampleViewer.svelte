<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom.js';
	import RegionsPlugin, { type RegionParams } from 'wavesurfer.js/dist/plugins/regions.js';

	let { audio, regions = [], clickableRegions = true, zoom = true, layout = 'default' } = $props();

	let node: HTMLElement;

	/* wavesurfer + state */
	let wavesurfer: WaveSurfer;
	let playing = $state(false);
	let time = $state(0);
	let duration = $state(0);

	let height = $derived(layout === 'compact' ? 70 : 128);

	$effect(() => {
		if (!audio) return;
		let plugins = [];
		if (zoom) plugins.push(ZoomPlugin.create({}));
		let regionsPlugin: RegionsPlugin | undefined;
		if (regions.length > 0) {
			regionsPlugin = RegionsPlugin.create();
			plugins.push(regionsPlugin);
		}
		wavesurfer = WaveSurfer.create({
			container: node,
			waveColor: '#4F4A85',
			progressColor: '#383351',
			barWidth: 2,
			cursorWidth: 2,
			height: height,
			mediaControls: false,
			dragToSeek: true,
			backend: 'WebAudio',
			hideScrollbar: !zoom,
			plugins: plugins
		});
		wavesurfer.on('timeupdate', (t) => (time = t));
		wavesurfer.on('interaction', () => wavesurfer.play());
		wavesurfer.on('play', () => (playing = true));
		wavesurfer.on('pause', () => (playing = false));
		wavesurfer.on('decode', () => (duration = wavesurfer.getDuration()));

		let cancelled = false;

		if (regionsPlugin) {
			if (clickableRegions) {
				regionsPlugin.on('region-clicked', (region, e) => {
					e.stopPropagation();
					region.play(true);
				});
			}
			regionsPlugin['avoidOverlapping'] = () => null;
			(async () => {
				await wavesurfer.loadBlob(audio);
				if (cancelled) return;
				regions.forEach((region: RegionParams) => {
					regionsPlugin.addRegion({
						drag: false,
						resize: false,
						color: 'rgba(255,255,197,0.2)',
						...region
					});
				});
			})();
		} else {
			wavesurfer.loadBlob(audio);
		}

		return () => {
			cancelled = true;
			if (wavesurfer) {
				wavesurfer.destroy();
			}
			for (const plugin of plugins) {
				plugin.destroy();
			}
		};
	});

	function format(time: number) {
		if (isNaN(time)) return '...';

		const minutes = Math.floor(time / 60);
		const seconds = Math.floor(time % 60);

		return `${minutes}:${seconds < 10 ? `0${seconds}` : seconds}`;
	}
</script>

<div class="sample-viewer" class:compact={layout === 'compact'}>
	{#if layout !== 'compact'}
		<button onclick={() => wavesurfer.playPause()}>
			{#if playing}
				<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"
					><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg
				>
			{:else}
				<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"
					><path d="M8 5v14l11-7z" /></svg
				>
			{/if}
		</button>
	{/if}
	<div class="labels" style:--height={height + 'px'}>
		{#if layout !== 'compact'}
			<span id="time">{time !== null ? format(time) : '--:--'}</span>
		{/if}
		<div id="wavesurfer" bind:this={node}></div>
		<span id="duration">{duration ? format(duration) : '--:--'}</span>
	</div>
	{#if layout === 'compact'}
		<button onclick={() => wavesurfer.playPause()}>
			{#if playing}
				Pause
			{:else}
				Play
			{/if}
		</button>
	{/if}
</div>

<style>
	.sample-viewer {
		display: flex;
		align-items: center;
		gap: 1em;
	}

	button {
		cursor: pointer;
		background-color: var(--surface-color);
		color: var(--foreground-color);
		border: 1px solid var(--border-color);
		transition: background-color 0.2s;
	}
	button:hover {
		background-color: var(--background-color);
	}

	.sample-viewer:not(.compact) button {
		border-radius: 50%;
		width: 48px;
		height: 48px;
		display: flex;
		justify-content: center;
		align-items: center;
		padding: 0;
		flex-shrink: 0;
	}

	.sample-viewer.compact {
		flex-direction: column;
		gap: 0.5em;
		align-items: stretch;
	}

	.sample-viewer.compact button {
		border-radius: 4px;
		padding: 0.5em 0.75em;
	}

	.labels {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.5em;
		flex-grow: 1;
		height: var(--height);
	}
	.labels span {
		width: 3em;
	}
	.labels span#duration {
		text-align: right;
	}

	#wavesurfer {
		flex-grow: 1;
		/* Prevent the host flex item from resizing when the inner shadow scroll width changes */
		overflow-x: hidden;
		/* Ensure child overflow doesn't affect intrinsic inline sizing */
		contain: inline-size;
	}

	#wavesurfer :global ::part(region) {
		border: 1px solid var(--foreground-color);
		/* overflow: hidden;
		text-overflow: clip; */
	}
	#wavesurfer :global ::part(region-content) {
		padding-left: 0.2em !important;
	}
</style>
