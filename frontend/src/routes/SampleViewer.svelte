<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom.js';
	import RegionsPlugin, { type RegionParams } from 'wavesurfer.js/dist/plugins/regions.js';

	let {
		audio,
		regions = [],
		clickableRegions = true,
		clickToPlay = true,
		zoom = true,
		layout = 'default',
		compareWith = null,
		currentTime = $bindable(0)
	} = $props();

	let node: HTMLElement;

	/* wavesurfer + state */
	let wavesurfer: WaveSurfer;
	let playing = $state(false);
	let duration = $state(0);
	let dragStart: number | undefined = $state();

	/* For comparing audio */
	let shiftDown = $state(false);
	let playOther = $state(false);

	let height = $derived(layout === 'compact' ? 70 : 128);

	// Track Shift key globally so drag handlers can read the state reliably
	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Shift') shiftDown = true;
	}
	function handleKeyup(e: KeyboardEvent) {
		if (e.key === 'Shift') shiftDown = false;
	}
	function handleWindowBlur() {
		// Reset when the window loses focus to avoid sticky state
		shiftDown = false;
	}

	function mapTime(t: number): number {
		if (!compareWith.alignedTimes || compareWith.alignedTimes.length === 0) return 0;
		const times = compareWith.alignedTimes;

		// Binary search for the interval
		let low = 0;
		let high = times.length - 1;

		if (t <= times[0][0]) return times[0][1];
		if (t >= times[high][0]) return times[high][1];

		while (low <= high) {
			const mid = Math.floor((low + high) / 2);
			if (times[mid][0] < t) {
				low = mid + 1;
			} else {
				high = mid - 1;
			}
		}

		// low is now the index of the first element > t
		// so the interval is [low-1, low]
		const idx = low;
		const t0 = times[idx - 1];
		const t1 = times[idx];

		const ratio = (t - t0[0]) / (t1[0] - t0[0]);
		return t0[1] + ratio * (t1[1] - t0[1]);
	}

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
			backend: 'WebAudio',
			hideScrollbar: !zoom,
			plugins: plugins
		});
		wavesurfer.on('timeupdate', (t) => (currentTime = t));
		wavesurfer.on('play', () => (playing = true));
		wavesurfer.on('pause', () => (playing = false));
		wavesurfer.on('decode', () => (duration = wavesurfer.getDuration()));
		(wavesurfer as any).renderer.initDrag(); // hack to enable drag events without dragToSeek set
		wavesurfer.on('dragstart', (relativeX) => {
			wavesurfer.stop();
			dragStart = relativeX * duration;
			playOther = shiftDown;
		});
		if (clickToPlay) {
			wavesurfer.on('interaction', () => {
				if (!dragStart) wavesurfer.play();
			});
		}
		wavesurfer.on('dragend', (relativeX) => {
			if (!dragStart) return;
			const dragEnd = relativeX * duration;
			if (dragStart > dragEnd) {
				// probably just meant to seek
				dragStart = undefined;
				playOther = false;
				if (clickToPlay) wavesurfer.play();
				return;
			}
			if (playOther) {
				if (compareWith && compareWith.alignedTimes) {
					const otherStartTime = mapTime(dragStart);
					const otherEndTime = mapTime(dragEnd);
					compareWith.other.play(otherStartTime, otherEndTime);
				} else {
					console.warn('No compareWith data to play other audio.');
				}
			} else {
				wavesurfer.play(dragStart, dragEnd);
			}
			dragStart = undefined;
			playOther = false;
		});

		let cancelled = false;

		if (regionsPlugin) {
			if (clickableRegions) {
				regionsPlugin.on('region-clicked', (region, e) => {
					e.stopPropagation();
					if (e.shiftKey) {
						if (compareWith && compareWith.alignedTimes) {
							const otherStartTime = mapTime(region.start);
							const otherEndTime = mapTime(region.end);
							compareWith.other.play(otherStartTime, otherEndTime);
						}
					} else {
						region.play(true);
					}
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

	export function play(start?: number, end?: number) {
		if (wavesurfer) {
			if (start && start >= duration) return;
			wavesurfer.play(start, end);
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} on:keyup={handleKeyup} on:blur={handleWindowBlur} />

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
			<span id="time">{format(currentTime)}</span>
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
		contain: paint;
		position: relative;
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
