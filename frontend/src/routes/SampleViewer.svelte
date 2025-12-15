<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';

	type Region = {
		start: number;
		end: number;
		color?: string;
		content?: string;
	};

	type CompareWith = {
		other: { play: (start?: number, end?: number) => void };
		alignedTimes?: number[][];
	};

	let {
		audio,
		regions = [],
		clickToPlay = true,
		zoom = true,
		layout = 'default',
		compareWith = null,
		currentTime = $bindable(0)
	}: {
		audio?: Blob;
		regions?: Region[];
		clickToPlay?: boolean;
		zoom?: boolean;
		layout?: 'default' | 'compact';
		compareWith?: CompareWith | null;
		currentTime?: number;
	} = $props();

	let node: HTMLElement;
	let waveformContainer: HTMLElement;
	let scrollWrapper: HTMLElement;

	/* wavesurfer + state */
	let wavesurfer: WaveSurfer;
	let playing = $state(false);
	let duration = $state(0);
	let dragStart: number | undefined = $state();

	/* For comparing audio */
	let shiftDown = $state(false);
	let playOther = $state(false);

	let height = $derived(layout === 'compact' ? 70 : 128);

	let pxPerSec = $state(0);
	let accumulatedDelta = 0;

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

	async function handleWheel(e: WheelEvent) {
		if (!zoom || !duration || !waveformContainer) return;

		// Allow horizontal scrolling if deltaX dominates
		if (Math.abs(e.deltaX) >= Math.abs(e.deltaY)) {
			return;
		}

		e.preventDefault();

		accumulatedDelta += -e.deltaY;

		const deltaThreshold = 5;
		if (Math.abs(accumulatedDelta) < deltaThreshold) return;

		const rect = waveformContainer.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const pointerTime = (waveformContainer.scrollLeft + x) / pxPerSec;

		const scale = 0.5;
		let newPxPerSec = Math.max(0, pxPerSec + accumulatedDelta * scale);

		// Limit min zoom to fit width
		const minPxPerSec = waveformContainer.clientWidth / duration;
		if (newPxPerSec < minPxPerSec) newPxPerSec = minPxPerSec;

		// Limit max zoom to match wavesurfer default (container width)
		const maxPxPerSec = waveformContainer.clientWidth;
		if (newPxPerSec > Math.max(maxPxPerSec, minPxPerSec))
			newPxPerSec = Math.max(maxPxPerSec, minPxPerSec);

		accumulatedDelta = 0;
		pxPerSec = newPxPerSec;

		if (scrollWrapper) {
			scrollWrapper.style.width = `${Math.ceil(duration * pxPerSec)}px`;
		}
		wavesurfer.zoom(pxPerSec);
		waveformContainer.scrollLeft = pointerTime * pxPerSec - x;
	}

	function mapTime(t?: number): number | undefined {
		if (t === undefined) return undefined;
		if (!compareWith || !compareWith.alignedTimes || compareWith.alignedTimes.length === 0) return 0;
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
		wavesurfer = WaveSurfer.create({
			container: node,
			waveColor: '#4F4A85',
			progressColor: '#383351',
			barWidth: 2,
			cursorWidth: 2,
			height: height,
			mediaControls: false,
			backend: 'WebAudio',
			hideScrollbar: false,
			autoCenter: false,
			autoScroll: false
		});

		wavesurfer.on('ready', (d) => {
			duration = d;
			const wrapper = (wavesurfer as any).renderer.wrapper as HTMLElement;
			if (wrapper && duration > 0) {
				if (wavesurfer.options.minPxPerSec === 0) {
					pxPerSec = wrapper.clientWidth / duration;
				}
			}
		});

		wavesurfer.on('redraw', () => {
			if (!wavesurfer) return;
			// This runs on initial draw and subsequent redraws.
			const wrapper = (wavesurfer as any).renderer.wrapper as HTMLElement;
			if (wrapper && duration > 0) {
				if (wavesurfer.options.minPxPerSec === 0) {
					pxPerSec = wrapper.clientWidth / duration;
				}
			}
		});

		if (zoom) {
			wavesurfer.on('zoom', (newPxPerSec) => {
				pxPerSec = newPxPerSec;
			});
		}

		wavesurfer.on('timeupdate', (t) => {
			currentTime = t;
			if (playing && zoom && waveformContainer) {
				const cursorX = t * pxPerSec;
				const containerWidth = waveformContainer.clientWidth;
				const scrollLeft = waveformContainer.scrollLeft;

				// Keep cursor centered
				const targetScrollLeft = cursorX - containerWidth / 2;

				const maxScroll = waveformContainer.scrollWidth - containerWidth;
				const clampedScroll = Math.max(0, Math.min(targetScrollLeft, maxScroll));

				if (Math.abs(scrollLeft - clampedScroll) > 1) {
					waveformContainer.scrollLeft = clampedScroll;
				}
			}
		});
		wavesurfer.on('play', () => (playing = true));
		wavesurfer.on('pause', () => (playing = false));
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

		wavesurfer.loadBlob(audio);

		return () => {
			cancelled = true;
			if (wavesurfer) {
				wavesurfer.destroy();
			}
		};
	});

	function handleRegionClick(region: Region, e: MouseEvent) {
		e.stopPropagation();
		if (e.shiftKey) {
			if (compareWith && compareWith.alignedTimes) {
				const otherStartTime = mapTime(region.start);
				const otherEndTime = mapTime(region.end);
				compareWith.other.play(otherStartTime, otherEndTime);
			}
		} else {
			wavesurfer.play(region.start, region.end);
		}
	}

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
	<div class="labels">
		{#if layout !== 'compact'}
			<span id="time">{format(currentTime)}</span>
		{/if}
		<div class="waveform-container" bind:this={waveformContainer} onwheel={handleWheel}>
			<div
				class="scroll-wrapper"
				bind:this={scrollWrapper}
				style:width={zoom && duration && pxPerSec ? Math.ceil(duration * pxPerSec) + 'px' : '100%'}
			>
				<div id="wavesurfer" bind:this={node}></div>
				{#if regions.length > 0}
					<div class="regions-bar">
						<div class="regions-timeline" style:width="{duration * pxPerSec}px">
							{#if duration}
								{#each regions as region}
									<!-- svelte-ignore a11y_click_events_have_key_events -->
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										class="region-under"
										style:left="{region.start * pxPerSec}px"
										style:width="{(region.end - region.start) * pxPerSec}px"
										style:background-color={region.color ?? 'rgba(255, 255, 197, 0.5)'}
										onclick={(e) => handleRegionClick(region, e)}
									>
										{#if region.content}
											<div class="region-content-under">{region.content}</div>
										{/if}
									</div>
								{/each}
							{/if}
						</div>
					</div>
				{/if}
			</div>
		</div>
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
	}
	.labels span {
		width: 3em;
	}
	.labels span#duration {
		text-align: right;
	}

	.waveform-container {
		flex-grow: 1;
		/* Prevent the host flex item from resizing when the inner shadow scroll width changes */
		overflow-x: auto;
		/* Ensure child overflow doesn't affect intrinsic inline sizing */
		contain: inline-size;
		display: flex;
		flex-direction: column;
		transform: rotateX(180deg);
	}

	.scroll-wrapper {
		transform: rotateX(180deg);
	}

	.regions-bar {
		position: relative;
		height: 24px;
		background: #f3f4f6;
		border-top: 1px solid #e5e7eb;
		overflow: hidden;
	}

	.regions-timeline {
		position: absolute;
		top: 0;
		bottom: 0;
		left: 0;
	}

	.region-under {
		position: absolute;
		top: 0;
		height: 100%;
		background-color: rgba(255, 255, 197, 0.5);
		border-left: 1px solid #e5e7eb;
		border-right: 1px solid #e5e7eb;
		box-sizing: border-box;
		overflow: hidden;
		cursor: pointer;
	}

	.region-content-under {
		padding: 4px;
		font-size: 12px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		user-select: none;
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
