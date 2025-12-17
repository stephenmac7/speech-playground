<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import { onMount } from 'svelte';

	type Region = {
		start: number;
		end: number;
		color?: string;
		content?: string;
	};

	type CompareWith = {
		other: { play: (start?: number, end?: number) => void, seek: (time?: number) => void };
		alignedTimes?: number[][];
	};

	let {
		audio,
		regions = [],
		clickToPlay = false,
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
	let playButton: HTMLButtonElement | undefined = $state();

	/* For comparing audio */
	let shiftDown = $state(false);
	let playOther = $state(false);

	let height = $derived(layout === 'compact' ? 70 : 128);

	let pxPerSec = $state(0);
	let isFitToView = $state(true);
	let accumulatedDelta = 0;
	let containerWidth = $state(0);

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
		isFitToView = Math.abs(pxPerSec - minPxPerSec) < 0.001;

		wavesurfer.zoom(pxPerSec);
		waveformContainer.scrollLeft = pointerTime * pxPerSec - x;
	}

	function mapTime(t?: number): number | undefined {
		if (t === undefined) return undefined;
		if (!compareWith || !compareWith.alignedTimes || compareWith.alignedTimes.length === 0) return 0;
		const times = compareWith.alignedTimes;

		let low = 0;
		let high = times.length - 1;

		while (low <= high) {
			const mid = Math.floor((low + high) / 2);
			if (times[mid][0] <= t) {
				low = mid + 1;
			} else {
				high = mid - 1;
			}
		}
		const idx = low - 1;

		if (idx < 0) return times[0][1];
		if (idx >= times.length - 1) return times[times.length - 1][1];

		const t0 = times[idx];
		const t1 = times[idx + 1];

		if (t1[0] === t0[0]) return t1[1];

		const ratio = (t - t0[0]) / (t1[0] - t0[0]);
		return t0[1] + ratio * (t1[1] - t0[1]);
	}

	function getEffectiveRegion(time: number): { start: number; end: number } {
		if (!regions || regions.length === 0) {
			return { start: 0, end: duration };
		}

		const sortedRegions = [...regions].sort((a, b) => a.start - b.start);

		for (const region of sortedRegions) {
			if (time >= region.start && time <= region.end) {
				return { start: region.start, end: region.end };
			}
		}

		if (time < sortedRegions[0].start) {
			return { start: 0, end: sortedRegions[0].start };
		}

		for (let i = 0; i < sortedRegions.length - 1; i++) {
			const current = sortedRegions[i];
			const next = sortedRegions[i + 1];
			if (time > current.end && time < next.start) {
				return { start: current.end, end: next.start };
			}
		}

		const last = sortedRegions[sortedRegions.length - 1];
		if (time > last.end) {
			return { start: last.end, end: duration };
		}

		return { start: 0, end: duration };
	}

	function playSelection(start?: number, end?: number, playOnOtherViewer: boolean = false) {
		if (playOnOtherViewer && compareWith) {
			const otherStart = mapTime(start);
			const otherEnd = mapTime(end);
			if (otherStart !== undefined && otherEnd !== undefined) {
				// wavesurfer.play handles start > end by playing from min(start, end) to max(start, end)
				compareWith.other.play(otherStart, otherEnd);
			}
		} else {
			wavesurfer.play(start, end);
		}
	}

	function handleRegionBarMouseDown(e: MouseEvent) {
		if (e.button !== 0) return;
		e.preventDefault();

		if (!wavesurfer) return;
		if (!waveformContainer) return;
		const rect = waveformContainer.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const time = (waveformContainer.scrollLeft + x) / pxPerSec;

		const region = getEffectiveRegion(time);
		dragStart = region.start;
		playOther = shiftDown;

		wavesurfer.pause();

		window.addEventListener('mousemove', handleRegionBarMouseMove);
		window.addEventListener('mouseup', handleRegionBarMouseUp);
	}

	function handleRegionBarMouseMove(e: MouseEvent) {
		if (dragStart === undefined || !waveformContainer) return;

		const rect = waveformContainer.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const minGap = 30;

		if (x > rect.width - minGap) {
			waveformContainer.scrollLeft += minGap;
		} else if (x < minGap) {
			waveformContainer.scrollLeft -= minGap;
		}
	}

	function handleRegionBarMouseUp(e: MouseEvent) {
		window.removeEventListener('mousemove', handleRegionBarMouseMove);
		window.removeEventListener('mouseup', handleRegionBarMouseUp);

		if (dragStart === undefined || !waveformContainer || !wavesurfer) return;

		const rect = waveformContainer.getBoundingClientRect();
		const x = e.clientX - rect.left;
		let dragEndTime = (waveformContainer.scrollLeft + x) / pxPerSec;

		dragEndTime = Math.max(0, Math.min(dragEndTime, duration));

		const region = getEffectiveRegion(dragEndTime);
		
		let start: number, end: number;

		if (dragEndTime < dragStart) {
			start = region.start;
			end = region.end;
		} else {
			start = dragStart;
			end = region.end;
		}

		playSelection(start, end, playOther);

		dragStart = undefined;
		playOther = false;
	}

	onMount(() => {
		if (!node) return;
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
			if (duration > 0 && wavesurfer.options.minPxPerSec === 0 && containerWidth > 0) {
				pxPerSec = containerWidth / duration;
			}
		});

		wavesurfer.on('timeupdate', (t) => {
			currentTime = t;
			if (playing && zoom && waveformContainer) {
				const cursorX = t * pxPerSec;
				const { scrollLeft, clientWidth } = waveformContainer;

				if (cursorX < scrollLeft) {
					waveformContainer.scrollLeft = cursorX;
				} else if (cursorX > scrollLeft + clientWidth) {
					waveformContainer.scrollLeft += clientWidth / 2;
				}
			}
		});
		wavesurfer.on('play', () => (playing = true));
		wavesurfer.on('pause', () => (playing = false));
		(wavesurfer as any).renderer.initDrag(); // hack to enable drag events without dragToSeek set
		wavesurfer.on('drag', (relativeX) => {
			if (zoom && waveformContainer) {
				const cursorX = relativeX * duration * pxPerSec;
				const { scrollLeft, clientWidth } = waveformContainer;
				const minGap = 30;

				if (cursorX + minGap > scrollLeft + clientWidth) {
					waveformContainer.scrollLeft += minGap;
				} else if (cursorX - minGap < scrollLeft) {
					waveformContainer.scrollLeft -= minGap;
				}
			}
		});
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
		wavesurfer.on('seeking', (currentTime) => {
			playButton?.focus({focusVisible: false} as any);
		});
		wavesurfer.on('click', (relativeX) => {
			if (shiftDown && compareWith) {
				const otherTime = mapTime(relativeX * duration);
				compareWith.other.seek(otherTime);
			}
		});
		wavesurfer.on('dragend', (relativeX) => {
			if (!dragStart) return;
			const dragEnd = relativeX * duration;
			if (dragStart >= dragEnd) {
				// probably just meant to seek
				dragStart = undefined;
				playOther = false;
				if (clickToPlay) wavesurfer.play();
				return;
			}
			playSelection(dragStart, dragEnd, playOther);
			dragStart = undefined;
			playOther = false;
		});

		if (audio) {
			wavesurfer.loadBlob(audio);
		}

		return () => {
			if (wavesurfer) {
				wavesurfer.destroy();
			}
			window.removeEventListener('mousemove', handleRegionBarMouseMove);
			window.removeEventListener('mouseup', handleRegionBarMouseUp);
		};
	});

	$effect(() => {
		if (isFitToView && duration > 0 && containerWidth > 0) {
			pxPerSec = containerWidth / duration;
		}
	});

	$effect(() => {
		if (wavesurfer && audio) {
			duration = 0;
			wavesurfer.setOptions({ minPxPerSec: 0 });
			isFitToView = true;
			pxPerSec = 0;
			accumulatedDelta = 0;
			wavesurfer.loadBlob(audio);
		}
	});

	$effect(() => {
		if (wavesurfer) {
			wavesurfer.setOptions({ height });
		}
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

	export function seek(time?: number) {
		if (time === undefined || !wavesurfer || time >= duration) return;
		wavesurfer.seekTo(time / duration);
	}
</script>

<svelte:window on:keydown={handleKeydown} on:keyup={handleKeyup} on:blur={handleWindowBlur} />

<div class="sample-viewer" class:compact={layout === 'compact'}>
	{#if layout !== 'compact'}
		<button bind:this={playButton} onclick={() => wavesurfer.playPause()}>
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
		<div
			class="waveform-container"
			bind:this={waveformContainer}
			bind:clientWidth={containerWidth}
			onwheel={handleWheel}
		>
			<div
				class="scroll-wrapper"
				bind:this={scrollWrapper}
				style:width={!isFitToView && zoom && duration && pxPerSec
					? Math.ceil(duration * pxPerSec) + 'px'
					: '100%'}
			>
				<div id="wavesurfer" bind:this={node}></div>
				{#if regions.length > 0}
					<div class="regions-bar">
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div
							class="regions-timeline"
							style:width="{duration * pxPerSec}px"
							onmousedown={handleRegionBarMouseDown}
						>
							{#if duration}
								{#each regions as region}
									<!-- svelte-ignore a11y_click_events_have_key_events -->
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										class="region-under"
										style:left="{region.start * pxPerSec}px"
										style:width="{(region.end - region.start) * pxPerSec}px"
										style:background-color={region.color ?? 'rgba(255, 255, 197, 0.5)'}
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
		<button bind:this={playButton} onclick={() => wavesurfer.playPause()}>
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
