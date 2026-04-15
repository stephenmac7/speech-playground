<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import { onMount } from 'svelte';
	import { visibleRegions } from '$lib/regions';
	import type { Region, Tier } from '$lib/regions';
	import { postJson } from '$lib/api';
	import { reportError } from '$lib/errors';

	const ALIGNMENT_TIER_PREFIX = 'alignment:';
	const ALIGNMENT_COLOR = 'rgba(170, 220, 180, 0.6)';
	const PHONOLOGICAL_TIER_PREFIX = 'phonological:';

	type CompareWith = {
		other: { play: (start?: number, end?: number) => void; seek: (time?: number) => void };
		alignedTimes?: number[][];
	};

	let {
		audio,
		tiers = [],
		transcript,
		clickToPlay = false,
		zoom = true,
		layout = 'default',
		compareWith = null,
		currentTime = $bindable(0)
	}: {
		audio?: Blob;
		tiers?: Tier[];
		transcript?: string;
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
	let scrollLeft = $state(0);
	let scrollRafPending = false;

	const OVERSCAN_SEC = 0.5;
	let viewStart = $derived(pxPerSec ? scrollLeft / pxPerSec : 0);
	let viewEnd = $derived(
		pxPerSec ? (scrollLeft + containerWidth) / pxPerSec : duration
	);

	function handleScroll() {
		if (scrollRafPending || !waveformContainer) return;
		scrollRafPending = true;
		requestAnimationFrame(() => {
			scrollRafPending = false;
			if (waveformContainer) scrollLeft = waveformContainer.scrollLeft;
		});
	}

	// Track Shift key globally so drag handlers can read the state reliably
	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Shift') shiftDown = true;
		if (e.key === 'Escape' && tierMenuOpen) closeTierMenu();
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
		if (!compareWith || !compareWith.alignedTimes || compareWith.alignedTimes.length === 0)
			return 0;
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

	/* Tier visibility */
	let hiddenTierNames = $state(new Set<string>());
	let tierMenuOpen = $state(false);

	/* Forced alignment */
	let alignmentEnabled = $state(false);
	let alignmentTiers = $state<Tier[]>([]);
	let alignmentLoading = $state(false);

	type AlignmentResponse = { phones?: Region[]; words?: Region[] };

	$effect(() => {
		if (!transcript) alignmentEnabled = false;
	});

	$effect(() => {
		if (!alignmentEnabled) {
			alignmentTiers = [];
			alignmentLoading = false;
			return;
		}
		if (!audio || !transcript) return;

		const controller = new AbortController();
		let aborted = false;

		(async () => {
			alignmentLoading = true;
			alignmentTiers = [];
			const formData = new FormData();
			formData.append('audio', audio, 'recording.wav');
			formData.append('transcript', transcript);
			try {
				const result = await postJson<AlignmentResponse>(
					'/api/align',
					formData,
					controller.signal
				);
				if (aborted) return;
				const tiersOut: Tier[] = [];
				for (const name of ['phones', 'words'] as const) {
					const regions = result[name];
					if (Array.isArray(regions)) {
						tiersOut.push({
							name: `${ALIGNMENT_TIER_PREFIX}${name}`,
							regions: regions.map((r) => ({ ...r, color: ALIGNMENT_COLOR }))
						});
					}
				}
				alignmentTiers = tiersOut;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError') {
					reportError('Error fetching forced alignment.', e);
					if (!aborted) {
						alignmentTiers = [];
						alignmentEnabled = false;
					}
				}
			} finally {
				if (!aborted) alignmentLoading = false;
			}
		})();

		return () => {
			aborted = true;
			controller.abort();
		};
	});

	/* Phonological tiers — master toggle */
	let phonologicalEnabled = $state(true);

	type TierGroup = {
		prefix: string;
		masterLabel: string;
		alwaysShowMaster?: boolean;
		enabled: boolean;
		toggle: (v: boolean) => void;
		disabled?: boolean;
		title?: string;
		suffix?: string;
	};

	const tierGroups: TierGroup[] = $derived([
		{
			prefix: ALIGNMENT_TIER_PREFIX,
			masterLabel: 'Forced alignment',
			alwaysShowMaster: true,
			enabled: alignmentEnabled,
			toggle: (v: boolean) => (alignmentEnabled = v),
			disabled: !transcript,
			title: !transcript ? 'Track has no transcript' : undefined,
			suffix: alignmentLoading ? ' (loading…)' : ''
		},
		{
			prefix: PHONOLOGICAL_TIER_PREFIX,
			masterLabel: 'Phonological vectors',
			enabled: phonologicalEnabled,
			toggle: (v: boolean) => (phonologicalEnabled = v)
		}
	]);

	function groupForTier(name: string | undefined): TierGroup | undefined {
		if (!name) return undefined;
		return tierGroups.find((g) => name.startsWith(g.prefix));
	}

	function stripTierPrefix(name: string): string {
		const g = groupForTier(name);
		return g ? name.slice(g.prefix.length) : name;
	}

	let allTiers = $derived([...tiers, ...alignmentTiers]);
	let tiersWithRegions = $derived(allTiers.filter((t) => t.regions.length > 0));
	let textgridTiersWithRegions = $derived(
		tiersWithRegions.filter((t) => !groupForTier(t.name))
	);
	function groupTiers(prefix: string): Tier[] {
		return tiersWithRegions.filter((t) => t.name?.startsWith(prefix));
	}
	let canvasRefs: HTMLCanvasElement[] = $state([]);

	let visibleTiers = $derived(
		tiersWithRegions.filter((t) => {
			if (hiddenTierNames.has(t.name!)) return false;
			const g = groupForTier(t.name);
			if (g && !g.enabled) return false;
			return true;
		})
	);

	function toggleTier(label: string) {
		const next = new Set(hiddenTierNames);
		if (next.has(label)) next.delete(label);
		else next.add(label);
		hiddenTierNames = next;
	}

	function closeTierMenu() {
		tierMenuOpen = false;
	}

	let activeTierRegions: Region[] = [];

	function getEffectiveRegion(time: number): { start: number; end: number } {
		if (!activeTierRegions || activeTierRegions.length === 0) {
			return { start: 0, end: duration };
		}

		const sortedRegions = [...activeTierRegions].sort((a, b) => a.start - b.start);

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
				console.log('Playing other viewer from', otherStart, 'to', otherEnd);
				compareWith.other.play(otherStart, otherEnd);
			}
		} else {
			wavesurfer.play(start, end);
		}
	}

	function handleRegionBarMouseDown(e: MouseEvent, tierRegions: Region[]) {
		if (e.button !== 0) return;
		e.preventDefault();

		if (!wavesurfer) return;
		if (!waveformContainer) return;
		activeTierRegions = tierRegions;
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
		playButton?.focus({ preventScroll: true });
	}

	onMount(() => {
		if (!node) return;
		wavesurfer = WaveSurfer.create({
			container: node,
			waveColor: '#4F4A85',
			progressColor: '#383351',
			barWidth: 2,
			cursorWidth: 0,
			height: height,
			mediaControls: false,
			backend: 'WebAudio',
			hideScrollbar: true,
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
		wavesurfer.on('pause', () => {
			playing = false;
			currentTime = wavesurfer.getCurrentTime();
		});
		// Enable drag events without dragToSeek (which adds its own conflicting handler).
		// Wrap setOptions so the drag stream survives option updates (7.12+ cleans it up).
		// @ts-expect-error private WaveSurfer renderer API
		wavesurfer.renderer.initDrag();
		const origSetOptions = wavesurfer.setOptions.bind(wavesurfer);
		wavesurfer.setOptions = (...args: Parameters<typeof origSetOptions>) => {
			origSetOptions(...args);
			// @ts-expect-error private WaveSurfer renderer API
			wavesurfer.renderer.initDrag();
		};
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
		wavesurfer.on('seeking', (t) => {
			if (typeof t === 'number') currentTime = t;
			playButton?.focus({ preventScroll: true });
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
		void viewStart;
		void viewEnd;
		void pxPerSec;
		void containerWidth;
		void duration;
		const tiers = visibleTiers;
		if (!duration || !pxPerSec || !containerWidth) return;
		const dpr = window.devicePixelRatio || 1;
		const wCss = containerWidth;
		const hCss = 24;
		const t0 = viewStart - OVERSCAN_SEC;
		const t1 = viewEnd + OVERSCAN_SEC;
		for (let i = 0; i < tiers.length; i++) {
			const canvas = canvasRefs[i];
			if (!canvas) continue;
			const bw = Math.max(1, Math.round(wCss * dpr));
			const bh = Math.max(1, Math.round(hCss * dpr));
			if (canvas.width !== bw || canvas.height !== bh) {
				canvas.width = bw;
				canvas.height = bh;
			}
			const ctx = canvas.getContext('2d');
			if (!ctx) continue;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, wCss, hCss);
			const regions = visibleRegions(tiers[i], t0, t1);
			ctx.font = '12px system-ui, sans-serif';
			ctx.textBaseline = 'middle';
			for (const r of regions) {
				const x = (r.start - viewStart) * pxPerSec;
				const w = (r.end - r.start) * pxPerSec;
				let y = 0;
				let h = hCss;
				if (r.lane === 'top') {
					y = 0;
					h = hCss / 2;
				} else if (r.lane === 'bottom') {
					y = hCss / 2;
					h = hCss / 2;
				}
				ctx.fillStyle = r.color ?? 'rgba(255, 255, 197, 0.5)';
				ctx.fillRect(x, y, w, h);
				ctx.strokeStyle = '#e5e7eb';
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(x + 0.5, y);
				ctx.lineTo(x + 0.5, y + h);
				ctx.moveTo(x + w - 0.5, y);
				ctx.lineTo(x + w - 0.5, y + h);
				ctx.stroke();
				if (r.content && w > 14) {
					ctx.save();
					ctx.beginPath();
					ctx.rect(x + 2, y, w - 4, h);
					ctx.clip();
					ctx.fillStyle = '#111';
					ctx.fillText(r.content, x + 4, y + h / 2);
					ctx.restore();
				}
			}
		}
	});

	$effect(() => {
		if (wavesurfer && audio) {
			duration = 0;
			wavesurfer.setOptions({ minPxPerSec: 0 });
			isFitToView = true;
			pxPerSec = 0;
			accumulatedDelta = 0;
			hiddenTierNames = new Set();
			tierMenuOpen = false;
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

<div class="sample-viewer" class:compact={layout === 'compact'} style:--waveform-height="{height}px">
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
			onscroll={handleScroll}
		>
			<div
				class="scroll-wrapper"
				bind:this={scrollWrapper}
				style:width={!isFitToView && zoom && duration && pxPerSec
					? Math.ceil(duration * pxPerSec) + 'px'
					: '100%'}
			>
				<div id="wavesurfer" bind:this={node}></div>
				{#each visibleTiers as tier, i (tier)}
					<div class="regions-bar" style:width="{duration * pxPerSec}px">
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<canvas
							bind:this={canvasRefs[i]}
							class="regions-canvas"
							style:width="{containerWidth}px"
							style:height="24px"
							onmousedown={(e) => handleRegionBarMouseDown(e, tier.regions)}
						></canvas>
					</div>
				{/each}
				{#if duration}
					<div class="playhead" style:left="{currentTime * pxPerSec}px"></div>
				{/if}
			</div>
		</div>
		{#if tiersWithRegions.length > 0 || transcript !== undefined}
			<div class="tier-labels">
				{#each visibleTiers as tier, i (tier)}
					<div class="tier-label-row">
						{#if i === 0}
							<button
								type="button"
								class="tier-menu-toggle"
								aria-label="Toggle tiers"
								aria-expanded={tierMenuOpen}
								onclick={() => (tierMenuOpen = !tierMenuOpen)}
							>
								<svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor">
									<path d="M4 6l4 4 4-4z" />
								</svg>
					{#if tierMenuOpen}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div
						class="tier-menu-backdrop"
						onclick={(e) => {
							e.stopPropagation();
							closeTierMenu();
						}}
					></div>
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<div class="tier-menu" onclick={(e) => e.stopPropagation()}>
							{#each textgridTiersWithRegions as tier}
								{@const isVisible = !hiddenTierNames.has(tier.name!)}
								{@const isLastVisible = isVisible && visibleTiers.length === 1}
								<label class="tier-menu-item" class:disabled={isLastVisible}>
									<input
										type="checkbox"
										checked={isVisible}
										disabled={isLastVisible}
										onchange={() => toggleTier(tier.name!)}
									/>
									{tier.name}
								</label>
							{/each}
							{#each tierGroups as group}
								{@const members = groupTiers(group.prefix)}
								{#if group.alwaysShowMaster || members.length > 0}
									<div class="tier-menu-separator"></div>
									<label
										class="tier-menu-item"
										class:disabled={group.disabled}
										title={group.title}
									>
										<input
											type="checkbox"
											checked={group.enabled}
											disabled={group.disabled}
											onchange={(e) =>
												group.toggle((e.currentTarget as HTMLInputElement).checked)}
										/>
										{group.masterLabel}{group.suffix ?? ''}
									</label>
									{#if group.enabled}
										{#each members as tier}
											{@const isVisible = !hiddenTierNames.has(tier.name!)}
											{@const isLastVisible = isVisible && visibleTiers.length === 1}
											<label class="tier-menu-item sub" class:disabled={isLastVisible}>
												<input
													type="checkbox"
													checked={isVisible}
													disabled={isLastVisible}
													onchange={() => toggleTier(tier.name!)}
												/>
												{stripTierPrefix(tier.name!)}
											</label>
										{/each}
									{/if}
								{/if}
							{/each}
						</div>
					{/if}
							</button>
						{/if}
						<span class="tier-label-name" title={tier.name}
							>{stripTierPrefix(tier.name ?? '')}</span
						>
					</div>
				{/each}
			</div>
		{/if}
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
		align-items: start;
		gap: 1em;
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
		/* Center vertically against the waveform, not the full column including tiers */
		margin-top: calc((var(--waveform-height) - 48px) / 2);
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
		align-items: start;
		gap: 0.5em;
		flex-grow: 1;
		position: relative;
	}
	.labels > span {
		width: 3em;
		/* Center vertically against the waveform height */
		height: var(--waveform-height);
		display: flex;
		align-items: center;
	}
	.labels > span#duration {
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
		scrollbar-width: none;
	}
	.waveform-container::-webkit-scrollbar {
		display: none;
	}

	.scroll-wrapper {
		position: relative;
		transform: rotateX(180deg);
	}

	.regions-bar {
		position: relative;
		height: 24px;
		background: #f3f4f6;
		border-top: 1px solid #e5e7eb;
	}

	.regions-canvas {
		position: sticky;
		left: 0;
		display: block;
	}

	.playhead {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 1px;
		background: var(--playhead-color, #000);
		pointer-events: none;
		z-index: 5;
	}

	.tier-labels {
		/* Stack of per-tier labels sitting in the left gutter, aligned with the
		   rows of .regions-bar on the right. The first row also hosts the menu
		   toggle (dropdown arrow) to the left of the label name. */
		position: absolute;
		left: 0;
		top: var(--waveform-height);
		width: calc(3em + 0.5em);
		display: flex;
		flex-direction: column;
		z-index: 10;
	}
	.tier-label-row {
		height: 24px;
		/* mirror .regions-bar's 1px top border so rows stay aligned */
		border-top: 1px solid transparent;
		box-sizing: content-box;
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: 4px;
		padding-right: 6px;
	}
	.tier-label-name {
		font-size: 11px;
		line-height: 1;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		color: var(--foreground-color);
		opacity: 0.8;
	}

	.sample-viewer:not(.compact) .tier-menu-toggle,
	.sample-viewer:not(.compact) .tier-menu-toggle:hover,
	.sample-viewer.compact .tier-menu-toggle,
	.sample-viewer.compact .tier-menu-toggle:hover {
		cursor: pointer;
		line-height: 0;
		position: relative;
		background: none;
		border: none;
		border-radius: 0;
		padding: 0;
		margin: 0;
		width: auto;
		height: auto;
		color: inherit;
		display: inline-flex;
	}
	.tier-menu-toggle svg {
		opacity: 0.5;
		transition: opacity 0.15s;
	}
	.tier-menu-toggle:hover svg {
		opacity: 0.9;
	}

	.tier-menu-backdrop {
		position: fixed;
		inset: 0;
		z-index: 99;
	}

	.tier-menu {
		position: absolute;
		left: 0;
		top: 100%;
		z-index: 100;
		background: var(--surface-color, #fff);
		border: 1px solid var(--border-color, #e5e7eb);
		border-radius: 6px;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
		padding: 4px 0;
		min-width: 140px;
		white-space: nowrap;
	}

	.tier-menu-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		font-size: 13px;
		cursor: pointer;
		user-select: none;
	}
	.tier-menu-item:hover {
		background: var(--background-color, #f3f4f6);
	}
	.tier-menu-item.sub {
		padding-left: 24px;
	}
	.tier-menu-separator {
		border-top: 1px solid var(--border-color, #e5e7eb);
		margin: 4px 0;
	}
	.tier-menu-item.disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}
	.tier-menu-item.disabled:hover {
		background: transparent;
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
