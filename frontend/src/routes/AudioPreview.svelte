<script lang="ts">
	let { audio }: { audio?: Blob | null } = $props();

	let audioEl: HTMLAudioElement | undefined = $state();
	let playing = $state(false);
	let duration = $state(0);
	let currentTime = $state(0);
	let audioUrl = $state<string | undefined>();
	let animFrame = 0;

	let prevBlob: Blob | null | undefined;
	$effect.pre(() => {
		if (audio === prevBlob) return;
		prevBlob = audio;
		const old = audioUrl;
		audioUrl = audio ? URL.createObjectURL(audio) : undefined;
		playing = false;
		duration = 0;
		currentTime = 0;
		if (old) URL.revokeObjectURL(old);
	});

	function tick() {
		if (audioEl && playing) {
			currentTime = audioEl.currentTime;
			animFrame = requestAnimationFrame(tick);
		}
	}

	function toggle() {
		if (!audioEl) return;
		if (playing) {
			audioEl.pause();
			audioEl.currentTime = 0;
			currentTime = 0;
		} else {
			audioEl.play();
		}
	}

	function onPlay() {
		playing = true;
		animFrame = requestAnimationFrame(tick);
	}

	function onPause() {
		playing = false;
		cancelAnimationFrame(animFrame);
	}

	function onEnded() {
		playing = false;
		cancelAnimationFrame(animFrame);
		currentTime = 0;
		if (audioEl) audioEl.currentTime = 0;
	}

	const RADIUS = 17;
	const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
	let progress = $derived(duration > 0 ? currentTime / duration : 0);
	let dashOffset = $derived(CIRCUMFERENCE * (1 - progress));

	function formatTime(t: number): string {
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60).toString().padStart(2, '0');
		return `${m}:${s}`;
	}
</script>

{#if audioUrl}
	<audio
		bind:this={audioEl}
		src={audioUrl}
		onloadedmetadata={() => { if (audioEl) duration = audioEl.duration; }}
		onplay={onPlay}
		onpause={onPause}
		onended={onEnded}
	></audio>
{/if}

<div class="audio-preview">
	<button class="play-button" onclick={toggle} disabled={!audioUrl} title={playing ? 'Stop' : 'Play'}>
		<svg class="progress-ring" viewBox="0 0 40 40">
			<circle class="progress-track" cx="20" cy="20" r={RADIUS} />
			{#if playing || progress > 0}
				<circle
					class="progress-fill"
					cx="20" cy="20" r={RADIUS}
					stroke-dasharray={CIRCUMFERENCE}
					stroke-dashoffset={dashOffset}
				/>
			{/if}
		</svg>
		<span class="icon">
			{#if playing}
				<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
					<rect x="6" y="5" width="12" height="14" />
				</svg>
			{:else}
				<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
					<path d="M8 5v14l11-7z" />
				</svg>
			{/if}
		</span>
	</button>
	{#if duration > 0}
		<span class="duration">{formatTime(duration)}</span>
	{/if}
</div>

<style>
	.audio-preview {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.2em;
	}
	.play-button {
		position: relative;
		width: 2.4em;
		height: 2.4em;
		border-radius: 50%;
		border: 1px solid var(--border-color);
		background-color: var(--surface-color);
		color: var(--foreground-color);
		cursor: pointer;
		padding: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		transition: background-color 0.2s;
	}
	.play-button:hover {
		background-color: var(--background-color);
	}
	.play-button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.progress-ring {
		position: absolute;
		inset: -3px;
		width: calc(100% + 6px);
		height: calc(100% + 6px);
		transform: rotate(-90deg);
		pointer-events: none;
	}
	.progress-track {
		fill: none;
		stroke: var(--border-color);
		stroke-width: 2;
	}
	.progress-fill {
		fill: none;
		stroke: var(--primary-color);
		stroke-width: 2.5;
		stroke-linecap: round;
		transition: stroke-dashoffset 0.1s linear;
	}
	.icon {
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.duration {
		font-size: 0.8rem;
		color: var(--secondary-color);
		font-variant-numeric: tabular-nums;
	}
</style>
