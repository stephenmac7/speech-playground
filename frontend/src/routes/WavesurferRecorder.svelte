<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import RecordPlugin from 'wavesurfer.js/dist/plugins/record.js';

	let wavesurfer: WaveSurfer;
	let record: RecordPlugin;

	let actualStartTime: number | undefined = $state();
	let resolveEndRecording: ((result: { blob: Blob; duration: number }) => void) | undefined;
	let innerHeight = $state(typeof window !== 'undefined' ? window.innerHeight : 1067);
	let height = $derived(Math.round(Math.min(128, Math.max(64, innerHeight * 0.12))));

	function waveform(node: HTMLDivElement) {
		wavesurfer = WaveSurfer.create({
			container: node,
			waveColor: '#4F4A85',
			progressColor: '#383351',
			barWidth: 2,
			cursorWidth: 2,
			height: height,
			dragToSeek: true
		});
		record = wavesurfer.registerPlugin(
			RecordPlugin.create({
				scrollingWaveform: true,
				renderRecordedAudio: false
			})
		);
		record.on('record-end', (blob: Blob) => {
			if (resolveEndRecording) {
				if (!actualStartTime) {
					throw new Error('Actual start time is undefined on record-end');
				}
				const duration = performance.now() - actualStartTime;
				resolveEndRecording({ blob, duration });
				resolveEndRecording = undefined;
			}
			actualStartTime = undefined;
		});

		return {
			destroy() {
				if (wavesurfer) {
					record.destroy();
					wavesurfer.destroy();
				}
			}
		};
	}

	$effect(() => {
		if (wavesurfer) {
			wavesurfer.setOptions({ height });
		}
	});

	export function startRecording(): Promise<void> {
		if (!record) {
			return Promise.reject('Recorder not initialized');
		}
		if (record.isRecording()) {
			return Promise.reject('Recorder is already recording');
		}

		return new Promise((resolve, reject) => {
			const updateRecordingState = () => {
				// unfortunate hack
				// @ts-expect-error private record plugin API
				const data = record.dataWindow as ArrayLike<number> | undefined;
				if (data && Array.from(data).some((v) => v !== 0)) {
					actualStartTime = performance.now();
					wavesurfer.un('redraw', updateRecordingState);
					resolve(); // Resolve the promise now that data is flowing
				}
			};

			wavesurfer.on('redraw', updateRecordingState);

			record.startRecording().catch((err: unknown) => {
				wavesurfer.un('redraw', updateRecordingState);
				reject(err);
			});
		});
	}

	export function stopRecording(): Promise<{ blob: Blob; duration: number }> {
		if (!record) {
			return Promise.reject('Recorder not initialized');
		}
		if (!record.isRecording()) {
			return Promise.reject('Recorder is not recording');
		}
		return new Promise((resolve) => {
			resolveEndRecording = resolve;
			record.stopRecording();
		});
	}

	export function cancelRecording(): void {
		if (record && record.isRecording()) {
			resolveEndRecording = undefined;
			record.stopRecording();
		}
	}
</script>

<svelte:window bind:innerHeight />

<div class="recorder-container">
	<div use:waveform class="recorder-inner" class:hide={!actualStartTime} style:height="{height}px"></div>
</div>

<style>
	.recorder-container {
		margin-top: -0.5rem;
		margin-bottom: 0.5rem;
		border: 1px dashed var(--border-color);
		border-radius: 4px;
	}
	.recorder-inner {
		opacity: 1;
		overflow: hidden;

		/* Base (visible) state */
		background-color: transparent;
		box-shadow: none;

		/* Define the transitions */
		transition:
			opacity 300ms ease,
			/* These create the highlight fade-out */ background-color 1s ease-out,
			box-shadow 1s ease-out;
	}

	.hide {
		opacity: 0;
		pointer-events: none;

		background-color: rgba(255, 150, 0, 0.5); /* Highlight color */
		box-shadow: 0 0 12px 3px rgba(255, 150, 0, 0.5); /* Glow */

		transition-duration: 0s;
	}
</style>
