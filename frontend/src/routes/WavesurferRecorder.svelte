<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';
	import RecordPlugin from 'wavesurfer.js/dist/plugins/record.js';

	let wavesurfer: WaveSurfer;
	let record: RecordPlugin;

	let isRecording = $state(false);
	let resolveEndRecording: ((blob: Blob) => void) | undefined;

	function waveform(node: HTMLDivElement) {
		wavesurfer = WaveSurfer.create({
			container: node,
			waveColor: '#4F4A85',
			progressColor: '#383351',
			barWidth: 2,
			cursorWidth: 2,
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
				resolveEndRecording(blob);
				resolveEndRecording = undefined;
			}
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

	export function startRecording(): Promise<void> {
		if (!record) {
			return Promise.reject('Recorder not initialized');
		}
		if (record.isRecording()) {
			return Promise.reject('Recorder is already recording');
		}
		isRecording = true;
		return record.startRecording();
	}

	export function stopRecording(): Promise<Blob> {
		if (!record) {
			return Promise.reject('Recorder not initialized');
		}
		if (!record.isRecording()) {
			return Promise.reject('Recorder is not recording');
		}
		return new Promise<Blob>((resolve) => {
			resolveEndRecording = resolve;
			record.stopRecording();
			isRecording = false;
		});
	}

	export function cancelRecording(): void {
		if (record && record.isRecording()) {
			resolveEndRecording = undefined;
			record.stopRecording();
			isRecording = false;
		}
	}
</script>

<div use:waveform class:hide={!isRecording}></div>

<style>
	div {
		opacity: 1;
		height: 128px; /* Give the element a fixed height */
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
