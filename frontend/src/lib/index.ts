// place files you want to import through the `$lib` alias in this folder.

/**
 * Height in px of a waveform track, scaled to the viewport and clamped so it
 * stays legible on short windows without dominating tall ones. Shared by the
 * live recorder and the sample viewers so their waveforms line up.
 */
export function waveformHeight(innerHeight: number): number {
	return Math.round(Math.min(128, Math.max(64, innerHeight * 0.14)));
}
