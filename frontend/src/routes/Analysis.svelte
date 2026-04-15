<script lang="ts">
	import SampleViewer from './SampleViewer.svelte';
	import ArticulatoryFeatures from './ArticulatoryFeatures.svelte';
	import EncoderFieldset from '$lib/EncoderFieldset.svelte';
	import { postJson } from '$lib/api';
	import { reportError } from '$lib/errors';
	import { buildNeutralSegmentRegions, buildPhonologicalTiers, type Tier } from '$lib/regions';
	import type { ModelsResponse, EncoderConfig } from '$lib/types';

	let {
		tracks,
		modelsConfig,
		encoderConfig = $bindable()
	}: {
		tracks: Record<string, import('./AudioLibrary.svelte').TrackData>;
		modelsConfig: ModelsResponse;
		encoderConfig: EncoderConfig;
	} = $props();

	const audio = $derived(tracks['Audio']?.data ?? undefined);

	let learnerSegments = $state<number[][] | undefined>();
	let articulatoryFeatures = $state<number[][] | undefined>();
	let phonologicalActivations = $state<number[][] | undefined>();
	let phonologicalFeatureNames = $state<string[] | undefined>();
	let currentTime = $state(0);
	let loading = $state(false);

	const currentFrame = $derived(Math.floor(currentTime * 50));

	const regions = $derived(learnerSegments ? buildNeutralSegmentRegions(learnerSegments) : []);

	const phonologicalTiers = $derived.by<Tier[]>(() =>
		phonologicalActivations && phonologicalFeatureNames
			? buildPhonologicalTiers(phonologicalActivations, phonologicalFeatureNames)
			: []
	);

	function textgridTiers(): Tier[] {
		const tg = tracks['Audio']?.textgrid;
		if (!tg) return [];
		return Object.entries(tg).map(([name, intervals]) => ({
			name,
			regions: intervals.map((iv) => ({
				start: iv.start,
				end: iv.end,
				content: iv.content,
				color: 'rgba(160, 200, 255, 0.6)'
			}))
		}));
	}

	$effect(() => {
		const controller = new AbortController();

		(async () => {
			if (!audio) return;

			loading = true;
			learnerSegments = undefined;
			articulatoryFeatures = undefined;
			phonologicalActivations = undefined;
			phonologicalFeatureNames = undefined;

			const formData = new FormData();
			formData.append('file', audio, 'recording.wav');
			formData.append('encoder', encoderConfig.encoder);
			formData.append('discretize', String(encoderConfig.discretize));
			if (encoderConfig.discretize) {
				formData.append('discretizer', encoderConfig.discretizer);
				formData.append('dpdp', String(encoderConfig.dpdp));
				if (encoderConfig.dpdp) {
					formData.append('gamma', encoderConfig.gamma);
				}
			}

			try {
				const data = await postJson<{
					learnerSegments?: number[][];
					articulatoryFeatures?: number[][];
					phonologicalActivations?: number[][];
					phonologicalFeatureNames?: string[];
				}>(`/api/analyze`, formData, controller.signal);
				learnerSegments = data.learnerSegments;
				articulatoryFeatures = data.articulatoryFeatures;
				phonologicalActivations = data.phonologicalActivations;
				phonologicalFeatureNames = data.phonologicalFeatureNames;
			} catch (e: unknown) {
				if ((e as { name?: string })?.name !== 'AbortError')
					reportError('Error running analysis.', e);
			} finally {
				loading = false;
			}
		})();

		return () => controller.abort();
	});
</script>

<div class={loading ? 'waiting' : ''}>
	<div class="viewer-card">
		<SampleViewer
			{audio}
			tiers={[{ name: 'Segments', regions }, ...textgridTiers(), ...phonologicalTiers]}
			transcript={tracks['Audio']?.transcript ?? undefined}
			bind:currentTime
		/>
	</div>

	{#if articulatoryFeatures}
		<div class="viewer-card">
			<h3>Articulatory Features</h3>
			<ArticulatoryFeatures
				learnerFeatures={articulatoryFeatures[currentFrame]}
				referenceFeatures={undefined}
			/>
		</div>
	{/if}

	<div class="controls">
		<EncoderFieldset {modelsConfig} bind:config={encoderConfig} />
	</div>
</div>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: flex-start;
		justify-content: stretch;
		margin-top: 1rem;
	}
	.viewer-card {
		background-color: var(--surface-color);
		border-radius: 8px;
		border: 1px solid var(--border-color);
		padding: 1rem;
		margin-bottom: 1rem;
	}
</style>
