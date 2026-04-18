<script lang="ts">
	import Tooltip from '$lib/Tooltip.svelte';
	import type { ModelsResponse, EncoderConfig } from '$lib/types';

	let {
		modelsConfig,
		config = $bindable()
	}: {
		modelsConfig: ModelsResponse;
		config: EncoderConfig;
	} = $props();

	const encoderOptions = $derived(modelsConfig.encoders);
	const selectedEncoderOption = $derived(encoderOptions.find((o) => o.value === config.encoder));
	const supportsDiscretize = $derived((selectedEncoderOption?.discretizers.length ?? 0) > 0);
	const supportsDpdp = $derived(selectedEncoderOption?.supports_dpdp ?? false);

	$effect(() => {
		if (!selectedEncoderOption) return;
		if (!selectedEncoderOption.discretizers.includes(config.discretizer)) {
			config.discretizer = selectedEncoderOption.discretizers[0];
		}
		if (!supportsDiscretize) {
			config.discretize = false;
		}
		if (!supportsDpdp) {
			config.dpdp = false;
		}
	});
</script>

<fieldset>
	<legend>Encoder</legend>
	<label>
		Encoder:
		<select bind:value={config.encoder}>
			{#each encoderOptions as opt (opt.value)}
				<option value={opt.value}>{opt.label}</option>
			{/each}
		</select>
	</label>
	<div class="radio-group">
		<label>
			<input type="radio" bind:group={config.discretize} value={false} />
			Continuous
		</label>
		<label class:disabled={!supportsDiscretize}>
			<input
				type="radio"
				bind:group={config.discretize}
				value={true}
				disabled={!supportsDiscretize}
			/>
			Discrete
		</label>
	</div>
	{#if config.discretize}
		<fieldset class="sub-fieldset">
			<label>
				Discretizer:
				<select bind:value={config.discretizer}>
					{#if selectedEncoderOption}
						{#each selectedEncoderOption.discretizers as opt (opt)}
							<option value={opt}>{opt}</option>
						{/each}
					{/if}
				</select>
			</label>
			<label class="label-with-tooltip" class:disabled={!supportsDpdp}>
				<span>DPDP:</span>
				<Tooltip align="left">
					Dynamic programming method that produces coarser speech units.
					<br /><br />
					<i
						>Word Segmentation on Discovered Phone Units With Dynamic Programming and
						Self-Supervised Scoring</i
					>
					(Herman Kamper).
					<a
						href="https://doi.org/10.1109/TASLP.2022.3229264"
						target="_blank"
						rel="noopener noreferrer"
					>
						doi:10.1109/TASLP.2022.3229264
					</a>
				</Tooltip>
				<input type="checkbox" bind:checked={config.dpdp} disabled={!supportsDpdp} />
			</label>
			<label class:disabled={!config.dpdp}>
				Gamma: <input
					type="number"
					bind:value={config.gamma}
					min="0.0"
					max="1.0"
					step="0.1"
					class="short-input"
				/>
				<input
					type="range"
					min="0.0"
					max="1.0"
					step="0.01"
					bind:value={config.gamma}
					disabled={!config.dpdp}
				/>
			</label>
		</fieldset>
	{/if}
</fieldset>

<style>
	fieldset {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		border: 1px solid var(--border-color);
		border-radius: 4px;
		padding: 0.75rem;
		background-color: var(--surface-color);
		min-width: 0;
	}
	.sub-fieldset {
		padding: 0.5rem;
		border-style: dashed;
	}
	label {
		display: flex;
		align-items: center;
		gap: 0.5em;
		min-width: 0;
	}
	label select {
		min-width: 0;
	}
	.radio-group {
		display: flex;
		gap: 1rem;
	}
	.disabled {
		opacity: 0.5;
	}
	.label-with-tooltip {
		display: flex;
		align-items: center;
		gap: 0.5em;
	}
	.short-input {
		width: 4em;
	}
</style>
