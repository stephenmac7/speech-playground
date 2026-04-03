<script lang="ts">
	import type { Snippet } from 'svelte';

	let { active, children }: { active: boolean; children: Snippet } = $props();

	let hasBeenActive = $state(false);
	$effect(() => {
		if (active) {
			hasBeenActive = true;
		}
	});
</script>

{#if hasBeenActive}
	<div style="display: {active ? 'block' : 'none'}">
		{@render children()}
	</div>
{/if}
