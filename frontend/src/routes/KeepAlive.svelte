<script lang="ts">
	let { active, children } = $props();

	let hasBeenActive = $state(active);
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
