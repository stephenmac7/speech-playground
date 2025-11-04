<script lang="ts">
	let { align = 'right', children } = $props()
</script>

<div class="tooltip-container">
	<div class="tooltip-trigger">
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="help-icon"
        >
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
	</div>
	<div class="tooltip-text" class:align-left={align === 'left'}>
		{@render children()}
	</div>
</div>

<style>
	.tooltip-container {
		position: relative;
		display: inline-flex;
		align-items: center;
	}

	.tooltip-trigger {
		cursor: help;
		display: inline-flex;
	}

	.tooltip-text {
		visibility: hidden;
		width: max-content;
		max-width: 400px;
		background-color: var(--background-color);
		color: var(--foreground-color);
		text-align: left;
		border-radius: 6px;
		padding: 10px;
		position: absolute;
		z-index: 10;
		bottom: 125%;
		opacity: 0;
		transition: opacity 0.3s;
		border: 1px solid var(--border-color);
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
		font-size: 0.9em;
		line-height: 1.4;
	}

	.tooltip-text.align-left {
		left: 0;
	}

	.tooltip-text:not(.align-left) {
		right: 0;
	}

	.tooltip-container:hover .tooltip-text {
		visibility: visible;
		opacity: 1;
	}

	.help-icon {
		cursor: help;
		color: var(--foreground-color);
		opacity: 0.7;
	}
</style>
