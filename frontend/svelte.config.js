import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter(),
		// Set BASE_PATH (e.g. BASE_PATH=/myapp pnpm run build) when serving the app
		// under a subdirectory behind a reverse proxy. It is baked in at build
		// time, so changing the public path means rebuilding.
		paths: { base: process.env.BASE_PATH ?? '' }
	},
	compilerOptions: { experimental: { async: true } }
};

export default config;
