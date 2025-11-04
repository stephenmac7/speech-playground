# Speech Playground

Sandbox for speech experiments with a FastAPI backend and a Svelte frontend.

## Quick start

Recommended prerequisites:
- [uv](https://docs.astral.sh/uv/)
- [pnpm](https://pnpm.io/)

### 1) Run the backend (FastAPI)

```bash
cd backend
cp .env.example .env
uv run --all-extras fastapi dev main.py

# optional: skip --all-extras to use only some functionality
```

For Kanade with Flash Attention, you can optionally also run:
```bash
uv pip install ninja
uv pip install flash-attn --no-build-isolation
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 2) Run the frontend (SvelteKit)

```bash
cd frontend
cp .env.example .env
pnpm install
pnpm run dev
```

This will start the app on http://localhost:5173.

### Optional: aligner service

The forced alignment tool requires a separate service. If you want it, run this in a separate process:

- Repo: https://github.com/stephenmac7/mfa-service
- Expected endpoint: http://localhost:8001

## Lazy-loaded modules

This project uses lazy loading for various speech processing models and encoders. Dependencies are loaded on-demand when specific endpoints are first accessed, rather than at application startup. This improves startup time but means:

- First requests to each endpoint may be slower as models are loaded into memory
- Additional dependencies may be required depending on which features you use
- Some imports will fail until you install the necessary optional dependencies for the encoder/tokenizer you want to use

Note on Kanade: Most dependencies are listed properly in the `kanade` group, but you'll need to install flash-attn by hand. See the Kanade repository instructions; a common sequence is shown above.

## Deploying
(Instructions from Svelte -- have not tried this yet)

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.
