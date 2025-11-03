# Speech Playground

Sandbox for speech experiments with a FastAPI backend and a Svelte frontend.

## Quick start

Recommended prerequisites:
- [uv](https://docs.astral.sh/uv/)
- [pnpm](https://pnpm.io/)

### 1) Run the backend (FastAPI)

```bash
cd backend
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
pnpm install
pnpm run dev
```

This will start the app on http://localhost:5173.

### Optional: aligner service

The forced alignment tool requires a separate service. If you want it, run this in a separate process:

- Repo: https://github.com/stephenmac7/mfa-service
- Expected endpoint: http://localhost:8001

## Configuration
Some values like the backend ports and the data directory can be configured at `frontend/.env`.

## Notes
- Models and heavy dependencies are lazily loaded on first use; the first request to some endpoints may be slow.
- See `backend/README.md` for deeper backend details if you need them.

## Deploying
(Instructions from Svelte -- have not tried this yet)

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.
