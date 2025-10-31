# Speech Playground Backend

A FastAPI-based backend for speech processing, featuring audio comparison, voice conversion, and speech tokenization using various encoders.

This can be used along with the Svelte frontend (see parent directory) to create interactive speech experiments. You can also play with the included tools.

## Quick Start

Run the development server using:

```bash
uv run fastapi dev main.py
```

Add the `--all-extras` flag if you want to be able to use all features (see [below](#lazy-loaded-modules)).

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Important Notes

### Lazy-Loaded Modules

This project uses lazy loading for various speech processing models and encoders. Dependencies are loaded on-demand when specific endpoints are first accessed, rather than at application startup. This improves startup time but means:

- **First requests to each endpoint may be slower** as models are loaded into memory
- **Additional dependencies may be required** depending on which features you use
- **Some imports will fail** until you install the necessary dependencies for the specific encoder/tokenizer you want to use. These are specified as optional dependencies.

#### Note on Kanade
Most dependencies are listed properly in the `kanade` group, but you'll need to install flash-attn by hand. See the README.md in `kanade-tokenizer`.

## Configuration

Key paths that may need adjustment in `main.py`:
- `KMEANS_PATH`: Path to pre-trained K-Means model
- `KANADE_REPO_ROOT`: Root directory of the Kanade tokenizer repository
- Data directory: Top-level directory that can be read by users of the application (for development convenience)
