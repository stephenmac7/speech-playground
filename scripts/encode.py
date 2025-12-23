# Adapted from bshall/dusted
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_encoder(args):
    if args.encoder == "hubert":
        from speech_playground.encoder.hubert import HubertEncoder

        return HubertEncoder(language=args.language, layer=args.layer)
    elif args.encoder in ["wavlm-l69", "wavlm-l69-reconstruction"]:
        from speech_playground.encoder.kanade import KanadeWavLMEncoder

        output_key = "ssl_real" if args.encoder == "wavlm-l69" else "ssl_recon"
        return KanadeWavLMEncoder(
            config_path="/home/smcintosh/kanade-tokenizer/config/model/25hz.yaml",
            weights_path="/home/smcintosh/kanade-tokenizer/weights/25hz_with_feature_decoder.safetensors",
            return_only=output_key,
        )
    else:
        # try to load from models_config -- add to python path from parent directory
        import sys
        import dotenv
        from pathlib import Path


        backend_path = Path(__file__).resolve().parent.parent / "backend"
        dotenv.load_dotenv(dotenv_path=backend_path / ".env")
        sys.path.append(str(backend_path))
        import models_config
        for model_meta in models_config.MODELS:
            if model_meta.slug == args.encoder:
                return model_meta.load()
    raise ValueError(f"Unknown encoder: {args.encoder}")


def encode_dataset(args):
    print(f"Loading model")
    encoder = get_encoder(args)

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        if out_path.exists():
            continue

        try:
            wav = encoder.load_audio(in_path)
        except RuntimeError:
            print("Skipping corrupted file:", in_path)
            continue

        x = encoder.encode_one(wav)
        assert x.ndim == 2, "Encoded output must be 2D (T, D)"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="name of the encoder to use (default to hubert).",
        default="hubert",
    )
    parser.add_argument(
        "--language",
        choices=["english", "chinese", "french"],
        help="pre-training language of the HuBERT content encoder.",
        default="english",
    )
    parser.add_argument(
        "--layer",
        help="HuBERT layer to extract features from (defaults to 7).",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .wav).",
        default=".wav",
        type=str,
    )

    args = parser.parse_args()
    if args.encoder != "hubert":
        assert args.language == "english", "Language argument is only supported for HuBERT encoder."
        assert (
            args.layer == 7
        ), "Layer argument is ignored for WavLM encoders. Uses average of layers 6 and 9."

    encode_dataset(args)
