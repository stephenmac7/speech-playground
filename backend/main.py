from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from pathlib import Path
import os

from dotenv import load_dotenv

import io
import numpy as np
import tgt
import soundfile
import tempfile
import torch
import torchaudio

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw_ndim

import torchaudio
import torchaudio.functional as F

from alignment import score_frames, plot_waveform, build_alignments
from tokenizer.kmeans import default_kmeans_model
from encoder.articulatory_inversion import animate_two_scatter

# Loads from backend/.env
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Configuration via environment variables (with conservative fallbacks)
PARENT_DIR = Path(__file__).parent
KMEANS_PATH = os.getenv("KMEANS_PATH")

KANADE_REPO_ROOT = Path(os.getenv("KANADE_REPO_ROOT", "/home/smcintosh/kanade-tokenizer"))
KANADE_MODELS = [
    {
        "variant": "kanade-12hz",
        "name": "Kanade 12.5 Hz",
        "slug": "12hz",
    },
    {
        "variant": "kanade-25hz",
        "name": "Kanade 25 Hz",
        "slug": "25hz",
    },
    {
        "variant": "kanade-25hz-small-vocab",
        "name": "Kanade 25 Hz (Small Vocab)",
        "slug": "25hz_small_vocab",
    }
]
KANADE_SLUGS = {model["variant"]: model["slug"] for model in KANADE_MODELS}

INVERSION_TOP = Path(os.getenv("INVERSION_TOP", "/home/smcintosh/default/ume_erj/"))
INVERSION_WEIGHTS_PATH = Path(
    os.getenv(
        "INVERSION_WEIGHTS_PATH",
        str(INVERSION_TOP / "checkpoints/inversion/vvn_distilled_baseplus"),
    )
)
INVERSION_MU_PATH = Path(
    os.getenv("INVERSION_MU_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_mean_EMA.npy"))
)
INVERSION_STD_PATH = Path(
    os.getenv("INVERSION_STD_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_std_EMA.npy"))
)

# Base directory for /data endpoint
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/work/smcintosh/data"))

Encoder = Literal["hubert", "kanade-12hz", "kanade-25hz", "kanade-25hz-small-vocab", "inversion"]


# --- Lazy, cached factories (loaded on first use) ---
@lru_cache()
def get_kmeans_model():
    if KMEANS_PATH is None:
        return default_kmeans_model()

    import joblib

    kmeans_path = Path(KMEANS_PATH)
    if not kmeans_path.is_absolute():
        kmeans_path = PARENT_DIR / kmeans_path

    return joblib.load(kmeans_path)


@lru_cache()
def get_hubert():
    from encoder.hubert import HubertEncoder

    return HubertEncoder()


@lru_cache()
def get_kanade(variant: str):
    from encoder.kanade import KanadeEncoder

    return KanadeEncoder(
        config_path=KANADE_REPO_ROOT / f"config/model/{KANADE_SLUGS[variant]}.yaml",
        weights_path=KANADE_REPO_ROOT / f"weights/{KANADE_SLUGS[variant]}.safetensors",
    )


@lru_cache()
def get_articulatory_inversion():
    from encoder.articulatory_inversion import ArticulatoryInversionEncoder

    return ArticulatoryInversionEncoder(
        weights=INVERSION_WEIGHTS_PATH, mu_path=INVERSION_MU_PATH, std_path=INVERSION_STD_PATH
    )


@lru_cache()
def get_kanade_vocoder(variant: str):
    from kanade_tokenizer.util import load_vocoder

    kanade = get_kanade(variant)

    return load_vocoder().to(kanade.device)


@lru_cache()
def get_kmeans_tokenizer():
    from tokenizer.kmeans import KMeansTokenizer

    return KMeansTokenizer(kmeans=get_kmeans_model())


@lru_cache()
def get_dpdp_tokenizer(encoder: Encoder):
    from tokenizer.dpdp import DPDPTokenizer

    if encoder == "hubert":
        cluster_centers = get_kmeans_model().cluster_centers_
    elif encoder == "inversion":
        raise NotImplementedError("DPDP tokenizer for inversion encoder is not implemented yet.")
    else:
        cluster_centers = get_kanade(encoder).codebook

    return DPDPTokenizer(cluster_centers=cluster_centers)


@lru_cache()
def get_sylber():
    from encoder.sylber import SylberEncoder

    return SylberEncoder()


@lru_cache()
def get_ifmdd():
    from ifmdd import IFMDD

    return IFMDD()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with tempfile.TemporaryDirectory(prefix="speech-playground") as tmpdirname:
        global tmpdir
        tmpdir = Path(tmpdirname)
        yield


app = FastAPI(lifespan=lifespan)


def parse_textgrid_to_json(textgrid_path: str) -> dict:
    tg = tgt.io.read_textgrid(textgrid_path)
    response_data = {}
    for tier in tg.tiers:
        if not isinstance(tier, tgt.IntervalTier):
            continue

        intervals_data = []
        for interval in tier.intervals:
            text = interval.text
            if not text:
                continue

            if interval.text:
                intervals_data.append(
                    {
                        "start": round(interval.start_time, 4),
                        "end": round(interval.end_time, 4),
                        "content": text,
                    }
                )
        response_data[tier.name] = intervals_data
    return response_data


@app.get("/models")
def models_endpoint():
    """
    Tells the frontend what models are available. Returns flat lists for UI simplicity.
    """
    inversion_disabled = not (
        INVERSION_WEIGHTS_PATH.exists()
        and INVERSION_MU_PATH.exists()
        and INVERSION_STD_PATH.exists()
    )
    if inversion_disabled:
        print("WARNING: Disabling inversion encoder in /models endpoint since files are missing.")

    kanade_encoders = [
        {
            "value": model["variant"],
            "label": model["name"],
            "supports_discretization": True,
        }
        for model in KANADE_MODELS
    ]
    vc_models = [
        {"value": model["variant"], "label": model["name"]} for model in KANADE_MODELS
    ]

    return {
        "encoders": [
            {"value": "hubert", "label": "HuBERT", "supports_discretization": True},
            {
                "value": "inversion",
                "label": "Articulatory Inversion",
                "supports_discretization": False,
                "disabled": inversion_disabled,
            },
            *kanade_encoders,
        ],
        "vc_models": vc_models,
    }


@app.get("/tg")
def textgrid_endpoint(filename: str):
    path = Path(filename)
    if not path.exists() or path.suffix != ".TextGrid":
        raise HTTPException(status_code=404, detail="TextGrid file not found.")

    return parse_textgrid_to_json(filename)


def vad(wav, *args, **kwargs):
    trimmed_start = F.vad(wav, *args, **kwargs)
    trimmed_end = F.vad(torch.flip(trimmed_start, dims=[1]), *args, **kwargs)
    return torch.flip(trimmed_end, dims=[1])


def streaming_response_of_audio(waveform: torch.Tensor, sample_rate: int) -> StreamingResponse:
    buffer = io.BytesIO()
    soundfile.write(buffer, waveform.squeeze().cpu().numpy(), sample_rate, format="wav")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


def streaming_response_of_audio_file(file, *, apply_vad: bool, debug_save_filename: str = None):
    ywav, sr = torchaudio.load_with_torchcodec(file)

    if apply_vad:
        ywav = F.vad(ywav, sr)
        if ywav.shape[1] == 0:
            raise HTTPException(status_code=400, detail="No speech detected after VAD.")

    if debug_save_filename is not None:
        debug_save_path = tmpdir / debug_save_filename
        torchaudio.save(debug_save_path, ywav, sr)

    max_abs = ywav.abs().max() + 1e-8
    if max_abs > 0:
        ywav = ywav / max_abs

    return streaming_response_of_audio(ywav, sr)


@app.post("/process_audio")
def process_audio_endpoint(file: UploadFile = File(...), apply_vad: bool = Form(True)):
    return streaming_response_of_audio_file(
        file.file, apply_vad=apply_vad, debug_save_filename="recorded_audio.wav"
    )


# Instead of app.mount("/data", StaticFiles(directory="/work/smcintosh/data", follow_symlink=True), name="data")
# Someday: add caching?
@app.get("/data/{filename:path}")
def data_endpoint(filename: str):
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=404, detail="Only .wav files are supported.")
    parent = DATA_ROOT
    path = (parent / filename).resolve()
    if not str(path).startswith(str(parent.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return streaming_response_of_audio_file(path, apply_vad=False)


@app.post("/ifmdd")
def ifmdd_transcribe_endpoint(file: UploadFile = File(...)):
    ywav, sr = torchaudio.load_with_torchcodec(file.file)
    if sr != 16000:
        ywav = F.resample(ywav, sr, 16000)
    assert ywav.shape[0] == 1, "Only mono audio is supported."

    predicted_tokens, sample_indices = get_ifmdd().transcribe_aligned(ywav.squeeze(0))
    timestamps = (sample_indices / 16000.0).tolist()

    # timestamps are [start_1, start_2, ..., start_n] but we want [(start_1, start_2), (start_2, start_3), ..., (start_n-1, end)]
    intervals = zip(timestamps, timestamps[1:] + [ywav.shape[1] / 16000.0])

    result = [
        {"start": start, "end": end, "content": token}
        for (token, (start, end)) in zip(predicted_tokens, intervals)
        if token != "sil"
    ]
    return {"intervals": result}


@app.post("/compare")
def compare_endpoint(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    encoder: Encoder = Form(...),
    discretize: bool = Form(...),
):
    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = xwav.squeeze(0)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = ywav.squeeze(0)

    extra_results = {}
    if encoder == "hubert":
        hubert = get_hubert()
        frame_duration = hubert.frame_shift
        x = hubert.encode_one(F.resample(xwav, xsr, hubert.sample_rate))
        y = hubert.encode_one(F.resample(ywav, ysr, hubert.sample_rate))

        def get_tokens():
            tok = get_kmeans_tokenizer()
            x_tokens = tok.tokenize_one(x)
            y_tokens = tok.tokenize_one(y)
            return x_tokens, y_tokens

    elif encoder == "inversion":
        inversion = get_articulatory_inversion()
        frame_duration = inversion.frame_shift
        x = inversion.encode_one(F.resample(xwav, xsr, inversion.sample_rate))
        y = inversion.encode_one(F.resample(ywav, ysr, inversion.sample_rate))

        x_norm = x[:, :12] * inversion.std + inversion.mu
        y_norm = y[:, :12] * inversion.std + inversion.mu
        extra_results["articulatoryFeatures"] = [x_norm.tolist(), y_norm.tolist()]

        def get_tokens():
            raise HTTPException(
                status_code=501, detail="Tokenization for inversion encoder is not available."
            )

    else:
        kanade = get_kanade(encoder)
        frame_duration = kanade.frame_shift
        xfeatures = kanade.encode_one(F.resample(xwav, xsr, kanade.sample_rate))
        yfeatures = kanade.encode_one(F.resample(ywav, ysr, kanade.sample_rate))

        x = xfeatures.content_embedding.cpu().float().numpy()
        y = yfeatures.content_embedding.cpu().float().numpy()

        def get_tokens():
            return (
                xfeatures.content_token_indices.cpu().numpy(),
                yfeatures.content_token_indices.cpu().numpy(),
            )

    if discretize:
        xtokens, ytokens = get_tokens()
        y_positions, path = score_frames(ytokens, xtokens, normalize=True)
        alignment_map = build_alignments(path, len(ytokens), len(xtokens))
    else:
        cosine_sims = cosine_similarity(x, y)
        path = dtw_ndim.warping_path(x, y)
        alignment_map = np.zeros(len(y), dtype=int)
        y_scores_sum = np.zeros(len(y))
        y_scores_count = np.zeros(len(y))
        for i, j in path:
            alignment_map[j] = i  # overwrites previous, but that's what we want
            y_scores_sum[j] += cosine_sims[i, j]
            y_scores_count[j] += 1

        y_positions = np.divide(
            y_scores_sum,
            y_scores_count,
            out=np.zeros_like(y_scores_sum),
            where=y_scores_count != 0,
        )

        # visualize path for debugging
        plt.imshow(cosine_sims, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title("Cosine Similarity")
        plt.xlabel("Y Segments")
        plt.ylabel("X Segments")
        # Overlay the DTW path: x-axis is columns (j -> Y segments), y-axis is rows (i -> X segments)
        # plt.plot([j for i, j in path], [i for i, j in path], color="red", linewidth=1)
        # plt.savefig("/tmp/cosine_similarity.png")
        # plt.close()

    # Plot for debugging
    # plot_waveform(ywav, sr, agreement_scores=y_positions)
    # plt.savefig("/tmp/comparison.png")
    # plt.close()
    return {
        "scores": y_positions.tolist(),
        "frameDuration": frame_duration,
        "alignmentMap": alignment_map.tolist(),
        **extra_results,
    }


@app.post("/compare_dpdp")
def compare_dpdp_endpoint(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    encoder: Encoder = Form(...),
    gamma: float = Form(...),
):
    if encoder == "inversion":
        raise HTTPException(status_code=501, detail="DPDP for inversion encoder is not available.")
    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = xwav.squeeze(0)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = ywav.squeeze(0)

    if encoder == "hubert":
        hubert = get_hubert()
        frame_duration = hubert.frame_shift
        x = hubert.encode_one(F.resample(xwav, xsr, hubert.sample_rate))
        y = hubert.encode_one(F.resample(ywav, ysr, hubert.sample_rate))

    else:
        kanade = get_kanade(encoder)
        frame_duration = kanade.frame_shift
        xfeatures = kanade.encode_one(F.resample(xwav, xsr, kanade.sample_rate))
        yfeatures = kanade.encode_one(F.resample(ywav, ysr, kanade.sample_rate))

        x = xfeatures.content_embedding.cpu().float().numpy()
        y = yfeatures.content_embedding.cpu().float().numpy()

    xcodes, xboundaries = get_dpdp_tokenizer(encoder).tokenize_one(x, gamma=gamma)
    ycodes, yboundaries = get_dpdp_tokenizer(encoder).tokenize_one(y, gamma=gamma)

    y_mismatches, path = score_frames(ycodes, xcodes, normalize=True)
    alignment_map_codes = build_alignments(path, len(ycodes), len(xcodes))
    alignment_map = np.zeros(len(y), dtype=int)
    for j in range(len(ycodes)):
        alignment_map[j] = alignment_map_codes[j]

    return {
        "scores": y_mismatches.tolist(),
        "boundaries": yboundaries.tolist(),
        "modelBoundaries": xboundaries.tolist(),
        "frameDuration": frame_duration,
        "alignmentMap": alignment_map.tolist(),
    }


@app.post("/compare_sylber")
def compare_sylber_endpoint(file: UploadFile = File(...), model_file: UploadFile = File(...)):
    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = xwav.squeeze(0)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = ywav.squeeze(0)

    # z-score normalization
    xwav = (xwav - xwav.mean()) / xwav.std()
    ywav = (ywav - ywav.mean()) / ywav.std()

    sylber = get_sylber()
    x = sylber.encode_one(F.resample(xwav, xsr, sylber.sample_rate))
    y = sylber.encode_one(F.resample(ywav, ysr, sylber.sample_rate))

    # plot_waveform(ywav, sr, agreement_scores=y_positions)
    # plt.savefig("/tmp/comparison.png")
    # plt.close()

    # cosine_sims = np.einsum("nd,nd->n", x["segment_features"], y["segment_features"]) / (
    #     np.linalg.norm(x["segment_features"], axis=1)
    #     * np.linalg.norm(y["segment_features"], axis=1)
    # )

    cosine_sims = cosine_similarity(x["segment_features"], y["segment_features"])
    path = dtw_ndim.warping_path(x["segment_features"], y["segment_features"])

    xsegments = x["segments"].tolist()
    ysegments = y["segments"].tolist()

    y_scores_sum = np.zeros(len(ysegments))
    y_scores_count = np.zeros(len(ysegments))
    y_to_x_mappings = [[] for _ in range(len(ysegments))]
    for i, j in path:
        y_scores_sum[j] += cosine_sims[i, j]
        y_scores_count[j] += 1
        y_to_x_mappings[j].append(i)

    y_avg_score = np.divide(
        y_scores_sum, y_scores_count, out=np.zeros_like(y_scores_sum), where=y_scores_count != 0
    )

    return {
        "scores": y_avg_score.tolist(),
        "xsegments": xsegments,
        "ysegments": ysegments,
        "y_to_x_mappings": y_to_x_mappings,
    }


@app.post("/convert_voice")
def convert_voice_endpoint(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    model: str = Form(...),
):
    from kanade_tokenizer.util import vocode

    kanade = get_kanade(model)
    source_wav, source_sr = torchaudio.load_with_torchcodec(source.file)
    source_wav = source_wav.squeeze(0)
    reference_wav, reference_sr = torchaudio.load_with_torchcodec(reference.file)
    reference_wav = reference_wav.squeeze(0)

    mel_spectrogram = kanade.model.voice_conversion(
        source_waveform=F.resample(source_wav, source_sr, kanade.sample_rate).to(kanade.device),
        reference_waveform=F.resample(reference_wav, reference_sr, kanade.sample_rate).to(
            kanade.device
        ),
    )
    vocoder = get_kanade_vocoder(model)
    converted_waveform = vocode(vocoder, mel_spectrogram.unsqueeze(0))

    return streaming_response_of_audio(converted_waveform, kanade.sample_rate)


@app.post("/reconstruct")
def reconstruct_endpoint(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    from kanade_tokenizer.util import vocode

    kanade = get_kanade(model)
    source_wav, source_sr = torchaudio.load_with_torchcodec(file.file)
    source_wav = source_wav.squeeze(0)
    source_wav_resampled = F.resample(source_wav, source_sr, kanade.sample_rate).to(kanade.device)

    features = kanade.encode_one(source_wav_resampled)

    mel_spectrogram = kanade.model.decode(
        content_embedding=features.content_embedding,
        global_embedding=features.global_embedding,
        target_audio_length=len(source_wav_resampled),
    )
    vocoder = get_kanade_vocoder(model)
    reconstructed_waveform = vocode(vocoder, mel_spectrogram.unsqueeze(0))

    return streaming_response_of_audio(reconstructed_waveform, kanade.sample_rate)


# @app.post("/transfer_intervals/{tier}")
# def transfer_intervals_endpoint(tier: str, file: UploadFile = File(...)):
#     EXAMPLE_FILE = Path("/work/smcintosh/experiments/phrase/AE/F01/S_PH_B_1_234.wav")
#     EXAMPLE_TEXTGRID = EXAMPLE_FILE.with_suffix('.TextGrid')
#
#     xwav, _ = torchaudio.load_with_torchcodec(EXAMPLE_FILE)
#     xwav = xwav.unsqueeze(0)
#     x = encode(hubert, xwav.cuda()).squeeze().cpu().numpy()
#     xcodes, xboundaries = segment(x, kmeans.cluster_centers_, gamma=0.2)
#
#     ywav, sr = torchaudio.load_with_torchcodec(file.file, normalize=True)
#     if sr != 16000:
#         raise HTTPException(status_code=400, detail="Uploaded audio must be 16kHz sample rate.")
#
#     ywav = ywav.unsqueeze(0)
#     y = encode(hubert, ywav.cuda()).squeeze().cpu().numpy()
#     ycodes, yboundaries = segment(y, kmeans.cluster_centers_, gamma=0.2)
#     x_sample_map, y_sample_map = create_alignment_map(
#         xcodes, ycodes, xboundaries, yboundaries, xwav[0][0], ywav[0][0],
#         gap_penalty=-1, match_score=1, mismatch_score=-1
#     )
#     tg = tgt.io.read_textgrid(EXAMPLE_TEXTGRID)
#     transferred = transfer_intervals(tg.get_tier_by_name(tier).annotations, (x_sample_map, y_sample_map))
#     result = [
#         {
#             "start": interval.start_time,
#             "end": interval.end_time,
#             "content": interval.text
#         }
#         for interval in transferred
#     ]
#     return {"intervals": result}
