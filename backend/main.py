from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

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
import time

import dtw

import torchaudio
import torchaudio.functional as F

from speech_playground.alignment import score_frames, score_continuous_frames, build_alignments

# Loads from backend/.env
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from models_config import (
    MODELS,
    MODELS_MAP,
    KANADE_MODELS,
    get_kanade,
    get_kanade_vocoder,
)

# Base directory for /data endpoint
DATA_ROOT = Path(os.getenv("DATA_ROOT"))


@lru_cache()
def get_sylber():
    from speech_playground.encoder.sylber import SylberEncoder

    return SylberEncoder()


@lru_cache()
def get_ifmdd():
    from speech_playground.ifmdd import IFMDD

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


@lru_cache()
def get_encoders():
    return {
        model.slug: {
            "label": model.name,
            "discretizers": model.discretizers(),
            "default_dist_method": model.default_dist_method,
        } for model in MODELS
    }


@app.get("/models")
def models_endpoint():
    """
    Tells the frontend what models are available. Returns flat lists for UI simplicity.
    """
    encoders = [{"value": k, **v} for k, v in get_encoders().items()]
    vc_models = [
        {"value": model["variant"], "label": model["name"]} for model in KANADE_MODELS
    ]

    return {
        "encoders": encoders,
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
    encoder: str = Form(...),
    discretizer: Optional[str] = Form(None),
    dist_method: Optional[str] = Form(None),
):
    model = MODELS_MAP.get(encoder)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown encoder '{encoder}'.")

    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = model.resample(xwav.squeeze(0), xsr)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = model.resample(ywav.squeeze(0), ysr)

    start_time = time.time()
    x = model.encode(xwav)
    y = model.encode(ywav)
    end_time = time.time()
    print(f"Encoding time for model '{model.slug}': {end_time - start_time:.2f} seconds")

    if hasattr(model, "extra_results"):
        extra_results = model.extra_results(x, y)
    else:
        extra_results = {}

    if discretizer is not None:
        xtokens = model.discretize(x, discretizer)
        ytokens = model.discretize(y, discretizer)
        scores, path = score_frames(ytokens, xtokens, normalize=True)
        alignment_map = build_alignments(path, len(ytokens), len(xtokens))
    else:
        x = model.to_continuous_features(x)
        y = model.to_continuous_features(y)
        dist_method = dist_method or model.default_dist_method

        alignment = dtw.dtw(x, y, dist_method=dist_method, keep_internals=True)
        alignment_map = dtw.warp(alignment, index_reference=False)

        y_scores_sum = np.zeros(len(y))
        y_scores_count = np.zeros(len(y))

        for i, j in zip(alignment.index1, alignment.index2):
            y_scores_sum[j] += alignment.localCostMatrix[i, j] 
            y_scores_count[j] += 1

        y_positions = np.divide(
            y_scores_sum,
            y_scores_count,
            out=np.full_like(y_scores_sum, np.nan), # Use nan or inf for 0 counts
            where=y_scores_count != 0,
        )

        if dist_method == "euclidean":
            scores = np.exp(model.score_alpha * (y_positions**2))
        else:
            scores = y_positions

        # visualize path for debugging
        # plt.imshow(cosine_sims, aspect="auto", origin="lower")
        # plt.colorbar()
        # plt.title("Cosine Similarity")
        # plt.xlabel("Y Segments")
        # plt.ylabel("X Segments")
        # Overlay the DTW path: x-axis is columns (j -> Y segments), y-axis is rows (i -> X segments)
        # plt.plot([j for i, j in path], [i for i, j in path], color="red", linewidth=1)
        # plt.savefig("/tmp/cosine_similarity.png")
        # plt.close()
        # plt.figure(figsize=(10,5))
        # plt.plot(y_positions)
        # plt.plot(scores)
        # plt.savefig("/tmp/y_positions.png")
        # plt.close()

    # Plot for debugging
    # plot_waveform(ywav, sr, agreement_scores=y_positions)
    # plt.savefig("/tmp/comparison.png")
    # plt.close()
    return {
        "scores": scores.tolist(),
        "frameDuration": model.frame_duration,
        "alignmentMap": alignment_map.tolist(),
        **extra_results
    }


@app.post("/compare_dpdp")
def compare_dpdp_endpoint(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    encoder: str = Form(...),
    gamma: float = Form(...),
    discretizer: str = Form(...),
):
    from speech_playground.tokenizer.dpdp import DPDPTokenizer

    model = MODELS_MAP.get(encoder)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown encoder '{encoder}'.")

    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = model.resample(xwav.squeeze(0), xsr)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = model.resample(ywav.squeeze(0), ysr)

    x = model.to_continuous_features(model.encode(xwav))
    y = model.to_continuous_features(model.encode(ywav))

    cluster_centers = model.cluster_centers(discretizer)
    tokenizer = DPDPTokenizer(cluster_centers=cluster_centers)

    xcodes, xboundaries = tokenizer.tokenize_one(x, gamma=gamma)
    ycodes, yboundaries = tokenizer.tokenize_one(y, gamma=gamma)

    y_mismatches, path = score_frames(ycodes, xcodes, normalize=True)
    alignment_map_codes = build_alignments(path, len(ycodes), len(xcodes))
    alignment_map = np.zeros(len(y), dtype=int)
    for j in range(len(ycodes)):
        alignment_map[j] = alignment_map_codes[j]

    return {
        "scores": y_mismatches.tolist(),
        "boundaries": yboundaries.tolist(),
        "modelBoundaries": xboundaries.tolist(),
        "frameDuration": model.frame_duration,
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

    xsegments = x["segments"].tolist()
    ysegments = y["segments"].tolist()

    scores, path = score_continuous_frames(y["segment_features"], x["segment_features"], normalize=True)
    alignment_map = build_alignments(path, len(ysegments), len(xsegments), fill_backwards=False)

    result = {
        "scores": scores.tolist(), #y_avg_score.tolist(),
        "xsegments": xsegments,
        "ysegments": ysegments,
        "y_to_x_mappings": alignment_map.tolist(), #y_to_x_mappings,
    }
    return result


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

