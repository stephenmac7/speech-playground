from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from pathlib import Path

import io
import numpy as np
import tgt
import soundfile
import torch
import torchaudio

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw_ndim

import torchaudio
import torchaudio.functional as F

from alignment import find_mismatches, plot_waveform

KMEANS_PATH = "/home/smcintosh/default/dusted/kmeans_en+ja_200.joblib"
KANADE_REPO_ROOT = Path("/home/smcintosh/kanade-tokenizer")


# --- Lazy, cached factories (loaded on first use) ---
@lru_cache()
def get_kmeans_model():
    import joblib

    return joblib.load(KMEANS_PATH)


@lru_cache()
def get_hubert():
    from encoder.hubert import HubertEncoder

    return HubertEncoder()


@lru_cache()
def get_kanade(variant : Literal["kanade-12hz", "kanade-25hz"]):
    from encoder.kanade import KanadeEncoder

    # remote kanade-
    name = variant.replace("kanade-", "")

    return KanadeEncoder(
        config_path=KANADE_REPO_ROOT / f"config/model/{name}.yaml",
        weights_path=KANADE_REPO_ROOT / f"weights/{name}.safetensors",
    )

@lru_cache()
def get_kanade_vocoder(variant : Literal["kanade-12hz", "kanade-25hz"]):
    from kanade_tokenizer.util import load_vocoder
    kanade = get_kanade(variant)

    return load_vocoder().to(kanade.device)

@lru_cache()
def get_kmeans_tokenizer():
    from tokenizer.kmeans import KMeansTokenizer

    return KMeansTokenizer(kmeans=get_kmeans_model())


@lru_cache()
def get_dpdp_tokenizer(encoder : Literal["hubert", "kanade-12hz", "kanade-25hz"]):
    from tokenizer.dpdp import DPDPTokenizer

    if encoder == "hubert":
        cluster_centers = get_kmeans_model().cluster_centers_
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


app = FastAPI()


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


def streaming_response_of_audio_file(file, *, apply_vad: bool, debug_save_path: str = None):
    ywav, sr = torchaudio.load_with_torchcodec(file)

    if apply_vad:
        ywav = F.vad(ywav, sr)

    if debug_save_path is not None:
        torchaudio.save(debug_save_path, ywav, sr)

    max_abs = ywav.abs().max() + 1e-8
    if max_abs > 0:
        ywav = ywav / max_abs

    return streaming_response_of_audio(ywav, sr)

@app.post("/process_audio")
def process_audio_endpoint(file: UploadFile = File(...), apply_vad: bool = Form(True)):
    return streaming_response_of_audio_file(
        file.file, apply_vad=apply_vad, debug_save_path="/tmp/uploaded.wav"
    )


# Instead of app.mount("/data", StaticFiles(directory="/work/smcintosh/data", follow_symlink=True), name="data")
# Someday: add caching?
@app.get("/data/{filename:path}")
def data_endpoint(filename: str):
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=404, detail="Only .wav files are supported.")
    parent = Path("/work/smcintosh/data")
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
    gamma: float = Form(...),
    encoder: str = Form(...),
    dpdp: bool = Form(...),
    discretize: bool = Form(...),
):
    xwav, xsr = torchaudio.load_with_torchcodec(model_file.file)
    xwav = xwav.squeeze(0)
    ywav, ysr = torchaudio.load_with_torchcodec(file.file)
    ywav = ywav.squeeze(0)

    if encoder == "hubert":
        hubert = get_hubert()
        x = hubert.encode_one(F.resample(xwav, xsr, hubert.sample_rate))
        y = hubert.encode_one(F.resample(ywav, ysr, hubert.sample_rate))

        def normal_tokens():
            tok = get_kmeans_tokenizer()
            x_tokens = tok.tokenize_one(x)
            y_tokens = tok.tokenize_one(y)
            return x_tokens, y_tokens

    else:
        kanade = get_kanade(encoder)
        xfeatures = kanade.encode_one(F.resample(xwav, xsr, kanade.sample_rate))
        yfeatures = kanade.encode_one(F.resample(ywav, ysr, kanade.sample_rate))

        x = xfeatures.content_embedding.cpu().float().numpy()
        y = yfeatures.content_embedding.cpu().float().numpy()

        def normal_tokens():
            return xfeatures.content_token_indices.cpu().numpy(), yfeatures.content_token_indices.cpu().numpy()

    if discretize:
        if dpdp:
            xcodes, xboundaries = get_dpdp_tokenizer(encoder).tokenize_one(x, gamma=gamma)
            ycodes, yboundaries = get_dpdp_tokenizer(encoder).tokenize_one(y, gamma=gamma)

            y_mismatches = find_mismatches(ycodes, xcodes, normalize=True)
            y_positions = np.zeros(len(y))
            for i in range(len(y_mismatches)):
                l, r = yboundaries[i], yboundaries[i + 1]
                y_positions[l:r] = y_mismatches[i] / (r - l)
        else:
            xtokens, ytokens = normal_tokens()
            y_positions = find_mismatches(ytokens, xtokens, normalize=True)
    else:
        cosine_sims = cosine_similarity(x, y)
        path = dtw_ndim.warping_path(x, y)
        y_scores_sum = np.zeros(len(y))
        y_scores_count = np.zeros(len(y))
        for i, j in path:
            y_scores_sum[j] = cosine_sims[i, j]
            y_scores_count[j] += 1

        y_positions = np.divide(
            y_scores_sum, y_scores_count, out=np.zeros_like(y_scores_sum), where=y_scores_count != 0
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

    return {"scores": y_positions.tolist(), "frameDuration": 0.02 if encoder == "hubert" else 0.08}


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
        y_scores_sum[j] = cosine_sims[i, j]
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
    model : Literal["kanade-12hz", "kanade-25hz"] = Form(...),
):
    from kanade_tokenizer.util import vocode

    kanade = get_kanade(model)
    source_wav, source_sr = torchaudio.load_with_torchcodec(source.file)
    source_wav = source_wav.squeeze(0)
    reference_wav, reference_sr = torchaudio.load_with_torchcodec(reference.file)
    reference_wav = reference_wav.squeeze(0)

    mel_spectrogram = kanade.model.voice_conversion(
        source_waveform=F.resample(source_wav, source_sr, kanade.sample_rate).to(kanade.device),
        reference_waveform=F.resample(reference_wav, reference_sr, kanade.sample_rate).to(kanade.device),
    )
    vocoder = get_kanade_vocoder(model)
    converted_waveform = vocode(vocoder, mel_spectrogram.unsqueeze(0))

    return streaming_response_of_audio(converted_waveform, kanade.sample_rate)

@app.post("/reconstruct")
def reconstruct_endpoint(
    file: UploadFile = File(...),
    model : Literal["kanade-12hz", "kanade-25hz"] = Form(...),
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
