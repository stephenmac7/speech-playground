"""PhonSeg: ZeroSyl-style segment encoder driven by phonological-vector signals.

Derives phoneme-like boundaries from projections of WavLM-Large features into
a panphon-based phonological feature space (current phone / previous phone /
next phone), combined via the "multiply" method from
``~/phonetic-arithmetic/segment_phones.ipynb``.
"""
from typing import Optional

import numpy as np
import scipy.signal
import torch
import torchaudio.compliance.kaldi as kaldi

from .wavlm import WavLMEncoder


_MEL_FRAME_SHIFT_MS = 10.0


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise ``scipy.spatial.distance.cosine``, vectorized."""
    return 1.0 - np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )


def _frame_delta_like(proj: np.ndarray, offset: int) -> np.ndarray:
    T = proj.shape[0]
    delta = np.full(T, np.nan, dtype=np.float32)
    if offset < T:
        delta[: T - offset] = _cos_dist(proj[:-offset], proj[offset:])
    return delta


def _fwd_contrast(
    proj_ipa: np.ndarray, proj_r1: np.ndarray, W: np.ndarray, lookahead: int
) -> np.ndarray:
    T = proj_ipa.shape[0]
    fwd_proj = proj_r1 @ W
    contrast = np.full(T, np.nan, dtype=np.float32)
    if lookahead < T:
        end = T - lookahead
        contrast[:end] = _cos_dist(fwd_proj[:end], proj_ipa[:end]) - _cos_dist(
            fwd_proj[:end], proj_ipa[lookahead : lookahead + end]
        )
    return contrast


def _bwd_contrast(
    proj_ipa: np.ndarray, proj_l1: np.ndarray, W: np.ndarray, lookbehind: int
) -> np.ndarray:
    T = proj_ipa.shape[0]
    bwd_proj = proj_l1 @ W
    contrast = np.full(T, np.nan, dtype=np.float32)
    if lookbehind < T:
        contrast[lookbehind:] = _cos_dist(bwd_proj[lookbehind:], proj_ipa[lookbehind:]) - _cos_dist(
            bwd_proj[lookbehind:], proj_ipa[: T - lookbehind]
        )
    return contrast


def _melspec_kaldi(audio: np.ndarray, sr: int = 16000, n_mels: int = 40) -> np.ndarray:
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        wav,
        sample_frequency=float(sr),
        frame_length=25.0,
        frame_shift=_MEL_FRAME_SHIFT_MS,
        num_mel_bins=n_mels,
        use_power=True,
        use_energy=False,
        dither=0.0,
        snip_edges=False,
    )
    return feats.cpu().numpy()


def _mel_svf(mel: np.ndarray, left: int, right: int) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    n = mel.shape[0]
    signal = np.full(n, np.nan)
    norms = np.linalg.norm(mel, axis=1)
    for t in range(left, n - right):
        denom = norms[t - left] * norms[t + right]
        if denom > 0:
            signal[t] = 1.0 - np.dot(mel[t - left], mel[t + right]) / denom
    finite = np.isfinite(signal)
    if finite.any():
        lo, hi = np.nanmin(signal), np.nanmax(signal)
        if hi > lo:
            signal[finite] = (signal[finite] - lo) / (hi - lo)
    return signal


def _mel_svf_signal(audio: np.ndarray, left: int, right: int, target_len: int) -> np.ndarray:
    mel = _melspec_kaldi(audio)
    sig = _mel_svf(mel, left=left, right=right)
    if len(sig) == 0 or target_len == 0:
        return np.full(target_len, np.nan, dtype=np.float32)
    indices = np.round(np.linspace(0, len(sig) - 1, target_len)).astype(int)
    return sig[indices].astype(np.float32)


def _shift_signal(signal: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return signal.copy()
    shifted = np.full(signal.shape, np.nan, dtype=signal.dtype)
    if abs(shift_frames) >= len(signal):
        return shifted
    if shift_frames > 0:
        shifted[shift_frames:] = signal[:-shift_frames]
    else:
        shifted[:shift_frames] = signal[-shift_frames:]
    return shifted


def _fill_gaps(raw_mask: np.ndarray) -> np.ndarray:
    neighbors_silent = np.logical_and(
        np.concatenate(([False], raw_mask[:-1])),
        np.concatenate((raw_mask[1:], [False])),
    )
    return np.logical_or(raw_mask, neighbors_silent)


def handle_silence(
    preds: np.ndarray, silence_mask: np.ndarray, snap_tolerance: int = 1
) -> np.ndarray:
    """Port of ``handle_silence`` from ``~/phonetic-arithmetic/segment_phones.ipynb``."""
    preds = np.asarray(preds, dtype=int)
    n_frames = len(silence_mask)

    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(silence_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(silence_mask)))

    snapped: set[int] = set()
    silence_boundaries: list[int] = []

    for s, e in spans:
        if s > 0:
            nearby = preds[(preds >= s - snap_tolerance) & (preds <= s + snap_tolerance)]
            if len(nearby) > 0:
                outside = nearby[nearby <= s]
                silence_boundaries.append(
                    int(outside.min()) if len(outside) > 0 else int(nearby.min())
                )
                snapped.update(nearby.tolist())
            else:
                silence_boundaries.append(s)
        if e < n_frames:
            nearby = preds[(preds >= e - snap_tolerance) & (preds <= e + snap_tolerance)]
            if len(nearby) > 0:
                outside = nearby[nearby >= e]
                silence_boundaries.append(
                    int(outside.max()) if len(outside) > 0 else int(nearby.max())
                )
                snapped.update(nearby.tolist())
            else:
                silence_boundaries.append(e)

    if len(preds) > 0:
        keep = ~silence_mask[preds]
        for i, p in enumerate(preds):
            if int(p) in snapped:
                keep[i] = False
        kept = preds[keep]
    else:
        kept = preds

    return np.unique(
        np.concatenate([kept, np.array(silence_boundaries, dtype=int)])
    )


class PhonSegEncoder(WavLMEncoder):
    """ZeroSyl-shaped segment encoder driven by phonological-vector signals."""

    FILTERED_SIGNALS = (
        "frame_delta",
        "fwd_delta",
        "bwd_delta",
        "fwd_contrast",
        "bwd_contrast",
        "mel_svf",
    )
    SIGNAL_KWARGS = {
        "frame_delta": {"offset": 1},
        "fwd_delta": {"offset": 1},
        "bwd_delta": {"offset": 2},
        "fwd_contrast": {"lookahead": 2},
        "bwd_contrast": {"lookbehind": 2},
        "mel_svf": {"left": 1, "right": 1},
    }
    SHIFTS = {
        "frame_delta": 0,
        "fwd_delta": 0,
        "bwd_delta": 1,
        "fwd_contrast": 1,
        "bwd_contrast": -2,
        "mel_svf": 0,
    }
    DROP_K = 2
    PROMINENCE = 0.001

    def __init__(
        self,
        *,
        weights_path: str,
        model_name: str = "microsoft/wavlm-large",
        device: Optional[torch.device] = "cuda",
    ):
        super().__init__(model_name=model_name, layer=None, device=device)
        data = np.load(weights_path, allow_pickle=False)

        def _bank(prefix: str):
            pos = data[f"{prefix}_pos_vecs"].astype(np.float32)
            zero = data[f"{prefix}_zero_vecs"].astype(np.float32)
            return (
                (pos - zero),  # W_proj: (F, 1024)
                data[f"{prefix}_biases"].astype(np.float32),
                data[f"{prefix}_scales"].astype(np.float32),
            )

        self.W_ipa, self.b_ipa, self.s_ipa = _bank("ipa")
        self.W_l1, self.b_l1, self.s_l1 = _bank("l1")
        self.W_r1, self.b_r1, self.s_r1 = _bank("r1")
        self.W_r1_to_ipa = data["W_r1_to_ipa"].astype(np.float32)
        self.W_l1_to_ipa = data["W_l1_to_ipa"].astype(np.float32)
        self.silence_coef = data["silence_coef"].astype(np.float32)
        self.silence_intercept = float(data["silence_intercept"])

    def _project(self, feats: np.ndarray, W: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
        return (feats @ W.T + b[None, :]) * s[None, :]

    def _signal(
        self,
        name: str,
        proj_ipa: np.ndarray,
        proj_r1: np.ndarray,
        proj_l1: np.ndarray,
        waveform_np: np.ndarray,
    ) -> np.ndarray:
        kw = self.SIGNAL_KWARGS[name]
        if name == "frame_delta":
            return _frame_delta_like(proj_ipa, kw["offset"])
        if name == "fwd_delta":
            return _frame_delta_like(proj_r1, kw["offset"])
        if name == "bwd_delta":
            return _frame_delta_like(proj_l1, kw["offset"])
        if name == "fwd_contrast":
            return _fwd_contrast(proj_ipa, proj_r1, self.W_r1_to_ipa, kw["lookahead"])
        if name == "bwd_contrast":
            return _bwd_contrast(proj_ipa, proj_l1, self.W_l1_to_ipa, kw["lookbehind"])
        if name == "mel_svf":
            return _mel_svf_signal(
                waveform_np, left=kw["left"], right=kw["right"], target_len=proj_ipa.shape[0]
            )
        raise ValueError(f"Unknown signal: {name}")

    def _predict_silence(self, feats: np.ndarray) -> np.ndarray:
        logits = feats @ self.silence_coef + self.silence_intercept
        raw_mask = logits > 0
        return _fill_gaps(raw_mask)

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> dict:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        feats_t = super().encode_one(waveform)  # (T, 1024)
        feats = feats_t.cpu().numpy().astype(np.float32)
        T = feats.shape[0]
        if T == 0:
            return {
                "segment_features": torch.empty((0, feats.shape[1])),
                "segments": np.empty((0, 2), dtype=np.float64),
            }

        proj_ipa = self._project(feats, self.W_ipa, self.b_ipa, self.s_ipa)
        proj_r1 = self._project(feats, self.W_r1, self.b_r1, self.s_r1)
        proj_l1 = self._project(feats, self.W_l1, self.b_l1, self.s_l1)

        waveform_np = waveform.detach().cpu().numpy().astype(np.float32)

        components: list[np.ndarray] = []
        for name in self.FILTERED_SIGNALS:
            sig = self._signal(name, proj_ipa, proj_r1, proj_l1, waveform_np)
            sig = _shift_signal(sig.astype(np.float32), self.SHIFTS[name])
            sig = sig - np.nanmin(sig)  # norm="min"
            components.append(sig)

        stacked = np.stack(components, axis=0)  # (n_signals, T)
        stacked = np.sort(stacked, axis=0)[self.DROP_K :]
        with np.errstate(invalid="ignore"):
            combined = np.prod(stacked, axis=0)
        combined = np.nan_to_num(combined, nan=0.0)

        peaks, _ = scipy.signal.find_peaks(combined, prominence=self.PROMINENCE)
        silence_mask = self._predict_silence(feats)
        peaks = handle_silence(peaks, silence_mask, snap_tolerance=2)

        boundaries = np.unique(
            np.concatenate([[0], np.asarray(peaks, dtype=int), [T]])
        )
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= T)]

        starts = boundaries[:-1]
        ends = boundaries[1:]
        seg_feats = [
            feats_t[s:e].mean(dim=0) for s, e in zip(starts, ends) if e > s
        ]
        if seg_feats:
            segment_features = torch.stack(seg_feats).cpu()
        else:
            segment_features = torch.empty((0, feats.shape[1]))
        valid = ends > starts
        segments = (
            np.stack([starts[valid], ends[valid]], axis=1).astype(np.float64)
            * self.frame_shift
        )
        return {"segment_features": segment_features, "segments": segments}
