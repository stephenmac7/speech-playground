"""SPAM encoder backed by the phonological-posteriogram package.

Wraps a trained :class:`phonological_posteriogram.phone_model.PhoneModel`
artifact (e.g. ``juice500/wavlm-24-phonemodel``) and exposes both the
phonological activation map (``encode_one``) and phone-boundary segmentation
(``segment_one``).

The segmenter overrides the artifact's stored ``combined_signals`` with the
library default, matching ``scripts/evaluate_metrics.py`` in the
phonological-posteriogram repo: stored hparams may predate the blessed
configuration (notably the mel-SVF spec).
"""

from typing import Optional

import numpy as np
import torch


class SpamEncoder:
    """Phonological activation maps and phone segmentation from a PhoneModel."""

    def __init__(self, model_path: str, device: Optional[str] = "cuda"):
        from phonological_posteriogram.phone_model import PhoneModel
        from phonological_posteriogram.segmenter import Segmenter

        self.model = PhoneModel.from_pretrained(model_path, device=device)
        self.segmenter = self.model.segmenter(
            {"combined_signals": Segmenter.default_hparams()["combined_signals"]}
        )

    @property
    def sample_rate(self) -> int:
        return self.model.net_spec["sr"]

    @property
    def frame_shift(self) -> float:
        return self.model.encoder.stride_size / self.sample_rate

    @property
    def featnames(self) -> list[str]:
        return self.model.posteriogram.featnames

    def _features(self, waveform: torch.Tensor) -> np.ndarray:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.model.extract_features(
            waveform.detach().cpu().numpy().astype(np.float32)
        )

    def encode_one(self, waveform: torch.Tensor) -> np.ndarray:
        """Raw (calibrated linear) phonological activations, shape (T, F)."""
        return self.model.posteriogram.project(
            self._features(waveform), view="ipa", act="none"
        )

    def segment_one(self, waveform: torch.Tensor) -> dict:
        """Phone-like segments with mean-pooled SSL features.

        Returns ``{"segment_features": (S, D) torch.Tensor,
        "segments": (S, 2) seconds}``, the same shape of output the old
        PhonSeg encoder produced.
        """
        feats = self._features(waveform)
        T, D = feats.shape
        if T == 0:
            return {
                "segment_features": torch.empty((0, D)),
                "segments": np.empty((0, 2), dtype=np.float64),
            }

        wav_np = waveform.detach().cpu().numpy().astype(np.float32)
        preds = np.asarray(self.segmenter.segment(feats, wav_np), dtype=int)
        boundaries = np.unique(np.concatenate([[0], preds, [T]]))
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= T)]

        starts, ends = boundaries[:-1], boundaries[1:]
        valid = ends > starts
        starts, ends = starts[valid], ends[valid]
        if len(starts):
            segment_features = torch.from_numpy(
                np.stack([feats[s:e].mean(axis=0) for s, e in zip(starts, ends)])
            )
        else:
            segment_features = torch.empty((0, D))
        segments = np.stack([starts, ends], axis=1).astype(np.float64) * self.frame_shift
        return {"segment_features": segment_features, "segments": segments}
