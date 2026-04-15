from typing import Optional

import numpy as np
import torch

from .wavlm import WavLMEncoder


class PhonologicalVectorEncoder(WavLMEncoder):
    """Projects WavLM-Large features into a panphon-based phonological feature space.

    Weights are trained by ``scripts/train_phonological_vectors.py`` using the
    ``PhonologicalVectors`` method from ``~/phonetic-arithmetic/segment_phones.ipynb``.
    """

    def __init__(
        self,
        weights_path: str,
        model_name: str = "microsoft/wavlm-large",
        layer: Optional[int] = None,
        device: Optional[torch.device] = "cuda",
    ):
        super().__init__(model_name=model_name, layer=layer, device=device)
        data = np.load(weights_path, allow_pickle=False)
        self.pos_vecs = data["ipa_pos_vecs"].astype(np.float32)
        self.zero_vecs = data["ipa_zero_vecs"].astype(np.float32)
        self.scales = data["ipa_scales"].astype(np.float32)
        self.biases = data["ipa_biases"].astype(np.float32)
        self.featnames = [str(n) for n in data["ipa_featnames"]]
        self.W = self.pos_vecs - self.zero_vecs  # (F, 1024)

    def project_raw(self, feats: np.ndarray) -> np.ndarray:
        return (feats @ self.W.T + self.biases[None, :]) * self.scales[None, :]

    def project(self, feats: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.project_raw(feats)))

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> np.ndarray:
        feats = super().encode_one(waveform).cpu().numpy()  # (T, 1024)
        return self.project_raw(feats)  # (T, F)
