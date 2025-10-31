from typing import Optional
from functools import cached_property
import torch
import numpy as np

from pathlib import Path

from kanade_tokenizer.model import KanadeFeatures, KanadeModel
from kanade_tokenizer.util import load_audio


class KanadeEncoder:
    def __init__(
        self, *, config_path: Path | str, weights_path: Path | str, device: Optional[torch.device] = "cuda"
    ):
        self.device = device
        # TODO: update kanade to accept path objects
        self.model = (
            KanadeModel.from_pretrained(str(config_path), weights_path=str(weights_path)).eval().to(device)
        )

    def load_audio(self, filepath: str) -> torch.Tensor:
        return load_audio(filepath, sample_rate=self.model.config.sample_rate).to(self.device)

    def encode_one(self, waveform: torch.Tensor, **kwargs) -> KanadeFeatures:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        if waveform.device != self.device:
            waveform = waveform.to(self.device)
        return self.model.encode(waveform, **kwargs)

    def encode(self, waveforms: torch.Tensor, **kwargs) -> np.ndarray:
        raise NotImplementedError("Batch encoding not implemented for KanadeEncoder")

    @property
    def sample_rate(self) -> int:
        return self.model.config.sample_rate

    @cached_property
    def codebook(self) -> np.ndarray:
        with torch.no_grad():
            return self.model.local_quantizer.decode(
                torch.arange(self.model.local_quantizer.all_codebook_size, device=self.device)
            ).float().cpu().numpy()
