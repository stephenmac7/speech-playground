from typing import Optional
from functools import cached_property
import torch
import numpy as np

from pathlib import Path

from kanade_tokenizer.model import KanadeFeatures, KanadeModel
from kanade_tokenizer.util import load_audio


class KanadeEncoder:
    def __init__(
        self, *, device: Optional[torch.device] = "cuda", **kwargs
    ):
        self.device = device
        self.model = KanadeModel.from_pretrained(**kwargs).eval().to(device)

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

    @property
    def frame_shift(self) -> float:
        return self.model.downsample_factor * 0.02 # since WavLM uses 20ms frames

    @cached_property
    def codebook(self) -> np.ndarray:
        with torch.no_grad():
            return self.model.local_quantizer.decode(
                torch.arange(self.model.local_quantizer.all_codebook_size, device=self.device)
            ).float().cpu().numpy()


class KanadeWavLMEncoder(KanadeEncoder):
    def __init__(self, *, return_only : str = None, **kwargs):
        super().__init__(**kwargs)
        self.return_only = return_only

    def encode_one(self, waveform: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            audio_length = waveform.size(0)
            padding = self.model._calculate_waveform_padding(audio_length, ensure_recon_length=True)
            ssl_real, _ = self.model.forward_ssl_features(waveform.unsqueeze(0).cuda(), padding=padding)
            if self.return_only == "ssl_real":
                return ssl_real.squeeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, _, ssl_recon, _ = self.model.forward_content(ssl_real)

        assert ssl_real.shape == ssl_recon.shape, "Reconstructed features must match original features in shape"

        if self.return_only == "ssl_recon":
            return ssl_recon.squeeze(0)

        return {
            "ssl_real": ssl_real.squeeze(0),
            "ssl_recon": ssl_recon.squeeze(0),
        }

    @property
    def frame_shift(self) -> float:
        return 0.02  # WavLM uses 20ms frames