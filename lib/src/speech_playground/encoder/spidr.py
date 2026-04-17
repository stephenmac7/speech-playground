from typing import Optional

import torch
import torchaudio
from torchaudio.functional import resample


class SpidREncoder:
    """SpidR encoder (facebookresearch/spidr).

    Extracts intermediate representations from the SpidR self-supervised model.
    Features: 768-dim, 16kHz, ~50fps.
    Audio must be standardized (mean=0, var=1) before encoding.
    """

    def __init__(self, *, layer: int = 5, device: Optional[torch.device] = "cuda"):
        from spidr.models import spidr_base

        self.device = device
        self.layer = layer
        self.model = spidr_base().to(device)
        self.model.eval()

    def load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        # SpidR expects standardized audio
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        outputs = self.model.get_intermediate_outputs(waveform.unsqueeze(0).to(self.device))
        # get_intermediate_outputs returns list of (1, T, 768), one per layer (0-indexed)
        return outputs[self.layer - 1].squeeze(0)  # (T, 768)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02
