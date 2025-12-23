from typing import Optional
import torch
import torchaudio
from torchaudio.functional import resample


class HubertEncoder:
    def __init__(self, *, language="english", layer=7, device: Optional[torch.device] = "cuda"):
        self.hubert, self.encode_f = torch.hub.load(
            "bshall/dusted:main", "hubert", language=language, trust_repo=True, verbose=False
        )
        self.hubert.to(device)

        self.layer = layer
        self.device = device

    def load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)

    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.encode_f(
            self.hubert, waveform.view(1, 1, -1).to(self.device), layer=self.layer
        ).squeeze()

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"
        return self.encode_f(self.hubert, waveforms.unsqueeze(1).to(self.device), layer=self.layer)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02
