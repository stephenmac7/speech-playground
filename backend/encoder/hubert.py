from typing import Optional
import torch
import numpy as np


class HubertEncoder:
    def __init__(self, *, device: Optional[torch.device] = "cuda"):
        self.hubert, self.encode_f = torch.hub.load(
            "bshall/dusted:main", "hubert", language="english", trust_repo=True, verbose=False
        )
        self.hubert.to(device)

        self.device = device

    def encode_one(self, waveform: torch.Tensor) -> np.ndarray:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return (
            self.encode_f(self.hubert, waveform.unsqueeze(0).unsqueeze(0).to(self.device))
            .squeeze()
            .cpu()
            .numpy()
        )

    def encode(self, waveforms: torch.Tensor) -> np.ndarray:
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"
        return self.encode_f(self.hubert, waveforms.unsqueeze(1).to(self.device)).cpu().numpy()

    @property
    def sample_rate(self) -> int:
        return 16000
