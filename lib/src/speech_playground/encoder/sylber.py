from typing import Optional
import torch
import numpy as np
from sylber import Segmenter

class SylberEncoder:
    def __init__(self, *, device : Optional[torch.device] = 'cuda'):
        self.segmenter = Segmenter(device=device)

    def encode_one(self, waveform : torch.Tensor, *, validate=True) -> dict:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.encode([waveform], validate=validate)[0]

    def encode(self, waveforms : list[torch.Tensor], *, validate=True) -> list[dict]:
        assert len(waveforms) > 0, "Input waveforms list must not be empty"
        assert waveforms[0].ndim == 1, "Input waveforms must be 1D (samples,)"
        if validate:
            # check normalized (mean is approx 0, std is approx 1)
            assert (waveforms[0].mean().abs() < 1e-2).all(), "Input waveforms must be normalized (mean approx 0)"
            assert ((waveforms[0].std() - 1.0).abs() < 3e-2).all(), "Input waveforms must be normalized (std approx 1)"
        # Convert from (batch, samples) to list of (channels, samples)
        wav_list = [waveforms[i].unsqueeze(0) for i in range(len(waveforms))]
        return self.segmenter(wav=wav_list)

    @property
    def sample_rate(self) -> int:
        return 16000
