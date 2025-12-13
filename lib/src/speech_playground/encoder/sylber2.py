from typing import Optional
import torch
import numpy as np
from sylber2_encoder import Sylber2
import importlib
import os
from torch.nn.utils.rnn import pad_sequence

class Sylber2ContentEncoder:
    def __init__(self, *, checkpoint_path : str | os.PathLike, device : Optional[torch.device] = 'cuda'):
        self.device = device
        with importlib.resources.as_file(importlib.resources.files('sylber2_encoder').joinpath('sylber2.yaml')) as config_path:
            self.model = Sylber2.load_model(config_path, checkpoint_path)
        self.model.to(device)

    def encode_one(self, waveform : torch.Tensor) -> dict:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.encode([waveform])[0]

    def encode(self, waveforms : list[torch.Tensor]) -> list[dict]:
        assert len(waveforms) > 0, "Input waveforms list must not be empty"
        assert waveforms[0].ndim == 1, "Input waveforms must be 1D (samples,)"
        if waveforms[0].device != self.device:
            waveforms = [w.to(self.device) for w in waveforms]

        attention_mask = [torch.ones_like(w) for w in waveforms]
        input_values = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)
        with torch.no_grad():
            outputs = self.model.encode_content(input_values, sample_rate=self.sample_rate, attention_mask=attention_mask)
        # content: (batch, n_segments, content_dim)
        # ft_mask: (batch, n_segments) tensor
        # segments: list of (n_segments, 2) np.ndarray
        results = []
        for i in range(len(waveforms)):
            result = {
                'segment_features': outputs['content'][i][:outputs['segments'][i].shape[0]].cpu(),
                'segments': outputs['segments'][i] * self.frame_shift,
            }
            results.append(result)
        return results

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02  # WavLM uses 20ms frames
