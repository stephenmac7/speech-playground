from typing import Optional
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

class WavLMEncoder:
    def __init__(self, device: Optional[torch.device] = "cuda"):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (batch, samples)
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"
        
        # processor expects numpy
        inputs = self.processor(
            [w.cpu().numpy() for w in waveforms], 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            return outputs.last_hidden_state

    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (samples,)
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.encode(waveform.unsqueeze(0)).squeeze(0)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02
