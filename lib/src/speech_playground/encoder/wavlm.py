from typing import Optional

import torch
import torchaudio
from torchaudio.functional import resample

from transformers import Wav2Vec2FeatureExtractor, WavLMModel


class WavLMLargeL6Encoder:
    """WavLM Large, layer 6 (0-indexed) via bshall/knn-vc.

    Uses the bshall/knn-vc checkpoint, which differs from microsoft/wavlm-large
    on HuggingFace. Matches the feature extraction in the original LinearVC paper.
    Features: 1024-dim, 16kHz, ~50fps, no normalization.
    """

    def __init__(self, device: Optional[torch.device] = "cuda"):
        self.device = device
        self.model = torch.hub.load(
            "bshall/knn-vc", "wavlm_large",
            trust_repo=True, progress=True, device=device,
        )
        self.model.eval()

    def load_audio(self, filepath) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)  # (samples,)

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        x, _ = self.model.extract_features(waveform.unsqueeze(0), output_layer=6)
        return x.squeeze(0)  # (T, 1024)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02


class WavLMEncoder:
    """WavLM encoder using HuggingFace transformers.

    Args:
        model_name: HuggingFace model ID (default: "microsoft/wavlm-base-plus-sv")
        layer: transformer layer to extract (0-indexed). None = last hidden state.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus-sv",
        layer: Optional[int] = None,
        device: Optional[torch.device] = "cuda",
    ):
        self.device = device
        self.layer = layer
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)

    @torch.inference_mode()
    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (batch, samples)
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"

        inputs = self.processor(
            [w.cpu().numpy() for w in waveforms],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(
            input_values,
            output_hidden_states=(self.layer is not None),
        )
        if self.layer is not None:
            # hidden_states[0] = embeddings, hidden_states[i+1] = layer i (0-indexed)
            return outputs.hidden_states[self.layer + 1]
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
