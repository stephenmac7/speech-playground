from typing import Optional
import torch
import torchaudio
from torchaudio.functional import resample

from transformers import Wav2Vec2FeatureExtractor, HubertModel


class HuBERTHFEncoder:
    """HuBERT encoder using HuggingFace transformers.

    Args:
        model_name: HuggingFace model ID (default: "TencentGameMate/chinese-hubert-large")
        layer: transformer layer to extract (1-indexed). None = last hidden state.
    """

    def __init__(
        self,
        model_name: str = "TencentGameMate/chinese-hubert-large",
        layer: Optional[int] = None,
        device: Optional[torch.device] = "cuda",
    ):
        self.device = device
        self.layer = layer
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)

    def _preprocess(self, waveforms: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            [w.cpu().numpy() for w in waveforms],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values.to(self.device)

    @torch.inference_mode()
    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"
        input_values = self._preprocess(waveforms)
        if self.layer is not None:
            outputs = self.model(input_values, output_hidden_states=True)
            return outputs.hidden_states[self.layer]
        outputs = self.model(input_values)
        return outputs.last_hidden_state

    @torch.inference_mode()
    def encode_all_layers(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, ...]:
        assert waveforms.ndim == 2, "Input waveforms must be 2D (batch, samples)"
        input_values = self._preprocess(waveforms)
        outputs = self.model(input_values, output_hidden_states=True)
        return outputs.hidden_states

    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        return self.encode(waveform.unsqueeze(0)).squeeze(0)

    def encode_one_all_layers(self, waveform: torch.Tensor) -> tuple[torch.Tensor, ...]:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        hidden_states = self.encode_all_layers(waveform.unsqueeze(0))
        return tuple(h.squeeze(0) for h in hidden_states)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02


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
