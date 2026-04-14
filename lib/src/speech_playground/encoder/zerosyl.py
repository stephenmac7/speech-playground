from typing import Optional
import torch
import numpy as np

from zerosyl import ZeroSylContinuous


class ZeroSylEncoder:
    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = "cuda",
    ):
        self.device = device
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            from zerosyl.wavlm.WavLM import WavLMConfig

            cfg = WavLMConfig(ckpt["cfg"])
            self.model = ZeroSylContinuous(cfg)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
        else:
            self.model = ZeroSylContinuous.from_remote()
        self.model.to(device)

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> dict:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)
        starts, ends, embeddings = self.model.encode(waveform.unsqueeze(0))
        segments = np.stack(
            [starts.cpu().numpy(), ends.cpu().numpy()], axis=1
        ).astype(np.float64) / self.model.feature_rate
        return {
            "segment_features": embeddings.cpu(),
            "segments": segments,
        }

    def encode(self, waveforms: list[torch.Tensor]) -> list[dict]:
        return [self.encode_one(w) for w in waveforms]

    @property
    def sample_rate(self) -> int:
        return 16000
