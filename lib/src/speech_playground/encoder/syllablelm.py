import torch
from typing import Optional
import numpy as np

from speech_playground.vendor.syllablelm import (
    SylBoostFeatureReader,
)


class SyllableLMEncoder:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        kmeans_centroids_path: str,
        agglom_indices_path: str,
        model_key: str,
        device: Optional[torch.device] = "cuda",
    ):
        self.device = device
        self.reader = SylBoostFeatureReader(
            sylboost_checkpoint=checkpoint_path,
            kmeans_centroids_path=kmeans_centroids_path,
            agglom_indices_path=agglom_indices_path,
            model_key=model_key,
            device=device,
        )

    def _process_result(self, features, clusters_with_times, cluster_features):
        tokens = clusters_with_times[0]
        start_frames = clusters_with_times[1]
        end_frames = clusters_with_times[2]
        
        segments = np.stack([start_frames, end_frames], axis=1) * self.frame_shift

        return {
            "features": cluster_features,
            "tokens": tokens,
            "segments": segments,
        }

    @torch.inference_mode()
    def encode_one(self, waveform: torch.Tensor) -> dict:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        if waveform.device != self.device:
            waveform = waveform.to(self.device)
        result = self.reader.forward(waveform.unsqueeze(0))
        
        return self._process_result(result["features"][0], result["clusters_with_times"][0], result["cluster_features"][0])

    @torch.inference_mode()
    def encode(self, waveforms: list[torch.Tensor]) -> list[dict]:
        assert len(waveforms) > 0, "Input waveforms list must not be empty"
        assert waveforms[0].ndim == 1, "Input waveforms must be 1D (samples,)"

        batched_waveforms = torch.stack(waveforms)
        if batched_waveforms.device != self.device:
            batched_waveforms = batched_waveforms.to(self.device)
            
        result = self.reader.forward(batched_waveforms)

        results = []
        for i in range(len(waveforms)):
            results.append(
                self._process_result(result["features"][i], result["clusters_with_times"][i], result["cluster_features"][i])
            )
        return results

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02
