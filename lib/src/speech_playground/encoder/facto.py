import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample

from ._facto_codec import Encoder

# Private HF repo holding the Facto encoder checkpoint (and the ADEPT
# content/prosody bases). Override the repo with FACTO_HF_REPO, or point
# FACTO_CONTENT_CHECKPOINT at a local file to skip the download entirely.
FACTO_HF_REPO = os.getenv("FACTO_HF_REPO", "stephenmcintosh/facto-50hz-disentangle")
FACTO_CONTENT_CHECKPOINT_FILE = "wavlm-content.ckpt"

# Local checkpoint produced by the linearvc experiment; used directly when present.
_LOCAL_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "linearvc"
    / "pretrained_models"
    / "facto"
    / FACTO_CONTENT_CHECKPOINT_FILE
)


def _resolve_device(device: Optional[str]) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_facto_checkpoint(checkpoint: Optional[Path] = None) -> Path:
    """Locate the Facto encoder checkpoint, downloading from HF if needed.

    Resolution order: explicit ``checkpoint`` arg, ``FACTO_CONTENT_CHECKPOINT``
    env var, the local linearvc copy, then ``hf_hub_download`` from
    ``FACTO_HF_REPO`` (cached under the usual HF cache).
    """
    if checkpoint is not None:
        return Path(checkpoint)
    env_path = os.getenv("FACTO_CONTENT_CHECKPOINT")
    if env_path:
        return Path(env_path)
    if _LOCAL_CHECKPOINT.exists():
        return _LOCAL_CHECKPOINT
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=FACTO_HF_REPO, filename=FACTO_CONTENT_CHECKPOINT_FILE)
    )


class FactoContentEncoder:
    """Facto continuous content codec (the encoder half).

    Produces the 12-dim continuous "content" code at 50 Hz, i.e. the ``prepool``
    stage of the codec (before its internal avg-pool). This is the disentangled
    content representation from the Facto/LinearVC line of work; downstream code
    can split these 12 dims into content/prosody subspaces with a learned
    orthonormal basis.

    The model definition is vendored (``_facto_codec``) and the checkpoint is
    fetched from a (private) HF repo when no local copy is present, so the
    encoder runs without the linearvc experiment checkout.

    The forward pass is memoized on the waveform contents so that callers which
    derive several views (e.g. content and prosody projections) from the same
    audio only pay for one transformer pass.
    """

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = "cuda",
        memo_size: int = 2,
    ):
        self.device = _resolve_device(device)
        self.checkpoint = resolve_facto_checkpoint(checkpoint)

        self.model = Encoder()
        self.model.load_state_dict(torch.load(str(self.checkpoint), map_location="cpu"))
        self.model.eval().to(self.device)

        self._memo: dict[str, torch.Tensor] = {}
        self._memo_order: list[str] = []
        self._memo_size = memo_size

    def load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load_with_torchcodec(filepath)
        wav = resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(self.device)  # (samples,)

    @torch.inference_mode()
    def _forward(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.detach().to(self.device).reshape(1, 1, -1)
        # prepool: replicate the encoder forward but skip the codec avg-pool so
        # we keep the full 50 Hz rate.
        hidden = self.model.feature_extractor(wav).transpose(1, 2)
        hidden = self.model.feature_projection(hidden)
        hidden = self.model.encoder(hidden)
        content, _residual = self.model.content_projection(hidden)
        return content.squeeze(0)  # (T, 12)

    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        key = hashlib.blake2b(
            np.ascontiguousarray(waveform.detach().cpu().numpy()).tobytes(),
            digest_size=16,
        ).hexdigest()
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        code = self._forward(waveform)
        self._memo[key] = code
        self._memo_order.append(key)
        if len(self._memo_order) > self._memo_size:
            del self._memo[self._memo_order.pop(0)]
        return code

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        # 16 kHz / (5*2*2*2*2*2*2 = 320) = 50 Hz.
        return 0.02
