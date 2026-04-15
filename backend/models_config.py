from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
import os
import json
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
import torch
import torchaudio.functional as F

# Load configuration via environment variables
PARENT_DIR = Path(__file__).parent

KANADE_MODELS_PATH = Path(os.getenv("KANADE_MODELS_PATH"))
if KANADE_MODELS_PATH is None:
    raise ValueError(
        "KANADE_MODELS_PATH environment variable must be set. Copy backend/.env.example to backend/.env."
    )
if not KANADE_MODELS_PATH.is_absolute():
    KANADE_MODELS_PATH = PARENT_DIR / KANADE_MODELS_PATH
with open(KANADE_MODELS_PATH, "r") as f:
    KANADE_MODELS = json.load(f)
KANADE_VARIANTS = {model["variant"]: model for model in KANADE_MODELS}


class ModelMetadata(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def discretizers(self): ...

    @abstractmethod
    def encode(self, waveform: torch.Tensor): ...

    def to_continuous_features(self, encoded):
        return encoded

    def resample(self, waveform: torch.Tensor, sr: int):
        model = self.load()
        if sr != model.sample_rate:
            waveform = F.resample(waveform, sr, model.sample_rate)
        return waveform

    @property
    def frame_duration(self) -> Optional[float]:
        model = self.load()
        return model.frame_shift

    @property
    def euclidean_alpha(self):
        """Alpha parameter for converting distances to scores."""
        return 0.00173

    @property
    def cosine_alpha(self):
        """Alpha parameter for sharpening cosine distance."""
        return 2.0

    @property
    def default_dist_method(self):
        return "cosine"

    @property
    def has_fixed_frame_rate(self) -> bool:
        print(f"{self.__class__.__name__}: has_fixed_frame_rate is not implemented; using frame_duration which may be slow.")
        return self.frame_duration is not None

    def get_segments(self, encoded):
        fd = self.frame_duration
        if fd is None:
            raise NotImplementedError(
                "This model does not have a fixed frame rate. Please override get_segments()."
            )
        features = self.to_continuous_features(encoded)
        length = len(features)
        return [[i * fd, (i + 1) * fd] for i in range(length)]


HUBERT_KMEANS_PATH = os.getenv("HUBERT_KMEANS_PATH")
WAVLM_BASE_PLUS_KMEANS_PATH = os.getenv("WAVLM_BASE_PLUS_KMEANS_PATH")


@lru_cache()
def load_kmeans(path):
    import joblib

    kmeans_path = Path(path)
    if not kmeans_path.is_absolute():
        kmeans_path = PARENT_DIR / kmeans_path

    return joblib.load(kmeans_path)


DEFAULT_KMEANS_URL = (
    "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt"
)


@lru_cache()
def load_kmeans_default():
    model = KMeans(100)
    checkpoint = torch.hub.load_state_dict_from_url(DEFAULT_KMEANS_URL, progress=False)
    model.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
    model.__dict__["_n_threads"] = checkpoint["_n_threads"]
    model.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return model


class HubertMetadata(ModelMetadata):
    slug = "hubert_l7"
    name = "HuBERT L7"

    def __init__(self, *, device: Optional[torch.device] = "cuda"):
        self.device = device

    @lru_cache()
    def load(self):
        from speech_playground.encoder.hubert import HubertEncoder

        return HubertEncoder(device=self.device)

    def load_kmeans(self, name: str):
        if HUBERT_KMEANS_PATH is None:
            return load_kmeans_default()
        return load_kmeans(Path(HUBERT_KMEANS_PATH) / f"{name}.joblib")

    def cluster_centers(self, name: str):
        return self.load_kmeans(name).cluster_centers_

    @lru_cache()
    def discretizers(self):
        if HUBERT_KMEANS_PATH is None:
            return ["dusted-english-100"]
        return sorted([p.stem for p in Path(HUBERT_KMEANS_PATH).glob("*.joblib")])

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform).cpu().numpy()

    def discretize(self, features: np.ndarray, discretizer_name: str):
        from speech_playground.tokenizer.kmeans import KMeansTokenizer

        return KMeansTokenizer(self.load_kmeans(discretizer_name)).tokenize_one(features)

    @property
    def has_fixed_frame_rate(self) -> bool:
        return True

    @property
    def euclidean_alpha(self):
        return 0.02

    @property
    def cosine_alpha(self):
        return 4.0


class WavLMMetadata(ModelMetadata):
    def __init__(
        self,
        slug: str = "wavlm-base-plus",
        name: str = "WavLM Base Plus",
        model_name: str = "microsoft/wavlm-base-plus",
        layer: Optional[int] = None,
        transform_path: Optional[str] = None,
    ):
        self._slug = slug
        self._name = name
        self.model_name = model_name
        self.layer = layer
        self._transform_weight = None
        self._transform_bias = None
        if transform_path is not None:
            data = np.load(transform_path)
            self._transform_weight = data["weight"]  # (out, in)
            self._transform_bias = data["bias"]  # (out,)

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def name(self) -> str:
        return self._name

    def load(self, *, layer=None):
        if layer is not None:
            return self._load(layer)
        return self._load(self.layer)

    @lru_cache()
    def _load(self, layer):
        from speech_playground.encoder.wavlm import WavLMEncoder

        return WavLMEncoder(model_name=self.model_name, layer=layer)

    def load_kmeans(self, name: str):
        return load_kmeans(Path(WAVLM_BASE_PLUS_KMEANS_PATH) / f"{name}.joblib")

    def cluster_centers(self, name: str):
        return self.load_kmeans(name).cluster_centers_

    @lru_cache()
    def discretizers(self):
        if WAVLM_BASE_PLUS_KMEANS_PATH is None:
            return []
        return sorted([p.stem for p in Path(WAVLM_BASE_PLUS_KMEANS_PATH).glob("*.joblib")])

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform).cpu().numpy()

    def to_continuous_features(self, encoded):
        if self._transform_weight is not None:
            out = encoded @ self._transform_weight.T
            if self._transform_bias is not None:
                out = out + self._transform_bias
            return out
        return encoded

    def discretize(self, features: np.ndarray, discretizer_name: str):
        from speech_playground.tokenizer.kmeans import KMeansTokenizer

        return KMeansTokenizer(self.load_kmeans(discretizer_name)).tokenize_one(features)

    @property
    def has_fixed_frame_rate(self) -> bool:
        return True

    @property
    def cosine_alpha(self):
        return 2.0


@lru_cache()
def get_kanade(variant: str):
    from speech_playground.encoder.kanade import KanadeEncoder

    model = KANADE_VARIANTS[variant]
    return KanadeEncoder(**model["source"])


INVERSION_TOP = os.getenv("INVERSION_TOP")
if INVERSION_TOP is None:
    INVERSION_WEIGHTS_PATH = None
    INVERSION_MU_PATH = None
    INVERSION_STD_PATH = None
else:
    if not Path(INVERSION_TOP).is_absolute():
        INVERSION_TOP = PARENT_DIR / INVERSION_TOP
    if not Path(INVERSION_TOP).exists():
        raise ValueError(f"INVERSION_TOP directory '{INVERSION_TOP}' does not exist.")
    INVERSION_TOP = Path(INVERSION_TOP)
    INVERSION_WEIGHTS_PATH = Path(
        os.getenv(
            "INVERSION_WEIGHTS_PATH",
            str(INVERSION_TOP / "checkpoints/inversion/wavlm_baseplus_inversion_distilled"),
        )
    )
    INVERSION_MU_PATH = Path(
        os.getenv(
            "INVERSION_MU_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_mean_EMA.npy")
        )
    )
    INVERSION_STD_PATH = Path(
        os.getenv(
            "INVERSION_STD_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_std_EMA.npy")
        )
    )


PHONVEC_WEIGHTS_PATH = os.getenv("PHONVEC_WEIGHTS_PATH", "weights/phonological_vectors.npz")
if not Path(PHONVEC_WEIGHTS_PATH).is_absolute():
    PHONVEC_WEIGHTS_PATH = PARENT_DIR / PHONVEC_WEIGHTS_PATH
else:
    PHONVEC_WEIGHTS_PATH = Path(PHONVEC_WEIGHTS_PATH)


class PhonologicalVectorMetadata(ModelMetadata):
    slug = "phonological-vector"
    name = "Phonological Vector"

    @lru_cache()
    def load(self):
        from speech_playground.encoder.phonological_vector import PhonologicalVectorEncoder

        return PhonologicalVectorEncoder(weights_path=str(PHONVEC_WEIGHTS_PATH))

    def discretizers(self):
        return []

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform)  # (T, F) np.ndarray

    def extra_results(self, x):
        model = self.load()
        return {
            "phonologicalActivations": x.tolist(),
            "phonologicalFeatureNames": model.featnames,
        }

    @property
    def has_fixed_frame_rate(self) -> bool:
        return True

    @property
    def cosine_alpha(self):
        """Alpha parameter for sharpening cosine distance."""
        return 9.0


class PhonSegMetadata(ModelMetadata):
    slug = "phonseg"
    name = "PhonSeg"

    @lru_cache()
    def load(self):
        from speech_playground.encoder.phonseg import PhonSegEncoder

        return PhonSegEncoder(weights_path=str(PHONVEC_WEIGHTS_PATH))

    def discretizers(self):
        return []

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform)

    def to_continuous_features(self, encoded):
        return encoded["segment_features"].cpu().numpy()

    def get_segments(self, encoded):
        return encoded["segments"].tolist()

    @property
    def frame_duration(self) -> Optional[float]:
        return None

    @property
    def has_fixed_frame_rate(self) -> bool:
        return False

    @property
    def cosine_alpha(self):
        return 2.0


class InversionMetadata(ModelMetadata):
    slug = "inversion"
    name = "Articulatory Inversion"

    @lru_cache()
    def load(self):
        from speech_playground.encoder.articulatory_inversion import ArticulatoryInversionEncoder

        return ArticulatoryInversionEncoder(
            weights=INVERSION_WEIGHTS_PATH, mu_path=INVERSION_MU_PATH, std_path=INVERSION_STD_PATH
        )

    def discretizers(self):
        return []

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform).cpu().numpy()

    def extra_results(self, x):
        model = self.load()
        x_norm = x[:, :12] * model.std + model.mu
        return {"articulatoryFeatures": x_norm.tolist()}

    @property
    def has_fixed_frame_rate(self) -> bool:
        return True

    @property
    def euclidean_alpha(self):
        """Alpha parameter for converting distances to scores."""
        # need high gap penalty -- about -0.5
        return 0.35

    @property
    def default_dist_method(self):
        return "euclidean"


class SylberMetadata(ModelMetadata):
    def discretizers(self):
        return []

    def to_continuous_features(self, encoded):
        return encoded["segment_features"]

    @property
    def frame_duration(self):
        return None

    def get_segments(self, encoded):
        return encoded["segments"].tolist()

    @property
    def has_fixed_frame_rate(self) -> bool:
        return False


class SylberV1Metadata(SylberMetadata):
    def __init__(self):
        pass

    @property
    def slug(self) -> str:
        return "sylber-v1"

    @property
    def name(self) -> str:
        return "Sylber v1"

    @lru_cache()
    def load(self):
        from speech_playground.encoder.sylber import SylberEncoder

        return SylberEncoder()

    def encode(self, waveform: torch.Tensor):
        # Sylber expects normalized audio
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        return self.load().encode_one(waveform, validate=False)

    @property
    def cosine_alpha(self):
        """Alpha parameter for sharpening cosine distance."""
        return 6.0


class ZeroSylMetadata(ModelMetadata):
    def __init__(self, *, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path

    @property
    def slug(self) -> str:
        return "zerosyl"

    @property
    def name(self) -> str:
        return "ZeroSyl"

    @lru_cache()
    def load(self):
        from speech_playground.encoder.zerosyl import ZeroSylEncoder

        return ZeroSylEncoder(checkpoint_path=self.checkpoint_path)

    def discretizers(self):
        return []

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform)

    def to_continuous_features(self, encoded):
        return encoded["segment_features"].cpu().numpy()

    @property
    def frame_duration(self) -> Optional[float]:
        return None

    @property
    def has_fixed_frame_rate(self) -> bool:
        return False

    def get_segments(self, encoded):
        return encoded["segments"].tolist()

    @property
    def cosine_alpha(self):
        return 6.0


MODELS = [
    WavLMMetadata(),
    WavLMMetadata(
        slug="wavlm-large",
        name="WavLM Large",
        model_name="microsoft/wavlm-large",
    ),
    HubertMetadata(),
    SylberV1Metadata(),
    ZeroSylMetadata(checkpoint_path=os.getenv("ZEROSYL_CHECKPOINT_PATH")),
]
if INVERSION_TOP is not None:
    MODELS.append(InversionMetadata())
else:
    print("WARNING: Articulatory Inversion model files not found; skipping inversion encoder.")

if Path(PHONVEC_WEIGHTS_PATH).exists():
    MODELS.append(PhonologicalVectorMetadata())
    MODELS.append(PhonSegMetadata())
else:
    print(
        f"WARNING: Phonological vector weights not found at {PHONVEC_WEIGHTS_PATH}; "
        "skipping phonological-vector encoder."
    )

try:
    from models_local import EXTRA_MODELS

    MODELS.extend(EXTRA_MODELS)
except ImportError:
    pass

MODELS_MAP = {model.slug: model for model in MODELS}


@lru_cache()
def get_kanade_vocoder(variant: str):
    from kanade_tokenizer.util import load_vocoder

    kanade = get_kanade(variant)

    return load_vocoder().to(kanade.device)
