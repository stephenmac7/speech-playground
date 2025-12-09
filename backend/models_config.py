from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
import os
import json
import numpy as np
from sklearn.cluster import KMeans
import torch
import torchaudio.functional as F

# Load configuration via environment variables
PARENT_DIR = Path(__file__).parent

KANADE_MODELS_PATH = Path(os.getenv("KANADE_MODELS_PATH"))
if KANADE_MODELS_PATH is None:
    raise ValueError("KANADE_MODELS_PATH environment variable must be set. Copy backend/.env.example to backend/.env.")
if not KANADE_MODELS_PATH.is_absolute():
    KANADE_MODELS_PATH = PARENT_DIR / KANADE_MODELS_PATH
with open(KANADE_MODELS_PATH, "r") as f:
    KANADE_MODELS = json.load(f)
KANADE_VARIANTS = {model["variant"]: model for model in KANADE_MODELS}

class ModelMetadata(ABC):
    @property
    @abstractmethod
    def slug(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def discretizers(self):
        ...

    @abstractmethod
    def encode(self, waveform: torch.Tensor):
        ...

    def to_continuous_features(self, encoded):
        return encoded

    def resample(self, waveform: torch.Tensor, sr: int):
        model = self.load()
        if sr != model.sample_rate:
            waveform = F.resample(waveform, sr, model.sample_rate)
        return waveform

    @property
    def frame_duration(self) -> float:
        model = self.load()
        return model.frame_shift

    @property
    def score_alpha(self):
        """Alpha parameter for converting distances to scores."""
        return -0.00173

    @property
    def default_dist_method(self):
        return "cosine"

HUBERT_KMEANS_PATH = os.getenv("HUBERT_KMEANS_PATH")

@lru_cache()
def load_kmeans(path):
    import joblib

    print("Loading KMeans model for HuBERT from", path)
    kmeans_path = Path(path)
    if not kmeans_path.is_absolute():
        kmeans_path = PARENT_DIR / kmeans_path

    return joblib.load(kmeans_path)

DEFAULT_KMEANS_URL = "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt"

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

    @lru_cache()
    def load(self):
        from speech_playground.encoder.hubert import HubertEncoder

        return HubertEncoder()

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
    def score_alpha(self):
        return -0.02



@lru_cache()
def get_kanade(variant: str):
    from speech_playground.encoder.kanade import KanadeEncoder

    model = KANADE_VARIANTS[variant]
    return KanadeEncoder(**model["source"])


class KanadeMetadata(ModelMetadata):
    def __init__(self, variant: str, name: str):
        self.variant = variant
        self._name = name

    @property
    def slug(self) -> str:
        return f"kanade-{self.variant}"

    @property
    def name(self) -> str:
        return self._name

    def load(self):
        return get_kanade(self.variant)

    def cluster_centers(self, name: str):
        return get_kanade(self.variant).codebook

    def discretizers(self):
        return [ "DEFAULT" ]

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform)

    def discretize(self, features, discretizer_name: str):
        return features.content_token_indices.cpu().numpy()

    def to_continuous_features(self, encoded):
        return encoded.content_embedding.float().cpu().numpy()


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
        os.getenv("INVERSION_MU_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_mean_EMA.npy"))
    )
    INVERSION_STD_PATH = Path(
        os.getenv("INVERSION_STD_PATH", str(INVERSION_TOP / "normalising_vectors/JW13_std_EMA.npy"))
    )

class InversionMetadata(ModelMetadata):
    slug = "inversion"
    name = "Articulatory Inversion"

    def load(self):
        from speech_playground.encoder.articulatory_inversion import ArticulatoryInversionEncoder

        return ArticulatoryInversionEncoder(
            weights=INVERSION_WEIGHTS_PATH, mu_path=INVERSION_MU_PATH, std_path=INVERSION_STD_PATH
        )

    def discretizers(self):
        return []

    def encode(self, waveform: torch.Tensor):
        return self.load().encode_one(waveform).cpu().numpy()

    def extra_results(self, x, y):
        model = self.load()
        x_norm = x[:, :12] * model.std + model.mu
        y_norm = y[:, :12] * model.std + model.mu
        return {
            "articulatoryFeatures": [x_norm.tolist(), y_norm.tolist()]
        }

    @property
    def score_alpha(self):
        """Alpha parameter for converting distances to scores."""
        return -0.7
    
    @property
    def default_dist_method(self):
        return "euclidean"

MODELS = [
    HubertMetadata(),
    *(KanadeMetadata(variant=model["variant"], name=model["name"]) for model in KANADE_MODELS),
]
if INVERSION_TOP is not None:
    MODELS.append(InversionMetadata())
else:
    print("WARNING: Articulatory Inversion model files not found; skipping inversion encoder.")

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
