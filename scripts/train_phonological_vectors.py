"""Train PhonologicalVectors (panphon-based linear probes) on TIMIT center-frame
WavLM-Large features.

Run with panphon injected ad-hoc:

    uv run --with panphon python scripts/train_phonological_vectors.py \
        --timit-root /path/to/TIMIT

Writes an ``.npz`` with ``pos_vecs, zero_vecs, scales, biases, featnames, model_name``.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import panphon
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


# TIMIT phone -> IPA mapping (from phonetic-arithmetic/prepare_datasets.py).
TIMIT_TO_IPA: dict[str, str | None] = {
    # Stops
    "b": "b", "d": "d", "g": "ɡ", "p": "p", "t": "t", "k": "k",
    "dx": "ɾ", "q": "ʔ",
    # Affricates
    "jh": "d͡ʒ", "ch": "t͡ʃ",
    # Fricatives
    "s": "s", "sh": "ʃ", "z": "z", "zh": "ʒ",
    "f": "f", "th": "θ", "v": "v", "dh": "ð",
    # Nasals
    "m": "m", "n": "n", "ng": "ŋ",
    "em": "m̩", "en": "n̩", "eng": "ŋ̍", "nx": "ɾ̃",
    # Semivowels/Glides
    "l": "l", "r": "ɹ", "w": "w", "y": "j",
    "hh": "h", "hv": "ɦ", "el": "l̩",
    # Vowels
    "iy": "i", "ih": "ɪ", "eh": "ɛ", "ae": "æ", "aa": "ɑ",
    "ah": "ʌ", "ao": "ɔ", "uh": "ʊ", "uw": "u", "ux": "ʉ",
    "er": "ɝ", "ax": "ə", "ix": "ɨ", "axr": "ɚ", "ax-h": "ə̯",
    # Diphthongs -- ignored (None)
    "ey": None, "aw": None, "ay": None, "oy": None, "ow": None,
    # Closures -- merged into the following stop in the loop
    "bcl": None, "dcl": None, "gcl": None,
    "pcl": None, "tcl": None, "kcl": None,
    # Silence
    "pau": "_", "h#": "_",
    # Epenthetic silence -- drop entirely
    "epi": None,
}

CLOSURE_FOR = {
    "b": "bcl", "d": "dcl", "g": "gcl",
    "p": "pcl", "t": "tcl", "k": "kcl",
    "jh": "dcl", "ch": "tcl",
}


def find_timit_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Enumerate (wav, phn) pairs under the TRAIN/ subtree, case-insensitive."""
    train_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.upper() == "TRAIN"]
    if not train_dirs:
        train_dirs = [root]
    pairs: list[tuple[Path, Path]] = []
    for base in train_dirs:
        for wav in base.rglob("*"):
            if wav.suffix.upper() != ".WAV":
                continue
            phn = wav.with_suffix(".PHN")
            if not phn.exists():
                phn = wav.with_suffix(".phn")
            if phn.exists():
                pairs.append((wav, phn))
    return sorted(pairs)


def read_phn(path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    # TIMIT .WAV uses NIST SPHERE headers; soundfile handles them.
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    assert sr == target_sr, f"Expected {target_sr} Hz, got {sr} Hz: {path}"
    return x


@torch.inference_mode()
def extract_features(
    wav: np.ndarray,
    processor: Wav2Vec2FeatureExtractor,
    model: WavLMModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    out = model(input_values)
    return out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 1024)


def build_feature_cache(
    pairs: list[tuple[Path, Path]],
    model_name: str,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device).eval()

    stride = 320  # WavLM CNN stride @ 16kHz => 20ms frames
    ipas: list[str] = []
    feats: list[np.ndarray] = []

    for wav_path, phn_path in tqdm(pairs, desc="Extracting features"):
        rows = read_phn(phn_path)
        if not rows:
            continue
        try:
            wav = load_wav(wav_path)
        except Exception as e:
            print(f"skip {wav_path}: {e}")
            continue

        hidden = extract_features(wav, processor, model, device)  # (T, 1024)
        T = hidden.shape[0]

        # Closures get merged into the following stop.
        merged: list[tuple[int, int, str]] = []
        for start, stop, phn in rows:
            if merged and phn in CLOSURE_FOR and merged[-1][2] == CLOSURE_FOR[phn]:
                pstart, _pstop, _pphn = merged[-1]
                merged[-1] = (pstart, stop, phn)
            else:
                merged.append((start, stop, phn))

        for start, stop, phn in merged:
            ipa = TIMIT_TO_IPA.get(phn)
            if ipa is None:
                continue
            center_sample = (start + stop) // 2
            frame = min(max(center_sample // stride, 0), T - 1)
            ipas.append(ipa)
            feats.append(hidden[frame])

    return ipas, np.stack(feats).astype(np.float32)


class PhonologicalVectors:
    """Linear probes for panphon phonological features. Copied from
    ``~/phonetic-arithmetic/segment_phones.ipynb`` (lines ~675-770)."""

    @staticmethod
    def prep_featmap(vocab, ft):
        names = ["speech+"] + [f"{n}+" for n in ft.fts("a").names] + [
            f"{n}-" for n in ft.fts("a").names
        ]
        featmap = {}
        for v in vocab:
            if v == "_":
                featmap[v] = [1] + ([0] * (len(names) - 1))
            elif ft.seg_known(v):
                nums = ft.fts(v).numeric()
                featmap[v] = (
                    [0]
                    + [1 if n == 1 else 0 for n in nums]
                    + [1 if n == -1 else 0 for n in nums]
                )
        return names, featmap

    def split_phns(self, featname):
        i = self.featnames.index(featname)
        pos = {p for p, v in self.featmap.items() if v[i] == 1}
        zero = {p for p, v in self.featmap.items() if v[i] == 0}
        return pos, zero

    def calc_phnvectors(self, df, group_col):
        pos_vecs, zero_vecs, scales, biases = [], [], [], []
        self.in_dim = len(df[~df.feat.isna()].iloc[0].feat)
        for featname in self.featnames:
            pos_phns, zero_phns = self.split_phns(featname)
            if len(pos_phns) > 0 and len(zero_phns) > 0:
                pos = np.stack(df[df[group_col].isin(pos_phns)].feat.tolist())
                zero = np.stack(df[df[group_col].isin(zero_phns)].feat.tolist())
                pv = pos.mean(0)
                zv = zero.mean(0)
                w = pv - zv
                pc = (pos @ w.T).mean(0)
                zc = (zero @ w.T).mean(0)
                bias = -(zc + pc) / 2.0
                scale = 4.0 / (pc - zc)
            else:
                pv = np.zeros(self.in_dim)
                zv = np.zeros(self.in_dim)
                bias = 0.0
                scale = 1.0
            pos_vecs.append(pv)
            zero_vecs.append(zv)
            scales.append(scale)
            biases.append(bias)
        self.pos_vecs = np.stack(pos_vecs)
        self.zero_vecs = np.stack(zero_vecs)
        self.scales = np.stack(scales)
        self.biases = np.stack(biases)

    def _filter_features(self):
        speech_pos, speech_zero = self.split_phns("speech+")
        keep = []
        for i, name in enumerate(self.featnames):
            pos_phns, zero_phns = self.split_phns(name)
            if len(pos_phns) == 0 or len(zero_phns) == 0:
                continue
            if name != "speech+" and pos_phns == speech_pos and zero_phns == speech_zero:
                continue
            keep.append(i)
        keep = np.array(keep)
        self.featnames = [self.featnames[i] for i in keep]
        self.pos_vecs = self.pos_vecs[keep]
        self.zero_vecs = self.zero_vecs[keep]
        self.scales = self.scales[keep]
        self.biases = self.biases[keep]
        self.featmap = {p: [vals[i] for i in keep] for p, vals in self.featmap.items()}

    def __init__(self, df, vocab, group_col="ipa", filter_features=True):
        ft = panphon.FeatureTable()
        self.featnames, self.featmap = self.prep_featmap(vocab, ft)
        self.calc_phnvectors(df, group_col)
        if filter_features:
            self._filter_features()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timit-root", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/wavlm-large")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    default_cache = Path("backend/weights/phonvec_train_feats.npz")
    parser.add_argument("--feature-cache", type=Path, default=default_cache)
    default_out = os.getenv("PHONVEC_WEIGHTS_PATH") or "backend/weights/phonological_vectors.npz"
    parser.add_argument("--output", type=Path, default=Path(default_out))
    args = parser.parse_args()

    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_cache.exists():
        print(f"Loading cached features from {args.feature_cache}")
        cached = np.load(args.feature_cache, allow_pickle=False)
        ipas = [str(s) for s in cached["ipas"]]
        feats = cached["feats"]
    else:
        pairs = find_timit_pairs(args.timit_root)
        if not pairs:
            raise SystemExit(f"No .WAV/.PHN pairs found under {args.timit_root}")
        print(f"Found {len(pairs)} utterances")
        ipas, feats = build_feature_cache(pairs, args.model, torch.device(args.device))
        np.savez(args.feature_cache, ipas=np.array(ipas), feats=feats)
        print(f"Cached {len(ipas)} phones to {args.feature_cache}")

    df = pd.DataFrame({"ipa": ipas, "feat": list(feats)})
    df = df[df.ipa.notna()]
    vocab = df.ipa.unique().tolist()
    pv = PhonologicalVectors(df, vocab=vocab, group_col="ipa")

    print(f"Kept {len(pv.featnames)} features: {pv.featnames}")
    np.savez(
        args.output,
        pos_vecs=pv.pos_vecs.astype(np.float32),
        zero_vecs=pv.zero_vecs.astype(np.float32),
        scales=pv.scales.astype(np.float32),
        biases=pv.biases.astype(np.float32),
        featnames=np.array(pv.featnames),
        model_name=np.array(args.model),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
