from huggingface_hub import hf_hub_download
import importlib.util

import sys
import inspect
import torch
import numpy as np
import torch.nn.functional as F

def load_module():
    module_path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")
    module_name = "IFMDD_MyEncoderASR"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module from {module_path}")
        
    custom_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = custom_module
    spec.loader.exec_module(custom_module)
    
    return custom_module

def load_model():
    ifmdd_myencoderasr = load_module()
    return ifmdd_myencoderasr.MyEncoderASR.from_hparams(source="Haopeng/CTC_for_IF-MDD", hparams_file="inference.yaml")

class IFMDD:
    def __init__(self, model=None):
        if model is None:
            model = load_model()
        self.model = model

    def load_audio(self, filepath):
        return self.model.load_audio(filepath)

    def transcribe_aligned(self, wav, sr=16000):
        # in case wav isn't already normalized
        wav = self.model.audio_normalizer(wav, sr)

        # The receptive field is 400 and the stride is 320.
        # So if we want frame 0 to correspond to samples [0, 320), we need to pad with 40 samples on each side.
        batch = F.pad(wav.unsqueeze(0), (40, 40))

        rel_length = torch.tensor([1.])
        ctc_p = self.model.encode_batch(batch, rel_length)

        # Have to use custom beam searcher from the model module
        MyCTCPrefixBeamSearcher = inspect.getmodule(self.model).MyCTCPrefixBeamSearcher
        searcher = MyCTCPrefixBeamSearcher(
            tokens=list(dict(sorted(self.model.tokenizer.ind2lab.items())).values()),
            blank_index=self.model.tokenizer.lab2ind["<blank>"],
            sil_index=self.model.tokenizer.lab2ind["sil"]
        )

        hyp = searcher(ctc_p, rel_length)[0][0]

        predicted_tokens = hyp.text
        assert type(predicted_tokens) is list
        timesteps = np.array(hyp.text_frames)
        # for now ifmdd is slightly wrong, so we need to decrease each index by 1
        timesteps[1:] -= 1

        sample_indices = timesteps * 320  # Each frame corresponds to 320 samples

        return predicted_tokens, sample_indices