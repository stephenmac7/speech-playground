# Copied from McGhee's repository
# TODO: Release separately and import as a package
# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
from dataclasses import dataclass
import os
import copy
from pathlib import Path
import torch
import argparse
import numpy as np
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2
import transformers.models.wavlm.modeling_wavlm as wavlm

from functools import lru_cache
from torchaudio.compliance import kaldi

from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2Processor,
    AutoProcessor,
    WavLMModel,
    AutoFeatureExtractor,
)
from torch.nn.utils import weight_norm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if self.config.finetune_method == "lora":
            self.feed_forward.intermediate_dense = lora.Linear(
                config.hidden_size,
                config.intermediate_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.feed_forward.output_dense = lora.Linear(
                config.intermediate_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.k_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.q_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.v_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.out_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        index=0,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # Adapter
        outputs = (hidden_states, position_bias)

        return outputs


class WavLMWrapper(nn.Module):
    def __init__(self, args, hidden_dim=256, output_class_num=4):
        super(WavLMWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.args = args
        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large", output_hidden_states=True
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = args.finetune_method
        self.model_config.lora_rank = args.lora_rank
        self.model_config.lora_alpha = args.lora_alpha

        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [
                WavLMEncoderLayer(self.model_config, has_relative_position_bias=(i == 0))
                for i in range(self.model_config.num_hidden_layers)
            ]
        )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if (
            self.args.finetune_method == "adapter"
            or self.args.finetune_method == "adapter_l"
            or self.args.finetune_method == "embedding_prompt"
            or self.args.finetune_method == "finetune"
            or self.args.finetune_method == "lora"
            or self.args.finetune_method == "combined"
        ):
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        # keep_list = [0,23]
        ##
        # new_list = []

        # for i in range(self.model_config.num_hidden_layers):
        #    if i in keep_list:
        #        new_list.append(self.backbone_model.encoder.layers[i])

        # self.backbone_model.encoder.layers = nn.ModuleList(new_list)

        # self.finetune_method = self.args.finetune_method

        # self.upsample = weight_norm(torch.nn.ConvTranspose1d(self.model_config.hidden_size,
        #                                    self.model_config.hidden_size,
        #                                    4, 2, padding=1))

        # self.relu = nn.ReLU()
        # self.conv1 = nn.Conv1d(self.model_config.hidden_size,12,4,padding=2)
        self.regression_head = nn.Linear(self.model_config.hidden_size, 36)

    def forward(self, x, inp_mask=None, length=None):
        # 1. feature extraction and projections
        if inp_mask != None:
            x = self.backbone_model(x, attention_mask=inp_mask).last_hidden_state
        else:
            x = self.backbone_model(x).last_hidden_state

        # x = rearrange(x, "N T C -> N C T")
        #
        # x = self.upsample(x)
        #
        # x = rearrange(x, "N C T -> N T C")

        predicted = self.regression_head(x)

        return predicted

# @dataclass
# class WavLMWrapperArgs:
#     finetune_method: str
#     lora_rank: int
#     lora_alpha: int
# 
# args = WavLMWrapperArgs('lora', 4, 4)

class WavLMBPWrapper(nn.Module):
    def __init__(
        self, 
    ):
        super(WavLMBPWrapper, self).__init__()
        
        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus",
            output_hidden_states=True
        )
        self.model_config = self.backbone_model.config

        self.regression_head = nn.Linear(self.model_config.hidden_size,14)
        
    def forward(self, x,inp_mask=None, length=None):
        if inp_mask!=None:
            x = self.backbone_model(x,attention_mask=inp_mask).last_hidden_state
        else:
            x = self.backbone_model(x).last_hidden_state
         
        predicted = self.regression_head(x)

        return predicted

class ArticulatoryInversionEncoder:
    def __init__(self, *, weights: Path, mu_path: Path, std_path: Path, device: Optional[torch.device] = "cuda"):
        self.model = WavLMBPWrapper()
        state_dict = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        self.mu = np.expand_dims(np.load(mu_path), 0)
        self.std = np.expand_dims(np.load(std_path), 0)
        self.device = device

    def encode_one(self, waveform: torch.Tensor) -> np.ndarray:
        assert waveform.ndim == 1, "Input waveform must be 1D (samples,)"
        with torch.no_grad():
            encoded = self.model(
                waveform.unsqueeze(0).to(self.device)
            ).squeeze().detach().cpu().numpy()
        return encoded

    def encode(self, waveforms: torch.Tensor) -> np.ndarray:
        raise NotImplementedError("Batch encoding not implemented for ArticulatoryInversionEncoder")

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_shift(self) -> float:
        return 0.02 # Sample as WavLM



def animate_two_scatter(
    seq1: np.ndarray,
    seq2: np.ndarray,
    fps: int = 30,
    point_size: int = 80,
    margin: float = 0.05,
    save_path: str | None = None,
):
    """
    Animate two normalized (s*std)+mu sequences of shape (L, 12).
    Each frame has 6 (x,y) pairs: columns [0,1], [2,3], ..., [10,11].
    """

    assert seq1.shape[1] == 12 and seq2.shape[1] == 12, "Shape must be (L, 12)."

    # Reshape to (L, 6, 2)
    P1 = seq1.reshape(seq1.shape[0], 6, 2)
    P2 = seq2.reshape(seq2.shape[0], 6, 2)

    # Sync lengths (stop at min length)
    T = min(P1.shape[0], P2.shape[0])
    P1, P2 = P1[:T], P2[:T]

    #Axis limits
    allx = np.concatenate([P1[...,0].ravel(), P2[...,0].ravel()])
    ally = np.concatenate([P1[...,1].ravel(), P2[...,1].ravel()])
    xmin, xmax = allx.min(), allx.max()
    ymin, ymax = ally.min(), ally.max()
    xr = xmax - xmin or 1.0
    yr = ymax - ymin or 1.0
    xmin -= xr * margin; xmax += xr * margin
    ymin -= yr * margin; ymax += yr * margin

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    #if not show_axes:
    #    ax.axis("off")

    # Initial scatters
    scat1 = ax.scatter(P1[0,:,0], P1[0,:,1], s=point_size, c="#E69F00", label="Learner",marker='x')
    scat2 = ax.scatter(P2[0,:,0], P2[0,:,1], s=point_size, c="#009E73", label="Template")
    ax.legend(loc="lower left")

    def update(f):
        scat1.set_offsets(P1[f])
        scat2.set_offsets(P2[f])
        return scat1, scat2
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False, repeat=True)

    if save_path is not None:
        ext = save_path.lower().split(".")[-1]
        if ext == "gif":
            anim.save(save_path, fps=fps, writer="pillow")
        else:
            anim.save(save_path, fps=fps)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.show()
    return anim