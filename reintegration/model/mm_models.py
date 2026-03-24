#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian

Modified for reintegration experiments — Phase 1.

    SERClassifier  : utterance-level encoder, unchanged from FedMultimodal.
    SceneGRUWrapper: wraps SERClassifier with a cross-utterance GRU so that
                     hidden state persists across utterances within a scene.

Phase 1 only: the model is trained on stable (full-audio) scenes.
At test time, run_reintegration_eval() calls forward() twice — once with an
all-ones mask and once with the Markov mask — to measure the reintegration dip.
forward_two_pass and all auxiliary-loss infrastructure have been removed.
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from typing import Dict, Iterable, Optional


'''
Utterance-level encoder — UNCHANGED from FedMultimodal
'''

class SERClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        audio_input_dim: int,
        text_input_dim: int,
        d_hid: int=64,
        n_filters: int=32,
        en_att: bool=False,
        att_name: str='',
        d_head: int=6,
        audio_only: bool=False,
        text_only: bool=False,
    ):
        super(SERClassifier, self).__init__()
        assert not (audio_only and text_only), "audio_only and text_only are mutually exclusive"
        self.dropout_p = 0.1
        self.en_att    = en_att
        self.att_name  = att_name
        self.audio_only = audio_only
        self.text_only  = text_only

        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        self.audio_rnn = nn.GRU(
            input_size=n_filters*4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        self.text_rnn = nn.GRU(
            input_size=text_input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, num_heads=4, dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, num_heads=4, dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(d_hid=d_hid)
            self.text_att  = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(d_hid=d_hid, d_head=d_head)

        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64), nn.ReLU(),
                nn.Dropout(self.dropout_p), nn.Linear(64, num_classes)
            )
        else:
            self.audio_proj = nn.Linear(d_hid, d_hid//2)
            self.text_proj  = nn.Linear(d_hid, d_hid//2)
            self.init_weight()
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64), nn.ReLU(),
                nn.Dropout(self.dropout_p), nn.Linear(64, num_classes)
            )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def downsample_mask_or(self, mask: torch.Tensor, factor: int, target_len: int):
        B, T = mask.shape
        pad = (-T) % factor
        if pad:
            mask = torch.cat(
                [mask, torch.zeros(B, pad, device=mask.device, dtype=torch.bool)], dim=1
            )
        mask = mask.view(B, -1, factor).any(dim=2)
        return mask[:, :target_len]

    def forward(self, x_audio, x_text, len_a, len_t, mask_a=None, mask_b=None):
        """
        Utterance-level forward pass.
        Returns (preds, x_mm) — x_mm is consumed by SceneGRUWrapper.
        """
        x_audio = self.audio_conv(x_audio)

        len_a = len_a // 8
        len_a[len_a == 0] = 1
        a_max_len = x_audio.shape[1]

        mask_a_reduced = None
        if mask_a is not None:
            mask_a_reduced = self.downsample_mask_or(mask_a, 8, a_max_len)
            time = torch.arange(a_max_len, device=x_audio.device).unsqueeze(0)
            valid_len = time < len_a.unsqueeze(1)
            mask_a_reduced = mask_a_reduced & valid_len
            x_audio = x_audio * mask_a_reduced.unsqueeze(-1).float()

        # Text-level availability mask (symmetric to audio masking above).
        # mask_b[t]==False → zero out all tokens for utterance t before the RNN.
        mask_b_reduced = None
        if mask_b is not None:
            t_max_len = x_text.shape[1]
            mask_b_reduced = mask_b[:, :t_max_len]
            time_b = torch.arange(t_max_len, device=x_text.device).unsqueeze(0)
            len_t_clamped = len_t.clone()
            len_t_clamped[len_t_clamped == 0] = 1
            valid_len_b = time_b < len_t_clamped.unsqueeze(1)
            mask_b_reduced = mask_b_reduced & valid_len_b
            x_text = x_text * mask_b_reduced.unsqueeze(-1).float()

        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio, len_a.cpu().numpy(), batch_first=True, enforce_sorted=False
            )
        if len_t[0] != 0:
            len_t = len_t.clone()
            len_t[len_t == 0] = 1
            x_text = pack_padded_sequence(
                x_text, len_t.cpu().numpy(), batch_first=True, enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio)
        x_text,  _ = self.text_rnn(x_text)

        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(x_audio, batch_first=True)
        if len_t[0] != 0:
            x_text,  _ = pad_packed_sequence(x_text,  batch_first=True)

        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text,  _ = self.text_att(x_text,  x_text,  x_text)
                x_audio = torch.mean(x_audio, axis=1)
                x_text  = torch.mean(x_text,  axis=1)
            elif self.att_name == 'base':
                x_audio = self.audio_att(x_audio)
                x_text  = self.text_att(x_text, len_t)
            elif self.att_name == "fuse_base":
                a_len_this  = x_audio.shape[1]

                Tb = x_text.shape[1]
                if mask_b is not None:
                    mask_b_reduced = mask_b[:, :Tb]
                    timeb = torch.arange(Tb, device=len_t.device).unsqueeze(0)
                    valid_len_b = timeb < len_t.unsqueeze(1)
                    mask_b_reduced = mask_b_reduced & valid_len_b
                else:
                    mask_b_reduced = None

                if self.audio_only:
                    x_text = torch.zeros_like(x_text)
                    if mask_b_reduced is not None:
                        mask_b_reduced = torch.zeros_like(mask_b_reduced, dtype=torch.bool)

                if self.text_only:
                    x_audio = torch.zeros_like(x_audio)
                    if mask_a_reduced is not None:
                        mask_a_reduced = torch.zeros_like(mask_a_reduced, dtype=torch.bool)

                x_mm_input  = torch.cat((x_audio, x_text), dim=1)

                x_mm = self.fuse_att(
                    x_mm_input, len_a, len_t, a_len_this,
                    mask_a=mask_a_reduced, mask_b=mask_b_reduced
                )
        else:
            x_audio = torch.mean(x_audio, axis=1)
            x_text  = torch.mean(x_text,  axis=1)
            x_mm    = torch.cat((x_audio, x_text), dim=1)

        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_text  = self.text_proj(x_text)
            x_mm    = torch.cat((x_audio, x_text), dim=1)

        preds = self.classifier(x_mm)
        return preds, x_mm


"""
    Scene-level temporal wrapper

    Wraps SERClassifier with a cross-utterance GRU so that hidden state
    persists across utterances t=0..T within a scene.

    Data flow:
    Scene (T utterances) → SERClassifier per utterance → x_mm embeddings
    → scene_gru (hidden state carries absence history) → per-timestep preds

    The reintegration effect is measured at the utterance t where audio returns:
    h_{t-1} was shaped by absence turns → does pred[t] dip vs stable?

    only forward() is used. The model is trained on stable scenes
    (all audio present). At test time, run_reintegration_eval() calls forward()
    twice on each test scene — once with an all-ones mask (stable condition)
    and once with the Markov mask (reintegration condition) — and records the
    per-timestep delta at t_reint.
"""


class SceneGRUWrapper(nn.Module):
    """
    Scene-level temporal model for reintegration experiments (Phase 1).

    Architecture:
        utterance_encoder : SERClassifier  (intra-utterance features)
        scene_gru         : GRU            (cross-utterance hidden state)
        scene_classifier  : Linear         (per-timestep prediction head)

    The scene_gru hidden state is the mechanism through which availability
    history shapes predictions at the reintegration boundary. Because the
    model is never trained on absent audio, h_{t-1} after an absence run
    carries a degraded representation that produces the dip at t_reint.
    """

    def __init__(
        self,
        utterance_encoder: SERClassifier,
        num_classes: int,
        d_hid: int = 64,
        scene_gru_layers: int = 1,
        dropout: float = 0.1,
        mask_modality: str = "audio",
    ):
        super().__init__()

        assert mask_modality in ("audio", "text"), (
            f"mask_modality must be 'audio' or 'text', got '{mask_modality}'"
        )

        self.utterance_encoder = utterance_encoder
        self.num_classes = num_classes
        self.d_hid = d_hid
        self.mask_modality = mask_modality

        # Infer embedding dim from utterance encoder output.
        # Without fuse_base: x_mm = cat(audio_proj, text_proj) → d_hid*2
        # With fuse_base:    x_mm = d_hid * d_head
        first_linear = utterance_encoder.classifier[0]
        self.utt_emb_dim = first_linear.in_features

        # Cross-utterance GRU — hidden state carries absence history
        self.scene_gru = nn.GRU(
            input_size=self.utt_emb_dim,
            hidden_size=d_hid,
            num_layers=scene_gru_layers,
            batch_first=True,
            dropout=dropout if scene_gru_layers > 1 else 0.0,
            bidirectional=False,    # causal: must not look ahead
        )

        # Per-timestep classification head
        self.scene_classifier = nn.Linear(d_hid, num_classes)

        self.dropout = nn.Dropout(dropout)

    def encode_utterances(
        self,
        x_a_scene,   # list of T tensors, each (1, T_frames, D_audio)
        x_b_scene,   # list of T tensors, each (1, T_tokens, D_text)
        len_a_scene, # list of T tensors, each (1,)
        len_b_scene, # list of T tensors, each (1,)
        mask_scene,  # (T,) int tensor — 1=modality present, 0=modality absent
        device,
    ):
        """
        Run SERClassifier on all T utterances in a single batched forward pass.

        Which modality is masked is controlled by self.mask_modality:
            "audio" → mask_scene gates audio availability (text always present)
            "text"  → mask_scene gates text availability  (audio always present)

        Returns:
            embeddings: (1, T, utt_emb_dim)
        """
        T = len(x_a_scene)

        len_a = torch.cat([len_a_scene[t] for t in range(T)], dim=0).to(device)  # (T,)
        len_b = torch.cat([len_b_scene[t] for t in range(T)], dim=0).to(device)  # (T,)

        # Pad audio to (T, max_T_frames, D_audio)
        max_a = max(x_a_scene[t].shape[1] for t in range(T))
        D_a   = x_a_scene[0].shape[2]
        x_a_batch = torch.zeros(T, max_a, D_a, device=device, dtype=torch.float32)
        for t in range(T):
            fa = x_a_scene[t].squeeze(0).to(device)
            x_a_batch[t, :fa.shape[0], :] = fa

        # Pad text to (T, max_T_tokens, D_text)
        max_b = max(x_b_scene[t].shape[1] for t in range(T))
        D_b   = x_b_scene[0].shape[2]
        x_b_batch = torch.zeros(T, max_b, D_b, device=device, dtype=torch.float32)
        for t in range(T):
            fb = x_b_scene[t].squeeze(0).to(device)
            x_b_batch[t, :fb.shape[0], :] = fb

        # Build per-frame availability mask for the target modality.
        # The non-target modality gets None (= all present).
        if self.mask_modality == "audio":
            mask_a = torch.zeros(T, max_a, device=device, dtype=torch.bool)
            for t in range(T):
                if mask_scene[t].item() == 1:
                    fa_len = x_a_scene[t].shape[1]
                    mask_a[t, :fa_len] = True
            mask_b = None
        else:
            mask_a = None
            mask_b = torch.zeros(T, max_b, device=device, dtype=torch.bool)
            for t in range(T):
                if mask_scene[t].item() == 1:
                    fb_len = x_b_scene[t].shape[1]
                    mask_b[t, :fb_len] = True

        _, x_mm_batch = self.utterance_encoder(
            x_a_batch, x_b_batch, len_a, len_b,
            mask_a=mask_a, mask_b=mask_b,
        )
        # x_mm_batch: (T, utt_emb_dim)

        embeddings = x_mm_batch.unsqueeze(0)   # (1, T, utt_emb_dim)
        return embeddings

    def forward(
        self,
        x_a_scene,
        x_b_scene,
        len_a_scene,
        len_b_scene,
        mask_scene,
        device,
    ):
        """
        Full scene forward pass.

        Called during training (stable mask) and by run_reintegration_eval
        (stable mask and Markov mask on the same scene).

        Returns:
            preds        : (T, num_classes) — per-utterance logits
            scene_hidden : (T, d_hid)       — GRU hidden states (for analysis)
        """
        embeddings = self.encode_utterances(
            x_a_scene, x_b_scene, len_a_scene, len_b_scene,
            mask_scene, device
        )
        embeddings = self.dropout(embeddings)

        scene_out, _ = self.scene_gru(embeddings)   # (1, T, d_hid)
        scene_out     = scene_out.squeeze(0)         # (T, d_hid)
        preds         = self.scene_classifier(scene_out)  # (T, num_classes)

        return preds, scene_out


"""
Remaining classes — UNCHANGED
"""

class Conv1dEncoder(nn.Module):
    def __init__(self, input_dim: int, n_filters: int, dropout: float=0.1):
        super().__init__()
        self.conv1   = nn.Conv1d(input_dim,     n_filters,   kernel_size=5, padding=2)
        self.conv2   = nn.Conv1d(n_filters,     n_filters*2, kernel_size=5, padding=2)
        self.conv3   = nn.Conv1d(n_filters*2,   n_filters*4, kernel_size=5, padding=2)
        self.relu    = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.dropout(self.pooling(self.relu(self.conv1(x))))
        x = self.dropout(self.pooling(self.relu(self.conv2(x))))
        x = self.dropout(self.pooling(self.relu(self.conv3(x))))
        x = x.permute(0, 2, 1)
        return x


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class BaseSelfAttention(nn.Module):
    def __init__(self, d_hid: int=64):
        super().__init__()
        self.att_fc1  = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2  = nn.Linear(1, 1)

    def forward(self, x: Tensor, val_l=None):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x   = (att.unsqueeze(2) * x).sum(axis=1)
        return x


class FuseBaseSelfAttention(nn.Module):
    def __init__(self, d_hid: int=64, d_head: int=4):
        super().__init__()
        self.att_fc1  = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2  = nn.Linear(512, d_head)
        self.d_hid    = d_hid
        self.d_head   = d_head

    def forward(self, x: Tensor, val_a=None, val_b=None, a_len=None,
                mask_a=None, mask_b=None):
        att     = self.att_pool(self.att_fc1(x))
        att     = self.att_fc2(att)
        att     = att.transpose(1, 2)
        dev     = att.device
        B       = att.shape[0]
        seq_len = att.shape[2]
        Tb      = (seq_len - a_len) if a_len is not None else seq_len

        if val_a is not None and a_len is not None:
            time_a        = torch.arange(a_len, device=dev).unsqueeze(0).expand(B, -1)
            valid_len_a   = time_a < val_a.unsqueeze(1)
            mask_a_slice  = mask_a[:, :a_len] if (mask_a is not None and mask_a.shape[1] != a_len) else mask_a
            combined_mask_a = valid_len_a if mask_a_slice is None else (valid_len_a & mask_a_slice.bool().to(dev))
            att[:, :, :a_len].masked_fill_(~combined_mask_a.unsqueeze(1), -1e5)

        if val_b is not None and a_len is not None and Tb > 0:
            time_b        = torch.arange(Tb, device=dev).unsqueeze(0).expand(B, -1)
            valid_len_b   = time_b < val_b.unsqueeze(1)
            mask_b_slice  = mask_b[:, :Tb] if (mask_b is not None and mask_b.shape[1] != Tb) else mask_b
            combined_mask_b = valid_len_b if mask_b_slice is None else (valid_len_b & mask_b_slice.bool().to(dev))
            att[:, :, a_len:].masked_fill_(~combined_mask_b.unsqueeze(1), -1e5)

        att = torch.softmax(att, dim=2)
        x   = torch.matmul(att, x)
        x   = x.reshape(x.shape[0], self.d_head * self.d_hid)
        return x