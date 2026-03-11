#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian

Modified for reintegration experiments:
    - SERClassifier unchanged (utterance-level encoder)
    - SceneGRUWrapper added: wraps SERClassifier with a cross-utterance GRU
      so that hidden state persists across utterances within a scene.
      This is the mechanism through which absence history at t=3,4 shapes
      the model's behaviour at the reintegration boundary t=5.
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


# ─────────────────────────────────────────────────────────────────────────────
# Utterance-level encoder — UNCHANGED from FedMultimodal
# ─────────────────────────────────────────────────────────────────────────────

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
        d_head: int=6
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
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
            self.aux_head = nn.Sequential(
                nn.Linear(d_hid, 64), nn.ReLU(),
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
            self.aux_head = nn.Sequential(
                nn.Linear(d_hid, 64), nn.ReLU(),
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

    def forward(self, x_audio, x_text, len_a, len_t, mask_a=None, mask_b=None, return_aux=False):
        """
        Utterance-level forward pass.
        Returns (preds, x_mm) — the embedding x_mm is consumed by SceneGRUWrapper.
        return_aux=False by default here because the scene wrapper manages
        per-timestep classification; aux_head is not used in the scene pipeline.
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

        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio, len_a.cpu().numpy(), batch_first=True, enforce_sorted=False
            )
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text, len_t.cpu().numpy(), batch_first=True, enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio)
        x_text,  _ = self.text_rnn(x_text)

        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(x_audio, batch_first=True)
        if len_t[0] != 0:
            x_text,  _ = pad_packed_sequence(x_text,  batch_first=True)

        aux_logits = None
        if return_aux:
            masked_audio = (
                x_audio * mask_a_reduced.unsqueeze(-1).float()
                if mask_a_reduced is not None else x_audio
            )
            aux_logits = self.aux_head(masked_audio)

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
                x_mm_input  = torch.cat((x_audio, x_text), dim=1)
                Tb = x_text.shape[1]
                if mask_b is not None:
                    mask_b_reduced = mask_b[:, :Tb]
                    timeb = torch.arange(Tb, device=len_t.device).unsqueeze(0)
                    valid_len_b = timeb < len_t.unsqueeze(1)
                    mask_b_reduced = mask_b_reduced & valid_len_b
                else:
                    mask_b_reduced = None
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

        if return_aux:
            return preds, x_mm, aux_logits
        return preds, x_mm


# ─────────────────────────────────────────────────────────────────────────────
# Scene-level temporal wrapper — NEW
#
# Wraps SERClassifier with a cross-utterance GRU so that hidden state
# persists across utterances t=0..T within a scene.
#
# Data flow:
#   Scene (T utterances) → SERClassifier per utterance → x_mm embeddings
#   → scene_gru (hidden state carries absence history) → per-timestep preds
#
# The reintegration effect is measured at the utterance t where audio returns:
#   h_{t-1} was shaped by absence turns → does pred[t] dip vs stable?
#
# Input contract (from dataloader):
#   x_a_scene:  (T, B_utt, T_frames, D_audio)   — audio frames per utterance
#   x_b_scene:  (T, B_utt, T_tokens, D_text)    — text tokens per utterance
#   len_a_scene:(T, B_utt)
#   len_b_scene:(T, B_utt)
#   mask_scene: (T,) int {0,1}                  — per-utterance audio mask
#   labels:     (T,)                             — per-utterance emotion label
#
# During training B_utt=1 (one scene at a time); batching is across scenes.
# ─────────────────────────────────────────────────────────────────────────────

class SceneGRUWrapper(nn.Module):
    """
    Scene-level temporal model for reintegration experiments.

    Architecture:
        utterance_encoder  : SERClassifier  (intra-utterance features)
        scene_gru          : GRU            (cross-utterance hidden state)
        scene_classifier   : Linear         (per-timestep prediction head)

    The scene_gru hidden state is the mechanism through which availability
    history shapes predictions at the reintegration boundary.
    """

    def __init__(
        self,
        utterance_encoder: SERClassifier,
        num_classes: int,
        d_hid: int = 64,
        scene_gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.utterance_encoder = utterance_encoder
        self.num_classes = num_classes
        self.d_hid = d_hid

        # Infer embedding dim from utterance encoder output.
        # SERClassifier without fuse_base: x_mm = cat(audio_proj, text_proj) → d_hid*2
        # SERClassifier with fuse_base:    x_mm = d_hid * d_head
        # We detect this from the classifier's first Linear in_features.
        first_linear = utterance_encoder.classifier[0]
        self.utt_emb_dim = first_linear.in_features   # e.g. 128 or d_hid*d_head

        # Cross-utterance GRU — hidden state carries absence history
        self.scene_gru = nn.GRU(
            input_size=self.utt_emb_dim,
            hidden_size=d_hid,
            num_layers=scene_gru_layers,
            batch_first=True,       # input: (batch=1, T_scene, utt_emb_dim)
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
        mask_scene,  # (T,) int tensor — 1=audio present, 0=audio absent
        device,
    ):
        """
        Run SERClassifier on each utterance and return stacked embeddings.

        Args:
            mask_scene: per-utterance audio availability mask.
                        When mask_scene[t]==0 the audio is zeroed inside
                        SERClassifier via mask_a, matching training behaviour.

        Returns:
            embeddings: (1, T, utt_emb_dim)  — scene embedding sequence
        """
        embeddings = []
        T = len(x_a_scene)

        for t in range(T):
            x_a = x_a_scene[t].to(device)   # (1, T_frames, D_audio)
            x_b = x_b_scene[t].to(device)   # (1, T_tokens, D_text)
            l_a = len_a_scene[t].to(device)  # (1,)
            l_b = len_b_scene[t].to(device)  # (1,)

            # Build frame-level audio mask for SERClassifier.
            # mask_scene[t]==0 → zero all audio frames for this utterance.
            # Shape required by SERClassifier.downsample_mask_or: (B, T_frames)
            if mask_scene[t].item() == 0:
                # audio absent: zero mask across all frames
                mask_a = torch.zeros(
                    1, x_a.shape[1], device=device, dtype=torch.bool
                )
            else:
                # audio present: ones mask
                mask_a = torch.ones(
                    1, x_a.shape[1], device=device, dtype=torch.bool
                )

            # text always present: no mask_b
            _, x_mm = self.utterance_encoder(
                x_a, x_b, l_a, l_b,
                mask_a=mask_a, mask_b=None,
                return_aux=False,
            )
            embeddings.append(x_mm)         # x_mm: (1, utt_emb_dim)

        # Stack into (1, T, utt_emb_dim) for scene_gru
        embeddings = torch.stack(embeddings, dim=1)   # (1, T, utt_emb_dim)
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

        Returns:
            preds:      (T, num_classes)  — per-utterance logits
            scene_hidden: (T, d_hid)     — scene GRU hidden states (for analysis)
        """
        # 1. Encode each utterance → (1, T, utt_emb_dim)
        embeddings = self.encode_utterances(
            x_a_scene, x_b_scene, len_a_scene, len_b_scene,
            mask_scene, device
        )
        embeddings = self.dropout(embeddings)

        # 2. Scene GRU: hidden state persists across utterances
        #    Output shape: (1, T, d_hid)
        #    h[t] encodes everything seen up to and including utterance t,
        #    including any absence runs leading up to a reintegration event.
        scene_out, _ = self.scene_gru(embeddings)   # (1, T, d_hid)
        scene_out = scene_out.squeeze(0)             # (T, d_hid)

        # 3. Per-timestep classification
        preds = self.scene_classifier(scene_out)     # (T, num_classes)

        return preds, scene_out

    def forward_two_pass(
        self,
        x_a_scene,
        x_b_scene,
        len_a_scene,
        len_b_scene,
        mask_scene,
        device,
    ):
        """
        Two-pass forward for training: stable pass + masked pass on same scene.

        Pass 1 (stable):  all-ones mask  → full (A,T) fusion every utterance
        Pass 2 (masked):  mask_scene     → Markov audio availability

        Returns:
            preds_stable: (T, num_classes)
            preds_masked: (T, num_classes)

        During training, loss is computed over both passes:
            loss = loss_stable + loss_masked
        This ensures the model is trained on both full availability
        and reintegration sequences without splitting the dataset.
        """
        T = len(x_a_scene)

        # Stable pass: audio always present
        ones_mask = torch.ones(T, device=device, dtype=torch.long)
        preds_stable, _ = self.forward(
            x_a_scene, x_b_scene, len_a_scene, len_b_scene,
            ones_mask, device
        )

        # Masked pass: Markov availability
        preds_masked, _ = self.forward(
            x_a_scene, x_b_scene, len_a_scene, len_b_scene,
            mask_scene, device
        )

        return preds_stable, preds_masked


# ─────────────────────────────────────────────────────────────────────────────
# Remaining classes — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

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
        # print('conv1', x, x.shape)
        # pdb.set_trace()
        x = self.dropout(self.pooling(self.relu(self.conv2(x))))
        # print('conv2', x, x.shape)
        x = self.dropout(self.pooling(self.relu(self.conv3(x))))
        # print('conv3', x, x.shape)
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