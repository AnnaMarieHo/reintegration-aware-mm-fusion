#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

# typing import
from typing import Dict, Iterable, Optional


class SERClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
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

        # Self attention module
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.text_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            self.aux_head = nn.Sequential(
                nn.Linear(d_hid, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.audio_proj = nn.Linear(d_hid, d_hid//2)
            self.text_proj = nn.Linear(d_hid, d_hid//2)
            self.init_weight()

            # classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            self.aux_head = nn.Sequential(
                nn.Linear(d_hid, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS  
    def downsample_mask_or(self, mask: torch.Tensor, factor: int, target_len: int):
        B, T = mask.shape
        pad = (-T) % factor
        if pad:
            mask = torch.cat([mask, torch.zeros(B, pad, device=mask.device, dtype=torch.bool)], dim=1) #type: ignore
        mask = mask.view(B, -1, factor).any(dim=2)      
        return mask[:, :target_len] #type: ignore
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    # def forward(self, x_audio, x_text, len_a, len_t):
    def forward(self, x_audio, x_text, len_a, len_t, mask_a=None, mask_b=None, return_aux=True):
    #------------------------------------------------------------------------------------------------
    
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        
        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a//8
        len_a[len_a==0] = 1
        a_max_len = x_audio.shape[1]
        # Reintegration: compute mask_a_reduced once (availability + valid length), zero audio at OFF before GRU
        mask_a_reduced = None
        if mask_a is not None:
            mask_a_reduced = self.downsample_mask_or(mask_a, 8, a_max_len)
            time = torch.arange(a_max_len, device=x_audio.device).unsqueeze(0)
            valid_len = time < len_a.unsqueeze(1)
            mask_a_reduced = mask_a_reduced & valid_len
            x_audio = x_audio * mask_a_reduced.unsqueeze(-1).float()
        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio, 
                len_a.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text, 
                len_t.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio) 
        x_text, _ = self.text_rnn(x_text)
        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(   
                x_audio,
                batch_first=True
            )
        if len_t[0] != 0:
            x_text, _ = pad_packed_sequence(
                x_text,
                batch_first=True
            )

        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS: aux from masked audio (OFF timesteps zeroed)
        if return_aux:
            masked_audio = x_audio * mask_a_reduced.unsqueeze(-1).float() if mask_a_reduced is not None else x_audio
            aux_logits = self.aux_head(masked_audio)
        else:
            aux_logits = None
        #------------------------------------------------------------------------------------------------
        
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                # x_text = self.text_att(x_text, l_b)
                x_text = self.text_att(x_text, len_t)
            elif self.att_name == "fuse_base":
                # get attention output; mask_a_reduced already computed above (reuse)
                # Use current x_audio time dim so it matches this batch (att segment lengths)
                a_len_this = x_audio.shape[1]
                x_mm_input = torch.cat((x_audio, x_text), dim=1)

                Tb = x_text.shape[1]
                if mask_b is not None:
                    mask_b_reduced = mask_b[:, :Tb]
                    timeb = torch.arange(Tb, device=len_t.device).unsqueeze(0)
                    valid_len_b = timeb < len_t.unsqueeze(1)
                    mask_b_reduced = mask_b_reduced & valid_len_b
                else:
                    mask_b_reduced = None

                x_mm = self.fuse_att(x_mm_input, len_a, len_t, a_len_this, mask_a=mask_a_reduced, mask_b=mask_b_reduced)
                #------------------------------------------------------------------------------------------------
        
        else:
            # 4. Average pooling Projection
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_audio, x_text), dim=1)
        
        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_text = self.text_proj(x_text)
            x_mm = torch.cat((x_audio, x_text), dim=1)
        
        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        if return_aux:
            return preds, x_mm, aux_logits
        else:
            return preds, x_mm
        #------------------------------------------------------------------------------------------------
        # return preds, x_mm



class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x
    
    
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

class AdditiveAttention(nn.Module):
    def __init__(
        self, 
        d_hid:  int=64, 
        d_att:  int=256
    ):
        super().__init__()

        self.query_proj = nn.Linear(d_hid, d_att, bias=False)
        self.key_proj = nn.Linear(d_hid, d_att, bias=False)
        self.bias = nn.Parameter(torch.rand(d_att).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_att, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, 
        query: Tensor,
        key: Tensor, 
        value: Tensor,
        valid_lens: Tensor
    ):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        # attn = F.softmax(score, dim=-1)
        attn = masked_softmax(scores, valid_lens)
        attn = self.dropout(attn)
        output = torch.bmm(attn.unsqueeze(1), value)
        return output
    

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            pdb.set_trace()
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i
    

class BaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(
        self,
        x: Tensor,
        val_l=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x = (att.unsqueeze(2) * x).sum(axis=1)
        return x
    
class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor, #x_mm_input fused audio and text features
        val_a=None, #len_a audio length/time downsampled by a factor of 8
        val_b=None, #len_t text length
        a_len=None, #a_max_len
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        mask_a=None, #mask_a_reduced audio mask downsampled by a factor of 8
        mask_b=None, #mask_b_reduced text mask
        #------------------------------------------------------------------------------------------------
    ):

        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        # Single combined mask per modality (length + availability); avoid double-applying
        dev = att.device
        B = att.shape[0]
        seq_len = att.shape[2]
        Tb = (seq_len - a_len) if a_len is not None else seq_len
        if val_a is not None and a_len is not None:
            time_a = torch.arange(a_len, device=dev).unsqueeze(0).expand(B, -1)
            valid_len_a = time_a < val_a.unsqueeze(1)
            mask_a_slice = mask_a[:, :a_len] if mask_a is not None and mask_a.shape[1] != a_len else mask_a
            combined_mask_a = valid_len_a if mask_a_slice is None else (valid_len_a & mask_a_slice.bool().to(dev))
            att[:, :, :a_len].masked_fill_(~combined_mask_a.unsqueeze(1), -1e5)
        if val_b is not None and a_len is not None and Tb > 0:
            time_b = torch.arange(Tb, device=dev).unsqueeze(0).expand(B, -1)
            valid_len_b = time_b < val_b.unsqueeze(1)
            mask_b_slice = mask_b[:, :Tb] if mask_b is not None and mask_b.shape[1] != Tb else mask_b
            combined_mask_b = valid_len_b if mask_b_slice is None else (valid_len_b & mask_b_slice.bool().to(dev))
            att[:, :, a_len:].masked_fill_(~combined_mask_b.unsqueeze(1), -1e5)
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x