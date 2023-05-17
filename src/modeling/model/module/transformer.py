"""
----------------------------------------------------------------------------------------------
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
Modified from FastMETRO (https://github.com/postech-ami/FastMETRO)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see https://github.com/postech-ami/FastMETRO/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see https://github.com/facebookresearch/detr/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

import copy
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor

class Transformer(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, model_dim=512, nhead=8, num_enc_layers=3, num_dec_layers=3, 
                feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features, attention_mask=None):
        hw, bs, _ = img_features.shape

        # Transformer Encoder
        cjv_tokens = self.encoder(cjv_tokens, mask=attention_mask,  memory_key_padding_mask=mask, query_pos=zero_tgt)

        return cjv_tokens


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)


    def forward(self, tgt,
                src_mask: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=src_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_transformer(transformer_config):
    return Transformer(model_dim=transformer_config['model_dim'],
                       dropout=transformer_config['dropout'],
                       nhead=transformer_config['nhead'],
                       feedforward_dim=transformer_config['feedforward_dim'],
                       num_enc_layers=transformer_config['num_enc_layers'],
                       num_dec_layers=transformer_config['num_dec_layers'])