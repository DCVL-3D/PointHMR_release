"""
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
"""

import torch
import numpy as np
from torch import nn
from .module.transformer import *
from .module.position_encoding import build_position_encoding

class MeshRegressor(nn.Module):
    def __init__(self, args, mesh_sampler, model_dim=(512,128), model_name="L"):
        super(MeshRegressor, self).__init__()

        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = 14
        self.num_vertices = 431

        if model_name=="L":
            num_enc_layers = 4

        # build transformers
        encoder_layer1 = TransformerEncoderLayer(model_dim[0], args.transformer_nhead, args.feedforward_dim_1, args.dropout, args.activation)
        encoder_layer2 = TransformerEncoderLayer(model_dim[1], args.transformer_nhead, args.feedforward_dim_2, args.dropout, args.activation)
        encoder_norm1 = nn.LayerNorm(model_dim[0])
        encoder_norm2 = nn.LayerNorm(model_dim[1])
        self.transformer_1 = TransformerEncoder(encoder_layer1, num_enc_layers//2, None)
        self.transformer_2 = TransformerEncoder(encoder_layer1, num_enc_layers//2, encoder_norm1)
        self.transformer_3 = TransformerEncoder(encoder_layer2, num_enc_layers//2, None)
        self.transformer_4 = TransformerEncoder(encoder_layer2, num_enc_layers//2, encoder_norm2)

        # dimensionality reduction
        self.dim_reduce = nn.Linear(model_dim[0], model_dim[1])

        # estimators
        self.xyz_regressor = nn.Linear(model_dim[1], 3)
        self.cam_predictor = nn.Linear(model_dim[1], 3)

        zeros_1 = torch.tensor(np.zeros((self.num_vertices, self.num_joints+1+49)).astype(bool))
        zeros_2 = torch.tensor(np.zeros((self.num_joints+1+49, (1 +49+ self.num_joints + self.num_vertices))).astype(bool))
        adjacency_indices = torch.load('src/modeling/data/smpl_431_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('src/modeling/data/smpl_431_adjmat_values.pt')
        adjacency_matrix_size = torch.load('src/modeling/data/smpl_431_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()

        self.attention_masks = []
        # attention mask
        update_matrix = (adjacency_matrix.clone() > 0)
        ref_matrix = (adjacency_matrix.clone() > 0)

        temp_mask_1 = (update_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1.clone()], dim=1)
        self.attention_masks.append(torch.cat([zeros_2, temp_mask_2], dim=0))

        for n in range(6):
            arr = torch.arange(adjacency_matrix_size[0])
            for i in range(adjacency_matrix_size[0]):
                idx = arr[update_matrix[i] > 0]
                for j in idx:
                    update_matrix[i] += ref_matrix[j]
            if n in [1, 3, 5]:
                temp_mask_1 = (update_matrix == 0)
                temp_mask_2 = torch.cat([zeros_1, temp_mask_1.clone()], dim=1)
                self.attention_masks.append(torch.cat([zeros_2, temp_mask_2], dim=0))

    def forward(self, cjv_token):
        device = cjv_token.device
        batch_size = cjv_token.size(0)

        # first transformer encoder-decoder
        cjv_features_1 = self.transformer_1(cjv_token.transpose(0, 1).contiguous(),mask=self.attention_masks[-1].to(device))
        cjv_features_1 = self.transformer_2(cjv_features_1, mask=self.attention_masks[-2].to(device))

        # progressive dimensionality reduction
        reduced_cjv_features_1 = self.dim_reduce(cjv_features_1)

        # second transformer encoder-decoder
        cjv_features_2 = self.transformer_3(reduced_cjv_features_1, mask=self.attention_masks[-3].to(device))
        cjv_features_2 = self.transformer_4(cjv_features_2, mask=self.attention_masks[-4].to(device))

        cam_features, _, jv_features_2 = cjv_features_2.split([1, 49, cjv_features_2.shape[0] - 50], dim=0)

        # estimators
        cam_parameter = self.cam_predictor(cam_features).view(batch_size, 3)
        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(0, 1).contiguous())
        pred_3d_joints = pred_3d_coordinates[:, :self.num_joints, :]
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.num_joints:, :]

        # coarse-to-fine mesh upsampling  # 431 -> 6890
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine
