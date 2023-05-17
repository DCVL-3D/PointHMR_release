"""
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.modeling.backbone.hrnet_32 import HigherResolutionNet
from src.modeling.model.regressor import MeshRegressor
from src.modeling.model.module.basic_modules import BasicBlock
from src.modeling.model.module.position_encoding import build_position_encoding

import torchvision.transforms as transforms
BN_MOMENTUM = 0.1

class PointHMR(torch.nn.Module):
    def __init__(self, args, mesh_sampler):
        super(PointHMR, self).__init__()

        self.num_joints = 14
        self.num_vertices = 431

        self.build_head(args)
        self.pos_embedding = build_position_encoding(hidden_dim=args.model_dim)

        self.cam_token_embed = nn.Embedding(1, args.model_dim)
        self.joint_token_embed = nn.Embedding(self.num_joints, args.model_dim)
        self.vertex_token_embed = nn.Embedding(self.num_vertices, args.model_dim)

        self.token_position_embed = nn.Embedding(1+49+self.num_joints+self.num_vertices, args.position_dim)

        self.transformer = MeshRegressor(args, mesh_sampler, model_dim=(args.model_dim+args.position_dim,128))

    def build_head(self, args):
        self.backbone = HigherResolutionNet(args) # 128, 56, 56
        self.inplanes = self.backbone.backbone_channels

        self.feature_extract_layer = self._make_resnet_layer(BasicBlock, args.model_dim, 2)
        self.heatmap_layer = self._make_resnet_layer(BasicBlock, self.num_vertices, 2)
        self.inplanes = self.backbone.backbone_channels*8
        self.grid_layer = self._make_resnet_layer(BasicBlock, args.model_dim, 1)
        self.cam_layer = nn.Linear(431,1)

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )  # ,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        layers.append(nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, images):
        x, x2 = self.backbone(images)

        batch, c, h, w = x.shape
        device = x.device

        joint_token = self.joint_token_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        position_embedding = self.token_position_embed.weight.unsqueeze(0).repeat(batch, 1, 1)

        heatmap_ = self.heatmap_layer(x)
        heatmap = heatmap_.clone()
        heatmap = F.softmax(heatmap.flatten(2), dim=-1)

        heatmap_ = F.interpolate(heatmap_, scale_factor=2)
        heatmap_ = F.softmax(heatmap_.flatten(2), dim=-1)

        feature_map = self.feature_extract_layer(x).flatten(2)  # BXCXHW
        feature_map += self.pos_embedding(batch, h, w, device).flatten(2)

        grid_feature = self.grid_layer(x2).flatten(2)
        grid_feature += self.pos_embedding(batch, h // 8, w // 8, device).flatten(2)

        sampled_feature = heatmap @ feature_map.transpose(1, 2).contiguous()  # BX431XC
        cam_feature = self.cam_layer(sampled_feature.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # BX1XC

        input_feature = torch.cat(
            [cam_feature, grid_feature.transpose(1, 2).contiguous(), joint_token, sampled_feature], 1)  # BX456XC
        input_feature = torch.cat([input_feature, position_embedding], 2)  # BX456XC+P

        cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine = self.transformer(
            input_feature)  # BX456XC+P

        return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine, heatmap_

