"""
----------------------------------------------------------------------------------------------
MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
"""

from __future__ import division
import torch
import torch.nn.functional as F
import math

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body'):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)

    def forward(self, x):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)

        z = x+y

        return z

def laplacian_decomposition(x):
    down2 = F.interpolate(x, scale_factor=1 / 2, mode="linear")
    down4 = F.interpolate(down2, scale_factor=1 / 2, mode="linear")

    up2 = F.interpolate(down4, scale_factor=2, mode="linear")
    up1 = F.interpolate(down2, scale_factor=2, mode="linear")

    lap1 = x - up1
    lap2 = down2 - up2
    lap3 = down4
    return lap1, lap2, lap3


def laplacian_composition(lap1, lap2, lap3):
    up2 = F.interpolate(lap3, scale_factor=2, mode="linear") + lap2
    up1 = F.interpolate(up2, scale_factor=2, mode="linear") + lap1
    return up1

class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True):
        super(GraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features

        if mesh=='body':
            adj_indices = torch.load('./src/modeling/data/smpl_431_adjmat_indices.pt')
            adj_mat_value = torch.load('./src/modeling/data/smpl_431_adjmat_values.pt')
            adj_mat_size = torch.load('./src/modeling/data/smpl_431_adjmat_size.pt')
        elif mesh=='hand':
            adj_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
            adj_mat_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
            adj_mat_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')

        self.adjmat = torch.sparse_coo_tensor(adj_indices, adj_mat_value, size=adj_mat_size).to(device)

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(self.adjmat, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'