#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def convnet(num_conv_filts, num_conv_layers, stride, dilation, max_n_filts=512):
    """
    Implements num_conv_layers conv layers a la VGG.
    """
    layers = []
    in_channels = 1
    n_filts = num_conv_filts
    conv_red_dim = 1  # subsampling factor
    for _ in range(num_conv_layers):
        if len(layers) == 0:
            layers += [nn.Conv2d(in_channels,
                                 n_filts,
                                 kernel_size=(12, 3),
                                 padding=1,
                                 stride=(9, stride),
                                 dilation=dilation)]
        else:
            layers += [nn.Conv2d(in_channels,
                                 n_filts,
                                 kernel_size=3,
                                 padding=1,
                                 stride=stride,
                                 dilation=dilation)]
        layers += [nn.BatchNorm2d(n_filts, affine=True)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(n_filts,
                             n_filts,
                             kernel_size=3,
                             padding=1,
                             stride=stride,
                             dilation=dilation)]
        layers += [nn.BatchNorm2d(n_filts, affine=True)]
        layers += [nn.ReLU()]
        if conv_red_dim <= 32:
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            conv_red_dim *= 2  # max pooled (only correct for frequency dim)
        in_channels = n_filts
        n_filts = 2 * n_filts if n_filts < max_n_filts else n_filts
    return nn.Sequential(*layers), in_channels, conv_red_dim


def coordinates(x, y, start=-1, end=1):
    """
    Returns a map of coordinates with x rows and y columns.
    Input:
    - x: rows
    - y: columns
    Returns:
    - xy_coords: 1 x 2 x 'x' x y
    """
    x_row = torch.linspace(start, end, steps=y)  # y
    y_row = torch.linspace(start, end, steps=x)  # x
    x_coords = x_row.unsqueeze(0).expand(x, y).unsqueeze(0)  # 1 x y
    y_coords = y_row.unsqueeze(1).expand(x, y).unsqueeze(0)  # 1 x y
    # 1 2 x y
    return torch.autograd.Variable(torch.cat([x_coords, y_coords], 0).unsqueeze(0))


class StackedAttention1D(nn.Module):
    """
    Adapted from clevr-iep/blob/master/iep/models/baselines.py
    """
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention1D, self).__init__()
        self.Wa = nn.Linear(input_dim, hidden_dim)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Linear(hidden_dim, input_dim)

    def forward(self, a, u):
        """
        Input:
        - a: N x D
        - u: N x D
        Returns:
        - next_u: N x D
        """
        a_proj = self.Wa(a)  # N x K
        u_proj = self.Wu(u)  # N x K
        h = torch.tanh(a_proj + u_proj)
        p = F.softmax(self.Wp(h), dim=1)  # N x D
        a_tilde = p * a  # N x D
        next_u = a_tilde + u  # N x D
        return next_u


class StackedAttention(nn.Module):
    """
    Adapted from clevr-iep/blob/master/iep/models/baselines.py
    """
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def forward(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D
        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        H, W = v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = torch.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W), dim=1).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum((2, 3))
        next_u = u + v_tilde
        return next_u
