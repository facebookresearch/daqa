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

from layers import convnet, coordinates


class FiLMed(nn.Module):
    """
    Implements a FiLMed block.
    """
    def __init__(self, num_conv_filts_in, num_conv_filts, stride, dilation):
        super(FiLMed, self).__init__()
        self.conv1 = nn.Conv2d(num_conv_filts_in,
                               num_conv_filts,
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               dilation=dilation)
        self.conv2 = nn.Conv2d(num_conv_filts,
                               num_conv_filts,
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(num_conv_filts, affine=False)

    def forward(self, x, gamma, beta):
        b1 = F.relu(self.conv1(x))
        b2 = self.batchnorm2(self.conv2(b1))
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(b2)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(b2)
        b2 = F.relu((b2 * gamma) + beta)
        return (b1 + b2)


class FiLM(nn.Module):
    """
    Implements FiLM.
    """
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 num_conv_filts_base,
                 num_conv_layers_base,
                 stride_base,
                 dilation_base,
                 use_coordinates,
                 num_conv_filts_film,
                 num_conv_layers_film,
                 stride_film,
                 dilation_film,
                 fcn_output_dim,
                 fcn_coeff_dim,
                 fcn_temp_dim,
                 aggregate,
                 output_hidden_dim,
                 output_dim):
        super(FiLM, self).__init__()
        self.bidirectional = bidirectional
        self.use_coordinates = use_coordinates

        self.embeddings = nn.Embedding(vocab_dim,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        self.lstm_q = nn.LSTM(embedding_dim,
                              lstm_hidden_dim_q,
                              num_lstm_layers_q,
                              batch_first=True,
                              bidirectional=bidirectional)
        if bidirectional:
            lstm_output_dim_q = (2 * lstm_hidden_dim_q)
        else:
            lstm_output_dim_q = lstm_hidden_dim_q

        # Compute required output dimension given convnet specs
        # * 2 for gamma and beta. Assumes constant num filters per layer
        num_feats = num_conv_filts_film * num_conv_layers_film * 2
        self.num_conv_filts_film = num_conv_filts_film
        self.num_conv_layers_film = num_conv_layers_film
        self.decoder = nn.Linear(lstm_output_dim_q, num_feats)

        # Base convnet
        self.conv, num_channels, _ = convnet(num_conv_filts_base,
                                             num_conv_layers_base,
                                             stride_base,
                                             dilation_base)

        # Filmed convnet
        self.film_conv_modules = []
        for i in range(num_conv_layers_film):
            num_channels += 2 if use_coordinates else 0
            fcm = FiLMed(num_channels,
                         num_conv_filts_film,
                         stride_film,
                         dilation_film)
            num_channels = num_conv_filts_film
            self.film_conv_modules.append(fcm)
            self.add_module('film_module_%d' % i, fcm)

        num_conv_filts_film += 2 if use_coordinates else 0
        self.conv1 = nn.Conv2d(num_conv_filts_film,
                               fcn_output_dim,
                               kernel_size=1,
                               padding=0)
        if aggregate == 'max':
            self.pool = nn.AdaptiveMaxPool2d((fcn_coeff_dim, fcn_temp_dim))
        elif aggregate == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((fcn_coeff_dim, fcn_temp_dim))
        else:
            assert False, 'Unknown aggregate function.'

        self.use_coordinates_class = (use_coordinates
                                      and fcn_coeff_dim > 1
                                      and fcn_temp_dim > 1)
        fcn_output_dim += 2 if self.use_coordinates_class else 0
        adaptive_pool_dim = fcn_output_dim * fcn_coeff_dim * fcn_temp_dim
        self.output = nn.Sequential(nn.Linear(adaptive_pool_dim, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
        embeddings = self.embeddings(q)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings,
                                                         len_q,
                                                         batch_first=True)
        lstm_q, _ = self.lstm_q(packed)
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_q,
                                                                batch_first=True)
        if self.bidirectional:
            bid_q = unpacked.view(unpacked.size(0),
                                  unpacked.size(1),
                                  2,
                                  int(unpacked.size(2) / 2))
            enc_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
        else:
            enc_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]
        gammas_betas = self.decoder(enc_q)
        gammas_betas = gammas_betas.view(gammas_betas.size(0),
                                         self.num_conv_layers_film,
                                         self.num_conv_filts_film,
                                         2)

        a = torch.unsqueeze(a, 1)
        a = self.conv(a)

        for i, fcm in enumerate(self.film_conv_modules):
            # Append coordinate maps
            if self.use_coordinates:
                coordinates_maps = coordinates(a.shape[2], a.shape[3]).to(a.device)
                a = torch.cat((a, coordinates_maps.expand(a.size(0), -1, -1, -1)), 1)
            # see FiLM appendix for + 1
            a = fcm(a, gammas_betas[:, i, :, 0] + 1, gammas_betas[:, i, :, 1])

        if self.use_coordinates:
            coordinates_maps = coordinates(a.shape[2], a.shape[3]).to(a.device)
            a = torch.cat((a, coordinates_maps.expand(a.size(0), -1, -1, -1)), 1)
        a = self.conv1(a)
        a = self.pool(a)

        if self.use_coordinates_class:
            coordinates_maps = coordinates(a.shape[2], a.shape[3]).to(a.device)
            a = torch.cat((a, coordinates_maps.expand(a.size(0), -1, -1, -1)), 1)
        a = a.view(a.size(0), -1)
        output = self.output(a)
        return output
