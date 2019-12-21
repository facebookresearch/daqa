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


class FiLM(nn.Module):
    """
    Implements a FiLM block.
    """
    def __init__(self, num_conv_filts_in, num_conv_filts, stride, dilation):
        super(FiLM, self).__init__()
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


class MALiMo(nn.Module):
    """
    Implements MALiMo.
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
                 input_dim,
                 a_aggregate,
                 lstm_hidden_dim_a,
                 num_lstm_layers_a,
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
        super(MALiMo, self).__init__()
        self.bidirectional = bidirectional
        self.use_coordinates = use_coordinates

        # Base convnet
        self.conv, num_channels, freq_red = convnet(num_conv_filts_base,
                                                    num_conv_layers_base,
                                                    stride_base,
                                                    dilation_base)

        # Compute required output dimension given convnet specs
        # * 2 for gamma and beta. Assumes constant num filters per layer
        num_feats = num_conv_filts_film * num_conv_layers_film * 2
        self.num_conv_filts_film = num_conv_filts_film
        self.num_conv_layers_film = num_conv_layers_film

        # Audio Controller
        if a_aggregate == 'max':
            self.a_decoder_pool = nn.MaxPool2d(
                kernel_size=(input_dim // freq_red, 8),
                stride=(input_dim // freq_red, 8))
        elif a_aggregate == 'mean':
            self.a_decoder_pool = nn.MaxPool2d(
                kernel_size=(input_dim // freq_red, 8),
                stride=(input_dim // freq_red, 8))
        else:
            assert False, 'Unknown aggregate function.'
        self.lstm_a = nn.LSTM(num_channels,
                              lstm_hidden_dim_a,
                              num_lstm_layers_a,
                              batch_first=True,
                              bidirectional=bidirectional)
        if bidirectional:
            lstm_output_dim_a = (2 * lstm_hidden_dim_a)
        else:
            lstm_output_dim_a = lstm_hidden_dim_a
        self.audio_decoder = nn.Linear(lstm_output_dim_a, num_feats)

        # Question Controller
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
        self.question_decoder = nn.Linear(lstm_output_dim_q, num_feats)

        # Modulated Layers
        self.a_modulated_modules = []
        self.q_modulated_modules = []
        for i in range(num_conv_layers_film):
            num_channels += 2 if use_coordinates else 0
            afcm = FiLM(num_channels,
                        num_conv_filts_film,
                        stride_film,
                        dilation_film)
            self.a_modulated_modules.append(afcm)
            self.add_module('a_modulated_module_%d' % i, afcm)
            num_channels = num_conv_filts_film
            num_channels += 2 if use_coordinates else 0
            qfcm = FiLM(num_channels,
                        num_conv_filts_film,
                        stride_film,
                        dilation_film)
            self.q_modulated_modules.append(qfcm)
            self.add_module('q_modulated_module_%d' % i, qfcm)
            num_channels = num_conv_filts_film

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

        # Classifier
        self.use_coordinates_class = (use_coordinates
                                      and fcn_coeff_dim > 1
                                      and fcn_temp_dim > 1)
        fcn_output_dim += 2 if self.use_coordinates_class else 0
        adaptive_pool_dim = fcn_output_dim * fcn_coeff_dim * fcn_temp_dim
        self.output = nn.Sequential(nn.Linear(adaptive_pool_dim, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, a, len_a, q, len_q):
        # Base convnet
        a = torch.unsqueeze(a, 1)
        a = self.conv(a)

        # Audio Controller
        pooled_a = self.a_decoder_pool(a)
        pooled_a = torch.transpose(pooled_a, 1, 2)
        pooled_a = pooled_a.view(pooled_a.size(0),
                                 pooled_a.size(1),
                                 pooled_a.size(2) * pooled_a.size(3))
        lstm_a, _ = self.lstm_a(pooled_a)
        if self.bidirectional:
            bid_a = lstm_a.view(lstm_a.size(0),
                                lstm_a.size(1),
                                2,
                                int(lstm_a.size(2) / 2))
            enc_a = torch.cat((bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 0],
                               bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 1]),
                              dim=1)
        else:
            enc_a = lstm_a[torch.arange(lstm_a.size(0), dtype=torch.long), -1]
        a_gammas_betas = self.audio_decoder(enc_a)
        a_gammas_betas = a_gammas_betas.view(a_gammas_betas.size(0),
                                             self.num_conv_layers_film,
                                             self.num_conv_filts_film,
                                             2)

        # Question Controller
        embeddings = self.embeddings(q)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings,
                                                         len_q,
                                                         batch_first=True)
        # self.lstm_q.flatten_parameters()
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
        q_gammas_betas = self.question_decoder(enc_q)
        q_gammas_betas = q_gammas_betas.view(q_gammas_betas.size(0),
                                             self.num_conv_layers_film,
                                             self.num_conv_filts_film,
                                             2)

        # Modulated Layers
        for i, (afcm, qfcm) in enumerate(zip(self.a_modulated_modules,
                                             self.q_modulated_modules)):
            # Append coordinate maps
            if self.use_coordinates:
                coordinates_maps = coordinates(a.shape[2], a.shape[3]).to(a.device)
                a = torch.cat((a, coordinates_maps.expand(a.size(0), -1, -1, -1)), 1)
            # see FiLM appendix for + 1
            a = afcm(a, a_gammas_betas[:, i, :, 0] + 1, a_gammas_betas[:, i, :, 1])
            # Append coordinate maps
            if self.use_coordinates:
                coordinates_maps = coordinates(a.shape[2], a.shape[3]).to(a.device)
                a = torch.cat((a, coordinates_maps.expand(a.size(0), -1, -1, -1)), 1)
            # see FiLM appendix for + 1
            a = qfcm(a, q_gammas_betas[:, i, :, 0] + 1, q_gammas_betas[:, i, :, 1])

        # Classifier
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
