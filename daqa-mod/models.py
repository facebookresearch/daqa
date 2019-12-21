#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from layers import StackedAttention, StackedAttention1D, convnet, coordinates


class LSTMN(nn.Module):
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 input_dim,
                 lstm_hidden_dim_a,
                 num_lstm_layers_a,
                 output_hidden_dim,
                 output_dim):
        super(LSTMN, self).__init__()
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(vocab_dim,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        self.lstm_q = nn.LSTM(embedding_dim,
                              lstm_hidden_dim_q,
                              num_lstm_layers_q,
                              batch_first=True,
                              bidirectional=bidirectional)

        self.lstm_a = nn.LSTM(input_dim,
                              lstm_hidden_dim_a,
                              num_lstm_layers_a,
                              batch_first=True,
                              bidirectional=bidirectional)

        if bidirectional:
            lstm_output_dim_q = (2 * lstm_hidden_dim_q)
            lstm_output_dim_a = (2 * lstm_hidden_dim_a)
        else:
            lstm_output_dim_q = lstm_hidden_dim_q
            lstm_output_dim_a = lstm_hidden_dim_a
        lstm_output_dim = lstm_output_dim_q + lstm_output_dim_a
        self.output = nn.Sequential(nn.Linear(lstm_output_dim, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
        embeddings = self.embeddings(q)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings,
                                                         len_q,
                                                         batch_first=True)
        # self.lstm_q.flatten_parameters()
        lstm_q, _ = self.lstm_q(packed)
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_q,
                                                                batch_first=True)

        # self.lstm_a.flatten_parameters()
        lstm_a, _ = self.lstm_a(a)

        if self.bidirectional:
            bid_q = unpacked.view(unpacked.size(0),
                                  unpacked.size(1),
                                  2,
                                  int(unpacked.size(2) / 2))
            bid_a = lstm_a.view(lstm_a.size(0),
                                lstm_a.size(1),
                                2,
                                int(lstm_a.size(2) / 2))

            cat_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
            cat_a = torch.cat((bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     len_a - 1, 0],
                               bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     len_a - 1, 1]),
                              dim=1)
        else:
            cat_a = lstm_a[torch.arange(lstm_a.size(0), dtype=torch.long), len_a - 1]
            cat_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]

        cat = torch.cat((cat_a, cat_q), 1)
        output = self.output(cat)
        return output


class FCNLSTMN(nn.Module):
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 num_conv_filts,
                 num_conv_layers,
                 stride,
                 dilation,
                 fcn_output_dim,
                 fcn_coeff_dim,
                 fcn_temp_dim,
                 aggregate,
                 output_hidden_dim,
                 output_dim):
        super(FCNLSTMN, self).__init__()
        self.bidirectional = bidirectional

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

        self.conv, num_channels, _ = convnet(num_conv_filts,
                                             num_conv_layers,
                                             stride,
                                             dilation)
        self.conv1 = nn.Conv2d(num_channels,
                               fcn_output_dim,
                               kernel_size=1,
                               padding=0)
        if aggregate == 'max':
            self.pool = nn.AdaptiveMaxPool2d((fcn_coeff_dim, fcn_temp_dim))
        elif aggregate == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((fcn_coeff_dim, fcn_temp_dim))
        else:
            assert False, 'Unknown aggregate function.'

        lstm_output_dim = lstm_output_dim_q \
                          + (fcn_output_dim * fcn_coeff_dim * fcn_temp_dim)
        self.output = nn.Sequential(nn.Linear(lstm_output_dim, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
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
            cat_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
        else:
            cat_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]

        a = torch.unsqueeze(a, 1)
        conv_a = self.conv(a)
        conv1_a = self.conv1(conv_a)
        pool_a = self.pool(conv1_a)
        cat_a = pool_a.view(pool_a.size(0), -1)

        cat = torch.cat((cat_a, cat_q), 1)
        output = self.output(cat)
        return output


class CONVLSTMN(nn.Module):
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 input_dim,
                 num_conv_filts,
                 num_conv_layers,
                 stride,
                 dilation,
                 lstm_hidden_dim_a,
                 num_lstm_layers_a,
                 output_hidden_dim,
                 output_dim):
        super(CONVLSTMN, self).__init__()
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(vocab_dim,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        self.lstm_q = nn.LSTM(embedding_dim,
                              lstm_hidden_dim_q,
                              num_lstm_layers_q,
                              batch_first=True,
                              bidirectional=bidirectional)

        self.conv, num_channels, conv_red_dim = convnet(num_conv_filts,
                                                        num_conv_layers,
                                                        stride,
                                                        dilation)
        self.lstm_a = nn.LSTM(num_channels * int(input_dim / conv_red_dim),
                              lstm_hidden_dim_a,
                              num_lstm_layers_a,
                              batch_first=True,
                              bidirectional=bidirectional)

        if bidirectional:
            lstm_output_dim_q = (2 * lstm_hidden_dim_q)
            lstm_output_dim_a = (2 * lstm_hidden_dim_a)
        else:
            lstm_output_dim_q = lstm_hidden_dim_q
            lstm_output_dim_a = lstm_hidden_dim_a
        lstm_output_dim = lstm_output_dim_q + lstm_output_dim_a

        self.output = nn.Sequential(nn.Linear(lstm_output_dim, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
        embeddings = self.embeddings(q)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings,
                                                         len_q,
                                                         batch_first=True)
        # self.lstm_q.flatten_parameters()
        lstm_q, _ = self.lstm_q(packed)
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_q,
                                                                batch_first=True)

        a = torch.unsqueeze(a, 1)
        a = self.conv(a)
        a = a.permute(0, 2, 1, 3).contiguous()
        a = a.view(a.size(0), a.size(1), a.size(2) * a.size(3))
        lstm_a, _ = self.lstm_a(a)

        if self.bidirectional:
            bid_a = lstm_a.view(lstm_a.size(0),
                                lstm_a.size(1),
                                2,
                                int(lstm_a.size(2) / 2))
            bid_q = unpacked.view(unpacked.size(0),
                                  unpacked.size(1),
                                  2,
                                  int(unpacked.size(2) / 2))

            cat_a = torch.cat((bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 0],
                               bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 1]),
                              dim=1)
            cat_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
        else:
            cat_a = lstm_a[torch.arange(lstm_a.size(0), dtype=torch.long), -1]
            cat_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]

        cat = torch.cat((cat_a, cat_q), 1)
        output = self.output(cat)
        return output


class FCNLSTMNSA(nn.Module):
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 num_conv_filts,
                 num_conv_layers,
                 stride,
                 dilation,
                 fcn_output_dim,
                 fcn_coeff_dim,
                 fcn_temp_dim,
                 aggregate,
                 use_coordinates,
                 stacked_att_dim,
                 num_stacked_att,
                 output_hidden_dim,
                 output_dim):
        super(FCNLSTMNSA, self).__init__()
        self.bidirectional = bidirectional

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

        self.conv, num_channels, _ = convnet(num_conv_filts,
                                             num_conv_layers,
                                             stride,
                                             dilation)
        self.conv1 = nn.Conv2d(num_channels,
                               fcn_output_dim,
                               kernel_size=1,
                               padding=0)
        if aggregate == 'max':
            self.pool = nn.AdaptiveMaxPool2d((fcn_coeff_dim, fcn_temp_dim))
        elif aggregate == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((fcn_coeff_dim, fcn_temp_dim))
        else:
            assert False, 'Unknown aggregate function.'

        self.use_coordinates = (use_coordinates
                                and fcn_coeff_dim > 1
                                and fcn_temp_dim > 1)
        fcn_output_dim += 2 if self.use_coordinates else 0
        self.projection = nn.Conv2d(fcn_output_dim,
                                    lstm_output_dim_q,
                                    kernel_size=1,
                                    padding=0)
        self.stacked_att = []
        for i in range(num_stacked_att):
            sa = StackedAttention(lstm_output_dim_q, stacked_att_dim)
            self.stacked_att.append(sa)
            self.add_module('stacked_att_%d' % i, sa)

        self.output = nn.Sequential(nn.Linear(lstm_output_dim_q, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
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
            cat_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
        else:
            cat_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]

        a = torch.unsqueeze(a, 1)
        conv_a = self.conv(a)
        conv1_a = self.conv1(conv_a)
        pool_a = self.pool(conv1_a)

        if self.use_coordinates:
            coo = coordinates(pool_a.shape[2], pool_a.shape[3]).to(pool_a.device)
            pool_a = torch.cat((pool_a, coo.expand(pool_a.size(0), -1, -1, -1)), 1)

        pool_a = torch.tanh(self.projection(pool_a))
        for sa in self.stacked_att:
            cat_q = sa(pool_a, cat_q)

        output = self.output(cat_q)
        return output


class CONVLSTMNSA(nn.Module):
    def __init__(self,
                 vocab_dim,
                 embedding_dim,
                 padding_idx,
                 lstm_hidden_dim_q,
                 num_lstm_layers_q,
                 bidirectional,
                 input_dim,
                 num_conv_filts,
                 num_conv_layers,
                 stride,
                 dilation,
                 lstm_hidden_dim_a,
                 num_lstm_layers_a,
                 stacked_att_dim,
                 num_stacked_att,
                 output_hidden_dim,
                 output_dim):
        super(CONVLSTMNSA, self).__init__()
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(vocab_dim,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        self.lstm_q = nn.LSTM(embedding_dim,
                              lstm_hidden_dim_q,
                              num_lstm_layers_q,
                              batch_first=True,
                              bidirectional=bidirectional)

        self.conv, num_channels, conv_red_dim = convnet(num_conv_filts,
                                                        num_conv_layers,
                                                        stride,
                                                        dilation)
        self.lstm_a = nn.LSTM(num_channels * int(input_dim / conv_red_dim),
                              lstm_hidden_dim_a,
                              num_lstm_layers_a,
                              batch_first=True,
                              bidirectional=bidirectional)

        if bidirectional:
            lstm_output_dim_q = (2 * lstm_hidden_dim_q)
            lstm_output_dim_a = (2 * lstm_hidden_dim_a)
        else:
            lstm_output_dim_q = lstm_hidden_dim_q
            lstm_output_dim_a = lstm_hidden_dim_a

        self.projection = nn.Linear(lstm_output_dim_a, lstm_output_dim_q)

        self.stacked_att = []
        for i in range(num_stacked_att):
            sa = StackedAttention1D(lstm_output_dim_q, stacked_att_dim)
            self.stacked_att.append(sa)
            self.add_module('stacked_att_%d' % i, sa)

        self.output = nn.Sequential(nn.Linear(lstm_output_dim_q, output_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_hidden_dim, output_dim))

    def forward(self, a, len_a, q, len_q):
        embeddings = self.embeddings(q)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings,
                                                         len_q,
                                                         batch_first=True)
        # self.lstm_q.flatten_parameters()
        lstm_q, _ = self.lstm_q(packed)
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_q,
                                                                batch_first=True)

        a = torch.unsqueeze(a, 1)
        a = self.conv(a)
        a = a.permute(0, 2, 1, 3).contiguous()
        a = a.view(a.size(0), a.size(1), a.size(2) * a.size(3))
        # self.lstm_a.flatten_parameters()
        lstm_a, _ = self.lstm_a(a)

        if self.bidirectional:
            bid_a = lstm_a.view(lstm_a.size(0),
                                lstm_a.size(1),
                                2,
                                int(lstm_a.size(2) / 2))
            bid_q = unpacked.view(unpacked.size(0),
                                  unpacked.size(1),
                                  2,
                                  int(unpacked.size(2) / 2))

            cat_a = torch.cat((bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 0],
                               bid_a[torch.arange(bid_a.size(0), dtype=torch.long),
                                     -1, 1]),
                              dim=1)
            cat_q = torch.cat((bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 0],
                               bid_q[torch.arange(bid_q.size(0), dtype=torch.long),
                                     lens - 1, 1]),
                              dim=1)
        else:
            cat_a = lstm_a[torch.arange(lstm_a.size(0), dtype=torch.long), -1]
            cat_q = unpacked[torch.arange(unpacked.size(0), dtype=torch.long), lens - 1]

        cat_a = torch.tanh(self.projection(cat_a))  # cat_a.size() == cat_q.size()
        for sa in self.stacked_att:
            cat_q = sa(cat_a, cat_q)

        output = self.output(cat_q)
        return output
