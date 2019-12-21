#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import re

import h5py
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets  # NOQA F401


class DAQA(torch.utils.data.Dataset):
    _special_ix = {'<pad>': 0}

    def __init__(self, audio_pt, ques_ans_pt, stats=None,
                 word_to_ix=None, answer_to_ix=None):
        # Read audio HDF5 file
        self.audio = h5py.File(audio_pt, 'r')
        if stats is None:
            self.stats = {}
            self.stats['mean'] = self.audio['mean'][:]
            self.stats['stddev'] = self.audio['stddev'][:]
        else:
            self.stats = stats

        # h5py doesnt support using one file handle for multithreaded ops
        # uncomment the following two lines if using >1 worker
        # and ammend __getitem__ accordingly.
        self.audio.close()
        self.audio = audio_pt

        # Read JSON file
        with open(ques_ans_pt, 'r') as f:
            questions_answers = json.load(f)

        # Audio, questions, and answers to a nice list
        dataset = []
        for i in range(len(questions_answers['questions'])):
            aud = questions_answers['questions'][i]['audio_filename'][:-4]
            ques = questions_answers['questions'][i]['question']
            ans = questions_answers['questions'][i]['answer_token']
            dataset.append({'audio': aud, 'question': ques, 'answer': ans})

        if word_to_ix is None:
            self.word_to_ix = DAQA.build_vocab_questions(dataset, DAQA._special_ix)
        else:
            self.word_to_ix = word_to_ix
        dataset = DAQA.encode_questions(dataset, self.word_to_ix)

        if answer_to_ix is None:
            self.answer_to_ix = DAQA.build_vocab_answers(dataset)
        else:
            self.answer_to_ix = answer_to_ix
        dataset = DAQA.encode_answers(dataset, self.answer_to_ix)

        self.dataset = dataset

        # Pack questions and answers for each audio into a nice dictionary.
        dataset_wrt_audio = {}
        for i in range(len(dataset)):
            aud = dataset[i]['audio']
            ques = dataset[i]['question']
            ans = dataset[i]['answer']
            if aud not in dataset_wrt_audio:
                dataset_wrt_audio[aud] = [{'question': ques, 'answer': ans}]
            else:
                dataset_wrt_audio[aud] += [{'question': ques, 'answer': ans}]

        self.dataset_wrt_audio = dataset_wrt_audio

    def __len__(self):
        # return len(self.dataset)
        return len(self.dataset_wrt_audio)

    def __getitem__(self, index):
        sub_mini_batch = []
        audio = sorted(self.dataset_wrt_audio)[index]  # maybe move up
        audio_pt = h5py.File(self.audio, 'r')  # swmr=True
        a = audio_pt[audio][:]
        a = torch.tensor((a - self.stats['mean']) / self.stats['stddev'])
        # The previous 3 lines should be commented if reading audio from memory,
        # as well as audio_pt.close() below.
        # The following line should be uncommented if reading audio from memory.
        # a = torch.tensor((self.audio[audio][:] - self.stats['mean'])
        #                  / self.stats['stddev'])
        len_a = torch.tensor(a.shape[0], dtype=torch.long)
        for qas in range(len(self.dataset_wrt_audio[audio])):
            q = torch.tensor(self.dataset_wrt_audio[audio][qas]['question'],
                             dtype=torch.long)
            len_q = torch.tensor(len(q), dtype=torch.long)
            y = torch.tensor(self.dataset_wrt_audio[audio][qas]['answer'],
                             dtype=torch.long)
            sub_mini_batch += [(a, len_a, q, len_q, y)]
        audio_pt.close()
        return sub_mini_batch

    @staticmethod
    def build_vocab_questions(d, special_ix):
        to_ix = special_ix  # start with special tokens
        for i in range(len(d)):
            # Remove punctuation, lower case, convert to list of words
            qr = re.sub(r'[^\w\s]', '', d[i]['question']).lower().split()
            for w in qr:
                if w not in to_ix:
                    to_ix[w] = len(to_ix)
        return to_ix

    @staticmethod
    def build_vocab_answers(d):
        to_ix = {}
        for i in range(len(d)):
            if d[i]['answer'] not in to_ix:
                to_ix[d[i]['answer']] = len(to_ix)
        return to_ix

    @staticmethod
    def encode_questions(d, to_ix):
        for i in range(len(d)):
            qr = re.sub(r'[^\w\s]', '', d[i]['question']).lower().split()
            d[i]['question'] = [to_ix[w] for w in qr if w in to_ix]
            # if w in to_ix is potentially dangerous
        return d

    @staticmethod
    def encode_answers(d, to_ix):
        for i in range(len(d)):
            d[i]['answer'] = to_ix[d[i]['answer']]
        return d

    @staticmethod
    def pad_collate_fn(batch):
        """
        Input: a list of list((A, len_A, Q, len_Q, Ans)).
        """
        batch = [i for j in batch for i in j]  # unpack list of lists to list
        pad_idx = DAQA._special_ix['<pad>']
        # Sort batch wrt to length of question
        batch = sorted(batch, key=lambda x: x[3], reverse=True)  # sort wrt Q
        max_len_q = batch[0][3]
        # Pad questions with pad_idx
        for i in range(len(batch)):
            x = torch.ones(max_len_q, dtype=batch[i][2].dtype) * pad_idx
            x[:batch[i][2].size(0)] = batch[i][2]
            batch[i] = (batch[i][0], batch[i][1], x, batch[i][3], batch[i][4])
        return default_collate(batch)
