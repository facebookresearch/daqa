#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os

import h5py
import numpy as np
import scipy
import scipy.io.wavfile

import librosa

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--input-wavs', default='wavs', type=str,
                    help='Path to folder with wavs to process.')
parser.add_argument('--input-features', default='features', type=str,
                    help='Path to folder with mels to process.')
# Settings
parser.add_argument('--compute-features', action='store_true', default=False,
                    help='Compute features.')
parser.add_argument('--window', default=0.025, type=float,
                    help='Window size (s).')
parser.add_argument('--stride', default=0.01, type=float,
                    help='Window stride (s).')
parser.add_argument('--num-mels', default=64, type=int,
                    help='Number of Mel coefficients.')
parser.add_argument('--astype', default='float32', type=str,
                    help='Data type for storage.')
parser.add_argument('--pack-features', action='store_true', default=False,
                    help='Pack features.')
parser.add_argument('--compressed', action='store_true', default=False,
                    help='Compress features.')

# Output
parser.add_argument('--output-features', default='features', type=str,
                    help='Path to folder with processed features.')
parser.add_argument('--output-file', default='features.hdf5', type=str,
                    help='Path to file with processed features.')


def compute_features(args):
    """
    Compute MFSCs for all audio wav files in a given directory.
    """
    print('Computing features...')
    if not os.path.isdir(args.output_features):
        os.makedirs(args.output_features)
    lst_wavs = os.listdir(args.input_wavs)
    lst_wavs = [e[:-4] for e in lst_wavs if e.endswith('.wav')]
    counter = 0
    for i in lst_wavs:
        try:
            fs, audio = scipy.io.wavfile.read(os.path.join(args.input_wavs,
                                                           i + '.wav'))
            mfsc = librosa.feature.melspectrogram(y=audio.astype(float),
                                                  sr=fs,
                                                  n_fft=int(fs * args.window),
                                                  n_mels=args.num_mels,
                                                  hop_length=int(fs * args.stride),
                                                  power=1)
            mfsc = librosa.power_to_db(mfsc, ref=np.max).T.astype(args.astype)
            np.save(os.path.join(args.output_features, i), mfsc)
        except Exception:
            print('Error processing: ' + str(i))
        counter += 1
        if counter % 1000 == 0:
            print('Finished processing: ' + str(counter) + ' files.')


def pack_features(args):
    """
    Pack all npy MFSCs in a given directory into a single hdf file.
    """
    print('Packing features...')
    lst_npys = os.listdir(args.input_features)
    lst_npys = [e[:-4] for e in lst_npys if e.endswith('.npy')]
    counter = 0
    # Variables for Welford’s mean and variance
    n, mean, v = 0, np.zeros(args.num_mels), np.zeros(args.num_mels)
    kwargs = {'compression': 'gzip', 'compression_opts': 9} if args.compressed else {}

    with h5py.File(args.output_file, 'w') as f:
        for i in lst_npys:
            mfsc = np.load(os.path.join(args.output_features, i + '.npy'))
            f.create_dataset(i, data=mfsc, dtype=args.astype,
                             **kwargs)

            for w in range(mfsc.shape[0]):
                n += 1
                delta = mfsc[w] - mean
                mean += delta / n
                v += (mfsc[w] - mean) * delta

            counter += 1
            if counter % 1000 == 0:
                print('Finished packing: ' + str(counter) + ' files.')

        var = v / (n - 1)
        stddev = np.sqrt(var)

        f.create_dataset('mean',
                         data=mean.astype(args.astype),
                         dtype=args.astype,
                         **kwargs)
        f.create_dataset('variance',
                         data=var.astype(args.astype),
                         dtype=args.astype,
                         **kwargs)
        f.create_dataset('stddev',
                         data=stddev.astype(args.astype),
                         dtype=args.astype,
                         **kwargs)


def main(args):
    if args.compute_features:
        compute_features(args)
    if args.pack_features:
        pack_features(args)
    if not args.compute_features and not args.pack_features:
        print('P.S. I didnt do anything. Both compute and pack features are false.')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print('Success!')
