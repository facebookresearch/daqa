# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import json
import os
import random
import numpy as np
import scipy
import scipy.io.wavfile

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--dataset', default='daqa.json', type=str,
                    help='JSON file describing the dataset.')
parser.add_argument('--events', default='events', type=str,
                    help='Location of individual audio events.')
parser.add_argument('--backgrounds', default='backgrounds', type=str,
                    help='Location of some background noise audio.')
parser.add_argument('--data_fs', default=16000, type=int,
                    help='Sampling frequency (Hz).')

# Settings
parser.add_argument('--min_num_events', default=5, type=int,
                    help='Minimum number of events per generated audio.')
parser.add_argument('--max_num_events', default=12, type=int,
                    help='Maximum number of events per generated audio.')
parser.add_argument('--rand_overlap', default=0.5, type=float,
                    help='Maximum overlap between adjacent events (seconds).')
parser.add_argument('--seed', default=0, type=int, help='Random Seed.')
parser.add_argument('--version', default='1.0', type=str, help='Version.')
parser.add_argument('--date',
                    default=datetime.datetime.today().strftime("%m/%d/%Y"),
                    help="Date.")
parser.add_argument('--license',
                    default='Creative Commons Attribution (CC-BY 4.0)',
                    help='License.')

# Output
parser.add_argument('--start_idx', default=0, type=int,
                    help='Start numbering from start_idx.')
parser.add_argument('--num_audio', default=10, type=int,
                    help='Number of audio to generate.')
parser.add_argument('--filename_prefix', default='daqa', type=str,
                    help='Filename prefix to audio and JSON files.')
parser.add_argument('--set', default='new',
                    help='Set name: train / val / test.')
parser.add_argument('--num_digits', default=6, type=int,
                    help='Number of digits to enumerate the generated files.')
parser.add_argument('--output_audio_dir', default='../daqa/audio/',
                    help='Directory to output generated audio.')
parser.add_argument('--output_narrative_dir', default='../daqa/narratives/',
                    help='Directory to output generated narratives.')
parser.add_argument('--output_narrative_file',
                    default='../daqa/daqa_narratives.json',
                    help="Path to narratives JSON file.")


def main(args):
    """Randomly sample audio events to form sequences of events."""

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Read dataset description
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    # Define naming conventions and directories
    prefix = '%s_%s_' % (args.filename_prefix, args.set)
    audio_template = '%s%%0%dd.wav' % (prefix, args.num_digits)
    audio_template = os.path.join(args.output_audio_dir, audio_template)
    narrative_template = '%s%%0%dd.json' % (prefix, args.num_digits)
    narrative_template = os.path.join(args.output_narrative_dir,
                                      narrative_template)

    if not os.path.isdir(args.output_audio_dir):
        os.makedirs(args.output_audio_dir)
    if not os.path.isdir(args.output_narrative_dir):
        os.makedirs(args.output_narrative_dir)

    # Get list of events and backgrounds
    lst_events = list(dataset['origins'].keys())  # without .wav
    lst_events_wav = os.listdir(args.events)
    lst_events_wav = [e[:-4] for e in lst_events_wav if e.endswith('.wav')]
    assert len(lst_events) == len(lst_events_wav), 'Dataset mismatch.'
    assert sorted(lst_events) == sorted(lst_events_wav), 'Dataset mismatch.'
    lst_bckgrnds = os.listdir(args.backgrounds)
    lst_bckgrnds = [e for e in lst_bckgrnds if e.endswith('.wav')]
    x_consctvs = [k for k, v in dataset['consecutive'].items() if v is False]
    num_fails = 0

    # Generate audio and narratives from events
    lst_narrative_paths = []
    for i in range(args.num_audio):
        idx = args.start_idx + i
        audio_path = audio_template % idx
        narrative_path = narrative_template % idx
        lst_narrative_paths.append(narrative_path)
        num_events = random.randint(args.min_num_events, args.max_num_events)

        # Sample num_events number of events (not unique)
        sel_events = None
        while sel_events is None:
            sel_events = random.sample(lst_events, num_events)
            # The following checks if the sequence of selected events is ok
            sel_events_dx = [x.split('_')[0] for x in sel_events]
            # Check if the list has any identical consective events
            consecutives = []
            for x in range(len(sel_events_dx) - 1):
                if sel_events_dx[x] == sel_events_dx[x + 1]:
                    consecutives.append(sel_events_dx[x])
            # Check if any of the events in consecutives are not allowed
            if len([x for x in consecutives if x in x_consctvs]) > 0:
                sel_events = None  # retry
                num_fails += 1
        sel_bckgrnd = random.sample(lst_bckgrnds, 1)
        audio, narrative = gen_audio_narrative(dataset=dataset,
                                               args=args,
                                               selcted_events=sel_events,
                                               selcted_bckgrnd=sel_bckgrnd,
                                               output_index=idx,
                                               output_audio=audio_path,
                                               )
        scipy.io.wavfile.write(audio_path, args.data_fs, audio)
        with open(narrative_path, 'w') as f:
            json.dump(narrative, f)

    print('Generated ' + str(args.num_audio) + ' audio sequences ('
          + str(num_fails) + ' failed attempts). Compiliing narratives...')

    # Combine all narratives into a single JSON file
    lst_narratives = []
    for narrative_path in lst_narrative_paths:
        with open(narrative_path, 'r') as f:
            lst_narratives.append(json.load(f))
    output = {
        'info': {
            'set': args.set,
            'version': args.version,
            'date': args.date,
            'license': args.license,
        },
        'narratives': lst_narratives
    }
    with open(args.output_narrative_file, 'w') as f:
        json.dump(output, f)

    return True


def gen_audio_narrative(dataset,
                        args,
                        selcted_events,
                        selcted_bckgrnd,
                        output_index,
                        output_audio):

    # Read audio events
    lst_audio_events = []
    for e in selcted_events:
        e_wav = os.path.join(args.events, e + '.wav')
        event_fs, event = scipy.io.wavfile.read(e_wav)
        assert event_fs == args.data_fs, \
            'Audio event sampling frequency != ' + str(args.data_fs) + ' Hz.'
        lst_audio_events.append(event)

    # Toss an unbiased coin to concatenate or add events
    if random.random() < 0.5:
        # concatenate
        audio = np.concatenate(lst_audio_events)
    else:
        # add (allows overlap between adjacent events)
        audio = lst_audio_events[0]
        for event in lst_audio_events[1:]:
            idx_overlap = random.randint(0, (args.rand_overlap * args.data_fs))
            plhldr = np.zeros(event.shape[0] - idx_overlap, event.dtype)
            audio = np.concatenate((audio, plhldr))
            audio[-event.shape[0]:] += event
    assert len(audio.shape) == 1, 'Audio events not concatenated properly.'

    # Toss an unbiased coin to add background noise
    background = 'None'
    if random.random() < 0.5:
        selec_bckgrnd = os.path.join(args.backgrounds, selcted_bckgrnd[0])
        bckgrnd_fs, bckgrnd = scipy.io.wavfile.read(selec_bckgrnd)
        assert event_fs == args.data_fs, \
            'Bckgrnd sampling frequency != ' + str(args.data_fs) + ' Hz.'
        idx_trim = random.randint(0, bckgrnd.shape[0] - audio.shape[0])
        trim_bckgrnd = bckgrnd[idx_trim:(audio.shape[0] + idx_trim)]
        audio += trim_bckgrnd
        background = selcted_bckgrnd[0][:-4]

    events = []
    for idx, sel_event in enumerate(selcted_events):
        event_dx = sel_event.split('_')[0]
        event = {  # 'start_time': 'end_time':
            'order': idx,
            'event': event_dx,
            'audio': sel_event,
            'source': random.choice(dataset['sources'][event_dx]),
            'action': random.choice(dataset['actions'][event_dx]),
            'duration': (float(lst_audio_events[idx].shape[0]) / args.data_fs),
            'loudness': dataset['origins'][sel_event]['loudness'],
        }
        events.append(event)

    # Generate JSON
    narrative = {
        'set': args.set,
        'audio_index': output_index,
        'audio_filename': os.path.basename(output_audio),
        'background': background,
        'events': events,
    }

    return audio, narrative


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print('Success!')
