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

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--outline', default='daqa_outline.json', type=str,
                    help='Location of outline file.')
parser.add_argument('--sources', default='daqa_sources.json', type=str,
                    help='Location of sources file.')
parser.add_argument('--loudness', default='daqa_loudness.json', type=str,
                    help='Location of loudness file.')
# Settings
parser.add_argument('--version', default='1.0', type=str,
                    help='Version.')
parser.add_argument('--date',
                    default=datetime.datetime.today().strftime("%m/%d/%Y"),
                    help="Date.")
parser.add_argument('--license',
                    default='Creative Commons Attribution (CC-BY 4.0)',
                    help='License.')
# Output
parser.add_argument('--output', default='daqa.json', type=str,
                    help='Location of dataset file.')


def main(args):

    # Read files
    with open(args.outline, 'r') as f:
        outline = json.load(f)
    with open(args.sources, 'r') as f:
        sources = json.load(f)
    with open(args.loudness, 'r') as f:
        loudness = json.load(f)

    dataset = {
        'info': {
            'version': args.version,
            'date': args.date,
            'license': args.license,
        },
        'events': outline['events'],
        'sources': outline['sources'],
        'actions': outline['actions'],
        'consecutive': outline['consecutive'],
        'origins': {},
    }

    counter = {}
    for i in range(len(dataset['events'])):
        counter[dataset['events'][i]] = 0

    for i in range(1, len(sources.keys()) + 1):
        counter[sources[str(i)]['event']] += 1
        ins = sources[str(i)]['event'] + '_' + \
            str(counter[sources[str(i)]['event']])
        dataset['origins'][ins] = sources[str(i)]
        dataset['origins'][ins]['filename'] = ins + '.wav'
        dataset['origins'][ins]['loudness'] = loudness[ins]

    with open(args.output, 'w') as f:
        json.dump(dataset, f)  # indent=2


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print('Success!')
