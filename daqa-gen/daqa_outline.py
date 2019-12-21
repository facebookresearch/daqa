# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json


def main():
    dataset = {
        'events': ['a000', 'b000', 'b001', 'c000', 'c001', 'c002', 'c003',
                   'c004', 'd000', 'd001', 'd002', 'f000', 'f001', 'h000',
                   'h001', 'h002', 'h003', 'h004', 'p000', 't000'],  # unique
        'sources': {
            'a000': ['aircraft', 'plane'],
            'b000': ['band'],
            'b001': ['bird'],
            'c000': ['crowd'],
            'c001': ['crowd'],
            'c002': ['crowd'],
            'c003': ['driver', 'car', 'vehicle'],
            'c004': ['car', 'vehicle'],
            'd000': ['door'],
            'd001': ['doorbell'],
            'd002': ['dog'],
            'f000': ['fire truck', 'fire engine', 'emergency vehicle'],
            'f001': ['fire alarm', 'alarm'],
            'h000': ['human'],
            'h001': ['human'],
            'h002': ['human'],
            'h003': ['human'],
            'h004': ['human'],
            'p000': ['phone'],
            't000': ['storm'],
        },
        'actions': {
            'a000': ['passing by', 'flying over'],
            'b000': ['playing'],
            'b001': ['singing'],
            'c000': ['babbling'],
            'c001': ['applauding', 'clapping'],
            'c002': ['rioting', 'making noise'],
            'c003': ['honking'],
            'c004': ['passing by'],
            'd000': ['slamming', 'closing', 'shutting'],
            'd001': ['ringing'],
            'd002': ['barking', 'making noise'],
            'f000': ['passing by'],
            'f001': ['going off'],
            'h000': ['speaking', 'talking'],
            'h001': ['laughing'],
            'h002': ['typing on a keyboard', 'typing'],
            'h003': ['whistling'],
            'h004': ['operating a machine'],
            'p000': ['ringing'],
            't000': ['thundering'],
        },
        'consecutive': {
            'a000': True,
            'b000': False,
            'b001': False,
            'c000': False,
            'c001': False,
            'c002': False,
            'c003': False,
            'c004': True,
            'd000': True,
            'd001': False,
            'd002': False,
            'f000': False,
            'f001': False,
            'h000': True,
            'h001': True,
            'h002': False,
            'h003': False,
            'h004': False,
            'p000': False,
            't000': False,
        }
    }
    with open('daqa_outline.json', 'w') as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
    print('Success!')
