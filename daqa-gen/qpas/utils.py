# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import re
import numpy as np


def a_or_an(q):
    a_an_letter = re.findall(r'\[a,an\] \w', q)
    for e in a_an_letter:
        a_an, letter = e.split(' ')
        if letter in ['a', 'e', 'i', 'o', 'u']:
            q = q.replace('[a,an]', 'an', 1)  # 1 to denote first occurrence
        else:
            q = q.replace('[a,an]', 'a', 1)
    return q


def options(q):
    assert ('[a' not in q) or ('an]' not in q), '[a,an] choice cant be random.'
    opt = re.findall(r'\[(.*?)\]', q)
    for o in opt:
        q = q.replace('[' + o + ']', np.random.choice(o.split(',')))
    return q


def spaces(q):
    q = q.replace('  ', ' ')
    q = q.replace('   ', ' ')
    return q


def sanitize_question(q):
    q = a_or_an(q)
    q = options(q)
    q = spaces(q)
    q = q.lower()
    q = q.capitalize()  # capitalizes only first letter
    assert '<' not in q, 'Could not sanitize template: ' + q
    assert '>' not in q, 'Could not sanitize template: ' + q
    assert '[' not in q, 'Could not sanitize template: ' + q
    assert ']' not in q, 'Could not sanitize template: ' + q
    return q


def sample_conjunction():
    return str(np.random.choice(['and', 'or']))


def sample_preposition():
    return str(np.random.choice(['before', 'after']))


def sample_immediate_preposition():
    return '[just,immediately] ' + sample_preposition()


def numbers_to_ordinals(num):
    ordinals = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
        6: 'sixth',
        7: 'seventh',
        8: 'eighth',
        9: 'ninth',
        10: 'tenth',
        11: 'eleventh',
        12: 'twelveth',
        13: 'thirteenth',
        14: 'fourteenth',
        15: 'fifteenth',
    }
    return ordinals[num]


def sample_number(n):
    number = int(np.random.randint(1, n + 1, 1))  # human indexing
    return number, numbers_to_ordinals(number)


def sample_second_number(n, x_n):
    lst_x_n = list(range(1, n + 1))  # human indexing
    lst_x_n.remove(x_n)
    number = int(np.random.choice(lst_x_n))
    return number, numbers_to_ordinals(number)


def sample_loudness():
    return str(np.random.choice(['quiet', 'loud']))


def sample_rel_loudness():
    return str(np.random.choice(['quieter', 'louder']))


def sample_absolute_loudness():
    return str(np.random.choice(['quietest', 'loudest']))


def sample_duration():
    return str(np.random.choice(['short', 'long']))


def sample_rel_duration():
    return str(np.random.choice(['shorter', 'longer']))


def sample_absolute_duration():
    return str(np.random.choice(['shortest', 'longest']))


def get_lst_events(narrative):
    le = len(narrative['events'])
    return [narrative['events'][e]['event'] for e in range(le)]


def get_lst_sources(narrative):
    le = len(narrative['events'])
    return [narrative['events'][e]['source'] for e in range(le)]


def get_lst_all_sources(dataset, narrative):
    ls = []
    for e in range(len(narrative['events'])):
        ls += dataset['sources'][narrative['events'][e]['event']]
    return ls


def get_lst_actions(narrative):
    le = len(narrative['events'])
    return [narrative['events'][e]['action'] for e in range(le)]


def get_lst_durations(narrative):
    le = len(narrative['events'])
    return np.array([narrative['events'][e]['duration'] for e in range(le)])


def get_lst_loudness(narrative):
    le = len(narrative['events'])
    return np.array([narrative['events'][e]['loudness'] for e in range(le)])


def compute_rel_diff(actual, reference):
    return np.abs(actual - reference) / reference


def numbers_to_words(n):
    numbers = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten',
        11: 'eleven',
        12: 'twelve',
        13: 'thirteen',
        14: 'fourteen',
        15: 'fifteen',
    }
    return numbers[n]
