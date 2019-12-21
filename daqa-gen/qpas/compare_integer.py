# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from qpas.utils import get_lst_events, sanitize_question


def less_than(dataset, narrative, _):
    questions = ['Were there fewer <S1>s <A1> than <S2>s <A2>?',
                 'Was the number of [times,instances,occurrences] [a,an] <S1> <A1> less than the number of [times,instances,occurrences] [a,an] <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of [a,an] <S1> <A1> and [a,an] <S2> <A2>, were there fewer [times,instances,occurrences] of the former?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of [a,an] <S2> <A2> and [a,an] <S1> <A1>, were there fewer [times,instances,occurrences] of the latter?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    event_1 = str(np.random.choice(lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    x_lst_events = [e for e in lst_events if e != event_1]
    assert len(x_lst_events) > 0, 'Question (less_than) illposed.'
    event_2 = str(np.random.choice(x_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert event_1 != event_2, 'Question (less_than) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    answer = 'yes' \
        if lst_events.count(event_1) < lst_events.count(event_2) \
        else 'no'

    return question, answer


def equal_to(dataset, narrative, _):
    questions = ['Was the number of times [a,an] <S1> <A1> equal to the number of times [a,an] <S2> <A2>?',  # noqa: E501
                 'Was the number of times [a,an] <S1> <A1> the same as the number of times [a,an] <S2> <A2>?',  # noqa: E501
                 'Was there an equal number of times [a,an] <S1> <A1> and [a,an] <S2> <A2>?',  # noqa: E501
                 'Was there the same number of <S1> <A1> and <S2> <A2>?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    event_1 = str(np.random.choice(lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    x_lst_events = [e for e in lst_events if e != event_1]
    assert len(x_lst_events) > 0, 'Question (equal_to) illposed.'
    event_2 = str(np.random.choice(x_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert event_1 != event_2, 'Question (equal_to) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    answer = 'yes' \
        if lst_events.count(event_1) == lst_events.count(event_2) \
        else 'no'

    return question, answer


def more_than(dataset, narrative, _):
    questions = ['Were there more <S1>s <A1> than <S2>s <A2>?',
                 'Was the number of [times,instances,occurrences] [a,an] <S1> <A1> more than the number of [times,instances,occurrences] [a,an] <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of [a,an] <S1> <A1> and [a,an] <S2> <A2>, were there more [times,instances,occurrences] of the former?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of [a,an] <S2> <A2> and [a,an] <S1> <A1>, were there more [times,instances,occurrences] of the latter?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    event_1 = str(np.random.choice(lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    x_lst_events = [e for e in lst_events if e != event_1]
    assert len(x_lst_events) > 0, 'Question (more_than) illposed.'
    event_2 = str(np.random.choice(x_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert event_1 != event_2, 'Question (more_than) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    answer = 'yes' \
        if lst_events.count(event_1) > lst_events.count(event_2) \
        else 'no'

    return question, answer
