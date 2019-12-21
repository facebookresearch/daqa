# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from qpas.utils import (compute_rel_diff, get_lst_durations, get_lst_events,
                        get_lst_loudness, sample_duration, sample_loudness,
                        sample_number, sample_second_number,
                        sample_rel_duration, sample_rel_loudness,
                        sanitize_question)


def compare_ordinal(dataset, narrative, _):
    questions = ['Was the <O1> [sound event,sound] [the same as,similar to] the <O2> [sound event,sound]?',  # noqa: E501
                 'Was the <O1> [sound event,sound] and <O2> [sound event,sound] [the same,similar]?',  # noqa: E501
                 'Were the <O1> and <O2> [sound events,sounds] [the same,similar]?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    assert number_1 != number_2, 'Question (compare_ordinal) illposed.'

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    answer = 'yes' if lst_events[number_1 - 1] == lst_events[number_2 - 1] \
        else 'no'

    return question, answer


def compare_ordinal_event(dataset, narrative, _):
    questions = ['Was the <O> [sound event,sound] [a,an] <S> <A>?',  # noqa: E501
                 'Did the <O> [sound event,sound] [sound,seem] like [a,an] <S> <A>?',  # noqa: E501
                 '[Listening to,Hearing] the <O> [sound event,sound], was it [a,an] <S> <A>?',  # noqa: E501
                 '[Listening to,Hearing] the <O> [sound event,sound], did it [sound,seem] like [a,an] <S> <A>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))
    event = str(np.random.choice(dataset['events']))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<O>', ordinal)  # insert ordinal
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    answer = 'yes' if lst_events[number - 1] == event else 'no'

    return question, answer


def compare_loudness(dataset, narrative, rel_diff):
    questions = ['Was the <S1> <A1> <RL> than the <S2> <A2>?',
                 'Was the sound of the <S1> <A1> <RL> than the sound of the <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S1> <A1> and the sound of the <S2> <A2>, was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S1> <A1> and the <S2> <A2>, was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S2> <A2> and the sound of the <S1> <A1>, was the latter <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S2> <A2> and the <S1> <A1>, was the latter <RL>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, 'Question (compare_loudness) illposed.'
    event_1 = str(np.random.choice(unique_lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    rel_loudness = sample_rel_loudness()
    x_unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(x_unique_lst_events) > 0, \
        'Question (compare_loudness) illposed.'
    event_2 = str(np.random.choice(x_unique_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert lst_events.count(event_1) == 1, \
        'Question (compare_loudness) illposed.'
    assert lst_events.count(event_2) == 1, \
        'Question (compare_loudness) illposed.'
    assert event_1 != event_2, 'Question (compare_loudness) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<RL>', rel_loudness)  # insert loudness
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[lst_events.index(event_1)]
    e_2_loudness = lst_loudness[lst_events.index(event_2)]
    # Assert a good margin in relative loudness
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (compare_loudness) illposed.'
    if 'quiet' in question:
        answer = 'yes' if e_1_loudness < e_2_loudness else 'no'
    elif 'loud' in question:
        answer = 'yes' if e_1_loudness > e_2_loudness else 'no'
    else:
        assert False, 'Loudness illdefined in Question (compare_loudness).'

    return question, answer


def compare_loudness_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O1> [sound event,sound] <RL> than the <O2> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O2> [sound event,sound] and the <O1> [sound event,sound], was the latter <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O2> and <O1> [sound events,sounds], was the latter <RL>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    rel_loudness = sample_rel_loudness()
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    assert number_1 != number_2, 'Question (compare_loudness_ordinal) illposed.'

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<RL>', rel_loudness)  # insert loudness
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[number_1 - 1]
    e_2_loudness = lst_loudness[number_2 - 1]
    # Assert a good margin in relative loudness
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (compare_loudness_ordinal) illposed.'
    if 'quiet' in question:
        answer = 'yes' if e_1_loudness < e_2_loudness else 'no'
    elif 'loud' in question:
        answer = 'yes' if e_1_loudness > e_2_loudness else 'no'
    else:
        assert False, 'Loudness illdefined in Question (compare_loudness_ordinal).'

    return question, answer


def compare_loudness_event_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S> <A> <RL> than the <O> [sound event,sound]?',
                 'Was the sound of the <S> <A> <RL> than the <O> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, was the latter <RL>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_loudness_event_ordinal) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    rel_loudness = sample_rel_loudness()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_loudness_event_ordinal) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_loudness_event_ordinal) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<RL>', rel_loudness)  # insert loudness
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[lst_events.index(event)]
    e_2_loudness = lst_loudness[number - 1]
    # Assert a good margin in relative loudness
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (compare_loudness_event_ordinal) illposed.'
    if 'quiet' in question:
        answer = 'yes' if e_1_loudness < e_2_loudness else 'no'
    elif 'loud' in question:
        answer = 'yes' if e_1_loudness > e_2_loudness else 'no'
    else:
        assert False, \
            'Loudness illdefined in Question (compare_loudness_event_ordinal).'

    return question, answer


def compare_loudness_ordinal_event(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O> [sound event,sound] <RL> than the <S> <A>?',
                 'Was the <O> [sound event,sound] <RL> than the sound of the <S> <A>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, was the former <RL>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], was the latter <RL>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_loudness_ordinal_event) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    rel_loudness = sample_rel_loudness()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_loudness_ordinal_event) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_loudness_ordinal_event) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<RL>', rel_loudness)  # insert loudness
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[number - 1]
    e_2_loudness = lst_loudness[lst_events.index(event)]
    # Assert a good margin in relative loudness
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (compare_loudness_ordinal_event) illposed.'
    if 'quiet' in question:
        answer = 'yes' if e_1_loudness < e_2_loudness else 'no'
    elif 'loud' in question:
        answer = 'yes' if e_1_loudness > e_2_loudness else 'no'
    else:
        assert False, \
            'Loudness illdefined in Question (compare_loudness_ordinal_event).'

    return question, answer


def compare_same_loudness(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S1> <A1> [roughly,approximately] as <L> as the <S2> <A2>?',  # noqa: E501
                 'Was the sound of the <S1> <A1> [roughly,approximately] as <L> as the sound of the <S2> <A2>?',  # noqa: E501
                 'Was the sound of the <S1> <A1> [roughly,approximately] the same loudness as the sound of the <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S1> <A1> and the sound of the <S2> <A2>, did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S1> <A1> and the <S2> <A2>, did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_same_loudness) illposed.'
    event_1 = str(np.random.choice(unique_lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    loudness = sample_loudness()
    x_unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(x_unique_lst_events) > 0, \
        'Question (compare_same_loudness) illposed.'
    event_2 = str(np.random.choice(x_unique_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert lst_events.count(event_1) == 1, \
        'Question (compare_same_loudness) illposed.'
    assert lst_events.count(event_2) == 1, \
        'Question (compare_same_loudness) illposed.'
    assert event_1 != event_2, 'Question (compare_same_loudness) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[lst_events.index(event_1)]
    e_2_loudness = lst_loudness[lst_events.index(event_2)]
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_loudness) illposed.'
    answer = 'yes' if rel_loudness_diff <= rel_diff else 'no'

    return question, answer


def compare_same_loudness_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O1> [sound event,sound] [roughly,approximately] as <L> as the <O2> [sound event,sound]?',  # noqa: E501
                 'Was the <O1> and <O2> [sound events,sounds] [roughly,approximately] as <L>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], did they have [roughly,approximately] the same loudness?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    loudness = sample_loudness()
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    assert number_1 != number_2, 'Question (compare_same_loudness_ordinal) illposed.'

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[number_1 - 1]
    e_2_loudness = lst_loudness[number_2 - 1]
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_loudness_ordinal) illposed.'
    answer = 'yes' if rel_loudness_diff <= rel_diff else 'no'

    return question, answer


def compare_same_loudness_event_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S> <A> [roughly,approximately] as <L> as the <O> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S> <A> and the <O> [sound event,sound], were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S> <A> and the <O> [sound event,sound], did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 'Was the <O> [sound event,sound] [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the sound of the <S> <A>, were they [roughly,approximately] as loud?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the sound of the <S> <A>, did they [roughly,approximately] have the same loudness?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_same_loudness_event_ordinal) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    loudness = sample_loudness()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_same_loudness_event_ordinal) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_same_loudness_event_ordinal) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    e_1_loudness = lst_loudness[lst_events.index(event)]
    e_2_loudness = lst_loudness[number - 1]
    rel_loudness_diff = compute_rel_diff(np.array(e_1_loudness),
                                         np.array(e_2_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_loudness_event_ordinal) illposed.'
    answer = 'yes' if rel_loudness_diff <= rel_diff else 'no'

    return question, answer


def compare_duration(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S1> <A1> <RD> than the <S2> <A2>?',
                 'Was the sound of the <S1> <A1> <RD> than the sound of the <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S1> <A1> and the sound of the <S2> <A2>, was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S1> <A1> and the <S2> <A2>, was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S2> <A2> and the sound of the <S1> <A1>, was the latter <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S2> <A2> and the <S1> <A1>, was the latter <RD>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_duration) illposed.'
    event_1 = str(np.random.choice(unique_lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    rel_duration = sample_rel_duration()
    x_unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(x_unique_lst_events) > 0, \
        'Question (compare_duration) illposed.'
    event_2 = str(np.random.choice(x_unique_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert lst_events.count(event_1) == 1, \
        'Question (compare_duration) illposed.'
    assert lst_events.count(event_2) == 1, \
        'Question (compare_duration) illposed.'
    assert event_1 != event_2, 'Question (compare_duration) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<RD>', rel_duration)  # insert duration
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[lst_events.index(event_1)]
    e_2_duration = lst_duration[lst_events.index(event_2)]
    # Assert a good margin in relative duration
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (compare_duration) illposed.'
    if 'short' in question:
        answer = 'yes' if e_1_duration < e_2_duration else 'no'
    elif 'long' in question:
        answer = 'yes' if e_1_duration > e_2_duration else 'no'
    else:
        assert False, 'Duration illdefined in Question (compare_duration).'

    return question, answer


def compare_duration_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O1> [sound event,sound] <RD> than the <O2> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O2> [sound event,sound] and the <O1> [sound event,sound], was the latter <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O2> and <O1> [sound events,sounds], was the latter <RD>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    rel_duration = sample_rel_duration()
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    assert number_1 != number_2, 'Question (compare_duration_ordinal) illposed.'

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<RD>', rel_duration)  # insert duration
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[number_1 - 1]
    e_2_duration = lst_duration[number_2 - 1]
    # Assert a good margin in relative duration
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (compare_duration_ordinal) illposed.'
    if 'short' in question:
        answer = 'yes' if e_1_duration < e_2_duration else 'no'
    elif 'long' in question:
        answer = 'yes' if e_1_duration > e_2_duration else 'no'
    else:
        assert False, 'Duration illdefined in Question (compare_duration_ordinal).'

    return question, answer


def compare_duration_event_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S> <A> <RD> than the <O> [sound event,sound]?',
                 'Was the sound of the <S> <A> <RD> than the <O> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, was the latter <RD>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_duration_event_ordinal) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    rel_duration = sample_rel_duration()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_duration_event_ordinal) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_duration_event_ordinal) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<RD>', rel_duration)  # insert duration
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[lst_events.index(event)]
    e_2_duration = lst_duration[number - 1]
    # Assert a good margin in relative duration
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (compare_duration_event_ordinal) illposed.'
    if 'short' in question:
        answer = 'yes' if e_1_duration < e_2_duration else 'no'
    elif 'long' in question:
        answer = 'yes' if e_1_duration > e_2_duration else 'no'
    else:
        assert False, \
            'Duration illdefined in Question (compare_duration_event_ordinal).'

    return question, answer


def compare_duration_ordinal_event(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O> [sound event,sound] <RD> than the <S> <A>?',
                 'Was the <O> [sound event,sound] <RD> than the sound of the <S> <A>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, was the former <RD>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], was the latter <RD>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_duration_ordinal_event) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    rel_duration = sample_rel_duration()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_duration_ordinal_event) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_duration_ordinal_event) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<RD>', rel_duration)  # insert duration
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[number - 1]
    e_2_duration = lst_duration[lst_events.index(event)]
    # Assert a good margin in relative duration
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (compare_duration_ordinal_event) illposed.'
    if 'short' in question:
        answer = 'yes' if e_1_duration < e_2_duration else 'no'
    elif 'long' in question:
        answer = 'yes' if e_1_duration > e_2_duration else 'no'
    else:
        assert False, \
            'Duration illdefined in Question (compare_duration_ordinal_event).'

    return question, answer


def compare_same_duration(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S1> <A1> [roughly,approximately] as <D> as the <S2> <A2>?',  # noqa: E501
                 'Was the sound of the <S1> <A1> [roughly,approximately] as <D> as the sound of the <S2> <A2>?',  # noqa: E501
                 'Was the sound of the <S1> <A1> [roughly,approximately] the same duration as the sound of the <S2> <A2>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S1> <A1> and the sound of the <S2> <A2>, did they [roughly,approximately] have the same duration?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sounds of the <S1> <A1> and the <S2> <A2>, did they [roughly,approximately] have the same duration?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_same_duration) illposed.'
    event_1 = str(np.random.choice(unique_lst_events))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    duration = sample_duration()
    x_unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(x_unique_lst_events) > 0, \
        'Question (compare_same_duration) illposed.'
    event_2 = str(np.random.choice(x_unique_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert lst_events.count(event_1) == 1, \
        'Question (compare_same_duration) illposed.'
    assert lst_events.count(event_2) == 1, \
        'Question (compare_same_duration) illposed.'
    assert event_1 != event_2, 'Question (compare_same_duration) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[lst_events.index(event_1)]
    e_2_duration = lst_duration[lst_events.index(event_2)]
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_duration_diff > rel_diff,
                                 rel_duration_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_duration) illposed.'
    answer = 'yes' if rel_duration_diff <= rel_diff else 'no'

    return question, answer


def compare_same_duration_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <O1> [sound event,sound] [roughly,approximately] as <D> as the <O2> [sound event,sound]?',  # noqa: E501
                 'Was the <O1> and <O2> [sound events,sounds] [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> [sound event,sound] and the <O2> [sound event,sound], did they [roughly,approximately] have the same duration?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O1> and <O2> [sound events,sounds], did they [roughly,approximately] have the same duration?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    duration = sample_duration()
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    assert number_1 != number_2, 'Question (compare_same_duration_ordinal) illposed.'

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[number_1 - 1]
    e_2_duration = lst_duration[number_2 - 1]
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_duration_diff > rel_diff,
                                 rel_duration_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_duration_ordinal) illposed.'
    answer = 'yes' if rel_duration_diff <= rel_diff else 'no'

    return question, answer


def compare_same_duration_event_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Was the <S> <A> [roughly,approximately] as <D> as the <O> [sound event,sound]?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S> <A> and the <O> [sound event,sound], were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <S> <A> and the <O> [sound event,sound], did they [roughly,approximately] have the same duration?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the sound of the <S> <A> and the <O> [sound event,sound], did they [roughly,approximately] have the same duration?',  # noqa: E501
                 'Was the <O> [sound event,sound] [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the sound of the <S> <A>, were they [roughly,approximately] as <D>?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the <S> <A>, did they [roughly,approximately] have the same duration?',  # noqa: E501
                 '[Comparing,Listening to,Hearing] the <O> [sound event,sound] and the sound of the <S> <A>, did they [roughly,approximately] have the same duration?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (compare_same_duration_event_ordinal) illposed.'
    event = str(np.random.choice(unique_lst_events))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    duration = sample_duration()
    number, ordinal = sample_second_number(len(lst_events), lst_events.index(event) + 1)

    assert lst_events.count(event) == 1, \
        'Question (compare_same_duration_event_ordinal) illposed.'
    assert lst_events.index(event) != (number - 1), \
        'Question (compare_same_duration_event_ordinal) illposed.'

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_duration = get_lst_durations(narrative)
    e_1_duration = lst_duration[lst_events.index(event)]
    e_2_duration = lst_duration[number - 1]
    rel_duration_diff = compute_rel_diff(np.array(e_1_duration),
                                         np.array(e_2_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_duration_diff > rel_diff,
                                 rel_duration_diff < (2 * rel_diff))) <= 0, \
        'Question (compare_same_duration_event_ordinal) illposed.'
    answer = 'yes' if rel_duration_diff <= rel_diff else 'no'

    return question, answer
