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
                        get_lst_loudness, sample_absolute_duration,
                        sample_absolute_loudness, sample_immediate_preposition,
                        sample_number, sample_preposition, sanitize_question)


def what_was(dataset, narrative, _):
    questions = ['What was the <O> sound you [heard,listened to]?',
                 'What was the <O> sound?',
                 'What did the <O> sound [sound,seem] like?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    event = lst_events[number - 1]
    answer = (str(np.random.choice(dataset['sources'][event]))
              + ' '
              + str(np.random.choice(dataset['actions'][event])))

    return question, answer


def what_was_relative(dataset, narrative, _):
    questions = ['What was the sound <RO> the <S> <A>?',
                 'What was the sound <RO> [hearing,listening to] the <S> <A>?',
                 'What was the sound <RO> the <S> <A> was heard?',
                 'What did you [hear,listen to] <RO> the <S> <A>?',
                 'What did you [hear,listen to] <RO> [hearing,listening to] the <S> <A>?',  # noqa: E501
                 'What did you [hear,listen to] <RO> the <S> <A> was heard?',
                 'What was the sound <IO> the <S> <A>?',
                 'What was the sound <IO> [hearing,listening to] the <S> <A>?',
                 'What was the sound <IO> the <S> <A> was heard?',
                 'What did you [hear,listen to] <IO> the <S> <A>?',
                 'What did you [hear,listen to] <IO> [hearing,listening to] the <S> <A>?',  # noqa: E501
                 'What did you [hear,listen to] <IO> the <S> <A> was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    preposition = sample_preposition()
    immediate_preposition = sample_immediate_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (what_was_relative) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    # Only one of the following two lines will have an effect
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<IO>', immediate_preposition)
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (what_was_relative) illposed.'

    event_idx = lst_events.index(event)
    if 'before' in question:
        if (event_idx - 1) < 0:
            answer = 'nothing'
        else:
            e = lst_events[event_idx - 1]
            answer = (str(np.random.choice(dataset['sources'][e]))
                      + ' '
                      + str(np.random.choice(dataset['actions'][e])))
    elif 'after' in question:
        if (event_idx + 1) >= len(lst_events):
            answer = 'nothing'
        else:
            e = lst_events[event_idx + 1]
            answer = (str(np.random.choice(dataset['sources'][e]))
                      + ' '
                      + str(np.random.choice(dataset['actions'][e])))
    else:
        assert False, 'Preposition illdefined in Question (what_was_relative).'

    return question, answer


def what_was_loudness(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AL> sound?',
                 'What was the <AL> sound you [heard,listened to]?',
                 'What was the <AL> sound that you [heard,listened to]?',
                 'What was the <AL> sound that was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_absolute_loudness()

    question = question.replace('<AL>', loudness)  # insert loudness
    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    lst_loudness = get_lst_loudness(narrative)
    if 'loud' in question:
        est = np.argmax(lst_loudness)
    elif 'quiet' in question:
        est = np.argmin(lst_loudness)
    else:
        assert False, \
            'Loudness illdefined in Question (what_was_loudness).'
    # Assert a good margin in relative loudness
    evt_loudness = lst_loudness[est]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != est]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (what_was_loudness) illposed.'
    e = lst_events[est]
    answer = (str(np.random.choice(dataset['sources'][e]))
              + ' '
              + str(np.random.choice(dataset['actions'][e])))

    return question, answer


def what_was_loudness_relative(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AL> sound <RO> the <S> <A>?',
                 'What was the <AL> sound <RO> [hearing,listening to] the <S> <A>?',
                 'What was the <AL> sound <RO> the <S> <A> was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_absolute_loudness()
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (what_was_loudness_relative) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<AL>', loudness)  # insert loudness
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (what_was_loudness_relative) illposed.'

    lst_loudness = get_lst_loudness(narrative)
    event_idx = lst_events.index(event)
    if 'before' in question:
        lst_events_e = lst_events[:event_idx]
        lst_events_l = lst_loudness[:event_idx]
    elif 'after' in question:
        lst_events_e = lst_events[(event_idx + 1):]
        lst_events_l = lst_loudness[(event_idx + 1):]
    else:
        assert False, \
            'Preposition illdefined in Question (what_was_loudness_relative).'
    assert len(lst_events_e) > 0, \
        'Question (what_was_loudness_relative) illposed.'
    if 'loud' in question:
        est = np.argmax(lst_events_l)
    elif 'quiet' in question:
        est = np.argmin(lst_events_l)
    else:
        assert False, \
            'Loudness illdefined in Question (what_was_loudness_relative).'
    # Assert a good margin in relative loudness
    evt_loudness = lst_events_l[est]
    x_loudness = [j for i, j in enumerate(lst_events_l) if i != est]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
        'Question (what_was_loudness_relative) illposed.'
    e = lst_events_e[est]
    answer = (str(np.random.choice(dataset['sources'][e]))
              + ' '
              + str(np.random.choice(dataset['actions'][e])))

    return question, answer


def what_was_loudness_relative_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AL> sound <RO> the <O> sound?',
                 'What was the <AL> sound <RO> [hearing,listening to] the <O> sound?',
                 'What was the <AL> sound <RO> the <O> sound was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_absolute_loudness()
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<AL>', loudness)  # insert loudness
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    event_idx = (number - 1)
    answer = None
    if 'before' in question:
        if (event_idx - 1) < 0:
            answer = 'nothing'
        else:
            lst_events_e = lst_events[:event_idx]
            lst_events_l = lst_loudness[:event_idx]
    elif 'after' in question:
        if (event_idx + 1) >= len(lst_events):
            answer = 'nothing'
        else:
            lst_events_e = lst_events[(event_idx + 1):]
            lst_events_l = lst_loudness[(event_idx + 1):]
    else:
        assert False, \
            'Preposition illdefined in Question (what_was_loudness_relative_ordinal).'
    if answer is None:
        assert len(lst_events_e) > 0, \
            'Question (what_was_loudness_relative_ordinal) illposed.'
        if 'loud' in question:
            est = np.argmax(lst_events_l)
        elif 'quiet' in question:
            est = np.argmin(lst_events_l)
        else:
            assert False, \
                'Loudness illdefined in Question (what_was_loudness_relative_ordinal).'
        # Assert a good margin in relative loudness
        evt_loudness = lst_events_l[est]
        x_loudness = [j for i, j in enumerate(lst_events_l) if i != est]
        rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                             np.array(evt_loudness))
        assert np.sum(rel_loudness_diff < rel_diff) <= 0, \
            'Question (what_was_loudness_relative_ordinal) illposed.'
        e = lst_events_e[est]
        answer = (str(np.random.choice(dataset['sources'][e]))
                  + ' '
                  + str(np.random.choice(dataset['actions'][e])))

    return question, answer


def what_was_duration(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AD> sound?',
                 'What was the <AD> sound you [heard,listened to]?',
                 'What was the <AD> sound that you [heard,listened to]?',
                 'What was the <AD> sound that was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_absolute_duration()

    question = question.replace('<AD>', duration)  # insert duration
    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    lst_durations = get_lst_durations(narrative)
    if 'long' in question:
        est = np.argmax(lst_durations)
    elif 'short' in question:
        est = np.argmin(lst_durations)
    else:
        assert False, \
            'Duration illdefined in Question (what_was_duration).'
    # Assert a good margin in relative duration
    evt_duration = lst_durations[est]
    x_durations = [j for i, j in enumerate(lst_durations) if i != est]
    rel_duration_diff = compute_rel_diff(np.array(x_durations),
                                         np.array(evt_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (what_was_duration) illposed.'
    e = lst_events[est]
    answer = (str(np.random.choice(dataset['sources'][e]))
              + ' '
              + str(np.random.choice(dataset['actions'][e])))

    return question, answer


def what_was_duration_relative(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AD> sound <RO> the <S> <A>?',
                 'What was the <AD> sound <RO> [hearing,listening to] the <S> <A>?',
                 'What was the <AD> sound <RO> the <S> <A> was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_absolute_duration()
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (what_was_duration_relative) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<AD>', duration)  # insert duration
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (what_was_duration_relative) illposed.'

    lst_durations = get_lst_durations(narrative)
    event_idx = lst_events.index(event)
    if 'before' in question:
        lst_events_e = lst_events[:event_idx]
        lst_events_d = lst_durations[:event_idx]
    elif 'after' in question:
        lst_events_e = lst_events[(event_idx + 1):]
        lst_events_d = lst_durations[(event_idx + 1):]
    else:
        assert False, \
            'Preposition illdefined in Question (what_was_duration_relative).'
    assert len(lst_events_e) > 0, \
        'Question (what_was_duration_relative) illposed.'
    if 'long' in question:
        est = np.argmax(lst_events_d)
    elif 'short' in question:
        est = np.argmin(lst_events_d)
    else:
        assert False, \
            'Duration illdefined in Question (what_was_duration_relative).'
    # Assert a good margin in relative duration
    evt_duration = lst_events_d[est]
    x_durations = [j for i, j in enumerate(lst_events_d) if i != est]
    rel_duration_diff = compute_rel_diff(np.array(x_durations),
                                         np.array(evt_duration))
    assert np.sum(rel_duration_diff < rel_diff) <= 0, \
        'Question (what_was_duration_relative) illposed.'
    e = lst_events_e[est]
    answer = (str(np.random.choice(dataset['sources'][e]))
              + ' '
              + str(np.random.choice(dataset['actions'][e])))

    return question, answer


def what_was_duration_relative_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['What was the <AD> sound <RO> the <O> sound?',
                 'What was the <AD> sound <RO> [hearing,listening to] the <O> sound?',
                 'What was the <AD> sound <RO> the <O> sound was heard?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_absolute_duration()
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<AD>', duration)  # insert duration
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_durations = get_lst_durations(narrative)
    event_idx = (number - 1)
    answer = None
    if 'before' in question:
        if (event_idx - 1) < 0:
            answer = 'nothing'
        else:
            lst_events_e = lst_events[:event_idx]
            lst_events_d = lst_durations[:event_idx]
    elif 'after' in question:
        if (event_idx + 1) >= len(lst_events):
            answer = 'nothing'
        else:
            lst_events_e = lst_events[(event_idx + 1):]
            lst_events_d = lst_durations[(event_idx + 1):]
    else:
        assert False, \
            'Preposition illdefined in Question (what_was_duration_relative_ordinal).'
    if answer is None:
        assert len(lst_events_e) > 0, \
            'Question (what_was_duration_relative_ordinal) illposed.'
        if 'long' in question:
            est = np.argmax(lst_events_d)
        elif 'short' in question:
            est = np.argmin(lst_events_d)
        else:
            assert False, \
                'Duration illdefined in Question (what_was_duration_relative_ordinal).'
        # Assert a good margin in relative duration
        evt_duration = lst_events_d[est]
        x_durations = [j for i, j in enumerate(lst_events_d) if i != est]
        rel_duration_diff = compute_rel_diff(np.array(x_durations),
                                             np.array(evt_duration))
        assert np.sum(rel_duration_diff < rel_diff) <= 0, \
            'Question (what_was_duration_relative_ordinal) illposed.'
        e = lst_events_e[est]
        answer = (str(np.random.choice(dataset['sources'][e]))
                  + ' '
                  + str(np.random.choice(dataset['actions'][e])))

    return question, answer
