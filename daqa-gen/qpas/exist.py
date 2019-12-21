# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from qpas.utils import (compute_rel_diff, get_lst_all_sources,
                        get_lst_durations, get_lst_events, get_lst_loudness,
                        sample_duration, sample_immediate_preposition,
                        sample_loudness, sample_number, sample_preposition,
                        sanitize_question)


def was_there(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S> <A>?',
                 'Have you [heard,listened to] [a,an] <S> <A>?',
                 'Did you [hear,listen to] any <S> <A>?',
                 'Have you [heard,listened to] any <S> <A>?',
                 'Did you [hear,listen to] a sound that [sounds like,sounded like,is,was] [a,an] <S> <A>?',  # noqa: E501
                 'Have you [heard,listened to] a sound that [sounds like,sounded like,is,was] [a,an] <S> <A>?',  # noqa: E501
                 'Was there [a,an] <S> <A>?',
                 'Were there any <S>s <A>?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    event = str(np.random.choice(dataset['events']))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    answer = 'yes' if event in get_lst_events(narrative) else 'no'

    return question, answer


def was_there_two_and(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> <A1> and [a,an] <S2> <A2>?',
                 'Have you [heard,listened to] [a,an] <S1> <A1> and [a,an] <S2> <A2>?',
                 'Did you [hear,listen to] any <S1> <A1> and any <S2> <A2>?',
                 'Have you [heard,listened to] any <S1> <A1> and any <S2> <A2>?',
                 'Did you [hear,listen to] a sound that [sounds like,is] [a,an] <S1> <A1> and a sound [sounds like,is] [a,an] <S2> <A2>?',  # noqa: E501
                 'Did you [hear,listen to] a sound that [sounded like,was] [a,an] <S1> <A1> and a sound [sounded like,was] [a,an] <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] a sound that [sounds like,is] [a,an] <S1> <A1> and a sound [sounds like,is] [a,an] <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] a sound that [sounded like,was] [a,an] <S1> <A1> and a sound [sounded like,was] [a,an] <S2> <A2>?',  # noqa: E501
                 'Was there [a,an] <S1> <A1> and [a,an] <S2> <A2>?',
                 'Were there any <S1>s <A1> and any <S2>s <A2>?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))  # sample source
    action_1 = str(np.random.choice(dataset['actions'][event_1]))  # sample action
    lst_events = [e for e in dataset['events'] if e != event_1]
    event_2 = str(np.random.choice(lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))  # sample source
    action_2 = str(np.random.choice(dataset['actions'][event_2]))  # sample action

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    answer = 'yes' if (event_1 in lst_events and event_2 in lst_events) else 'no'

    return question, answer


def was_there_two_or(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> <A1> or [a,an] <S2> <A2>?',
                 'Have you [heard,listened to] [a,an] <S1> <A1> or [a,an] <S2> <A2>?',
                 'Did you [hear,listen to] any <S1> <A1> or any <S2> <A2>?',
                 'Have you [heard,listened to] any <S1> <A1> or any <S2> <A2>?',
                 'Did you [hear,listen to] a sound that [sounds like,is] [a,an] <S1> <A1> or a sound [sounds like,is] [a,an] <S2> <A2>?',  # noqa: E501
                 'Did you [hear,listen to] a sound that [sounded like,was] [a,an] <S1> <A1> or a sound [sounded like,was] [a,an] <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] a sound that [sounds like,is] [a,an] <S1> <A1> or a sound [sounds like,is] [a,an] <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] a sound that [sounded like,was] [a,an] <S1> <A1> or a sound [sounded like,was] [a,an] <S2> <A2>?',  # noqa: E501
                 'Was there [a,an] <S1> <A1> or [a,an] <S2> <A2>?',
                 'Were there any <S1>s <A1> or any <S2>s <A2>?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))  # sample source
    action_1 = str(np.random.choice(dataset['actions'][event_1]))  # sample action
    lst_events = [e for e in dataset['events'] if e != event_1]
    event_2 = str(np.random.choice(lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))  # sample source
    action_2 = str(np.random.choice(dataset['actions'][event_2]))  # sample action

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    answer = 'yes' if (event_1 in lst_events or event_2 in lst_events) else 'no'

    return question, answer


def was_there_source(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S>?',
                 'Have you [heard,listened to] [a,an] <S>?'
                 'Did you [hear,listen to] any <S>?',
                 'Have you [heard,listened to] any <S>?',
                 'Was there a sound [produced,made] by [a,an] <S>?',
                 'Were there any sounds [produced,made] by [a,an] <S>?',
                 ]

    question = str(np.random.choice(questions))  # sample question
    event = str(np.random.choice(dataset['events']))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))  # sample source

    question = question.replace('<S>', source)  # insert source
    question = sanitize_question(question)  # correct grammar

    answer = 'yes' if source in get_lst_all_sources(dataset, narrative) else 'no'

    return question, answer


def was_there_source_two_and(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> and [a,an] <S2>?',
                 'Have you [heard,listened to] [a,an] <S1> and [a,an] <S2>?'
                 'Did you [hear,listen to] any <S1> and any <S2>?',
                 'Have you [heard,listened to] any <S1> and any <S2>?',
                 'Was there a sound [produced,made] by [a,an] <S1> and a sound [produced,made] by [a,an] <S2>?',  # noqa: E501
                 'Were there any sounds [produced,made] by [a,an] <S1> and any sounds [produced,made] by [a,an] <S2>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))  # sample source
    lst_events = [e for e in dataset['events'] if e != event_1]
    event_2 = str(np.random.choice(lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))  # sample source

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<S2>', source_2)  # insert source
    question = sanitize_question(question)  # correct grammar

    lst_sources = get_lst_all_sources(dataset, narrative)
    answer = 'yes' if (source_1 in lst_sources and source_2 in lst_sources) else 'no'

    return question, answer


def was_there_source_two_or(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> or [a,an] <S2>?',
                 'Have you [heard,listened to] [a,an] <S1> or [a,an] <S2>?'
                 'Did you [hear,listen to] any <S1> or any <S2>?',
                 'Have you [heard,listened to] any <S1> or any <S2>?',
                 'Was there a sound [produced,made] by [a,an] <S1> or a sound [produced,made] by [a,an] <S2>?',  # noqa: E501
                 'Were there any sounds [produced,made] by [a,an] <S1> or any sounds [produced,made] by [a,an] <S2>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))  # sample source
    lst_events = [e for e in dataset['events'] if e != event_1]
    event_2 = str(np.random.choice(lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))  # sample source

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<S2>', source_2)  # insert source
    question = sanitize_question(question)  # correct grammar

    lst_sources = get_lst_all_sources(dataset, narrative)
    answer = 'yes' if (source_1 in lst_sources or source_2 in lst_sources) else 'no'

    return question, answer


def was_there_relative(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> <A1> <RO> the <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] [a,an] <S1> <A1> <RO> the <S2> <A2>?',  # noqa: E501
                 'Did you [hear,listen to] any <S1> <A1> <RO> the <S2> <A2>?',
                 'Have you [heard,listened to] any <S1> <A1> <RO> the <S2> <A2>?',
                 'Was there [a,an] <S1> <A1> <RO> the <S2> <A2>?',
                 'Were there any <S1>s <A1> <RO> the <S2> <A2>?',
                 'Did you [hear,listen to] a sound that [sounds like,sounded like,is,was] [a,an] <S1> <A1> <RO> the <S2> <A2>?',  # noqa: E501
                 '<RO> the <S2> <A2>, did you [hear,listen to] [a,an] <S1> <A1> ?',  # noqa: E501
                 '<RO> the <S2> <A2>, did you [hear,listen to] any <S1> <A1>?',
                 '<RO> the <S2> <A2>, was there [a,an] <S1> <A1>?',
                 '<RO> the <S2> <A2>, were there any <S1>s <A1>?',
                 '<RO> the <S2> <A2>, did you [hear,listen to] a sound that [sounds like,sounded like,is,was] [a,an] <S1> <A1>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_relative) illposed.'
    event_2 = str(np.random.choice(unique_lst_events))
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event_2) == 1, \
        'Question (was_there_relative) illposed.'

    event_2_idx = lst_events.index(event_2)
    if 'before' in preposition:
        lst_events = lst_events[:event_2_idx]
    elif 'after' in preposition:
        lst_events = lst_events[(event_2_idx + 1):]
    else:
        assert False, 'Preposition illdefined in Question (was_there_relative).'
    answer = 'yes' if event_1 in lst_events else 'no'

    return question, answer


def was_there_immediate_relative(dataset, narrative, _):
    questions = ['Did you [hear,listen to] [a,an] <S1> <A1> <IO> the <S2> <A2>?',  # noqa: E501
                 'Have you [heard,listened to] [a,an] <S1> <A1> <IO> the <S2> <A2>?',  # noqa: E501
                 'Did you [hear,listen to] any <S1> <A1> <IO> the <S2> <A2>?',
                 'Have you [heard,listened to] any <S1> <A1> <IO> the <S2> <A2>?',
                 'Was there [a,an] <S1> <A1> <IO> the <S2> <A2>?',
                 'Were there any <S1>s <A1> <IO> the <S2> <A2>?',
                 'Did you [hear,listen to] a sound that [sounds like,sounded like,is,was] [a,an] <S1> <A1> <IO> the <S2> <A2>?',  # noqa: E501
                 '<IO> the <S2> <A2>, did you [hear,listen to] [a,an] <S1> <A1> ?',  # noqa: E501
                 '<IO> the <S2> <A2>, did you [hear,listen to] any <S1> <A1>?',
                 '<IO> the <S2> <A2>, was there [a,an] <S1> <A1>?',
                 '<IO> the <S2> <A2>, were there any <S1>s <A1>?',
                 '<IO> the <S2> <A2>, did you [hear,listen to] a sound that [sounds like,sounded like,is,was] [a,an] <S1> <A1>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    preposition = sample_immediate_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    unique_lst_events = [e for e in unique_lst_events if e != event_1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_immediate_relative) illposed.'
    event_2 = str(np.random.choice(unique_lst_events))
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<IO>', preposition)  # insert preposition
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event_2) == 1, \
        'Question (was_there_immediate_relative) illposed.'

    event_2_idx = lst_events.index(event_2)
    if 'before' in preposition:
        if (event_2_idx - 1) < 0:
            target_event = []
        else:
            target_event = lst_events[event_2_idx - 1]
    elif 'after' in preposition:
        if (event_2_idx + 1) >= len(lst_events):
            target_event = []
        else:
            target_event = lst_events[event_2_idx + 1]
    else:
        assert False, \
            'Preposition illdefined in Question (was_there_immediate_relative).'
    answer = 'yes' if event_1 == target_event else 'no'

    return question, answer


def was_there_similar_ordinal(dataset, narrative, _):
    questions = ['Were there any similar sounds to the <O> sound?',
                 'Were there any sounds that were similar to the <O> sound?',
                 'Was there at least a sound similar to the <O> sound?',
                 'Was there at least a sound that was similar to the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound similar to the <O> sound?',
                 'Was there at least [one,a single] sound that was similar to the <O> sound?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    event = lst_events[number - 1]
    answer = 'yes' if lst_events.count(event) > 1 else 'no'  # 1 for reference

    return question, answer


def was_there_similar_loudness(dataset, narrative, rel_diff=0.1):
    questions = ['Were there any sounds [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'Was there any sound [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] the same loudness as <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] the same loudness as <S> <A>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_loudness()  # sample loudness
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_similar_loudness) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (was_there_similar_loudness) illposed.'

    lst_loudness = get_lst_loudness(narrative)
    event_idx = lst_events.index(event)
    evt_loudness = lst_loudness[event_idx]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != event_idx]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_similar_loudness) illposed.'
    answer = 'yes' if np.sum(rel_loudness_diff <= rel_diff) >= 1 else 'no'

    return question, answer


def was_there_at_least_two_similar_loudness(dataset, narrative, rel_diff=0.1):
    questions = ['Were there at least two sounds [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 ]
    question = str(np.random.choice(questions))  # sample question
    loudness = sample_loudness()  # sample loudness
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_at_least_two_similar_loudness) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (was_there_at_least_two_similar_loudness) illposed.'

    lst_loudness = get_lst_loudness(narrative)
    event_idx = lst_events.index(event)
    evt_loudness = lst_loudness[event_idx]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != event_idx]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_at_least_two_similar_loudness) illposed.'
    answer = 'yes' if np.sum(rel_loudness_diff <= rel_diff) >= 2 else 'no'

    return question, answer


def was_there_similar_loudness_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Were there any sounds [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'Was there any sound [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'Was there at least a sound [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_loudness()  # sample loudness
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    evt_loudness = lst_loudness[number - 1]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != (number - 1)]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_similar_loudness_ordinal) illposed.'
    answer = 'yes' if np.sum(rel_loudness_diff <= rel_diff) >= 1 else 'no'

    return question, answer


def was_there_at_least_two_similar_loudness_ordinal(dataset,
                                                    narrative,
                                                    rel_diff=0.1):
    questions = ['Were there at least two sounds [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'Was there more than a sound [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_loudness()  # sample loudness
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_loudness = get_lst_loudness(narrative)
    evt_loudness = lst_loudness[number - 1]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != (number - 1)]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_at_least_two_similar_loudness_ordinal) illposed.'
    answer = 'yes' if np.sum(rel_loudness_diff <= rel_diff) >= 2 else 'no'

    return question, answer


def was_there_similar_duration(dataset, narrative, rel_diff=0.1):
    questions = ['Were there any sounds [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'Was there any sound [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] the same duration as <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] the same duration as <S> <A>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_duration()  # sample duration
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_similar_duration) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (was_there_similar_duration) illposed.'

    lst_durations = get_lst_durations(narrative)
    event_idx = lst_events.index(event)
    evt_duration = lst_durations[event_idx]
    x_durations = [j for i, j in enumerate(lst_durations) if i != event_idx]
    rel_durations_diff = compute_rel_diff(np.array(x_durations),
                                          np.array(evt_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_similar_duration) illposed.'
    answer = 'yes' if np.sum(rel_durations_diff <= rel_diff) >= 1 else 'no'

    return question, answer


def was_there_at_least_two_similar_duration(dataset, narrative, rel_diff=0.1):
    questions = ['Were there at least two sounds [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_duration()  # sample duration
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (was_there_at_least_two_similar_duration) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (was_there_at_least_two_similar_duration) illposed.'

    lst_durations = get_lst_durations(narrative)
    event_idx = lst_events.index(event)
    evt_duration = lst_durations[event_idx]
    x_durations = [j for i, j in enumerate(lst_durations) if i != event_idx]
    rel_durations_diff = compute_rel_diff(np.array(x_durations),
                                          np.array(evt_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_at_least_two_similar_duration) illposed.'
    answer = 'yes' if np.sum(rel_durations_diff <= rel_diff) >= 2 else 'no'

    return question, answer


def was_there_similar_duration_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['Were there any sounds [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Were there any sounds that were [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'Was there any sound [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there any sound that was [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'Was there at least a sound [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there at least a sound that was [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there at least [one,a single] sound that was [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_duration()  # sample duration
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_durations = get_lst_durations(narrative)
    evt_duration = lst_durations[number - 1]
    x_durations = [j for i, j in enumerate(lst_durations) if i != (number - 1)]
    rel_durations_diff = compute_rel_diff(np.array(x_durations),
                                          np.array(evt_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_similar_duration_ordinal) illposed.'
    answer = 'yes' if np.sum(rel_durations_diff <= rel_diff) >= 1 else 'no'

    return question, answer


def was_there_at_least_two_similar_duration_ordinal(dataset,
                                                    narrative,
                                                    rel_diff=0.1):
    questions = ['Were there at least two sounds [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Were there at least two sounds that were [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'Was there more than a sound [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there more than a sound that was [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'Was there more than [one,a single] sound that was [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_duration()  # sample duration
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    lst_durations = get_lst_durations(narrative)
    evt_duration = lst_durations[number - 1]
    x_durations = [j for i, j in enumerate(lst_durations) if i != (number - 1)]
    rel_durations_diff = compute_rel_diff(np.array(x_durations),
                                          np.array(evt_duration))
    # Assert a good margin in relative duration
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (was_there_at_least_two_similar_duration_ordinal) illposed.'
    answer = 'yes' if np.sum(rel_durations_diff <= rel_diff) >= 2 else 'no'

    return question, answer
