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
                        get_lst_loudness, numbers_to_words, sample_duration,
                        sample_loudness, sample_number, sample_second_number,
                        sample_preposition, sanitize_question)


def how_many(dataset, narrative, _):
    questions = ['How many [sound events,sounds] were there?',
                 'How many [sound events,sounds] [did,could] you [hear,listen to]?',
                 'How many [sound events,sounds] have you [heard,listened to]?',
                 'What is the number of [sound events,sounds]?',
                 'What is the number of [sound events,sounds] [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] have you [heard,listened to]?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question

    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    answer = numbers_to_words(len(lst_events))

    return question, answer


def how_many_event(dataset, narrative, _):
    questions = ['How many times was [a,an] <S> <A>?',
                 'How many times did you [hear,listen to] [a,an] <S> <A>?',
                 'How many times have you [heard,listened to] [a,an] <S> <A>?',
                 'What is the number of times [a,an] <S> <A>?',
                 'What is the number of times did you [hear,listen to] [a,an] <S> <A>?',  # noqa: E501
                 'What is the number of times you [heard,listened to] [a,an] <S> <A>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    event = str(np.random.choice(lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    answer = numbers_to_words(lst_events.count(event))

    return question, answer


def how_many_ordinal(dataset, narrative, _):
    questions = ['How many times did you [hear,listen to] a sound that [sounded,seemed] like the <O> [sound event,sound]?',  # noqa: E501
                 'What is the number of times did you [hear,listen to] a sound that [sounded,seemed] like the <O> [sound event,sound]?',  # noqa: E501
                 '[Hearing,Listening to] the <O> [sound event,sound], how many sounds were [the same, similar]?',  # noqa: E501
                 '[Hearing,Listening to] the <O> [sound event,sound], what is the number of sounds that were [the same, similar]?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    event = lst_events[number - 1]
    answer = numbers_to_words(lst_events.count(event) - 1)  # -1 for base event

    return question, answer


def how_many_event_two(dataset, narrative, _):
    questions = ['How many times was [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',
                 'How many times did you [hear,listen to] [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',  # noqa: E501
                 'How many times have you [heard,listened to] [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',  # noqa: E501
                 'What is the number of times [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',  # noqa: E501
                 'What is the number of times did you [hear,listen to] [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',  # noqa: E501
                 'What is the number of times you [heard,listened to] [a,an] <S1> <A1> [or,and] [a,an] <S2> <A2>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event_1 = str(np.random.choice(dataset['events']))  # sample event
    source_1 = str(np.random.choice(dataset['sources'][event_1]))
    action_1 = str(np.random.choice(dataset['actions'][event_1]))
    x_lst_events = [e for e in dataset['events'] if e != event_1]
    event_2 = str(np.random.choice(x_lst_events))  # sample event
    source_2 = str(np.random.choice(dataset['sources'][event_2]))
    action_2 = str(np.random.choice(dataset['actions'][event_2]))

    assert event_1 != event_2, 'Question (how_many_event_two) illposed.'

    question = question.replace('<S1>', source_1)  # insert source
    question = question.replace('<A1>', action_1)  # insert action
    question = question.replace('<S2>', source_2)  # insert source
    question = question.replace('<A2>', action_2)  # insert action
    question = sanitize_question(question)  # correct grammar

    lst_events = get_lst_events(narrative)
    answer = numbers_to_words(lst_events.count(event_1)
                              + lst_events.count(event_2))

    return question, answer


def how_many_event_two_ordinal(dataset, narrative, _):
    questions = ['How many times did you [hear,listen to] a sound that [sounded,seemed] like the <O1> [sound event,sound] [or,and] the <O2> [sound event,sound]?',  # noqa: E501
                 'What is the number of times did you [hear,listen to] a sound that [sounded,seemed] like the <O1> [sound event,sound] [or,and] the <O2> [sound event,sound]?',  # noqa: E501
                 '[Hearing,Listening to] the <O1> [sound event,sound] and the <O2> [sound event,sound], how many sounds were [the same,similar]?',  # noqa: E501
                 '[Hearing,Listening to] the <O1> [sound event,sound] and the <O2> [sound event,sound], what is the number of sounds that were [the same,similar]?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number_1, ordinal_1 = sample_number(len(lst_events))
    number_2, ordinal_2 = sample_second_number(len(lst_events), number_1)

    question = question.replace('<O1>', ordinal_1)  # insert ordinal
    question = question.replace('<O2>', ordinal_2)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    event_1 = lst_events[number_1 - 1]
    event_2 = lst_events[number_2 - 1]
    answer = numbers_to_words((lst_events.count(event_1) - 1)   # -1 for base event
                              + (lst_events.count(event_2) - 1))

    return question, answer


def how_many_sounds_relative(dataset, narrative, _):
    questions = ['How many [sound events,sounds] <RO> the <S> <A> were there?',
                 'How many [sound events,sounds] <RO> the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] <RO> the <S> <A> have you [heard,listened to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] <RO> the <S> <A>?',
                 'What is the number of [sound events,sounds] <RO> the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] <RO> the <S> <A> have you [heard,listened to]?',  # noqa: E501
                 'There is [a,an] <S> <A>; how many [sound events,sounds] [did,could] you hear <RO>?',  # noqa: E501
                 'There is [a,an] <S> <A>; how many [sound events,sounds] have you heard <RO>?',  # noqa: E501
                 'There is [a,an] <S> <A>; what is the number of [sound events,sounds] [did,could] you hear <RO>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (how_many_sounds_relative) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (how_many_sounds_relative) illposed.'

    event_idx = lst_events.index(event)
    if 'before' in question:
        lst_events_e = lst_events[:event_idx]
    elif 'after' in question:
        lst_events_e = lst_events[(event_idx + 1):]
    else:
        assert False, \
            'Preposition illdefined in Question (how_many_sounds_relative).'
    answer = numbers_to_words(len(lst_events_e))

    return question, answer


def how_many_sounds_relative_ordinal(dataset, narrative, _):
    questions = ['How many [sound events,sounds] after the <O> [sound event,sound] were there?',  # noqa: E501
                 'How many [sound events,sounds] after the <O> [sound event,sound] [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] after the <O> [sound event,sound] have you [heard,listened to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] after the <O> [sound event,sound]?',  # noqa: E501
                 'What is the number of [sound events,sounds] after the <O> [sound event,sound] [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] after the <O> [sound event,sound] have you [heard,listened to]?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    assert number < (len(lst_events) - 1), \
        'Question (how_many_sounds_relative_ordinal) illposed.'
    lst_events_e = lst_events[number:]
    answer = numbers_to_words(len(lst_events_e))

    return question, answer


def how_many_event_relative(dataset, narrative, _):
    questions = ['How many <S1>s <A1> <RO> the <S2> <A2> were there?',
                 'How many <S1>s <A1> <RO> the <S2> <A2> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many <S1>s <A1> <RO> the <S2> <A2> have you [heard,listened to]?',  # noqa: E501
                 'What is the number of <S1>s <A1> <RO> the <S2> <A2>?',
                 'What is the number of <S1>s <A1> <RO> the <S2> <A2> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of <S1>s <A1> <RO> the <S2> <A2> have you [heard,listened to]?',  # noqa: E501
                 'There is [a,an] <S2> <A2>; how many <S1>s <A1> [did,could] you hear <RO>?',  # noqa: E501
                 'There is [a,an] <S2> <A2>; how many <S1>s <A1> have you heard <RO>?',  # noqa: E501
                 'There is [a,an] <S2> <A2>; what is the number of <S1>s <A1> [did,could] you hear <RO>?',  # noqa: E501
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
        'Question (how_many_event_relative) illposed.'
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
        'Question (how_many_event_relative) illposed.'

    event_2_idx = lst_events.index(event_2)
    if 'before' in question:
        lst_events_e = lst_events[:event_2_idx]
    elif 'after' in question:
        lst_events_e = lst_events[(event_2_idx + 1):]
    else:
        assert False, \
            'Relative preposition illdefined in Question (how_many_event_relative).'
    answer = numbers_to_words(lst_events_e.count(event_1))

    return question, answer


def how_many_event_relative_ordinal(dataset, narrative, _):
    questions = ['How many <S>s <A> <RO> the <O> [sound event,sound] were there?',
                 'How many <S>s <A> <RO> the <O> [sound event,sound] [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many <S>s <A> <RO> the <O> [sound event,sound] have you [heard,listened to]?',  # noqa: E501
                 'What is the number of <S>s <A> <RO> the <O> [sound event,sound]?',
                 'What is the number of <S>s <A> <RO> the <O> [sound event,sound] [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of <S>s <A> <RO> the <O> [sound event,sound] have you [heard,listened to]?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    event = str(np.random.choice(dataset['events']))  # sample event
    source = str(np.random.choice(dataset['sources'][event]))
    action = str(np.random.choice(dataset['actions'][event]))
    preposition = sample_preposition()
    lst_events = get_lst_events(narrative)
    number, ordinal = sample_number(len(lst_events))

    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = question.replace('<RO>', preposition)  # insert preposition
    question = question.replace('<O>', ordinal)  # insert ordinal
    question = sanitize_question(question)  # correct grammar

    if 'before' in question:
        assert number > 1, 'Question (how_many_event_relative_ordinal) illposed.'
        lst_events_e = lst_events[:(number - 1)]
    elif 'after' in question:
        assert number < (len(lst_events) - 1), \
            'Question (how_many_event_relative_ordinal) illposed.'
        lst_events_e = lst_events[number:]
    else:
        assert False, \
            'Relative preposition illdefined in Question (how_many_event_relative_ordinal).'   # noqa: E501
    answer = numbers_to_words(lst_events_e.count(event))

    return question, answer


def how_many_sounds_loudness_event(dataset, narrative, rel_diff=0.1):
    questions = ['How many [sound events,sounds] [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <L> as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <L> as the <S> <A> have you heard?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A> have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] [roughly,approximately] as <L> as the <S> <A>?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <L> as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <L> as the <S> <A> have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A>?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <S> <A> have you heard?',  # noqa: E501
                 'There is [a,an] <S> <A>; how many [sound events,sounds] that are [roughly,approximately] as <L>?',  # noqa: E501
                 'There is [a,an] <S> <A>; what is the number of [sound events,sounds] that are [roughly,approximately] as <L>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    loudness = sample_loudness()  # sample loudness
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (how_many_sounds_loudness_event) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<L>', loudness)  # insert loudness
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (how_many_sounds_loudness_event) illposed.'

    lst_loudness = get_lst_loudness(narrative)
    event_idx = lst_events.index(event)
    evt_loudness = lst_loudness[event_idx]
    x_loudness = [j for i, j in enumerate(lst_loudness) if i != event_idx]
    rel_loudness_diff = compute_rel_diff(np.array(x_loudness),
                                         np.array(evt_loudness))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_loudness_diff > rel_diff,
                                 rel_loudness_diff < (2 * rel_diff))) <= 0, \
        'Question (how_many_sounds_loudness_event) illposed.'
    answer = numbers_to_words(np.sum(rel_loudness_diff <= rel_diff))

    return question, answer


def how_many_sounds_loudness_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['How many [sound events,sounds] [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound have you heard?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <L> as the <O> sound have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same loudness as the <O> sound have you heard?',  # noqa: E501
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
        'Question (how_many_sounds_loudness_ordinal) illposed.'
    answer = numbers_to_words(np.sum(rel_loudness_diff <= rel_diff))

    return question, answer


def how_many_sounds_duration_event(dataset, narrative, rel_diff=0.1):
    questions = ['How many [sound events,sounds] [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <D> as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <D> as the <S> <A> have you heard?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A> have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] [roughly,approximately] as <D> as the <S> <A>?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <D> as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <D> as the <S> <A> have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A>?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A> [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <S> <A> have you heard?',  # noqa: E501
                 'There is [a,an] <S> <A>; how many [sound events,sounds] that are [roughly,approximately] as <D>?',  # noqa: E501
                 'There is [a,an] <S> <A>; what is the number of [sound events,sounds] that are [roughly,approximately] as <D>?',  # noqa: E501
                 ]

    question = str(np.random.choice(questions))  # sample question
    duration = sample_duration()  # sample duration
    lst_events = get_lst_events(narrative)
    unique_lst_events = [e for e in lst_events if lst_events.count(e) == 1]
    assert len(unique_lst_events) > 0, \
        'Question (how_many_sounds_duration_event) illposed.'
    event = str(np.random.choice(unique_lst_events))
    source = str(np.random.choice(dataset['sources'][event]))  # sample source
    action = str(np.random.choice(dataset['actions'][event]))  # sample action

    question = question.replace('<D>', duration)  # insert duration
    question = question.replace('<S>', source)  # insert source
    question = question.replace('<A>', action)  # insert action
    question = sanitize_question(question)  # correct grammar

    assert lst_events.count(event) == 1, \
        'Question (how_many_sounds_duration_event) illposed.'

    lst_durations = get_lst_durations(narrative)
    event_idx = lst_events.index(event)
    evt_duration = lst_durations[event_idx]
    x_durations = [j for i, j in enumerate(lst_durations) if i != event_idx]
    rel_durations_diff = compute_rel_diff(np.array(x_durations),
                                          np.array(evt_duration))
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (how_many_sounds_duration_event) illposed.'
    answer = numbers_to_words(np.sum(rel_durations_diff <= rel_diff))

    return question, answer


def how_many_sounds_duration_ordinal(dataset, narrative, rel_diff=0.1):
    questions = ['How many [sound events,sounds] [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound have you heard?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'How many [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that are [roughly,approximately] as <D> as the <O> sound have you heard?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound [did,could] you [hear,listen to]?',  # noqa: E501
                 'What is the number of [sound events,sounds] that have [roughly,approximately] the same duration as the <O> sound have you heard?',  # noqa: E501
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
    # Assert a good margin in relative loudness
    assert np.sum(np.logical_and(rel_durations_diff > rel_diff,
                                 rel_durations_diff < (2 * rel_diff))) <= 0, \
        'Question (how_many_sounds_duration_ordinal) illposed.'
    answer = numbers_to_words(np.sum(rel_durations_diff <= rel_diff))

    return question, answer
