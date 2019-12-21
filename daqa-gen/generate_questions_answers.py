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
import random
import numpy as np

from qpas.exist import (was_there,
                        was_there_two_and,
                        was_there_two_or,
                        # was_there_source,
                        # was_there_source_two_and,
                        # was_there_source_two_or,
                        was_there_relative,
                        was_there_immediate_relative,
                        was_there_similar_ordinal,
                        was_there_similar_loudness,
                        was_there_at_least_two_similar_loudness,
                        was_there_similar_loudness_ordinal,
                        was_there_at_least_two_similar_loudness_ordinal,
                        was_there_similar_duration,
                        was_there_at_least_two_similar_duration,
                        was_there_similar_duration_ordinal,
                        was_there_at_least_two_similar_duration_ordinal,
                        )
from qpas.query import (what_was,
                        what_was_relative,
                        what_was_loudness,
                        what_was_loudness_relative,
                        what_was_loudness_relative_ordinal,
                        what_was_duration,
                        what_was_duration_relative,
                        what_was_duration_relative_ordinal,
                        )
from qpas.count import (how_many,
                        how_many_event,
                        how_many_ordinal,
                        how_many_event_two,
                        how_many_event_two_ordinal,
                        how_many_sounds_relative,
                        how_many_sounds_relative_ordinal,
                        how_many_event_relative,
                        how_many_event_relative_ordinal,
                        how_many_sounds_loudness_event,
                        how_many_sounds_loudness_ordinal,
                        how_many_sounds_duration_event,
                        how_many_sounds_duration_ordinal,
                        )
from qpas.compare import (compare_ordinal,
                          compare_ordinal_event,
                          compare_loudness,
                          compare_loudness_ordinal,
                          compare_loudness_event_ordinal,
                          compare_loudness_ordinal_event,
                          compare_same_loudness,
                          compare_same_loudness_ordinal,
                          compare_same_loudness_event_ordinal,
                          compare_duration,
                          compare_duration_ordinal,
                          compare_duration_event_ordinal,
                          compare_duration_ordinal_event,
                          compare_same_duration,
                          compare_same_duration_ordinal,
                          compare_same_duration_event_ordinal,
                          )
from qpas.compare_integer import (less_than,
                                  equal_to,
                                  more_than,
                                  )

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--dataset', default='daqa.json', type=str,
                    help='JSON file describing the dataset.')
parser.add_argument('--input_narrative_file',
                    default='../daqa/daqa_narratives.json',
                    help="Path to narratives JSON file.")
parser.add_argument('--start_narrative_idx', default=0, type=int,
                    help='Start reading from start_narrative_idx.')

# Settings
parser.add_argument('--set', default='new',
                    help='Set name: train / val / test.')
parser.add_argument('--num_questions_per_narrative', default=10, type=int,
                    help='Number of questions per narrative.')
parser.add_argument('--patience_narrative', default=10, type=int,
                    help='Number of failed attempts to reach num_q_per_narr.')
parser.add_argument('--patience_template', default=10, type=int,
                    help='Number of failed attempts to reach num_q_per_narr.')
parser.add_argument('--rel_diff', default=0.1, type=int,
                    help='Loudness sensitivity (%).')
parser.add_argument('--max_diff', default=0.05, type=float,
                    help='Maximum difference between (in)frequent answers.')
parser.add_argument('--seed', default=0, type=int, help='Random Seed.')
parser.add_argument('--version', default='1.0', type=str, help='Version.')
parser.add_argument('--license',
                    default='Creative Commons Attribution (CC-BY 4.0)',
                    help='License.')
parser.add_argument('--date',
                    default=datetime.datetime.today().strftime("%m/%d/%Y"),
                    help="Date.")

# Output
parser.add_argument('--start_output_idx', default=0, type=int,
                    help='Start numbering from start_output_idx.')
parser.add_argument('--output_qa_file',
                    default='../daqa/daqa_questions_answers.json',
                    help="Path to questions answers JSON file.")


def tokenize_answer(dataset, ans):
    # Tokenize answer
    anss = ans.split(' ')
    for e in dataset['events']:
        lst_syn = dataset['sources'][e] + dataset['actions'][e]
        lst_syn = ' '.join(s for s in lst_syn)
        lst_check = []
        for a in anss:
            lst_check.append((' ' + a + ' ') in (' ' + lst_syn + ' '))
        if all(lst_check):
            ans = e
    return ans


def add_answer(ans_dist_per_temp, ques_temp, ans_tk, max_diff):
    # Only one answer seen so far for this template
    if len(ans_dist_per_temp[ques_temp].keys()) <= 1:
        return True
    # First instance of this answer in this template
    if ans_dist_per_temp[ques_temp][ans_tk] == 0:
        return True
    num_occ = sorted(((v, k) for k, v in ans_dist_per_temp[ques_temp].items()))
    # Not the most frequent answer
    if num_occ[-1][1] != ans_tk:
        return True
    # Difference between the (most + 1) and least frequent is less than max_diff
    if ((num_occ[-1][0] + 1) - num_occ[0][0]) <= max_diff:
        return True
    return False


def main(args):
    """Randomly sample questions for given narrative and deduce answer."""

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Read dataset description and narratives
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    with open(args.input_narrative_file, 'r') as f:
        narratives = json.load(f)

    assert args.set == narratives['info']['set'], 'train/val/test mismatch.'

    templates = [was_there,
                 was_there_two_and,
                 was_there_two_or,
                 # was_there_source,
                 # was_there_source_two_and,
                 # was_there_source_two_or,
                 was_there_relative,
                 was_there_immediate_relative,
                 was_there_similar_ordinal,
                 was_there_similar_loudness,
                 was_there_at_least_two_similar_loudness,
                 was_there_similar_loudness_ordinal,
                 was_there_at_least_two_similar_loudness_ordinal,
                 was_there_similar_duration,
                 was_there_at_least_two_similar_duration,
                 was_there_similar_duration_ordinal,
                 was_there_at_least_two_similar_duration_ordinal,
                 what_was,
                 what_was_relative,
                 what_was_loudness,
                 what_was_loudness_relative,
                 what_was_loudness_relative_ordinal,
                 what_was_duration,
                 what_was_duration_relative,
                 what_was_duration_relative_ordinal,
                 how_many,
                 how_many_event,
                 how_many_ordinal,
                 how_many_event_two,
                 how_many_event_two_ordinal,
                 how_many_sounds_relative,
                 how_many_sounds_relative_ordinal,
                 how_many_event_relative,
                 how_many_event_relative_ordinal,
                 how_many_sounds_loudness_event,
                 how_many_sounds_loudness_ordinal,
                 how_many_sounds_duration_event,
                 how_many_sounds_duration_ordinal,
                 compare_ordinal,
                 compare_ordinal_event,
                 compare_loudness,
                 compare_loudness_ordinal,
                 compare_loudness_event_ordinal,
                 compare_loudness_ordinal_event,
                 compare_same_loudness,
                 compare_same_loudness_ordinal,
                 compare_same_loudness_event_ordinal,
                 compare_duration,
                 compare_duration_ordinal,
                 compare_duration_event_ordinal,
                 compare_duration_ordinal_event,
                 compare_same_duration,
                 compare_same_duration_ordinal,
                 compare_same_duration_event_ordinal,
                 less_than,
                 equal_to,
                 more_than,
                 ]

    print('Generating ' + str(args.num_questions_per_narrative)
          + ' questions for each of the ' + str(len(narratives['narratives']))
          + ' narratives.')
    idx = args.start_output_idx
    lst_questions = []
    num_skewed_answers = 0
    num_illposed_questions = 0
    ans_dist_per_temp = {}
    # The delta between (in)frequent answers is irrespective of the set size
    max_diff = (args.max_diff
        * ((len(narratives['narratives']) - args.start_narrative_idx)
        * args.num_questions_per_narrative) / len(templates))

    for n in range(args.start_narrative_idx, len(narratives['narratives'])):
        narrative = narratives['narratives'][n]
        num_questions, patience_narrative = 0, 0
        while num_questions < args.num_questions_per_narrative:
            question_template = random.choice(templates)
            try:  # catch illposed questions
                patience_template = 0
                while patience_template < args.patience_template:
                    ques, ans = question_template(dataset, narrative, args.rel_diff)

                    ans_tk = tokenize_answer(dataset, ans)

                    ques_temp_name = question_template.__name__
                    if ques_temp_name not in ans_dist_per_temp:
                        ans_dist_per_temp[ques_temp_name] = {}
                    if ans_tk not in ans_dist_per_temp[ques_temp_name]:
                        ans_dist_per_temp[ques_temp_name][ans_tk] = 0

                    if add_answer(ans_dist_per_temp, ques_temp_name,
                                  ans_tk, max_diff):
                        question = {
                            'set': narrative['set'],
                            'audio_index': narrative['audio_index'],
                            'audio_filename': narrative['audio_filename'],
                            'question_template': ques_temp_name,
                            'question': ques,
                            'answer': ans,
                            'answer_token': ans_tk,
                        }
                        lst_questions.append(question)
                        ans_dist_per_temp[ques_temp_name][ans_tk] += 1
                        idx += 1
                        num_questions += 1
                        break
                    else:
                        patience_template += 1
                        num_skewed_answers += 1
                        if patience_template >= args.patience_template:
                            print('R1. Out of patience for narrative #' + str(n)
                                  + ' for template: ' + ques_temp_name + '.')

            except AssertionError as error:
                print(error)
                patience_narrative += 1
                num_illposed_questions += 1
                if patience_narrative >= args.patience_narrative:
                    print('R2. Out of patience for narrative #' + str(n) + '.')
                    break

    print('Generated ' + str(idx) + ' questions.')
    print('Failed to generate ' + str(num_skewed_answers) + ' questions.'
          + ' Reason: skewed answers.')
    print('Failed to generate ' + str(num_illposed_questions) + ' questions.'
          + ' Reason: illposed questions.')
    print('Total number of attempts: '
          + str(idx + num_skewed_answers + num_illposed_questions))

    output = {
        'info': {
            'set': args.set,
            'version': args.version,
            'date': args.date,
            'license': args.license,
        },
        'questions': lst_questions
    }
    with open(args.output_qa_file, 'w') as f:
        json.dump(output, f)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print('Success!')
