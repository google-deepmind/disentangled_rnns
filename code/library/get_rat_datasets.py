# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Load rat datasets."""

import json
from typing import List

import numpy as np

from disentangled_rnns.code.library import rnn_utils


def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_datasets(data_file: str,
                     ) -> List[rnn_utils.DatasetRNN]:
  """Packages up the rat datasets.

  Requires downloading the dataset file "tab_dataset.json", which is available
  on Figshare.

  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

  Args:
    data_file: Complete path to the dataset file, including the filename.

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be an episode, padded with NaNs
    to match length. "ys" will be the choices on each trial
    (left=0, right=1) "xs" will be the choice and reward (0 or 1) from
    the previous trial. Invalid xs and ys will be -1
  """

  with open(data_file, 'r') as f:
    dataset = json.load(f)

  n_rats = len(dataset)

  dataset_list = []
  # Each creates a DatasetRNN object for a single rat, adds it to the list
  for rat_i in range(n_rats):
    ratdata = dataset[rat_i]
    sides = ratdata['sides']
    n_trials = len(sides)

    # Left choices will be 0s, right choices will be 1s, viols will be removed
    rights = find(sides, 'r')
    choices = np.zeros(n_trials)
    choices[rights] = 1

    vs = find(sides, 'v')
    viols = np.zeros(n_trials, dtype=bool)
    viols[vs] = True

    # Free will be 0 and forced will be 1
    free = find(ratdata['trial_types'], 'f')
    instructed_choice = np.ones(n_trials)
    instructed_choice[free] = 0

    rewards = np.array(ratdata['rewards'])
    new_sess = np.array(ratdata['new_sess'])

    n_sess = np.sum(new_sess)
    sess_starts = np.nonzero(np.concatenate((new_sess, [1])))[0]
    max_session_length = np.max(np.diff(sess_starts, axis=0))

    # Populate matrices for rewards and choices. size (n_trials, n_sessions, 1)
    rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    instructed_by_session = -1 * np.ones((max_session_length, n_sess, 1))

    # Each iteration processes one session
    for sess_i in np.arange(n_sess):
      sess_start = sess_starts[sess_i]
      sess_end = sess_starts[sess_i + 1]

      viols_sess = viols[sess_start:sess_end]
      rewards_sess = rewards[sess_start:sess_end]
      choices_sess = choices[sess_start:sess_end]
      instructed_choice_sess = instructed_choice[sess_start:sess_end]

      rewards_sess = np.delete(rewards_sess, viols_sess)
      choices_sess = np.delete(choices_sess, viols_sess)
      instructed_choice_sess = np.delete(instructed_choice_sess, viols_sess)

      sess_length_noviols = len(choices_sess)

      rewards_by_session[0:sess_length_noviols, sess_i, 0] = rewards_sess
      choices_by_session[0:sess_length_noviols, sess_i, 0] = choices_sess
      instructed_by_session[0:sess_length_noviols, sess_i, 0] = (
          instructed_choice_sess
      )

    # Inputs: choices and rewards, offset by one trial
    choice_and_reward = np.concatenate(
        (choices_by_session, rewards_by_session), axis=2
    )
    # Add a dummy input at the beginning. First step has a target but no input
    xs = np.concatenate(
        (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
    )
    # Targets: choices on each free-choice trial
    free_choices = choices_by_session
    free_choices[instructed_by_session == 1] = -1
    # Add a dummy target at the end -- last step has input but no target
    ys = np.concatenate((free_choices, -1*np.ones((1, n_sess, 1))), axis=0)

    dataset_rat = rnn_utils.DatasetRNN(ys=ys, xs=xs)
    dataset_list.append(dataset_rat)

  return dataset_list
