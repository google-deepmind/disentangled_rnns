# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load datasets."""

import json
import os
from typing import Optional, Literal, cast
import urllib.request

from disentangled_rnns.library import pclicks
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import numpy as np
import scipy.io


def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_bandit_dataset(rat_i: int = 0) -> rnn_utils.DatasetRNN:
  """Downloads and packages rat two-armed bandit datasets.

  Dataset is from the following paper:
  From predictive models to cognitive models: Separable behavioral processes
  underlying reward learning in the rat. Miller, Botvinick, and Brody,
  bioRxiv, 2018

  Dataset is available from Figshare at the following link:
  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

  Args:
    rat_i: Integer specifying which rat to load.

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be an episode, padded with -1s
    to match length. "ys" will be the choices on each trial
    (left=0, right=1) "xs" will be the choice and reward (0 or 1) from
    the previous trial. Invalid xs and ys will be -1
  """

  # Download the file to the current folder
  url = 'https://figshare.com/ndownloader/files/40442660'
  data_path = 'tab_dataset.json'
  urllib.request.urlretrieve(url, data_path)

  # Load dataset into memory
  with open(data_path, 'r') as f:
    dataset = json.load(f)

  # Clean up after ourselves by removing the downloaded file
  os.remove(data_path)

  # "dataset" will be a list in which each element is a dict. Each of these
  # dicts holds data from a single rat.

  ratdata = dataset[rat_i]
  # "sides" is a list of characters in which each character specifies the
  # choice made on a trial. 'r' for right, 'l' for left, 'v' for a violation
  # Here, we'll code left choices as 0s, right choices as 1s, and will flag
  # violations for later removal
  sides = ratdata['sides']
  n_trials = len(sides)
  rights = find(sides, 'r')
  choices = np.zeros(n_trials)
  choices[rights] = 1
  vs = find(sides, 'v')
  viols = np.zeros(n_trials, dtype=bool)
  viols[vs] = True

  # Free will be 0 and forced will be 1
  # "trial_types" is a list of characters, each giving the type of a trial.
  # 'f' for free-choice, 'l' for instructed-left, 'r' for instructed-right
  free = find(ratdata['trial_types'], 'f')
  instructed_choice = np.ones(n_trials)
  instructed_choice[free] = 0

  # "rewards" is a list of 1s (rewarded trials) and 0s (unrewarded trials)
  rewards = np.array(ratdata['rewards'])

  # "new_sess" is a list of 1s (trials that are the first of a new session)
  # and 0s (all other trials)
  new_sess = np.array(ratdata['new_sess'])
  n_sess = np.sum(new_sess)
  sess_starts = np.nonzero(np.concatenate((new_sess, [1])))[0]
  # We will pad each session so they all have length of the longest session
  max_session_length = np.max(np.diff(sess_starts, axis=0))

  # Instantiate blank matrices for rewards and choices.
  # size (n_trials, n_sessions, 1)
  rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
  choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))
  instructed_by_session = -1 * np.ones((max_session_length, n_sess, 1))

  # Each iteration processes one session
  for sess_i in np.arange(n_sess):
    # Get the choices, rewards, viols, and instructed for just this session
    sess_start = sess_starts[sess_i]
    sess_end = sess_starts[sess_i + 1]
    viols_sess = viols[sess_start:sess_end]
    rewards_sess = rewards[sess_start:sess_end]
    choices_sess = choices[sess_start:sess_end]
    instructed_choice_sess = instructed_choice[sess_start:sess_end]

    # Remove violation trials
    rewards_sess = np.delete(rewards_sess, viols_sess)
    choices_sess = np.delete(choices_sess, viols_sess)
    instructed_choice_sess = np.delete(instructed_choice_sess, viols_sess)
    sess_length_noviols = len(choices_sess)

    # Add them to the matrices
    rewards_by_session[0:sess_length_noviols, sess_i, 0] = rewards_sess
    choices_by_session[0:sess_length_noviols, sess_i, 0] = choices_sess
    instructed_by_session[0:sess_length_noviols, sess_i, 0] = (
        instructed_choice_sess
    )

  # Define neural network inputs:
  # for each trial the choice and reward from the previous trial.
  choice_and_reward = np.concatenate(
      (choices_by_session, rewards_by_session), axis=2
  )
  xs = np.concatenate(
      (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
  )  # First trial's input will arbitrarily always be 0

  # Define neural network targets:
  # choices on all free-choice trial, -1 on all instructed trials
  free_choices = choices_by_session
  free_choices[instructed_by_session == 1] = -1
  # Add a dummy target at the end -- last step has input but no target
  ys = np.concatenate((free_choices, -1*np.ones((1, n_sess, 1))), axis=0)

  # Pack into a DatasetRNN object
  dataset_rat = rnn_utils.DatasetRNN(ys=ys, xs=xs, y_type='categorical')

  return dataset_rat


def get_pclicks_dataset(rat_i: int = 0) -> rnn_utils.DatasetRNN:
  """Packages up rat poisson clicks datasets.

  Dataset is from the following paper:
  Rats and humans can optimally accumulate evidence for decision-making.
  Brunton, Botvinick, and Brody. Science, 2013

  This dataset is available at the following link:
  https://github.com/Brody-Lab/brunton_dataset

  Args:
    rat_i: Integer specifying which rat to load.

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be a trial. xs has two elements per
    timestep per trial, one for binned left bup counts and the other for binned
    right bup counts. ys has one element per timestep per trial, which is -1
    (masked out) on all timesteps but the final timestep, and in the final
    timestep is either 0 or 1 to indicate the choice of the rat.
  """

  # A list of URLs for datasets from individual rats
  url_path = 'https://github.com/Brody-Lab/brunton_dataset/raw/main/data/'
  url_filenames = ['chrono_B052_rawdata.mat',
                   'chrono_B053_rawdata.mat',
                   'chrono_B065_rawdata.mat',
                   'chrono_B069_rawdata.mat',
                   'chrono_B074_rawdata.mat',
                   'chrono_B083_rawdata.mat',
                   'chrono_B090_rawdata.mat',
                   'chrono_B093_rawdata.mat',
                   'chrono_B097_rawdata.mat',
                   'chrono_B102_rawdata.mat',
                   'chrono_B103_rawdata.mat',
                   'chrono_B104_rawdata.mat',
                   'chrono_B105_rawdata.mat',
                   'chrono_B106_rawdata.mat',
                   'chrono_B107_rawdata.mat',
                   'chrono_B111_rawdata.mat',
                   'chrono_B112_rawdata.mat',
                   'chrono_B113_rawdata.mat',
                   'chrono_B115_rawdata.mat',
                  ]

  # Download the file for this rat, package a datasetRNN object
  url_filename = url_filenames[rat_i]

  # Download data file
  url = url_path + url_filename
  data_path = 'pclick_ratdata.mat'
  urllib.request.urlretrieve(url, data_path)
  # Load dataset into memory
  ratdata_mat = scipy.io.loadmat(data_path)
  # Clean up after ourselves by removing the downloaded file
  os.remove(data_path)

  n_trials = ratdata_mat['total_trials'][0, 0]

  # Get the choice of the rat. 0 for left, 1 for right
  choices = np.array(ratdata_mat['rawdata']['pokedR'][0])

  # Get the stimulus duration on each trial, in 10ms bins
  stim_dur = ratdata_mat['rawdata']['T'][0]  # In seconds
  n_stim_timesteps = np.ceil(stim_dur * 100)  # In number of 10ms bins

  # Get the click counts in each bin for each trial
  binned_left_bups = np.zeros((n_trials, 100))
  binned_right_bups = np.zeros((n_trials, 100))
  for trial_i in np.arange(n_trials):
    stim_dur = n_stim_timesteps[trial_i]
    stim_start_bin = 101 - stim_dur
    # Click times, in seconds
    left_bups_trial = ratdata_mat['rawdata']['leftbups'][0, trial_i][0]
    right_bups_trial = ratdata_mat['rawdata']['rightbups'][0, trial_i][0]
    # Assemble into binned click counts
    binned_left_bups_trial, _ = np.histogram(left_bups_trial*100,
                                             bins=np.arange(stim_dur))
    binned_right_bups_trial, _ = np.histogram(right_bups_trial*100,
                                              bins=np.arange(stim_dur))

    binned_left_bups[trial_i, stim_start_bin:] = binned_left_bups_trial
    binned_right_bups[trial_i, stim_start_bin:] = binned_right_bups_trial

  # Re-arrange into inputs (xs) and targets (ys) for training RNN
  xs = np.zeros((101, n_trials, 2))
  xs[:-1, :, 0] = np.swapaxes(binned_left_bups, 1, 0)
  xs[:-1, :, 1] = np.swapaxes(binned_right_bups, 1, 0)

  ys = -1*np.ones((101, n_trials, 1))
  ys[-1,:, 0] = choices

  dataset_rat = rnn_utils.DatasetRNN(xs, ys, y_type='categorical')

  return dataset_rat


def get_q_learning_dataset(
    alpha: float = 0.3,
    beta: float = 3.,
    sigma: float = 0.1,
    n_trials: int = 500,
    n_sessions: int = 20000,
    np_rng_seed: float = 0
) -> rnn_utils.DatasetRNN:
  """Generates synthetic dataset from Q-Learning agent, using standard parameters."""
  np.random.seed(np_rng_seed)
  agent = two_armed_bandits.AgentQ(alpha=alpha, beta=beta)
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=sigma)
  dataset = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset


def get_actor_critic_dataset(
    alpha_critic: float = 0.3,
    alpha_actor_learn: float = 1.,
    alpha_actor_forget: float = 0.05,
    sigma: float = 0.1,
    n_trials: int = 500,
    n_sessions: int = 20000,
    np_rng_seed: float = 0,
) -> rnn_utils.DatasetRNN:
  """Generates synthetic dataset from Actor-Critic agent, using standard parameters."""
  np.random.seed(np_rng_seed)
  agent = two_armed_bandits.AgentLeakyActorCritic(
      alpha_critic=alpha_critic,
      alpha_actor_learn=alpha_actor_learn,
      alpha_actor_forget=alpha_actor_forget,
  )
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=sigma)
  dataset = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset


def get_bounded_accumulator_dataset(
    n_trials: int = 200000,
    stim_duration_max: int = 50,
    stim_duration_min: int = 10,
    base_click_rate: float = 10,
    click_rate_diffs: np.ndarray = np.array([-2.5, -1, 0, 1, 2.5]),
    noise_per_click: float = 0.01,
    noise_per_timestep: float = 0.0,
    click_depression: float = 1.0,
    depression_tau: float = 8.0,
    bound: float = 2.9,
    lapse: float = 0.0,
) -> rnn_utils.DatasetRNN:
  """Generates synthetic dataset from Bounded Accumulator."""
  xs, _ = pclicks.generate_clicktrains(
      n_trials=n_trials,
      stim_duration_max=stim_duration_max,
      stim_duration_min=stim_duration_min,
      base_click_rate=base_click_rate,
      click_rate_diffs=click_rate_diffs,
  )

  decisions, _ = pclicks.drift_diffusion_model(
      xs,
      noise_per_click=noise_per_click,
      noise_per_timestep=noise_per_timestep,
      click_depression=click_depression,
      depression_tau=depression_tau,
      bound=bound,
      lapse=lapse,
  )
  ys = -1 * np.ones((stim_duration_max + 1, n_trials, 1))
  ys[-1, :, 0] = decisions
  dataset = rnn_utils.DatasetRNN(xs, ys, y_type='categorical')
  return dataset


def dataset_list_to_multisubject(
    dataset_list: list[rnn_utils.DatasetRNN],
    add_subj_id: bool = True,
) -> rnn_utils.DatasetRNN:
  """Turn a list of single-subject datasets into a multisubject dataset.

  Multisubject dataset has a new first column containing an integer subject ID.
  DisRNN in multisubject mode will convert this first to a one-hot then to a
  subject embedding.

  Args:
    dataset_list: List of single-subject datasets
    add_subj_id: Whether to add a subject ID column to the xs. If True, dataset
      is suitable for multisubject mode. If False, dataset is suitable for
      single-subject mode, treating all data as if from a single subject.

  Returns:
    A single DatasetRNN containing data from all datasets in the list
  """
  xs_dataset, ys_dataset = dataset_list[0].get_all()
  x_names = dataset_list[0].x_names
  y_names = dataset_list[0].y_names
  y_type_str = dataset_list[0].y_type
  n_classes = dataset_list[0].n_classes

  # Runtime check for y_type_str before casting
  allowed_y_types = ('categorical', 'scalar', 'mixed')
  if y_type_str not in allowed_y_types:
    raise ValueError(
        f'Invalid y_type "{y_type_str}" found in dataset_list. '
        f'Expected one of {allowed_y_types}.')
  # Cast for pytype
  y_type = cast(Literal['categorical', 'scalar', 'mixed'], y_type_str)

  # If we're adding a subject ID, we'll add a feature to the xs
  if add_subj_id:
    x_dim = np.shape(xs_dataset)[2] + 1
  else:
    x_dim = np.shape(xs_dataset)[2]
  y_dim = np.shape(ys_dataset)[2]

  xs = np.zeros((0, 0, x_dim))
  ys = np.zeros((0, 0, y_dim))

  max_n_trials = 0

  # For each dataset in the list, add the subject ID column, then add it to the
  # multisubject dataset
  for dataset_i in range(len(dataset_list)):
    # Check datasets are compatible
    assert x_names == dataset_list[dataset_i].x_names, (
        f'x_names do not match across datasets. Expected {x_names}, got'
        f' {dataset_list[dataset_i].x_names}'
    )
    assert y_names == dataset_list[dataset_i].y_names, (
        f'y_names do not match across datasets. Expected {y_names}, got'
        f' {dataset_list[dataset_i].y_names}'
    )
    assert y_type == dataset_list[dataset_i].y_type, (
        f'y_type does not match across datasets. Expected {y_type}, got'
        f' {dataset_list[dataset_i].y_type}'
    )
    assert n_classes == dataset_list[dataset_i].n_classes, (
        f'n_classes does not match across datasets. Expected {n_classes}, got'
        f' {dataset_list[dataset_i].n_classes}'
    )

    xs_dataset, ys_dataset = dataset_list[dataset_i].get_all()
    n_sessions = np.shape(xs_dataset)[1]
    n_trials = np.shape(xs_dataset)[0]
    if add_subj_id:
      subj_ids = dataset_i * np.ones([n_trials, n_sessions, 1])
      xs_dataset = np.concatenate([subj_ids, xs_dataset], axis=2)

    # If this dataset has more trials than all previous datasets, add dummy
    # trials to the previous datasets so all lengths will match
    if n_trials > max_n_trials:
      # Add dummy trials to existing xs and ys
      xs = np.concatenate(
          (xs, -1 * np.ones((n_trials - max_n_trials, np.shape(xs)[1], x_dim))),
          axis=0,
      )
      ys = np.concatenate(
          (
              ys,
              -1
              * np.ones(
                  (n_trials - max_n_trials, np.shape(ys)[1], np.shape(ys)[2])
              ),
          ),
          axis=0,
      )
      max_n_trials = n_trials
    # If this dataset has fewer trials than any previous dataset, add dummy
    # trials to this dataset so all lengths will match
    elif max_n_trials > n_trials:
      xs_dataset = np.concatenate(
          (
              xs_dataset,
              -1
              * np.ones(
                  (max_n_trials - n_trials, np.shape(xs_dataset)[1], x_dim)
              ),
          ),
          axis=0,
      )
      ys_dataset = np.concatenate(
          (
              ys_dataset,
              -1
              * np.ones((
                  max_n_trials - n_trials,
                  np.shape(ys_dataset)[1],
                  np.shape(ys_dataset)[2],
              )),
          ),
          axis=0,
      )

    # Concatenate new xs and ys
    xs = np.concatenate((xs, xs_dataset), axis=1)
    ys = np.concatenate((ys, ys_dataset), axis=1)

  if add_subj_id:
    x_names = ['Subject ID'] + x_names

  dataset = rnn_utils.DatasetRNN(
      xs=xs,
      ys=ys,
      x_names=x_names,
      y_names=y_names,
      y_type=y_type,
      n_classes=n_classes,
  )

  return dataset


def get_q_learning_multisubject_dataset(
    n_trials: int = 200,
    n_sessions: int = 300,
    alphas: Optional[list[float]] = None,
    np_rng_seed: float = 0,
) -> rnn_utils.DatasetRNN:
  """Returns a multisubject dataset for the Q-learning task."""
  if alphas is None:
    alphas = [0.1, 0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
  dataset_list = []
  for alpha in alphas:
    dataset_list.append(
        get_q_learning_dataset(
            n_trials=n_trials,
            n_sessions=n_sessions,
            alpha=alpha,
            np_rng_seed=np_rng_seed,
        )
    )
  return dataset_list_to_multisubject(dataset_list)


def get_rat_bandit_multisubject_dataset(
    n_rats: int = 20,
) -> rnn_utils.DatasetRNN:
  """Returns a multisubject dataset for the rat bandit task."""
  dataset_list = []
  for rat_i in range(n_rats):
    dataset = get_rat_bandit_dataset(rat_i=rat_i)
    dataset_list.append(dataset)
  return dataset_list_to_multisubject(dataset_list)


def get_pclick_multisubject_dataset(
    n_rats: int = 19,
) -> rnn_utils.DatasetRNN:
  """Returns a multisubject dataset for the pClick task."""
  dataset_list = []
  for rat_i in range(n_rats):
    dataset = get_pclicks_dataset(rat_i=rat_i)
    dataset_list.append(dataset)
  return dataset_list_to_multisubject(dataset_list)
