# Copyright 2024 The disentangled_rnns Authors.
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

"""Load rat datasets."""

import json
import os
from typing import List
import urllib.request

from disentangled_rnns.library import pclicks
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import numpy as np
import scipy.io


def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_bandit_datasets() -> List[rnn_utils.DatasetRNN]:
  """Downloads and packages rat two-armed bandit datasets.

  Dataset is from the following paper:
  From predictive models to cognitive models: Separable behavioral processes
  underlying reward learning in the rat. Miller, Botvinick, and Brody,
  bioRxiv, 2018

  Dataset is available from Figshare at the following link:
  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

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

  # Each iteration of the loop processes data from one rat, converting the dict
  # into inputs (xs) and targets (ys) for training a neural network, packaging
  # these into a DatasetRNN object, and appending this to dataset_list
  n_rats = len(dataset)
  dataset_list = []
  for rat_i in range(n_rats):
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

    # Pack into a DatasetRNN object and append to the list
    dataset_rat = rnn_utils.DatasetRNN(ys=ys, xs=xs)
    dataset_list.append(dataset_rat)

  return dataset_list


def get_pclicks_datasets() -> List[rnn_utils.DatasetRNN]:
  """Packages up rat poisson clicks datasets.

  Dataset is from the following paper:
  Rats and humans can optimally accumulate evidence for decision-making.
  Brunton, Botvinick, and Brody. Science, 2013

  This dataset is available at the following link:
  https://github.com/Brody-Lab/brunton_dataset

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

  # For each URL, download the file, package a dataset, and append it to the
  # dataset_list
  dataset_list = []
  for url_filename in url_filenames:

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

    dataset_list.append(rnn_utils.DatasetRNN(xs, ys))

  return dataset_list


def get_q_learning_dataset():
  """Generates synthteic dataset from Q-Learning agent, using standard parameters."""
  np.random.seed(0)
  agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
  n_trials = 500
  n_sessions = 20000
  dataset = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset


def get_actor_critic_dataset():
  """Generates synthetic dataset from Actor-Critic agent, using standard parameters."""
  np.random.seed(0)
  agent = two_armed_bandits.AgentLeakyActorCritic(
      alpha_critic=0.3, alpha_actor_learn=1, alpha_actor_forget=0.05
  )
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
  n_trials = 500
  n_sessions = 20000
  dataset = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset


def get_bounded_accumulator_dataset():
  """Generates synthetic dataset from Bounded Accumulator agent, using standard parameters."""
  n_trials = 200000
  stim_duration_max = 50
  stim_duration_min = 10
  base_click_rate = 10
  click_rate_diffs = np.array([-5., -2., 0., 2., 5.,]) / 2.

  noise_per_click = 0.01
  noise_per_timestep = 0.0
  click_depression = 1.0
  depression_tau = 8.0
  bound = 2.9
  lapse = 0.0

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
  dataset = rnn_utils.DatasetRNN(xs, ys)
  return dataset
