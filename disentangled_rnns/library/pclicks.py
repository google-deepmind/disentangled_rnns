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

"""Code for generating synthetic data from the poisson clicks task.
"""
import numpy as np


def generate_clicktrains(n_trials=10000,
                         base_click_rate=20,
                         click_rate_diffs=(-38, -34, -27, -12, 12, 27, 34, 38),
                         stim_duration_max=100,
                         stim_duration_min=20):
  """Generates random clicktrain stimuli for Poisson clicks task.

  Args:
    n_trials:
    base_click_rate: Average number of clicks on each side
    click_rate_diffs: Difference in click rates on different trial types
    stim_duration_max: Number of 10ms time bins. Each trial's stim duration will
      be sampled uniformly between this and stim_durtion_min
    stim_duration_min: Number of 10ms time bins

  Returns:

  """

  # Leave a blank timestep at the end for the response
  xs = np.zeros((stim_duration_max + 1, n_trials, 2))
  stim_durations = np.zeros(n_trials)

  for trial_i in range(n_trials):
    stim_duration = np.random.randint(stim_duration_min, stim_duration_max)

    diff_i = np.random.randint(0, len(click_rate_diffs))
    click_rate_diff = click_rate_diffs[diff_i]
    rate_left = (base_click_rate - click_rate_diff / 2) / stim_duration_max
    rate_right = (base_click_rate + click_rate_diff / 2) / stim_duration_max

    left_clicks = np.random.binomial(1, rate_left, (stim_duration, 1))
    right_clicks = np.random.binomial(1, rate_right, (stim_duration, 1))

    xs_trial = [left_clicks, right_clicks]
    xs_trial = np.swapaxes(xs_trial, 0, 2)

    xs[-(stim_duration + 1) : -1, trial_i, :] = xs_trial

    stim_durations[trial_i] = stim_duration

  assert np.all(xs[-1, :, :] == 0)  # Check that the last timestep has no clicks

  return xs, stim_durations


def drift_diffusion_model(
    xs,
    noise_per_click=0.5,
    noise_per_timestep=0,
    click_depression=0.3,
    depression_tau=8,
    bound=5,
    lapse=0,
):
  """Runs DDM on clicktrains to produce choices.

  Args:
    xs: Click trains, as output by generate_clicktrains
    noise_per_click: Sensory noise per click
    noise_per_timestep: Accumulator noise per timestep
    click_depression: Amount of sensory depression caused by a single click
    depression_tau: Timeconstant for recovery from click depression
    bound: Sticky decision bound, in units of clicks
    lapse: Lapse rate

  Returns:

  """
  (n_timesteps, n_trials, _) = np.shape(xs)

  click_any = np.logical_or(xs[:, :, 0], xs[:, :, 1]).astype(float)
  depression_variable = np.nan * np.ones((n_timesteps, n_trials))

  depression_variable[0, :] = 1.0
  for timestep_i in range(n_timesteps - 1):
    # First, decay the depression variable towards one
    depression_variable[timestep_i + 1, :] = (
        depression_variable[timestep_i, :]
        + (1 - depression_variable[timestep_i, :]) / depression_tau
    )
    # Next multiply it by phi if there has been a click
    depression_variable[timestep_i + 1, :] += (
        (click_depression - 1)
        * depression_variable[timestep_i, :]
        * click_any[timestep_i, :]
    )

  diffusion_noise = np.random.normal(
      loc=0, scale=noise_per_timestep, size=(n_timesteps, n_trials)
  )

  noisy_clicks_left = np.random.normal(
      loc=xs[:, :, 0],
      scale=noise_per_click * xs[:, :, 0],
      size=(n_timesteps, n_trials),
  )
  noisy_clicks_right = np.random.normal(
      loc=xs[:, :, 1],
      scale=noise_per_click * xs[:, :, 1],
      size=(n_timesteps, n_trials),
  )

  click_evidence = np.multiply(
      noisy_clicks_left - noisy_clicks_right, depression_variable
  )
  decision_variable = np.cumsum(click_evidence, axis=0) + np.cumsum(
      diffusion_noise, axis=0
  )

  crossed_bound = np.logical_or(
      decision_variable > bound, decision_variable < -1 * bound
  )
  first_bound_crossing = np.argmax(crossed_bound, axis=0)

  decisions = np.zeros(n_trials)
  for trial_i in range(n_trials):
    if first_bound_crossing[trial_i] > 0:
      decision_variable[first_bound_crossing[trial_i] :, trial_i] = (
          bound
          * np.sign(decision_variable[first_bound_crossing[trial_i], trial_i])
      )

    decision = (np.sign(decision_variable[-1, trial_i]) + 1) / 2
    decision = (decision + 1) / 2  # Convert from +-1 to 0 or 1

    if np.random.random() < lapse:
      decision = 1 - decision
    decisions[trial_i] = int(decision)

  return decisions, (decision_variable, depression_variable)
