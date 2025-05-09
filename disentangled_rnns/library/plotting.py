# Copyright 2024 DeepMind Technologies Limited.
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

"""Plotting functions for inspecting Disentangled RNNs."""

import copy
from typing import Optional

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Fontsizes and formatting for plots.
small = 15
medium = 18
large = 20
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'


def plot_bottlenecks(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.DisentangledRNN."""

  params_disrnn = params['hk_disentangled_rnn']

  latent_dim = params_disrnn['latent_sigma_params'].shape[0]
  obs_dim = params_disrnn['update_net_sigma_params'].shape[0] - latent_dim

  update_input_names = disrnn_config.x_names

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params_disrnn['latent_sigma_params'])
  )
  update_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params_disrnn['update_net_sigma_params'])
      )
  )
  choice_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params_disrnn['choice_net_sigma_params'])
      )
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    latent_sigmas = latent_sigmas[latent_sigma_order]
    choice_sigmas = choice_sigmas[latent_sigma_order]

    update_sigma_order = np.concatenate(
        (np.arange(0, obs_dim, 1), obs_dim + latent_sigma_order), axis=0
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, latent_dim + 1)
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  # Plot Latent Bottlenecks on axes[0]
  im1 = axes[0].imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  im1.set_clim(vmin=0, vmax=1)
  axes[0].set_yticks(
      ticks=range(latent_dim), labels=latent_names, fontsize=small
  )
  axes[0].set_xticks(ticks=[])
  axes[0].set_ylabel('Latent #', fontsize=medium)
  axes[0].set_title('Latent Bottlenecks', fontsize=large)

  # Plot Choice Bottlenecks on axes[1]
  im2 = axes[1].imshow(np.swapaxes([1 - choice_sigmas], 0, 1), cmap='Oranges')
  im2.set_clim(vmin=0, vmax=1)
  axes[1].set_yticks(
      ticks=range(latent_dim), labels=latent_names, fontsize=small
  )
  axes[1].set_xticks(ticks=[])
  axes[1].set_ylabel('Latent #', fontsize=medium)
  axes[1].set_title('Choice Network Bottlenecks', fontsize=large)

  # Plot Update Bottlenecks on axes[2]
  im3 = axes[2].imshow(1 - update_sigmas, cmap='Oranges')
  im3.set_clim(vmin=0, vmax=1)
  cbar = fig.colorbar(im3, ax=axes[2])
  cbar.ax.tick_params(labelsize=small)
  axes[2].set_yticks(
      ticks=range(latent_dim), labels=latent_names, fontsize=small
  )
  xlabels = np.concatenate((np.array(update_input_names), latent_names))
  axes[2].set_xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
      fontsize=small,
  )
  axes[2].set_ylabel('Latent #', fontsize=medium)
  axes[2].set_title('Update Network Bottlenecks', fontsize=large)
  fig.tight_layout()  # Adjust layout to prevent overlap
  return fig


def plot_update_rules(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    subj_ind: Optional[int] = None,
    axis_lim: float = 2.1,
) -> plt.Figure:
  """Generates visualizations of the update rules of a HkDisentangledRNN."""

  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True  # Turn off noise for plotting

  make_network = lambda: disrnn.HkDisentangledRNN(disrnn_config)
  obs_names = disrnn_config.x_names
  param_prefix = 'hk_disentangled_rnn'

  def step(xs, state):
    core = make_network()
    output, new_state = core(jnp.expand_dims(jnp.array(xs), axis=0), state)
    return output, new_state

  _, step_hk = hk.transform(step)
  key = jax.random.PRNGKey(0)
  step_hk = jax.jit(step_hk)

  initial_state = np.array(rnn_utils.get_initial_state(make_network))
  reference_state = np.zeros(initial_state.shape)

  def plot_update_1d(params, unit_i, observations, titles):
    state_bins = np.linspace(-axis_lim, axis_lim, 20)
    colormap = mpl.colormaps['viridis'].resampled(3)
    colors = colormap.colors

    fig, axes = plt.subplots(
        1, len(observations), figsize=(len(observations) * 4, 5.5), sharey=True
    )
    # Ensure axes is always an array for consistent indexing
    if len(observations) == 1:
      axes = [axes]
    axes[0].set_ylabel('Δ Activity')

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      if subj_ind is not None:
        observation = [subj_ind] + observation
      ax = axes[observation_i]
      delta_states = np.zeros(shape=(len(state_bins), 1))
      for s_i in np.arange(len(state_bins)):
        state = reference_state
        state[0, unit_i] = state_bins[s_i]
        _, next_state = step_hk(
            params, key, observation, state
        )
        next_state = np.array(next_state)
        delta_states[s_i] = next_state[0, unit_i] - state_bins[s_i]

      ax.plot((-axis_lim, axis_lim), (0, 0), color='black')
      ax.plot(state_bins, delta_states, color=colors[1])
      ax.set_title(titles[observation_i], fontsize=large)
      ax.set_xlim(-axis_lim, axis_lim)
      ax.set_xlabel(
          'Latent ' + str(unit_i + 1) + ' Activity', fontsize=medium
      )
      ax.set_aspect('equal')
      ax.tick_params(axis='both', labelsize=small)

    return fig

  def plot_update_2d(params, unit_i, unit_input, observations, titles):

    state_bins = np.linspace(-axis_lim, axis_lim, 50)
    state_bins_input = np.linspace(-axis_lim/2, axis_lim/2, 5)
    colormap = mpl.colormaps['viridis'].resampled(len(state_bins_input))
    colors = colormap.colors

    fig, axes = plt.subplots(
        1,
        len(observations),
        figsize=(len(observations) * 2 + 10, 5.5),
        sharey=True,
    )
    # Ensure axes is always an array for consistent indexing
    if len(observations) == 1:
      axes = [axes]
    axes[0].set_ylabel('Δ Activity', fontsize=medium)

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      if subj_ind is not None:
        observation = [subj_ind] + observation
      legend_elements = []
      ax = axes[observation_i]
      for si_i in np.arange(len(state_bins_input)):
        delta_states = np.zeros(shape=(len(state_bins), 1))
        for s_i in np.arange(len(state_bins)):
          state = reference_state
          state[0, unit_i] = state_bins[s_i]
          state[0, unit_input] = state_bins_input[si_i]

          _, next_state = step_hk(params, key, observation, state)
          next_state = np.array(next_state)
          delta_states[s_i] = next_state[0, unit_i] - state_bins[s_i]

        lines = ax.plot(state_bins, delta_states, color=colors[si_i])
        legend_elements.append(lines[0])

        if observation_i == 0:
          legend_labels = [f'{num:.1f}' for num in state_bins_input]  # pylint: disable=bad-whitespace
          ax.legend(legend_elements, legend_labels, fontsize=small)

      ax.plot((-axis_lim, axis_lim), (0, 0), color='black')
      ax.set_title(titles[observation_i], fontsize=large)
      ax.set_xlim(-axis_lim, axis_lim)
      ax.set_xlabel(
          'Latent ' + str(unit_i + 1) + ' Activity', fontsize=medium
      )
      ax.tick_params(axis='both', labelsize=small)

    return fig

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(
          params[param_prefix]['latent_sigma_params']
      )
  )
  update_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params[param_prefix]['update_net_sigma_params'])
      )
  )

  obs_size = 2

  latent_order = np.argsort(latent_sigmas)
  figs = []

  # Loop over latents. Plot update rules
  for latent_i in latent_order:
    # If this latent's bottleneck is open
    if latent_sigmas[latent_i] < 0.5:

      # Which of its input bottlenecks are open?
      update_net_inputs = np.argwhere(update_sigmas[latent_i] < 0.5)
      # TODO(kevinjmiller): Generalize to allow different observation length
      obs1_sensitive = np.any(update_net_inputs == 0)
      obs2_sensitive = np.any(update_net_inputs == 1)
      # Choose which observations to use based on input bottlenecks
      if obs1_sensitive and obs2_sensitive:
        observations = ([0, 0], [0, 1], [1, 0], [1, 1])
        titles = (
            obs_names[0] + ': 0\n' + obs_names[1] + ': 0',
            obs_names[0] + ': 0\n' + obs_names[1] + ': 1',
            obs_names[0] + ': 1\n' + obs_names[1] + ': 0',
            obs_names[0] + ': 1\n' + obs_names[1] + ': 1',
        )
      elif obs1_sensitive:
        observations = ([0, 0], [1, 0])
        titles = (obs_names[0] + ': 0', obs_names[0] + ': 1')
      elif obs2_sensitive:
        observations = ([0, 0], [0, 1])
        titles = (obs_names[1] + ': 0', obs_names[1] + ': 1')
      else:
        observations = ([0, 0],)
        titles = ('All Trials',)

      # Choose whether to condition on other latent values
      update_net_input_latents = (
          update_net_inputs[obs_size:, 0]
          - (obs_size)
      )
      # Doesn't count if it depends on itself (this'll be shown no matter what)
      latent_sensitive = np.delete(
          update_net_input_latents, update_net_input_latents == latent_i
      )
      if not latent_sensitive.size:  # Depends on no other latents
        fig = plot_update_1d(params, latent_i, observations, titles)
      else:  # It depends on latents other than itself.
        fig = plot_update_2d(
            params,
            latent_i,
            latent_sensitive[0],
            observations,
            titles,
        )
      if len(latent_sensitive) > 1:
        print(
            'WARNING: This update rule depends on more than one '
            'other latent. Plotting just one of them'
        )
      figs.append(fig)
      fig.tight_layout()

  return figs


def plot_choice_rule(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the choice rule of a DisRNN.

  Args:
    params: The parameters of the DisRNN
    disrnn_config: A DisRnnConfig object
    axis_lim: The axis limit for the plot.

  Returns:
    A matplotlib Figure object, or None if choice depends on no latents.
  """

  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True  # Turn off noise for plotting
  activation_fn = getattr(jax.nn, disrnn_config.activation)

  params_prefix = 'hk_disentangled_rnn'

  n_vals = 100

  def forward(xs):
    choice_net_output = disrnn.ResMLP(
        input_size=disrnn_config.latent_size,
        output_size=disrnn_config.output_size,
        n_units_per_layer=disrnn_config.choice_net_n_units_per_layer,
        n_layers=disrnn_config.choice_net_n_layers,
        activation_fn=activation_fn,
        name='choice_net',
    )(xs)
    return choice_net_output

  model = hk.transform(forward)
  apply = jax.jit(model.apply)

  choice_net_params = {
      'choice_net': params[params_prefix + '/~predict_targets/choice_net']
  }
  choice_net_sigmas = disrnn.reparameterize_sigma(
      params[params_prefix]['choice_net_sigma_params']
  )
  n_inputs = np.sum(choice_net_sigmas < 0.5)
  choice_net_input_order = np.argsort(choice_net_sigmas)

  if n_inputs == 0:
    print('Choice does not depend on any latents')
    return None
  elif n_inputs == 1:
    # Choice Rule 1D: A curve
    policy_latent_ind = choice_net_input_order[0]
    policy_latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)
    xs = np.zeros((
        n_vals,
        disrnn_config.latent_size,
    ))
    xs[:, policy_latent_ind] = policy_latent_vals
    y_hats, _ = apply(choice_net_params, jax.random.PRNGKey(0), xs)
    choice_logits = y_hats[:, 0] - y_hats[:, 1]

    fig, ax = plt.subplots()
    ax.plot(policy_latent_vals, choice_logits, 'g')
    ax.set_title('Choice Rule', fontsize=large)
    ax.set_xlabel(f'Latent {policy_latent_ind + 1}', fontsize=medium)
    ax.set_ylabel('Choice Logit', fontsize=medium)
    ax.tick_params(axis='both', labelsize=small)

  else:
    # Choice Rule 2D: A colormap
    if n_inputs > 2:
      print(
          'WARNING: More than two latents contribute to choice. Plotting only',
          ' the first two.'
      )

    policy_latent_inds = choice_net_input_order[:2]

    latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)

    xv, yv = np.meshgrid(latent_vals, latent_vals)
    latent0_vals = np.reshape(xv, (xv.size,))
    latent1_vals = np.reshape(yv, (yv.size,))

    xs = np.zeros(
        shape=(
            n_vals**2,
            disrnn_config.latent_size,
        )
    )
    xs[:, policy_latent_inds[0]] = latent0_vals
    xs[:, policy_latent_inds[1]] = latent1_vals

    y_hats, _ = apply(choice_net_params, jax.random.PRNGKey(0), xs)
    # TODO(kevinjmiller): This assumes two-alternative logits. Generalize to
    # allow more alternatives and/or scalar outputs
    choice_logits = y_hats[:, 1] - y_hats[:, 0]

    cmax = np.max(np.abs(choice_logits))

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        latent0_vals, latent1_vals, c=choice_logits, s=100, cmap='bwr'
    )
    scatter.set_clim(-cmax, cmax)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=small)
    cbar.set_label('Choice Logit', fontsize=medium)
    ax.set_title('Choice Rule', fontsize=large)
    ax.set_xlabel(f'Latent {policy_latent_inds[0]+1}', fontsize=medium)
    ax.set_ylabel(f'Latent {policy_latent_inds[1]+1}', fontsize=medium)
    ax.tick_params(axis='both', labelsize=small)

  return fig
