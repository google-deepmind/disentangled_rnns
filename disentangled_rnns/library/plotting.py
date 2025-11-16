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

"""Plotting functions for inspecting Disentangled RNNs."""

import copy
from typing import Optional

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import multisubject_disrnn
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

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    params_disrnn = params['multisubject_dis_rnn']
    subject_embedding_size = disrnn_config.subject_embedding_size
    update_input_names = [
        f'SubjEmb {i+1}' for i in range(subject_embedding_size)
    ] + disrnn_config.x_names[1:]
    # For update_sigmas: concatenate transposed reparameterized sigmas
    # Order of inputs to update nets: subject_embedding, observations, latents
    update_subj_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_subj_sigma_params']
        )
    )
    update_obs_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_obs_sigma_params']
        )
    )
    update_latent_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_latent_sigma_params']
        )
    )
    update_sigmas = np.concatenate(
        (update_subj_sigmas_t, update_obs_sigmas_t, update_latent_sigmas_t),
        axis=1,
    )

    # For choice_sigmas: concatenate reparameterized sigmas
    # Order of inputs to choice net: subject_embedding, latents
    choice_subj_sigmas = disrnn.reparameterize_sigma(
        params_disrnn['choice_net_subj_sigma_params']
    )
    choice_latent_sigmas = disrnn.reparameterize_sigma(
        params_disrnn['choice_net_latent_sigma_params']
    )
    choice_sigmas = np.concatenate((choice_subj_sigmas, choice_latent_sigmas))
  elif isinstance(disrnn_config, disrnn.DisRnnConfig):
    params_disrnn = params['hk_disentangled_rnn']
    subject_embedding_size = 0
    update_input_names = disrnn_config.x_names
    # For update_sigmas: concatenate transposed reparameterized sigmas
    # Order of inputs to update nets: observations, latents
    update_obs_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_obs_sigma_params']
        )
    )
    update_latent_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_latent_sigma_params']
        )
    )
    update_sigmas = np.concatenate(
        (update_obs_sigmas_t, update_latent_sigmas_t), axis=1)
    choice_sigmas = np.array(
        disrnn.reparameterize_sigma(
            np.transpose(params_disrnn['choice_net_sigma_params'])
        )
    )
  else:
    raise ValueError(
        'plot_bottlenecks only supports DisRnnConfig and'
        ' MultisubjectDisRnnConfig.'
    )

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params_disrnn['latent_sigma_params'])
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    latent_sigmas = latent_sigmas[latent_sigma_order]

    # Sort choice sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist.
    choice_sigma_order = np.concatenate(
        (
            np.arange(0, subject_embedding_size),
            subject_embedding_size + latent_sigma_order,
        ),
        axis=0,
    )
    choice_sigmas = choice_sigmas[choice_sigma_order]

    # Sort update sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist, then observations, then latents.
    non_latent_input_size = subject_embedding_size + disrnn_config.obs_size
    update_sigma_order = np.concatenate(
        (
            np.arange(0, non_latent_input_size, 1),
            non_latent_input_size + latent_sigma_order,
        ),
        axis=0,
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, disrnn_config.latent_size + 1)
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  # Plot Latent Bottlenecks on axes[0]
  im1 = axes[0].imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  im1.set_clim(vmin=0, vmax=1)
  axes[0].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  axes[0].set_xticks(ticks=[])
  axes[0].set_ylabel('Latent #', fontsize=medium)
  axes[0].set_title('Latent Bottlenecks', fontsize=large)

  # Plot Choice Bottlenecks on axes[1]
  # These bottlenecks apply to the inputs of the choice network:
  # [subject embeddings, latents]
  choice_input_dim = subject_embedding_size + disrnn_config.latent_size
  choice_input_names = np.concatenate((
      [f'SubjEmb {i+1}' for i in range(subject_embedding_size)],
      [f'Latent {i}' for i in latent_names]
  ))
  im2 = axes[1].imshow(np.swapaxes([1 - choice_sigmas], 0, 1), cmap='Oranges')
  im2.set_clim(vmin=0, vmax=1)
  axes[1].set_yticks(
      ticks=range(choice_input_dim), labels=choice_input_names, fontsize=small
  )
  axes[1].set_xticks(ticks=[])
  axes[1].set_ylabel('Choice Network Input', fontsize=medium)
  axes[1].set_title('Choice Network Bottlenecks', fontsize=large)

  # Plot Update Bottlenecks on axes[2]
  im3 = axes[2].imshow(1 - update_sigmas, cmap='Oranges')
  im3.set_clim(vmin=0, vmax=1)
  cbar = fig.colorbar(im3, ax=axes[2])
  # Y-axis corresponds to the target latent (sorted if sort_latents=True)
  cbar.ax.tick_params(labelsize=small)
  axes[2].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  # X-axis corresponds to the inputs to the update network:
  # [subject embeddings, observations, latents]
  xlabels = update_input_names + [f'Latent {i}' for i in latent_names]
  axes[2].set_xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
      fontsize=small,
  )
  axes[2].set_ylabel('Latent #', fontsize=medium)
  axes[2].set_xlabel('Update Network Inputs', fontsize=medium)
  axes[2].set_title('Update Network Bottlenecks', fontsize=large)
  fig.tight_layout()  # Adjust layout to prevent overlap
  return fig


def plot_update_rules(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    subj_ind: Optional[int] = None,
    axis_lim: float = 2.1,
) -> list[plt.Figure]:
  """Generates visualizations of the update rules of a HkDisentangledRNN."""

  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True  # Turn off noise for plotting

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    if subj_ind is None:
      print('In multisubject mode, but subj_ind not provided. Defaulting to 0')
      subj_ind = 0
  elif subj_ind is not None:
    print('subj_ind provided, but not in multisubject mode. Ignoring it')
    subj_ind = None

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    make_network = lambda: multisubject_disrnn.MultisubjectDisRnn(disrnn_config)
    obs_names = disrnn_config.x_names[1:]  # First x_name is "Subject ID"
    param_prefix = 'multisubject_dis_rnn'
    subj_embedding_size = disrnn_config.subject_embedding_size
    update_subj_s_t = np.transpose(disrnn.reparameterize_sigma(
        params[param_prefix]['update_net_subj_sigma_params']))
    update_obs_s_t = np.transpose(disrnn.reparameterize_sigma(
        params[param_prefix]['update_net_obs_sigma_params']))
    update_latent_s_t = np.transpose(disrnn.reparameterize_sigma(
        params[param_prefix]['update_net_latent_sigma_params']))
    update_sigmas = np.concatenate(
        (update_subj_s_t, update_obs_s_t, update_latent_s_t), axis=1
    )
  elif isinstance(disrnn_config, disrnn.DisRnnConfig):
    make_network = lambda: disrnn.HkDisentangledRNN(disrnn_config)
    obs_names = disrnn_config.x_names
    param_prefix = 'hk_disentangled_rnn'
    subj_embedding_size = 0
    update_obs_s_t = np.transpose(
        disrnn.reparameterize_sigma(
            params[param_prefix]['update_net_obs_sigma_params']
        )
    )
    update_latent_s_t = np.transpose(
        disrnn.reparameterize_sigma(
            params[param_prefix]['update_net_latent_sigma_params']
        ))
    update_sigmas = np.concatenate((update_obs_s_t, update_latent_s_t), axis=1)

  else:
    raise ValueError(
        f'Unsupported config type: {type(disrnn_config)} for plot_update_rules.'
    )

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

  # TODO(kevinjmiller): Generalize to allow different observation length
  if disrnn_config.obs_size != 2:
    raise NotImplementedError(
        'Plot update rules currently assumes that there are exactly two'
        f' observations. Instead founc observarions {obs_names}'
    )

  latent_order = np.argsort(latent_sigmas)
  figs = []

  # Loop over latents. Plot update rules
  for latent_i in latent_order:
    # If this latent's bottleneck is open
    if latent_sigmas[latent_i] < 0.5:

      # Which of its input bottlenecks are open?
      update_net_inputs = np.argwhere(update_sigmas[latent_i] < 0.5)
      obs1_sensitive = np.any(update_net_inputs == subj_embedding_size)
      obs2_sensitive = np.any(update_net_inputs == subj_embedding_size + 1)

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

      # Choose which other latents to condition on, based on input bottlenecks
      start_idx_of_latents = subj_embedding_size + disrnn_config.obs_size
      is_latent_input_mask = update_net_inputs >= start_idx_of_latents
      influential_latent_input_indices = update_net_inputs[is_latent_input_mask]
      update_net_input_latents = (
          influential_latent_input_indices - start_idx_of_latents
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
    subj_embedding: Optional[np.ndarray] = None,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the choice rule of a DisRNN.

  Args:
    params: The parameters of the DisRNN
    disrnn_config: A DisRnnConfig object
    subj_embedding: The subject embedding to use. If None, use a zero vector
      (loosely: the average subject)
    axis_lim: The axis limit for the plot.

  Returns:
    A matplotlib Figure object, or None if choice depends on no latents.
  """

  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True  # Turn off noise for plotting
  activation_fn = getattr(jax.nn, disrnn_config.activation)

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    subj_embedding_size = disrnn_config.subject_embedding_size
    params_prefix = 'multisubject_dis_rnn'
    choice_subj_s = disrnn.reparameterize_sigma(
        params[params_prefix]['choice_net_subj_sigma_params']
    )
    choice_latent_s = disrnn.reparameterize_sigma(
        params[params_prefix]['choice_net_latent_sigma_params']
    )
    choice_net_sigmas = np.concatenate((choice_subj_s, choice_latent_s))
  elif isinstance(disrnn_config, disrnn.DisRnnConfig):
    subj_embedding_size = 0
    params_prefix = 'hk_disentangled_rnn'
    choice_net_sigmas = disrnn.reparameterize_sigma(
        params[params_prefix]['choice_net_sigma_params']
    )
  else:
    raise ValueError(
        'DisRnnConfig is neither MultisubjectDisRnnConfig nor DisRnnConfig,'
        f' but got {type(disrnn_config).__name__}'
    )

  if subj_embedding is None:
    subj_embedding = np.zeros(shape=(subj_embedding_size,))

  n_vals = 100

  def forward(xs):
    choice_net_output = disrnn.ResMLP(
        input_size=disrnn_config.latent_size + subj_embedding_size,
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

  # Determine which latents to vary based on their choice_net_sigma_params.
  # choice_net_sigmas has shape (subj_embedding_size + latent_size,).
  latent_to_choice_net_sigmas = choice_net_sigmas[subj_embedding_size:]

  sorted_latent_indices = np.argsort(
      latent_to_choice_net_sigmas
  )

  influential_latents_indices_in_latent_space = [
      latent_idx  # This is an index in the latent space (0 to latent_size-1)
      for latent_idx in sorted_latent_indices
      if latent_to_choice_net_sigmas[latent_idx] < 0.5
  ]
  n_latents_to_plot = min(len(influential_latents_indices_in_latent_space), 2)

  if n_latents_to_plot == 0:
    print(
        'Choice rule: No latents have a choice_net_input_sigma < 0.5.'
        ' Plotting not possible.'
    )
    return None

  # Select the actual latents to vary (indices within the latent space)
  varying_latents_plot_indices = influential_latents_indices_in_latent_space[
      :n_latents_to_plot
  ]

  if n_latents_to_plot == 1:
    # Choice Rule 1D: A curve
    policy_latent_idx_in_latent_space = varying_latents_plot_indices[0]
    policy_latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)
    xs = np.zeros((
        n_vals,
        subj_embedding_size + disrnn_config.latent_size,
    ))
    xs[:, :subj_embedding_size] = subj_embedding
    # Vary the selected latent; other latents remain 0
    xs[
        :, subj_embedding_size + policy_latent_idx_in_latent_space
    ] = policy_latent_vals
    choice_net_output = apply(choice_net_params, jax.random.PRNGKey(0), xs)
    y_hats = choice_net_output[0]
    choice_logits = y_hats[:, 1] - y_hats[:, 0]

    fig, ax = plt.subplots()
    ax.plot(policy_latent_vals, choice_logits, 'g')
    ax.set_title('Choice Rule', fontsize=large)
    ax.set_xlabel(
        f'Latent {policy_latent_idx_in_latent_space + 1}', fontsize=medium
    )
    ax.set_ylabel('Choice Logit', fontsize=medium)
    ax.tick_params(axis='both', labelsize=small)

  else:
    # Choice Rule 2D: A colormap
    if len(influential_latents_indices_in_latent_space) > 2:
      print(
          'WARNING: More than two latents have choice_net_input_sigma < 0.5.'
          ' Plotting only the two with the smallest choice_net_input_sigmas.'
      )

    policy_latent_idx1_in_latent_space = varying_latents_plot_indices[0]
    policy_latent_idx2_in_latent_space = varying_latents_plot_indices[1]

    latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)

    xv, yv = np.meshgrid(latent_vals, latent_vals)
    latent0_vals = xv.flatten()
    latent1_vals = yv.flatten()

    xs = np.zeros((
        n_vals * n_vals,
        subj_embedding_size + disrnn_config.latent_size,
    ))
    xs[:, :subj_embedding_size] = subj_embedding
    # Vary the selected latents; other latents remain 0
    xs[
        :, subj_embedding_size + policy_latent_idx1_in_latent_space
    ] = latent0_vals
    xs[
        :, subj_embedding_size + policy_latent_idx2_in_latent_space
    ] = latent1_vals

    y_hats = apply(choice_net_params, jax.random.PRNGKey(0), xs)
    # TODO(kevinjmiller): This assumes two-alternative logits. Generalize to
    # K-alternative choice. For now, take difference between first two.
    choice_logits_2d = y_hats[0][:, 1] - y_hats[0][:, 0]
    choice_logits_2d = choice_logits_2d.reshape((n_vals, n_vals))

    fig, ax = plt.subplots()
    scatter = ax.imshow(
        choice_logits_2d,
        cmap='coolwarm',
        origin='lower',
        extent=[-axis_lim, axis_lim, -axis_lim, axis_lim],
        aspect='auto',
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=small)
    cbar.set_label('Choice Logit', fontsize=medium)
    ax.set_title('Choice Rule', fontsize=large)
    ax.set_xlabel(
        f'Latent {policy_latent_idx1_in_latent_space + 1}', fontsize=medium
    )
    ax.set_ylabel(
        f'Latent {policy_latent_idx2_in_latent_space + 1}', fontsize=medium
    )
    ax.tick_params(axis='both', labelsize=small)

  return fig
