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

"""Disentangled RNN and plotting functions for choice and neural data."""

from collections.abc import Callable
import copy
import dataclasses
from typing import Optional

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class DisRnnWNeuralActivityConfig(disrnn.DisRnnConfig):
  """Specifies configuration for a Disentangled RNN with neural_activity prediction.

  Attributes:
    neural_activity_net_n_units_per_layer: Number of units per layer in the
      neural_activity network
    neural_activity_net_n_layers: Number of layers in the neural_activity net
      targets bottlenecks
  """

  neural_activity_net_n_units_per_layer: int = 8
  neural_activity_net_n_layers: int = 2
  neural_activity_net_latent_penalty: float = 1.0


class HkNeuroDisentangledRNN(disrnn.HkDisentangledRNN):
  """Disentangled RNN with an extra target for neural_activity.

  DisRNN typically have the choice prediction as the output. This class adds an
  extra output which predicts the neural_activity. Currently the neural_activity
  is expected to be a scalar floating point value.
  """

  def __init__(
      self,
      config: DisRnnWNeuralActivityConfig,
  ):
    # We are subclassing the classic DisRNN for this model. Classic disRNN is
    # used as is for the choice prediction.
    # On top of those latents we add new nets to predict the neural_activity.
    if config.obs_size != 2:
      raise ValueError('Only implemented for observation size of 2')

    super().__init__(config)
    self._neural_activity_net_n_units_per_layer = (
        config.neural_activity_net_n_units_per_layer
    )
    self._neural_activity_net_n_layers = config.neural_activity_net_n_layers
    self._neural_activity_net_latent_penalty = (
        config.neural_activity_net_latent_penalty
    )
    self._num_predictions = config.output_size + 1  # +1 for neural_activity
    self._get_haiku_parameters()

  def _build_neural_activity_bottlenecks(self):
    """Initializes parameters for the neural_activity net."""
    input_size = self._latent_size + 2
    self._neural_activity_net_sigmas, self._neural_activity_net_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(input_size,),
            name='neural_activity_net',
        )
    )

  def _get_haiku_parameters(self):
    """Initializes all haiku parameters for this network."""
    super()._get_haiku_parameters()
    self._build_neural_activity_bottlenecks()

  def __call__(self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = observations.shape[0]
    penalty = jnp.zeros(shape=(batch_size,))

    curr_choices = observations[:, 0]
    curr_rewards = observations[:, 1]
    # Convention: First two columns are choice and reward.
    # Right now we don't have anything else in the observation but this allows
    # us to add more easily.
    observations = observations[:, 0:2]
    choice_disrnn_output, new_latents = super().__call__(
        observations, prev_latents
    )

    next_choice_prediction = choice_disrnn_output[:, :-1]
    penalty += choice_disrnn_output[:, -1]

    ####################
    # Neural Activity Net #
    ####################
    # Predict neural_activity for current time step
    # The input to the net are the latents, curr choice and curr reward.
    # The expected output of the net is the current neural_activity value.
    latents_and_curr_obs = jnp.concatenate(
        (new_latents, curr_choices[..., None], curr_rewards[..., None]), axis=1
    )
    # Apply multipliers. We want to do this whether or not we are in noiseless
    # mode, so that the model will produce similar outputs in both modes.
    neural_activity_net_inputs, neural_activity_net_kl_cost = (
        disrnn.information_bottleneck(
            inputs=latents_and_curr_obs,
            sigmas=self._neural_activity_net_sigmas,
            multipliers=self._neural_activity_net_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
    )
    penalty += (
        neural_activity_net_kl_cost * self._neural_activity_net_latent_penalty
    )

    current_neural_activity_prediction, neural_activity_net_l2 = disrnn.ResMLP(
        input_size=neural_activity_net_inputs.shape[1],
        # NOTE: Neural Activity net outputs single scalars for now.
        output_size=1,
        n_units_per_layer=self._neural_activity_net_n_units_per_layer,
        n_layers=self._neural_activity_net_n_layers,
        activation_fn=self._activation,
        name='neural_activity_net',
    )(neural_activity_net_inputs)
    # Neural Activity prediction should be 1D
    current_neural_activity_prediction = jnp.squeeze(
        current_neural_activity_prediction, axis=1
    )

    if not self._noiseless_mode:
      penalty += self._l2_scale * neural_activity_net_l2

    # Note that the final output is the next choice predictions and the current
    # neural_activity prediction. The two outputs don't correspond to the same
    # trial. Neural Activity prediction is for trial n and choice prediction is
    # for trial n+1.
    output_shape = (batch_size, self._num_predictions + 1)

    output = jnp.zeros(output_shape)
    output = output.at[:, :-2].set(next_choice_prediction)
    output = output.at[:, -2].set(current_neural_activity_prediction)
    output = output.at[:, -1].set(penalty)

    return output, new_latents


def plot_bottlenecks(
    params: hk.Params,
    disrnn_w_neural_activity_config: DisRnnWNeuralActivityConfig,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.DisentangledRNN."""
  params = params['hk_neuro_disentangled_rnn']
  latent_dim = params['latent_sigma_params'].shape[0]

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params['latent_sigma_params'])
  )
  neural_activity_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params['neural_activity_net_sigma_params'])
      )
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    neural_activity_sigmas = np.concatenate((
        neural_activity_sigmas[latent_sigma_order],
        neural_activity_sigmas[-2:],
    ))

  latent_names = np.concatenate(
      (np.arange(1, latent_dim + 1), ['prev choice', 'prev reward'])
  )

  # Plot bottlenecks from base network

  fig = plotting.plot_bottlenecks(
      {'hk_disentangled_rnn': params},
      disrnn_w_neural_activity_config,
      sort_latents,
  )

  plt.close()

  # Regenerate the figure with the neural_activity bottlenecks added to the end.
  base_axes = fig.axes
  fig, axes = plt.subplots(1, 4, figsize=(25, 5))

  im = axes[-1].imshow(
      np.swapaxes([1 - neural_activity_sigmas], 0, 1), cmap='Oranges'
  )
  im.set_clim(vmin=0, vmax=1)
  axes[-1].set_title('Neural Activity Bottlenecks')
  axes[-1].set_ylabel('Latent # (Sorted)' if sort_latents else 'Latent #')
  axes[-1].set_xticks([])  # Remove x-axis ticks as it's a 1D representation
  axes[-1].set_yticks(ticks=range(len(latent_names)), labels=latent_names)

  for i, ax in enumerate(base_axes):
    if len(ax.images) < 1:
      continue
    image = ax.images[0].get_array()
    im = axes[i].imshow(image, cmap='Oranges')
    im.set_clim(vmin=0, vmax=1)
    axes[i].set_title(ax.get_title())
    axes[i].set_ylabel(ax.get_ylabel())
    axes[i].set_xticks(
        ticks=ax.get_xticks(),
        labels=ax.get_xticklabels(),
        rotation='vertical',
    )
    axes[i].set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels())
    axes[i].set_ylim(ax.get_ylim())
    axes[i].set_xlim(ax.get_xlim())

  # fig.tight_layout()
  return fig


def plot_neural_activity_rules(
    params: hk.Params,
    disrnn_config: DisRnnWNeuralActivityConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the neural_activity rule of a DisRNN with neural_activity prediction.

  This function visualizes how the predicted neural_activity level changes based
  on
  the model's latents, the choice made, and the reward received. It creates a
  2x2 grid of plots for different combinations of choice and reward. Within each
  subplot, it shows the neural_activity prediction as a function of the most
  influential latent variables.

  Args:
    params: The parameters of the DisRNN.
    disrnn_config: A DisRnnWNeuralActivityConfig object, expected to have
      neural_activity-related attributes.
    axis_lim: The axis limit for the plot.

  Returns:
    A matplotlib Figure object, or None if no inputs are influential.
  """

  config = copy.deepcopy(disrnn_config)
  config.noiseless_mode = True

  params_prefix = 'hk_neuro_disentangled_rnn'
  if params_prefix not in params:
    raise ValueError(f"'{params_prefix}' not found in params. ")
  params_disrnn = params[params_prefix]
  # Get neural_activity-specific attributes from config.
  try:
    neural_activity_net_n_units_per_layer = (
        config.neural_activity_net_n_units_per_layer
    )
    neural_activity_net_n_layers = config.neural_activity_net_n_layers
  except AttributeError:
    print(
        'Neural Activity net attributes not found in config. Is this a'
        ' DisRnnWNeuralActivityConfig?'
    )
    return None

  activation_fn = getattr(jax.nn, config.activation)
  latent_size = config.latent_size

  # Get sigmas and multipliers for the neural_activity network's inputs.
  neural_activity_sigmas = disrnn.reparameterize_sigma(
      params_disrnn['neural_activity_net_sigma_params']
  )
  neural_activity_multipliers = params_disrnn['neural_activity_net_multipliers']

  # Identify influential latents (sigmas < 0.5).
  influential_latents = np.where(neural_activity_sigmas[:latent_size] < 0.1)[0]
  n_influential_latents = len(influential_latents)

  if n_influential_latents == 0:
    print(
        'Neural Activity rule: No latents have a'
        ' neural_activity_net_input_sigma < 0.5. Plotting not possible.'
    )
    return None

  # Define the forward pass for the neural_activity network.
  def forward(xs):
    # Apply the same bottleneck function used during training
    neural_activity_net_inputs, _ = disrnn.information_bottleneck(
        inputs=xs,
        sigmas=neural_activity_sigmas,
        multipliers=neural_activity_multipliers,
        noiseless_mode=True,  # Keep noiseless for plotting
    )
    neural_activity_net = disrnn.ResMLP(
        input_size=neural_activity_net_inputs.shape[1],
        output_size=1,
        n_units_per_layer=neural_activity_net_n_units_per_layer,
        n_layers=neural_activity_net_n_layers,
        activation_fn=activation_fn,
        name='neural_activity_net',
    )
    prediction, _ = neural_activity_net(neural_activity_net_inputs)
    return jnp.squeeze(prediction, axis=-1)

  model = hk.transform(forward)
  apply = jax.jit(model.apply)
  neural_activity_net_params = {
      'neural_activity_net': params[params_prefix + '/neural_activity_net']
  }

  # Plotting logic.
  fig, axes = plt.subplots(
      2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True
  )
  small = 8
  medium = 12
  large = 16
  fig.suptitle('Neural Activity Prediction vs. Latents', fontsize=large)

  n_vals = 50
  latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)

  # Sort influential latents by their sigma values to plot the most important.
  sorted_influential_latents = sorted(
      influential_latents, key=lambda i: neural_activity_sigmas[i]
  )
  n_latents_to_plot = min(n_influential_latents, 2)
  varying_latents_plot_indices = sorted_influential_latents[:n_latents_to_plot]

  if n_influential_latents > 2:
    print(
        'WARNING: More than two latents are influential for neural_activity.'
        ' Plotting only the two with the smallest sigmas.'
    )

  if n_latents_to_plot == 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    latent_idx = varying_latents_plot_indices[0]

    for choice_val, reward_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
      base_inputs = np.zeros(latent_size + 2)
      base_inputs[latent_size] = choice_val
      base_inputs[latent_size + 1] = reward_val

      xs = np.tile(base_inputs, (n_vals, 1))
      xs[:, latent_idx] = latent_vals
      neural_activity_preds = apply(
          neural_activity_net_params, jax.random.PRNGKey(0), xs
      )

      ax.plot(
          latent_vals,
          neural_activity_preds,
          label=f'Choice={choice_val}, Reward={reward_val}',
      )

    ax.set_xlabel(f'Latent {latent_idx + 1}', fontsize=medium)
    ax.set_ylabel('Neural Activity', fontsize=medium)
    ax.set_title('Neural Activity Prediction vs. Latent', fontsize=large)
    ax.legend(fontsize=small)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.suptitle('Neural Activity Prediction vs. Latents', fontsize=large)
    return fig

  # This part is for n_latents_to_plot == 2
  fig, axes = plt.subplots(
      2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True
  )
  fig.suptitle('Neural Activity Prediction vs. Latents', fontsize=large)
  overall_min = None
  overall_max = None
  for i, (choice_val, reward_val) in enumerate(
      [(0, 0), (0, 1), (1, 0), (1, 1)]
  ):
    ax = axes.flatten()[i]
    # Add choice and reward as inputs to the neural_activity net.
    base_inputs = np.zeros(latent_size + 2)
    base_inputs[latent_size] = choice_val
    base_inputs[latent_size + 1] = reward_val

    latent_idx1, latent_idx2 = varying_latents_plot_indices
    xv, yv = np.meshgrid(latent_vals, latent_vals)
    xs = np.tile(base_inputs, (n_vals * n_vals, 1))
    xs[:, latent_idx1] = xv.flatten()
    xs[:, latent_idx2] = yv.flatten()
    neural_activity_preds = apply(
        neural_activity_net_params, jax.random.PRNGKey(0), xs
    )
    neural_activity_preds = neural_activity_preds.reshape((n_vals, n_vals))

    if i == 0:
      # Set the overall min and max to be a bit larger than the max absolute
      # prediction. We use the first plot to set the overall min and max as
      # the plots are all on the same scale.
      max_abs_pred = np.max(np.abs(neural_activity_preds))
      overall_min = -max_abs_pred * 1.1
      overall_max = max_abs_pred * 1.1

    im = ax.imshow(
        neural_activity_preds,
        cmap='coolwarm',
        origin='lower',
        aspect='auto',
        extent=[-axis_lim, axis_lim, -axis_lim, axis_lim],
        vmin=overall_min,
        vmax=overall_max,
    )
    if i == 1 or i == 3:  # Add colorbar to right plots
      cbar = fig.colorbar(im, ax=ax)
      cbar.ax.tick_params(labelsize=small)

    ax.set_xlabel(f'Latent {latent_idx1 + 1}', fontsize=medium)
    if i % 2 == 0:
      ax.set_ylabel(f'Latent {latent_idx2 + 1}', fontsize=medium)

    ax.set_title(f'Choice={choice_val}, Reward={reward_val}', fontsize=medium)
    ax.tick_params(axis='both', labelsize=small)
    ax.grid(True, linestyle='--', alpha=0.6)

  return fig


def plot_choice_rule(
    params: hk.Params,
    disrnn_config: DisRnnWNeuralActivityConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the choice rule of a DisRNN with neural_activity prediction."""

  params = {
      key.replace('hk_neuro_disentangled_rnn', 'hk_disentangled_rnn'): value
      for key, value in params.items()
  }
  plotting.plot_choice_rule(
      params=params, disrnn_config=disrnn_config, axis_lim=axis_lim
  )


def plot_update_rules(
    params: hk.Params,
    disrnn_config: DisRnnWNeuralActivityConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the update rules of a DisRNN with neural_activity prediction."""
  params = {
      key.replace('hk_neuro_disentangled_rnn', 'hk_disentangled_rnn'): value
      for key, value in params.items()
  }
  plotting.plot_update_rules(
      params=params, disrnn_config=disrnn_config, axis_lim=axis_lim
  )


def log_bottlenecks(
    params: hk.Params,
    open_thresh: float = 0.1,
    partially_open_thresh: float = 0.25,
    closed_thresh: float = 0.9,
) -> dict[str, int]:
  """Computes info about bottlenecks."""
  bnecks = disrnn.log_bottlenecks(
      params, open_thresh, partially_open_thresh, closed_thresh
  )

  params_disrnn = params['hk_disentangled_rnn']

  neural_activity_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params_disrnn['neural_activity_net_sigma_params'])
      )
  )

  neural_activity_bottlenecks_open = np.sum(
      neural_activity_sigmas < open_thresh
  )
  neural_activity_bottlenecks_partial = np.sum(
      neural_activity_sigmas < partially_open_thresh
  )
  neural_activity_bottlenecks_closed = np.sum(
      neural_activity_sigmas > closed_thresh
  )

  bottleneck_dict = {
      **bnecks,
      'neural_activity_bottlenecks_open': int(neural_activity_bottlenecks_open),
      'neural_activity_bottlenecks_partial': int(
          neural_activity_bottlenecks_partial
      ),
      'neural_activity_bottlenecks_closed': int(
          neural_activity_bottlenecks_closed
      ),
  }
  return bottleneck_dict


def get_total_sigma(params):
  """Get sum of reparameterized sigmas of a DisRNN."""

  prev_sigma_total = disrnn.get_total_sigma(params)

  params_disrnn = params['hk_disentangled_rnn']

  neural_activity_bottlenecks = disrnn.reparameterize_sigma(
      params_disrnn['neural_activity_net_sigma_params']
  )

  return float(jnp.sum(neural_activity_bottlenecks) + prev_sigma_total)


def get_auxiliary_metrics(
    params: hk.Params,
    make_model_fn: Callable[[], hk.RNNCore],
    dataset_train: rnn_utils.DatasetRNN,
    dataset_eval: rnn_utils.DatasetRNN,
) -> dict[str, np.ndarray]:
  """Compute auxiliary metrics for DisRNN with Neural Activity."""
  xs, ys = dataset_train.get_all()
  network_outputs, _ = rnn_utils.eval_network(
      make_model_fn,
      params,
      xs,
  )
  y_hats = network_outputs[:, :, :-1]
  choice_normlik_train = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=1.0
  )
  neural_activity_mse_train = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=0.0
  )

  xs, ys = dataset_eval.get_all()
  network_outputs, _ = rnn_utils.eval_network(
      make_model_fn,
      params,
      xs,
  )
  y_hats = network_outputs[:, :, :-1]
  choice_normlik_eval = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=1.0
  )
  neural_activity_mse_eval = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=0.0
  )

  bottleneck_metrics = log_bottlenecks(params)
  total_sigma_val = get_total_sigma(params)
  total_sigma_val = np.array(total_sigma_val)

  neural_activity_specific_metrics = {
      'choice_normlik_train': np.array(choice_normlik_train),
      'choice_normlik_eval': np.array(choice_normlik_eval),
      'neural_activity_mse_train': np.array(neural_activity_mse_train),
      'neural_activity_mse_eval': np.array(neural_activity_mse_eval),
  }
  return {
      **neural_activity_specific_metrics,
      **bottleneck_metrics,
      'total_sigma': total_sigma_val,
  }
