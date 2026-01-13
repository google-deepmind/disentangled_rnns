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

"""DisRNN extension. Converts subject ID into a disentangled embedding."""

import dataclasses
from typing import Any

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import multisubject_disrnn
from disentangled_rnns.library import neuro_disrnn
from disentangled_rnns.library import plotting
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class MultisubjectNeuroDisRnnConfig(
    multisubject_disrnn.MultisubjectDisRnnConfig,
    neuro_disrnn.DisRnnWNeuralActivityConfig,
):
  """Combines MultisubjectDisRnnConfig and DisRnnWNeuralActivityConfig."""

  neural_activity_net_subj_penalty: float = 0.0

  # post_init will be called in the ordrer of inheritance.
  # This means that MultisubjectDisRnnConfig will be called first,
  # then DisRnnWNeuralActivityConfig.


class MultisubjectNeuroDisRnn(multisubject_disrnn.MultisubjectDisRnn):
  """Disentangled RNN with multisubject mode."""

  def __init__(self, config: MultisubjectNeuroDisRnnConfig):

    # IMPORTANT: super().__init__, will call helper method _get_haiku_parameters
    # which we have overridden below. This way of doing things is necessary for
    # Haiku to create the parameters correctly.
    config.enable_aux_outputs = True

    super().__init__(config)
    self._neural_activity_net_n_units_per_layer = (
        config.neural_activity_net_n_units_per_layer
    )
    self._neural_activity_net_n_layers = config.neural_activity_net_n_layers
    self._neural_activity_net_latent_penalty = (
        config.neural_activity_net_latent_penalty
    )
    self._num_predictions = config.output_size + 1  # +1 for neural_activity
    self._neural_activity_net_subj_penalty = (
        config.neural_activity_net_subj_penalty
    )
    self._get_haiku_parameters()

  def _get_haiku_parameters(self):
    """Initializes all haiku parameters for multisubject DisRNN."""
    super()._get_haiku_parameters()
    self._build_neural_activity_bottlenecks()

  def _build_neural_activity_bottlenecks(self):
    """Initializes parameters for the neural_activity net."""
    input_size = self._latent_size + 2
    self._neural_activity_net_sigmas, self._neural_activity_net_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(input_size,),
            name='neural_activity_net',
        )
    )

    (
        self._neural_activity_net_subj_sigmas,
        self._neural_activity_net_subj_multipliers,
    ) = disrnn.get_initial_bottleneck_params(
        shape=(self._subject_embedding_size,),
        name='neural_activity_net_subj',
    )

  def __call__(self, inputs: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = inputs.shape[0]

    # Let the parent class do the heavy lifting of predicting the next choice
    # and updating the latents.
    choice_disrnn_output, new_latents, aux_output = super().__call__(
        inputs, prev_latents
    )

    # Unpack the output from the parent class
    next_choice_prediction = choice_disrnn_output[:, :-1]
    penalty = choice_disrnn_output[:, -1]

    # Now, predict the neural activity
    # The input to the neural activity net are the latents, current choice, and
    # current reward.
    observations = inputs[:, 1:]
    curr_choices = observations[:, 0]
    curr_rewards = observations[:, 1]
    latents_and_curr_obs = jnp.concatenate(
        (new_latents, curr_choices[..., None], curr_rewards[..., None]), axis=1
    )

    subject_embeddings = aux_output['subject_embeddings']

    neural_activity_net_subj_inputs, kl_cost = disrnn.information_bottleneck(
        inputs=subject_embeddings,
        sigmas=self._neural_activity_net_subj_sigmas,
        multipliers=self._neural_activity_net_subj_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._neural_activity_net_subj_penalty * kl_cost

    # Pass the inputs through an information bottleneck
    neural_activity_net_inputs, kl_cost = disrnn.information_bottleneck(
        inputs=latents_and_curr_obs,
        sigmas=self._neural_activity_net_sigmas,
        multipliers=self._neural_activity_net_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._neural_activity_net_latent_penalty * kl_cost

    neural_activity_net_inputs = jnp.concatenate(
        (neural_activity_net_subj_inputs, neural_activity_net_inputs), axis=1
    )

    # Use an MLP to predict the neural activity
    current_neural_activity_prediction, neural_activity_net_l2 = disrnn.ResMLP(
        input_size=neural_activity_net_inputs.shape[1],
        output_size=1,  # Predict a single value for neural activity
        n_units_per_layer=self._neural_activity_net_n_units_per_layer,
        n_layers=self._neural_activity_net_n_layers,
        activation_fn=self._activation,
        name='neural_activity_net',
    )(neural_activity_net_inputs)
    current_neural_activity_prediction = jnp.squeeze(
        current_neural_activity_prediction, axis=1
    )

    if not self._noiseless_mode:
      penalty += self._l2_scale * neural_activity_net_l2

    # Format the output to include the choice prediction, neural activity
    # prediction, and the penalty.
    output_shape = (batch_size, self._num_predictions + 1)
    output = jnp.zeros(output_shape)
    output = output.at[:, :-2].set(next_choice_prediction)
    output = output.at[:, -2].set(current_neural_activity_prediction)
    output = output.at[:, -1].set(penalty)

    return output, new_latents


def get_auxiliary_metrics(
    params: hk.Params,
    open_thresh: float = 0.1,
    closed_thresh: float = 0.9,
) -> dict[str, Any]:
  """Computes auxiliary metrics for Multisubject Neuro DisRNN."""
  assert 'multisubject_neuro_dis_rnn' in params, (
      'This is the auxiliary metrics function for MultisubjectNeuroDisRnn, but'
      ' params is not a MultisubjectNeuroDisRnn params. Instead it contains'
      f' {params.keys()}'
  )
  module_params = params['multisubject_neuro_dis_rnn']

  def _count_states(
      sigmas_array: np.ndarray, open_thresh: float, closed_thresh: float
  ):
    is_open = np.sum(sigmas_array < open_thresh)
    is_closed = np.sum(sigmas_array > closed_thresh)
    return int(is_open), int(is_closed)

  # Latent bottlenecks
  latent_s = disrnn.reparameterize_sigma(module_params['latent_sigma_params'])
  lat_open, lat_closed = _count_states(
      np.array(latent_s), open_thresh, closed_thresh
  )

  # Update bottlenecks (sum over subj, obs, latent inputs)
  upd_subj_s = disrnn.reparameterize_sigma(
      module_params['update_net_subj_sigma_params']
  )
  upd_obs_s = disrnn.reparameterize_sigma(
      module_params['update_net_obs_sigma_params']
  )
  upd_lat_s = disrnn.reparameterize_sigma(
      module_params['update_net_latent_sigma_params']
  )

  us_o, us_c = _count_states(np.array(upd_subj_s), open_thresh, closed_thresh)
  uo_o, uo_c = _count_states(np.array(upd_obs_s), open_thresh, closed_thresh)
  ul_o, ul_c = _count_states(np.array(upd_lat_s), open_thresh, closed_thresh)

  update_open = us_o + uo_o + ul_o
  update_closed = us_c + uo_c + ul_c

  # Choice bottlenecks (sum over subj, latent inputs)
  choice_subj_s = disrnn.reparameterize_sigma(
      module_params['choice_net_subj_sigma_params']
  )
  choice_lat_s = disrnn.reparameterize_sigma(
      module_params['choice_net_latent_sigma_params']
  )

  cs_o, cs_c = _count_states(
      np.array(choice_subj_s), open_thresh, closed_thresh
  )
  cl_o, cl_c = _count_states(np.array(choice_lat_s), open_thresh, closed_thresh)

  choice_open = cs_o + cl_o
  choice_closed = cs_c + cl_c

  # Neural activity bottlenecks
  na_s = disrnn.reparameterize_sigma(
      module_params['neural_activity_net_sigma_params']
  )
  na_o, na_c = _count_states(np.array(na_s), open_thresh, closed_thresh)

  bottleneck_dict = {
      'latent_bottlenecks_open': lat_open,
      'latent_bottlenecks_closed': lat_closed,
      'choice_bottlenecks_open': choice_open,
      'choice_bottlenecks_closed': choice_closed,
      'update_bottlenecks_open': update_open,
      'update_bottlenecks_closed': update_closed,
      'neural_activity_bottlenecks_open': na_o,
      'neural_activity_bottlenecks_closed': na_c,
  }

  # Get total sigma (the sum of all bottleneck sigmas)
  all_sigmas_sum = (
      jnp.sum(latent_s)
      + jnp.sum(upd_subj_s)
      + jnp.sum(upd_obs_s)
      + jnp.sum(upd_lat_s)
      + jnp.sum(choice_subj_s)
      + jnp.sum(choice_lat_s)
      + jnp.sum(na_s)
  )
  total_sigma_val = float(all_sigmas_sum)

  return {'total_sigma': total_sigma_val, **bottleneck_dict}


def plot_bottlenecks(
    params: hk.Params,
    multisubject_neuro_disrnn_config: MultisubjectNeuroDisRnnConfig,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.MultisubjectNeuroDisentangledRNN."""
  params_multisubject_dis_rnn = {
      key.replace(
          'multisubject_neuro_dis_rnn',
          'multisubject_dis_rnn',
      ): value
      for key, value in params.items()
  }

  fig_multisubject_dis_rnn = plotting.plot_bottlenecks(
      params_multisubject_dis_rnn,
      multisubject_neuro_disrnn_config,
      sort_latents,
  )

  # generate the bottlenecks for the neural activity net
  params = params['multisubject_neuro_dis_rnn']
  latent_dim = params['latent_sigma_params'].shape[0]

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params['latent_sigma_params'])
  )
  neural_activity_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params['neural_activity_net_sigma_params'])
      )
  )
  neural_activity_net_subj_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params['neural_activity_net_subj_sigma_params'])
      )
  )

  neural_activity_sigmas = np.concatenate(
      (neural_activity_net_subj_sigmas, neural_activity_sigmas)
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)

    neural_activity_sigma_order = np.concatenate(
        (
            np.arange(
                0, multisubject_neuro_disrnn_config.subject_embedding_size
            ),
            multisubject_neuro_disrnn_config.subject_embedding_size
            + latent_sigma_order,
        ),
        axis=0,
    )
    neural_activity_sigmas = neural_activity_sigmas[neural_activity_sigma_order]

    neural_activity_sigmas = np.concatenate((
        neural_activity_sigmas,
        neural_activity_sigmas[-2:],
    ))
    print(
        'Lenght of neural activity sigmas (if sorted): ',
        len(neural_activity_sigmas),
    )

  subject_embedding_names = [
      f'SubjEmb {i+1}'
      for i in range(multisubject_neuro_disrnn_config.subject_embedding_size)
  ]
  latent_names = np.arange(1, latent_dim + 1)
  non_latent_input_names = ['prev choice', 'prev reward']

  bottleneck_names = (
      subject_embedding_names + list(latent_names) + non_latent_input_names
  )

  plt.close()

  # combine the two figures
  fig_combined = plotting.append_bottleneck(
      fig=fig_multisubject_dis_rnn,
      bottleneck_values=neural_activity_sigmas,
      bottleneck_names=list(bottleneck_names),
      title='Neural Activity Bottlenecks',
      sort_latents=sort_latents,
  )
  return fig_combined
