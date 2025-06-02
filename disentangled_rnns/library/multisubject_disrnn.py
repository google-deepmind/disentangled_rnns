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
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class MultisubjectDisRnnConfig(disrnn.DisRnnConfig):
  """Extends DisRnnConfig with multisubject-specific parameters.

  Attributes:
   max_n_subjects: If multisubject_mode is True, this is the maximum number of
     subjects in the dataset. Otherwise must be zero
   subject_embedding_size: If multisubject_mode is True, this is the number of
     dimensions in the subject embedding. Otherwise must be zero
   subject_embedding_penalty_scale: Multiplier for KL cost on subject embedding
     bottleneck. This refers to the global subject embedding bottleneck.
  """

  max_n_subjects: int = 0
  subject_embedding_size: int = 0
  subj_penalty: float = 0.0
  update_net_subj_penalty: float = 0.0
  choice_net_subj_penalty: float = 0.0

  def __post_init__(self):
    # If the first observation is a subject ID, remove it from the list of
    # observation names before calling the base class's __post_init__
    if self.x_names is not None and self.x_names[0] == 'Subject ID':
      self.x_names = self.x_names[1:]
    super().__post_init__()
    # We expect that the first element of the input will be an integer subject
    # ID. Add this to the list of observation names to make it explicit.
    self.x_names = ['Subject ID'] + self.x_names


class MultisubjectDisRnn(disrnn.HkDisentangledRNN):
  """Disentangled RNN with multisubject mode."""

  def __init__(self, config: MultisubjectDisRnnConfig):

    # Set up subject embedding parameters
    self._max_n_subjects = config.max_n_subjects
    self._subject_embedding_size = config.subject_embedding_size
    self._subj_penalty = config.subj_penalty
    self._update_net_subj_penalty = config.update_net_subj_penalty
    self._choice_net_subj_penalty = config.choice_net_subj_penalty

    # IMPORTANT: super().__init__, will call helper method _get_haiku_parameters
    # which we have overridden below. This way of doing things is necessary for
    # Haiku to create the parameters correctly.
    super().__init__(config)
    self._get_haiku_parameters()

  def _get_haiku_parameters(self):
    """Initializes all haiku parameters for multisubject DisRNN."""
    # Build the update and choice bottlenecks, which now include the subject
    # embedding in their inputs.
    self._build_subj_emb_global_bottleneck()
    self._build_multisubject_update_bottlenecks()
    self._build_multisubject_choice_bottlenecks()
    # Build the latent bottlenecks, which are unchanged from the base class.
    super()._build_latent_bottlenecks()
    # Build the latent initializations, which are unchanged from the base class.
    self._latent_inits = hk.get_parameter(
        'latent_inits',
        (self._latent_size,),
        init=hk.initializers.RandomUniform(minval=-1, maxval=1),
    )

  def _build_subj_emb_global_bottleneck(self):
    """Initializes subject embedding bottleneck parameters."""
    # Input: subject_embedding
    # This bottleneck is applied directly to the subject_embedding.
    self._subj_emb_global_sigma, _ = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._subject_embedding_size,),
            name='subj_emb_global',
        )
    )

  def _build_multisubject_update_bottlenecks(self):
    """Initializes update net bottleneck parameters, including subject embedding.
    """
    # Input: subject_embedding + observations + latents
    # Needs _obs_size from base class config to be set first.
    self._update_net_subj_sigmas, self._update_net_subj_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._subject_embedding_size, self._latent_size),
            name='update_net_subj',
        )
    )
    self._update_net_obs_sigmas, self._update_net_obs_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._obs_size, self._latent_size),
            name='update_net_obs',
        )
    )
    self._update_net_latent_sigmas, self._update_net_latent_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._latent_size, self._latent_size),
            name='update_net_latent',
        )
    )

  def _build_multisubject_choice_bottlenecks(self):
    """Initializes choice MLP bottleneck parameters, including subject embedding.
    """
    # Input: subject_embedding + latents (output from update_latents)
    self._choice_net_latent_sigmas, self._choice_net_latent_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._latent_size,),
            name='choice_net_latent',
        )
    )
    self._choice_net_subj_sigmas, self._choice_net_subj_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(self._subject_embedding_size,),
            name='choice_net_subj',
        )
    )

  def __call__(self, inputs: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = inputs.shape[0]
    penalty = jnp.zeros(shape=(batch_size,))

    subject_id = inputs[:, 0]
    observations = inputs[:, 1:]

    # Compute the subject embeddings
    # TODO(kevinjmiller): Figure out how to add an assert here
    subject_one_hot = jax.nn.one_hot(subject_id, self._max_n_subjects)
    subject_embeddings = hk.Linear(
        self._subject_embedding_size, name='subject_embedding_weights'
    )(
        subject_one_hot,
    )  # Shape: (batch_size, subject_embedding_size)

    # Pass subject embeddings through a global bottleneck
    subject_embeddings, kl_cost = disrnn.information_bottleneck(
        inputs=subject_embeddings,
        sigmas=self._subj_emb_global_sigma,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._subj_penalty * kl_cost

    # Set up update net inputs. These will include subject embeddings,
    # observations, and previous latents.
    # Subject embeddings for update net input
    # (batch_size, subj_emb_size, latent_size)
    subj_emb_for_update_net = jnp.tile(
        jnp.expand_dims(subject_embeddings, 2), (1, 1, self._latent_size)
    )
    subj_emb_for_update_net, kl_cost = disrnn.information_bottleneck(
        inputs=subj_emb_for_update_net,
        sigmas=self._update_net_subj_sigmas,
        multipliers=self._update_net_subj_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._update_net_subj_penalty * kl_cost
    # Observations for update net input (batch_size, obs_size, latent_size)
    obs_for_update_net = jnp.tile(
        jnp.expand_dims(observations, 2), (1, 1, self._latent_size)
    )
    obs_for_update_net, kl_cost = disrnn.information_bottleneck(
        inputs=obs_for_update_net,
        sigmas=self._update_net_obs_sigmas,
        multipliers=self._update_net_obs_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._update_net_obs_penalty * kl_cost
    # Latents for update net input
    # (batch_size, latent_size, latent_size)
    prev_latents_for_update_net = jnp.tile(
        jnp.expand_dims(prev_latents, 2), (1, 1, self._latent_size)
    )
    prev_latents_for_update_net, kl_cost = disrnn.information_bottleneck(
        inputs=prev_latents_for_update_net,
        sigmas=self._update_net_latent_sigmas,
        multipliers=self._update_net_latent_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._update_net_latent_penalty * kl_cost
    # Concatenate all inputs for the update networks
    # Shape: (batch_size, subj_emb_size + obs_size + latent_size, latent_size)
    update_net_inputs = jnp.concatenate(
        (
            subj_emb_for_update_net,
            obs_for_update_net,
            prev_latents_for_update_net,
        ),
        axis=1,
    )
    # Use update networks to compute new latents
    new_latents, update_net_penalty = super().update_latents(
        update_net_inputs, prev_latents
    )
    penalty += update_net_penalty

    # Predict targets
    # Subject embeddings for choice net input: (batch_size, subj_emb_size,)
    subj_emb_for_choice_net, subj_emb_kl_cost = disrnn.information_bottleneck(
        inputs=subject_embeddings,
        sigmas=self._choice_net_subj_sigmas,
        multipliers=self._choice_net_subj_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._choice_net_subj_penalty * subj_emb_kl_cost
    # Latents for choice net input: (batch_size, latent_size)
    latents_for_choice_net, latent_kl_cost = disrnn.information_bottleneck(
        inputs=new_latents,
        sigmas=self._choice_net_latent_sigmas,
        multipliers=self._choice_net_latent_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._choice_net_latent_penalty * latent_kl_cost
    # Choice net input: concatenate subject embeddings and latents
    choice_net_inputs = jnp.concatenate(
        (subj_emb_for_choice_net, latents_for_choice_net), axis=1
    )
    # Use choice network to predict targets
    predicted_targets, choice_net_penalty = super().predict_targets(
        choice_net_inputs
    )
    penalty += choice_net_penalty

    # Format the output. Shape is (batch_size, output_size + 1).
    output_shape = (batch_size, self._output_size + 1)
    output = jnp.zeros(output_shape)
    output = output.at[:, :-1].set(predicted_targets)
    output = output.at[:, -1].set(penalty)

    return output, new_latents


def get_auxiliary_metrics(
    params: hk.Params,
    open_thresh: float = 0.1,
    closed_thresh: float = 0.9,
) -> dict[str, Any]:
  """Computes auxiliary metrics for Multisubject DisRNN."""
  assert 'multisubject_dis_rnn' in params, (
      'This is the auxiliary metrics function for Multisubject DisRNN, but'
      ' params is not a MultisubjectDisRnn params. Instead it contains'
      f' {params.keys()}'
  )
  module_params = params['multisubject_dis_rnn']

  def _count_states(sigmas_array: np.ndarray,
                    open_thresh: float, closed_thresh: float):
    is_open = np.sum(sigmas_array < open_thresh)
    is_closed = np.sum(sigmas_array > closed_thresh)
    return int(is_open), int(is_closed)

  # Latent bottlenecks
  latent_s = disrnn.reparameterize_sigma(module_params['latent_sigma_params'])
  lat_open, lat_closed = _count_states(
      np.array(latent_s), open_thresh, closed_thresh)

  # Update bottlenecks (sum over subj, obs, latent inputs)
  upd_subj_s = disrnn.reparameterize_sigma(
      module_params['update_net_subj_sigma_params'])
  upd_obs_s = disrnn.reparameterize_sigma(
      module_params['update_net_obs_sigma_params'])
  upd_lat_s = disrnn.reparameterize_sigma(
      module_params['update_net_latent_sigma_params'])

  us_o, us_c = _count_states(
      np.array(upd_subj_s), open_thresh, closed_thresh)
  uo_o, uo_c = _count_states(
      np.array(upd_obs_s), open_thresh, closed_thresh)
  ul_o, ul_c = _count_states(
      np.array(upd_lat_s), open_thresh, closed_thresh)

  update_open = us_o + uo_o + ul_o
  update_closed = us_c + uo_c + ul_c

  # Choice bottlenecks (sum over subj, latent inputs)
  choice_subj_s = disrnn.reparameterize_sigma(
      module_params['choice_net_subj_sigma_params'])
  choice_lat_s = disrnn.reparameterize_sigma(
      module_params['choice_net_latent_sigma_params'])

  cs_o, cs_c = _count_states(
      np.array(choice_subj_s), open_thresh, closed_thresh)
  cl_o, cl_c = _count_states(
      np.array(choice_lat_s), open_thresh, closed_thresh)

  choice_open = cs_o + cl_o
  choice_closed = cs_c + cl_c

  bottleneck_dict = {
      'latent_bottlenecks_open': lat_open,
      'latent_bottlenecks_closed': lat_closed,
      'choice_bottlenecks_open': choice_open,
      'choice_bottlenecks_closed': choice_closed,
      'update_bottlenecks_open': update_open,
      'update_bottlenecks_closed': update_closed,
  }

  # Get total sigma (the sum of all bottleneck sigmas)
  all_sigmas_sum = (
      jnp.sum(latent_s) + jnp.sum(upd_subj_s) + jnp.sum(upd_obs_s) +
      jnp.sum(upd_lat_s) + jnp.sum(choice_subj_s) + jnp.sum(choice_lat_s)
      )
  total_sigma_val = float(all_sigmas_sum)

  return {'total_sigma': total_sigma_val, **bottleneck_dict}
