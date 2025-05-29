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
     bottleneck
  """

  max_n_subjects: int = 0
  subject_embedding_size: int = 0
  subject_embedding_penalty_scale: float = 0.0

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
    self._subject_embedding_penalty_scale = (
        config.subject_embedding_penalty_scale
    )

    # IMPORTANT: super().__init__, will call helper method _get_haiku_parameters
    # which we have overridden below. This way of doing things is necessary for
    # Haiku to create the parameters correctly.
    super().__init__(config)
    self._get_haiku_parameters()

  def _get_haiku_parameters(self):
    """Initializes all haiku parameters for multisubject DisRNN."""
    # Subject embedding bottlenecks do not use multipliers, since the
    # embedding is already a linear projection from the one-hots.
    self._subject_embedding_sigmas, _ = disrnn.get_initial_bottleneck_params(
        shape=(self._subject_embedding_size,),
        name='subject_embedding',
    )
    # Build the update and choice bottlenecks, which now include the subject
    # embedding in their inputs.
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

  def _build_multisubject_update_bottlenecks(self):
    """Initializes update net bottleneck parameters, including subject embedding.
    """
    # Input: subject_embedding + observations + latents
    # Needs _obs_size from base class config to be set first.
    update_net_input_size = (
        self._subject_embedding_size + self._obs_size + self._latent_size
    )
    self._update_net_sigmas, self._update_net_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(update_net_input_size, self._latent_size),
            name='update_net',
        )
    )

  def _build_multisubject_choice_bottlenecks(self):
    """Initializes choice MLP bottleneck parameters, including subject embedding.
    """
    # Input: subject_embedding + latents (output from update_latents)
    choice_net_input_size = self._subject_embedding_size + self._latent_size
    self._choice_net_sigmas, self._choice_net_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(choice_net_input_size,),
            name='choice_net',
        )
    )

  def __call__(self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = observations.shape[0]
    penalty = jnp.zeros(shape=(batch_size,))

    subject_id = observations[:, 0]
    # TODO(kevinjmiller): Figure out how to add an assert here
    subject_one_hot = jax.nn.one_hot(subject_id, self._max_n_subjects)
    subject_embeddings = hk.Linear(
        self._subject_embedding_size, name='subject_embedding_weights'
    )(
        subject_one_hot,
    )  # Shape: (batch_size, subject_embedding_size)

    # Put subject embeddings through an information bottleneck. If we are in
    # noiseless mode, we can skip this and use the embeddings directly.
    if not self._noiseless_mode:
      subject_embeddings, subject_kl_cost = disrnn.information_bottleneck(
          mus=subject_embeddings,
          sigmas=self._subject_embedding_sigmas
      )
      # Include subject embedding penalty in the overall penalty
      penalty += self._subject_embedding_penalty_scale * subject_kl_cost

    # Use subject embeddings as additional observations for the update nets
    update_net_inputs = jnp.concatenate(
        (subject_embeddings, observations[:, 1:]), axis=1
    )
    update_net_outputs, update_net_penalty = super().update_latents(
        update_net_inputs, prev_latents
    )
    penalty += update_net_penalty

    # Concatenate subject embeddings to the inputs of the choice net
    choice_net_inputs = jnp.concatenate(
        (subject_embeddings, update_net_outputs), axis=1
    )
    predicted_targets, choice_net_penalty = super().predict_targets(
        choice_net_inputs
    )
    penalty += choice_net_penalty

    # Output has shape (batch_size, output_size + 1).
    output_shape = (batch_size, self._output_size + 1)
    output = jnp.zeros(output_shape)
    output = output.at[:, :-1].set(predicted_targets)
    output = output.at[:, -1].set(penalty)

    return output, update_net_outputs


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
  # We'd like to use the functions in disrnn, but these expect params to be in
  # params['hk_disentangled_rnn'].
  module_params = params['multisubject_dis_rnn']
  base_bnecks = disrnn.log_bottlenecks(
      {'hk_disentangled_rnn': module_params},
      open_thresh,
      closed_thresh,
  )
  subject_embedding_sigmas = np.array(
      disrnn.reparameterize_sigma(
          module_params['subject_embedding_sigma_params']
      )
  )
  subject_bottlenecks_open = np.sum(subject_embedding_sigmas < open_thresh)
  subject_bottlenecks_closed = np.sum(subject_embedding_sigmas > closed_thresh)
  bottleneck_dict = {
      **base_bnecks,
      'subject_bottlenecks_open': int(subject_bottlenecks_open),
      'subject_bottlenecks_closed': int(subject_bottlenecks_closed),
  }

  # Get total sigma (the sum of all bottleneck sigmas)
  base_total_sigma = disrnn.get_total_sigma(
      {'hk_disentangled_rnn': module_params}
  )
  # subject_embedding_sigmas is already computed above
  total_sigma_val = float(base_total_sigma + jnp.sum(subject_embedding_sigmas))

  return {'total_sigma': total_sigma_val, **bottleneck_dict}
