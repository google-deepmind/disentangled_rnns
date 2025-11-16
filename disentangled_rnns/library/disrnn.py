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

"""Disentangled RNN and plotting functions."""
import dataclasses
from typing import Optional, Callable, Any, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def information_bottleneck(
    inputs: jnp.ndarray,
    sigmas: jnp.ndarray,
    multipliers: Optional[jnp.ndarray] = None,
    noiseless_mode: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  r"""Output from an information bottleneck given a vector of means and std devs.

  Bottleneck outputs are sampled independently from Gaussian distributions with
  the given means and variances. Bottleneck costs are computed as the KL
  divergence between this sampling distribution and the unit Gaussian.

  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

  Args:
    inputs: The inputs to the bottleneck. Shape is (batch_size, bottleneck_dims)
    sigmas: The standard deviations of the sampling distribution (diagonal of
      the sqrt of the covariance matrix). Shape is (bottleneck_dims).
    multipliers: The multipliers to apply to the inputs. Shape is
      (batch_size, bottleneck_dims)
    noiseless_mode: If True, no noise is added and no penalty is computed.

  Returns:
    bottleneck_output: The noisy output of the bottleneck. Shape is the same as
      mus.
    bottleneck_penalty: The KL cost of the bottleneck sample. Shape is
      (batch_size,).
  """
  # Apply multipliers if any are provided
  # Do this regardless of whether we are in noiseless mode, so the average
  # output will be the same
  if multipliers is None:
    mus = inputs
  else:
    mus = inputs * multipliers

  if noiseless_mode:
    # In noiseless mode, do add noise and do not include penalty
    bottleneck_output = mus
    bottleneck_penalty = jnp.zeros(shape=(mus.shape[0],))
  else:
    # In noisey mode, add noise based on sigmas and compute KL penalty
    # Shape is (batch_size, bottleneck_dims)
    bottleneck_output = mus + sigmas * jax.random.normal(
        hk.next_rng_key(), jnp.shape(mus)
    )

    # KL divergence KL( N(mu, var) || N(0, 1))
    variances = jnp.square(sigmas)
    # Shape is (batch_size, bottleneck_dims)
    elementwise_kl = jnp.square(mus) + variances - 1.0 - jnp.log(variances)
    # Shape is (batch_size,)
    bottleneck_penalty = 0.5 * jnp.sum(
        elementwise_kl, axis=tuple(range(1, elementwise_kl.ndim))
    )

  return bottleneck_output, bottleneck_penalty


def reparameterize_sigma(
    hk_param: jnp.ndarray, min_sigma: float = 1e-5
) -> jnp.ndarray:
  """Reparamaterizes bottleneck sigma for easy fitting.

  Args:
    hk_param: The haiku parameter corresponding to a bottleneck sigma. Range
      from -inf to +inf
    min_sigma: The minimum value of the standard deviation.
  Returns:
    sigma: The bottleneck standard deviation. Range from min_sigma to inf.
  """
  return jnp.abs(hk_param) + min_sigma


@dataclasses.dataclass
class DisRnnConfig:
  """Specifies an architecture and configuration for a Disentangled RNN.

  Attributes:
    obs_size: Number of dimensions in the observation vector
    output_size: Number of dimensions the disRNN will output
      (logits or predicted targets)
    latent_size: Number of recurrent variables
    update_net_n_units_per_layer: Number of units in each layer of the update
      networks
    update_net_n_layers: Number of layers in the update networks
    choice_net_n_units_per_layer: Number of units in each layer of the choice
      network
    choice_net_n_layers: Number of layers in the choice network
    noiseless_mode: Allows turning off the bottlenecks e.g. for evaluation
    latent_penalty: Multiplier for KL cost on the latent bottlenecks
    choice_net_latent_penalty: Multiplier for bottleneck cost on latent inputs
      to the choice network
    update_net_obs_penalty: Multiplier for bottleneck cost on observation
      inputs to the update network
    update_net_latent_penalty: Multiplier for latent inputs to the update
      networks
    l2_scale: Multiplier for L2 penalty on hidden layer weights in both update
      and choice networks
    activation: String defining an activation function. Must be in jax.nn.
    max_latent_value: Cap on the possible absolute value of a latent. Used to
      prevent runaway latents resulting in NaNs
    x_names: Names of the observation vector elements. Must have length obs_size
    y_names: Names of the target vector elements. Must have length target_size
  """

  obs_size: int = 2
  output_size: int = 2
  latent_size: int = 10

  update_net_n_units_per_layer: int = 10
  update_net_n_layers: int = 2
  choice_net_n_units_per_layer: int = 2
  choice_net_n_layers: int = 2
  activation: str = 'leaky_relu'

  noiseless_mode: bool = False

  latent_penalty: float = 1.0
  update_net_obs_penalty: float = 0.0
  update_net_latent_penalty: float = 0.0
  choice_net_latent_penalty: float = 0.0

  l2_scale: float = 0.01

  max_latent_value: float = 2.

  x_names: Optional[list[str]] = None
  y_names: Optional[list[str]] = None

  def __post_init__(self):
    """Checks that the configuration is valid."""

    expected_len_xnames = self.obs_size
    if self.x_names is None:
      self.x_names = [f'Observation {i}' for i in range(expected_len_xnames)]
    if len(self.x_names) != expected_len_xnames:
      raise ValueError(
          f'Based on obs_size {self.obs_size}, expected x_names to have '
          f'length {self.obs_size} but got {self.x_names}'
      )

    # Check activation is in jax.nn
    try:
      getattr(jax.nn, self.activation)
    except AttributeError as e:
      raise ValueError(
          f'Activation {self.activation} not found in jax.nn. Provided value '
          f'was {self.activation}'
      ) from e


class ResMLP(hk.Module):
  """MLP modified to apply serial updates to a residual stream.

  Attributes:
    input_size: Dimension of the input vector
    output_size: Dimensions of the output vector
    n_layers: Number of layers
    n_units_per_layer: Dimension of the stream and of each layer
    activation_fn: Activation function
    w_initializer: Initializer for the weights
    b_initializer: Initializer for the biases
    name: Optional name, which affects the names of the haiku parameters
  """

  def __init__(self,
               input_size: int,
               output_size: int,
               n_layers: int = 5,
               n_units_per_layer: int = 5,
               activation_fn: Callable[[Any], Any] = jax.nn.relu,
               name=None):
    super().__init__(name=name)

    self.n_layers = n_layers
    self.n_units_per_layer = n_units_per_layer
    self.activation_fn = activation_fn

    # Input layer will be a linear projection from input size to stream size
    # To keep activation magnitudes similar, we initialise weights in the range
    # 1/sqrt(input_size)
    scale = 1 / jnp.sqrt(jnp.float32(input_size))
    self._input_weights = hk.get_parameter(
        'input_weights',
        (input_size, self.n_units_per_layer),
        init=hk.initializers.RandomNormal(stddev=scale),
    )
    self._input_biases = hk.get_parameter(
        'input_biases',
        (self.n_units_per_layer,),
        init=hk.initializers.Constant(0.0),
    )

    # Output layer will be a linear projection from stream size to output size,
    # To keep activation magnitudes similar, we scale initial weights by
    # 1 / sqrt(n_units_per_layer)
    scale = 1 / jnp.sqrt(jnp.float32(self.n_units_per_layer))
    self._output_weights = hk.get_parameter(
        'output_weights',
        (n_units_per_layer, output_size),
        init=hk.initializers.RandomNormal(stddev=scale),
    )
    self._output_biases = hk.get_parameter(
        'output_biases', (output_size,), init=hk.initializers.Constant(0.0)
    )

    # Hidden layers will each be a single fully connected layer.
    # Each layer will increase the variance of the stream, so we scale initial
    # weights both by n_units_per_layer and n_layers.
    hidden_w_init_scale = 1 / jnp.sqrt(
        jnp.float32(n_units_per_layer * n_layers)
    )
    hidden_w_init = hk.initializers.RandomNormal(stddev=hidden_w_init_scale)
    self._hidden_layer_weights = []
    self._hidden_layer_biases = []
    for hidden_layer_i in range(self.n_layers):
      self._hidden_layer_weights.append(
          hk.get_parameter(
              f'layer_{hidden_layer_i}_weights',
              (self.n_units_per_layer, self.n_units_per_layer),
              init=hidden_w_init,
          )
      )
      self._hidden_layer_biases.append(
          hk.get_parameter(
              f'layer_{hidden_layer_i}_biases',
              (self.n_units_per_layer,),
              init=hk.initializers.Constant(0.0),
          )
      )

    # Compute sum of squares of all hidden layer weights. This will be passed on
    # and can be used to compute an L2 (ridge) penalty.
    self.l2 = (
        jnp.sum(jnp.square(jnp.array(self._hidden_layer_weights)))
    )

  def __call__(self, inputs):

    # Linear projection of inputs to the size of the residual stream
    stream = jnp.dot(inputs, self._input_weights) + self._input_biases

    # Each iteration adds a layer.
    # Each hidden layer additively modifies the residual stream.
    for hidden_layer_i in range(self.n_layers):
      # (batch_size, stream_size)
      layer_activations = (
          jnp.dot(stream, self._hidden_layer_weights[hidden_layer_i])
          + self._hidden_layer_biases[hidden_layer_i]
      )
      layer_output = self.activation_fn(layer_activations)
      stream += layer_output
    # Linear projection to the appropriate output size
    output = jnp.dot(stream, self._output_weights) + self._output_biases

    return output, self.l2


def get_initial_bottleneck_params(
    shape: Sequence[int], name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Defines a bottleneck with a sigma and a multiplier."""
  # At init the bottlenecks should all be open: sigmas small and multipliers 1
  sigma_params = hk.get_parameter(
      name + '_sigma_params',
      shape,
      init=hk.initializers.RandomUniform(minval=0.0, maxval=0.05),
  )
  sigmas = reparameterize_sigma(sigma_params)
  multipliers = hk.get_parameter(
      name + '_multipliers',
      shape,
      init=hk.initializers.Constant(constant=1),
  )
  return sigmas, multipliers


class HkDisentangledRNN(hk.RNNCore):
  """Disentangled RNN."""

  def __init__(
      self,
      config: DisRnnConfig,
  ):
    super().__init__()
    self._l2_scale = config.l2_scale
    self._noiseless_mode = config.noiseless_mode

    self._obs_size = config.obs_size
    self._output_size = config.output_size
    self._latent_size = config.latent_size

    self._update_net_n_units_per_layer = config.update_net_n_units_per_layer
    self._update_net_n_layers = config.update_net_n_layers
    self._choice_net_n_units_per_layer = config.choice_net_n_units_per_layer
    self._choice_net_n_layers = config.choice_net_n_layers

    self._latent_penalty = config.latent_penalty
    self._update_net_obs_penalty = config.update_net_obs_penalty
    self._update_net_latent_penalty = config.update_net_latent_penalty
    self._choice_net_latent_penalty = config.choice_net_latent_penalty
    self._activation = getattr(jax.nn, config.activation)
    self._max_latent_value = config.max_latent_value

    # Get Haiku parameters. IMPORTANT: if you are subclassing HkDisentangledRNN,
    # you must override _get_haiku_parameters to add any new parameters that you
    # need. This way of doing things is necessary for Haiku to work correctly.
    self._get_haiku_parameters()

  def _get_haiku_parameters(self):
    """Initializes parameters for the bottlenecks."""
    self._build_update_bottlenecks()
    self._build_latent_bottlenecks()
    self._build_choice_bottlenecks()
    self._latent_inits = hk.get_parameter(
        'latent_inits',
        (self._latent_size,),
        init=hk.initializers.RandomUniform(minval=-0.1, maxval=0.1),
    )

  def _build_update_bottlenecks(self):
    """Initializes parameters for the update network bottlenecks."""
    # There is one Update Network per latent. Each one gets input from all
    # latents and all observations. These inputs pass through separate
    # bottlenecks.
    # Bottlenecks for observations feeding into update nets
    self._update_net_obs_sigmas, self._update_net_obs_multipliers = (
        get_initial_bottleneck_params(
            shape=(self._obs_size, self._latent_size),
            name='update_net_obs',
        )
    )
    # Bottlenecks for previous latents feeding into update nets
    self._update_net_latent_sigmas, self._update_net_latent_multipliers = (
        get_initial_bottleneck_params(
            shape=(self._latent_size, self._latent_size),
            name='update_net_latent',
        )
    )

  def _build_latent_bottlenecks(self):
    """Initializes parameters for the latent bottlenecks."""
    # Latents will also go through a bottleneck after being updated. These
    # bottlenecks do not need multipliers, the network output can rescale them
    self._latent_sigmas, _ = (
        get_initial_bottleneck_params(
            shape=(self._latent_size,),
            name='latent',
        )
    )

  def _build_choice_bottlenecks(self):
    """Initializes parameters for the choice network bottlenecks."""
    # Choice network gets inputs from the latents, and has a bottleneck on each
    self._choice_net_sigmas, self._choice_net_multipliers = (
        get_initial_bottleneck_params(
            shape=(self._latent_size,),
            name='choice_net',
        )
    )

  def initial_state(self, batch_size: Optional[int]) -> Any:
    # (batch_size, latent_size)
    latents = jnp.ones([batch_size, self._latent_size]) * self._latent_inits
    return latents

  def update_latents(
      self, update_net_inputs: jnp.ndarray, prev_latent_values: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Updates the latents using the update rules.

    Each latent is updated by a separate Update Network, which takes as input
    all previous latents and all additional update rule inputs, and outputs an
    update and a weight. New value of the latent will be the weighted average of
    the previous latent and the update.

    Args:
      update_net_inputs: Additional inputs for the update rules.
      prev_latent_values: The latents from the previous time step.
    Returns:
      new_latent_values: The updated latents.
      penalty_increment: A penalty associated with the update.
    """
    # penalty_increment: (batch_size,)
    batch_size = prev_latent_values.shape[0]
    penalty_increment = jnp.zeros(shape=(batch_size,))

    # Loop over latents. Update each using its own network.
    new_latent_values = jnp.zeros(
        shape=(prev_latent_values.shape[0], self._latent_size)
    )
    for net_i in jnp.arange(self._latent_size):
      update_net_output, update_net_l2 = ResMLP(
          input_size=update_net_inputs.shape[1],
          output_size=2,
          n_units_per_layer=self._update_net_n_units_per_layer,
          n_layers=self._update_net_n_layers,
          activation_fn=self._activation,
          name='update_net',
      )(update_net_inputs[:, :, net_i])
      # Add L2 to the penalty based on weights of the network
      penalty_increment += self._l2_scale * update_net_l2
      # Update the latent multiplicatively, using a weight and a new target
      # derived from the MLP output.
      # Weight needs to be in [0, 1]
      new_latent_weight = jax.nn.sigmoid(update_net_output[:, 0])
      # Target needs to be in [-max_latent_value, max_latent_value]
      new_latent_target = update_net_output[:, 1]
      new_latent_target = self._max_latent_value * jax.nn.tanh(
          new_latent_target / self._max_latent_value
      )
      # New latent value is weighted average of previous value and new target.
      prev_latent_value = prev_latent_values[:, net_i]
      new_latent_value = (
          1 - new_latent_weight
      ) * prev_latent_value + new_latent_weight * new_latent_target
      new_latent_values = new_latent_values.at[:, net_i].set(new_latent_value)

    # Put latent values through an information bottleneck. If we are in
    # noiseless mode, we can skip this.
    new_latent_values, latent_kl_cost = information_bottleneck(
        inputs=new_latent_values,
        sigmas=self._latent_sigmas,
        noiseless_mode=self._noiseless_mode,
    )
    penalty_increment += self._latent_penalty * latent_kl_cost

    return new_latent_values, penalty_increment

  def predict_targets(
      self,
      choice_net_inputs: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predicts the targets using the choice network."""
    batch_size = choice_net_inputs.shape[0]
    penalty_increment = jnp.zeros(shape=(batch_size,))

    predicted_targets, choice_net_l2 = ResMLP(
        input_size=choice_net_inputs.shape[1],
        output_size=self._output_size,
        n_units_per_layer=self._choice_net_n_units_per_layer,
        n_layers=self._choice_net_n_layers,
        activation_fn=self._activation,
        name='choice_net'
        )(choice_net_inputs)
    penalty_increment += self._l2_scale * choice_net_l2

    return predicted_targets, penalty_increment

  def __call__(self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = prev_latents.shape[0]
    penalty = jnp.zeros(shape=(batch_size,))

    # Prepare observations for update nets
    # Expand to have a separate copy per update network:
    # (batch_size, obs_size, latent_size)
    obs_for_update_net = jnp.tile(
        jnp.expand_dims(observations, 2), (1, 1, self._latent_size)
    )
    obs_for_update_net, obs_update_kl = information_bottleneck(
        inputs=obs_for_update_net,
        sigmas=self._update_net_obs_sigmas,
        multipliers=self._update_net_obs_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._update_net_obs_penalty * obs_update_kl

    # Prepare previous latents for update nets
    # Expand to have a separate copy per update network:
    # (batch_size, latent_size, latent_size)
    prev_latents_for_update_net = jnp.tile(
        jnp.expand_dims(prev_latents, 2), (1, 1, self._latent_size)
    )
    prev_latents_for_update_net, latent_update_kl = information_bottleneck(
        inputs=prev_latents_for_update_net,
        sigmas=self._update_net_latent_sigmas,
        multipliers=self._update_net_latent_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._update_net_latent_penalty * latent_update_kl

    # Concatenate processed inputs for the update networks
    # Shape: (batch_size, obs_size + latent_size, latent_size)
    update_net_inputs = jnp.concatenate(
        (obs_for_update_net, prev_latents_for_update_net), axis=1
    )

    # Update latents using update networks
    new_latents, penalty_increment = self.update_latents(
        update_net_inputs, prev_latents
    )
    penalty += penalty_increment

    # Set up choice net inputs from new_latents.
    choice_net_inputs, choice_net_input_kl = information_bottleneck(
        inputs=new_latents,
        sigmas=self._choice_net_sigmas,
        multipliers=self._choice_net_multipliers,
        noiseless_mode=self._noiseless_mode,
    )
    penalty += self._choice_net_latent_penalty * choice_net_input_kl

    predicted_targets, penalty_increment = self.predict_targets(
        choice_net_inputs
    )
    penalty += penalty_increment

    # Output has shape (batch_size, output_size + 1).
    # The first output_size elements are the predicted targets, and the last
    # element is the penalty. We preassign instead of using concatenate to avoid
    # errors caused by silent broadcasting.
    output_shape = (batch_size, self._output_size + 1)
    output = jnp.zeros(output_shape)
    output = output.at[:, :-1].set(predicted_targets)
    output = output.at[:, -1].set(penalty)

    return output, new_latents


def log_bottlenecks(params,
                    open_thresh: float = 0.1,
                    partially_open_thresh: float = 0.25,
                    closed_thresh: float = 0.9) -> dict[str, int]:
  """Computes info about bottlenecks for the base DisRNN."""

  params_disrnn = params['hk_disentangled_rnn']

  latent_sigmas = np.array(
      reparameterize_sigma(params_disrnn['latent_sigma_params'])
  )
  update_obs_sigmas = np.array(
      reparameterize_sigma(
          np.transpose(params_disrnn['update_net_obs_sigma_params'])
      )
  )
  update_latent_sigmas = np.array(
      reparameterize_sigma(
          np.transpose(params_disrnn['update_net_latent_sigma_params'])
      )
  )
  choice_sigmas = np.array(
      reparameterize_sigma(
          np.transpose(params_disrnn['choice_net_sigma_params'])
      )
  )

  latent_bottlenecks_open = np.sum(latent_sigmas < open_thresh)
  choice_bottlenecks_open = np.sum(choice_sigmas < open_thresh)
  update_obs_bottlenecks_open = np.sum(update_obs_sigmas < open_thresh)
  update_latent_bottlenecks_open = np.sum(update_latent_sigmas < open_thresh)
  update_bottlenecks_open = (
      update_obs_bottlenecks_open + update_latent_bottlenecks_open
  )

  latent_bottlenecks_partial = np.sum(latent_sigmas < partially_open_thresh)
  choice_bottlenecks_partial = np.sum(choice_sigmas < partially_open_thresh)
  update_obs_bottlenecks_partial = np.sum(
      update_obs_sigmas < partially_open_thresh
  )
  update_latent_bottlenecks_partial = np.sum(
      update_latent_sigmas < partially_open_thresh
  )
  update_bottlenecks_partial = (
      update_obs_bottlenecks_partial + update_latent_bottlenecks_partial
  )

  latent_bottlenecks_closed = np.sum(latent_sigmas > closed_thresh)
  choice_bottlenecks_closed = np.sum(choice_sigmas > closed_thresh)
  update_obs_bottlenecks_closed = np.sum(update_obs_sigmas > closed_thresh)
  update_latent_bottlenecks_closed = np.sum(
      update_latent_sigmas > closed_thresh
  )
  update_bottlenecks_closed = (
      update_obs_bottlenecks_closed + update_latent_bottlenecks_closed
  )

  bottleneck_dict = {
      'latent_bottlenecks_open': int(latent_bottlenecks_open),
      'latent_bottlenecks_partial': int(latent_bottlenecks_partial),
      'latent_bottlenecks_closed': int(latent_bottlenecks_closed),
      'choice_bottlenecks_open': int(choice_bottlenecks_open),
      'choice_bottlenecks_partial': int(choice_bottlenecks_partial),
      'choice_bottlenecks_closed': int(choice_bottlenecks_closed),
      'update_bottlenecks_open': int(update_bottlenecks_open),
      'update_bottlenecks_partial': int(update_bottlenecks_partial),
      'update_bottlenecks_closed': int(update_bottlenecks_closed),
      }
  return bottleneck_dict


def get_total_sigma(params):
  """Get sum of reparameterized sigmas of a DisRNN."""

  params_disrnn = params['hk_disentangled_rnn']

  latent_bottlenecks = reparameterize_sigma(
      params_disrnn['latent_sigma_params'])
  update_obs_bottlenecks = reparameterize_sigma(
      params_disrnn['update_net_obs_sigma_params'])
  update_latent_bottlenecks = reparameterize_sigma(
      params_disrnn['update_net_latent_sigma_params'])
  choice_bottlenecks = reparameterize_sigma(
      params_disrnn['choice_net_sigma_params'])

  return float(
      jnp.sum(latent_bottlenecks)
      + jnp.sum(update_obs_bottlenecks)
      + jnp.sum(update_latent_bottlenecks)
      + jnp.sum(choice_bottlenecks)
  )


def get_auxiliary_metrics(params: hk.Params) -> dict[str, Any]:
  """Computes auxiliary metrics for the base DisRNN.

  Args:
    params: Haiku parameters of the model.

  Returns:
    A dictionary of auxiliary metrics.
  """

  bottleneck_metrics = log_bottlenecks(params)
  total_sigma_val = get_total_sigma(params)
  return {'total_sigma': total_sigma_val, **bottleneck_metrics}
