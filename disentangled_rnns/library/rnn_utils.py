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

"""Functions to train, evaluate, and examine Haiku RNNs."""

from collections.abc import Callable
import json
import sys
from typing import Any, Literal, Optional
import warnings

from absl import logging
import chex
import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# If we're running on colab, try to import IPython.display so we can display
# progress that way. Otherwise, we will just print.
if 'google.colab' in sys.modules:
  try:
    from IPython import display  # pylint: disable=g-bad-import-order, g-import-not-at-top

    _display_available = True
  except ImportError:
    _display_available = False
else:
  _display_available = False


class DatasetRNN:
  """Holds a dataset for training an RNN, consisting of inputs and targets.

  Both inputs and targets are stored as [timestep, episode, feature]
  Serves them up in batches
  """

  def __init__(
      self,
      xs: np.ndarray,
      ys: np.ndarray,
      y_type: Literal['categorical', 'scalar', 'mixed'] = 'categorical',
      n_classes: Optional[int] = None,
      x_names: Optional[list[str]] = None,
      y_names: Optional[list[str]] = None,
      batch_size: Optional[int] = None,
      batch_mode: Literal['single', 'rolling', 'random'] = 'single',
  ):
    """Do error checking and bin up the dataset into batches.

    Args:
      xs: Values to become inputs to the network. Should have dimensionality
        [timestep, episode, feature]
      ys: Values to become output targets for the RNN. Should have
        dimensionality [timestep, episode, feature]
      y_type: Either 'categorical','scalar' or 'mixed'. If 'categorical',
        targets must be integers. If 'mixed', first element is assumed to be
        categorical.
      n_classes: The number of classes in the categorical targets. If not
        specified, will be inferred from the data.
      x_names: A list of names for the features in xs. If not supplied, will be
        generated automatically.
      y_names: A list of names for the features in ys. If not supplied, will be
        generated automatically.
      batch_size: The size of the batch (number of episodes) to serve up each
        time next() is called. If not specified, all episodes in the dataset
        will be served
      batch_mode: How to batch the dataset. Options are 'single', 'rolling', and
        'random'. In 'single' mode, all episodes are served together. In
        'rolling' mode, a new batch is formed by rolling the episodes forward in
        time. In 'random' mode, a new batch is formed by randomly sampling
        episodes. If not specified, will default to 'single'.
    """
    ##################
    # Error checking #
    ##################

    if y_type not in ['categorical', 'scalar', 'mixed']:
      raise ValueError(
          f'y_type {y_type} must be either "categorical","scalar" or "mixed".'
      )

    if y_type == 'categorical':
      # Check validity and determine the number of classes
      if ys.shape[-1] != 1:
        raise NotImplementedError(
            'Categorical targets are assumed to have dimensionality'
            f'(n_timesteps, n_episodes, 1). Got {ys.shape} instead. If you need'
            'multiple distinct types of categorical targets, feel free to '
            'implement this and send a CL'
        )

    if y_type in ['categorical', 'mixed']:
      # NOTE: By convention, for y_type=='mixed' the first element of the target
      # is assumed to be categorical.
      categorical_index = 0
      categorical_ys = ys[:, :, categorical_index]
      uniques = np.unique(categorical_ys)
      if not np.all(np.isclose(uniques, np.round(uniques))):
        raise ValueError(
            f'Categorical targets must be integers. Got {uniques} instead'
        )
      # Infer or check the number of classes. It should be equal to or greater
      # than the number of unique nonnegative values
      n_classes_expected = np.sum(uniques >= 0)
      if n_classes is None:
        n_classes = n_classes_expected
      else:
        if n_classes < n_classes_expected:
          raise ValueError(
              f'Based on unique y values {uniques}, expected n_classes to be at'
              f' least {n_classes_expected}. Instead it is {n_classes}'
          )
    else:
      # If not categorical, n_classes is not defined
      n_classes = None

    # Do xs and ys have the same number of timesteps?
    if xs.shape[0] != ys.shape[0]:
      raise ValueError(
          f'Number of timesteps in xs {xs.shape[0]} must be equal to number of'
          f' timesteps in ys {xs.shape[0]}.'
      )

    # Do xs and ys have the same number of episodes?
    if xs.shape[1] != ys.shape[1]:
      raise ValueError(
          f'Number of episodes in xs {xs.shape[1]} must be equal to number of'
          ' episodes in ys {ys.shape[1]}.'
      )

    # Are xs and ys non-empty?
    if xs.shape[0] == 0 or xs.shape[1] == 0:
      raise ValueError(
          'xs and ys must be non-empty. Got xs.shape = {xs.shape} and'
          f' ys.shape = {ys.shape} instead.'
      )

    # Process feature and target names
    if x_names is None:
      x_names = [f'Observation {i}' for i in range(xs.shape[2])]
    else:
      if len(x_names) != xs.shape[2]:
        raise ValueError(
            f'Number of x_names {len(x_names)} must be equal to number of'
            f' features in xs {xs.shape[-1]}.'
        )

    if y_names is None:
      y_names = [f'Target {i}' for i in range(ys.shape[2])]
    else:
      if len(y_names) != ys.shape[2]:
        raise ValueError(
            f'Number of y_names {len(y_names)} must be equal to number of'
            f' features in ys {ys.shape[-1]}.'
        )

    ####################
    # Property setting #
    ####################
    # If batch size not specified, use all episodes in the dataset
    if batch_size is None:
      batch_size = xs.shape[1]

    self.x_names = x_names
    self.y_names = y_names
    self.y_type = y_type
    self.n_classes = n_classes
    self.batch_size = batch_size
    self._xs = xs
    self._ys = ys
    self._n_episodes = self._xs.shape[1]
    self._n_timesteps = self._xs.shape[0]
    self.batch_mode = batch_mode
    self._current_start_index = 0  # For batch_mode='rolling'

  def __iter__(self):
    return self

  def get_all(self):
    """Returns all the data in the dataset."""
    return self._xs, self._ys

  def __next__(self):
    """Return a batch of data, including both xs and ys."""

    if self.batch_size == 0:
      # Return empty arrays with correct number of dimensions
      empty_xs = np.empty(
          (self._n_timesteps, 0, self._xs.shape[2]), dtype=self._xs.dtype
      )
      empty_ys = np.empty(
          (self._n_timesteps, 0, self._ys.shape[2]), dtype=self._ys.dtype
      )
      warnings.warn('DatasetRNN batch_size is 0. Returning an empty batch.')
      return empty_xs, empty_ys

    if self.batch_mode == 'single':
      return self.get_all()

    elif self.batch_mode == 'rolling':
      # Generate indices starting from the current index, wrapping using modulo
      indices = np.arange(
          self._current_start_index,
          self._current_start_index + self.batch_size,
      )
      batch_inds = indices % self._n_episodes

      # Get the chunks of data
      xs_batch, ys_batch = self._xs[:, batch_inds], self._ys[:, batch_inds]
      # Update the starting index for the next batch, wrapping around
      self._current_start_index = (
          self._current_start_index + self.batch_size
      ) % self._n_episodes
      return xs_batch, ys_batch

    elif self.batch_mode == 'random':
      inds_to_get = np.random.choice(self._n_episodes, size=self.batch_size)
      return self._xs[:, inds_to_get], self._ys[:, inds_to_get]

    else:
      raise ValueError(
          f'Batch mode {self.batch_mode} not recognized. Must be one of'
          ' "single", "rolling", or "random".'
      )


def split_dataset(
    dataset: DatasetRNN, eval_every_n: int
) -> tuple[DatasetRNN, DatasetRNN]:
  """Split a dataset into train and eval sets."""
  xs, ys = dataset.get_all()
  n_sessions = xs.shape[1]
  train_sessions = np.ones(n_sessions, dtype=bool)
  train_sessions[np.arange(eval_every_n - 1, n_sessions, eval_every_n)] = False
  eval_sessions = np.logical_not(train_sessions)

  dataset_train = DatasetRNN(
      xs[:, train_sessions, :],
      ys[:, train_sessions, :],
      x_names=dataset.x_names,
      y_names=dataset.y_names,
      y_type=dataset.y_type,
      n_classes=dataset.n_classes,
      batch_size=dataset.batch_size,
      batch_mode=dataset.batch_mode,
  )
  dataset_eval = DatasetRNN(
      xs[:, eval_sessions, :],
      ys[:, eval_sessions, :],
      x_names=dataset.x_names,
      y_names=dataset.y_names,
      y_type=dataset.y_type,
      n_classes=dataset.n_classes,
      batch_size=dataset.batch_size,
      batch_mode=dataset.batch_mode,
  )
  return dataset_train, dataset_eval


def nan_in_dict(d: np.ndarray | dict[str, Any]):
  """Check a nested dict (e.g. hk.params) for nans."""
  if not isinstance(d, dict):
    return np.any(np.isnan(d))
  else:
    return any(nan_in_dict(v) for v in d.values())


@jax.jit
def sse(ys: np.ndarray, y_hats: np.ndarray) -> float:
  """Compute the sum of squared errors between ys and y_hats."""
  # We want to allow the training code to pass NaNs for missing targets. These
  # missing targets should generate no gradients. The mask here does that.
  mask = jnp.logical_not(jnp.isnan(ys))
  errors = ys - y_hats
  masked_errors = jnp.where(mask, errors, 0.0)
  sum_squared_error = jnp.sum(jnp.square(masked_errors))
  return sum_squared_error  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def mse(ys: np.ndarray, y_hats: np.ndarray) -> float:
  """Compute the mean squared error between ys and y_hats."""
  # We want to allow the training code to pass NaNs for missing targets. These
  # missing targets should generate no gradients. The mask here does that.
  mask = jnp.logical_not(jnp.isnan(ys))
  sq_errors = sse(ys, y_hats)
  loss = sq_errors / jnp.sum(mask)
  return loss  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def categorical_neg_log_likelihood(
    labels: np.ndarray, output_logits: np.ndarray
) -> float:
  """Compute total log-likelihood of a set of labels given a set of logits."""
  # Mask any errors for which label is negative
  mask = jnp.logical_not(labels < 0)
  log_probs = jax.nn.log_softmax(output_logits)
  if labels.shape[2] != 1:
    raise ValueError(
        'Categorical loss function requires targets to be of dimensionality'
        ' (n_timesteps, n_episodes, 1)'
    )
  one_hot_labels = jax.nn.one_hot(
      labels[:, :, 0], num_classes=output_logits.shape[-1]
  )
  log_liks = one_hot_labels * log_probs
  masked_log_liks = jnp.where(mask, log_liks, 0.0)
  loss = -jnp.sum(masked_log_liks)
  n_unmasked_samples = jnp.sum(mask)
  return loss, n_unmasked_samples  # pytype: disable=bad-return-type  # jnp-type


def likelihood_and_sse(
    ys: np.ndarray,
    y_hats: np.ndarray,
    likelihood_weight: float = 1.0,
    n_categorical_targets: int = 2,
    n_continuous_targets: int = 1,
) -> float:
  """Compute a weighted average of categorical log-likelihood and MSE."""
  categorical_y_hats = y_hats[:, :, 0:n_categorical_targets]
  categorical_ys = ys[:, :, 0:1]

  # All trials with a negative target are masked.
  mask = jnp.logical_not(categorical_ys < 0)

  continuous_y_hats = y_hats[
      :, :, n_categorical_targets : n_categorical_targets + n_continuous_targets
  ]
  continuous_ys = ys[:, :, 1 : 1 + n_continuous_targets]

  log_likelihood, _ = categorical_neg_log_likelihood(
      categorical_ys, categorical_y_hats
  )
  # All trials with an invalid categorical target are masked.
  continuous_ys = jnp.where(mask, continuous_ys, jnp.nan)
  sum_squared_error = sse(continuous_ys, continuous_y_hats)

  return log_likelihood * likelihood_weight + sum_squared_error * (
      1 - likelihood_weight
  )


def normalized_likelihood_and_mse(
    ys: np.ndarray,
    yhats: np.ndarray,
    likelihood_weight: float = 1.0,
    n_categorical_targets: int = 2,
    n_continuous_targets: int = 1,
) -> float | jnp.ndarray:
  """Compute a weighted average of normalized categorical log-likelihood and MSE."""

  # Convention is that the first two elements of yhats are for categorical
  # targets, and the next element is for continuous targets.
  # Optionally the last element of yhats is a penalty.
  categorical_y_hats = yhats[:, :, 0:n_categorical_targets]
  categorical_ys = ys[:, :, 0:1]

  mask = jnp.logical_not(categorical_ys < 0)
  continuous_y_hats = yhats[
      :, :, n_categorical_targets : n_categorical_targets + n_continuous_targets
  ]
  continuous_ys = ys[:, :, 1 : 1 + n_continuous_targets]
  continuous_ys = jnp.where(mask, continuous_ys, jnp.nan)

  normlik_categorical = normalized_likelihood(
      categorical_ys, categorical_y_hats
  )
  normlik_mse = mse(continuous_ys, continuous_y_hats)
  return normlik_categorical * likelihood_weight + normlik_mse * (
      1 - likelihood_weight
  )


def normalized_likelihood(
    labels: np.ndarray, output_logits: np.ndarray
) -> float:
  total_nll, n_unmasked_samples = categorical_neg_log_likelihood(
      labels, output_logits
  )
  normlik = np.exp((-1 * total_nll) / (n_unmasked_samples))
  return normlik


@jax.jit
def compute_penalty(targets: np.ndarray, outputs: np.ndarray) -> float:
  """Compute the penalty, masking invalid timesteps."""
  # Categorical mask: Exclude targets exactly equal to -1
  categorical_mask = jnp.logical_not(targets == -1)
  # Continuous mask: exclude targets that are NaN
  continuous_mask = jnp.logical_not(jnp.isnan(targets))
  mask = jnp.logical_and(categorical_mask, continuous_mask)
  # A trial is unmasked if any of its targets are unmasked.
  mask = jnp.any(mask, axis=-1)

  trialwise_penalty = outputs[:, :, -1]
  penalty = jnp.sum(jnp.multiply(trialwise_penalty, mask))
  n_unmasked_samples = jnp.sum(mask)

  return penalty, n_unmasked_samples  # pytype: disable=bad-return-type  # jnp-type


## Training Loop
def train_network(
    make_network: Callable[[], hk.RNNCore],
    training_dataset: DatasetRNN,
    validation_dataset: Optional[DatasetRNN],
    opt: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[chex.PRNGKey] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[hk.Params] = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1e10,
    loss_param: dict[str, float] | float = 1.0,
    loss: Literal[
        'mse',
        'penalized_mse',
        'categorical',
        'penalized_categorical',
        'hybrid',
        'penalized_hybrid',
    ] = 'mse',
    log_losses_every: int = 10,
    do_plot: bool = False,
    report_progress_by: Literal['print', 'log', 'none'] = 'print',
) -> tuple[hk.Params, optax.OptState, dict[str, np.ndarray]]:
  """Trains a network.

  Args:
    make_network: A function that, when called, returns a Haiku RNN
    training_dataset: A DatasetRNN, containing the data you wish to train on
    validation_dataset: A DatasetRNN, containing the data you wish to use for
      validation
    opt: The optimizer yuo'd like to use to train the network
    random_key: A jax random key, to be used in initializing the network
    opt_state: An optimzier state suitable for opt If not specified, will
      initialize a new optimizer from scratch
    params:  A set of parameters suitable for the network given by make_network
      If not specified, will begin training a network from scratch
    n_steps: An integer giving the number of steps you'd like to train for
    max_grad_norm:  Gradient clipping. Default to a very high ceiling
    loss_param: Parameters to pass to the loss function. Can be a dictionary for
      fine-grained control over different loss components (e.g.,
      {'penalty_scale': 0.1, 'likelihood_weight': 0.8}) or a single float for
      simpler losses.
    loss: The loss function to use. Options are 'mse', 'penalized_mse',
      'categorical', 'penalized_categorical', 'hybrid', 'penalized_hybrid'.
    log_losses_every: How many training steps between each time we check for
      errors and log the loss
    do_plot: Boolean that controls whether a learning curve is plotted
    report_progress_by: Mode for reporting real-time progress. Options are
      "print" for printing to the console, "log" for using absl logging, and
      "none" for no output.

  Returns:
    params: Trained parameters
    opt_state: Optimizer state at the end of training
    losses: Losses on both datasets
  """
  sample_xs, _ = next(training_dataset)  # Get a sample input, for shape

  # Haiku, step one: Define the batched network
  def unroll_network(xs):
    core = make_network()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

  # Haiku, step two: Transform the network into a pair of functions
  # (model.init and model.apply)
  model = hk.transform(unroll_network)

  ################
  # PARSE INPUTS #
  ################
  if random_key is None:
    random_key = jax.random.PRNGKey(0)
  # If params have not been supplied, start training from scratch
  if params is None:
    random_key, key1 = jax.random.split(random_key)
    params = model.init(key1, sample_xs)
  # It an optimizer state has not been supplied, start optimizer from scratch
  if opt_state is None:
    opt_state = opt.init(params)

  loss_param_dict = (
      loss_param if isinstance(loss_param, dict) else {'value': loss_param}
  )

  def get_loss_param(
      loss_param_dict: dict[str, float], key: str, default: float
  ) -> float:
    """Helper to get a value from loss_param_dict."""
    return loss_param_dict.get(key, loss_param_dict.get('value', default))

  ############################################
  # Define possible losses and training step #
  ###########################################
  def mse_loss(params, xs, ys, random_key) -> float:
    y_hats = model.apply(params, random_key, xs)
    loss = mse(ys, y_hats)
    return loss

  def penalized_mse_loss(
      params, xs, ys, random_key, loss_param=loss_param_dict
  ) -> float:
    """Treats the last element of the model outputs as a penalty."""
    # (n_steps, n_episodes, n_targets+1)
    model_output = model.apply(params, random_key, xs)
    y_hats = model_output[:, :, :-1]
    penalty, n_unmasked_samples = compute_penalty(ys, model_output)
    penalty_scale = get_loss_param(loss_param, 'penalty_scale', 1.0)
    loss = mse(ys, y_hats) + penalty_scale * penalty / n_unmasked_samples
    return loss

  def categorical_loss(
      params, xs: np.ndarray, labels: np.ndarray, random_key
  ) -> float:
    output_logits = model.apply(params, random_key, xs)
    nll, n_unmasked_samples = categorical_neg_log_likelihood(
        labels, output_logits
    )
    return nll / n_unmasked_samples

  def penalized_categorical_loss(
      params, xs, targets, random_key, loss_param=loss_param_dict
  ) -> float:
    """Treats the last element of the model outputs as a penalty."""
    # (n_steps, n_episodes, n_targets)
    model_output = model.apply(params, random_key, xs)
    output_logits = model_output[:, :, :-1]
    penalty, _ = compute_penalty(targets, model_output)
    nll, n_unmasked_samples = categorical_neg_log_likelihood(
        targets, output_logits
    )
    penalty_scale = get_loss_param(loss_param, 'penalty_scale', 1.0)
    avg_nll = nll / n_unmasked_samples
    avg_penalty = penalty / n_unmasked_samples
    loss = avg_nll + penalty_scale * avg_penalty
    return loss

  def hybrid_loss(
      params, xs, ys, random_key, loss_param=loss_param_dict
  ) -> float:
    """A loss that combines categorical and continuous targets."""

    model_output = model.apply(params, random_key, xs)
    y_hats = model_output
    likelihood_weight = get_loss_param(loss_param, 'likelihood_weight', 1.0)
    loss = jax.jit(likelihood_and_sse)(
        ys, y_hats, likelihood_weight=likelihood_weight
    )
    return loss

  def penalized_hybrid_loss(
      params, xs, ys, random_key, loss_param=loss_param_dict
  ) -> float:
    """A hybrid loss with a penalty."""

    penalty_scale = get_loss_param(loss_param, 'penalty_scale', 1.0)
    model_output = model.apply(params, random_key, xs)

    # model_output has the continuous and categorical targets first followed by
    # the penalty. The likelihood_and_sse functions handles
    # ignoring the penalty, hence we don't need to do anything special here.
    y_hats = model_output
    likelihood_weight = get_loss_param(loss_param, 'likelihood_weight', 1.0)
    supervised_loss = likelihood_and_sse(
        ys, y_hats, likelihood_weight=likelihood_weight
    )
    # Supervised loss here is a sum not an average, so use the raw penalty
    # without dividing by n_unmasked_samples
    # TODO(siddhantjain): Evaluate whether we should use averaging here too.
    penalty, _ = compute_penalty(ys, y_hats)
    loss = supervised_loss + penalty_scale * penalty
    return loss

  losses = {
      'mse': mse_loss,
      'penalized_mse': penalized_mse_loss,
      'categorical': categorical_loss,
      'penalized_categorical': penalized_categorical_loss,
      'hybrid': hybrid_loss,
      'penalized_hybrid': penalized_hybrid_loss,
  }
  compute_loss = jax.jit(losses[loss])

  @jax.jit
  def train_step(
      params, opt_state, xs, ys, random_key
  ) -> tuple[float, Any, Any]:
    """One training step."""
    loss, grads = jax.value_and_grad(compute_loss, argnums=0)(
        params, xs, ys, random_key
    )
    grads, opt_state = opt.update(grads, opt_state)
    clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
    params = optax.apply_updates(params, clipped_grads)
    return loss, params, opt_state

  #################
  # Training Loop #
  #################

  training_loss = []
  validation_loss = []
  l_validation = np.nan

  xs_train, ys_train = next(training_dataset)

  if validation_dataset is not None:
    xs_eval, ys_eval = validation_dataset.get_all()
  else:
    xs_eval = None
    ys_eval = None
  # Train the network!
  for step in range(n_steps):
    random_key, subkey_train, subkey_validation = jax.random.split(
        random_key, 3
    )
    # If the training dataset uses batching, get a new batch
    if training_dataset.batch_mode != 'single':
      xs_train, ys_train = next(training_dataset)

    loss, params, opt_state = train_step(
        params, opt_state, xs_train, ys_train, subkey_train
    )

    # Check for errors and report progress
    if step % log_losses_every == 0 or step == n_steps - 1:
      if nan_in_dict(params):
        raise ValueError('NaN in params')
      if np.isnan(loss):
        raise ValueError('NaN in loss')
      if loss > 1e50:
        raise ValueError('Loss is too large')

      # Test on validation data
      if validation_dataset is not None:
        l_validation = compute_loss(params, xs_eval, ys_eval, subkey_validation)

      validation_loss.append(float(l_validation))
      training_loss.append(float(loss))

      log_str = (
          f'Step {step + 1} of {n_steps}. '
          f'Training Loss: {loss:.2e}. '
          f'Validation Loss: {l_validation:.2e}'
      )

      if report_progress_by == 'print':
        # On colab, print does not always work, so try to use display
        if _display_available:
          display.clear_output(wait=True)
          display.display(log_str)
        else:
          print(log_str)
      elif report_progress_by == 'log':
        logging.info(log_str)
      elif report_progress_by == 'none':
        pass
      else:
        warnings.warn(f'Unknown report_progress_by mode: {report_progress_by}')

  # If we actually did any training, print final loss and make a nice plot
  if n_steps > 1 and do_plot:

    plt.figure()
    plt.semilogy(training_loss, color='black')
    plt.semilogy(validation_loss, color='tab:red', linestyle='dashed')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Loss')
    plt.legend(('Training Set', 'Validation Set'))
    plt.title('Loss over Training')

  losses = {
      'training_loss': np.array(training_loss),
      'validation_loss': np.array(validation_loss),
  }

  if training_loss and np.isnan(training_loss[-1]):
    raise ValueError('NaN in loss')
  return params, opt_state, losses


def eval_network(
    make_network: Callable[[], hk.RNNCore],
    params: hk.Params,
    xs: np.ndarray,
) -> tuple[np.ndarray, Any]:
  """Run an RNN with specified params and inputs. Track internal state.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    xs: A batch of inputs [timesteps, episodes, features] suitable for the model

  Returns:
    y_hats: Network outputs at each timestep
    states: Network states at each timestep
  """

  def unroll_network(xs):
    core = make_network()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, states = hk.dynamic_unroll(core, xs, state, return_all_states=True)
    return ys, states

  model = hk.transform(unroll_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  apply = jax.jit(model.apply)
  y_hats, states = apply(params, key, xs)

  if isinstance(states, tuple) or isinstance(states, list) and len(states) == 1:
    states = np.array(states[0])

  # States should now be (n_timesteps, n_episodes, n_hidden)
  assert (
      states.shape[0] == xs.shape[0]
  ), 'States and inputs should have the same number of timesteps.'
  assert (
      states.shape[1] == xs.shape[1]
  ), 'States and inputs should have the same number of episodes.'

  return np.asarray(y_hats), states


def get_apply(
    make_network: Callable[[], hk.RNNCore],
    verbose: bool = False,
) -> tuple[np.ndarray, Any]:
  """Get a jitted apply function for a network.

  Args:
    make_network: Network constructor
    verbose: if True, print a statement when calling step_sub

  Returns:
    apply: a jitted apply function for the network
  """

  def step_sub(x, state):
    core = make_network()
    y_hat, new_state = core(x, state)
    if verbose:
      print('[step_sub] this should only print once per jit.')
    return y_hat, new_state

  model = hk.transform(step_sub)
  apply = jax.jit(model.apply)

  return apply  # pytype: disable=bad-return-type  # jax-api-types


def step_network(
    make_network: Callable[[], hk.RNNCore],
    params: hk.Params,
    state: Any,
    xs: Any,
    apply: Any = None,
) -> tuple[Any, Any, Any]:
  """Run an RNN for just a single step on a single input, with batching.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    state: An RNN state suitable for that network
    xs: An input for a single timestep from a single episode, with shape
      [n_features]
    apply: A jitted function that applies the network to a single input. If not
      supplied, will be generated from the network architecture

  Returns:
    y_hat: The output given by the network, with dimensionality [n_features]
    new_state: The new RNN state of the network
  """

  if apply is None:
    apply = get_apply(make_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hat, new_state = apply(params, key, xs, state)

  return y_hat, new_state, apply


def get_initial_state(
    make_network: Callable[[], hk.RNNCore],
    params: Optional[Any] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Any:
  """Get the default initial state for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: Optional parameters for the Hk function. If not passed, will init
      new parameters. For many models this will not affect initial state
    batch_size: The batch size to use when generating the initial state
    seed: A seed for the random number generator that makes params (for most
      networks this will not affect initial state)

  Returns:
    initial_state: An initial state from that network
  """

  def unroll_network():
    core = make_network()
    state = core.initial_state(batch_size=batch_size)

    return state

  model = hk.transform(unroll_network)

  # If no params were supplied, generate random params.
  random_key = jax.random.PRNGKey(seed)
  if params is None:
    params = model.init(random_key)

  apply = jax.jit(model.apply)
  initial_state = apply(params, random_key)

  return initial_state


def get_new_params(
    make_network: Callable[..., hk.RNNCore],
    random_key: Optional[jax.Array] = None,
) -> Any:
  """Get a new set of random parameters for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    random_key: a Jax random key

  Returns:
    params: A set of parameters suitable for the architecture
  """

  # If no key has been supplied, pick a random one.
  if random_key is None:
    random_key = jax.random.PRNGKey(np.random.randint(2**32))

  def unroll_network():
    core = make_network()
    state = core.initial_state(batch_size=1)

    return state

  model = hk.transform(unroll_network)
  init = jax.jit(model.init)
  params = init(random_key)

  return params


def eval_feedforward_network(
    make_network: Callable[..., Any], params: hk.Params, xs: np.ndarray
) -> np.ndarray:
  """Run a feedforward network with specified params and inputs."""

  def forward(xs):
    net = make_network()
    y_hats = net(xs)
    return y_hats

  model = hk.transform(forward)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  apply = jax.jit(model.apply)
  y_hats = apply(params, key, xs)

  return np.array(y_hats)


def to_np(list_dict: dict[str, Any]):
  """Converts all numerical lists in a dict to np arrays.

  Elements that are convertible to numpy are converted. Elements that are dicts
  are recursively unpacked in the same way. Other elements are left unchanged.
  The intended use case is reconstructing a dict from json that was saved with
  NpEncoder and has had all its np arrays converted to lists.

  Args:
    list_dict: A dict or hierarchical tree of dicts

  Returns:
    np_dict: A dict or tree of dicts with the same structure, in which any
      numerical lists have been converted into np arrays.
  """
  np_dict = dict()
  # Traverse the tree of dicts. If we find anything that can be converted to an
  # np array, convert it. Otherwise, leave it unchanged.
  for key, value in list_dict.items():
    if isinstance(value, dict):
      np_dict[key] = to_np(value)
    else:
      try:
        np_dict[key] = np.array(value)
      except ValueError:
        print(f'Not converting {key}. Type is {type(value)}')
        np_dict[key] = value
  return np_dict


class NpJnpJsonEncoder(json.JSONEncoder):
  """Encode Numpy and JAX arrays/types in a format suitable for json.dump."""

  def default(self, o: Any):
    if o is None:
      return None

    # Lists and dicts: Handle recursively
    if isinstance(o, list):
      return [self.default(x) for x in o]
    if isinstance(o, dict):
      return {k: self.default(v) for k, v in o.items()}

    # Basic Python types
    if isinstance(o, bool):
      return bool(o)  # Ensure standard Python bool
    if isinstance(o, (int, float, str)):
      return o  # Already JSON serializable

    # NumPy types
    if isinstance(o, (np.bool_)):
      return bool(o)
    if isinstance(o, np.integer):
      return int(o)
    if isinstance(o, np.floating):
      return float(o)
    if isinstance(o, np.ndarray):
      return o.tolist()

    # JAX types
    if isinstance(o, (jnp.bool_)):
      return bool(o)
    if isinstance(o, jnp.integer):
      return int(o)
    if isinstance(o, jnp.floating):
      return float(o)
    if isinstance(o, jnp.ndarray):
      return o.tolist()

    # Other types: attempt to fallback to base class default method
    try:
      return super().default(o)
    except TypeError as exc:
      raise TypeError(
          f'Object of type {type(o).__name__} is not JSON serializable'
      ) from exc
