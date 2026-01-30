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
from typing import Any, Literal, Mapping
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

RnnParams = Mapping[str, Mapping[str, Any]]

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
      xs: np.typing.NDArray[np.number],
      ys: np.typing.NDArray[np.number],
      y_type: Literal['categorical', 'scalar', 'mixed'] = 'categorical',
      batch_mode: Literal['single', 'rolling', 'random'] = 'random',
      batch_size: int | None = 1024,
      n_classes: int | None = None,
      x_names: list[str] | None = None,
      y_names: list[str] | None = None,
      rng: np.random.Generator | None = None,
  ):
    """Do error checking and define properties.

    Args:
      xs: Values to become inputs to the network. Should have dimensionality
        [timestep, episode, feature]. Must be numeric, will be cast to float32.
      ys: Values to become output targets for the RNN. Should have
        dimensionality [timestep, episode, feature].
      y_type: The type of the target variable(s). Options are:
        'categorical': Targets must be integers representing classes.
        'scalar': Targets must be numeric and will be cast to float32.
        'mixed': Assumes the first target feature is categorical and the rest
          are scalar.
      batch_mode: How to batch the dataset. Options are:
        'random' [default]: Batches are formed by sampling episodes randomly
           with replacement.
        'rolling': Batches are formed by taking consecutive episodes in time,
           wrapping around at the end of the dataset.
        'single': All episodes are served together in a single batch.
      batch_size: The size of the batch (number of episodes) to serve up each
        time next() is called. If batch_mode is 'single', this is ignored and
        all episodes are served together in a single batch.
      n_classes: The number of classes in the categorical targets. If not
        specified, will be inferred from the data.
      x_names: A list of names for the features in xs. If not supplied, will be
        generated automatically.
      y_names: A list of names for the features in ys. If not supplied, will be
        generated automatically.
      rng: A numpy random number generator. If not supplied, a new one will be
        created.
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
      # For categorical ys, ensure they are integers or close to integers
      uniques = np.unique(ys)
      if not np.all(np.isclose(uniques, np.round(uniques))):
        raise ValueError(
            'For y_type="categorical", ys must be integers or floats close to'
            f' integers. Got unique values: {uniques}'
        )

    if y_type in ['categorical', 'mixed']:
      # By convention, for y_type=='mixed' the first element of the target
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
          f'xs and ys must be non-empty. Got xs.shape = {xs.shape} and'
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
    # If batch size is None, set it to the number of episodes.
    if batch_size is None:
      assert batch_mode not in ['rolling', 'random'], (
          f'batch_mode was {batch_mode}, which requires batch_size to be'
          f' specified. Instead, batch_size was {batch_size}.'
      )

    self.x_names = x_names
    self.y_names = y_names
    self.y_type = y_type
    self.n_classes = n_classes
    self.batch_size = batch_size
    self._xs = xs.astype(np.float32)
    self._ys = ys.astype(np.float32)
    self._n_episodes = self._xs.shape[1]
    self._n_timesteps = self._xs.shape[0]
    self.batch_mode = batch_mode
    self._current_start_index = 0  # For batch_mode='rolling'
    self.rng = rng if rng is not None else np.random.default_rng()

  def __iter__(self):
    return self

  def get_all(self):
    """Returns all the data in the dataset. Use this for evaluation."""
    return {'xs': self._xs, 'ys': self._ys}

  def __next__(self):
    """Return a batch of data. Use this for training."""

    if self.batch_size == 0:
      # Return empty arrays with correct number of dimensions
      empty_xs = np.empty(
          (self._n_timesteps, 0, self._xs.shape[2]), dtype=self._xs.dtype
      )
      empty_ys = np.empty(
          (self._n_timesteps, 0, self._ys.shape[2]), dtype=self._ys.dtype
      )
      warnings.warn('DatasetRNN batch_size is 0. Returning an empty batch.')
      return {'xs': empty_xs, 'ys': empty_ys}

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
      return {'xs': xs_batch, 'ys': ys_batch}

    elif self.batch_mode == 'random':
      inds_to_get = self.rng.choice(self._n_episodes, size=self.batch_size)
      return {'xs': self._xs[:, inds_to_get], 'ys': self._ys[:, inds_to_get]}

    else:
      raise ValueError(
          f'Batch mode {self.batch_mode} not recognized. Must be one of'
          ' "single", "rolling", or "random".'
      )


def split_dataset(
    dataset: DatasetRNN, eval_every_n: int, eval_offset: int = 1
) -> tuple[DatasetRNN, DatasetRNN]:
  """Split a dataset into train and eval sets."""
  data = dataset.get_all()
  xs, ys = data['xs'], data['ys']
  if data.keys() != {'xs', 'ys'}:
    raise NotImplementedError(
        f'Splitting is only implemented for datasets with keys xs and ys. This'
        f' one has keys {data.keys()}.'
        ' If you need to split a dataset with other data, feel free to '
        'implement this and send a CL!'
    )

  n_sessions = xs.shape[1]
  train_sessions = np.ones(n_sessions, dtype=bool)
  if eval_offset < 0 or eval_offset > eval_every_n - 1:
    raise ValueError(
        f'eval_offset {eval_offset} must be between 0 and {eval_every_n - 1}.'
        f' Got {eval_offset} instead.'
    )
  train_sessions[np.arange(eval_offset, n_sessions, eval_every_n)] = False
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
      rng=dataset.rng,
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
      rng=dataset.rng,
  )
  return dataset_train, dataset_eval


def nan_in_dict(d: np.ndarray | dict[str, Any]):
  """Check a nested dict (e.g. RnnParams) for nans."""
  if not isinstance(d, dict):
    return np.any(np.isnan(d))
  else:
    return any(nan_in_dict(v) for v in d.values())


@jax.jit
def sse(ys: np.ndarray, y_hats: np.ndarray) -> float:
  """Computes the sum of squared errors, ignoring NaNs in the targets.

  Args:
    ys: The ground truth targets. NaNs are treated as missing values and are
      ignored in the loss calculation.
    y_hats: The predicted values.

  Returns:
    The sum of squared errors.
  """
  mask = jnp.logical_not(jnp.isnan(ys))
  errors = ys - y_hats
  masked_errors = jnp.where(mask, errors, 0.0)
  sum_squared_error = jnp.sum(jnp.square(masked_errors))
  return sum_squared_error  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def mse(ys: np.ndarray, y_hats: np.ndarray) -> float:
  """Computes the mean squared error, ignoring NaNs in the targets.

  Args:
    ys: The ground truth targets. NaNs are treated as missing values and are
      ignored in the loss calculation.
    y_hats: The predicted values.

  Returns:
    The mean squared error.
  """
  mask = jnp.logical_not(jnp.isnan(ys))
  sq_errors = sse(ys, y_hats)
  loss = sq_errors / jnp.sum(mask)
  return loss  # pytype: disable=bad-return-type  # jnp-type


@jax.jit
def categorical_neg_log_likelihood(
    labels: np.ndarray,
    output_logits: np.ndarray,
    valid_actions_mask: np.ndarray | None = None,
) -> tuple[float, int]:
  """Compute total log-likelihood of a set of labels given a set of logits.

  Also computes the total number of valid samples, for use in calculating
  normalized likelihood.

  Args:
    labels: An array of shape (n_timesteps, n_episodes, 1) containing the
      categorical labels. Negative values are treated as masked.
    output_logits: An array of shape (n_timesteps, n_episodes, n_classes)
      containing the logits output by the network.
    valid_actions_mask: An array of shape (n_timesteps, n_episodes, n_classes)
      containing a binary mask for the logits. If None, all logits are
      considered valid actions.

  Returns:
    A tuple containing:
      - loss: The total negative log-likelihood.
      - n_unmasked_samples: The total number of valid (unmasked) samples.
  """
  # Mask any errors for which label is negative
  mask = jnp.logical_not(labels < 0)
  log_probs = jax.nn.log_softmax(output_logits, where=valid_actions_mask)
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
  """Computes a weighted sum of categorical NLL and SSE for mixed targets.

  This loss is unnormalized.

  By convention, for mixed targets:
  - `ys` contains categorical targets at index 0 and continuous targets at
    indices [1, 1 + n_continuous_targets).
  - `y_hats` contains logits for categorical targets at indices
    [0, n_categorical_targets) and predictions for continuous targets at indices
    [n_categorical_targets, n_categorical_targets + n_continuous_targets).

  Args:
    ys: Ground truth targets.
    y_hats: Network predictions (logits for categorical, values for continuous).
    likelihood_weight: The weight for the categorical negative log-likelihood.
      The SSE is weighted by (1 - likelihood_weight).
    n_categorical_targets: The number of output logits for the categorical
      target.
    n_continuous_targets: The number of continuous targets.

  Returns:
    The weighted sum of the total negative log-likelihood and sum of squared
    errors.
  """
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
  """Computes a weighted average of normalized categorical likelihood and MSE.

  By convention, for mixed targets:
  - `ys` contains categorical targets at index 0 and continuous targets at
    indices [1, 1 + n_continuous_targets).
  - `yhats` contains logits for categorical targets at indices
    [0, n_categorical_targets) and predictions for continuous targets at indices
    [n_categorical_targets, n_categorical_targets + n_continuous_targets).

  Args:
    ys: Ground truth targets.
    yhats: Network predictions (logits for categorical, values for continuous).
    likelihood_weight: The weight for the normalized categorical likelihood. The
      MSE is weighted by (1 - likelihood_weight).
    n_categorical_targets: The number of output logits for the categorical
      target.
    n_continuous_targets: The number of continuous targets.

  Returns:
    The weighted average of the normalized likelihood and mean squared error.
  """

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
  """Computes the normalized likelihood (geometric mean of probabilities).

  Args:
    labels: An array of shape (n_timesteps, n_episodes, 1) containing the
      categorical labels. Negative values are treated as masked.
    output_logits: An array of shape (n_timesteps, n_episodes, n_classes)
      containing the logits output by the network.

  Returns:
    The normalized likelihood.
  """
  total_nll, n_unmasked_samples = categorical_neg_log_likelihood(
      labels, output_logits
  )
  normlik = np.exp((-1 * total_nll) / (n_unmasked_samples))
  return normlik


def avg_nll_and_log_mse(
    ys: np.ndarray,
    y_hats: np.ndarray,
    likelihood_weight: float = 0.5,
    n_categorical_targets: int = 2,
    n_continuous_targets: int = 1,
) -> float:
  """Computes a scale-normalized weighted sum of avg NLL and log(MSE).

  This loss is suitable for joint training on heterogeneous objectives
  (categorical + continuous) by putting both losses on the same log scale.
  The log(MSE) formulation helps keep the gradients from the MSE term from
  dominating the overall gradient.

  By convention, for mixed targets:
  - `ys` contains categorical targets at index 0 and continuous targets at
    indices [1, 1 + n_continuous_targets).
  - `y_hats` contains logits for categorical targets at indices
    [0, n_categorical_targets) and predictions for continuous targets at indices
    [n_categorical_targets, n_categorical_targets + n_continuous_targets).

  Args:
    ys: Ground truth targets.
    y_hats: Network predictions (logits for categorical, values for continuous).
    likelihood_weight: The weight for the average NLL. The log(MSE) is weighted
      by (1 - likelihood_weight). Default is 0.5 for equal weighting.
    n_categorical_targets: The number of output logits for the categorical
      target.
    n_continuous_targets: The number of continuous targets.

  Returns:
    The weighted sum of average NLL and log(MSE + eps).
  """
  categorical_y_hats = y_hats[:, :, 0:n_categorical_targets]
  categorical_ys = ys[:, :, 0:1]

  mask = jnp.logical_not(categorical_ys < 0)
  continuous_y_hats = y_hats[
      :, :, n_categorical_targets : n_categorical_targets + n_continuous_targets
  ]
  continuous_ys = ys[:, :, 1 : 1 + n_continuous_targets]
  continuous_ys = jnp.where(mask, continuous_ys, jnp.nan)

  nll, n_unmasked_samples = categorical_neg_log_likelihood(
      categorical_ys, categorical_y_hats
  )
  avg_nll = nll / n_unmasked_samples

  mse_val = mse(continuous_ys, continuous_y_hats)

  # This is a trick to scale the gradients from the mse as:
  # derivative(log(1+mse)) = 1 / (1 + mse) * derivative(1+mse)
  # So early in the training when mse is large, the gradient is damped, later
  # when mse is small, the gradient is almost the same as the original gradient.
  # Note that log(mse) is follows the same monotonic trend as mse, so this
  # transform will lead to a similar optimization as the original mse.
  log_mse_val = jnp.log(1.0 + mse_val)

  # Likelihood weight should ideally be set to 0.5, but can be used as a toggle
  # to train on just one objective at a time (e.g. likelihood or mse).
  return avg_nll * likelihood_weight + log_mse_val * (1 - likelihood_weight)


@jax.jit
def compute_penalty(
    targets: np.ndarray, outputs: np.ndarray
) -> tuple[float, int]:
  """Computes the total penalty from network outputs, masking invalid timesteps.

  A timestep is considered invalid and masked if all its target values are
  invalid. For categorical targets, -1 is invalid. For continuous targets, NaN
  is invalid. The penalty is assumed to be the last feature in the `outputs`
  array.

  Args:
    targets: The ground truth targets, used for masking.
    outputs: The network outputs, where the last feature is the penalty.

  Returns:
    A tuple containing:
      - penalty: The total penalty over all valid timesteps.
      - n_unmasked_samples: The number of valid timesteps.
  """
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
    validation_dataset: DatasetRNN | None,
    opt: optax.GradientTransformation = optax.adam(1e-3),
    random_key: chex.PRNGKey | None = None,
    opt_state: optax.OptState | None = None,
    params: RnnParams | None = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1,
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
    report_progress_by: Literal['print', 'log', 'wandb', 'none'] = 'print',
    wandb_run: Any | None = None,
    wandb_step_offset: int = 0,
) -> tuple[RnnParams, optax.OptState, dict[str, np.ndarray]]:
  """Trains a Haiku recurrent neural network.

  Args:
    make_network: A function that, when called, returns a Haiku RNN.
    training_dataset: A DatasetRNN, containing the data you wish to train on.
    validation_dataset: A DatasetRNN, containing the data you wish to use for
      validation.
    opt: An optax optimizer you'd like to use to train the network. Default is
      Adam with learning rate 1e-3.
    random_key: A jax random key, used for network initialization and during
      training.
    opt_state: An optax.OptState object containing an optimizer state suitable
      for the optimizer specified in opt. If None, will initialize an optimizer
      state from scratch.
    params:  An RnnParams object containing a set of parameters suitable for the
      network given by make_network. If not specified, will randomly initialize
      new parameters.
    n_steps: An integer giving the number of steps you'd like to train for.
    max_grad_norm:  Gradient clipping. Default to a very high ceiling.
    loss_param: Parameters to pass to the loss function. Can be a dictionary for
      fine-grained control over different loss components (e.g.,
      {'penalty_scale': 0.1, 'likelihood_weight': 0.8}) or a single float for
      simpler losses.
    loss: The loss function to use. Options are 'mse', 'penalized_mse',
      'categorical', 'penalized_categorical', 'hybrid', 'penalized_hybrid',
      'penalized_log_hybrid'.
    log_losses_every: How many training steps between each time we check for
      errors and log the loss.
    do_plot: Boolean that controls whether a learning curve is plotted.
    report_progress_by: Mode for reporting real-time progress. Options are
      'print' for printing to the console, 'log' for using absl logging, 'wandb'
      for both W&B logging and printing, and 'none' for no output.
    wandb_run: Optional W&B run object used for logging metrics during train.
      W&B logging occurs only if both wandb_run is provided and
      report_progress_by is 'wandb'.
    wandb_step_offset: Integer used to shift the W&B step count, if necessary
      (e.g. to include warmup steps logged beforehand in the same W&B run).

  Returns:
    params: RnnParams object containing the trained parameters. Typically this
      can be treated as a nested dictionary with a format that depends on the
      structure of the network.
    opt_state: optax.OptState object containing the optimizer state at the end
      of training.
    losses: A dictionary with keys 'training_loss' and 'validation_loss'
      mapping to numpy arrays containing a timeseries of losses on each
      dataset. Losses are recorded every log_losses_every steps.
  """
  # If loaded from json, params might be a nested dict of lists. Convert to np.
  if params is not None:
    params = to_np(params)
  sample_xs = next(training_dataset)['xs']  # Get a sample input, for shape

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
    """A hybrid loss with a penalty.

    Useful for jointly training on categorical and continuous targets. Uses a
    log of MSE loss for the continuous targets, so that the loss is similar
    units as the categorical loss.

    Args:
      params: The network parameters.
      xs: The input data.
      ys: The target data.
      random_key: A JAX random key.
      loss_param: Parameters for the loss function, potentially including
        'penalty_scale' and 'likelihood_weight'.

    Returns:
      The computed penalized hybrid loss.
    """

    penalty_scale = get_loss_param(loss_param, 'penalty_scale', 1.0)
    model_output = model.apply(params, random_key, xs)

    y_hats = model_output
    likelihood_weight = get_loss_param(loss_param, 'likelihood_weight', 0.5)
    supervised_loss = avg_nll_and_log_mse(
        ys, y_hats, likelihood_weight=likelihood_weight
    )
    penalty, n_unmasked_samples = compute_penalty(ys, y_hats)
    avg_penalty = penalty / n_unmasked_samples
    loss = supervised_loss + penalty_scale * avg_penalty
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

  data = next(training_dataset)
  xs_train, ys_train = data['xs'], data['ys']

  if validation_dataset is not None:
    data = validation_dataset.get_all()
    xs_eval, ys_eval = data['xs'], data['ys']
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
      data = next(training_dataset)
      xs_train, ys_train = data['xs'], data['ys']

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

      if report_progress_by == 'wandb' and hasattr(wandb_run, 'log'):
        try:
          wandb_run.log(
              {'train/loss': loss, 'valid/loss': l_validation},
              step=step + wandb_step_offset,
          )
        except RuntimeError as e:
          warnings.warn(f'W&B logging failed: {e}')

      if report_progress_by in ('print', 'wandb'):
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
  return params, opt_state, losses  # pytype: disable=bad-return-type


def eval_network(
    make_network: Callable[[], hk.RNNCore],
    params: RnnParams,
    xs: np.ndarray,
) -> tuple[np.ndarray, Any]:
  """Runs an RNN and returns its outputs and all internal states.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    xs: A batch of inputs `[timesteps, episodes, features]` suitable for the
      model

  Returns:
    network_outputs: Output provided by the network on each timestep. A numpy
      array of shape [timesteps, episodes, output_size]. The meaning of each
      element depends on the network configuration. For DisRNNs the final
      element will be the penalty attributable to that timestep. Previous
      elements will reflect predicted targets -- logits in the case of
      categorical targets and means in the case of continuous targets.
      states: Network states at each timestep. A numpy array of shape
        [timesteps, episodes, hidden_size].
  """
  # If loaded from json, params might be a nested dict of lists. Convert to np.
  params = to_np(params)

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

  return np.asarray(y_hats), np.asarray(states)


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
    params: RnnParams,
    state: Any,
    xs: Any,
    apply: Any = None,
    convert_params_to_np: bool = True,
) -> tuple[Any, Any, Any]:
  """Run an RNN for just a single step on a single input, with batching.

  Args:
    make_network: A Haiku function that defines a network architecture.
    params: A set of params suitable for that network.
    state: An RNN state suitable for that network.
    xs: An input for a single timestep, with shape `[n_features]` or
      `[batch_size, n_features]`.
    apply: A jitted function that applies the network to a single input. If not
      supplied, will be generated from the network architecture.
    convert_params_to_np: Whether to convert params to np. This is helpful if
      the params are loaded from json, but will cause issues if `step_network`
      is jitted.

  Returns:
    A tuple containing:
      - y_hat: The network output, with shape `[output_size]` or
        `[batch_size, output_size]`.
      - new_state: The new RNN state.
      - apply: The (possibly newly created) jitted apply function.
  """
  # If loaded from json, params might be a nested dict of lists. Convert to np.
  if convert_params_to_np:
    params = to_np(params)

  if apply is None:
    apply = get_apply(make_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hat, new_state = apply(params, key, xs, state)

  return y_hat, new_state, apply


def get_initial_state(
    make_network: Callable[[], hk.RNNCore],
    params: RnnParams | None = None,
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
  # If loaded from json, params might be a nested dict of lists. Convert to np.
  if params is not None:
    params = to_np(params)

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
    input_size: int,
    random_key: jax.Array | None = None,
) -> RnnParams:
  """Get a new set of random parameters for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    input_size: The dimensionality of the input features. This is needed because
      in some networks, the number of params depends on this and is not
      inferrable from the network architecture alone.
    random_key: a Jax random key

  Returns:
    params: A set of parameters suitable for the architecture
  """

  # If no key has been supplied, initialize a fixed key.
  if random_key is None:
    random_key = jax.random.PRNGKey(0)

  def unroll_network(xs):
    core = make_network()
    batch_size = xs.shape[1]
    state = core.initial_state(batch_size=batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

  model = hk.transform(unroll_network)

  dummy_input = jnp.zeros((1, 1, input_size))

  init = jax.jit(model.init)
  params = init(random_key, dummy_input)

  return params


def eval_feedforward_network(
    make_network: Callable[..., Any], params: RnnParams, xs: np.ndarray
) -> np.ndarray:
  """Runs a feedforward network with specified parameters and inputs.

  Args:
    make_network: A Haiku function that defines a feedforward network.
    params: A set of parameters suitable for the network.
    xs: A batch of inputs with shape `[batch_size, n_features]`.

  Returns:
    The network outputs with shape `[batch_size, n_outputs]`.
  """
  # If loaded from json, params might be a nested dict of lists. Convert to np.
  params = to_np(params)

  def forward(xs):
    net = make_network()
    y_hats = net(xs)
    return y_hats

  model = hk.transform(forward)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  apply = jax.jit(model.apply)
  y_hats = apply(params, key, xs)

  return np.array(y_hats)


def to_np(list_dict: dict[str, Any] | RnnParams):
  """Converts all numerical lists in a dict to np arrays.

  Elements that are convertible to numpy are converted. Elements that are dicts
  are recursively unpacked in the same way. Other elements are left unchanged.
  The intended use case is reconstructing a dict from json that was saved with
  NpJnpJsonEncoder and has had all its np arrays converted to lists.

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

    # Lists, tuples, and dicts: Handle recursively
    if isinstance(o, (list, tuple)):
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
