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

"""Utility functions for checkpoints."""

from flax import traverse_util
import haiku as hk
import optax


def get_optimizer_with_frozen_params(
    opt: optax.GradientTransformation,
    params: hk.Params,
    trainable_param_names: list[str] | None = None,
) -> optax.GradientTransformation:
  """Modified an optax optimizer to freeze some parameters.

  This function partitions parameters into 'trainable' and 'frozen'.
  'trainable' parameters are optimized using `opt`. 'frozen' parameters
  have their gradients set to zero.

  Args:
    opt: The base Optax optimizer to use for trainable parameters.
    params: The model parameters (e.g., a Haiku parameter dictionary-like
      structure). This is used to identify parameter paths and partition them.
    trainable_param_names: An optional list of strings. If a parameter's path
      (as a tuple of strings) contains any of these strings as a substring in
      any part of the path, it's marked as trainable. If None or empty, all
      parameters will be frozen.

  Returns:
    An Optax `GradientTransformation` that applies `opt` to trainable
    parameters and zeros out gradients for frozen parameters.
  """
  partition_optimizers = {
      'trainable': opt,
      'frozen': optax.set_to_zero(),
  }

  # checks if any of the trainable param names are present in the path as a
  # substring. If so, we mark it as trainable.
  def _is_trainable(path):
    for p in path:
      for trainable_param_name in trainable_param_names:
        if trainable_param_name in p:
          return True
    return False

  # path is a tuple of strings corresponding to the path to each parameter in
  # nested param tree. Example paths:
  # ('hk_disentangled_rnn/choice_mlp', 'input_biases')
  # ('hk_disentangled_rnn', 'dopamine_mlp_multipliers')
  # ('hk_disentangled_rnn/dopamine_mlp', 'input_biases')
  #
  # For each parameter, we check if any of the trainable param names are
  # present in the path. If so, we mark that parameter as trainable.
  # rest of the parameters are frozen.
  param_partitions = traverse_util.path_aware_map(
      lambda path, _: 'trainable' if _is_trainable(path) else 'frozen',
      params,
  )

  traverse_util.path_aware_map(
      lambda path, v: print(path),
      params,
  )
  # Choose the optimizer based on the partition.
  opt = optax.multi_transform(partition_optimizers, param_partitions)
  return opt
