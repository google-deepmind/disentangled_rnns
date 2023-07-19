# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config for training RNNs on rat bandit data."""

from ml_collections import config_dict


def get_config():
  """Config."""
  config = {
      "jax_random_seed": 0,
      "rat_i": 0,  # Which rat to fit (0-19)
      ## Network Params
      "learning_rate": 1e-4,  # Learning rate for the optimizer
      "n_hidden": 10,  # Number of hidden units in the LSTM
      "n_steps": 2e4,  # Number of steps to train the network for
  }

  config = config_dict.ConfigDict(config)

  return config
