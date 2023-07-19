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

"""Cross-validates an LSTM against the dataset for a particular rat."""

import sys

from absl import flags
from absl import logging
import haiku as hk
import jax
from ml_collections.config_flags import config_flags
import optax

from disentangled_rnns.code.library import get_rat_datasets
from disentangled_rnns.code.library import rnn_utils


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "code/library/config_lstm.py",
)


def main():

  # Define the datasets
  dataset_list = get_rat_datasets.get_rat_datasets(FLAGS.config.dataset_path)
  dataset_rat = dataset_list[FLAGS.config.rat_i]

  xs, ys = next(dataset_rat)
  dataset_odd = rnn_utils.DatasetRNN(xs[:, 0::2, :], ys[:, 0::2, :])
  dataset_even = rnn_utils.DatasetRNN(xs[:, 1::2, :], ys[:, 1::2, :])

  # Define the architecture of the networks we'd like to train
  def make_network():
    model = hk.DeepRNN(
        [hk.LSTM(FLAGS.config.n_hidden), hk.Linear(output_size=2)]
    )
    return model

  # Define the optimizer we'd like to train with
  opt = optax.rmsprop(learning_rate=FLAGS.config.learning_rate)

  # Init the networks
  logging.info("Initializing Network One")
  params_1, opt_state_1, _ = rnn_utils.train_network(
      make_network,
      dataset_even,
      dataset_odd,
      random_key=jax.random.PRNGKey(FLAGS.config.jax_random_seed),
      loss="categorical",
      opt=opt,
      n_steps=0,
  )

  params_2, opt_state_2, _ = rnn_utils.train_network(
      make_network,
      dataset_odd,
      dataset_even,
      random_key=jax.random.PRNGKey(FLAGS.config.jax_random_seed),
      loss="categorical",
      opt=opt,
      n_steps=0,
  )

  n_steps_complete_1 = 0
  n_steps_complete_2 = 0

  # Train network 1
  while n_steps_complete_1 < FLAGS.config.n_steps:
    logging.info("Training Network One")
    params_1, opt_state_1, _ = rnn_utils.train_network(
        make_network,
        dataset_even,
        dataset_odd,
        opt=opt,
        loss="softmax",
        params=params_1,
        opt_state=opt_state_1,
        n_steps=100,
    )
    n_steps_complete_1 += 100

  # Train network 2
  while n_steps_complete_2 < FLAGS.config.n_steps:
    logging.info("Training Network One")
    params_2, opt_state_2, _ = rnn_utils.train_network(
        make_network,
        dataset_odd,
        dataset_even,
        opt=opt,
        loss="softmax",
        params=params_2,
        opt_state=opt_state_2,
        n_steps=100,
    )
    n_steps_complete_2 += 100

  logging.info("Done!")


if __name__ == "__main__":
  sys.exit(main())
