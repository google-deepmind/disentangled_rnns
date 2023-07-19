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

"""Run compartmentalized RNN on a full dataset. Save and plot some output."""

import datetime
import json
import os
import sys

from absl import flags
from absl import logging
import jax
from ml_collections.config_flags import config_flags
import numpy as np
import optax

from disentangled_rnns.code.library import disrnn
from disentangled_rnns.code.library import get_rat_datasets
from disentangled_rnns.code.library import rnn_utils


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "/code/library/config_disrnn.py",
)


def main():

  # Define the datasets
  dataset_list = get_rat_datasets.get_rat_datasets(FLAGS.config.dataset_path)
  dataset_rat = dataset_list[FLAGS.config.rat_i]

  # Define the architecture of the networks we'd like to train
  def make_network():
    model = disrnn.HkCompartmentalizedRNN(
        update_mlp_shape=FLAGS.config.update_mlp_shape,
        choice_mlp_shape=FLAGS. config.choice_mlp_shape,
        latent_size=FLAGS.config.n_latents,
        obs_size=2,
        target_size=2,
    )
    return model

  def make_network_eval():
    model = disrnn.HkCompartmentalizedRNN(
        update_mlp_shape=FLAGS.config.update_mlp_shape,
        choice_mlp_shape=FLAGS.config.choice_mlp_shape,
        latent_size=FLAGS.config.n_latents,
        obs_size=2,
        target_size=2,
        eval_mode=1,
    )
    return model

  # Define the optimizer we'd like to train with
  opt = optax.adam(learning_rate=FLAGS.config.learning_rate)

  # Init the network
  key = jax.random.PRNGKey(FLAGS.config.jax_random_seed)
  key, subkey = jax.random.split(key)
  logging.info("Initializing Network")
  params, opt_state_init, _ = rnn_utils.train_network(
      make_network,
      dataset_rat,
      dataset_rat,
      random_key=subkey,
      loss="penalized_categorical",
      opt=opt,
      n_steps=0,
      max_grad_norm=FLAGS.config.max_grad_norm,
  )
  opt_state = opt_state_init

  n_steps_per_save = 1000
  n_steps_complete = 0

  ##################
  # TRAIN NETWORKS #
  ##################
  # Train network 1
  logging.info("Training Network")

  # First train with weak penalty (warmup). Do not update opt_state
  if n_steps_complete < FLAGS.config.n_steps_warmup:
    key, subkey = jax.random.split(key)
    params, _, _ = (
        rnn_utils.train_network(
            make_network,
            dataset_rat,
            dataset_rat,
            opt=opt,
            loss="penalized_categorical",
            penalty_scale=FLAGS.config.warmup_penalty,
            params=params,
            opt_state=opt_state,
            n_steps=FLAGS.config.n_steps_warmup,
            random_key=subkey,
            max_grad_norm=FLAGS.config.max_grad_norm,
        )
    )
    n_steps_complete = FLAGS.config.n_steps_warmup

  while n_steps_complete < FLAGS.config.n_steps:
    key, subkey = jax.random.split(key)
    params, opt_state, _ = (
        rnn_utils.train_network(
            make_network,
            dataset_rat,
            dataset_rat,
            opt=opt,
            loss="penalized_softmax",
            penalty_scale=FLAGS.config.penalty_scale,
            params=params,
            opt_state=opt_state,
            n_steps=n_steps_per_save,
            random_key=subkey,
            max_grad_norm=FLAGS.config.max_grad_norm,
        )
    )
    n_steps_complete += n_steps_per_save

  ################
  # EVAL NETWORK #
  ################
  def softmax(labels: np.ndarray, output_logits: np.ndarray) -> float:
    # Mask any errors for which label is negative
    mask = np.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    masked_log_liks = np.multiply(log_liks, mask)
    loss = -np.nansum(masked_log_liks)
    return loss

  logging.info("Running network in eval mode")
  # Run the network in eval mode
  xs, ys = next(dataset_rat)
  network_outputs, network_states = rnn_utils.eval_network(
      make_network_eval, params, xs
  )
  logits = network_outputs[:, :, :-1]
  ll = softmax(ys, logits)
  n_trials = np.sum(ys != -1)

  # Put it together to create overall cross-validated normalized likelihood
  normlik = np.exp((ll) / (n_trials))

  # Get a unique identifier for this particular job
  run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S%f")

  # Save final params
  logging.info("Saving params")

  save_path = os.path.join(FLAGS.config.path, run_id)
  if not os.path.isdir(save_path):
    os.mkdir(save_path)

  with open(os.path.join(save_path, run_id + "-params.json"), "w") as f:
    f.write(json.dumps(params, cls=rnn_utils.NpEncoder))

  # Save eval results
  logging.info("Saving eval results")
  save_dict = {
      "network_outputs": network_outputs,
      "network_states": network_states,
      "xs": xs,
      "ys": ys,
      "normLik": normlik,
  }
  with open(os.path.join(save_path, run_id + "-eval.json"), "w") as f:
    f.write(json.dumps(save_dict, cls=rnn_utils.NpEncoder))

  logging.info("Done!")


if __name__ == "__main__":
  sys.exit(main())
