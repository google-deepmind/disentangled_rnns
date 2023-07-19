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

"""Cross-validates the compartmentalized RNN against the dataset for a particular rat."""

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
    "code/library/config_compartmentalized_rnn.py",
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

  # Init the networks
  key = jax.random.PRNGKey(FLAGS.config.jax_random_seed)
  key, subkey = jax.random.split(key)
  logging.info("Initializing Network One")
  params_even, opt_state_even, _ = rnn_utils.train_network(
      make_network,
      dataset_even,
      dataset_odd,
      random_key=subkey,
      loss="penalized_categorical",
      opt=opt,
      n_steps=0,
      max_grad_norm=FLAGS.config.max_grad_norm,
  )

  key, subkey = jax.random.split(key)
  params_odd, opt_state_odd, _ = rnn_utils.train_network(
      make_network,
      dataset_odd,
      dataset_even,
      random_key=subkey,
      loss="penalized_categorical",
      opt=opt,
      n_steps=0,
      max_grad_norm=FLAGS.config.max_grad_norm,
  )

  n_steps_per_save = 1000
  n_steps_complete_even = 0
  n_steps_complete_odd = 0

  ##################
  # TRAIN NETWORKS #
  ##################
  # Train network 1
  logging.info("Training Network One")
  # First train without any penalty
  if n_steps_complete_even < FLAGS.config.n_steps_warmup:
    key, subkey = jax.random.split(key)
    params_even, opt_state_even, _ = (
        rnn_utils.train_network(
            make_network,
            dataset_even,
            dataset_odd,
            opt=opt,
            loss="penalized_categorical",
            penalty_scale=FLAGS.config.warmup_penalty,
            params=params_even,
            opt_state=opt_state_even,
            n_steps=FLAGS.config.n_steps_warmup,
            random_key=subkey,
            max_grad_norm=FLAGS.config.max_grad_norm,
        )
    )
    n_steps_complete_even = FLAGS.config.n_steps_warmup

  while n_steps_complete_even < FLAGS.config.n_steps:
    key, subkey = jax.random.split(key)
    params_even, opt_state_even, _ = (
        rnn_utils.train_network(
            make_network,
            dataset_even,
            dataset_odd,
            opt=opt,
            loss="penalized_categorical",
            penalty_scale=FLAGS.config.penalty_scale,
            params=params_even,
            opt_state=opt_state_even,
            n_steps=n_steps_per_save,
            random_key=subkey,
            max_grad_norm=FLAGS.config.max_grad_norm,
        )
    )
    n_steps_complete_even += n_steps_per_save

  # Train network 2
  logging.info("Training Network One")
  # First train without any penalty
  if n_steps_complete_odd < FLAGS.config.n_steps_warmup:
    key, subkey = jax.random.split(key)
    params_odd, opt_state_odd, _ = rnn_utils.train_network(
        make_network,
        dataset_odd,
        dataset_even,
        opt=opt,
        loss="penalized_categorical",
        penalty_scale=FLAGS.config.warmup_penalty,
        params=params_odd,
        opt_state=opt_state_odd,
        n_steps=FLAGS.config.n_steps_warmup,
        random_key=subkey,
        max_grad_norm=FLAGS.config.max_grad_norm,
    )
    n_steps_complete_odd = FLAGS.config.n_steps_warmup

  while n_steps_complete_odd < FLAGS.config.n_steps:
    key, subkey = jax.random.split(key)
    params_odd, opt_state_odd, _ = rnn_utils.train_network(
        make_network,
        dataset_odd,
        dataset_even,
        opt=opt,
        loss="penalized_categorical",
        penalty_scale=FLAGS.config.penalty_scale,
        params=params_odd,
        opt_state=opt_state_odd,
        n_steps=n_steps_per_save,
        random_key=subkey,
        max_grad_norm=FLAGS.config.max_grad_norm,
    )
    n_steps_complete_odd += n_steps_per_save

  # Get a unique identifier for this particular job
  run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S%f")

  # Create a save folder
  logging.info("Creating save folder")

  save_path = os.path.join(FLAGS.config.path, run_id)
  if not os.path.isdir(save_path):
    os.mkdir(save_path)

  # Save fit params
  save_dict = {
      "params_fit_even": params_even,
      "params_fit_odd": params_odd,
  }
  with open(os.path.join(save_path, run_id + "-params.json"), "w") as f:
    f.write(json.dumps(save_dict, cls=rnn_utils.NpEncoder))

  ###########################
  # COMPUTE XVAL LIKELIHOOD #
  ###########################
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

  # Unpack datasets
  xs_odd, ys_odd = next(dataset_odd)
  xs_even, ys_even = next(dataset_even)
  # Run network in eval mode on unseen dataset
  even_to_odd_outputs, even_to_odd_states = (
      rnn_utils.eval_network(make_network_eval, params_even, xs_odd)
  )
  odd_to_even_outputs, odd_to_even_states = (
      rnn_utils.eval_network(make_network_eval, params_odd, xs_even)
  )
  # Compute likelihoods: log-likelihood and normalized likelihood
  logits = even_to_odd_outputs[:, :, :-1]
  even_to_odd_ll = softmax(ys_odd, logits)
  n_trials_odd = np.sum(ys_odd != -1)
  even_to_odd_normlik = np.exp(even_to_odd_ll / n_trials_odd)

  logits = odd_to_even_outputs[:, :, :-1]
  odd_to_even_ll = softmax(ys_even, logits)
  n_trials_even = np.sum(ys_even != -1)
  odd_to_even_normlik = np.exp(odd_to_even_ll / n_trials_even)

  # Put it together to create overall cross-validated normalized likelihood
  normlik = np.exp(
      (even_to_odd_ll + odd_to_even_ll)
      / (n_trials_even + n_trials_odd)
  )

  # Save eval results
  logging.info("Saving noiseless eval results")
  save_dict = {
      "even_to_odd_outputs": even_to_odd_outputs,
      "even_to_odd_states": even_to_odd_states,
      "odd_to_even_outputs": odd_to_even_outputs,
      "odd_to_even_states": odd_to_even_states,
      "even_to_odd_normlik": even_to_odd_normlik,
      "odd_to_even_normlik": odd_to_even_normlik,
      "xs_even": xs_even,
      "xs_odd": xs_odd,
      "ys_even": ys_even,
      "ys_odd": ys_odd,
      "normlik": normlik,
  }
  with open(
      os.path.join(save_path, run_id + "-noiseless-eval.json"), "w"
  ) as f:
    f.write(json.dumps(save_dict, cls=rnn_utils.NpEncoder))

  ############################
  # COMPUTE BOTTLENECK USAGE #
  ############################
  # Run on unseen dataset in noisy mode
  even_to_odd_outputs, even_to_odd_states = rnn_utils.eval_network(
      make_network, params_even, xs_odd
  )
  odd_to_even_outputs, odd_to_even_states = rnn_utils.eval_network(
      make_network, params_odd, xs_even
  )

  even_to_odd_penalty = np.sum(even_to_odd_outputs[:, :, -1])
  odd_to_even_penalty = np.sum(odd_to_even_outputs[:, :, -1])

  logging.info("Saving noisy eval results")
  save_dict = {
      "even_to_odd_outputs": even_to_odd_outputs,
      "even_to_odd_states": even_to_odd_states,
      "odd_to_even_outputs": odd_to_even_outputs,
      "odd_to_even_states": odd_to_even_states,
      "even_to_odd_penalty": even_to_odd_penalty,
      "odd_to_even_penalty": odd_to_even_penalty,
      "xs_even": xs_even,
      "xs_odd": xs_odd,
      "ys_even": ys_even,
      "ys_odd": ys_odd,
      "normlik": normlik,
  }
  with open(
      os.path.join(save_path, run_id + "-noisy-eval.json"), "w"
  ) as f:
    f.write(json.dumps(save_dict, cls=rnn_utils.NpEncoder))

  logging.info("Done!")


if __name__ == "__main__":
  sys.exit(main())
