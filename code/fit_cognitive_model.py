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

"""Fits and cross-validates the cognitive model using dataset for a particular rat."""

import datetime
import json
import os
import sys

from absl import flags
from absl import logging
import jax
from ml_collections.config_flags import config_flags
import numpy as np

from disentangled_rnns.code.library import cog_model
from disentangled_rnns.code.library import get_rat_datasets
from disentangled_rnns.code.library import rnn_utils


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "experimental/deepmind/kevinjmiller/predictive_cognitive/config_cog.py",
)


def main():

  # Define the datasets
  logging.info("Setting up datasets")
  rat_i = FLAGS.config.rat_i
  logging.info(f"Rat number is {rat_i}".format(rat_i=rat_i))

  dataset_list = get_rat_datasets.get_rat_datasets(FLAGS.config.dataset_path)
  dataset_rat = dataset_list[rat_i]

  xs, ys = next(dataset_rat)
  dataset_odd = rnn_utils.DatasetRNN(xs[:, 0::2, :], ys[:, 0::2, :])
  dataset_even = rnn_utils.DatasetRNN(xs[:, 1::2, :], ys[:, 1::2, :])

  # Define the model we'd like to train
  model = cog_model.HkCognitiveModel
  key = jax.random.PRNGKey(FLAGS.config.jax_random_seed)
  # Train on everything to get param fits
  logging.info("Fitting Cognitive model to whole dataset")
  params_hk, fit_ll, _ = cog_model.fit_model(
      model,
      dataset_rat,
      random_key=key,
  )
  params = params_hk["hk_cognitive_model"]
  logging.info("Done fitting. Params:")
  logging.info(params)

  # Train on even/odd split to get xval likelihood
  logging.info("Cross-validating cognitive model")
  _, _, xval_loss_even_to_odd = cog_model.fit_model(
      model,
      dataset_even,
      dataset_odd,
      random_key=key,
  )
  logging.info("Done fit-even/test-odd. Log likelihood was:")
  logging.info(xval_loss_even_to_odd)
  _, _, xval_loss_odd_to_even = cog_model.fit_model(
      model,
      dataset_odd,
      dataset_even,
      random_key=key,
  )
  logging.info("Done fit-odd/test-even. Log likelihood was:")
  logging.info(xval_loss_odd_to_even)

  # Compute normalized likelihoods: exp(log-likelihood / ntrials)
  _, ys = next(dataset_odd)
  n_trials_odd = np.sum(ys != -1)
  _, ys = next(dataset_even)
  n_trials_even = np.sum(ys != -1)

  xval_normlik_even_to_odd = np.exp(-1 * xval_loss_even_to_odd / n_trials_odd)
  xval_normlik_odd_to_even = np.exp(-1 * xval_loss_odd_to_even / n_trials_even)

  xval_normlik = np.exp(
      (-1 * xval_loss_even_to_odd + -1 * xval_loss_odd_to_even)
      / (n_trials_even + n_trials_odd)
  )

  # Get a unique identifier for this particular job
  run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S%f")

  logging.info("Creating save folder")

  save_path = FLAGS.config.path
  if not os.path.isdir(save_path):
    os.mkdir(save_path)

  logging.info("Saving results to JSON")

  save_dict = {
      "rat_i": int(rat_i),
      "fit_ll": float(fit_ll),
      "alpha_gf_logit": float(params["alpha_gf_logit"]),
      "alpha_h_logit": float(params["alpha_h_logit"]),
      "alpha_rs_logit": float(params["alpha_rs_logit"]),
      "beta_bias": float(params["beta_bias"]),
      "beta_gf": float(params["beta_gf"]),
      "beta_h": float(params["beta_h"]),
      "beta_rs": float(params["beta_rs"]),
      "xval_loss_even_to_odd": float(xval_loss_even_to_odd),
      "xval_loss_odd_to_even": float(xval_loss_odd_to_even),
      "xval_normlik_even_to_odd": float(xval_normlik_even_to_odd),
      "xval_normlik_odd_to_even": float(xval_normlik_odd_to_even),
      "xval_normlik": float(xval_normlik),
      "Run_ID": run_id,
  }

  with open(
      os.path.join(
          save_path,
          "fit_rat"
          + str(FLAGS.config.rat_i)
          + "_seed"
          + str(FLAGS.config.jax_random_seed)
          + ".json",
      ),
      "w",
  ) as f:
    f.write(json.dumps(save_dict, cls=rnn_utils.NpEncoder))

  logging.info("Done!")


if __name__ == "__main__":
  sys.exit(main())
