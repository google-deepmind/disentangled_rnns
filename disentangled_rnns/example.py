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

"""Example DisRNN workflow: Define a dataset, train network, inspect the fit.
"""

import copy

from absl import app
from absl import flags
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import optax


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "n_steps_per_session", 200, "Number of steps per session in the dataset."
)
flags.DEFINE_integer("n_sessions", 300, "Number of sessions in the dataset.")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
flags.DEFINE_integer("n_warmup_steps", 1000, "Number of training warmup steps.")
flags.DEFINE_integer(
    "n_training_steps", 3000, "Number of main training steps."
)


def main(_) -> None:

  ##############################
  # Define dataset and network #
  ##############################

  # Synthetic dataset from q-learning agent. Other options in get_datasets.py
  agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
  dataset = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=FLAGS.n_steps_per_session,
      n_sessions=FLAGS.n_sessions,
      batch_size=FLAGS.n_sessions,
  )

  # Second synthetic dataset for evaluation
  dataset_eval = two_armed_bandits.create_dataset(
      agent,
      environment,
      n_steps_per_session=FLAGS.n_steps_per_session,
      n_sessions=FLAGS.n_sessions,
      batch_size=FLAGS.n_sessions,
  )

  # Define the disRNN architecture
  disrnn_config = disrnn.DisRnnConfig(
      # Dataset related
      obs_size=2,
      output_size=2,
      x_names=dataset.x_names,
      y_names=dataset.y_names,
      # Network architecture
      latent_size=5,
      update_net_n_units_per_layer=8,
      update_net_n_layers=4,
      choice_net_n_units_per_layer=4,
      choice_net_n_layers=2,
      activation="leaky_relu",
      # Penalties
      noiseless_mode=False,
      latent_penalty=1e-2,
      update_net_obs_penalty=1e-3,
      update_net_latent_penalty=1e-3,
      choice_net_latent_penalty=1e-3,
      l2_scale=1e-5,
  )
  # Define a config for warmup training with no noise and no penalties
  disrnn_config_warmup = copy.deepcopy(disrnn_config)
  disrnn_config_warmup.latent_penalty = 0
  disrnn_config_warmup.choice_net_latent_penalty = 0
  disrnn_config_warmup.update_net_obs_penalty = 0
  disrnn_config_warmup.update_net_latent_penalty = 0
  disrnn_config_warmup.l2_scale = 0
  disrnn_config_warmup.noiseless_mode = True

  # Define network builder functions
  make_disrnn = lambda: disrnn.HkDisentangledRNN(disrnn_config)
  make_disrnn_warmup = lambda: disrnn.HkDisentangledRNN(disrnn_config_warmup)

  # Define an optimizer
  opt = optax.adam(learning_rate=FLAGS.learning_rate)

  #################################
  # Optimizing network parameters #
  #################################

  # Warmup training with no noise and no penalties
  params, _, _ = rnn_utils.train_network(
      make_disrnn_warmup,
      training_dataset=dataset,
      validation_dataset=dataset_eval,
      loss="penalized_categorical",
      params=None,
      opt_state=None,
      opt=opt,
      n_steps=FLAGS.n_warmup_steps,
      do_plot=True,
  )

  # Additional training using information penalty
  params, _, _ = rnn_utils.train_network(
      make_disrnn,
      training_dataset=dataset,
      validation_dataset=dataset_eval,
      loss="penalized_categorical",
      params=params,
      opt_state=None,
      opt=opt,
      n_steps=FLAGS.n_training_steps,
      do_plot=True,
  )

  ###########################
  # Inspecting a fit disRNN #
  ###########################
  # Plot bottleneck structure and update rules
  plotting.plot_bottlenecks(params, disrnn_config)
  plotting.plot_update_rules(params, disrnn_config)

  ##############################
  # Eval disRNN on unseen data #
  ##############################
  # Use the wamrup disrnn, so that there will be no noise
  xs, _ = next(dataset_eval)
  # pylint: disable-next=unused-variable
  _, network_states = rnn_utils.eval_network(make_disrnn_warmup, params, xs)


if __name__ == "__main__":
  app.run(main)
