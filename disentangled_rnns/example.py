# Copyright 2024 DeepMind Technologies Limited.
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

from absl import app
from absl import flags
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import optax


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "n_steps_per_session", 200, "Number of steps per session in the dataset."
)
flags.DEFINE_integer("n_sessions", 300, "Number of sessions in the dataset.")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
flags.DEFINE_float("penalty_scale", 1e-3, "Information penalty scale.")
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
  update_mlp_shape = (3, 5, 5)
  choice_mlp_shape = (2, 2)
  latent_size = 5

  def make_network():
    return disrnn.HkDisRNN(
        update_mlp_shape=update_mlp_shape,
        choice_mlp_shape=choice_mlp_shape,
        latent_size=latent_size,
        obs_size=2,
        target_size=2,
    )

  opt = optax.adam(learning_rate=FLAGS.learning_rate)

  #################################
  # Optimizing network parameters #
  #################################

  # Warmup training with no information penalty
  params, _, _ = rnn_utils.train_network(
      make_network,
      training_dataset=dataset,
      validation_dataset=dataset_eval,
      loss="penalized_categorical",
      params=None,
      opt_state=None,
      opt=opt,
      loss_param=0,
      n_steps=FLAGS.n_warmup_steps,
      do_plot=True,
  )

  # Additional training using information penalty
  params, _, _ = rnn_utils.train_network(
      make_network,
      training_dataset=dataset,
      validation_dataset=dataset_eval,
      loss="penalized_categorical",
      params=params,
      opt_state=None,
      opt=opt,
      loss_param=FLAGS.penalty_scale,
      n_steps=FLAGS.n_training_steps,
      do_plot=True,
  )

  ###########################
  # Inspecting a fit disRNN #
  ###########################

  # Eval mode runs the network with no noise
  def make_network_eval():
    return disrnn.HkDisRNN(
        update_mlp_shape=update_mlp_shape,
        choice_mlp_shape=choice_mlp_shape,
        latent_size=latent_size,
        obs_size=2,
        target_size=2,
        eval_mode=True,
    )

  disrnn.plot_bottlenecks(params, make_network_eval)
  disrnn.plot_update_rules(params, make_network_eval)

  ##############################
  # Eval disRNN on unseen data #
  ##############################

  xs, _ = next(dataset_eval)
  # pylint: disable-next=unused-variable
  _, network_states = rnn_utils.eval_network(make_network_eval, params, xs)


if __name__ == "__main__":
  app.run(main)
