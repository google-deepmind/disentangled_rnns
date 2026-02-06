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

from absl.testing import absltest
from disentangled_rnns.library import neuro_disrnn
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits_w_dopamine


class NeuroDisrnnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.neuro_disrnn_config = neuro_disrnn.DisRnnWNeuralActivityConfig(
        obs_size=2,
        # Choice net output size, doesn't include neural_activity
        output_size=2,
        latent_size=5,
        update_net_n_units_per_layer=4,
        update_net_n_layers=3,
        choice_net_n_units_per_layer=2,
        choice_net_n_layers=2,
        neural_activity_net_n_units_per_layer=2,
        neural_activity_net_n_layers=2,
    )
    self.q_dataset = (
        two_armed_bandits_w_dopamine.get_q_learning_with_dopamine_dataset(
            n_sessions=11, n_trials=7
        )
    )
    self.neuro_disrnn_params, _, _ = rnn_utils.train_network(
        make_network=lambda: neuro_disrnn.HkNeuroDisentangledRNN(
            self.neuro_disrnn_config
        ),
        training_dataset=self.q_dataset,
        validation_dataset=None,
        n_steps=0,
    )

  def test_neuro_disrnn_params(self):
    """Check that disRNN neural_activity params are as expected."""
    neuro_disrnn_config = self.neuro_disrnn_config
    neuro_disrnn_params = self.neuro_disrnn_params

    self.assertIn('hk_neuro_disentangled_rnn', neuro_disrnn_params)
    self.assertIn(
        'hk_neuro_disentangled_rnn/~update_latents/update_net',
        neuro_disrnn_params,
    )
    self.assertIn(
        'hk_neuro_disentangled_rnn/neural_activity_net', neuro_disrnn_params
    )

    params = neuro_disrnn_params['hk_neuro_disentangled_rnn']
    neural_activity_net_params = neuro_disrnn_params[
        'hk_neuro_disentangled_rnn/neural_activity_net'
    ]

    # Check neural_activity specific params
    self.assertIn('neural_activity_net_multipliers', params)
    self.assertIn('neural_activity_net_sigma_params', params)

    # Check shapes based on config
    latent_size = neuro_disrnn_config.latent_size
    neural_activity_net_units = (
        neuro_disrnn_config.neural_activity_net_n_units_per_layer
    )

    self.assertEqual(
        params['neural_activity_net_sigma_params'].shape, (latent_size + 2,)
    )  # +2 for choice and reward
    self.assertEqual(
        neural_activity_net_params['input_weights'].shape,
        (latent_size + 2, neural_activity_net_units),
    )

  def test_neuro_disrnn_output_shape(self):
    """Check that the output shapes of the disRNN with neural_activity are correct."""
    xs = self.q_dataset.get_all()['xs']
    n_sessions, n_trials = xs.shape[:2]

    network_outputs, network_states = rnn_utils.eval_network(
        lambda: neuro_disrnn.HkNeuroDisentangledRNN(self.neuro_disrnn_config),
        self.neuro_disrnn_params,
        xs,
    )
    # Output has shape (batch_size, output_size + 2).
    # The first output_size elements are the predicted targets (next choice),
    # the next element is the neural_activity prediction, and the last is
    # the penalty.
    self.assertEqual(network_outputs.shape, (n_sessions, n_trials, 4))
    self.assertEqual(
        network_states.shape,
        (n_sessions, n_trials, self.neuro_disrnn_config.latent_size),
    )

  def test_neuro_disrnn_trainable(self):
    """Smoke test to check that the disRNN with neural_activity can be trained."""
    n_steps = 10
    _, _, _ = rnn_utils.train_network(
        make_network=lambda: neuro_disrnn.HkNeuroDisentangledRNN(
            self.neuro_disrnn_config
        ),
        training_dataset=self.q_dataset,
        validation_dataset=None,
        loss='penalized_hybrid',
        params=self.neuro_disrnn_params,
        n_steps=n_steps,
    )

  def test_neuro_disrnn_plotting(self):
    neuro_disrnn_config = self.neuro_disrnn_config
    neuro_disrnn_params = self.neuro_disrnn_params
    neuro_disrnn.plot_bottlenecks(neuro_disrnn_params, neuro_disrnn_config)
    neuro_disrnn.plot_update_rules(neuro_disrnn_params, neuro_disrnn_config)
    neuro_disrnn.plot_choice_rule(neuro_disrnn_params, neuro_disrnn_config)


if __name__ == '__main__':
  absltest.main()
