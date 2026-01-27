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
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils


class DisrnnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.disrnn_config = disrnn.DisRnnConfig(
        latent_size=5,
        obs_size=2,
        output_size=2,
        update_net_n_units_per_layer=4,
        update_net_n_layers=2,
        choice_net_n_units_per_layer=2,
        choice_net_n_layers=2,
    )
    self.q_dataset = get_datasets.get_q_learning_dataset(
        n_sessions=11, n_trials=7
    )
    self.disrnn_params, _, _ = rnn_utils.train_network(
        make_network=lambda: disrnn.HkDisentangledRNN(self.disrnn_config),
        training_dataset=self.q_dataset,
        validation_dataset=None,
        n_steps=0,
    )

  def test_disrnn_params(self):
    """Check that disRNN params are as expected."""
    disrnn_config = self.disrnn_config
    disrnn_params = self.disrnn_params

    self.assertIn('hk_disentangled_rnn', disrnn_params)
    self.assertIn(
        'hk_disentangled_rnn/~update_latents/update_net',
        disrnn_params,
    )
    self.assertIn(
        'hk_disentangled_rnn/~predict_targets/choice_net', disrnn_params
    )

    # Parameters are stored under the module name given in hk.transform,
    # which is 'hk_disentangled_rnn' if not specified otherwise.
    params = disrnn_params['hk_disentangled_rnn']
    update_net_params = disrnn_params[
        'hk_disentangled_rnn/~update_latents/update_net'
    ]
    choice_net_params = disrnn_params[
        'hk_disentangled_rnn/~predict_targets/choice_net'
    ]

    self.assertIn('update_net_obs_sigma_params', params)
    self.assertIn('update_net_obs_multipliers', params)
    self.assertIn('update_net_latent_sigma_params', params)
    self.assertIn('update_net_latent_multipliers', params)

    self.assertIn('latent_sigma_params', params)
    self.assertIn('choice_net_sigma_params', params)
    self.assertIn('choice_net_multipliers', params)
    self.assertIn('latent_inits', params)

    # Check shapes based on config
    latent_size = disrnn_config.latent_size
    obs_size = disrnn_config.obs_size
    net_input_size = latent_size + obs_size

    self.assertEqual(
        params['update_net_obs_sigma_params'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        params['update_net_obs_multipliers'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        params['update_net_latent_sigma_params'].shape,
        (latent_size, latent_size),
    )
    self.assertEqual(
        params['update_net_latent_multipliers'].shape,
        (latent_size, latent_size),
    )

    self.assertEqual(params['latent_sigma_params'].shape, (latent_size,))
    self.assertEqual(params['choice_net_sigma_params'].shape, (latent_size,))
    self.assertEqual(params['choice_net_multipliers'].shape, (latent_size,))
    self.assertEqual(params['latent_inits'].shape, (latent_size,))
    self.assertEqual(
        update_net_params['input_weights'].shape,
        (net_input_size, disrnn_config.update_net_n_units_per_layer),
    )
    self.assertEqual(
        choice_net_params['input_weights'].shape,
        (latent_size, disrnn_config.choice_net_n_units_per_layer),
    )

  def test_disrnn_plotting(self):
    plotting.plot_bottlenecks(self.disrnn_params, self.disrnn_config)
    plotting.plot_update_rules(self.disrnn_params, self.disrnn_config)
    plotting.plot_choice_rule(self.disrnn_params, self.disrnn_config)

  def test_disrnn_output_shape(self):
    xs = self.q_dataset.get_all()['xs']
    n_sessions, n_trials = xs.shape[:2]

    network_outputs, network_states = rnn_utils.eval_network(
        lambda: disrnn.HkDisentangledRNN(self.disrnn_config),
        self.disrnn_params,
        xs,
    )
    # Output has shape (batch_size, output_size + 1).
    # The first output_size elements are the predicted targets, andt the last
    # element is the penalty
    self.assertEqual(network_outputs.shape, (n_sessions, n_trials, 3))
    self.assertEqual(network_states.shape, (n_sessions,
                                            n_trials,
                                            self.disrnn_config.latent_size))

  def test_disrnn_trainable(self):
    """Smoke test to check that the disRNN can be trained."""
    n_steps = 10
    _, _, _ = rnn_utils.train_network(
        make_network=lambda: disrnn.HkDisentangledRNN(self.disrnn_config),
        training_dataset=self.q_dataset,
        validation_dataset=None,
        params=self.disrnn_params,
        n_steps=n_steps,
    )

  def test_get_auxiliary_metrics(self):
    """Smoke test for get_auxiliary_metrics."""
    metrics = disrnn.get_auxiliary_metrics(self.disrnn_params)
    self.assertIsInstance(metrics, dict)
    self.assertIn('total_sigma', metrics)
    self.assertIn('latent_bottlenecks_open', metrics)
    self.assertIn('choice_bottlenecks_open', metrics)
    self.assertIn('update_bottlenecks_open', metrics)
    self.assertGreaterEqual(metrics['total_sigma'], 0)


if __name__ == '__main__':
  absltest.main()
