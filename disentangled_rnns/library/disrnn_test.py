# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for disRNN bottleneck structure, training, and plotting."""

import copy
import dataclasses

from absl.testing import absltest
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


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

    self.assertIn('update_net_obs_log_sigmas', params)
    self.assertIn('update_net_obs_multipliers', params)
    self.assertIn('update_net_latent_log_sigmas', params)
    self.assertIn('update_net_latent_multipliers', params)

    self.assertIn('latent_log_sigmas', params)
    self.assertIn('choice_net_log_sigmas', params)
    self.assertIn('choice_net_multipliers', params)
    self.assertIn('latent_inits', params)

    # Check shapes based on config
    latent_size = disrnn_config.latent_size
    obs_size = disrnn_config.obs_size
    net_input_size = latent_size + obs_size

    self.assertEqual(
        params['update_net_obs_log_sigmas'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        params['update_net_obs_multipliers'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        params['update_net_latent_log_sigmas'].shape,
        (latent_size, latent_size),
    )
    self.assertEqual(
        params['update_net_latent_multipliers'].shape,
        (latent_size, latent_size),
    )

    self.assertEqual(params['latent_log_sigmas'].shape, (latent_size,))
    self.assertEqual(params['choice_net_log_sigmas'].shape, (latent_size,))
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

  def test_compute_choice_rule_applies_multipliers(self):
    """Verify compute_choice_rule scales inputs by choice_net_multipliers."""
    params_zero_mult = copy.deepcopy(self.disrnn_params)
    params_zero_mult['hk_disentangled_rnn']['choice_net_multipliers'] = (
        params_zero_mult['hk_disentangled_rnn']['choice_net_multipliers'] * 0.0
    )
    rule_default = plotting.compute_choice_rule(
        self.disrnn_params, self.disrnn_config
    )
    rule_zero = plotting.compute_choice_rule(
        params_zero_mult, self.disrnn_config
    )
    # With zero multipliers, choice_net receives constant zero inputs across the
    # grid, whereas default multipliers=1 vary across the grid.
    self.assertGreater(float(rule_default['yhats'].std()), 0.0)
    self.assertAlmostEqual(float(rule_zero['yhats'].std()), 0.0, places=6)

  def test_compute_update_rules_all_trials_title(self):
    """Verify observation-insensitive latents use the 'All Trials' key."""
    params_closed_obs = copy.deepcopy(self.disrnn_params)
    obs_log_sigmas = params_closed_obs['hk_disentangled_rnn'][
        'update_net_obs_log_sigmas'
    ]
    params_closed_obs['hk_disentangled_rnn']['update_net_obs_log_sigmas'] = (
        disrnn.inverse_reparameterize_sigma(np.full_like(obs_log_sigmas, 1.0))
    )
    update_dict = plotting.compute_update_rules(
        params_closed_obs, self.disrnn_config
    )
    for latent_rules in update_dict.values():
      self.assertIn('All Trials', latent_rules)
      self.assertNotIn('A', latent_rules)

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
    self.assertEqual(
        network_states.shape,
        (n_sessions, n_trials, self.disrnn_config.latent_size),
    )

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

  def test_config_validation(self):
    """Check that DisRnnConfig rejects invalid attributes and values."""
    config = disrnn.DisRnnConfig()
    with self.assertRaises(AttributeError):
      setattr(config, 'subject_penalty', 1e-3)
    with self.assertRaises(ValueError):
      config.latent_penalty = -0.1
    with self.assertRaises(ValueError):
      setattr(config, 'latent_penalty', None)
    with self.assertRaises(ValueError):
      config.update_net_latent_self_penalty = -0.1
    config.update_net_latent_self_penalty = None
    self.assertIsNone(config.update_net_latent_self_penalty)
    with self.assertRaises(ValueError):
      config.max_latent_value = 0.0
    with self.assertRaises(ValueError):
      disrnn.DisRnnConfig(max_latent_value=-1.0)

  def test_separate_self_and_between_latent_penalties(self):
    """Verify separate self (diagonal) and between-latent (off-diag) penalties."""
    xs = self.q_dataset.get_all()['xs']
    base_cfg = dataclasses.replace(
        self.disrnn_config,
        latent_penalty=0.0,
        update_net_obs_penalty=0.0,
        choice_net_latent_penalty=0.0,
        l2_scale=0.0,
        update_net_latent_penalty=1.0,
        update_net_latent_self_penalty=None,
    )
    explicit_equal_cfg = dataclasses.replace(
        base_cfg,
        update_net_latent_penalty=1.0,
        update_net_latent_self_penalty=1.0,
    )
    self_only_cfg = dataclasses.replace(
        base_cfg,
        update_net_latent_penalty=0.0,
        update_net_latent_self_penalty=1.0,
    )
    between_only_cfg = dataclasses.replace(
        base_cfg,
        update_net_latent_penalty=1.0,
        update_net_latent_self_penalty=0.0,
    )
    weighted_cfg = dataclasses.replace(
        base_cfg,
        update_net_latent_penalty=3.0,
        update_net_latent_self_penalty=0.5,
    )

    key = jax.random.PRNGKey(0)
    xs_step = xs[0, :2]

    def _get_penalty(cfg):
      def step(x):
        core = disrnn.HkDisentangledRNN(cfg)
        outputs, _ = core(x, core.initial_state(x.shape[0]))
        return jnp.sum(outputs[:, -1])

      return float(
          hk.transform(step).apply(self.disrnn_params, key, xs_step)
      )

    pen_default = _get_penalty(base_cfg)
    pen_equal = _get_penalty(explicit_equal_cfg)
    pen_self = _get_penalty(self_only_cfg)
    pen_between = _get_penalty(between_only_cfg)
    pen_weighted = _get_penalty(weighted_cfg)

    self.assertGreater(pen_self, 0.0)
    self.assertGreater(pen_between, 0.0)
    np.testing.assert_allclose(pen_default, pen_equal, rtol=1e-5)
    np.testing.assert_allclose(pen_equal, pen_self + pen_between, rtol=1e-5)
    np.testing.assert_allclose(
        pen_weighted, 0.5 * pen_self + 3.0 * pen_between, rtol=1e-5
    )

  def test_zero_weight_underflowing_sigma_is_finite(self):
    """Zero-weighted bottlenecks must not produce NaN if sigma^2 underflows."""

    def loss_fn(sigmas, weights):
      _, penalty = disrnn.information_bottleneck(
          inputs=jnp.ones((2, 3, 3)),
          sigmas=sigmas,
          weights=weights,
      )
      return jnp.sum(penalty)

    # Diagonal sigmas square to 0.0 in float32; diagonal weight is 0.0.
    sigmas = jnp.where(jnp.eye(3, dtype=bool), 1e-25, 0.5)
    weights = 1.0 - jnp.eye(3)

    transformed = hk.transform(loss_fn)
    rng = jax.random.PRNGKey(0)
    params = transformed.init(rng, sigmas, weights)
    val, grad = jax.value_and_grad(
        lambda s: transformed.apply(params, rng, s, weights)
    )(sigmas)
    self.assertTrue(np.isfinite(float(val)))
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

  def test_inverse_reparameterize_sigma_round_trip(self):
    """inverse_reparameterize_sigma undoes reparameterize_sigma."""
    sigmas = np.array([1e-5, 1e-3, 0.01, 0.5, 1.0, 10.0])
    np.testing.assert_allclose(
        disrnn.reparameterize_sigma(
            disrnn.inverse_reparameterize_sigma(sigmas)
        ),
        sigmas,
        rtol=1e-6,
    )

  def test_inverse_reparameterize_sigma_nonpositive_raises(self):
    with self.assertRaises(ValueError):
      disrnn.inverse_reparameterize_sigma(np.array([0.5, 0.0]))

  def test_choice_net_zero_layers(self):
    """Test DisRnnConfig with choice_net_n_layers=0."""
    config = disrnn.DisRnnConfig(
        latent_size=5,
        obs_size=2,
        output_size=2,
        update_net_n_units_per_layer=4,
        update_net_n_layers=2,
        choice_net_n_units_per_layer=2,
        choice_net_n_layers=0,
    )
    params, _, losses = rnn_utils.train_network(
        make_network=lambda: disrnn.HkDisentangledRNN(config),
        training_dataset=self.q_dataset,
        validation_dataset=None,
        n_steps=5,
    )
    self.assertFalse(rnn_utils.nan_in_dict(params))
    self.assertEqual(losses['training_loss'].shape[0], 2)


if __name__ == '__main__':
  absltest.main()
