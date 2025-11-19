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

import dataclasses

from absl.testing import absltest
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import multisubject_neuro_disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits_w_dopamine


class MultisubjectNeuroDisrnnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.multisubject_neuro_disrnn_config = (
        multisubject_neuro_disrnn.MultisubjectNeuroDisRnnConfig(
            latent_size=5,
            output_size=2,  # Two possible choices
            obs_size=2,  # Choice and reward. Subj ID does not count
            update_net_n_units_per_layer=4,
            update_net_n_layers=2,
            choice_net_n_units_per_layer=2,
            choice_net_n_layers=2,
            neural_activity_net_n_units_per_layer=2,
            neural_activity_net_n_layers=2,
            neural_activity_net_latent_penalty=1.0,
            max_n_subjects=3,  # Corresponds to the 3 datasets created below
            subject_embedding_size=3,
            x_names=['Choice', 'Reward'],
            y_names=['Next Choice', 'Neural Activity'],
        )
    )
    multisubject_dataset_list = []
    for alpha in [0.1, 0.2, 0.5]:  # Create 3 datasets simulating 3 subjects
      dataset = (
          two_armed_bandits_w_dopamine.get_q_learning_with_dopamine_dataset(
              n_sessions=11, n_trials=7, alpha=alpha
          )
      )
      multisubject_dataset_list.append(dataset)
    self.multisubject_dataset = get_datasets.dataset_list_to_multisubject(
        multisubject_dataset_list
    )
    self.multisubject_neuro_disrnn_params, _, _ = rnn_utils.train_network(
        make_network=lambda: multisubject_neuro_disrnn.MultisubjectNeuroDisRnn(
            self.multisubject_neuro_disrnn_config
        ),
        training_dataset=self.multisubject_dataset,
        validation_dataset=None,
        n_steps=0,
    )

  def test_multisubject_neuro_disrnn_params(self):
    """Check that multisubject neuro disRNN params are as expected."""
    config = self.multisubject_neuro_disrnn_config
    params = self.multisubject_neuro_disrnn_params
    self.assertIn('multisubject_neuro_dis_rnn', params)
    self.assertIn(
        'multisubject_neuro_dis_rnn/~update_latents/update_net', params
    )
    self.assertIn(
        'multisubject_neuro_dis_rnn/~predict_targets/choice_net', params
    )
    self.assertIn('multisubject_neuro_dis_rnn/neural_activity_net', params)
    self.assertIn(
        'multisubject_neuro_dis_rnn/subject_embedding_weights', params
    )

    rnn_params = params['multisubject_neuro_dis_rnn']
    update_net_params = params[
        'multisubject_neuro_dis_rnn/~update_latents/update_net'
    ]
    choice_net_params = params[
        'multisubject_neuro_dis_rnn/~predict_targets/choice_net'
    ]
    neural_activity_net_params = params[
        'multisubject_neuro_dis_rnn/neural_activity_net'
    ]
    subject_embedding_linear_params = params[
        'multisubject_neuro_dis_rnn/subject_embedding_weights'
    ]

    # Check parameters directly under the main module
    self.assertIn('latent_sigma_params', rnn_params)
    self.assertIn('latent_inits', rnn_params)

    # Update network bottleneck parameters
    self.assertIn('update_net_subj_sigma_params', rnn_params)
    self.assertIn('update_net_subj_multipliers', rnn_params)
    self.assertIn('update_net_obs_sigma_params', rnn_params)
    self.assertIn('update_net_obs_multipliers', rnn_params)
    self.assertIn('update_net_latent_sigma_params', rnn_params)
    self.assertIn('update_net_latent_multipliers', rnn_params)

    # Choice network bottleneck parameters
    self.assertIn('choice_net_subj_sigma_params', rnn_params)
    self.assertIn('choice_net_subj_multipliers', rnn_params)
    self.assertIn('choice_net_latent_sigma_params', rnn_params)
    self.assertIn('choice_net_latent_multipliers', rnn_params)

    # Neural activity network bottleneck parameters
    self.assertIn('neural_activity_net_sigma_params', rnn_params)
    self.assertIn('neural_activity_net_multipliers', rnn_params)

    # Check shapes based on config
    latent_size = config.latent_size
    obs_size = config.obs_size
    subj_emb_size = config.subject_embedding_size
    max_n_subjects = config.max_n_subjects
    update_net_input_size = latent_size + obs_size + subj_emb_size
    choice_net_input_size = latent_size + subj_emb_size
    neural_activity_net_input_size = latent_size + obs_size
    update_net_units = config.update_net_n_units_per_layer
    choice_net_units = config.choice_net_n_units_per_layer
    neural_activity_net_units = config.neural_activity_net_n_units_per_layer

    self.assertEqual(rnn_params['latent_sigma_params'].shape, (latent_size,))
    self.assertEqual(rnn_params['latent_inits'].shape, (latent_size,))

    # Update network bottleneck shapes
    self.assertEqual(
        rnn_params['update_net_subj_sigma_params'].shape,
        (subj_emb_size, latent_size),
    )
    self.assertEqual(
        rnn_params['update_net_subj_multipliers'].shape,
        (subj_emb_size, latent_size),
    )
    self.assertEqual(
        rnn_params['update_net_obs_sigma_params'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        rnn_params['update_net_obs_multipliers'].shape, (obs_size, latent_size)
    )
    self.assertEqual(
        rnn_params['update_net_latent_sigma_params'].shape,
        (latent_size, latent_size),
    )
    self.assertEqual(
        rnn_params['update_net_latent_multipliers'].shape,
        (latent_size, latent_size),
    )

    # Choice network bottleneck shapes
    self.assertEqual(
        rnn_params['choice_net_subj_sigma_params'].shape, (subj_emb_size,)
    )
    self.assertEqual(
        rnn_params['choice_net_subj_multipliers'].shape, (subj_emb_size,)
    )
    self.assertEqual(
        rnn_params['choice_net_latent_sigma_params'].shape, (latent_size,)
    )
    self.assertEqual(
        rnn_params['choice_net_latent_multipliers'].shape, (latent_size,)
    )

    # Neural activity network bottleneck shapes
    self.assertEqual(
        rnn_params['neural_activity_net_sigma_params'].shape,
        (latent_size + obs_size,),
    )
    self.assertEqual(
        rnn_params['neural_activity_net_multipliers'].shape,
        (latent_size + obs_size,),
    )

    # Subject embedding linear layer shapes
    self.assertEqual(
        subject_embedding_linear_params['w'].shape,
        (max_n_subjects, subj_emb_size),
    )
    self.assertEqual(
        subject_embedding_linear_params['b'].shape, (subj_emb_size,)
    )

    # MLP input weight shapes
    self.assertEqual(
        update_net_params['input_weights'].shape,
        (update_net_input_size, update_net_units),
    )
    self.assertEqual(
        choice_net_params['input_weights'].shape,
        (choice_net_input_size, choice_net_units),
    )
    self.assertEqual(
        neural_activity_net_params['input_weights'].shape,
        (neural_activity_net_input_size, neural_activity_net_units),
    )

  def test_multisubject_neuro_disrnn_output_shape(self):
    """Check that the output shapes of the multisubject neuro disRNN are correct."""
    xs, _ = self.multisubject_dataset.get_all()
    n_batch, n_trials = xs.shape[:2]  # n_batch = n_subjects * n_sessions

    network_outputs, network_states = rnn_utils.eval_network(
        lambda: multisubject_neuro_disrnn.MultisubjectNeuroDisRnn(
            self.multisubject_neuro_disrnn_config
        ),
        self.multisubject_neuro_disrnn_params,
        xs,
    )
    # Output has shape (batch_size, output_size + 2).
    # The first output_size elements are the predicted targets (next choice),
    # the next element is the neural_activity prediction, and the last is
    # the penalty.
    self.assertEqual(network_outputs.shape, (n_batch, n_trials, 4))
    self.assertEqual(
        network_states.shape,
        (n_batch, n_trials, self.multisubject_neuro_disrnn_config.latent_size),
    )

  def test_multisubject_neuro_disrnn_trainable(self):
    """Smoke test to check that the multisubject neuro disRNN can be trained."""
    n_steps = 10
    _, _, _ = rnn_utils.train_network(
        make_network=lambda: multisubject_neuro_disrnn.MultisubjectNeuroDisRnn(
            self.multisubject_neuro_disrnn_config
        ),
        training_dataset=self.multisubject_dataset,
        validation_dataset=None,
        loss='penalized_hybrid',
        params=self.multisubject_neuro_disrnn_params,
        n_steps=n_steps,
    )

  def test_multisubject_neuro_disrnn_plotting(self):
    """Smoke test for plotting functions."""
    config = self.multisubject_neuro_disrnn_config
    params = self.multisubject_neuro_disrnn_params
    plotting.plot_bottlenecks(params, config)
    plotting.plot_update_rules(params, config, subj_ind=0)
    plotting.plot_choice_rule(params, config)

  def test_get_auxiliary_metrics(self):
    """Smoke test for get_auxiliary_metrics."""
    metrics = multisubject_neuro_disrnn.get_auxiliary_metrics(
        self.multisubject_neuro_disrnn_params
    )
    self.assertIsInstance(metrics, dict)
    self.assertIn('total_sigma', metrics)
    self.assertIn('latent_bottlenecks_open', metrics)
    self.assertIn('choice_bottlenecks_open', metrics)
    self.assertIn('update_bottlenecks_open', metrics)
    self.assertIn('neural_activity_bottlenecks_open', metrics)
    self.assertIn('neural_activity_bottlenecks_closed', metrics)
    self.assertGreaterEqual(metrics['total_sigma'], 0)


if __name__ == '__main__':
  absltest.main()
