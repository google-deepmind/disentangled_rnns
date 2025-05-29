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

from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import multisubject_disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils

from google3.testing.pybase import googletest


class MultisubjectDisrnnTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.multisubject_disrnn_config = (
        multisubject_disrnn.MultisubjectDisRnnConfig(
            latent_size=5,
            output_size=2,  # Two possible choices
            obs_size=2,  # Choice and reward. Subj ID does not count
            update_net_n_units_per_layer=4,
            update_net_n_layers=2,
            choice_net_n_units_per_layer=2,
            choice_net_n_layers=2,
            max_n_subjects=3,  # Corresponds to the 3 datasets created below
            subject_embedding_size=3,
            x_names=['Choice', 'Reward'],
            y_names=['Next Choice'],
        )
    )
    multisubject_dataset_list = []
    for alpha in [0.1, 0.2, 0.5]:  # Create 3 datasets simulating 3 subjects
      dataset = get_datasets.get_q_learning_dataset(
          n_sessions=11, n_trials=7, alpha=alpha
      )
      multisubject_dataset_list.append(dataset)
    self.multisubject_dataset = (
        get_datasets.dataset_list_to_multisubject(
            multisubject_dataset_list
        )
    )
    self.multisubject_disrnn_params, _, _ = rnn_utils.train_network(
        make_network=lambda: multisubject_disrnn.MultisubjectDisRnn(
            self.multisubject_disrnn_config
        ),
        training_dataset=self.multisubject_dataset,
        validation_dataset=None,
        n_steps=0,
    )

  def test_multisubject_disrnn_config(self):
    """Check that the multisubject disRNN config is as expected."""
    multisubject_disrnn_config = self.multisubject_disrnn_config
    self.assertEqual(
        multisubject_disrnn_config.x_names, ['Subject ID', 'Choice', 'Reward']
    )
    self.assertEqual(multisubject_disrnn_config.y_names, ['Next Choice'])

    new_config = dataclasses.replace(
        multisubject_disrnn_config,
        obs_size=2,
    )
    self.assertEqual(new_config.x_names, ['Subject ID', 'Choice', 'Reward'])
    self.assertEqual(new_config.obs_size, 2)

  def test_multisubject_disrnn_params(self):
    """Check that multisubject disRNN params are as expected."""
    multisubject_disrnn_config = self.multisubject_disrnn_config
    multisubject_disrnn_params = self.multisubject_disrnn_params
    self.assertIn(
        'multisubject_dis_rnn/~update_latents/update_net',
        multisubject_disrnn_params,
    )
    self.assertIn(
        'multisubject_dis_rnn/~predict_targets/choice_net',
        multisubject_disrnn_params,
    )
    self.assertIn(
        'multisubject_dis_rnn/subject_embedding_weights',
        multisubject_disrnn_params,
    )

    params = multisubject_disrnn_params['multisubject_dis_rnn']
    update_net_params = multisubject_disrnn_params[
        'multisubject_dis_rnn/~update_latents/update_net'
    ]
    choice_net_params = multisubject_disrnn_params[
        'multisubject_dis_rnn/~predict_targets/choice_net'
    ]
    subject_embedding_params = multisubject_disrnn_params[
        'multisubject_dis_rnn/subject_embedding_weights'
    ]

    # Check inherited params
    self.assertIn('update_net_sigma_params', params)
    self.assertIn('update_net_multipliers', params)
    self.assertIn('latent_sigma_params', params)
    self.assertIn('choice_net_sigma_params', params)
    self.assertIn('choice_net_multipliers', params)
    self.assertIn('latent_inits', params)
    # Check subject embedding specific params
    self.assertIn('subject_embedding_sigma_params', params)

    # Check shapes based on config
    latent_size = multisubject_disrnn_config.latent_size
    obs_size = multisubject_disrnn_config.obs_size
    subj_emb_size = multisubject_disrnn_config.subject_embedding_size
    max_n_subjects = multisubject_disrnn_config.max_n_subjects
    update_net_input_size = latent_size + obs_size + subj_emb_size
    choice_net_input_size = latent_size + subj_emb_size
    update_net_units = multisubject_disrnn_config.update_net_n_units_per_layer
    choice_net_units = multisubject_disrnn_config.choice_net_n_units_per_layer

    self.assertEqual(
        params['update_net_sigma_params'].shape,
        (update_net_input_size, latent_size),
    )
    self.assertEqual(params['latent_sigma_params'].shape, (latent_size,))
    self.assertEqual(
        params['choice_net_sigma_params'].shape, (choice_net_input_size,)
    )
    self.assertEqual(params['latent_inits'].shape, (latent_size,))
    self.assertEqual(
        params['subject_embedding_sigma_params'].shape, (subj_emb_size,)
    )
    self.assertEqual(
        subject_embedding_params['w'].shape, (max_n_subjects, subj_emb_size)
    )
    self.assertEqual(
        update_net_params['input_weights'].shape,
        (update_net_input_size, update_net_units),
    )
    self.assertEqual(
        choice_net_params['input_weights'].shape,
        (choice_net_input_size, choice_net_units),
    )

  def test_multisubject_disrnn_output_shape(self):
    """Check that the output shapes of the multisubject disRNN are correct."""
    xs, _ = self.multisubject_dataset.get_all()
    n_batch, n_trials = xs.shape[:2]  # n_batch = n_subjects * n_sessions

    network_outputs, network_states = rnn_utils.eval_network(
        lambda: multisubject_disrnn.MultisubjectDisRnn(
            self.multisubject_disrnn_config
        ),
        self.multisubject_disrnn_params,
        xs,
    )
    # Output has shape (batch_size, output_size + 1).
    # The first output_size elements are the predicted targets, and the last
    # element is the penalty
    self.assertEqual(network_outputs.shape, (n_batch, n_trials, 3))
    self.assertEqual(
        network_states.shape,
        (n_batch, n_trials, self.multisubject_disrnn_config.latent_size),
    )

  def test_multisubject_disrnn_trainable(self):
    """Smoke test to check that the multisubject disRNN can be trained."""
    n_steps = 10
    _, _, _ = rnn_utils.train_network(
        make_network=lambda: multisubject_disrnn.MultisubjectDisRnn(
            self.multisubject_disrnn_config
        ),
        training_dataset=self.multisubject_dataset,
        validation_dataset=None,
        params=self.multisubject_disrnn_params,
        n_steps=n_steps,
    )

  def test_multisubject_disrnn_plotting(self):
    multisubject_disrnn_config = self.multisubject_disrnn_config
    multisubject_disrnn_params = self.multisubject_disrnn_params
    plotting.plot_bottlenecks(
        multisubject_disrnn_params, multisubject_disrnn_config
    )
    plotting.plot_update_rules(
        multisubject_disrnn_params, multisubject_disrnn_config, subj_ind=0
    )
    plotting.plot_choice_rule(
        multisubject_disrnn_params, multisubject_disrnn_config
    )


if __name__ == '__main__':
  googletest.main()
