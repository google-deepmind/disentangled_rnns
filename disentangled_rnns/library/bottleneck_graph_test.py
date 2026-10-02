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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from disentangled_rnns.library import bottleneck_graph
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import multisubject_disrnn
from disentangled_rnns.library import neuro_disrnn
from disentangled_rnns.library import rnn_utils
import networkx as nx
import numpy as np


class BottleneckGraphTest(parameterized.TestCase):

  def test_disrnns_isomorphic_latent_swap(self):
    """Test isomorphism with swapped latent identities."""

    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=3,
        x_names=['obs1', 'obs2'],
    )

    # obs1 -> latent1; latent1 -> latent2; latent2 -> output
    params1 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9, 0.9],  # obs1 to latent0
                    [0.9, 0.9, 0.9],  # obs2 to nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.01, 0.9],  # latent0 to latent1
                        [0.9, 0.9, 0.9],  # latent to nowhere
                        [0.9, 0.9, 0.9],  # latent3 to nowhere
                    ])
                )
            ),
            # latent1 to output
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.9, 0.01, 0.9])
            ),
        }
    }

    # obs1 -> latent3; latent3 -> latent0; latent0 -> Output
    # Isomorphic to the above, with latent1 <-> latent3 and latent2 <-> latent1
    params2 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9, 0.01])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.9, 0.9, 0.01],  # obs1 to latent2
                    [0.9, 0.9, 0.9],  # obs2 to nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9, 0.9],  # latent1 to nowhere
                        [0.9, 0.9, 0.9],  # latent2 to nowhere
                        [0.01, 0.9, 0.9],  # latent3 to latent0
                    ])
                )
            ),
            # latent0 to output
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9, 0.9])
            ),
        }
    }

    self.assertTrue(
        bottleneck_graph.disrnns_isomorphic(config, params1, config, params2)
    )

  def test_disrnns_not_isomorphic_obs_mismatch(self):
    """Test non-isomorphism when observation connections differ."""

    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=2,
        x_names=['obs1', 'obs2'],
    )

    # params1: obs1 -> latent1 -> Output
    params1 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],  # obs1 to latent 0
                    [0.9, 0.9],  # obs2 to nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9],  # latent 0 to nowhere
                        [0.9, 0.9],  # latent 1 to nowhere
                    ])
                )
            ),
            # latent 0 to output
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }

    # params2: obs2 -> latent1 -> Output
    params2 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.9, 0.9],  # obs1 to nowhere
                    [0.01, 0.9],  # obs2 to latent 0
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9],  # latent 0 to nowhere
                        [0.9, 0.9],  # latent 1 to nowhere
                    ])
                )
            ),
            # latent 0 to output
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }
    self.assertFalse(
        bottleneck_graph.disrnns_isomorphic(config, params1, config, params2)
    )

  def test_get_networkx_graph_single_output(self):
    """Test that get_bottleneck_graph creates edges to a single output node."""

    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=2,
        x_names=['obs1', 'obs2'],
        y_names=['output1', 'output2'],
    )

    # latent 0 connected to the output via choice bottleneck.
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],  # obs1 -> latent 0
                    [0.9, 0.9],  # obs2 -> nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9],
                        [0.9, 0.9],
                    ])
                )
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }

    g = bottleneck_graph.get_bottleneck_graph(config, params)

    # latent 0 should connect to the single output node.
    self.assertIn(('Latent 1', 'Output'), g.edges())
    # latent 0 is open so it must have a self-connection even though
    # update_net_latent_sigma_params[0, 0] is closed (0.9).
    self.assertIn(('Latent 1', 'Latent 1'), g.edges())
    self.assertNotIn(('Latent 2', 'Latent 2'), g.edges())
    # The output node is named 'Output' regardless of y_names.
    self.assertIn('Output', g.nodes())
    self.assertNotIn('output1', g.nodes())
    self.assertNotIn('output2', g.nodes())

  def test_disrnns_isomorphic_single_output(self):
    """Test isomorphism with single output node."""

    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=2,
        x_names=['obs1', 'obs2'],
        y_names=['output1', 'output2'],
    )

    # Both params have latent0 -> output via choice bottleneck.
    params1 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],
                    [0.9, 0.9],
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9],
                        [0.9, 0.9],
                    ])
                )
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }

    # Isomorphic: latent1 -> output (latent swap).
    params2 = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.9, 0.01])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.9, 0.01],
                    [0.9, 0.9],
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(
                    np.array([
                        [0.9, 0.9],
                        [0.9, 0.9],
                    ])
                )
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.9, 0.01])
            ),
        }
    }

    self.assertTrue(
        bottleneck_graph.disrnns_isomorphic(config, params1, config, params2)
    )

  def test_get_networkx_graph_sort_latents(self):
    """Test sorted numbering matches plotting.plot_bottlenecks ranks."""
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=3,
        x_names=['obs0', 'obs1'],
    )
    # Params define a graph with the following order and structure:
    # - obs1 -> latent1
    # - obs0 -> latent0 -> output
    # - latent2 is closed
    params = {
        'hk_disentangled_rnn': {
            # These will determine ordering: 1, 0, 2
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.3, 0.01, 0.9])
            ),
            # These determine connections. 0.01 means open, 1.0 means closed.
            # obs0 -> latent0, obs1 -> latent1
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 1.0, 1.0],  # obs0 -> latent 0
                    [1.0, 0.01, 1.0],  # obs1 -> latent 1
                ])
            ),
            # No self-connections or connections between latents.
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((3, 3), 0.9))
            ),
            # latent0 -> output
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9, 0.9])
            ),
        }
    }
    unsorted = bottleneck_graph.get_bottleneck_graph(
        config, params, sort_latents=False
    )
    self.assertTrue(unsorted.has_edge('obs0', 'Latent 1'))
    self.assertTrue(unsorted.has_edge('obs1', 'Latent 2'))
    self.assertTrue(unsorted.has_edge('Latent 1', 'Output'))
    self.assertFalse(unsorted.has_edge('Latent 2', 'Output'))

    g = bottleneck_graph.get_bottleneck_graph(config, params)
    self.assertTrue(g.has_edge('obs1', 'Latent 1'))
    self.assertTrue(g.has_edge('obs0', 'Latent 2'))
    self.assertTrue(g.has_edge('Latent 2', 'Output'))
    self.assertFalse(g.has_edge('Latent 1', 'Output'))
    self.assertEqual(g.nodes['Latent 1']['index'], 0)
    self.assertEqual(g.nodes['Latent 2']['index'], 1)

  def test_get_networkx_graph_node_kinds(self):
    """Test that nodes carry role and index attributes."""
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=2,
        x_names=['obs1', 'obs2'],
    )
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.9, 0.01])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.9, 0.9],
                    [0.9, 0.01],
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.9))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.9, 0.01])
            ),
        }
    }
    g = bottleneck_graph.get_bottleneck_graph(
        config, params, sort_latents=False
    )
    self.assertEqual(g.nodes['obs2'], dict(name='obs2', kind='input', index=1))
    self.assertEqual(
        g.nodes['Latent 2'], dict(name='Latent 2', kind='latent', index=1)
    )
    self.assertEqual(
        g.nodes['Output'], dict(name='Output', kind='output', index=0)
    )

  def test_get_bottleneck_graph_prunes_closed_latents(self):
    """Test that closed latents are dropped along with all their edges."""
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=2,
        x_names=['obs0', 'obs1'],
    )
    # latent 0 is open, latent 1 is closed. Every other bottleneck is open.
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.full((2, 2), 0.01)
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.01))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.01])
            ),
        }
    }
    g = bottleneck_graph.get_bottleneck_graph(
        config, params, sort_latents=False
    )
    self.assertNotIn('Latent 2', g)
    self.assertCountEqual(
        g.edges(),
        [
            ('obs0', 'Latent 1'),
            ('obs1', 'Latent 1'),
            ('Latent 1', 'Latent 1'),
            ('Latent 1', 'Output'),
        ],
    )

  def test_get_bottleneck_graph_real_params(self):
    """Test a graph built from freshly initialized DisRNN params.

    This catches drift between the param names in `disrnn` and the ones read
    here. At initialization every bottleneck is open, so the graph is complete.
    """
    n_latents = 3
    x_names = ['prev choice', 'prev reward']
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=2,
        latent_size=n_latents,
        update_net_n_units_per_layer=4,
        update_net_n_layers=2,
        choice_net_n_units_per_layer=2,
        choice_net_n_layers=2,
        x_names=x_names,
    )
    dataset = get_datasets.get_q_learning_dataset(n_sessions=2, n_trials=5)
    params, _, _ = rnn_utils.train_network(
        make_network=lambda: disrnn.HkDisentangledRNN(config),
        training_dataset=dataset,
        validation_dataset=None,
        n_steps=0,
    )
    g = bottleneck_graph.get_bottleneck_graph(config, params)
    latent_names = [f'Latent {i + 1}' for i in range(n_latents)]
    expected_edges = (
        list(itertools.product(x_names, latent_names))
        + list(itertools.product(latent_names, latent_names))
        + [(latent, 'Output') for latent in latent_names]
    )
    self.assertCountEqual(g.edges(), expected_edges)

  @parameterized.named_parameters(
      (
          'multisubject',
          multisubject_disrnn.MultisubjectDisRnnConfig(
              obs_size=2, x_names=['obs0', 'obs1']
          ),
      ),
      (
          'neural_activity',
          neuro_disrnn.DisRnnWNeuralActivityConfig(
              obs_size=2, x_names=['obs0', 'obs1']
          ),
      ),
  )
  def test_get_bottleneck_graph_rejects_config_subclasses(self, config):
    with self.assertRaisesRegex(NotImplementedError, 'single-subject'):
      bottleneck_graph.get_bottleneck_graph(config, params={})

  @parameterized.named_parameters(
      ('unset', None),
      ('too_few', ['obs0']),
  )
  def test_get_bottleneck_graph_rejects_bad_x_names(self, x_names):
    config = disrnn.DisRnnConfig(obs_size=2, latent_size=2)
    config.x_names = x_names
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.full(2, 0.01)
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.full((2, 2), 0.01)
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.01))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.full(2, 0.01)
            ),
        }
    }
    with self.assertRaisesRegex(ValueError, 'one name per input'):
      bottleneck_graph.get_bottleneck_graph(config, params)

  def test_graphs_isomorphic_latents_match_regardless_of_name(self):
    """Test that any latent matches any other latent."""
    g1 = nx.DiGraph()
    g1.add_node('obs1', name='obs1', kind='input')
    g1.add_node('Latent 1', name='Latent 1', kind='latent')
    g1.add_node('Output', name='Output', kind='output')
    g1.add_edge('obs1', 'Latent 1')
    g1.add_edge('Latent 1', 'Output')

    g2 = nx.DiGraph()
    g2.add_node('obs1', name='obs1', kind='input')
    g2.add_node('Latent 3', name='Latent 3', kind='latent')
    g2.add_node('Output', name='Output', kind='output')
    g2.add_edge('obs1', 'Latent 3')
    g2.add_edge('Latent 3', 'Output')

    self.assertTrue(bottleneck_graph.graphs_isomorphic(g1, g2))

  def test_graphs_isomorphic_input_named_latent_is_not_a_latent(self):
    """Test that an input with 'latent' in its name is not a latent."""
    # input 'latent cue' -> latent -> output
    g1 = nx.DiGraph()
    g1.add_node('latent cue', name='latent cue', kind='input')
    g1.add_node('Latent 1', name='Latent 1', kind='latent')
    g1.add_node('Output', name='Output', kind='output')
    g1.add_edge('latent cue', 'Latent 1')
    g1.add_edge('Latent 1', 'Output')

    # latent -> latent -> output: same shape, but the first node is a latent
    g2 = nx.DiGraph()
    g2.add_node('Latent 1', name='Latent 1', kind='latent')
    g2.add_node('Latent 2', name='Latent 2', kind='latent')
    g2.add_node('Output', name='Output', kind='output')
    g2.add_edge('Latent 1', 'Latent 2')
    g2.add_edge('Latent 2', 'Output')

    self.assertFalse(bottleneck_graph.graphs_isomorphic(g1, g2))

  @parameterized.named_parameters(
      dict(
          testcase_name='input_matches_output',
          x_names=['Output', 'reward'],
      ),
      dict(
          testcase_name='input_matches_latent',
          x_names=['Latent 1', 'reward'],
      ),
      dict(
          testcase_name='input_matches_input',
          x_names=['reward', 'reward'],
      ),
  )
  def test_get_bottleneck_graph_duplicate_names_raise(self, x_names):
    """Test that clashing node names raise instead of merging nodes."""
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=2,
        x_names=x_names,
    )
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.full((2, 2), 0.01)
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.9))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }
    with self.assertRaisesRegex(ValueError, 'must be unique'):
      bottleneck_graph.get_bottleneck_graph(config, params)

  def test_disrnn_isomorphic_to_graph_input_names(self):
    """Test the three outcomes: isomorphic, not isomorphic, impossible."""
    config = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=2,
        x_names=['obs1', 'obs2'],
    )
    # obs1 -> latent 0 -> Output; obs2 is closed.
    params = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],  # obs1 -> latent 0
                    [0.9, 0.9],  # obs2 -> nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.9))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }

    def build_reference_graph(input_name):
      # input_name -> latent -> Output, with a latent self-connection.
      g = nx.DiGraph()
      g.add_node(input_name, name=input_name, kind='input')
      g.add_node('latent', name='latent', kind='latent')
      g.add_node('Output', name='Output', kind='output')
      g.add_edge(input_name, 'latent')
      g.add_edge('latent', 'latent')
      g.add_edge('latent', 'Output')
      return g

    with self.subTest('isomorphic'):
      self.assertTrue(
          bottleneck_graph.disrnn_isomorphic_to_graph(
              config, params, build_reference_graph('obs1')
          )
      )
    with self.subTest('reference_input_closed_in_disrnn'):
      self.assertFalse(
          bottleneck_graph.disrnn_isomorphic_to_graph(
              config, params, build_reference_graph('obs2')
          )
      )
    with self.subTest('reference_input_not_in_x_names'):
      with self.assertRaisesRegex(ValueError, r"\['cue'\]"):
        bottleneck_graph.disrnn_isomorphic_to_graph(
            config, params, build_reference_graph('cue')
        )

  def test_disrnns_isomorphic_input_names(self):
    """Test that an open input missing from the other config raises."""
    config_with_cue = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=2,
        x_names=['obs1', 'cue'],
    )
    config_without_cue = disrnn.DisRnnConfig(
        obs_size=2,
        output_size=1,
        latent_size=2,
        x_names=['obs1', 'obs2'],
    )
    # obs1 -> latent 0 -> Output, and the second input -> latent 0.
    params_second_input_open = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],  # obs1 -> latent 0
                    [0.01, 0.9],  # second input -> latent 0
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.9))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }
    # obs1 -> latent 0 -> Output; the second input is closed.
    params_second_input_closed = {
        'hk_disentangled_rnn': {
            'latent_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
            'update_net_obs_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([
                    [0.01, 0.9],  # obs1 -> latent 0
                    [0.9, 0.9],  # second input -> nowhere
                ])
            ),
            'update_net_latent_sigma_params': (
                disrnn.inverse_reparameterize_sigma(np.full((2, 2), 0.9))
            ),
            'choice_net_sigma_params': disrnn.inverse_reparameterize_sigma(
                np.array([0.01, 0.9])
            ),
        }
    }

    with self.subTest('open_input_not_in_other_x_names'):
      with self.assertRaisesRegex(ValueError, r"\['cue'\]"):
        bottleneck_graph.disrnns_isomorphic(
            config_with_cue,
            params_second_input_open,
            config_without_cue,
            params_second_input_closed,
        )
    with self.subTest('open_input_not_in_other_x_names_reversed'):
      with self.assertRaisesRegex(ValueError, r"\['cue'\]"):
        bottleneck_graph.disrnns_isomorphic(
            config_without_cue,
            params_second_input_closed,
            config_with_cue,
            params_second_input_open,
        )
    with self.subTest('closed_input_not_in_other_x_names'):
      self.assertTrue(
          bottleneck_graph.disrnns_isomorphic(
              config_with_cue,
              params_second_input_closed,
              config_without_cue,
              params_second_input_closed,
          )
      )
    with self.subTest('open_input_closed_in_other_disrnn'):
      self.assertFalse(
          bottleneck_graph.disrnns_isomorphic(
              config_with_cue,
              params_second_input_open,
              config_with_cue,
              params_second_input_closed,
          )
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='get_bottleneck_graph',
          fn=bottleneck_graph.get_bottleneck_graph,
      ),
      dict(
          testcase_name='plot_bottleneck_graph',
          fn=bottleneck_graph.plot_bottleneck_graph,
      ),
      dict(
          testcase_name='disrnn_isomorphic_to_graph',
          fn=lambda params, config: bottleneck_graph.disrnn_isomorphic_to_graph(
              params, config, nx.DiGraph()
          ),
      ),
  )
  def test_swapped_config_and_params_raise(self, fn):
    """Test that passing (params, disrnn_config) gives an informative error."""
    config = disrnn.DisRnnConfig(obs_size=2, x_names=['obs0', 'obs1'])
    params = {'hk_disentangled_rnn': {}}
    with self.assertRaisesRegex(TypeError, 'swapped order'):
      fn(params, config)


if __name__ == '__main__':
  absltest.main()
