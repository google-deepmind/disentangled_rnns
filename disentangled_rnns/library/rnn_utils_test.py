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

import json
from absl.testing import absltest
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


n_steps_per_session = 10
n_sessions = 3
batch_size = 1024
n_hidden = 20


def make_network():
  model = hk.DeepRNN([hk.GRU(n_hidden), hk.Linear(output_size=2)])
  return model


class TestRNNUtils(absltest.TestCase):
  """Tests for rnn_utils.
  """

  def setUp(self):
    super().setUp()
    self.make_network = make_network
    self.random_key = jax.random.PRNGKey(0)

    agent = two_armed_bandits.AgentQ(alpha=0, beta=0)
    environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
    self.dataset = two_armed_bandits.create_dataset(
        agent,
        environment,
        n_steps_per_session=n_steps_per_session,
        n_sessions=n_sessions,
        )

    # "Train" for zero steps to instantiate model variables
    self.params, self.opt_state, _ = rnn_utils.train_network(
        make_network,
        training_dataset=self.dataset,
        validation_dataset=self.dataset,
        random_key=self.random_key,
        loss='categorical',
        n_steps=0,
        opt=optax.adam(learning_rate=0.001))

  def test_dataset_rnn(self):
    data = next(self.dataset)
    xs, ys = data['xs'], data['ys']

    # Inputs for y-maze should be choice & reward
    self.assertEqual(np.shape(xs), (n_steps_per_session, batch_size, 2))
    self.assertTrue(np.all(np.logical_or(xs == 1, xs == 0)))

    # Targets for y-maze should be binary
    self.assertEqual(np.shape(ys), (n_steps_per_session, batch_size, 1))
    self.assertTrue(np.all(np.logical_or(ys == 1, ys == 0)))

  def test_dataset_rnn_batch_size(self):
    dataset = rnn_utils.DatasetRNN(
        xs=np.zeros((n_steps_per_session, n_sessions, 2)),
        ys=np.zeros((n_steps_per_session, n_sessions, 1)),
        y_type='categorical',
        batch_size=10)
    xs = next(dataset)['xs']
    self.assertEqual(np.shape(xs), (n_steps_per_session, 10, 2))

  def test_dataset_rnn_rolling_batch_gt_nepisodes(self):
    """Test rolling batch mode when batch_size > n_episodes."""
    n_episodes = 3
    batch_s = 5
    n_timesteps = 2
    n_features = 1

    # Create data where xs[0, episode_idx, 0] = episode_idx for easy checking
    xs_data = np.zeros((n_timesteps, n_episodes, n_features))
    for i in range(n_episodes):
      xs_data[:, i, 0] = i
    ys_data = np.zeros((n_timesteps, n_episodes, n_features))

    dataset = rnn_utils.DatasetRNN(
        xs=xs_data,
        ys=ys_data,
        y_type='scalar',
        batch_size=batch_s,
        batch_mode='rolling',
    )

    # Initial order: [0, 1, 2]
    # First batch expected indices: tile([0,1,2], 2)[:5] = [0, 1, 2, 0, 1]
    xs_batch1 = next(dataset)['xs']
    self.assertEqual(xs_batch1.shape, (n_timesteps, batch_s, n_features))
    np.testing.assert_array_equal(xs_batch1[0, :, 0], [0, 1, 2, 0, 1])
    # Expected start index after 1st batch: (0 + 5) % 3 = 2
    self.assertEqual(dataset._current_start_index, 2)

    # Second batch expected indices: tile([2,0,1], 2)[:5] = [2, 0, 1, 2, 0]
    xs_batch2 = next(dataset)['xs']
    self.assertEqual(xs_batch2.shape, (n_timesteps, batch_s, n_features))
    np.testing.assert_array_equal(xs_batch2[0, :, 0], [2, 0, 1, 2, 0])
    # Expected start index after 2nd batch: (2 + 5) % 3 = 1
    self.assertEqual(dataset._current_start_index, 1)

  def test_dataset_rnn_rolling_batch_lt_nepisodes(self):
    """Test rolling batch mode when batch_size < n_episodes."""
    n_episodes = 3
    batch_s = 2
    n_timesteps = 2
    n_features = 1

    xs_data = np.zeros((n_timesteps, n_episodes, n_features))
    for i in range(n_episodes):
      xs_data[:, i, 0] = i
    ys_data = np.zeros((n_timesteps, n_episodes, n_features))

    dataset = rnn_utils.DatasetRNN(
        xs=xs_data,
        ys=ys_data,
        y_type='scalar',
        batch_size=batch_s,
        batch_mode='rolling',
    )
    # Initial order: [0, 1, 2]
    # First batch: [0, 1]
    xs_batch1 = next(dataset)['xs']
    self.assertEqual(xs_batch1.shape, (n_timesteps, batch_s, n_features))
    np.testing.assert_array_equal(xs_batch1[0, :, 0], [0, 1])
    # Expected start index after 1st batch: (0 + 2) % 3 = 2
    self.assertEqual(dataset._current_start_index, 2)

    # Second batch: [2, 0]
    xs_batch2 = next(dataset)['xs']
    np.testing.assert_array_equal(xs_batch2[0, :, 0], [2, 0])
    # Expected start index after 2nd batch: (2 + 2) % 3 = 1
    self.assertEqual(dataset._current_start_index, 1)

  def test_dataset_rnn_rolling_batch_eq_nepisodes(self):
    """Test rolling batch mode when batch_size == n_episodes."""
    n_episodes = 3
    batch_s = 3
    n_timesteps = 2
    n_features = 1

    xs_data = np.zeros((n_timesteps, n_episodes, n_features))
    for i in range(n_episodes):
      xs_data[:, i, 0] = i
    ys_data = np.zeros((n_timesteps, n_episodes, n_features))

    dataset = rnn_utils.DatasetRNN(
        xs=xs_data,
        ys=ys_data,
        y_type='scalar',
        batch_size=batch_s,
        batch_mode='rolling',
    )
    # Initial order: [0, 1, 2]
    # First batch: [0, 1, 2]
    xs_batch1 = next(dataset)['xs']
    self.assertEqual(xs_batch1.shape, (n_timesteps, batch_s, n_features))
    np.testing.assert_array_equal(xs_batch1[0, :, 0], [0, 1, 2])
    # Expected start index after 1st batch: (0 + 3) % 3 = 0
    self.assertEqual(dataset._current_start_index, 0)

    # Second batch: [0, 1, 2]
    xs_batch2 = next(dataset)['xs']
    np.testing.assert_array_equal(xs_batch2[0, :, 0], [0, 1, 2])
    # Expected start index after 2nd batch: (0 + 3) % 3 = 0
    self.assertEqual(dataset._current_start_index, 0)

  def test_split_dataset(self):
    dataset_train, dataset_eval = rnn_utils.split_dataset(self.dataset, 2)
    xs_train_all = dataset_train.get_all()['xs']
    xs_eval_all = dataset_eval.get_all()['xs']
    self.assertEqual(np.shape(xs_train_all), (n_steps_per_session, 2, 2))
    self.assertEqual(np.shape(xs_eval_all), (n_steps_per_session, 1, 2))
    xs_train_batch = next(dataset_train)['xs']
    xs_eval_batch = next(dataset_eval)['xs']
    self.assertEqual(
        np.shape(xs_train_batch), (n_steps_per_session, batch_size, 2)
    )
    self.assertEqual(
        np.shape(xs_eval_batch), (n_steps_per_session, batch_size, 2)
    )

  def test_train_network(self):
    """Train the network for a few steps, check that the loss goes down.
    """
    # Train the network for a few steps
    new_params, new_opt_state, losses = rnn_utils.train_network(
        make_network,
        training_dataset=self.dataset,
        validation_dataset=self.dataset,
        random_key=self.random_key,
        loss='categorical',
        n_steps=100,
        opt=optax.adam(learning_rate=0.01),
        opt_state=self.opt_state,
        params=self.params)

    # Check that loss has gone down
    self.assertGreater(losses['training_loss'][0],
                       losses['training_loss'][-1])
    # Check that params have changed
    self.assertFalse(
        np.all(self.params['linear']['w'] == new_params['linear']['w']))
    # Check that opt state has changed
    self.assertNotEqual(self.opt_state, new_opt_state)

  def test_get_new_params(self):
    """Test that get_new_params returns a new set of params."""
    xs = next(self.dataset)['xs']
    input_size = xs.shape[-1]
    new_params = rnn_utils.get_new_params(self.make_network,
                                          input_size=input_size)
    self.assertNotEmpty(new_params, 'new_params should not be empty')
    self.assertIn('gru', new_params)

  def test_training_from_new_params(self):
    """Test that training from new params works."""
    xs = next(self.dataset)['xs']
    input_size = xs.shape[-1]
    new_params = rnn_utils.get_new_params(self.make_network,
                                          input_size=input_size)

    _, _, losses = rnn_utils.train_network(
        self.make_network,
        training_dataset=self.dataset,
        validation_dataset=self.dataset,
        random_key=self.random_key,
        loss='categorical',
        n_steps=100,
        opt=optax.adam(learning_rate=0.01),
        opt_state=None,
        params=new_params)
    self.assertGreater(losses['training_loss'][0],
                       losses['training_loss'][-1])

  def test_train_network_from_json_params(self):
    """Smoke test for training from params loaded from json."""

    # Generate params, convert them to json, convert them back
    initial_params, _, _ = rnn_utils.train_network(
        self.make_network,
        validation_dataset=None,
        training_dataset=self.dataset,
        loss='categorical',
        n_steps=0,
    )
    json_string = json.dumps(initial_params, cls=rnn_utils.NpJnpJsonEncoder)
    loaded_params = json.loads(json_string)

    # Check that we're able to train from loaded params
    new_params, _, losses = rnn_utils.train_network(
        self.make_network,
        training_dataset=self.dataset,
        validation_dataset=self.dataset,
        random_key=self.random_key,
        loss='categorical',
        n_steps=10,  # A small number of steps for a smoke test
        opt=optax.adam(learning_rate=0.01),
        params=loaded_params,
    )

    self.assertGreater(losses['training_loss'][0], losses['training_loss'][-1])
    initial_params_after_conversion = rnn_utils.to_np(loaded_params)
    self.assertFalse(
        np.all(
            initial_params_after_conversion['gru']['w_h']
            == new_params['gru']['w_h']
        )
    )

  def test_eval_network(self):
    """Eval a network on a set of inputs. Check shapes look right."""
    # Get a set of inputs
    xs = self.dataset.get_all()['xs']
    # Eval the network on that set of inputs
    y_hats, states = rnn_utils.eval_network(make_network, self.params, xs)

    # Check outputs have the right shape
    self.assertEqual(np.shape(y_hats), (n_steps_per_session, n_sessions, 2))
    # Check states have the right shape
    self.assertEqual(
        np.shape(states), (n_steps_per_session, n_sessions, n_hidden))

  def test_get_initial_state(self):

    state = rnn_utils.get_initial_state(self.make_network)
    # Check state has the right shape
    self.assertEqual(np.shape(state), (1, 1, n_hidden))

  def test_step_network(self):
    """Step the network forward two steps. Check the shapes all look right.
    """
    # Get a set of inputs
    xs = next(self.dataset)['xs']
    # Define a network state
    state = rnn_utils.get_initial_state(self.make_network)

    # Step the network from that state, using the first input
    _, new_state, _ = rnn_utils.step_network(
        self.make_network,
        params=self.params,
        state=state,
        xs=xs[0, 0])
    # Step it again
    y_hat, new_state, _ = rnn_utils.step_network(
        self.make_network,
        params=self.params,
        state=new_state,
        xs=xs[0, 0])

    # Check output has the right shape
    self.assertEqual(np.shape(y_hat)[0], 1)
    # Check new_state has the right shape
    self.assertEqual(np.shape(new_state), (1, 1, n_hidden))

  def test_NpJnpJsonEncoder(self):
    """Test that the NpEncoder works correctly with json.dumps."""
    encoder = rnn_utils.NpJnpJsonEncoder

    # Test basic types
    self.assertEqual(json.dumps(1, cls=encoder), '1')
    self.assertEqual(json.dumps(1.5, cls=encoder), '1.5')
    self.assertEqual(json.dumps(True, cls=encoder), 'true')
    self.assertEqual(json.dumps('hello', cls=encoder), '"hello"')
    self.assertEqual(json.dumps(None, cls=encoder), 'null')

    # Test NumPy types
    self.assertEqual(json.dumps(np.int32(5), cls=encoder), '5')
    self.assertEqual(json.dumps(np.float64(3.14), cls=encoder), '3.14')
    self.assertEqual(json.dumps(np.bool_(False), cls=encoder), 'false')
    self.assertEqual(json.dumps(np.array([1, 2, 3]), cls=encoder), '[1, 2, 3]')
    self.assertEqual(
        json.dumps(np.array([[1.1, 1.2], [2.1, 2.2]]), cls=encoder),
        '[[1.1, 1.2], [2.1, 2.2]]',
    )

    # Test JAX types
    self.assertEqual(json.dumps(jnp.int32(5), cls=encoder), '5')
    # JAX float64 serializes to a string with 15 decimal places
    self.assertEqual(
        json.dumps(jnp.float64(3.14), cls=encoder), '3.140000104904175'
    )
    self.assertEqual(json.dumps(jnp.array([1, 2, 3]), cls=encoder), '[1, 2, 3]')

    # Test nested structures
    data = {'a': np.array([1, 2]), 'b': [True, np.float32(1.0)], 'c': None}
    expected_json = '{"a": [1, 2], "b": [true, 1.0], "c": null}'
    self.assertEqual(json.dumps(data, cls=encoder), expected_json)

    # Test unsupported type
    with self.assertRaises(TypeError):
      json.dumps(object(), cls=encoder)

  def test_dataset_rnn_edge_cases(self):
    n_timesteps = 2
    n_features = 1
    xs_empty = np.empty((n_timesteps, 0, n_features))
    ys_empty = np.empty((n_timesteps, 0, n_features))

    # Empty dataset (n_episodes = 0) should raise ValueError on init
    for batch_mode in ['single', 'rolling', 'random']:
      with self.assertRaises(ValueError):
        _ = rnn_utils.DatasetRNN(
            xs=xs_empty,
            ys=ys_empty,
            y_type='scalar',
            batch_size=0,
            batch_mode=batch_mode,
        )


if __name__ == '__main__':
  absltest.main()
