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

from absl.testing import absltest
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import haiku as hk
import jax
import numpy as np
import optax


n_steps_per_session = 10
n_sessions = 3
batch_size = n_sessions
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
        batch_size=batch_size)

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
    xs, ys = next(self.dataset)

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
        batch_size=None)
    xs, _ = next(dataset)
    self.assertEqual(np.shape(xs), (n_steps_per_session, n_sessions, 2))

  def test_split_dataset(self):
    dataset_train, dataset_eval = rnn_utils.split_dataset(self.dataset, 2)
    xs_train, _ = next(dataset_train)
    xs_eval, _ = next(dataset_eval)
    self.assertEqual(np.shape(xs_train), (n_steps_per_session, 2, 2))
    self.assertEqual(np.shape(xs_eval), (n_steps_per_session, 1, 2))

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

  def test_eval_network(self):
    """Eval a network on a set of inputs. Check shapes look right.
    """
    # Get a set of inputs
    xs, _ = next(self.dataset)
    # Eval the network on that set of inputs
    y_hats, states = rnn_utils.eval_network(make_network, self.params, xs)

    # Check outputs have the right shape
    self.assertEqual(np.shape(y_hats), (n_steps_per_session, batch_size, 2))
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
    xs, _ = next(self.dataset)
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


if __name__ == '__main__':
  absltest.main()
