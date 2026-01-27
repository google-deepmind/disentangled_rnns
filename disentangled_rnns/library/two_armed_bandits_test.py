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
from absl.testing import parameterized
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import haiku as hk
import numpy as np


N_HIDDEN = 5


def make_gru_network():
  return hk.DeepRNN([hk.GRU(N_HIDDEN), hk.Linear(output_size=2)])


def get_agent_network():
  params, _, _ = rnn_utils.train_network(
      make_gru_network,
      training_dataset=rnn_utils.DatasetRNN(
          xs=np.zeros((10, 1, 2)),
          ys=np.zeros((10, 1, 1)),
          y_type='categorical',
          n_classes=2,
      ),
      validation_dataset=None,
      n_steps=0,
  )
  return two_armed_bandits.AgentNetwork(make_gru_network, params)


named_agents_list = [
    ('AgentQ', two_armed_bandits.AgentQ(alpha=0.3, beta=3)),
    (
        'AgentLeakyActorCritic',
        two_armed_bandits.AgentLeakyActorCritic(
            alpha_critic=0.2, alpha_actor_learn=0.2, alpha_actor_forget=0.05
        ),
    ),
]


class TwoArmedBanditsTest(parameterized.TestCase):

  @parameterized.named_parameters(named_agents_list)
  def test_update(self, agent):
    """Checks that agent increases its choice probability after getting a reward."""

    choice_probs_pre = agent.get_choice_probs()
    agent.update(choice=0, reward=1)
    choice_probs_post = agent.get_choice_probs()

    self.assertGreater(choice_probs_post[0], choice_probs_pre[0])

  @parameterized.named_parameters(named_agents_list)
  def test_step(self, agent):
    """Smoke test for stepping environment and agent together."""
    environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.01)
    agent.new_session()
    # Get a choice from the agent. Should be an int, 0 or 1
    attempted_choice = agent.get_choice()
    self.assertIsInstance(attempted_choice, int)
    self.assertIn(attempted_choice, [0, 1])
    # Get choice, reward, instructed from the environment
    choice, reward, instructed = environment.step(attempted_choice)
    # Choice is an int, 0 or 1
    self.assertIsInstance(choice, int)
    self.assertIn(choice, [0, 1])
    # Reward is a float between 0 and 1
    self.assertIsInstance(reward, float)
    self.assertGreaterEqual(reward, 0)
    self.assertLessEqual(reward, 1)
    # Instructed is an int, 0 or 1
    self.assertIsInstance(instructed, int)
    self.assertIn(instructed, [0, 1])

    # Update the agent with the choice and reward, get a new choice
    agent.update(choice=choice, reward=reward)
    attempted_choice = agent.get_choice()
    self.assertIsInstance(attempted_choice, int)
    self.assertIn(attempted_choice, [0, 1])

  @parameterized.named_parameters(named_agents_list)
  def test_generate_dataset(self, agent):
    """Smoke test generating a dataset the environment."""
    environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.01)
    dataset = two_armed_bandits.create_dataset(
        environment=environment,
        agent=agent,
        n_steps_per_session=10,
        n_sessions=10,
    )

    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_run_experiment_and_create_dataset_smoke_test(self):
    """Smoke test for run_experiment and create_dataset with different agents and environments."""
    n_steps = 10
    n_sessions = 2
    agents = [
        two_armed_bandits.AgentQ(alpha=0.3, beta=3),
        two_armed_bandits.AgentLeakyActorCritic(
            alpha_critic=0.2, alpha_actor_learn=0.2, alpha_actor_forget=0.05
        ),
        get_agent_network(),
    ]
    environments_builders = [
        lambda: two_armed_bandits.EnvironmentBanditsDrift(sigma=0.01),
        lambda: two_armed_bandits.EnvironmentPayoutMatrix(
            payout_matrix=np.array(
                [np.tile([0, 1], (n_steps, 1)), np.tile([1, 0], (n_steps, 1))]
            )
        ),
    ]

    for agent in agents:
      for env_builder in environments_builders:
        # Test run_experiment
        environment = env_builder()
        agent.new_session()
        environment.new_session()
        experiment = two_armed_bandits.run_experiment(
            agent, environment, n_steps
        )
        self.assertIsInstance(experiment, two_armed_bandits.SessData)
        if not hasattr(environment, 'reward_probs'):
          self.assertTrue(np.all(np.isnan(experiment.reward_probs)))
        else:
          self.assertFalse(np.any(np.isnan(experiment.reward_probs)))

        # Test create_dataset
        dataset_environment = env_builder()
        agent.new_session()
        dataset = two_armed_bandits.create_dataset(
            agent, dataset_environment, n_steps, n_sessions
        )
        self.assertIsInstance(dataset, rnn_utils.DatasetRNN)


if __name__ == '__main__':
  absltest.main()
