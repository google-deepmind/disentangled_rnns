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
from absl.testing import parameterized
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits  # For Environment
from disentangled_rnns.library import two_armed_bandits_w_dopamine


# Agents to be tested from two_armed_bandits_w_dopamine.py
named_agents_w_dopamine_list = [
    (
        'AgentRewardSeeking',
        two_armed_bandits_w_dopamine.AgentRewardSeeking(alpha=0.3, beta=3.0),
    ),
    (
        'AgentQ_w_dopamine',
        two_armed_bandits_w_dopamine.AgentQ(alpha=0.3, beta=3.0),
    ),
]


class TwoArmedBanditsWDopamineTest(parameterized.TestCase):

  @parameterized.named_parameters(named_agents_w_dopamine_list)
  def test_update_and_rpe(self, agent):
    agent.new_session()
    choice_probs_pre = agent.get_choice_probs()
    chosen_action = 0
    reward_received = 1

    agent.update(choice=chosen_action, reward=reward_received)
    choice_probs_post = agent.get_choice_probs()
    rpe = agent.get_rpe()

    # Choice_probs should increase after reward
    self.assertGreater(
        choice_probs_post[chosen_action],
        choice_probs_pre[chosen_action],
        msg=(
            f'Choice prob for action {chosen_action} should increase after'
            ' reward.'
        ),
    )

    # rpe should be non zero after update
    self.assertGreater(abs(rpe), 0.0)

  @parameterized.named_parameters(named_agents_w_dopamine_list)
  def test_create_dataset_w_dopamine(self, agent):
    """Test create_dataset from two_armed_bandits_w_dopamine."""
    environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.01)
    n_steps_per_session = 10
    n_sessions = 5
    batch_size = 5

    dataset = two_armed_bandits_w_dopamine.create_dataset(
        agent=agent,
        environment=environment,
        n_steps_per_session=n_steps_per_session,
        n_sessions=n_sessions,
        batch_size=batch_size,
    )

    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_get_q_learning_with_dopamine_dataset(self):
    """Smoke test for get_q_learning_with_dopamine_dataset."""
    n_sessions = 10
    n_trials = 20
    dataset = two_armed_bandits_w_dopamine.get_q_learning_with_dopamine_dataset(
        n_sessions=n_sessions,
        n_trials=n_trials,
        alpha=0.3,
        beta=3.0,
    )
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)

  def test_get_reward_seeking_with_dopamine_dataset(self):
    """Smoke test for get_reward_seeking_with_dopamine_dataset."""
    n_sessions = 10
    n_trials = 20
    dataset = (
        two_armed_bandits_w_dopamine.get_reward_seeking_with_dopamine_dataset(
            n_sessions=n_sessions,
            n_trials=n_trials,
            alpha=0.3,
            beta=3.0,
        )
    )
    self.assertIsInstance(dataset, rnn_utils.DatasetRNN)


if __name__ == '__main__':
  absltest.main()
