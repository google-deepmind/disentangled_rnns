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

"""Syntehtic two armed bandit datasets with simulated dopamine."""

from typing import NamedTuple, Union

from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
import numpy as np


class AgentRewardSeeking:
  """An agent that uses a single 'reward-seeking' variable.

  Computes prediction error as the difference between reward and this variable.
  """

  def __init__(
      self,
      alpha: float,  # Learning rate
      beta: float,  # softmax temp
  ):
    """Initialize the agent.

    Args:
      alpha: alpha: The agent's learning rate
      beta: The agent's softmax temperature
    """
    self._alpha_r = alpha
    self._beta_r = beta
    self.new_session()

  def new_session(self):
    """Reset the agent for the beginning of a new session."""
    self.r = 0.0
    self.last_rpe = 0.0

  def get_choice_probs(self) -> np.ndarray:

    choice_logits = np.array([-1, 1]) * (self._beta_r * self.r)
    choice_probs = np.exp(choice_logits) / np.sum(np.exp(choice_logits))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""

    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def get_rpe(self) -> float:
    return self.last_rpe

  def update(self, choice: int, reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    assert choice in [0, 1]
    assert reward in [0, 1]
    choice_for_update = 2 * choice - 1  # Convert 0, 1 or -1, 1
    reward_for_update = 2 * reward - 1  # Convert 0, 1 to -1,1

    # Since 'r' is a sided policy variable, the proxy for 'reward
    # expectation' is choice_for_update * r
    self.last_rpe = reward_for_update - choice_for_update * self.r

    # To update the sided variable we need to use a signed RPE
    self.r = self.r + self._alpha_r * choice_for_update * self.last_rpe


class AgentQ(two_armed_bandits.AgentQ):
  """An agent that runs "vanilla" Q-learning for the y-maze tasks.

  Also keeps track of the reward prediction error (RPE).
  """

  def __init__(
      self,
      alpha: float,  # Learning rate
      beta: float,  # softmax temp
  ):
    super().__init__(alpha, beta)
    self.last_rpe = 0.0

  def get_rpe(self) -> float:
    return self.last_rpe

  def update(self, choice: int, reward: float):
    """Update the agent after one step of the task."""
    # First, we calculate the RPE as the difference between the actual reward
    # and the predicted reward.
    self.last_rpe = reward - self.q[choice]
    # Now we update the agent's Q-values (i.e its predictions for rewards).
    super().update(choice, reward)


class SessData(NamedTuple):
  choices: np.ndarray
  rewards: np.ndarray
  reward_probs: np.ndarray
  rpes: np.ndarray
  n_trials: int


Agent = Union[AgentQ, AgentRewardSeeking]


def run_experiment(
    agent: Agent,
    environment: two_armed_bandits.EnvironmentBanditsDrift,
    n_steps: int,
) -> SessData:
  """Runs a behavioral session from a given agent and environment."""
  choices = np.zeros(n_steps)
  rewards = np.zeros(n_steps)
  reward_probs = np.zeros((n_steps, 2))
  rpes = np.zeros(n_steps)

  for step in np.arange(n_steps):
    # First record environment reward probs
    reward_probs[step] = environment.reward_probs

    attempted_choice = agent.get_choice()
    choice, reward, _ = environment.step(attempted_choice)
    agent.update(choice, reward)
    rpe = agent.get_rpe()

    choices[step] = choice
    rewards[step] = reward
    rpes[step] = rpe

  experiment = SessData(
      choices=choices,
      rewards=rewards,
      n_trials=n_steps,
      rpes=rpes,
      reward_probs=reward_probs,
  )
  return experiment


def create_dataset(
    agent: Agent,
    environment: two_armed_bandits.EnvironmentBanditsDrift,
    n_steps_per_session: int,
    n_sessions: int,
    batch_size: int,
) -> rnn_utils.DatasetRNN:
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices and RPEs. Note that this agent
      must be ones of the agents described in this class, i.e. something that
      generates the reward prediction error (RPE) as well.
    environment: An environment object to generate rewards
    n_steps_per_session: The number of trials in each behavioral session to be
      generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset

  Returns:
  """
  xs = np.zeros((n_steps_per_session, n_sessions, 2))
  ys = np.zeros((n_steps_per_session, n_sessions, 2))

  for sess_i in np.arange(n_sessions):
    experiment = run_experiment(agent, environment, n_steps_per_session)
    prev_choices = np.concatenate(([0], experiment.choices[0:-1]))
    prev_rewards = np.concatenate(([0], experiment.rewards[0:-1]))
    prev_rpes = np.concatenate(([0], experiment.rpes[0:-1]))
    xs[:, sess_i] = np.swapaxes(
        np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1
    )
    ys[:, sess_i] = np.swapaxes(
        np.concatenate(([experiment.choices], [prev_rpes]), axis=0), 0, 1
    )

  dataset = rnn_utils.DatasetRNN(
      xs=xs,
      ys=ys,
      x_names=['prev choice', 'prev reward'],
      y_names=['choice', 'dopamine'],
      y_type='mixed',
      n_classes=2,
      batch_size=batch_size,
  )
  return dataset


def get_q_learning_with_dopamine_dataset(
    n_sessions: int = 500,
    alpha: float = 0.3,
    beta: float = 3,
    n_trials: int = 500,
) -> rnn_utils.DatasetRNN:
  """Generates synthetic dataset of behavior and dopamine from Q-Learning agent."""
  np.random.seed(0)
  agent = AgentQ(alpha=alpha, beta=beta)
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
  dataset = create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset


def get_reward_seeking_with_dopamine_dataset(
    n_sessions: int = 500,
    alpha: float = 0.3,
    beta: float = 3,
    n_trials: int = 500,
) -> rnn_utils.DatasetRNN:
  """Generates synthetic dataset of behavior and dopamine from a reward-seeking agent."""
  np.random.seed(0)
  agent = AgentRewardSeeking(alpha=alpha, beta=beta)
  environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
  dataset = create_dataset(
      agent,
      environment,
      n_steps_per_session=n_trials,
      n_sessions=n_sessions,
      batch_size=n_sessions,
  )
  return dataset
