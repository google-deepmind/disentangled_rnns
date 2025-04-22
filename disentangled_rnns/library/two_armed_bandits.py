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

"""Two armed bandit experiments. Generate synthetic data, plot data."""

from collections.abc import Callable
from typing import NamedTuple, Optional, Union

from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np


class EnvironmentBanditsDrift:
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and
  1. On each trial, gaussian random noise is added to each, with the constraint
  that the probabilities cannot go below 0 or above 1.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
  """

  def __init__(self,
               sigma: float,
               seed: Optional[int] = None,
               ):

    # Check inputs
    if sigma < 0:
      msg = ('sigma was {}, but must be greater than 0')
      raise ValueError(msg.format(sigma))
    # Initialize persistent properties
    self._sigma = sigma
    self._random_state = np.random.RandomState(seed)

    # Sample new reward probabilities
    self.new_sess()

  def new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = self._random_state.rand(2)

  def step(self,
           choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: The choice made by the agent. 0 or 1

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    if not np.logical_or(choice == 0, choice == 1):
      msg = ('choice given was {}, but must be either 0 or 1')
      raise ValueError(msg.format(choice))

    # Sample reward with the probability of the chosen side
    reward = self._random_state.rand() < self._reward_probs[choice]
    # Add gaussian noise to reward probabilities
    drift = self._random_state.normal(loc=0, scale=self._sigma, size=2)
    self._reward_probs += drift

    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    return reward

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()


class AgentQ:
  """An agent that runs "vanilla" Q-learning for the y-maze tasks.

  Attributes:
    q: The agent's current estimate of the reward probability on each arm

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
    self._alpha = alpha
    self._beta = beta
    self.new_sess()

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self.q = 0.5 * np.ones(2)

  def get_choice_probs(self) -> np.ndarray:
    choice_probs = np.exp(self._beta * self.q) / np.sum(
        np.exp(self._beta * self.q))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""

    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    self.q[choice] = (1 - self._alpha) * self.q[choice] + self._alpha * reward


class AgentLeakyActorCritic:
  """An agent that runs Actor-Critic learning for the y-maze tasks.

  Attributes:
    theta: The actor's state variable. Updated trial-by-trial using a leaky
      gradient bandit algorithm.
    v: The critic's state variable. Updated trial-by-trial to keep a running
      estimate of recent reward rate.
  """

  def __init__(
      self,
      alpha_critic: float,
      alpha_actor_learn: float,
      alpha_actor_forget: float,
  ):
    """Initialize the agent.

    Args:
      alpha_critic: The critic's learning rate
      alpha_actor_learn: The actor's learning rate
      alpha_actor_forget: The actor's forgetting rate
    """
    self._alpha_critic = alpha_critic
    self._alpha_actor_learn = alpha_actor_learn
    self._alpha_actor_forget = alpha_actor_forget
    self.new_sess()

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self.theta = 0. * np.ones(2)
    self.v = 0.5

  def get_choice_probs(self) -> np.ndarray:
    choice_probs = np.exp(self.theta) / np.sum(
        np.exp(self.theta))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""

    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    unchosen = 1 - choice  # Convert 0 to 1 or 1 to 0
    # Actor learning
    # Follows "gradient bandit" algorithm from Sutton & Barto, chapter 2, but
    # with the addition of forgetting
    choice_probs = self.get_choice_probs()
    rpe = reward - self.v
    self.theta[choice] = (1 - self._alpha_actor_forget) * self.theta[
        choice] + self._alpha_actor_learn * rpe * (1 - choice_probs[choice])
    self.theta[unchosen] = (1 - self._alpha_actor_forget) * self.theta[
        unchosen] - self._alpha_actor_learn * rpe * (choice_probs[unchosen])

    # Critic learing: V moves towards reward
    self.v = (1 - self._alpha_critic) * self.v + self._alpha_critic * reward


class AgentNetwork:
  """A class that allows running a trained RNN as an agent.

  Attributes:
    make_network: A Haiku function that returns an RNN architecture
    params: A set of Haiku parameters suitable for that architecture
  """

  def __init__(self,
               make_network: Callable[[], hk.RNNCore],
               params: hk.Params):

    def step_network(
        xs: np.ndarray, state: hk.State
    ) -> tuple[np.ndarray, hk.State]:
      core = make_network()
      y_hat, new_state = core(xs, state)
      return y_hat, new_state

    def get_initial_state() -> hk.State:
      core = make_network()
      rnn_state = core.initial_state(1)
      return rnn_state

    model = hk.without_apply_rng(hk.transform(step_network))
    rnn_state = hk.without_apply_rng(hk.transform(get_initial_state))

    self._initial_state = rnn_state.apply(params)
    self._model_fun = jax.jit(
        lambda xs, state: model.apply(params, xs, rnn_state)
    )
    self._xs = np.zeros((1, 2))
    self.new_sess()

  def new_sess(self):
    self._rnn_state = self._initial_state

  def get_choice_probs(self) -> np.ndarray:
    output_logits, _ = self._model_fun(self._xs, self._rnn_state)
    choice_probs = np.asarray(jax.nn.softmax(output_logits[0]))
    return choice_probs

  def get_choice(self) -> tuple[int, np.ndarray]:
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    self._xs = np.array([[choice, reward]])
    _, self._rnn_state = self._model_fun(self._xs, self._rnn_state)


Agent = Union[AgentQ, AgentLeakyActorCritic, AgentNetwork]


class SessData(NamedTuple):
  choices: np.ndarray
  rewards: np.ndarray
  reward_probs: np.ndarray
  n_trials: int


def run_experiment(agent: Agent,
                   environment: EnvironmentBanditsDrift,
                   n_steps: int) -> SessData:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_steps: The number of steps in the session you'd like to generate

  Returns:
    experiment: A YMazeSession holding choices and rewards from the session
  """
  choices = np.zeros(n_steps)
  rewards = np.zeros(n_steps)
  reward_probs = np.zeros((n_steps, 2))

  for step in np.arange(n_steps):
    # First record environment reward probs
    reward_probs[step] = environment.reward_probs
    # First agent makes a choice
    choice = agent.get_choice()
    # Then environment computes a reward
    reward = environment.step(choice)
    # Finally agent learns
    agent.update(choice, reward)
    # Log choice and reward
    choices[step] = choice
    rewards[step] = reward

  experiment = SessData(choices=choices,
                        rewards=rewards,
                        n_trials=n_steps,
                        reward_probs=reward_probs)
  return experiment


def create_dataset(agent: Agent,
                   environment: EnvironmentBanditsDrift,
                   n_steps_per_session: int,
                   n_sessions: int,
                   batch_size: int) -> rnn_utils.DatasetRNN:
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices
    environment: An environment object to generate rewards
    n_steps_per_session: The number of trials in each behavioral session to
      be generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset

  Returns:
    rnn_utils.DatasetRNN object
  """
  xs = np.zeros((n_steps_per_session, n_sessions, 2))
  ys = np.zeros((n_steps_per_session, n_sessions, 1))

  for sess_i in np.arange(n_sessions):
    experiment = run_experiment(agent, environment, n_steps_per_session)
    prev_choices = np.concatenate(([0], experiment.choices[0:-1]))
    prev_rewards = np.concatenate(([0], experiment.rewards[0:-1]))
    xs[:, sess_i] = np.swapaxes(
        np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1
    )
    ys[:, sess_i] = np.expand_dims(experiment.choices, 1)

  dataset = rnn_utils.DatasetRNN(
      xs=xs,
      ys=ys,
      x_names=['prev choice', 'prev reward'],
      y_names=['choice'],
      y_type='categorical',
      n_classes=2,
      batch_size=batch_size,
  )
  return dataset


def plot_sessdata(sessdata: SessData):
  """Creates a figure showing data from a single behavioral session.

  Args:
    sessdata: A session of data to plot
  """

  choose_high = sessdata.choices == 1
  choose_low = sessdata.choices == 0
  rewarded = sessdata.rewards == 1

  # Make the plot
  plt.subplots(figsize=(10, 3))
  plt.plot(sessdata.reward_probs)

  # Rewarded high
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      1.1 * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker=3)
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      1.1 * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker='|')
  # Omission high
  plt.scatter(
      np.argwhere(choose_high & 1 - rewarded),
      1.1 * np.ones(np.sum(choose_high & 1 - rewarded)),
      color='red',
      marker='|')

  # Rewarded low
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      -0.1 * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker='|')
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      -0.1 * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker=2)
  # Omission Low
  plt.scatter(
      np.argwhere(choose_low & 1 - rewarded),
      -0.1 * np.ones(np.sum(choose_low & 1 - rewarded)),
      color='red',
      marker='|')

  plt.xlabel('Trial')
  plt.ylabel('Probability')
