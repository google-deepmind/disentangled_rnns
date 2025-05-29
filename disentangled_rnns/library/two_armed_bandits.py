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

"""Two armed bandit experiments. Generate synthetic data, plot data."""

import abc
from collections.abc import Callable
from typing import NamedTuple, Optional, Union

from disentangled_rnns.library import rnn_utils
import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np


abstractmethod = abc.abstractmethod

################
# ENVIRONMENTS #
################


class BaseEnvironment(abc.ABC):
  """Base class for two-armed bandit environments.

  Subclasses must implement the following methods:
    - new_session()
    - step(choice)

  Attributes:
    seed: The seed used to initialize the environment.
    n_arms: The number of arms in the environment.
  """

  def __init__(self, seed: Optional[int] = None, n_arms: int = 2):
    self._random_state = np.random.RandomState(seed)
    self._n_arms = n_arms

  @abstractmethod
  def new_session(self):
    """Starts a new session (e.g., resets environment parameters).

    This method should be implemented by subclasses to initialize or
    reset the environment's state at the beginning of a new session or episode.
    """

  @abstractmethod
  def step(self, attempted_choice: int) -> tuple[int, float, int]:
    """Executes a single step in the environment.

    Args:
      attempted_choice: The action chosen by the agent.

    Returns:
      choice: The action actually taken. May be different from the attempted
        choice if the environment decides the choice should be instructed on
        that trial.
      reward: The reward received after taking the action.
      instructed: 1 if the choice was instructed, 0 otherwise
    """

  @property
  def n_arms(self) -> int:
    """Returns the current reward probabilities for each arm."""
    return self._n_arms


class EnvironmentBanditsDrift(BaseEnvironment):
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and
  1. On each trial, gaussian random noise is added to each, with the constraint
  that the probabilities cannot go below 0 or above 1.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_arms: The number of arms in the environment.
  """

  def __init__(self,
               sigma: float,
               p_instructed: float = 0.0,
               seed: Optional[int] = None,
               n_arms: int = 2,
               ):
    super().__init__(seed=seed, n_arms=n_arms)

    # Check inputs
    if sigma < 0:
      msg = ('sigma was {}, but must be greater than 0')
      raise ValueError(msg.format(sigma))

    # Initialize persistent properties
    self._sigma = sigma
    self._p_instructed = p_instructed

    # Sample new reward probabilities
    self.new_session()

  def new_session(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = self._random_state.rand(self.n_arms)

  def step(self,
           attempted_choice: int) -> tuple[int, float, int]:
    """Run a single trial of the task.

    Args:
      attempted_choice: The choice made by the agent. 0 or 1

    Returns:
      choice: The action actually taken. May be different from the attempted
        choice if the environment decides the choice should be instructed on
        that trial.
      reward: The reward to be given to the agent. 0 or 1.
      instructed: 1 if the choice was instructed, 0 otherwise

    """
    if attempted_choice == -1:
      choice = -1
      reward = -1
      instructed = -1
      return choice, reward, instructed

    # Check inputs
    if attempted_choice not in list(range(self.n_arms)):
      msg = (f'choice given was {attempted_choice}, but must be one of '
             f'{list(range(self.n_arms))}.')
      raise ValueError(msg)

    # If choice was instructed, overrule it and decide randomly
    instructed = self._random_state.rand() < self._p_instructed
    if instructed:
      choice = self._random_state.choice(self.n_arms)
    else:
      choice = attempted_choice

    # Sample reward with the probability of the chosen side
    reward = self._random_state.rand() < self._reward_probs[choice]
    # Add gaussian noise to reward probabilities
    drift = self._random_state.normal(
        loc=0, scale=self._sigma, size=self.n_arms)
    self._reward_probs += drift

    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    return choice, float(reward), int(instructed)

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()


class NoMoreTrialsInSessionError(ValueError):
  pass


class NoMoreSessionsInDatasetError(ValueError):
  pass


class EnvironmentPayoutMatrix(BaseEnvironment):
  """Environment for a two-armed bandit task with a specified payout matrix."""

  def __init__(
      self,
      payout_matrix: np.ndarray,
      instructed_matrix: Optional[np.ndarray] = None,
  ):
    """Initialize the environment.

    Args:
      payout_matrix: A numpy array of shape (n_sessions, n_actions, n_trials)
        giving the reward for each session, action, and trial. These are
        deterministic, i.e. for the same trial_num, session_num, and action, the
        reward will always be the same. (If you'd like stochastic rewards you
        can populate this matrix ahead of time).
      instructed_matrix: A numpy array of shape (n_sessions, n_trials) giving
        the choice that should be made, if any, for each session and trial.
        Elements should be ints or nan. If nan, the choice is not instructed. If
        None, no choices are instructed.
    """
    n_arms = payout_matrix.shape[2]
    super().__init__(seed=None, n_arms=n_arms)

    self._payout_matrix = payout_matrix
    self._n_sessions = payout_matrix.shape[0]
    self._n_trials = payout_matrix.shape[1]

    if instructed_matrix is not None:
      self._instructed_matrix = instructed_matrix
    else:
      self._instructed_matrix = np.nan * np.zeros_like(payout_matrix)

    self._current_session = 0
    self._current_trial = 0

  def new_session(self):
    self._current_session += 1
    if self._current_session >= self._n_sessions:
      raise NoMoreSessionsInDatasetError(
          'No more sessions in dataset. '
          f'Current session {self._current_session} is out of range '
          f'[0, {self._n_sessions - 1})'
      )
    self._current_trial = 0

  def step(self, attempted_choice: int) -> tuple[int, float, int]:
    # If agent choice is default empty value -1, return -1 for all outputs.
    if attempted_choice == -1:
      choice = -1
      reward = -1
      instructed = -1
      return choice, reward, instructed

    # Check inputted choice is valid.
    if attempted_choice not in list(range(self.n_arms)):
      msg = (f'choice given was {attempted_choice}, but must be one of '
             f'{list(range(self.n_arms))}.')
      raise ValueError(msg)

    if self._current_trial >= self._n_trials:
      raise NoMoreTrialsInSessionError(
          'No more trials in session. '
          f'Current trial {self._current_trial} is out of range '
          f'[0, {self._n_trials})'
      )

    # If choice was instructed, overrule and replace with the instructed choice
    instruction = self._instructed_matrix[
        self._current_session, self._current_trial
    ]
    instructed = not np.isnan(instruction)
    if instructed:
      choice = int(instruction)
    else:
      choice = attempted_choice

    reward = self._payout_matrix[
        self._current_session, self._current_trial, choice
    ]
    self._current_trial += 1
    return choice, float(reward), int(instructed)

  @property
  def payout(self) -> np.ndarray:
    """Get possible payouts for current session, trial across actions."""
    return self._payout_matrix[
        self._current_session, self._current_trial, :].copy()

##########
# AGENTS #
##########


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
    self.new_session()

  def new_session(self):
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
             reward: float):
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
    self.new_session()

  def new_session(self):
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

  def update(self, choice: int, reward: float):
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
    self.new_session()

  def new_session(self):
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
    experiment: A SessData object holding choices and rewards from the session
  """
  choices = np.zeros(n_steps)
  rewards = np.zeros(n_steps)
  reward_probs = np.zeros((n_steps, 2))

  for step in np.arange(n_steps):
    # First record environment reward probs
    reward_probs[step] = environment.reward_probs
    # First agent makes a choice
    attempted_choice = agent.get_choice()
    # Then environment computes a reward
    choice, reward, _ = environment.step(attempted_choice)
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
    environment.new_session()

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
