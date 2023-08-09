# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Cognitive model from Miller, Botvinick, and Brody."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from disentangled_rnns.code.library import rnn_utils


class HkCognitiveModel(hk.RNNCore):
  """Cognitive model from Miller, Botvinick, and Brody, expressed in Haiku.

  Three agents: Reward-seeking, habit, gamblers fallacy
  """

  def __init__(self, n_cs=4):
    super().__init__()

    # Haiku parameters
    alpha_rs_logit = hk.get_parameter(
        'alpha_rs_logit',
        (1,),
        init=hk.initializers.RandomUniform(minval=0, maxval=1),
    )
    alpha_h_logit = hk.get_parameter(
        'alpha_h_logit',
        (1,),
        init=hk.initializers.RandomUniform(minval=-1, maxval=0),
    )
    alpha_gf_logit = hk.get_parameter(
        'alpha_gf_logit',
        (1,),
        init=hk.initializers.RandomUniform(minval=-2, maxval=-1),
    )

    beta_rs = hk.get_parameter(
        'beta_rs', (1,), init=hk.initializers.RandomUniform(minval=0, maxval=2)
    )
    beta_h = hk.get_parameter(
        'beta_h', (1,), init=hk.initializers.RandomUniform(minval=0, maxval=2)
    )
    beta_gf = hk.get_parameter(
        'beta_gf', (1,), init=hk.initializers.RandomUniform(minval=0, maxval=2)
    )
    beta_bias = hk.get_parameter(
        'beta_bias',
        (1,),
        init=hk.initializers.RandomUniform(minval=-0.5, maxval=0.5),
    )

    # Local parameters
    self.alpha_rs = 1 / (1 + jnp.exp(-alpha_rs_logit))
    self.alpha_h = 1 / (1 + jnp.exp(-alpha_h_logit))
    self.alpha_gf = 1 / (1 + jnp.exp(-alpha_gf_logit))
    self.beta_rs = beta_rs
    self.beta_h = beta_h
    self.beta_gf = beta_gf
    self.beta_bias = beta_bias

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    prev_values = prev_state
    prev_r = prev_values[:, 0]  # shape: (batch_size, 1)
    prev_h = prev_values[:, 1]  # shape: (batch_size, 1)
    prev_g = prev_values[:, 2]  # shape: (batch_size, 1)

    choice = inputs[:, 0]  # shape: (batch_size, 1)
    reward = inputs[:, 1]  # shape: (batch_size, 1)

    # Convert choice and reward from {0, 1} to {-1, +1}
    choice_for_update = 2 * choice - 1
    reward_for_update = 2 * reward - 1
    # If either choice or reward is -1, this is the first trial. Do not update
    choice_for_update = choice_for_update * (choice != -1) * (reward != -1)
    reward_for_update = reward_for_update * (choice != -1) * (reward != -1)

    # Update representations
    new_r = (
        1 - self.alpha_rs
    ) * prev_r + self.alpha_rs * choice_for_update * reward_for_update
    new_h = (1 - self.alpha_h) * prev_h + self.alpha_h * (choice_for_update)
    new_g = (1 - self.alpha_gf) * prev_g + self.alpha_gf * (
        choice_for_update - choice_for_update * reward_for_update
    )

    # Compute output logits
    choice_logit = (
        self.beta_rs * new_r
        + self.beta_h * new_h
        + self.beta_gf * new_g
        + self.beta_bias
    )
    output = jnp.stack((-1 * choice_logit, choice_logit), axis=-1)

    new_values = jnp.stack((new_r, new_h, new_g), axis=-1)
    new_state = new_values
    return output, new_state

  def initial_state(self, batch_size):
    values = jnp.zeros([batch_size, 3])  # shape: (batch_size, n_cs)
    return values


def fit_model(
    model_fun,
    dataset,
    dataset_eval=None,
    epsilon=1e-4,
    convergence_thresh=1e-5,
    n_match=5,
    max_n=20,
    random_key=None,
):
  """Fits parameters of the cognitive model from Miller, Botvinick, and Brody.
  """
  if dataset_eval is None:
    dataset_eval = dataset

  if random_key is None:
    random_key = jax.random.PRNGKey(np.random.randint(2**32))

  # Define a reasonable optimizer
  opt = optax.adabelief(learning_rate=1e-1)

  # Define what it means to do one round of fitting
  def fit_sub():
    # Train one step to initialize everything
    params, opt_state, losses = rnn_utils.train_network(
        model_fun,
        dataset,
        dataset_eval,
        loss='categorical',
        opt=opt,
        n_steps=0,
        random_key=random_key,
    )

    # Train until either the fit converges or the error is less than epsilon
    converged = False
    loss = np.inf
    while loss > epsilon and not converged:
      params, opt_state, losses = rnn_utils.train_network(
          model_fun,
          dataset,
          dataset_eval,
          loss='categorical',
          params=params,
          opt_state=opt_state,
          opt=opt,
          max_grad_norm=1,
          n_steps=100,
      )

      loss_new = losses['training_loss'][-1]
      # Declare "converged" if loss has not improved enough
      converged = loss_new > loss * (1 - convergence_thresh)
      loss = loss_new
    eval_loss = losses['validation_loss'][-1]
    return params, loss, eval_loss

  # Fit until a sufficient number have converged on the same solution
  done = False
  params_list = []
  loss_list = []
  eval_loss_list = []
  n_tried = 0
  while not done:
    n_tried += 1
    print('Fitting! Attempt ', n_tried)
    params, loss, eval_loss = fit_sub()
    params_list.append(params)
    loss_list.append(loss)
    eval_loss_list.append(eval_loss)
    best_loss = np.min(np.array(loss_list))
    loss_ratios = (np.array(loss_list) - best_loss) / best_loss
    n_like_best = np.sum(loss_ratios < epsilon)
    done = n_like_best >= n_match or n_tried > max_n

  best_ind = np.argmin(np.array(loss_list))
  best_loss = loss_list[best_ind]
  best_eval_loss = eval_loss_list[best_ind]
  best_params = params_list[best_ind]

  return best_params, best_loss, best_eval_loss
