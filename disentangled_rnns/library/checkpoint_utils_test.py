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

"""Tests for selective parameter freezing during training."""

from absl.testing import absltest
from absl.testing import parameterized
from disentangled_rnns.library import checkpoint_utils
import haiku as hk
import jax
import jax.numpy as jnp
import optax


class CheckpointUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("omitted", {}),
      ("none", {"trainable_param_names": None}),
      ("empty", {"trainable_param_names": []}),
  )
  def test_freeze_all_params(self, kwargs):
    """An omitted or empty selection freezes both gradients and weight decay."""
    params = {
        "linear1": {"w": jnp.array([1.0, -2.0])},
        "linear2": {"b": jnp.array([3.0])},
    }
    gradients = jax.tree.map(jnp.ones_like, params)
    optimizer = checkpoint_utils.get_optimizer_with_frozen_params(
        optax.adamw(1e-3, weight_decay=0.1), params, **kwargs
    )
    state = optimizer.init(params)
    for _ in range(2):
      updates, state = optimizer.update(gradients, state, params)
      updated_params = optax.apply_updates(params, updates)
      for update in jax.tree.leaves(updates):
        self.assertTrue(jnp.array_equal(update, jnp.zeros_like(update)))
      for before, after in zip(
          jax.tree.leaves(params), jax.tree.leaves(updated_params)
      ):
        self.assertTrue(jnp.array_equal(before, after))
      params = updated_params

  def test_get_optimizer_with_frozen_params(self):
    # Create dummy parameters
    def _dummy_model(x):
      x = hk.Linear(output_size=10, name="linear1")(x)
      x = hk.Linear(output_size=5, name="linear2")(x)
      return x

    dummy_input = hk.transform(_dummy_model).init(
        jax.random.PRNGKey(0), jnp.zeros(1)
    )

    # Create a dummy optimizer
    opt = optax.adam(1e-3)

    # Freeze 'linear1' parameters
    trainable_param_names = ["linear2"]
    modified_opt = checkpoint_utils.get_optimizer_with_frozen_params(
        opt, dummy_input, trainable_param_names
    )

    # Check if the optimizer state is correctly partitioned
    dummy_input_grad = hk.transform(_dummy_model).init(
        jax.random.PRNGKey(0), jnp.zeros(1)
    )

    opt_state = modified_opt.init(dummy_input)

    updates, _ = modified_opt.update(dummy_input_grad, opt_state, dummy_input)

    for path, x in updates.items():
      print(f"path: {path}, update: {x} ")

    for path, x in updates.items():
      if "linear1" in path:
        for arr in x.values():
          self.assertTrue(jnp.allclose(arr, 0.0))
      elif "linear2" in path:
        self.assertTrue(any(not jnp.allclose(arr, 0.0) for arr in x.values()))


if __name__ == "__main__":
  absltest.main()
