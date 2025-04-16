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
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import rnn_utils

import optax


class DisrnnTest(absltest.TestCase):

  def test_training_and_eval(self):
    """Test that training and eval work."""
    dataset = get_datasets.get_q_learning_dataset(n_trials=10, n_sessions=10)

    # Train for a few steps
    params, opt_state, losses = rnn_utils.train_network(
        training_dataset=dataset,
        validation_dataset=dataset,
        make_network=disrnn.HkDisRNN,
        opt=optax.adam(1e-3),
        n_steps=20,
        loss="penalized_categorical",
        params=None,
        opt_state=None,
        penalty_scale=1e-3,
        do_plot=True,
    )
    loss_init = losses["training_loss"][-1]

    # Train for a few more steps
    params, _, losses = rnn_utils.train_network(
        training_dataset=dataset,
        validation_dataset=dataset,
        make_network=disrnn.HkDisRNN,
        opt=optax.adam(1e-3),
        n_steps=20,
        loss="penalized_categorical",
        params=params,
        opt_state=opt_state,
        penalty_scale=1e-3,
        do_plot=True,
    )
    loss_final = losses["training_loss"][-1]

    # Check that loss has decreased
    self.assertLess(loss_final, loss_init)

    # Check that plotting functions work
    disrnn.plot_bottlenecks(params)
    disrnn.plot_update_rules(params, disrnn.HkDisRNN)

if __name__ == "__main__":
  absltest.main()
